"""Persona cloning via corpus-driven contrastive extraction.

Reuses the same contrastive-PCA pipeline as saklas.session.extract().
Instead of curated or LLM-synthesized pairs from a concept *name*,
pairs are `(persona_line, model_neutralized_rewrite)` — the model
flattens the persona's voice into plain prose, and the delta between
the two is the persona direction.

Only the entry point is public: `clone_from_corpus`. Tests hit the
module-private helpers directly to keep them CPU-only; the full
extraction path is covered by a GPU-gated end-to-end test.
"""
from __future__ import annotations

import json
import logging
import pathlib
import random
import re
from typing import Any, TYPE_CHECKING


from saklas.io.atomic import write_json_atomic
from saklas.io.datasource import DataSource
from saklas.core.errors import SaklasError
from saklas.core.generation import (
    GenerationConfig,
    GenerationState,
    build_chat_input,
    generate_steered,
)
from saklas.io.packs import NAME_REGEX, PackFormatError, PackMetadata, hash_file
from saklas.io.paths import concept_dir, safe_model_id
from saklas.core.vectors import load_profile as _load_profile

if TYPE_CHECKING:
    from saklas.core.profile import Profile
    from saklas.core.session import SaklasSession

_log = logging.getLogger(__name__)

_BATCH_SIZE = 5
_MIN_WORDS = 6
_MIN_CORPUS_LINES = 10
_TOKENS_PER_REWRITE = 80  # ~flat budget per rewrite line; 50 truncated long inputs

_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s*(.*?)\s*$")


class CorpusTooShortError(ValueError, SaklasError):
    """Corpus has fewer than the minimum number of usable lines."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class CorpusTooLongError(ValueError, SaklasError):
    """A batch prompt plus its generation budget does not fit in context."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class InsufficientPairsError(RuntimeError, SaklasError):
    """Too few pairs survived generation/parsing to extract a stable vector."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


# ---------------------------------------------------------------------------
# Pure helpers (tested directly)
# ---------------------------------------------------------------------------

def _filter_corpus(path: str | pathlib.Path) -> list[str]:
    """Load corpus from disk, filter short lines, dedupe.

    UTF-8 with errors='replace' (never raises on bad bytes). Leading
    BOM stripped. Whitespace stripped per line. Lines with fewer than
    `_MIN_WORDS` whitespace-separated tokens are dropped. Duplicates
    are collapsed, preserving first-seen order.
    """
    path = pathlib.Path(path)
    raw = path.read_bytes().decode("utf-8", errors="replace")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s.split()) < _MIN_WORDS:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _sample_lines(lines: list[str], n: int, rng: random.Random) -> list[str]:
    """Seeded random sample without replacement. Clamps n to len(lines)."""
    if not lines:
        raise ValueError("cannot sample from empty corpus")
    n = min(n, len(lines))
    return rng.sample(lines, n)


def _chunk(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into batches of at most `batch_size`. Last may be short."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _build_neutralize_prompt(batch: list[str]) -> str:
    """Build a numbered-list rewrite prompt for neutralizing a batch."""
    k = len(batch)
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(batch))
    return (
        f"Rewrite each of the following lines in a plain, neutral voice. "
        f"Preserve meaning exactly. Do not add commentary. Output exactly "
        f"{k} numbered lines, one per line, in the form \"<i>. <rewrite>\".\n\n"
        f"{numbered}"
    )


def _parse_numbered(raw: str, expected: int) -> list[str] | None:
    """Parse a numbered-list response. Returns None on any format failure.

    Tolerates leading preamble and trailing notes. Requires exactly
    `expected` numbered lines, numbered 1..expected in strict order.
    Empty rewrites are tolerated here and preserved in the returned
    list — the pair-assembly loop drops empty/whitespace-only rewrites
    individually so the rest of the batch still contributes.
    """
    rewrites: list[tuple[int, str]] = []
    for line in raw.splitlines():
        m = _NUMBERED_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()
        rewrites.append((idx, text))
    if len(rewrites) != expected:
        return None
    for pos, (idx, _text) in enumerate(rewrites, start=1):
        if idx != pos:
            return None
    return [text for _, text in rewrites]


def _fit_check(
    tokenizer: Any,
    batch: list[str],
    generation_budget: int,
    ctx_len: int,
) -> bool:
    """Rough upper-bound fit check for a single batch."""
    prompt = _build_neutralize_prompt(batch)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # Add slack for chat template wrapping.
    overhead = 64
    return (len(ids) + overhead + generation_budget) <= ctx_len


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _neutralize_batch(
    session: "SaklasSession",
    batch: list[str],
    seed: int | None,
) -> list[str] | None:
    """Run the neutralize prompt through `generate_steered`, parse result.

    Called after `SteeringManager.clear_all()`, so there are no active
    steering hooks — `generate_steered` behaves identically to a plain
    forward generate, while matching the spec's reuse of the canonical
    sampler path.
    """
    prompt = _build_neutralize_prompt(batch)
    system_msg = (
        "You rewrite text in plain neutral prose. You follow format "
        "instructions exactly."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    input_ids = build_chat_input(
        session._tokenizer, messages, system_prompt=None,
    ).to(session._device)

    max_new = len(batch) * _TOKENS_PER_REWRITE + 32
    cfg = GenerationConfig(
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
    )
    state = GenerationState()
    generated_ids = generate_steered(
        session._model, session._tokenizer, input_ids,
        cfg, state, thinking=False, seed=seed,
    )
    text = session._tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _parse_numbered(text, len(batch))


def clone_from_corpus(
    session: "SaklasSession",
    path: str | pathlib.Path,
    name: str,
    *,
    n_pairs: int = 90,
    seed: int | None = None,
    batch_size: int = _BATCH_SIZE,
    force: bool = False,
) -> tuple[str, "Profile"]:
    """Extract a monopolar steering vector from a persona corpus file.

    Pipeline: load + filter corpus, sample n_pairs lines, batch them,
    ask the model to rewrite each batch in neutral voice, pair each
    persona line with its rewrite, feed into the same contrastive PCA
    path as session.extract(). Returns `(canonical_name, profile)`.

    Cache: keyed on corpus sha256 + n_pairs + batch_size + seed +
    model. Pass `force=True` to bypass.
    """
    # Local import to avoid a circular import with saklas.session.
    from saklas.core.session import canonical_concept_name

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")

    canonical = canonical_concept_name(name)
    if not NAME_REGEX.match(canonical):
        raise ValueError(
            f"invalid concept name after slugging: {canonical!r}"
        )

    folder = concept_dir("local", canonical)
    folder.mkdir(parents=True, exist_ok=True)

    corpus_sha = hash_file(path)
    cache_path = folder / f"{safe_model_id(session.model_id)}.safetensors"

    # Cache hit check.
    #   seed is not None -> must match exactly.
    #   seed is None     -> ignore seed, reuse any prior run with matching
    #                       corpus/n_pairs/batch_size.
    if not force and cache_path.exists() and (folder / "pack.json").exists():
        try:
            raw_pack = json.loads((folder / "pack.json").read_text())
            cached_sha = raw_pack.get("corpus_sha256")
            cached_n = raw_pack.get("n_pairs")
            cached_bs = raw_pack.get("batch_size")
            cached_seed = raw_pack.get("seed")
            seed_ok = True if seed is None else cached_seed == seed
            if (
                cached_sha == corpus_sha
                and cached_n == n_pairs
                and cached_bs == batch_size
                and seed_ok
            ):
                profile, _m = _load_profile(str(cache_path))
                profile = session._promote_profile(profile)
                _log.info("cloned %s (cache hit) -> %s", canonical, folder)
                from saklas.core.profile import Profile
                return canonical, Profile(profile, metadata=_m)
        except (PackFormatError, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            pass

    # Placeholder pack.json so downstream helpers (_update_local_pack_files)
    # have something to operate on. Overwritten in full at the end.
    if not (folder / "pack.json").exists():
        PackMetadata(
            name=canonical,
            description=f"Cloned from {path.name}",
            version="1.0.0",
            license="AGPL-3.0-or-later",
            tags=["cloned"],
            recommended_alpha=0.5,
            source="local",
            files={},
        ).write(folder)

    lines = _filter_corpus(path)
    if len(lines) < _MIN_CORPUS_LINES:
        raise CorpusTooShortError(
            f"corpus has {len(lines)} usable lines after filtering; "
            f"need at least {_MIN_CORPUS_LINES}"
        )

    if n_pairs > len(lines):
        _log.warning(
            "clone: requested %d pairs but corpus only has %d lines; clamping",
            n_pairs, len(lines),
        )
        n_pairs = len(lines)

    # Record effective seed so reruns are reproducible.
    effective_seed = seed if seed is not None else random.Random().randint(0, 2**31 - 1)
    _log.info("clone: effective seed = %d", effective_seed)
    rng = random.Random(effective_seed)

    sample = _sample_lines(lines, n_pairs, rng)
    batches = _chunk(sample, batch_size)

    # Fit check: every batch, not just the one with the most lines — a
    # short batch of long lines can still overflow. Context length via
    # model config with a loud fallback.
    ctx_len = getattr(session._model.config, "max_position_embeddings", None)
    if ctx_len is None:
        _log.warning(
            "model config missing max_position_embeddings; assuming 4096 context window"
        )
        ctx_len = 4096
    for b in batches:
        budget = len(b) * _TOKENS_PER_REWRITE + 32
        if not _fit_check(session._tokenizer, b, budget, ctx_len):
            raise CorpusTooLongError(
                f"batch prompt + generation budget exceeds context length "
                f"({ctx_len}); try a smaller --n-pairs or shorter corpus lines"
            )

    # Save and clear steering state — extracted pairs must not be
    # contaminated by whatever's currently active. Shallow dict() is
    # safe: inner profile dicts are immutable references, we only
    # clear the outer mapping via clear_all().
    saved_vectors = dict(session._steering.vectors)
    session._steering.clear_all()

    pairs: list[tuple[str, str]] = []
    try:
        for i, batch in enumerate(batches):
            batch_seed = effective_seed + i
            rewrites = _neutralize_batch(session, batch, batch_seed)
            if rewrites is None:
                # Retry once with a perturbed seed.
                _log.warning("clone: batch %d parse failed; retrying", i)
                rewrites = _neutralize_batch(session, batch, batch_seed + 10_000)
            if rewrites is None:
                _log.warning("clone: batch %d dropped after retry", i)
                continue
            for persona_line, rewrite in zip(batch, rewrites):
                # Drop degenerate (empty / whitespace-only) rewrites at
                # the pair-assembly level — the rest of the batch still
                # contributes.
                if not persona_line.strip() or not rewrite.strip():
                    continue
                pairs.append((persona_line, rewrite))
    finally:
        # Restore prior vector registrations.
        for vname, vdata in saved_vectors.items():
            session._steering.add_vector(vname, vdata["profile"], vdata["alpha"])
        if saved_vectors:
            session._steering.apply_to_model(
                session._layers, session._device, session._dtype,
            )

    min_pairs = max(n_pairs // 2, 4)
    if len(pairs) < min_pairs:
        raise InsufficientPairsError(
            f"only {len(pairs)} pairs survived generation; "
            f"need at least {min_pairs}"
        )

    # Persist statements.json in the same shape as bundled packs.
    stmts_path = folder / "statements.json"
    stmt_objs = [{"positive": p, "negative": n} for p, n in pairs]
    write_json_atomic(stmts_path, stmt_objs)

    # Delegate the contrastive PCA + save + local pack refresh to the
    # shared session.extract() path via a DataSource. Clear any stale
    # safetensors first so extract() doesn't short-circuit on a cache
    # miss we already decided to bypass.
    if cache_path.exists():
        cache_path.unlink()
    ds = DataSource(pairs=pairs, name=canonical)
    _canonical, profile = session.extract(ds)

    # Single final pack.json overwrite: merge descriptive + clone
    # metadata on top of whatever session.extract() / _update_local_pack_files
    # produced, then refresh the files hash map one more time so the
    # manifest covers the statements.json we just wrote.
    raw_pack = json.loads((folder / "pack.json").read_text())
    raw_pack["description"] = f"Cloned from {path.name}"
    raw_pack["source"] = "local"
    raw_pack["tags"] = sorted(set((raw_pack.get("tags") or []) + ["cloned"]))
    raw_pack["corpus_sha256"] = corpus_sha
    raw_pack["n_pairs"] = n_pairs
    raw_pack["batch_size"] = batch_size
    raw_pack["seed"] = effective_seed
    write_json_atomic(folder / "pack.json", raw_pack)

    _log.info(
        "cloned %s -> %s (corpus not included in pack; statements.json contains "
        "model-generated pairs only)",
        canonical, folder,
    )
    return canonical, profile
