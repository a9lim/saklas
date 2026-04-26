"""ExtractionPipeline — concept-extraction orchestration lifted out of session.py.

The pipeline owns the full custom-concept flow: tensor cache → curated or
local ``statements.json`` (reused by default) → generate scenarios → save
→ generate pairs → save → contrastive PCA → save tensor.  Curated concepts
save under ``default/<c>/``; user concepts under ``local/<c>/``.

Dependencies are passed structurally (not as a back-reference to the
session) via three runtime-checkable Protocols — :class:`ModelHandle`,
:class:`PackWriter`, :class:`VectorRegistry` — plus an :class:`EventBus`
for ``VectorExtracted`` emission.  ``SaklasSession`` implements all three
protocols implicitly, so construction reads as
``ExtractionPipeline(self, self, self, self.events)``.

The session gates re-entry against ``GenState.IDLE`` before forwarding;
the pipeline itself does not touch generation state.  See ``Phase 7`` of
``docs/plans/audit-followups.md`` for the design rationale.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from saklas.core.events import EventBus, VectorExtracted
from saklas.core.profile import Profile
from saklas.core.sae import SaeBackend
from saklas.io.datasource import DataSource
from saklas.io.packs import hash_file
from saklas.io.paths import tensor_filename
from saklas.core.vectors import (
    extract_contrastive,
    save_profile as _save_profile,
    load_profile as _load_profile,
    load_contrastive_pairs,
)


# ----------------------------------------------------------------------
# Structural protocols.  Runtime-checkable so callers (and tests) can
# ``isinstance(session, ModelHandle)`` to sanity-check the implicit
# implementation.  ``SaklasSession`` satisfies each by virtue of carrying
# the listed attributes / methods.
# ----------------------------------------------------------------------

@runtime_checkable
class ModelHandle(Protocol):
    """Read-only surface the pipeline needs from the live HF session.

    Held as a *handle*, not a copy — the pipeline must see the same
    model object the session uses; otherwise device-side state diverges.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def tokenizer(self) -> Any: ...  # PreTrainedTokenizerBase

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def layers(self) -> Any: ...  # ``get_layers`` returns ``nn.ModuleList`` — list-like

    def _run_generator(
        self, system_msg: str, prompt: str, max_new_tokens: int,
    ) -> str:
        """Single-turn LLM call shared by scenario and pair generators.

        Underscore-prefixed because the override site is per-session
        (subclass-and-override is the established test pattern).  The
        protocol shape mirrors the existing ``SaklasSession._run_generator``
        signature exactly so the session satisfies it implicitly.
        """
        ...

    def generate_scenarios(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = ...,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]: ...

    def generate_pairs(
        self,
        concept: str,
        baseline: str | None = None,
        n: int = ...,
        *,
        scenarios: list[str] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[tuple[str, str]]: ...


@runtime_checkable
class PackWriter(Protocol):
    """Local-pack mutations the pipeline performs."""

    def _local_concept_folder(self, canonical: str) -> pathlib.Path:
        """Return ``local/<canonical>/`` with a placeholder ``pack.json``."""
        ...

    def _promote_profile(
        self, profile: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Cast a freshly-loaded profile onto the session's device + dtype."""
        ...

    def _update_local_pack_files(self, folder: pathlib.Path) -> None:
        """Recompute ``pack.json.files`` after writing tensors / statements."""
        ...


@runtime_checkable
class VectorRegistry(Protocol):
    """Optional registry the pipeline writes back to.

    The current pipeline does not auto-register profiles into the session
    registry — callers do that explicitly via ``session.steer(name, profile)``.
    The protocol is wired in for parity with the plan, and so future
    callers can ask the pipeline to populate a registry directly.
    """

    def __contains__(self, name: str) -> bool: ...
    def add(self, name: str, profile: Profile) -> None: ...


# ----------------------------------------------------------------------
# Pipeline.
# ----------------------------------------------------------------------


class ExtractionPipeline:
    """Self-contained concept-extraction pipeline.

    Construction accepts the structural dependencies extraction actually
    uses; the pipeline holds none of them as inherited "session" state.
    See module docstring for the audit reference.
    """

    __slots__ = ("_handle", "_packs", "_registry", "_events")

    def __init__(
        self,
        model_handle: ModelHandle,
        pack_writer: PackWriter,
        registry: VectorRegistry,
        events: EventBus,
    ) -> None:
        self._handle = model_handle
        self._packs = pack_writer
        self._registry = registry
        self._events = events

    # -- public entry point ------------------------------------------------

    def extract(
        self,
        source,
        baseline: str | None = None,
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
    ) -> tuple[str, Profile]:
        """Extract a steering vector profile and emit ``VectorExtracted``.

        Mirrors the historical ``SaklasSession.extract`` signature: ``sae``
        accepts a release-name string (resolved via :func:`load_sae_backend`)
        or an already-resolved :class:`SaeBackend` for direct injection.

        Cache-hit semantics:

        - Tensor cache hits short-circuit and emit ``VectorExtracted``
          with ``method`` set to the prior extraction method.
        - On tensor miss, ``statements.json`` is reused when present
          unless ``force_statements=True`` or explicit ``scenarios=[...]``
          were supplied.
        - On statements miss, the full pipeline runs end-to-end.
        """
        canonical, profile = self._extract_impl(
            source, baseline,
            scenarios=scenarios,
            reuse_scenarios=reuse_scenarios,
            force_statements=force_statements,
            on_progress=on_progress,
            sae=sae,
            sae_revision=sae_revision,
            namespace=namespace,
        )
        try:
            meta = dict(profile.metadata) if hasattr(profile, "metadata") else {}
        except Exception:
            meta = {}
        self._events.emit(
            VectorExtracted(name=canonical, profile=profile, metadata=meta),
        )
        return canonical, profile

    # -- internal pipeline -------------------------------------------------

    def _extract_impl(
        self,
        source,
        baseline: str | None = None,
        *,
        scenarios: list[str] | None = None,
        reuse_scenarios: bool = False,
        force_statements: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
    ) -> tuple[str, Profile]:
        """Extraction body.  See :meth:`extract` for the wrapper.

        Pipeline: tensor cache check → ``statements.json`` cache (curated
        or local) unless ``force_statements`` → otherwise generate
        scenarios + pairs → save both → extract contrastive → save tensor.
        Scenarios persist to ``local/<canonical>/scenarios.json`` alongside
        ``statements.json``.
        """
        # Local import to avoid an import cycle: ``session`` imports the
        # pipeline at module load, and ``canonical_concept_name`` lives on
        # the session module for back-compat with downstream callers
        # (cloning, tui, runners, selectors).  The function is pure; the
        # late import has no runtime cost after first use.
        from saklas.core.session import canonical_concept_name, _split_composite_source

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Normalize source.
        if isinstance(source, str):
            concept, baseline = _split_composite_source(source, baseline)
        elif isinstance(source, DataSource):
            concept = source.name
            baseline = None
        elif isinstance(source, list):
            concept = "custom"
            baseline = None
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        canonical = canonical_concept_name(concept, baseline)

        # Resolve the SAE backend once.  No-op when ``sae is None`` —
        # the ``load_sae_backend`` import is lazy so non-SAE callers
        # don't touch the SAE layer.  Pre-resolved ``SaeBackend`` instances
        # pass through verbatim.
        sae_backend: SaeBackend | None
        sae_release: str | None
        if sae is None:
            sae_backend = None
            sae_release = None
        elif isinstance(sae, str):
            from saklas.core.sae import load_sae_backend
            sae_backend = load_sae_backend(
                sae,
                revision=sae_revision,
                model_id=self._handle.model_id,
                device=self._handle.device,
            )
            sae_release = sae
        else:
            # Pre-resolved backend.
            sae_backend = sae
            sae_release = sae.release

        sae_metadata: dict = {}
        if sae_backend is not None:
            sae_metadata = {
                "sae_release": sae_backend.release,
                "sae_revision": sae_backend.revision,
                "sae_ids_by_layer": getattr(sae_backend, "sae_ids_by_layer", {}),
            }

        def _build_return(profile_dict: dict) -> tuple[str, Profile]:
            meta: dict = {
                "method": "pca_center_sae" if sae_backend is not None else "contrastive_pca",
            }
            meta.update(sae_metadata)
            out_name = (
                canonical
                if sae_backend is None
                else f"{canonical}:sae-{sae_backend.release}"
            )
            return out_name, Profile(profile_dict, metadata=meta)

        def _save_meta(extra: dict | None = None) -> dict:
            meta: dict = {
                "method": "pca_center_sae" if sae_backend is not None else "contrastive_pca",
            }
            if extra:
                meta.update(extra)
            meta.update(sae_metadata)
            return meta

        model = self._handle.model
        tokenizer = self._handle.tokenizer
        layers = self._handle.layers
        model_id = self._handle.model_id

        # For DataSource or raw pairs, skip the full pipeline — just extract.
        if isinstance(source, (DataSource, list)):
            if isinstance(source, list):
                ds = DataSource(pairs=source)
            else:
                ds = source
            folder = self._packs._local_concept_folder(canonical)
            cache_path = str(folder / tensor_filename(model_id, release=sae_release))
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._packs._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                out_name = (
                    canonical
                    if sae_backend is None
                    else f"{canonical}:sae-{sae_backend.release}"
                )
                return out_name, Profile(profile, metadata=_meta)
            except (FileNotFoundError, KeyError, ValueError):
                pass

            _progress(f"Extracting profile ({len(ds.pairs)} pairs)...")
            pairs = [{"positive": p, "negative": n} for p, n in ds.pairs]
            profile = extract_contrastive(
                model, tokenizer, pairs, layers=layers,
                sae=sae_backend,
            )
            _save_profile(profile, cache_path, _save_meta())
            self._packs._update_local_pack_files(folder)
            return _build_return(profile)

        # String source — full pipeline.  Pack lookup scans installed
        # namespaces, but bare names must not silently pick the first
        # duplicate.  If the caller supplies ``namespace=``, honor only that
        # namespace.
        from saklas.io.selectors import _all_concepts, AmbiguousSelectorError
        matches = [
            c for c in _all_concepts()
            if c.name == canonical and (namespace is None or c.namespace == namespace)
        ]
        if namespace is not None and not matches and namespace != "local":
            raise FileNotFoundError(
                f"concept pack '{namespace}/{canonical}' is not installed"
            )
        if namespace is None and len(matches) > 1:
            qualified = ", ".join(f"{c.namespace}/{c.name}" for c in matches)
            raise AmbiguousSelectorError(
                f"ambiguous concept '{canonical}': matches {qualified}. "
                f"Specify with a namespace."
            )
        pack_folder = matches[0].folder if matches else None

        if pack_folder is not None:
            cache_path = str(pack_folder / tensor_filename(model_id, release=sae_release))
        else:
            cache_path = str(
                pathlib.Path(self._packs._local_concept_folder(canonical))
                / tensor_filename(model_id, release=sae_release)
            )

        # 1. Vector cache — short-circuits unless a regen path is requested.
        #    ``force_statements=True`` or explicit ``scenarios=[...]`` both
        #    mean the caller wants fresh pairs, which definitionally
        #    invalidates any tensor trained on the old pairs — bypassing
        #    the tensor cache here is the only semantically coherent
        #    behavior for those flags.  No cache hit means the full
        #    pipeline runs end-to-end and overwrites the stale tensor.
        if not force_statements and scenarios is None:
            try:
                profile, _meta = _load_profile(cache_path)
                profile = self._packs._promote_profile(profile)
                _progress(f"Loaded cached profile for '{canonical}'.")
                out_name = (
                    canonical
                    if sae_backend is None
                    else f"{canonical}:sae-{sae_backend.release}"
                )
                return out_name, Profile(profile, metadata=_meta)
            except (FileNotFoundError, KeyError, ValueError):
                pass

        # 2. Installed-statements fast path — default reuses bundled / HF /
        #    local statements.json when present.  ``force_statements=True``
        #    skips this branch and falls through to regeneration.  Passing an
        #    explicit ``scenarios=`` also skips — if the caller supplied
        #    scenarios they're clearly opting into fresh pair generation.
        if pack_folder is not None and not force_statements and scenarios is None:
            pack_stmts = pack_folder / "statements.json"
            if pack_stmts.exists():
                _progress(f"Using curated statements for '{canonical}'...")
                ds = load_contrastive_pairs(str(pack_stmts))
                profile = extract_contrastive(
                    model, tokenizer, ds["pairs"],
                    layers=layers,
                    sae=sae_backend,
                )
                _save_profile(profile, cache_path, _save_meta({
                    "statements_sha256": hash_file(pack_stmts),
                }))
                self._packs._update_local_pack_files(pack_folder)
                return _build_return(profile)

        # 3. Local statements cache — default reuses if present.
        local_folder = self._packs._local_concept_folder(canonical)
        stmt_cache_path = str(local_folder / "statements.json")
        local_folder = pathlib.Path(stmt_cache_path).parent
        pairs = None
        if not force_statements and scenarios is None:
            try:
                with open(stmt_cache_path) as f:
                    cached = json.load(f)
                if isinstance(cached, list):
                    pairs = cached
                elif isinstance(cached, dict) and "pairs" in cached:
                    pairs = cached["pairs"]
                if pairs:
                    _progress(f"Using cached pairs for '{canonical}'.")
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                pass

        # 4. Generate scenarios + pairs if needed.
        if pairs is None:
            suffix = f" vs '{baseline}'" if baseline else ""
            local_folder.mkdir(parents=True, exist_ok=True)

            # 4a. Resolve effective scenarios.
            scn_path = local_folder / "scenarios.json"
            eff_scenarios: list[str] | None = None
            if scenarios is not None:
                eff_scenarios = list(scenarios)
                _progress(
                    f"Using {len(eff_scenarios)} caller-provided scenarios."
                )
            elif reuse_scenarios and scn_path.exists():
                try:
                    with open(scn_path) as f:
                        data = json.load(f)
                    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
                        eff_scenarios = [str(s) for s in data["scenarios"]]
                    elif isinstance(data, list):
                        eff_scenarios = [str(s) for s in data]
                    if eff_scenarios:
                        _progress(
                            f"Reusing {len(eff_scenarios)} cached scenarios "
                            f"for '{canonical}'."
                        )
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    eff_scenarios = None

            if not eff_scenarios:
                _progress(f"Generating scenarios for '{concept}'{suffix}...")
                eff_scenarios = self._handle.generate_scenarios(
                    concept, baseline, on_progress=_progress,
                )

            if not eff_scenarios:
                raise ValueError(
                    f"Could not generate scenarios for '{concept}'. "
                    f"Try a more specific concept."
                )

            # Persist scenarios to disk (overwriting any existing file).
            with open(scn_path, "w") as f:
                json.dump({"scenarios": eff_scenarios}, f, indent=2)

            _progress(
                f"Generating contrastive pairs for '{concept}'{suffix} "
                f"across {len(eff_scenarios)} domains..."
            )
            raw_pairs = self._handle.generate_pairs(
                concept, baseline,
                scenarios=eff_scenarios,
                on_progress=_progress,
            )
            pairs = [{"positive": p, "negative": n} for p, n in raw_pairs]
            with open(stmt_cache_path, "w") as f:
                json.dump(pairs, f, indent=2)

        if len(pairs) < 2:
            raise ValueError(
                f"Could only generate {len(pairs)} pairs (need >= 2). "
                f"Try a more specific concept."
            )

        # 5. Extract.
        _progress(f"Extracting contrastive profile ({len(pairs)} pairs)...")
        profile = extract_contrastive(
            model, tokenizer, pairs, layers=layers,
            sae=sae_backend,
        )
        _save_profile(profile, cache_path, _save_meta({
            "statements_sha256": hash_file(pathlib.Path(stmt_cache_path)),
        }))
        self._packs._update_local_pack_files(local_folder)
        return _build_return(profile)
