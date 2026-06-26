"""User-authored setup YAML parser for saklas -C <path>.

The ``vectors:`` key is a steering expression string (the same grammar
every surface speaks); the loader validates it and stores the raw text.
Parsing into a :class:`~saklas.core.steering.Steering` happens at
consumption time via :func:`saklas.core.steering_expr.parse_expr`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from saklas.core.errors import SaklasError

log = logging.getLogger(__name__)

_KNOWN_KEYS = {
    "model", "vectors", "thinking",
    "temperature", "top_p", "max_tokens", "system_prompt",
    "compile", "cuda_graphs",
    "return_top_k",
}


class ConfigFileError(ValueError, SaklasError):
    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


@dataclass
class ConfigFile:
    model: Optional[str] = None
    vectors: Optional[str] = None
    thinking: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    compile: Optional[bool] = None           # CUDA torch.compile auto-enable; None = default (on)
    cuda_graphs: Optional[bool] = None       # CUDA StaticCache + graph capture; None = default (on)
    # Session-level default for SamplingConfig.return_top_k — the number
    # of top-K alternatives the engine decodes (with text) per generated
    # token.  ``None`` means "use session default" (which is 0 — chosen
    # logprob only).  Per-call SamplingConfig.return_top_k > 0 overrides;
    # K=0 inherits this session-level value.  Phase 1 logit pass.
    return_top_k: Optional[int] = None

    @classmethod
    def load_default(cls) -> Optional["ConfigFile"]:
        """Load ``~/.saklas/config.yaml`` if it exists, else return ``None``."""
        from saklas.io.paths import saklas_home
        p = saklas_home() / "config.yaml"
        if not p.exists():
            return None
        return cls.load(p)

    @classmethod
    def effective(
        cls,
        extra_paths: list[Path] | None = None,
        *,
        include_default: bool = True,
    ) -> "ConfigFile":
        """Compose the default config + extras into a single ConfigFile.

        Order: ``~/.saklas/config.yaml`` (if present) → extras (in order).
        Later entries override earlier ones.
        """
        chain: list[ConfigFile] = []
        if include_default:
            default = cls.load_default()
            if default is not None:
                chain.append(default)
        chain.extend(cls.load(Path(p)) for p in extra_paths or [])
        return compose(chain) if chain else cls()

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for f in (
            "model", "thinking", "temperature", "top_p", "max_tokens",
            "system_prompt", "compile", "cuda_graphs", "return_top_k",
        ):
            v = getattr(self, f)
            if v is not None:
                out[f] = v
        if self.vectors:
            out["vectors"] = self.vectors
        return out

    def to_yaml(self, *, header: Optional[str] = None) -> str:
        import yaml
        body = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if header:
            return f"{header}\n{body}"
        return body

    @classmethod
    def load(cls, path: Path) -> "ConfigFile":
        import yaml
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigFileError(f"{path}: YAML parse error: {e}") from e
        if not isinstance(data, dict):
            raise ConfigFileError(f"{path}: top-level must be a mapping")

        unknown = set(data.keys()) - _KNOWN_KEYS
        for k in unknown:
            log.warning("unknown key %r in %s (ignored)", k, path)

        vectors_raw = data.get("vectors")
        vectors: Optional[str] = None
        if vectors_raw is not None:
            if not isinstance(vectors_raw, str):
                raise ConfigFileError(
                    f"{path}: vectors: must be a steering expression string "
                    f"(got {type(vectors_raw).__name__}). Example: "
                    f"`vectors: \"0.5 honest + 0.3 warm\"`."
                )
            text = vectors_raw.strip()
            if text:
                # Validate via the shared parser; raise a wrapped error on
                # failure so the YAML path surfaces the column/detail.
                from saklas.core.steering_expr import (
                    SteeringExprError, parse_expr,
                )
                try:
                    parse_expr(text)
                except SteeringExprError as e:
                    raise ConfigFileError(
                        f"{path}: vectors: {e}"
                    ) from e
                vectors = text

        compile_v = data.get("compile")
        if compile_v is not None and not isinstance(compile_v, bool):
            # ``compile: false`` is the documented opt-out; reject ints,
            # strings, etc. so a malformed YAML doesn't silently disable
            # compile via ``bool("false") == True``.
            raise ConfigFileError(
                f"{path}: compile must be a boolean (got "
                f"{type(compile_v).__name__} {compile_v!r}). Use "
                f"``compile: true`` or ``compile: false``."
            )

        cuda_graphs_v = data.get("cuda_graphs")
        if cuda_graphs_v is not None and not isinstance(cuda_graphs_v, bool):
            raise ConfigFileError(
                f"{path}: cuda_graphs must be a boolean (got "
                f"{type(cuda_graphs_v).__name__} {cuda_graphs_v!r}). "
                f"Use ``cuda_graphs: true`` or ``cuda_graphs: false``."
            )

        return_top_k_v = data.get("return_top_k")
        if return_top_k_v is not None:
            # Reject bool sneaking through as int (``return_top_k: false``
            # would silently set 0 without this guard; the user almost
            # certainly meant a number).
            if isinstance(return_top_k_v, bool) or not isinstance(return_top_k_v, int):
                raise ConfigFileError(
                    f"{path}: return_top_k must be an integer in [0, 256] "
                    f"(got {type(return_top_k_v).__name__} {return_top_k_v!r})"
                )
            if return_top_k_v < 0 or return_top_k_v > 256:
                raise ConfigFileError(
                    f"{path}: return_top_k out of range [0, 256] "
                    f"(got {return_top_k_v!r})"
                )

        return cls(
            model=data.get("model"),
            vectors=vectors,
            thinking=data.get("thinking"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
            compile=compile_v,
            cuda_graphs=cuda_graphs_v,
            return_top_k=return_top_k_v,
        )


def compose(configs: list[ConfigFile]) -> ConfigFile:
    """Combine multiple config files; later entries override earlier ones.

    The ``vectors`` string overrides wholesale — later configs replace the
    earlier expression rather than concatenating. Callers that want to
    extend should spell out the full expression in their override file.
    """
    out = ConfigFile()
    for c in configs:
        for f in (
            "model", "thinking", "temperature",
            "top_p", "max_tokens", "system_prompt", "vectors",
            "compile", "cuda_graphs",
            "return_top_k",
        ):
            v = getattr(c, f)
            if v is not None:
                setattr(out, f, v)
    return out


def apply_flag_overrides(cfg_in: ConfigFile, **flags: Any) -> ConfigFile:
    """Return a new ConfigFile with non-None flag values overriding cfg_in."""
    supplied = {k: v for k, v in flags.items() if v is not None}
    return replace(cfg_in, **supplied)


def _bare_concept_resolves(concept: str) -> bool:
    """True when a bare (un-namespaced) concept reference resolves to a manifold.

    Mirrors the steering read path (:mod:`saklas.core.steering_expr`): a bare
    pole or node label resolves via :func:`resolve_bare_name`, and a composite
    name (``happy.sad``) resolves to its 2-node ``pca`` manifold via
    :func:`resolve_manifold_name`.  An ambiguity counts as resolved (the
    reference matches more than one installed artifact, not zero).
    """
    from saklas.io.selectors import (
        AmbiguousSelectorError,
        resolve_bare_name,
        resolve_manifold_name,
    )

    try:
        manifold_hit = resolve_bare_name(concept)
        if manifold_hit is not None:
            return True
        return resolve_manifold_name(concept) is not None
    except AmbiguousSelectorError:
        return True


def ensure_vectors_installed(config: ConfigFile, *, strict: bool) -> list[str]:
    """Install any vectors referenced in ``config.vectors`` that are not
    present locally.

    Walks the raw expression string via :func:`referenced_selectors` so
    namespace-qualified references (``bob/honest``) retain their install
    coordinates even though the parsed ``Steering`` flattens them. Returns
    a list of coords that could not be installed; in strict mode, raises
    on any failure instead.
    """
    from saklas.core.steering_expr import referenced_selectors
    from saklas.io.paths import manifold_dir
    from saklas.io.manifolds import materialize_bundled_manifolds
    from saklas.io.templates import materialize_bundled_templates
    from saklas.io import selectors as _selectors

    if config.vectors is None:
        return []

    # Every concept is a manifold (4.0): bundled ones live under
    # ``manifolds/default/<name>/``.  Materialize them up front so a bare or
    # ``default/`` reference resolves against the just-dropped folders, and
    # drop any stale resolver cache so the new folders are seen.  Templates
    # first — a bundled manifold may ``template_ref`` a bundled template.
    materialize_bundled_templates()
    materialize_bundled_manifolds()
    _selectors.invalidate()

    missing: list[str] = []
    for ns, concept, _variant in referenced_selectors(config.vectors):
        if ns is None:
            if _bare_concept_resolves(concept):
                continue
            msg = f"vector {concept!r}: must be '<ns>/<name>' (no installed match)"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(concept)
            continue
        coord = f"{ns}/{concept}"
        if (manifold_dir(ns, concept) / "manifold.json").exists():
            continue
        if ns == "local":
            msg = f"vector {coord!r}: local namespace, cannot auto-install"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(coord)
            continue
        if ns == "default":
            msg = f"vector {coord!r}: bundled manifold missing from package data"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(coord)
            continue
        try:
            from saklas.io.hf_manifolds import install_manifold
            install_manifold(coord, force=False)
            _selectors.invalidate()
        except Exception as e:
            msg = f"vector {coord!r}: install failed ({e})"
            if strict:
                raise ConfigFileError(msg) from e
            log.warning(msg)
            missing.append(coord)
    return missing
