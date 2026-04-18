"""User-authored setup YAML parser for saklas -C <path>.

The ``vectors:`` key is a steering expression string (the same grammar
every surface speaks); the loader validates it and stores the raw text.
Parsing into a :class:`~saklas.core.steering.Steering` happens at
consumption time via :func:`saklas.core.steering_expr.parse_expr`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from saklas.core.errors import SaklasError

log = logging.getLogger(__name__)

_KNOWN_KEYS = {
    "model", "vectors", "thinking",
    "temperature", "top_p", "max_tokens", "system_prompt",
}


class ConfigFileError(ValueError, SaklasError):
    pass


@dataclass
class ConfigFile:
    model: Optional[str] = None
    vectors: Optional[str] = None
    thinking: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None

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
        for p in extra_paths or []:
            chain.append(cls.load(Path(p)))
        return compose(chain) if chain else cls()

    def to_dict(self) -> dict:
        out: dict = {}
        for f in ("model", "thinking", "temperature", "top_p", "max_tokens", "system_prompt"):
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

        return cls(
            model=data.get("model"),
            vectors=vectors,
            thinking=data.get("thinking"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt"),
        )


def compose(configs: list[ConfigFile]) -> ConfigFile:
    """Combine multiple config files; later entries override earlier ones.

    The ``vectors`` string overrides wholesale — later configs replace the
    earlier expression rather than concatenating. Callers that want to
    extend should spell out the full expression in their override file.
    """
    out = ConfigFile()
    for c in configs:
        for f in ("model", "thinking", "temperature",
                  "top_p", "max_tokens", "system_prompt", "vectors"):
            v = getattr(c, f)
            if v is not None:
                setattr(out, f, v)
    return out


def apply_flag_overrides(cfg_in: ConfigFile, **flags) -> ConfigFile:
    """Return a new ConfigFile with non-None flag values overriding cfg_in."""
    supplied = {k: v for k, v in flags.items() if v is not None}
    return replace(cfg_in, **supplied)


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
    from saklas.io.paths import concept_dir
    from saklas.io.packs import materialize_bundled
    from saklas.io import cache_ops

    if config.vectors is None:
        return []

    missing: list[str] = []
    for ns, concept, _variant in referenced_selectors(config.vectors):
        if ns is None:
            from saklas.cli.selectors import _all_concepts
            slug = concept.split(".")[0] if "." in concept else concept
            matches = [
                c for c in _all_concepts()
                if c.name == concept
                or (
                    "." in c.name
                    and slug in c.name.split(".")
                )
            ]
            if matches:
                continue
            msg = f"vector {concept!r}: must be '<ns>/<name>' (no installed match)"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(concept)
            continue
        coord = f"{ns}/{concept}"
        cdir = concept_dir(ns, concept)
        if cdir.exists():
            continue
        if ns == "local":
            msg = f"vector {coord!r}: local namespace, cannot auto-install"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(coord)
            continue
        if ns == "default":
            materialize_bundled()
            if cdir.exists():
                continue
            msg = f"vector {coord!r}: bundled concept missing from package data"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(coord)
            continue
        try:
            cache_ops.install(coord, as_=None, force=False)
        except Exception as e:
            msg = f"vector {coord!r}: install failed ({e})"
            if strict:
                raise ConfigFileError(msg) from e
            log.warning(msg)
            missing.append(coord)
    return missing
