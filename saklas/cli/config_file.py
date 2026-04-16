"""User-authored setup YAML parser for saklas -C <path>.

See docs/superpowers/specs/2026-04-12-story-a-portability-design.md §Component 7.
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
    vectors: dict[str, float] = field(default_factory=dict)
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
            out["vectors"] = dict(self.vectors)
        return out

    def to_yaml(self, *, header: Optional[str] = None) -> str:
        import yaml
        body = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if header:
            return f"{header}\n{body}"
        return body

    def resolve_poles(self) -> "ConfigFile":
        """Return a copy with ``vectors`` keys run through ``resolve_pole``.

        Bare poles like ``wolf`` resolve to ``deer.wolf`` with sign -1; the
        alpha is negated accordingly. Namespaced keys stay in their namespace.
        """
        from saklas.cli.selectors import resolve_pole, AmbiguousSelectorError
        out: dict[str, float] = {}
        for coord, alpha in self.vectors.items():
            if "/" in coord:
                ns, name = coord.split("/", 1)
            else:
                ns, name = None, coord
            try:
                canonical, sign, _match = resolve_pole(name, namespace=ns)
            except AmbiguousSelectorError:
                out[coord] = alpha
                continue
            key = f"{ns}/{canonical}" if ns else canonical
            out[key] = alpha * sign
        return replace(self, vectors=out)

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

        vectors_raw = data.get("vectors") or {}
        vectors: dict[str, float] = {}
        if isinstance(vectors_raw, dict):
            for coord, alpha in vectors_raw.items():
                try:
                    vectors[str(coord)] = float(alpha)
                except (TypeError, ValueError) as e:
                    raise ConfigFileError(f"vector {coord!r} alpha not a number: {e}") from e
        else:
            raise ConfigFileError("vectors: must be a mapping of coord -> alpha")

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
    """Combine multiple config files; later entries override earlier ones."""
    out = ConfigFile()
    for c in configs:
        for f in ("model", "thinking", "temperature",
                  "top_p", "max_tokens", "system_prompt"):
            v = getattr(c, f)
            if v is not None:
                setattr(out, f, v)
        for coord, alpha in c.vectors.items():
            out.vectors[coord] = alpha
    return out


def apply_flag_overrides(cfg_in: ConfigFile, **flags) -> ConfigFile:
    """Return a new ConfigFile with non-None flag values overriding cfg_in."""
    supplied = {k: v for k, v in flags.items() if v is not None}
    return replace(cfg_in, **supplied)


def ensure_vectors_installed(config: ConfigFile, *, strict: bool) -> list[str]:
    """Install any vectors in config.vectors that are not present locally.

    Returns a list of coords that could not be installed. In strict mode,
    raises on any failure instead of returning.
    """
    from saklas.io.paths import concept_dir
    from saklas.io.packs import materialize_bundled
    from saklas.io import cache_ops

    missing: list[str] = []
    for coord, _alpha in config.vectors.items():
        if "/" not in coord:
            # Bare name: resolve across namespaces. If a match exists locally,
            # treat as installed; otherwise require an explicit <ns>/<name>.
            from saklas.cli.selectors import _all_concepts
            matches = [c for c in _all_concepts() if c.name == coord]
            if matches:
                continue
            msg = f"vector {coord!r}: must be '<ns>/<name>' (no installed match)"
            if strict:
                raise ConfigFileError(msg)
            log.warning(msg)
            missing.append(coord)
            continue
        ns, name = coord.split("/", 1)
        cdir = concept_dir(ns, name)
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
