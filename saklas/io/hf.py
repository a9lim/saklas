"""Shared Hugging Face Hub primitives — the single HF seam for :mod:`saklas.io`.

Every Hub call in the package routes through the three indirections here
(``_hf_snapshot_download`` / ``_hf_hub_download`` / ``_hf_api``): manifold
distribution (:mod:`saklas.io.hf_manifolds`) and the external J-lens source
registry (:mod:`saklas.io.lens_sources`) both import them under these exact
names, so a test monkeypatching ``<module>._hf_hub_download`` blocks the
network on either path.  New HF work belongs here, not in a second local
copy.  Also owns the ``HFError`` type, the ``owner/name@revision`` splitter,
and the push-coord resolver.
"""
from __future__ import annotations

from typing import Any, Optional

from saklas.core.errors import SaklasError


class HFError(RuntimeError, SaklasError):
    def user_message(self) -> tuple[int, str]:
        return (502, str(self) or self.__class__.__name__)


def split_revision(target: str) -> tuple[str, Optional[str]]:
    """Split an ``owner/name@revision`` target into (coord, revision).

    Revisions can be any git ref HF accepts: tag, branch, or commit SHA.
    Concept/manifold names are restricted to ``NAME_REGEX``, so ``@`` is
    unambiguous as a separator.
    """
    if "@" not in target:
        return target, None
    coord, _, rev = target.partition("@")
    if not rev:
        raise HFError(f"empty revision after '@' in {target!r}")
    return coord, rev


def _hf_snapshot_download(
    repo_id: str, *, repo_type: str = "model", **kwargs: Any,
) -> str:
    """Thin indirection so tests can monkeypatch."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type=repo_type, **kwargs)


def _hf_hub_download(
    repo_id: str, filename: str, *, repo_type: str = "model", **kwargs: Any,
) -> str:
    """Thin indirection so tests can monkeypatch."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type=repo_type, **kwargs,
    )


def _hf_api():
    """Thin indirection so tests can monkeypatch."""
    from huggingface_hub import HfApi
    return HfApi()


def resolve_target_coord(name: str, as_: Optional[str]) -> str:
    """Decide the HF coord to push to. ``--as owner/name`` wins; else whoami()/<name>."""
    if as_:
        if "/" not in as_:
            raise HFError(f"--as must be '<owner>/<name>', got {as_!r}")
        return as_
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami()
    except Exception as e:
        raise HFError(
            f"could not resolve HF username ({e}); pass --as owner/name or run `hf auth login`"
        ) from e
    user = who.get("name") if hasattr(who, "get") else None
    if not user:
        raise HFError("could not resolve HF username; pass --as owner/name")
    return f"{user}/{name}"
