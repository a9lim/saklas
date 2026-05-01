"""DataFrame coercions for ``ResultCollector`` and bare result lists.

Thin wrapper layer over the existing structured types — keeps the
notebook's plot functions decoupled from how the user collected their
results (programmatic ``ResultCollector``, hand-built list, JSONL on
disk, or already a DataFrame).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import pandas as pd

    from saklas.core.results import GenerationResult, ResultCollector


def _require_pandas() -> Any:
    """Lazy ``pandas`` import.  Raises with the install hint when missing."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        from saklas.notebook.plots import NotebookExtraNotInstalled

        raise NotebookExtraNotInstalled("pandas") from None
    return pd


def to_dataframe(
    source: "Union[ResultCollector, list[GenerationResult], list[dict[str, Any]], pd.DataFrame]",
) -> "pd.DataFrame":
    """Coerce the common result containers into a flat pandas DataFrame.

    Accepted shapes:

    * :class:`saklas.core.results.ResultCollector` — delegates to its own
      ``to_dataframe`` (the single source of truth for column naming).
    * ``list[GenerationResult]`` — wraps each through a transient
      ``ResultCollector`` so the column shape matches programmatic
      collection.
    * ``list[dict[str, Any]]`` — already-flat row dicts (e.g. JSONL
      readback).  Pass through ``pd.DataFrame``.
    * ``pd.DataFrame`` — passes through unchanged.

    Raises :class:`NotebookExtraNotInstalled` when pandas is missing.
    """
    pd = _require_pandas()

    # Already a DataFrame: pass through.  Duck-typed so we don't force a
    # pandas import at module load when the source is something else.
    if hasattr(source, "to_records") and hasattr(source, "columns"):
        return source  # type: ignore[return-value]

    # ResultCollector: delegate to its own to_dataframe so column naming
    # stays in one place.  Imported here (not at module top) so users
    # without pandas don't pay an import-time cost.
    from saklas.core.results import GenerationResult, ResultCollector

    if isinstance(source, ResultCollector):
        return source.to_dataframe()

    if isinstance(source, list):
        if not source:
            return pd.DataFrame()
        first = source[0]
        if isinstance(first, dict):
            return pd.DataFrame(source)
        if isinstance(first, GenerationResult):
            # Route through a transient collector so column names match
            # what users get from session.generate_batch flows.
            rc = ResultCollector()
            for r in source:
                if not isinstance(r, GenerationResult):
                    raise TypeError(
                        f"to_dataframe: list contains mixed types "
                        f"({type(r).__name__}); expected uniform "
                        f"GenerationResult or dict"
                    )
                rc.add(r)
            return rc.to_dataframe()

    raise TypeError(
        f"to_dataframe: unsupported source type {type(source).__name__}; "
        f"expected ResultCollector, list[GenerationResult], "
        f"list[dict], or pd.DataFrame"
    )
