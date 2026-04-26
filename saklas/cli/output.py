"""CLI presentation helpers for pack listing / info / search.

``cache_ops`` returns structured data (``ConceptRow`` / ``ConceptInfo`` /
``PackListResult`` etc); this module turns it into text or JSON for stdout.
Programmatic callers of ``cache_ops`` get the structured shape; only the
CLI runners format it.
"""
from __future__ import annotations

import json as _json
import sys
from typing import Any

from saklas.io.cache_ops import (
    ConceptInfo,
    ConceptRow,
    HfRow,
    PackListResult,
    list_concepts as _list_concepts,
    pack_info as _pack_info,
    search_remote_packs as _search_remote_packs,
)
from saklas.io.selectors import Selector


# ---------------------------------------------------------------------------
# Internal print helpers — pure formatters over the structured shapes.
# ---------------------------------------------------------------------------

def _row_dict_from_concept(row: ConceptRow) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": row.name,
        "namespace": row.namespace,
        "status": row.status,
        "recommended_alpha": row.recommended_alpha,
        "tags": list(row.tags),
        "description": row.description,
        "source": row.source,
        "tensor_models": list(row.tensor_models),
    }
    if row.error is not None:
        payload["error"] = row.error
    return payload


def _row_dict_from_hf(row: HfRow) -> dict[str, Any]:
    return {
        "name": row.name,
        "namespace": row.namespace,
        "status": "hf",
        "recommended_alpha": row.recommended_alpha,
        "tags": list(row.tags),
        "description": row.description,
        "tensor_models": list(row.tensor_models),
    }


def _print_list(rows: list[ConceptRow], *, verbose: bool = False) -> None:
    if verbose:
        print(f"{'NAME':<24} {'NS':<12} {'STATUS':<13} {'ALPHA':<6} {'TAGS':<24} DESCRIPTION")
    else:
        print(f"{'NAME':<24} {'NS':<12} {'STATUS':<13} {'ALPHA':<6} TAGS")
    for r in rows:
        tags = ",".join(r.tags)
        tag = "[corrupt]    " if r.status == "corrupt" else "[installed]  "
        line = (
            f"{r.name:<24} {r.namespace:<12} {tag} "
            f"{r.recommended_alpha:<6.2f} {tags}"
        )
        if verbose:
            line = (
                f"{r.name:<24} {r.namespace:<12} {tag} "
                f"{r.recommended_alpha:<6.2f} {tags:<24} {r.description}"
            )
        print(line)
        if r.error:
            print(f"  ! {r.error}")


def _print_hf_rows(rows: list[HfRow], *, verbose: bool = False) -> None:
    for r in rows:
        line = (
            f"{r.name:<24} {r.namespace:<12} [hf]          "
            f"{r.recommended_alpha:<6.2f} {','.join(r.tags)}"
        )
        if verbose:
            line = (
                f"{r.name:<24} {r.namespace:<12} [hf]          "
                f"{r.recommended_alpha:<6.2f} "
                f"{','.join(r.tags):<24} {r.description}"
            )
        print(line)


def _print_info(info: ConceptInfo) -> None:
    tag = "installed" if info.status == "installed" else info.status
    print(f"{info.namespace}/{info.name} [{tag}]")
    print(f"  description: {info.description}")
    if info.long_description:
        print(f"  long:        {info.long_description}")
    print(f"  tags:        {', '.join(info.tags) or '(none)'}")
    if info.recommended_alpha is not None:
        print(f"  alpha:       {info.recommended_alpha}")
    if info.source:
        print(f"  source:      {info.source}")
    print(f"  tensors:     {', '.join(info.tensor_models) or '(none)'}")


def _info_to_dict(info: ConceptInfo) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": info.name,
        "namespace": info.namespace,
        "status": info.status,
        "description": info.description,
        "tags": list(info.tags),
        "tensor_models": list(info.tensor_models),
    }
    if info.long_description:
        payload["long_description"] = info.long_description
    if info.recommended_alpha is not None:
        payload["recommended_alpha"] = info.recommended_alpha
    if info.source:
        payload["source"] = info.source
    return payload


# ---------------------------------------------------------------------------
# Public CLI entry points (replace the old cache_ops print sites).
# ---------------------------------------------------------------------------

def render_concept_list(
    selector: Selector | None,
    *,
    hf: bool = True,
    installed_only: bool = False,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Print (or JSON-dump) concepts matching the selector.

    Mirrors the old ``cache_ops.list_concepts`` UX: name-with-namespace
    selectors render as info pages, broader selectors as a table.
    """
    if hf and installed_only:
        hf = False

    # Single-concept name selectors render as info pages.
    if (
        selector is not None
        and selector.kind == "name"
        and selector.namespace is not None
        and selector.value is not None
    ):
        info = _pack_info(selector.namespace, selector.value, hf=hf)
        if info is None:
            print(f"not found: {selector.namespace}/{selector.value}")
            return
        if json_output:
            print(_json.dumps(_info_to_dict(info), indent=2))
            return
        _print_info(info)
        return

    result: PackListResult = _list_concepts(selector, hf=hf)

    if json_output:
        payload = [_row_dict_from_concept(r) for r in result.installed]
        if result.error:
            print(_json.dumps({"error": result.error, "installed": payload}))
            return
        installed_keys = {(r["namespace"], r["name"]) for r in payload}
        for hf_row in result.hf_rows:
            if (hf_row.namespace, hf_row.name) in installed_keys:
                continue
            payload.append(_row_dict_from_hf(hf_row))
        print(_json.dumps(payload, indent=2))
        return

    _print_list(result.installed, verbose=verbose)
    if result.error:
        print(f"({result.error})")
        return
    if result.hf_rows:
        _print_hf_rows(result.hf_rows, verbose=verbose)


def render_local_pack_list(
    selector: Selector | None,
    *,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Print installed packs only (no HF query)."""
    render_concept_list(
        selector,
        hf=False,
        installed_only=True,
        json_output=json_output,
        verbose=verbose,
    )


def render_remote_search(
    query: str,
    *,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Search the HF hub for saklas-pack repos and print results."""
    try:
        result = _search_remote_packs(query)
    except ImportError as e:
        print(f"saklas pack search unavailable: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"hf search failed: {type(e).__name__}: {e}", file=sys.stderr)
        return
    if json_output:
        print(_json.dumps([_row_dict_from_hf(r) for r in result], indent=2))
        return
    if not result:
        print("(no matches)")
        return
    _print_hf_rows(result, verbose=verbose)
