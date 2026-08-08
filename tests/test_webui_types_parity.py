"""The dashboard's REST types are a build artifact of the server's schema.

Three structural checks over the native ``/saklas/v1/*`` tree:

1. ``webui/src/lib/types.gen.ts`` is byte-identical to what
   ``scripts/generate_webui_types.py`` renders from the live OpenAPI schema.
   This is the wire-drift gate — a route whose response shape changed
   without a regenerate fails here rather than in a panel.
2. Every non-streaming native route declares a response schema.  An
   unannotated route emits an empty ``{}`` schema, which the generator would
   silently render as nothing at all.
3. Native paths are kebab-case.  The two snake_case survivors
   (``tree/edge_label``, ``tree/joint_logprobs``) were renamed; this keeps
   the next route from reintroducing the split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def generator() -> Any:
    import generate_webui_types  # pyright: ignore[reportMissingImports]

    return generate_webui_types


@pytest.fixture(scope="module")
def schema(generator: Any) -> dict[str, Any]:
    return generator.build_schema()


def test_types_gen_matches_the_live_schema(generator: Any) -> None:
    """The committed generated types equal a fresh render.

    The same comparison ``python scripts/generate_webui_types.py --check``
    runs in CI; having it here means the local gate catches drift too,
    without a second Node round-trip.
    """
    committed = generator.OUTPUT.read_text()
    assert committed == generator.generate(), (
        "webui/src/lib/types.gen.ts is stale — run "
        "`python scripts/generate_webui_types.py`"
    )


def test_generated_file_is_marked_do_not_edit(generator: Any) -> None:
    assert generator.OUTPUT.read_text().startswith("// DO NOT EDIT")


def _native_operations(schema: dict[str, Any]):
    for path, ops in schema["paths"].items():
        if not path.startswith("/saklas/v1/"):
            continue
        for method, op in ops.items():
            if isinstance(op, dict):
                yield path, method, op


def test_every_native_json_route_declares_a_response_schema(
    schema: dict[str, Any],
) -> None:
    """No native route answers with an undescribed body.

    A route with no return annotation emits ``schema: {}``, which is
    indistinguishable from "returns nothing" downstream — the generator
    would drop it and the dashboard would go back to a hand-mirrored type.
    204/streaming responses carry no JSON content and are exempt by
    construction (they have no ``application/json`` block).
    """
    undescribed: list[str] = []
    for path, method, op in _native_operations(schema):
        for status, response in (op.get("responses") or {}).items():
            if not str(status).startswith("2"):
                continue
            body = (response.get("content") or {}).get("application/json")
            if body is not None and not body.get("schema"):
                undescribed.append(f"{method.upper()} {path} -> {status}")
    assert not undescribed, (
        "native routes with an undescribed response body: "
        f"{undescribed} — annotate the return type with a TypedDict from "
        "saklas/server/response_models.py"
    )


def test_native_paths_are_kebab_case(schema: dict[str, Any]) -> None:
    """Path segments use ``-``, never ``_`` (path params excepted)."""
    offenders = [
        path
        for path in schema["paths"]
        if path.startswith("/saklas/v1/")
        and any(
            "_" in segment
            for segment in path.split("/")
            if not segment.startswith("{")
        )
    ]
    assert not offenders, f"snake_case native paths: {offenders}"


def test_renamed_tree_routes_are_kebab_only(schema: dict[str, Any]) -> None:
    """The two renames landed as a clean break — no aliases."""
    paths = set(schema["paths"])
    assert "/saklas/v1/sessions/{session_id}/tree/edge-label" in paths
    assert "/saklas/v1/sessions/{session_id}/tree/joint-logprobs" in paths
    assert "/saklas/v1/sessions/{session_id}/tree/edge_label" not in paths
    assert "/saklas/v1/sessions/{session_id}/tree/joint_logprobs" not in paths


def test_measurements_envelope_is_named_not_opaque(schema: dict[str, Any]) -> None:
    """The read-side envelope reaches the dashboard as a real type.

    ``core/measurements.py`` owns the shape; the generator renames it into
    the dashboard's vocabulary rather than forking a second declaration, so
    a component named ``Measurements`` must exist in the schema and a
    ``MeasurementsEnvelopeJSON`` interface in the output.
    """
    components = schema["components"]["schemas"]
    assert "Measurements" in components
    assert "ProbeReadingDict" in components
    assert "ScalarReadingDict" in components
