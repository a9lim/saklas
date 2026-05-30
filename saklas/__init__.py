"""saklas — local activation steering + trait monitoring for HuggingFace causal LMs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "3.2.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    "SaklasSession": ("saklas.core.session", "SaklasSession"),
    "SaklasError": ("saklas.core.errors", "SaklasError"),
    "GenState": ("saklas.core.session", "GenState"),
    "LayerWhitener": ("saklas.core.mahalanobis", "LayerWhitener"),
    "WhitenerError": ("saklas.core.mahalanobis", "WhitenerError"),
    "Profile": ("saklas.core.profile", "Profile"),
    "ProfileError": ("saklas.core.profile", "ProfileError"),
    "SamplingConfig": ("saklas.core.sampling", "SamplingConfig"),
    "Steering": ("saklas.core.steering", "Steering"),
    "Trigger": ("saklas.core.triggers", "Trigger"),
    "EventBus": ("saklas.core.events", "EventBus"),
    "VectorExtracted": ("saklas.core.events", "VectorExtracted"),
    "SteeringApplied": ("saklas.core.events", "SteeringApplied"),
    "SteeringCleared": ("saklas.core.events", "SteeringCleared"),
    "ProbeScored": ("saklas.core.events", "ProbeScored"),
    "GenerationStarted": ("saklas.core.events", "GenerationStarted"),
    "GenerationFinished": ("saklas.core.events", "GenerationFinished"),
    "DataSource": ("saklas.io.datasource", "DataSource"),
    "GenerationResult": ("saklas.core.results", "GenerationResult"),
    "RunSet": ("saklas.core.results", "RunSet"),
    "TokenAlt": ("saklas.core.results", "TokenAlt"),
    "TokenEvent": ("saklas.core.results", "TokenEvent"),
    "ProbeReadings": ("saklas.core.results", "ProbeReadings"),
    "Manifold": ("saklas.core.manifold", "Manifold"),
    "ManifoldTokenReading": ("saklas.core.results", "ManifoldTokenReading"),
    "ManifoldAggregate": ("saklas.core.results", "ManifoldAggregate"),
    "ResultCollector": ("saklas.core.results", "ResultCollector"),
    "LoomTree": ("saklas.core.loom", "LoomTree"),
    "LoomNode": ("saklas.core.loom", "LoomNode"),
    "LoomMutated": ("saklas.core.loom", "LoomMutated"),
    "Recipe": ("saklas.core.loom", "Recipe"),
    "LoomTreeError": ("saklas.core.loom", "LoomTreeError"),
    "UnknownNodeError": ("saklas.core.loom", "UnknownNodeError"),
    "InvalidNodeOperationError": ("saklas.core.loom", "InvalidNodeOperationError"),
    "MutationDuringGenerationError": ("saklas.core.loom", "MutationDuringGenerationError"),
    "derive_seed_schedule": ("saklas.core.loom", "derive_seed_schedule"),
    "FilterClause": ("saklas.core.tree_filter", "FilterClause"),
    "FilterParseError": ("saklas.core.tree_filter", "FilterParseError"),
    "parse_filter": ("saklas.core.tree_filter", "parse_filter"),
    "DiffSpan": ("saklas.core.loom_diff", "DiffSpan"),
    "NodeDiff": ("saklas.core.loom_diff", "NodeDiff"),
    "ReadingDelta": ("saklas.core.loom_diff", "ReadingDelta"),
    "TokenDeltaSpan": ("saklas.core.loom_diff", "TokenDeltaSpan"),
    "per_token_diff": ("saklas.core.loom_diff", "per_token_diff"),
    "readings_diff": ("saklas.core.loom_diff", "readings_diff"),
    "steering_delta": ("saklas.core.loom_diff", "steering_delta"),
    "text_diff": ("saklas.core.loom_diff", "text_diff"),
    "ProbeRef": ("saklas.core.transcript", "ProbeRef"),
    "Transcript": ("saklas.core.transcript", "Transcript"),
    "TranscriptError": ("saklas.core.transcript", "TranscriptError"),
    "TranscriptFormatError": ("saklas.core.transcript", "TranscriptFormatError"),
    "TranscriptModelMismatch": ("saklas.core.transcript", "TranscriptModelMismatch"),
    "TranscriptProbeDriftError": ("saklas.core.transcript", "TranscriptProbeDriftError"),
    "TranscriptTurn": ("saklas.core.transcript", "Turn"),
}

__all__ = [
    "SaklasSession",
    "SaklasError",
    "GenState",
    "LayerWhitener",
    "WhitenerError",
    "Profile",
    "ProfileError",
    "SamplingConfig",
    "Steering",
    "Trigger",
    "EventBus",
    "VectorExtracted",
    "SteeringApplied",
    "SteeringCleared",
    "ProbeScored",
    "GenerationStarted",
    "GenerationFinished",
    "DataSource",
    "GenerationResult",
    "RunSet",
    "TokenAlt",
    "TokenEvent",
    "ProbeReadings",
    "Manifold",
    "ManifoldTokenReading",
    "ManifoldAggregate",
    "ResultCollector",
    "LoomTree",
    "LoomNode",
    "LoomMutated",
    "Recipe",
    "LoomTreeError",
    "UnknownNodeError",
    "InvalidNodeOperationError",
    "MutationDuringGenerationError",
    "derive_seed_schedule",
    "FilterClause",
    "FilterParseError",
    "parse_filter",
    "DiffSpan",
    "NodeDiff",
    "ReadingDelta",
    "TokenDeltaSpan",
    "per_token_diff",
    "readings_diff",
    "steering_delta",
    "text_diff",
    "ProbeRef",
    "Transcript",
    "TranscriptError",
    "TranscriptFormatError",
    "TranscriptModelMismatch",
    "TranscriptProbeDriftError",
    "TranscriptTurn",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(f"module 'saklas' has no attribute {name!r}") from e
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])


if TYPE_CHECKING:
    from saklas.core.errors import SaklasError as SaklasError
    from saklas.core.events import (
        EventBus as EventBus,
        GenerationFinished as GenerationFinished,
        GenerationStarted as GenerationStarted,
        ProbeScored as ProbeScored,
        SteeringApplied as SteeringApplied,
        SteeringCleared as SteeringCleared,
        VectorExtracted as VectorExtracted,
    )
    from saklas.core.loom import (
        InvalidNodeOperationError as InvalidNodeOperationError,
        LoomMutated as LoomMutated,
        LoomNode as LoomNode,
        LoomTree as LoomTree,
        LoomTreeError as LoomTreeError,
        MutationDuringGenerationError as MutationDuringGenerationError,
        Recipe as Recipe,
        UnknownNodeError as UnknownNodeError,
        derive_seed_schedule as derive_seed_schedule,
    )
    from saklas.core.loom_diff import (
        DiffSpan as DiffSpan,
        NodeDiff as NodeDiff,
        ReadingDelta as ReadingDelta,
        TokenDeltaSpan as TokenDeltaSpan,
        per_token_diff as per_token_diff,
        readings_diff as readings_diff,
        steering_delta as steering_delta,
        text_diff as text_diff,
    )
    from saklas.core.mahalanobis import LayerWhitener as LayerWhitener, WhitenerError as WhitenerError
    from saklas.core.manifold import Manifold as Manifold
    from saklas.core.profile import Profile as Profile, ProfileError as ProfileError
    from saklas.core.results import (
        GenerationResult as GenerationResult,
        ManifoldAggregate as ManifoldAggregate,
        ManifoldTokenReading as ManifoldTokenReading,
        ProbeReadings as ProbeReadings,
        ResultCollector as ResultCollector,
        RunSet as RunSet,
        TokenAlt as TokenAlt,
        TokenEvent as TokenEvent,
    )
    from saklas.core.sampling import SamplingConfig as SamplingConfig
    from saklas.core.session import GenState as GenState, SaklasSession as SaklasSession
    from saklas.core.steering import Steering as Steering
    from saklas.core.transcript import (
        ProbeRef as ProbeRef,
        Transcript as Transcript,
        TranscriptError as TranscriptError,
        TranscriptFormatError as TranscriptFormatError,
        TranscriptModelMismatch as TranscriptModelMismatch,
        TranscriptProbeDriftError as TranscriptProbeDriftError,
        Turn as TranscriptTurn,  # noqa: F401
    )
    from saklas.core.tree_filter import (
        FilterClause as FilterClause,
        FilterParseError as FilterParseError,
        parse_filter as parse_filter,
    )
    from saklas.core.triggers import Trigger as Trigger
    from saklas.io.datasource import DataSource as DataSource
