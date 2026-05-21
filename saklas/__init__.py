"""saklas — local activation steering + trait monitoring for HuggingFace causal LMs."""

__version__ = "3.1.0"

from saklas.core.errors import SaklasError
from saklas.core.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ProbeScored,
    SteeringApplied,
    SteeringCleared,
    VectorExtracted,
)
from saklas.core.loom import (
    InvalidNodeOperationError,
    LoomMutated,
    LoomNode,
    LoomTree,
    LoomTreeError,
    MutationDuringGenerationError,
    Recipe,
    UnknownNodeError,
    derive_seed_schedule,
)
from saklas.core.loom_diff import (
    DiffSpan,
    NodeDiff,
    ReadingDelta,
    TokenDeltaSpan,
    per_token_diff,
    readings_diff,
    steering_delta,
    text_diff,
)
from saklas.core.transcript import (
    ProbeRef,
    Transcript,
    TranscriptError,
    TranscriptFormatError,
    TranscriptModelMismatch,
    TranscriptProbeDriftError,
    Turn as TranscriptTurn,
)
from saklas.core.tree_filter import (
    FilterClause,
    FilterParseError,
    parse_filter,
)
from saklas.core.mahalanobis import LayerWhitener, WhitenerError
from saklas.core.profile import Profile, ProfileError
from saklas.core.results import (
    GenerationResult,
    ProbeReadings,
    ResultCollector,
    RunSet,
    TokenAlt,
    TokenEvent,
)
from saklas.core.sampling import SamplingConfig
from saklas.core.session import GenState, SaklasSession
from saklas.core.steering import Steering
from saklas.core.triggers import Trigger
from saklas.io.datasource import DataSource

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
    "ResultCollector",
    # Loom (v2.3) — engine-side tree of conversation nodes
    "LoomTree",
    "LoomNode",
    "LoomMutated",
    "Recipe",
    "LoomTreeError",
    "UnknownNodeError",
    "InvalidNodeOperationError",
    "MutationDuringGenerationError",
    "derive_seed_schedule",
    # Loom phase 5 — filter grammar, cross-branch diff, transcript IO
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
