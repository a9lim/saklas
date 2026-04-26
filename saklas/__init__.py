"""saklas — local activation steering + trait monitoring for HuggingFace causal LMs."""

__version__ = "1.5.0"

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
from saklas.core.profile import Profile, ProfileError
from saklas.core.sampling import SamplingConfig
from saklas.core.session import SaklasSession
from saklas.core.steering import Steering
from saklas.core.triggers import Trigger
from saklas.io.datasource import DataSource
from saklas.core.results import GenerationResult, TokenEvent, ProbeReadings, ResultCollector

__all__ = [
    "SaklasSession",
    "SaklasError",
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
    "TokenEvent",
    "ProbeReadings",
    "ResultCollector",
]
