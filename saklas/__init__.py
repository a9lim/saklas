"""saklas — local activation steering + trait monitoring for HuggingFace causal LMs."""

__version__ = "1.4.0"

from saklas.errors import SaklasError
from saklas.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ProbeScored,
    SteeringApplied,
    SteeringCleared,
    VectorExtracted,
)
from saklas.profile import Profile, ProfileError
from saklas.sampling import SamplingConfig
from saklas.session import SaklasSession
from saklas.steering import Steering
from saklas.datasource import DataSource
from saklas.results import GenerationResult, TokenEvent, ProbeReadings, ResultCollector

__all__ = [
    "SaklasSession",
    "SaklasError",
    "Profile",
    "ProfileError",
    "SamplingConfig",
    "Steering",
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
