"""Project integration for the Narcissus clean-label backdoor attack."""

from .integration import (
    TriggerGenerationConfig,
    apply_trigger_array,
    generate_trigger,
    load_trigger_artifact,
    trigger_sha256,
    validate_trigger,
)

__all__ = [
    "TriggerGenerationConfig",
    "apply_trigger_array",
    "generate_trigger",
    "load_trigger_artifact",
    "trigger_sha256",
    "validate_trigger",
]
