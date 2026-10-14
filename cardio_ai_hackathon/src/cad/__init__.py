"""CAD hackathon prototype package. Educational decision support only; not a diagnostic tool."""

from .inference import (  # noqa: F401
    ArtifactError,
    CADPredictor,
    InputValidationError,
    load_config,
    synthetic_example,
)
