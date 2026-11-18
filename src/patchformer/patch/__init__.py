"""Independent, schema-aware JSON Patch validation and dry-run application."""

from .oracle import PatchValidationError, apply_patch, validate_patch

__all__ = ["PatchValidationError", "apply_patch", "validate_patch"]
