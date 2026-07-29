"""Research-only V17 v5 runtime boundary."""

from __future__ import annotations

from .authority import (
    DELIVERY_STATUS,
    GLOBAL_ACTIVATION_STATE,
    RUN_STATE,
    STATE,
    authority_envelope,
)
from .factor_diagnostics import (
    FactorDiagnosticError,
    FactorDiagnosticStatus,
    FactorOriginSample,
    FactorSampleStratum,
    build_factor_diagnostic,
    build_unavailable_factor_diagnostic,
    validate_factor_diagnostic_replay,
)
from .factor_lifecycle import (
    FactorLifecycleDiagnosticError,
    FactorLifecycleDiagnosticStatus,
    build_factor_lifecycle_diagnostic,
    build_unavailable_factor_lifecycle_diagnostic,
    validate_factor_lifecycle_diagnostic_replay,
)
from .v4_factor_adapter import (
    V4FactorAdaptationStatus,
    V4FactorAdapterError,
    V4FactorEvidenceAdaptation,
    adapt_v4_factor_evidence,
    build_factor_diagnostic_from_v4,
)
from .v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityError,
    V4CompatibilityRead,
    read_v4_artifact,
)

__all__ = [
    "DELIVERY_STATUS",
    "FactorDiagnosticError",
    "FactorDiagnosticStatus",
    "FactorOriginSample",
    "FactorSampleStratum",
    "FactorLifecycleDiagnosticError",
    "FactorLifecycleDiagnosticStatus",
    "GLOBAL_ACTIVATION_STATE",
    "RUN_STATE",
    "STATE",
    "V4ClosureNode",
    "V4CompatibilityError",
    "V4CompatibilityRead",
    "V4FactorAdaptationStatus",
    "V4FactorAdapterError",
    "V4FactorEvidenceAdaptation",
    "adapt_v4_factor_evidence",
    "authority_envelope",
    "build_factor_diagnostic",
    "build_factor_diagnostic_from_v4",
    "build_factor_lifecycle_diagnostic",
    "build_unavailable_factor_diagnostic",
    "build_unavailable_factor_lifecycle_diagnostic",
    "read_v4_artifact",
    "validate_factor_diagnostic_replay",
    "validate_factor_lifecycle_diagnostic_replay",
]
