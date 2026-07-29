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
from .factor_regime_diagnostics import (
    FactorRegimeDiagnosticError,
    build_regime_conditioned_factor_diagnostic,
    build_unavailable_regime_conditioned_factor_diagnostic,
    validate_regime_conditioned_factor_diagnostic,
    validate_regime_conditioned_factor_diagnostic_replay,
)
from .factor_regime_origin_inventory import (
    ContentArtifactRef,
    FactorRegimeOriginInput,
    FactorRegimeOriginInventoryError,
    RegimeEvidenceSnapshot,
    build_factor_regime_origin_inventory,
    validate_factor_regime_origin_inventory,
    validate_factor_regime_origin_inventory_replay,
)
from .v4_factor_adapter import (
    V4ArtifactReference,
    V4FactorAdaptationStatus,
    V4FactorAdapterError,
    V4FactorEvidenceAdaptation,
    V4FactorOriginBinding,
    adapt_v4_factor_evidence,
    build_factor_diagnostic_from_v4,
)
from .v4_regime_adapter import (
    NormalizedV4RegimeEvidence,
    V4RegimeAdapterError,
    V4RegimeEvidenceStatus,
    adapt_v4_regime_evidence,
)
from .v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityError,
    V4CompatibilityRead,
    read_v4_artifact,
)

__all__ = [
    "DELIVERY_STATUS",
    "ContentArtifactRef",
    "FactorDiagnosticError",
    "FactorDiagnosticStatus",
    "FactorOriginSample",
    "FactorSampleStratum",
    "FactorLifecycleDiagnosticError",
    "FactorLifecycleDiagnosticStatus",
    "FactorRegimeDiagnosticError",
    "FactorRegimeOriginInput",
    "FactorRegimeOriginInventoryError",
    "GLOBAL_ACTIVATION_STATE",
    "RUN_STATE",
    "RegimeEvidenceSnapshot",
    "STATE",
    "V4ClosureNode",
    "V4CompatibilityError",
    "V4CompatibilityRead",
    "V4ArtifactReference",
    "V4FactorAdaptationStatus",
    "V4FactorAdapterError",
    "V4FactorEvidenceAdaptation",
    "V4FactorOriginBinding",
    "V4RegimeAdapterError",
    "V4RegimeEvidenceStatus",
    "NormalizedV4RegimeEvidence",
    "adapt_v4_factor_evidence",
    "adapt_v4_regime_evidence",
    "authority_envelope",
    "build_factor_diagnostic",
    "build_factor_diagnostic_from_v4",
    "build_factor_lifecycle_diagnostic",
    "build_factor_regime_origin_inventory",
    "build_regime_conditioned_factor_diagnostic",
    "build_unavailable_factor_diagnostic",
    "build_unavailable_factor_lifecycle_diagnostic",
    "build_unavailable_regime_conditioned_factor_diagnostic",
    "read_v4_artifact",
    "validate_factor_diagnostic_replay",
    "validate_factor_lifecycle_diagnostic_replay",
    "validate_factor_regime_origin_inventory",
    "validate_factor_regime_origin_inventory_replay",
    "validate_regime_conditioned_factor_diagnostic",
    "validate_regime_conditioned_factor_diagnostic_replay",
]
