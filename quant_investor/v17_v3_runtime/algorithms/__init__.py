"""Pure v17 v3 algorithm layer.

This package intentionally has no dependency on ``v17_v2_*`` modules and
performs no file, provider, broker, or network I/O.
"""

from .branch_fusion import (
    BranchFusionError,
    BranchOutput,
    BranchRecord,
    FusionDisposition,
    FusionResult,
    fuse_branches,
    validate_branch_output,
)
from .calibration import (
    BOOTSTRAP_MATRIX_HEADER,
    BOOTSTRAP_MATRIX_VERSION,
    CANDIDATE_QUANT_WEIGHTS,
    CalibrationError,
    CalibrationFold,
    CalibrationMonth,
    CalibrationResult,
    MonthEndOrigin,
    MonthlyFusionMetric,
    WeightAssessment,
    bootstrap_matrix_header_bytes,
    bootstrap_matrix_sha256,
    calibrate_fusion,
    circular_moving_block_bootstrap_matrix,
    schedule_month_end_origins,
    select_fusion_weight,
)
from .decimal_normalization import (
    DECIMAL_PRECISION,
    DECIMAL_QUANTUM,
    canonical_decimal_string,
    normalize_decimal,
)
from .deep_research import DeepResearchDecision, evaluate_deep_research
from .overlay import OverlayValidation, validate_monotonic_overlay
from .quant_preselection import (
    FactorIdentity,
    FactorInventoryConflict,
    FactorSpec,
    QuantDisposition,
    QuantPreselectionError,
    QuantPreselectionResult,
    SymbolObservation,
    run_quant_preselection,
    validate_disjoint_factor_inventories,
    validate_factor_inventory,
)

__all__ = [
    "BOOTSTRAP_MATRIX_HEADER",
    "BOOTSTRAP_MATRIX_VERSION",
    "CANDIDATE_QUANT_WEIGHTS",
    "DECIMAL_PRECISION",
    "DECIMAL_QUANTUM",
    "BranchFusionError",
    "BranchOutput",
    "BranchRecord",
    "CalibrationError",
    "CalibrationFold",
    "CalibrationMonth",
    "CalibrationResult",
    "DeepResearchDecision",
    "FactorIdentity",
    "FactorInventoryConflict",
    "FactorSpec",
    "FusionDisposition",
    "FusionResult",
    "MonthEndOrigin",
    "MonthlyFusionMetric",
    "OverlayValidation",
    "QuantDisposition",
    "QuantPreselectionError",
    "QuantPreselectionResult",
    "SymbolObservation",
    "WeightAssessment",
    "bootstrap_matrix_header_bytes",
    "bootstrap_matrix_sha256",
    "calibrate_fusion",
    "canonical_decimal_string",
    "circular_moving_block_bootstrap_matrix",
    "evaluate_deep_research",
    "fuse_branches",
    "normalize_decimal",
    "run_quant_preselection",
    "schedule_month_end_origins",
    "select_fusion_weight",
    "validate_branch_output",
    "validate_disjoint_factor_inventories",
    "validate_factor_inventory",
    "validate_monotonic_overlay",
]
