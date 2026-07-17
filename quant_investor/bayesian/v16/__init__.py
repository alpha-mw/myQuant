"""Explicit, unactivated Bayesian v16 contract.

Importing this package does not change the authoritative v15 runtime defaults.
"""

from quant_investor.bayesian.v16.branch_config import (
    BRANCH_WEIGHT_VERSION,
    CANONICAL_BRANCH_ORDER,
    DEFAULT_BRANCH_WEIGHTS,
)
from quant_investor.bayesian.v16.bootstrap import BlockBootstrapArtifact
from quant_investor.bayesian.v16.calibration import (
    CalibrationBucket,
    CalibrationObservation,
    CalibrationStore,
    LikelihoodCalibrationModel,
)
from quant_investor.bayesian.v16.likelihood import (
    BranchLikelihoodInput,
    SignalLikelihoodMapper,
)
from quant_investor.bayesian.v16.posterior import (
    BayesianPosteriorEngine,
    CostComponents,
    compute_correlation_vif,
    compute_equal_weight_evidence_increments,
)
from quant_investor.bayesian.v16.prior import BaseRateEvidence, BaseRatePriorBuilder
from quant_investor.bayesian.v16.return_calibration import (
    ArtifactReturnCalibration,
    ReturnCalibrationEstimate,
    ReturnCalibrationModel,
    RobustReturnModelArtifact,
)
from quant_investor.bayesian.v16.types import (
    LikelihoodSet,
    PosteriorResult,
    PriorSet,
)
from quant_investor.bayesian.v16.versioning import ARCHITECTURE_VERSION

__all__ = [
    "ARCHITECTURE_VERSION",
    "ArtifactReturnCalibration",
    "BRANCH_WEIGHT_VERSION",
    "BaseRateEvidence",
    "BaseRatePriorBuilder",
    "BayesianPosteriorEngine",
    "BlockBootstrapArtifact",
    "BranchLikelihoodInput",
    "CalibrationBucket",
    "CalibrationObservation",
    "CalibrationStore",
    "CANONICAL_BRANCH_ORDER",
    "CostComponents",
    "DEFAULT_BRANCH_WEIGHTS",
    "LikelihoodSet",
    "LikelihoodCalibrationModel",
    "PosteriorResult",
    "PriorSet",
    "ReturnCalibrationEstimate",
    "ReturnCalibrationModel",
    "RobustReturnModelArtifact",
    "SignalLikelihoodMapper",
    "compute_correlation_vif",
    "compute_equal_weight_evidence_increments",
]
