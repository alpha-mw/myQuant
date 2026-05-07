"""Offline factor governance contracts and stores."""

from quant_investor.factors import backtest as _backtest
from quant_investor.factors import capacity as _capacity
from quant_investor.factors import contribution as _contribution
from quant_investor.factors import correlation as _correlation
from quant_investor.factors import expression as _expression
from quant_investor.factors import library as _library
from quant_investor.factors import matrix as _matrix
from quant_investor.factors import metrics as _metrics
from quant_investor.factors import operators as _operators
from quant_investor.factors import report as _report
from quant_investor.factors import robustness as _robustness
from quant_investor.factors import schema as _schema
from quant_investor.factors.admission import (
    build_library_entry_from_decision,
    build_production_factor_library,
    evaluate_backtest_against_thresholds,
    propose_admission_decision,
)
from quant_investor.factors.backtest import *  # noqa: F403
from quant_investor.factors.capacity import *  # noqa: F403
from quant_investor.factors.contribution import *  # noqa: F403
from quant_investor.factors.correlation import *  # noqa: F403
from quant_investor.factors.expression import *  # noqa: F403
from quant_investor.factors.library import *  # noqa: F403
from quant_investor.factors.matrix import *  # noqa: F403
from quant_investor.factors.metrics import *  # noqa: F403
from quant_investor.factors.operators import *  # noqa: F403
from quant_investor.factors.report import *  # noqa: F403
from quant_investor.factors.robustness import *  # noqa: F403
from quant_investor.factors.schema import *  # noqa: F403
from quant_investor.factors.store import (
    FactorBacktestArtifactStore,
    FactorCorrelationContributionStore,
    FactorGovernanceStore,
    FactorLibraryAuditStore,
    FactorMatrixStore,
    FactorValidationArtifactStore,
)

__all__ = [
    *_schema.__all__,
    *_backtest.__all__,
    *_capacity.__all__,
    *_correlation.__all__,
    *_contribution.__all__,
    *_matrix.__all__,
    *_metrics.__all__,
    *_operators.__all__,
    *_expression.__all__,
    *_library.__all__,
    *_robustness.__all__,
    *_report.__all__,
    "FactorGovernanceStore",
    "FactorMatrixStore",
    "FactorBacktestArtifactStore",
    "FactorValidationArtifactStore",
    "FactorCorrelationContributionStore",
    "FactorLibraryAuditStore",
    "evaluate_backtest_against_thresholds",
    "propose_admission_decision",
    "build_library_entry_from_decision",
    "build_production_factor_library",
]
