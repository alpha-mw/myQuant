import importlib

from quant_investor.factors.store import (
    FactorAlignmentAuditStore,
    FactorBacktestArtifactStore,
    FactorCorrelationContributionStore,
    FactorEvidenceStore,
    FactorExecutionCostSimulationStore,
    FactorGovernanceStore,
    FactorLibraryAuditStore,
    FactorMatrixStore,
    FactorShadowScoringStore,
    FactorTradabilityAuditStore,
    FactorValidationArtifactStore,
)


def test_factor_stores_share_json_artifact_helpers() -> None:
    helpers = importlib.import_module("quant_investor.factors.store_helpers")
    store_classes = [
        FactorGovernanceStore,
        FactorMatrixStore,
        FactorBacktestArtifactStore,
        FactorValidationArtifactStore,
        FactorCorrelationContributionStore,
        FactorLibraryAuditStore,
        FactorShadowScoringStore,
        FactorEvidenceStore,
        FactorAlignmentAuditStore,
        FactorTradabilityAuditStore,
        FactorExecutionCostSimulationStore,
    ]
    for store_cls in store_classes:
        assert issubclass(store_cls, helpers.JsonArtifactStoreMixin)
        assert store_cls._append_jsonl is helpers.JsonArtifactStoreMixin._append_jsonl
        assert store_cls._read_jsonl_payloads is helpers.JsonArtifactStoreMixin._read_jsonl_payloads
