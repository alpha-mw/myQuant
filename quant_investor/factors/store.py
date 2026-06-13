"""Append-only local store for offline factor governance artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.store_helpers import JsonArtifactStoreMixin
from quant_investor.factors.backtest import (
    DEFAULT_FACTOR_BACKTEST_DIR,
    DEFAULT_FACTOR_BACKTEST_RUNS_FILENAME,
    DEFAULT_FACTOR_DAILY_RECORDS_FILENAME,
    DEFAULT_FACTOR_WEIGHT_MATRICES_FILENAME,
    FactorDailyBacktestRecord,
    FactorWeightMatrix,
    SingleFactorBacktestRun,
)
from quant_investor.factors.alignment_audit import (
    DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME,
    DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME,
    DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR,
    FactorBacktestAlignmentAuditReport,
)
from quant_investor.factors.capacity import FactorCostCapacityReport
from quant_investor.factors.contribution import FactorPortfolioContributionReport
from quant_investor.factors.correlation import FactorRedundancyReport
from quant_investor.factors.execution_cost import (
    DEFAULT_EXECUTION_ADJUSTED_DAILY_RECORDS_FILENAME,
    DEFAULT_EXECUTION_ADJUSTED_RUNS_FILENAME,
    DEFAULT_EXECUTION_COST_DASHBOARD_FILENAME,
    DEFAULT_EXECUTION_COST_MARKDOWN_FILENAME,
    DEFAULT_EXECUTION_COST_REPORTS_FILENAME,
    DEFAULT_FACTOR_EXECUTION_COST_DIR,
    DailyExecutionCostRecord,
    ExecutionAdjustedBacktestRun,
    FactorExecutionCostSimulationReport,
)
from quant_investor.factors.evidence import (
    DEFAULT_EVIDENCE_DASHBOARD_FILENAME,
    DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME,
    DEFAULT_EVIDENCE_MARKDOWN_FILENAME,
    DEFAULT_FACTOR_EVIDENCE_DIR,
    DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME,
    FactorShadowEvidenceDateResult,
    MultiDateFactorEvidenceReport,
)
from quant_investor.factors.matrix import (
    DEFAULT_EXPRESSION_RESULTS_FILENAME,
    DEFAULT_FACTOR_MATRICES_FILENAME,
    DEFAULT_FACTOR_MATRIX_DIR,
    DEFAULT_MATRIX_BUNDLES_FILENAME,
    DEFAULT_MATRIX_CONTRACTS_FILENAME,
    ExpressionEvaluationResult,
    FactorMatrix,
    MatrixDataBundle,
    MatrixDataContract,
)
from quant_investor.factors.schema import (
    DEFAULT_DEPRECATED_FACTORS_FILENAME,
    DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME,
    DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME,
    DEFAULT_FACTOR_DEFINITIONS_FILENAME,
    DEFAULT_FACTOR_LIBRARY_DIR,
    DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME,
    DEFAULT_PRODUCTION_FACTORS_FILENAME,
    FactorAdmissionDecision,
    FactorBacktestResult,
    FactorDefinition,
    FactorValidationReport,
    ProductionFactorLibrary,
)
from quant_investor.factors.library import FactorLibraryAuditReport
from quant_investor.factors.robustness import FactorRobustnessReport
from quant_investor.factors.shadow_scoring import (
    DEFAULT_FACTOR_SHADOW_SCORING_DIR,
    DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME,
    DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME,
    DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME,
    DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME,
    DEFAULT_SHADOW_FACTOR_SCORES_FILENAME,
    ShadowCandidateScore,
    ShadowFactorScore,
    ShadowScoringComparisonReport,
)
from quant_investor.factors.tradability import (
    DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME,
    DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME,
    DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR,
    DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME,
    DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME,
    DEFAULT_TRADABILITY_MASKS_FILENAME,
    AShareTradabilityMask,
    FactorExecutionFeasibilityReport,
    FactorTradabilityAuditReport,
)


DEFAULT_FACTOR_VALIDATION_DIR = Path("data/factor_library/validation")
DEFAULT_FACTOR_ROBUSTNESS_REPORTS_FILENAME = "factor_robustness_reports.jsonl"
DEFAULT_FACTOR_COST_CAPACITY_REPORTS_FILENAME = "factor_cost_capacity_reports.jsonl"
DEFAULT_ENHANCED_VALIDATION_REPORTS_FILENAME = "enhanced_validation_reports.jsonl"
DEFAULT_FACTOR_INCREMENTAL_DIR = Path("data/factor_library/incremental")
DEFAULT_FACTOR_REDUNDANCY_REPORTS_FILENAME = "factor_redundancy_reports.jsonl"
DEFAULT_FACTOR_CONTRIBUTION_REPORTS_FILENAME = "factor_contribution_reports.jsonl"
DEFAULT_FACTOR_LIBRARY_AUDIT_DIR = Path("data/factor_library/audit")
DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME = "factor_library_audit_reports.jsonl"
DEFAULT_FACTOR_LIBRARY_AUDIT_MARKDOWN_FILENAME = "factor_library_audit_report.md"
DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME = "factor_governance_dashboard.json"


class FactorGovernanceStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_LIBRARY_DIR
        self.factor_definitions_path = self.root_dir / DEFAULT_FACTOR_DEFINITIONS_FILENAME
        self.backtest_results_path = self.root_dir / DEFAULT_FACTOR_BACKTEST_RESULTS_FILENAME
        self.validation_reports_path = self.root_dir / DEFAULT_FACTOR_VALIDATION_REPORTS_FILENAME
        self.admission_decisions_path = self.root_dir / DEFAULT_FACTOR_ADMISSION_DECISIONS_FILENAME
        self.production_library_path = self.root_dir / DEFAULT_PRODUCTION_FACTORS_FILENAME
        self.deprecated_factors_path = self.root_dir / DEFAULT_DEPRECATED_FACTORS_FILENAME

    def append_factor_definition(self, definition: FactorDefinition) -> None:
        if definition.factor_id in self.get_factor_definition_ids():
            raise ValueError(f"Duplicate factor_id in factor definitions ledger: {definition.factor_id}")
        self._append_jsonl(self.factor_definitions_path, definition.to_dict())

    def append_backtest_result(self, result: FactorBacktestResult) -> None:
        if result.result_id in self.get_backtest_result_ids():
            raise ValueError(f"Duplicate result_id in factor backtest results ledger: {result.result_id}")
        self._append_jsonl(self.backtest_results_path, result.to_dict())

    def append_validation_report(self, report: FactorValidationReport) -> None:
        if report.report_id in self.get_validation_report_ids():
            raise ValueError(f"Duplicate report_id in factor validation reports ledger: {report.report_id}")
        self._append_jsonl(self.validation_reports_path, report.to_dict())

    def append_admission_decision(self, decision: FactorAdmissionDecision) -> None:
        if decision.decision_id in self.get_admission_decision_ids():
            raise ValueError(f"Duplicate decision_id in factor admission decisions ledger: {decision.decision_id}")
        self._append_jsonl(self.admission_decisions_path, decision.to_dict())

    def save_production_library(self, library: ProductionFactorLibrary) -> Path:
        return self._write_json(self.production_library_path, library.to_dict())

    def load_production_library(self) -> ProductionFactorLibrary:
        return ProductionFactorLibrary.from_dict(self._read_json(self.production_library_path))

    def read_factor_definitions(self) -> list[FactorDefinition]:
        return [
            FactorDefinition.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.factor_definitions_path)
        ]

    def read_backtest_results(self) -> list[FactorBacktestResult]:
        return [
            FactorBacktestResult.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.backtest_results_path)
        ]

    def read_validation_reports(self) -> list[FactorValidationReport]:
        return [
            FactorValidationReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.validation_reports_path)
        ]

    def read_admission_decisions(self) -> list[FactorAdmissionDecision]:
        return [
            FactorAdmissionDecision.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.admission_decisions_path)
        ]

    def get_factor_definition_ids(self) -> set[str]:
        return {definition.factor_id for definition in self.read_factor_definitions()}

    def get_backtest_result_ids(self) -> set[str]:
        return {result.result_id for result in self.read_backtest_results()}

    def get_validation_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_validation_reports()}

    def get_admission_decision_ids(self) -> set[str]:
        return {decision.decision_id for decision in self.read_admission_decisions()}


class FactorMatrixStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_MATRIX_DIR
        self.matrix_contracts_path = self.root_dir / DEFAULT_MATRIX_CONTRACTS_FILENAME
        self.matrix_bundles_path = self.root_dir / DEFAULT_MATRIX_BUNDLES_FILENAME
        self.factor_matrices_path = self.root_dir / DEFAULT_FACTOR_MATRICES_FILENAME
        self.expression_results_path = self.root_dir / DEFAULT_EXPRESSION_RESULTS_FILENAME

    def append_matrix_contract(self, contract: MatrixDataContract) -> None:
        if contract.contract_id in self.get_matrix_contract_ids():
            raise ValueError(f"Duplicate contract_id in matrix contracts ledger: {contract.contract_id}")
        self._append_jsonl(self.matrix_contracts_path, contract.to_dict())

    def append_matrix_bundle(self, bundle: MatrixDataBundle) -> None:
        if bundle.bundle_id in self.get_matrix_bundle_ids():
            raise ValueError(f"Duplicate bundle_id in matrix bundles ledger: {bundle.bundle_id}")
        self._append_jsonl(self.matrix_bundles_path, bundle.to_dict())

    def append_factor_matrix(self, matrix: FactorMatrix) -> None:
        if matrix.matrix_id in self.get_factor_matrix_ids():
            raise ValueError(f"Duplicate matrix_id in factor matrices ledger: {matrix.matrix_id}")
        self._append_jsonl(self.factor_matrices_path, matrix.to_dict())

    def append_expression_result(self, result: ExpressionEvaluationResult) -> None:
        if result.result_id in self.get_expression_result_ids():
            raise ValueError(f"Duplicate result_id in expression results ledger: {result.result_id}")
        self._append_jsonl(self.expression_results_path, result.to_dict())

    def read_matrix_contracts(self) -> list[MatrixDataContract]:
        return [
            MatrixDataContract.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.matrix_contracts_path)
        ]

    def read_matrix_bundles(self) -> list[MatrixDataBundle]:
        return [
            MatrixDataBundle.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.matrix_bundles_path)
        ]

    def read_factor_matrices(self) -> list[FactorMatrix]:
        return [
            FactorMatrix.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.factor_matrices_path)
        ]

    def read_expression_results(self) -> list[ExpressionEvaluationResult]:
        return [
            ExpressionEvaluationResult.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.expression_results_path)
        ]

    def get_matrix_contract_ids(self) -> set[str]:
        return {contract.contract_id for contract in self.read_matrix_contracts()}

    def get_matrix_bundle_ids(self) -> set[str]:
        return {bundle.bundle_id for bundle in self.read_matrix_bundles()}

    def get_factor_matrix_ids(self) -> set[str]:
        return {matrix.matrix_id for matrix in self.read_factor_matrices()}

    def get_expression_result_ids(self) -> set[str]:
        return {result.result_id for result in self.read_expression_results()}


class FactorBacktestArtifactStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_BACKTEST_DIR
        self.weight_matrices_path = self.root_dir / DEFAULT_FACTOR_WEIGHT_MATRICES_FILENAME
        self.backtest_runs_path = self.root_dir / DEFAULT_FACTOR_BACKTEST_RUNS_FILENAME
        self.daily_records_path = self.root_dir / DEFAULT_FACTOR_DAILY_RECORDS_FILENAME

    def append_weight_matrix(self, matrix: FactorWeightMatrix) -> None:
        if matrix.weights_id in self.get_weight_matrix_ids():
            raise ValueError(f"Duplicate weights_id in factor weight matrices ledger: {matrix.weights_id}")
        self._append_jsonl(self.weight_matrices_path, matrix.to_dict())

    def append_backtest_run(self, run: SingleFactorBacktestRun) -> None:
        if run.run_id in self.get_backtest_run_ids():
            raise ValueError(f"Duplicate run_id in factor backtest runs ledger: {run.run_id}")
        self._append_jsonl(self.backtest_runs_path, run.to_dict())

    def append_daily_records(self, records: Sequence[FactorDailyBacktestRecord]) -> int:
        existing_keys = self._get_daily_record_keys()
        pending_keys: set[tuple[str, str, str, str, str]] = set()
        for record in records:
            key = self._daily_record_key(record)
            if key in existing_keys or key in pending_keys:
                raise ValueError(f"Duplicate factor daily backtest record: {key}")
            pending_keys.add(key)
        for record in records:
            self._append_jsonl(self.daily_records_path, record.to_dict())
        return len(records)

    def read_weight_matrices(self) -> list[FactorWeightMatrix]:
        return [
            FactorWeightMatrix.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.weight_matrices_path)
        ]

    def read_backtest_runs(self) -> list[SingleFactorBacktestRun]:
        return [
            SingleFactorBacktestRun.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.backtest_runs_path)
        ]

    def read_daily_records(self) -> list[FactorDailyBacktestRecord]:
        return [
            FactorDailyBacktestRecord.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.daily_records_path)
        ]

    def get_weight_matrix_ids(self) -> set[str]:
        return {matrix.weights_id for matrix in self.read_weight_matrices()}

    def get_backtest_run_ids(self) -> set[str]:
        return {run.run_id for run in self.read_backtest_runs()}

    def _get_daily_record_keys(self) -> set[tuple[str, str, str, str, str]]:
        return {self._daily_record_key(record) for record in self.read_daily_records()}

    def _daily_record_key(self, record: FactorDailyBacktestRecord) -> tuple[str, str, str, str, str]:
        run_id = str(record.metadata.get("run_id") or "")
        return (
            run_id,
            record.date,
            record.signal_date,
            record.execution_start_date,
            record.execution_end_date,
        )


class FactorValidationArtifactStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_VALIDATION_DIR
        self.robustness_reports_path = (
            self.root_dir / DEFAULT_FACTOR_ROBUSTNESS_REPORTS_FILENAME
        )
        self.cost_capacity_reports_path = (
            self.root_dir / DEFAULT_FACTOR_COST_CAPACITY_REPORTS_FILENAME
        )
        self.enhanced_validation_reports_path = (
            self.root_dir / DEFAULT_ENHANCED_VALIDATION_REPORTS_FILENAME
        )

    def append_robustness_report(self, report: FactorRobustnessReport) -> None:
        if report.report_id in self.get_robustness_report_ids():
            raise ValueError(f"Duplicate report_id in factor robustness reports ledger: {report.report_id}")
        self._append_jsonl(self.robustness_reports_path, report.to_dict())

    def append_cost_capacity_report(self, report: FactorCostCapacityReport) -> None:
        if report.report_id in self.get_cost_capacity_report_ids():
            raise ValueError(f"Duplicate report_id in factor cost/capacity reports ledger: {report.report_id}")
        self._append_jsonl(self.cost_capacity_reports_path, report.to_dict())

    def append_enhanced_validation_report(self, report: FactorValidationReport) -> None:
        if report.report_id in self.get_enhanced_validation_report_ids():
            raise ValueError(f"Duplicate report_id in enhanced validation reports ledger: {report.report_id}")
        self._append_jsonl(self.enhanced_validation_reports_path, report.to_dict())

    def read_robustness_reports(self) -> list[FactorRobustnessReport]:
        return [
            FactorRobustnessReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.robustness_reports_path)
        ]

    def read_cost_capacity_reports(self) -> list[FactorCostCapacityReport]:
        return [
            FactorCostCapacityReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.cost_capacity_reports_path)
        ]

    def read_enhanced_validation_reports(self) -> list[FactorValidationReport]:
        return [
            FactorValidationReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.enhanced_validation_reports_path)
        ]

    def get_robustness_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_robustness_reports()}

    def get_cost_capacity_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_cost_capacity_reports()}

    def get_enhanced_validation_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_enhanced_validation_reports()}


class FactorCorrelationContributionStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_INCREMENTAL_DIR
        self.redundancy_reports_path = (
            self.root_dir / DEFAULT_FACTOR_REDUNDANCY_REPORTS_FILENAME
        )
        self.contribution_reports_path = (
            self.root_dir / DEFAULT_FACTOR_CONTRIBUTION_REPORTS_FILENAME
        )

    def append_redundancy_report(self, report: FactorRedundancyReport) -> None:
        if report.report_id in self.get_redundancy_report_ids():
            raise ValueError(f"Duplicate report_id in factor redundancy reports ledger: {report.report_id}")
        self._append_jsonl(self.redundancy_reports_path, report.to_dict())

    def append_contribution_report(self, report: FactorPortfolioContributionReport) -> None:
        if report.report_id in self.get_contribution_report_ids():
            raise ValueError(
                f"Duplicate report_id in factor contribution reports ledger: {report.report_id}"
            )
        self._append_jsonl(self.contribution_reports_path, report.to_dict())

    def read_redundancy_reports(self) -> list[FactorRedundancyReport]:
        return [
            FactorRedundancyReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.redundancy_reports_path)
        ]

    def read_contribution_reports(self) -> list[FactorPortfolioContributionReport]:
        return [
            FactorPortfolioContributionReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.contribution_reports_path)
        ]

    def get_redundancy_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_redundancy_reports()}

    def get_contribution_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_contribution_reports()}


class FactorLibraryAuditStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_LIBRARY_AUDIT_DIR
        self.audit_reports_path = self.root_dir / DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME
        self.audit_markdown_path = self.root_dir / DEFAULT_FACTOR_LIBRARY_AUDIT_MARKDOWN_FILENAME
        self.dashboard_payload_path = self.root_dir / DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME

    def append_audit_report(self, report: FactorLibraryAuditReport) -> None:
        if report.report_id in self.get_audit_report_ids():
            raise ValueError(f"Duplicate report_id in factor library audit ledger: {report.report_id}")
        self._append_jsonl(self.audit_reports_path, report.to_dict())

    def read_audit_reports(self) -> list[FactorLibraryAuditReport]:
        return [
            FactorLibraryAuditReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.audit_reports_path)
        ]

    def get_audit_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_audit_reports()}

    def save_audit_markdown(self, markdown: str) -> Path:
        return self._write_text(self.audit_markdown_path, markdown)

    def load_audit_markdown(self) -> str:
        return self.audit_markdown_path.read_text(encoding="utf-8")

    def save_dashboard_payload(self, payload: Mapping[str, Any]) -> Path:
        return self._write_json(self.dashboard_payload_path, payload)

    def load_dashboard_payload(self) -> dict[str, Any]:
        return self._read_json(self.dashboard_payload_path)


class FactorShadowScoringStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_SHADOW_SCORING_DIR
        self.factor_scores_path = self.root_dir / DEFAULT_SHADOW_FACTOR_SCORES_FILENAME
        self.candidate_scores_path = self.root_dir / DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME
        self.comparison_reports_path = self.root_dir / DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME
        self.comparison_markdown_path = self.root_dir / DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME
        self.dashboard_payload_path = self.root_dir / DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME

    def append_factor_scores(self, scores: Sequence[ShadowFactorScore]) -> int:
        for score in scores:
            self._append_jsonl(self.factor_scores_path, score.to_dict())
        return len(scores)

    def append_candidate_scores(self, scores: Sequence[ShadowCandidateScore]) -> int:
        for score in scores:
            self._append_jsonl(self.candidate_scores_path, score.to_dict())
        return len(scores)

    def append_comparison_report(self, report: ShadowScoringComparisonReport) -> None:
        if report.report_id in self.get_comparison_report_ids():
            raise ValueError(f"Duplicate report_id in shadow comparison reports ledger: {report.report_id}")
        self._append_jsonl(self.comparison_reports_path, report.to_dict())

    def read_factor_scores(self) -> list[ShadowFactorScore]:
        return [
            ShadowFactorScore.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.factor_scores_path)
        ]

    def read_candidate_scores(self) -> list[ShadowCandidateScore]:
        return [
            ShadowCandidateScore.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.candidate_scores_path)
        ]

    def read_comparison_reports(self) -> list[ShadowScoringComparisonReport]:
        return [
            ShadowScoringComparisonReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.comparison_reports_path)
        ]

    def get_comparison_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_comparison_reports()}

    def save_markdown(self, markdown: str) -> Path:
        return self._write_text(self.comparison_markdown_path, markdown)

    def load_markdown(self) -> str:
        return self.comparison_markdown_path.read_text(encoding="utf-8")

    def save_dashboard_payload(self, payload: Mapping[str, Any]) -> Path:
        return self._write_json(self.dashboard_payload_path, payload)

    def load_dashboard_payload(self) -> dict[str, Any]:
        return self._read_json(self.dashboard_payload_path)


class FactorEvidenceStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_EVIDENCE_DIR
        self.date_results_path = self.root_dir / DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME
        self.multi_date_reports_path = self.root_dir / DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME
        self.evidence_markdown_path = self.root_dir / DEFAULT_EVIDENCE_MARKDOWN_FILENAME
        self.evidence_dashboard_path = self.root_dir / DEFAULT_EVIDENCE_DASHBOARD_FILENAME

    def append_date_result(self, result: FactorShadowEvidenceDateResult) -> None:
        if result.result_id in self.get_date_result_ids():
            raise ValueError(f"Duplicate result_id in factor evidence date results ledger: {result.result_id}")
        self._append_jsonl(self.date_results_path, result.to_dict())

    def append_multi_date_report(self, report: MultiDateFactorEvidenceReport) -> None:
        if report.report_id in self.get_multi_date_report_ids():
            raise ValueError(f"Duplicate report_id in multi-date evidence reports ledger: {report.report_id}")
        self._append_jsonl(self.multi_date_reports_path, report.to_dict())

    def read_date_results(self) -> list[FactorShadowEvidenceDateResult]:
        return [
            FactorShadowEvidenceDateResult.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.date_results_path)
        ]

    def read_multi_date_reports(self) -> list[MultiDateFactorEvidenceReport]:
        return [
            MultiDateFactorEvidenceReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.multi_date_reports_path)
        ]

    def get_date_result_ids(self) -> set[str]:
        return {result.result_id for result in self.read_date_results()}

    def get_multi_date_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_multi_date_reports()}

    def save_evidence_markdown(self, markdown: str) -> Path:
        return self._write_text(self.evidence_markdown_path, markdown)

    def load_evidence_markdown(self) -> str:
        return self.evidence_markdown_path.read_text(encoding="utf-8")

    def save_evidence_dashboard(self, payload: Mapping[str, Any]) -> Path:
        return self._write_json(self.evidence_dashboard_path, payload)

    def load_evidence_dashboard(self) -> dict[str, Any]:
        return self._read_json(self.evidence_dashboard_path)


class FactorAlignmentAuditStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR
        self.alignment_audit_reports_path = (
            self.root_dir / DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME
        )
        self.alignment_audit_markdown_path = (
            self.root_dir / DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME
        )

    def append_alignment_audit_report(
        self,
        report: FactorBacktestAlignmentAuditReport,
    ) -> None:
        if report.report_id in self.get_alignment_audit_report_ids():
            raise ValueError(f"Duplicate report_id in alignment audit ledger: {report.report_id}")
        self._append_jsonl(self.alignment_audit_reports_path, report.to_dict())

    def read_alignment_audit_reports(self) -> list[FactorBacktestAlignmentAuditReport]:
        return [
            FactorBacktestAlignmentAuditReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.alignment_audit_reports_path)
        ]

    def get_alignment_audit_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_alignment_audit_reports()}

    def save_alignment_audit_markdown(self, markdown: str) -> Path:
        return self._write_text(self.alignment_audit_markdown_path, markdown)

    def load_alignment_audit_markdown(self) -> str:
        return self.alignment_audit_markdown_path.read_text(encoding="utf-8")


class FactorTradabilityAuditStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR
        self.tradability_masks_path = self.root_dir / DEFAULT_TRADABILITY_MASKS_FILENAME
        self.tradability_audit_reports_path = (
            self.root_dir / DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME
        )
        self.execution_feasibility_reports_path = (
            self.root_dir / DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME
        )
        self.tradability_audit_markdown_path = (
            self.root_dir / DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME
        )
        self.execution_feasibility_markdown_path = (
            self.root_dir / DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME
        )

    def append_tradability_mask(self, mask: AShareTradabilityMask) -> None:
        if mask.mask_id in self.get_tradability_mask_ids():
            raise ValueError(f"Duplicate mask_id in tradability audit ledger: {mask.mask_id}")
        self._append_jsonl(self.tradability_masks_path, mask.to_dict())

    def append_tradability_audit_report(self, report: FactorTradabilityAuditReport) -> None:
        if report.report_id in self.get_tradability_audit_report_ids():
            raise ValueError(
                f"Duplicate report_id in tradability audit ledger: {report.report_id}"
            )
        self._append_jsonl(self.tradability_audit_reports_path, report.to_dict())

    def append_execution_feasibility_report(
        self,
        report: FactorExecutionFeasibilityReport,
    ) -> None:
        if report.report_id in self.get_execution_feasibility_report_ids():
            raise ValueError(
                f"Duplicate report_id in execution feasibility ledger: {report.report_id}"
            )
        self._append_jsonl(self.execution_feasibility_reports_path, report.to_dict())

    def read_tradability_masks(self) -> list[AShareTradabilityMask]:
        return [
            AShareTradabilityMask.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.tradability_masks_path)
        ]

    def read_tradability_audit_reports(self) -> list[FactorTradabilityAuditReport]:
        return [
            FactorTradabilityAuditReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.tradability_audit_reports_path)
        ]

    def read_execution_feasibility_reports(self) -> list[FactorExecutionFeasibilityReport]:
        return [
            FactorExecutionFeasibilityReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.execution_feasibility_reports_path)
        ]

    def get_tradability_mask_ids(self) -> set[str]:
        return {mask.mask_id for mask in self.read_tradability_masks()}

    def get_tradability_audit_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_tradability_audit_reports()}

    def get_execution_feasibility_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_execution_feasibility_reports()}

    def save_tradability_audit_markdown(self, markdown: str) -> Path:
        return self._write_text(self.tradability_audit_markdown_path, markdown)

    def load_tradability_audit_markdown(self) -> str:
        return self.tradability_audit_markdown_path.read_text(encoding="utf-8")

    def save_execution_feasibility_markdown(self, markdown: str) -> Path:
        return self._write_text(self.execution_feasibility_markdown_path, markdown)

    def load_execution_feasibility_markdown(self) -> str:
        return self.execution_feasibility_markdown_path.read_text(encoding="utf-8")


class FactorExecutionCostSimulationStore(JsonArtifactStoreMixin):
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_FACTOR_EXECUTION_COST_DIR
        self.execution_cost_reports_path = self.root_dir / DEFAULT_EXECUTION_COST_REPORTS_FILENAME
        self.execution_adjusted_runs_path = (
            self.root_dir / DEFAULT_EXECUTION_ADJUSTED_RUNS_FILENAME
        )
        self.execution_adjusted_daily_records_path = (
            self.root_dir / DEFAULT_EXECUTION_ADJUSTED_DAILY_RECORDS_FILENAME
        )
        self.execution_cost_markdown_path = (
            self.root_dir / DEFAULT_EXECUTION_COST_MARKDOWN_FILENAME
        )
        self.execution_cost_dashboard_path = (
            self.root_dir / DEFAULT_EXECUTION_COST_DASHBOARD_FILENAME
        )

    def append_execution_cost_report(
        self,
        report: FactorExecutionCostSimulationReport,
    ) -> None:
        if report.report_id in self.get_execution_cost_report_ids():
            raise ValueError(f"Duplicate report_id in execution cost reports ledger: {report.report_id}")
        self._append_jsonl(self.execution_cost_reports_path, report.to_dict())

    def append_execution_adjusted_run(self, run: ExecutionAdjustedBacktestRun) -> None:
        if run.adjusted_run_id in self.get_execution_adjusted_run_ids():
            raise ValueError(
                f"Duplicate adjusted_run_id in execution adjusted runs ledger: {run.adjusted_run_id}"
            )
        self._append_jsonl(self.execution_adjusted_runs_path, run.to_dict())

    def append_daily_execution_cost_records(
        self,
        records: Sequence[DailyExecutionCostRecord],
    ) -> int:
        for record in records:
            self._append_jsonl(self.execution_adjusted_daily_records_path, record.to_dict())
        return len(records)

    def read_execution_cost_reports(self) -> list[FactorExecutionCostSimulationReport]:
        return [
            FactorExecutionCostSimulationReport.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.execution_cost_reports_path)
        ]

    def read_execution_adjusted_runs(self) -> list[ExecutionAdjustedBacktestRun]:
        return [
            ExecutionAdjustedBacktestRun.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.execution_adjusted_runs_path)
        ]

    def read_daily_execution_cost_records(self) -> list[DailyExecutionCostRecord]:
        return [
            DailyExecutionCostRecord.from_dict(payload)
            for payload in self._read_jsonl_payloads(self.execution_adjusted_daily_records_path)
        ]

    def get_execution_cost_report_ids(self) -> set[str]:
        return {report.report_id for report in self.read_execution_cost_reports()}

    def get_execution_adjusted_run_ids(self) -> set[str]:
        return {run.adjusted_run_id for run in self.read_execution_adjusted_runs()}

    def save_execution_cost_markdown(self, markdown: str) -> Path:
        return self._write_text(self.execution_cost_markdown_path, markdown)

    def load_execution_cost_markdown(self) -> str:
        return self.execution_cost_markdown_path.read_text(encoding="utf-8")

    def save_execution_cost_dashboard(self, payload: Mapping[str, Any]) -> Path:
        return self._write_json(self.execution_cost_dashboard_path, payload)

    def load_execution_cost_dashboard(self) -> dict[str, Any]:
        return self._read_json(self.execution_cost_dashboard_path)


__all__ = [
    "DEFAULT_FACTOR_VALIDATION_DIR",
    "DEFAULT_FACTOR_ROBUSTNESS_REPORTS_FILENAME",
    "DEFAULT_FACTOR_COST_CAPACITY_REPORTS_FILENAME",
    "DEFAULT_ENHANCED_VALIDATION_REPORTS_FILENAME",
    "DEFAULT_FACTOR_INCREMENTAL_DIR",
    "DEFAULT_FACTOR_REDUNDANCY_REPORTS_FILENAME",
    "DEFAULT_FACTOR_CONTRIBUTION_REPORTS_FILENAME",
    "DEFAULT_FACTOR_LIBRARY_AUDIT_DIR",
    "DEFAULT_FACTOR_LIBRARY_AUDIT_REPORTS_FILENAME",
    "DEFAULT_FACTOR_LIBRARY_AUDIT_MARKDOWN_FILENAME",
    "DEFAULT_FACTOR_GOVERNANCE_DASHBOARD_FILENAME",
    "DEFAULT_FACTOR_SHADOW_SCORING_DIR",
    "DEFAULT_SHADOW_FACTOR_SCORES_FILENAME",
    "DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME",
    "DEFAULT_FACTOR_EVIDENCE_DIR",
    "DEFAULT_EVIDENCE_DATE_RESULTS_FILENAME",
    "DEFAULT_MULTI_DATE_EVIDENCE_REPORTS_FILENAME",
    "DEFAULT_EVIDENCE_DASHBOARD_FILENAME",
    "DEFAULT_EVIDENCE_MARKDOWN_FILENAME",
    "DEFAULT_FACTOR_ALIGNMENT_AUDIT_DIR",
    "DEFAULT_ALIGNMENT_AUDIT_REPORTS_FILENAME",
    "DEFAULT_ALIGNMENT_AUDIT_MARKDOWN_FILENAME",
    "DEFAULT_FACTOR_TRADABILITY_AUDIT_DIR",
    "DEFAULT_TRADABILITY_MASKS_FILENAME",
    "DEFAULT_TRADABILITY_AUDIT_REPORTS_FILENAME",
    "DEFAULT_EXECUTION_FEASIBILITY_REPORTS_FILENAME",
    "DEFAULT_TRADABILITY_AUDIT_MARKDOWN_FILENAME",
    "DEFAULT_EXECUTION_FEASIBILITY_MARKDOWN_FILENAME",
    "DEFAULT_FACTOR_EXECUTION_COST_DIR",
    "DEFAULT_EXECUTION_COST_REPORTS_FILENAME",
    "DEFAULT_EXECUTION_ADJUSTED_DAILY_RECORDS_FILENAME",
    "DEFAULT_EXECUTION_ADJUSTED_RUNS_FILENAME",
    "DEFAULT_EXECUTION_COST_MARKDOWN_FILENAME",
    "DEFAULT_EXECUTION_COST_DASHBOARD_FILENAME",
    "FactorGovernanceStore",
    "FactorMatrixStore",
    "FactorBacktestArtifactStore",
    "FactorValidationArtifactStore",
    "FactorCorrelationContributionStore",
    "FactorLibraryAuditStore",
    "FactorShadowScoringStore",
    "FactorEvidenceStore",
    "FactorAlignmentAuditStore",
    "FactorTradabilityAuditStore",
    "FactorExecutionCostSimulationStore",
]
