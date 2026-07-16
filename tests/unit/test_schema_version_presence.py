from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

import pytest

from quant_investor.agent_protocol import (
    BayesianDecisionRecord,
    BranchVerdict,
    EvidenceItem,
    ICDecision,
    PortfolioPlan,
    ReportBundle,
    RiskDecision,
)
from quant_investor.branch_contracts import (
    BranchResult,
    PortfolioStrategy,
    ResearchPipelineResult,
    UnifiedDataBundle,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.pipeline.result_types import QuantInvestorPipelineResult
from quant_investor.learning.memory_promoter import (
    PromotionCandidate as MemoryPromotionCandidate,
    PromotionDecision,
    RuleProposal,
    TradeCase,
)
from quant_investor.learning.post_trade_reflector import (
    ReflectionEvidence,
    ReflectionLessonDraft,
    ReflectionReport,
)
from quant_investor.learning.pre_trade_recall import (
    MemoryItem,
    PromotionCandidate as RecallPromotionCandidate,
    RecallHit,
    RecallPacket,
    RecallQuery,
)
import quant_investor.versioning as versioning


def _field_names(obj) -> set[str]:
    return {field.name for field in fields(type(obj))}


def _assert_string_field(obj, field_name: str, expected: str | None = None, prefix: str | None = None) -> None:
    assert field_name in _field_names(obj)
    value = getattr(obj, field_name)
    assert isinstance(value, str)
    assert value
    if expected is not None:
        assert value == expected
    if prefix is not None:
        assert value.startswith(prefix)


def test_report_stack_objects_expose_non_empty_version_fields():
    branch_result = BranchResult(branch_name="quant")
    pipeline_result = ResearchPipelineResult(data_bundle=UnifiedDataBundle())
    branch_verdict = BranchVerdict(agent_name="KlineAgent", thesis="ok")
    risk_decision = RiskDecision()
    ic_decision = ICDecision()
    portfolio_plan = PortfolioPlan()
    report_bundle = ReportBundle()
    mainline_result = QuantInvestorPipelineResult()

    _assert_string_field(branch_result, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(branch_result, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(branch_result, "calibration_schema_version", versioning.CALIBRATION_SCHEMA_VERSION)
    _assert_string_field(pipeline_result, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(pipeline_result, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(pipeline_result, "likelihood_schema_version", versioning.LIKELIHOOD_SCHEMA_VERSION)
    _assert_string_field(pipeline_result, "ic_protocol_version", versioning.IC_PROTOCOL_VERSION)
    _assert_string_field(pipeline_result, "report_protocol_version", versioning.REPORT_PROTOCOL_VERSION)

    _assert_string_field(branch_verdict, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(branch_verdict, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)

    _assert_string_field(risk_decision, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(risk_decision, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(risk_decision, "ic_protocol_version", versioning.IC_PROTOCOL_VERSION)

    _assert_string_field(ic_decision, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(ic_decision, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(ic_decision, "ic_protocol_version", versioning.IC_PROTOCOL_VERSION)

    _assert_string_field(portfolio_plan, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(portfolio_plan, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(portfolio_plan, "ic_protocol_version", versioning.IC_PROTOCOL_VERSION)

    _assert_string_field(report_bundle, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(report_bundle, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(report_bundle, "likelihood_schema_version", versioning.LIKELIHOOD_SCHEMA_VERSION)
    _assert_string_field(report_bundle, "ic_protocol_version", versioning.IC_PROTOCOL_VERSION)
    _assert_string_field(report_bundle, "report_protocol_version", versioning.REPORT_PROTOCOL_VERSION)
    _assert_string_field(mainline_result, "likelihood_schema_version", versioning.LIKELIHOOD_SCHEMA_VERSION)
    _assert_string_field(mainline_result, "debate_template_version", versioning.DEBATE_TEMPLATE_VERSION)
    _assert_string_field(mainline_result, "agent_schema_version", versioning.AGENT_SCHEMA_VERSION)


def test_current_artifact_envelopes_reject_old_versions_and_nested_intelligence():
    with pytest.raises(ValueError, match="branch_schema_version mismatch"):
        ReportBundle(branch_schema_version="branch-schema.v13.four-branch")
    with pytest.raises(ValueError, match="non-v15 branch keys"):
        ReportBundle(branch_verdicts={"intelligence": BranchVerdict()})
    with pytest.raises(ValueError, match="noncanonical branch"):
        QuantInvestorPipelineResult(branch_results={"intelligence": object()})
    with pytest.raises(ValueError, match="noncanonical branch"):
        ResearchPipelineResult(
            data_bundle=UnifiedDataBundle(),
            branch_results={"intelligence": object()},  # type: ignore[dict-item]
        )
    old_verdict = BranchVerdict()
    old_verdict.branch_schema_version = "branch-schema.v13.four-branch"
    with pytest.raises(ValueError, match="BranchVerdict branch_schema_version mismatch"):
        ReportBundle(branch_verdicts={"quant": old_verdict})

    old_strategy = PortfolioStrategy()
    old_strategy.architecture_version = "13.0.0-stable"
    with pytest.raises(ValueError, match="PortfolioStrategy.architecture_version mismatch"):
        ResearchPipelineResult(
            data_bundle=UnifiedDataBundle(),
            final_strategy=old_strategy,
        )
    with pytest.raises(ValueError, match="PortfolioStrategy.architecture_version mismatch"):
        QuantInvestorPipelineResult(final_strategy=old_strategy)

    nested = ReportBundle(
        branch_verdicts={
            "quant": BranchVerdict(
                evidence=[EvidenceItem(metadata={"intelligence_score": 0.7})]
            )
        }
    )
    with pytest.raises(ValueError, match="retired Intelligence key"):
        nested.to_dict()

    with pytest.raises(ValueError, match="likelihood fields must match v15"):
        BayesianDecisionRecord(likelihoods={})


@pytest.mark.parametrize("missing_branch", CANONICAL_BRANCH_ORDER)
def test_bayesian_pipeline_result_requires_each_canonical_branch(
    missing_branch: str,
) -> None:
    branch_results = {
        name: BranchResult(branch_name=name)
        for name in CANONICAL_BRANCH_ORDER
        if name != missing_branch
    }

    with pytest.raises(
        ValueError,
        match=rf"bayesian branch_results.*missing branches: {missing_branch}",
    ):
        QuantInvestorPipelineResult(
            branch_results=branch_results,
            pipeline_mode="bayesian",
        )


def test_legacy_pipeline_result_keeps_partial_accumulator_construction() -> None:
    result = QuantInvestorPipelineResult(
        branch_results={"quant": BranchResult(branch_name="quant")},
        pipeline_mode="legacy",
    )

    assert set(result.branch_results) == {"quant"}


@pytest.mark.parametrize("missing_branch", CANONICAL_BRANCH_ORDER)
def test_research_pipeline_result_distinguishes_draft_from_bayesian_result(
    missing_branch: str,
) -> None:
    draft = ResearchPipelineResult(data_bundle=UnifiedDataBundle())
    assert draft.pipeline_mode == "draft"

    with pytest.raises(
        ValueError,
        match=rf"bayesian branch_results.*missing branches: {missing_branch}",
    ):
        ResearchPipelineResult(
            data_bundle=UnifiedDataBundle(),
            branch_results={
                name: BranchResult(branch_name=name)
                for name in CANONICAL_BRANCH_ORDER
                if name != missing_branch
            },
            pipeline_mode="bayesian",
        )


@pytest.mark.parametrize(
    "field_name,value,match",
    [
        ("posterior_win_rate", float("nan"), "finite"),
        ("posterior_confidence", float("inf"), "finite"),
        ("posterior_action_score", -1.1, r"\[-1.0, 1.0\]"),
        ("correlation_discount", -0.1, r"\[0.0, 1.0\]"),
        ("fallback_penalty", 1.1, r"\[0.0, 1.0\]"),
        ("rank", -1, "non-negative integer"),
    ],
)
def test_bayesian_decision_record_rejects_invalid_current_scalars(
    field_name, value, match
):
    with pytest.raises(ValueError, match=match):
        BayesianDecisionRecord(**{field_name: value})


def test_learning_stack_objects_expose_schema_version():
    recall_query = RecallQuery(
        symbol="000001.SZ",
        as_of=datetime.now(timezone.utc),
        market_regime="neutral",
        sector="bank",
        branch_support_pattern="mixed",
        consensus_count=1,
        candidate_action="watch",
        volatility_regime="normal",
    )

    learning_objects = [
        (
            MemoryPromotionCandidate(
                candidate_id="c1",
                source_case_ids=["x"],
                lesson_statement="s",
                lesson_type="risk",
                support_count=2,
                counter_count=0,
                regimes_seen=["neutral"],
                sectors_seen=["bank"],
                confidence=0.8,
                status="candidate_lesson",
                evidence_summary="ok",
            ),
            versioning.PROMOTION_CANDIDATE_SCHEMA_VERSION,
        ),
        (
            RecallPromotionCandidate(
                candidate_id="c2",
                source_case_ids=["x"],
                lesson_statement="s",
                lesson_type="risk",
                support_count=2,
                counter_count=0,
                regimes_seen=["neutral"],
                sectors_seen=["bank"],
                confidence=0.8,
                status="candidate_lesson",
                evidence_summary="ok",
            ),
            versioning.PROMOTION_CANDIDATE_SCHEMA_VERSION,
        ),
        (
            PromotionDecision(
                candidate_id="c1",
                target_status="validated_pattern",
                decision="validate_pattern",
                reason="ok",
                support_count=2,
                counter_count=0,
                confidence=0.8,
                missing_requirements=[],
                rule_proposal_recommended=False,
            ),
            versioning.PROMOTION_DECISION_SCHEMA_VERSION,
        ),
        (
            RuleProposal(
                proposal_id="p1",
                proposal_type="risk_guard_update",
                suggestion="x",
                evidence="y",
                expected_effect="z",
            ),
            versioning.RULE_PROPOSAL_SCHEMA_VERSION,
        ),
        (
            MemoryItem(
                memory_id="m1",
                source_case_id="c1",
                memory_type="episodic",
                title="t",
                statement="s",
                tags={},
            ),
            versioning.MEMORY_ITEM_SCHEMA_VERSION,
        ),
        (
            RecallHit(
                memory_id="m1",
                source_case_ids=["c1"],
                title="t",
                statement="s",
                relevance_score=0.5,
                memory_type="episodic",
                status="indexed_only",
                cautionary=False,
            ),
            versioning.RECALL_HIT_SCHEMA_VERSION,
        ),
        (recall_query, versioning.RECALL_QUERY_SCHEMA_VERSION),
        (RecallPacket(symbol="000001.SZ", query=recall_query), versioning.RECALL_PACKET_SCHEMA_VERSION),
        (
            ReflectionEvidence(evidence_type="return", observation="obs", implication="imp"),
            versioning.REFLECTION_EVIDENCE_SCHEMA_VERSION,
        ),
        (
            ReflectionLessonDraft(
                lesson_type="case_lesson",
                statement="s",
                rationale="r",
                confidence=0.7,
                promotion_recommendation="candidate_only",
            ),
            versioning.REFLECTION_LESSON_DRAFT_SCHEMA_VERSION,
        ),
        (
            ReflectionReport(
                case_id="c1",
                symbol="000001.SZ",
                thesis_validation="correct",
                timing_assessment="good",
                risk_control_assessment="good",
                human_override_assessment="neutral",
            ),
            versioning.REFLECTION_REPORT_SCHEMA_VERSION,
        ),
        (
            TradeCase(
                case_id="c1",
                symbol="000001.SZ",
                decision_time=datetime.now(timezone.utc),
                pretrade_snapshot=object(),
                human_decision=object(),
                execution_snapshot=object(),
            ),
            versioning.TRADE_CASE_SCHEMA_VERSION,
        ),
    ]

    for obj, expected in learning_objects:
        _assert_string_field(obj, "schema_version", expected)
        assert getattr(obj, "schema_version").startswith("learning.")
