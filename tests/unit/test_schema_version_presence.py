from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

from quant_investor.agent_protocol import BranchVerdict, ICDecision, PortfolioPlan, ReportBundle, RiskDecision
from quant_investor.branch_contracts import BranchResult, ResearchPipelineResult, UnifiedDataBundle
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
    branch_result = BranchResult(branch_name="kline")
    pipeline_result = ResearchPipelineResult(data_bundle=UnifiedDataBundle())
    branch_verdict = BranchVerdict(agent_name="KlineAgent", thesis="ok")
    risk_decision = RiskDecision()
    ic_decision = ICDecision()
    portfolio_plan = PortfolioPlan()
    report_bundle = ReportBundle()

    _assert_string_field(branch_result, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(branch_result, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
    _assert_string_field(branch_result, "calibration_schema_version", versioning.CALIBRATION_SCHEMA_VERSION)
    _assert_string_field(pipeline_result, "architecture_version", versioning.ARCHITECTURE_VERSION)
    _assert_string_field(pipeline_result, "branch_schema_version", versioning.BRANCH_SCHEMA_VERSION)
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
    _assert_string_field(report_bundle, "ic_protocol_version", versioning.IC_PROTOCOL_VERSION)
    _assert_string_field(report_bundle, "report_protocol_version", versioning.REPORT_PROTOCOL_VERSION)


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
