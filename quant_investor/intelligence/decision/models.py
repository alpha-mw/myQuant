"""Closed value models shared by the I1 investment-decision library.

The helpers in this module are intentionally pure.  They normalize caller-owned
values into new JSON-compatible values, grant no authority, and never read or
write external state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import re
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract.canonical import canonical_resource_bytes

from .._core import (
    IntelligenceContractError,
    decimal_text,
    decimal_value,
    exact_ref,
    sha256,
    timestamp,
)

POLICY_VERSION: Final = "myquant.v17.research-intelligence.decision-policy.v1"
CONTEXT_NOTE_VERSION: Final = "myquant.v17.research-intelligence.decision-context-note.v1"
CONTEXT_VERSION: Final = "myquant.v17.research-intelligence.investment-decision-context.v1"
RISK_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.risk-assessment-receipt.v1"
DECISION_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.investment-decision-receipt.v1"
MEMO_VERSION: Final = "myquant.v17.research-intelligence.investment-memo.v1"
DISCIPLINE_ENTRY_VERSION: Final = "myquant.v17.research-intelligence.decision-discipline-entry.v1"
PAPER_INTAKE_PROPOSAL_VERSION: Final = "myquant.v17.research-intelligence.paper-intake-proposal.v1"

DECISION_PROTOCOL: Final = "myquant.v17.v4"
MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
MAX_TEXT_BYTES: Final = 4_000
MAX_CONTEXT_NOTES: Final = 32
MAX_AI_DRAFTS: Final = 16
MAX_EVIDENCE_REFS: Final = 256
MAX_ASSESSMENTS_PER_DIMENSION: Final = 16
MAX_REASON_CODES: Final = 64
MAX_BLOCKER_CODES: Final = 64

REQUIREMENT_CLASSES: Final = frozenset(
    {
        "AI_DRAFT",
        "INDUSTRY_CONTEXT",
        "R22_EVALUATION",
        "THEME_CONTEXT",
        "VALUATION_CONTEXT",
        "WHY_NOW",
    }
)
RISK_DIMENSIONS: Final = frozenset({"BUSINESS", "FINANCIAL", "MARKET", "THESIS"})
CONTEXT_NOTE_KINDS: Final = frozenset(
    {
        "COMPANY_DISPLAY_NAME",
        "INDUSTRY_CONTEXT",
        "THEME_CONTEXT",
        "VALUATION_CONTEXT",
        "WHY_NOW",
    }
)
DECISION_STATES: Final = frozenset(
    {
        "INSUFFICIENT_EVIDENCE",
        "PAPER_CANDIDATE",
        "RESEARCH_APPROVED",
        "THESIS_INVALIDATED",
        "WATCHLIST",
    }
)
R22_HYPOTHESIS_STATUSES: Final = frozenset({"FAILED", "SUPPORTED", "UNCERTAIN"})
RISK_ASSESSMENT_KINDS: Final = frozenset({"NO_MATERIAL_RISK_IDENTIFIED", "RISK_IDENTIFIED"})

ALLOWED_ERROR_CODES: Final = frozenset(
    {
        "I1_AUTHORITY_OPEN",
        "I1_DISCIPLINE_TRANSITION_INVALID",
        "I1_FUTURE_INPUT",
        "I1_POLICY_INVALID",
        "I1_R22_CLOSURE_INVALID",
        "I1_REF_MISMATCH",
        "I1_REPLAY_MISMATCH",
        "I1_SHAPE_INVALID",
    }
)

CONTENT_REF_FIELDS: Final = frozenset(
    {"artifact_id", "artifact_version", "byte_sha256", "semantic_sha256"}
)
_ENTITY_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


class DecisionContractError(IntelligenceContractError):
    """Fail-closed I1 error carrying one stable machine-readable code."""

    exit_code = 2

    def __init__(self, code: str, message: str | None = None) -> None:
        if code not in ALLOWED_ERROR_CODES:
            raise ValueError("DecisionContractError code is not allowlisted")
        self.code = code
        super().__init__(message if message is not None else code)


def fail(code: str, message: str) -> NoReturn:
    raise DecisionContractError(code, message)


def canonical_timestamp(value: Any, *, label: str, code: str = "I1_SHAPE_INVALID") -> str:
    try:
        return timestamp(value, label=label)
    except IntelligenceContractError as exc:
        fail(code, str(exc))


def bounded_text(value: Any, *, label: str, maximum: int = MAX_TEXT_BYTES) -> str:
    if type(value) is not str or not value.strip() or len(value.strip().encode("utf-8")) > maximum:
        fail("I1_SHAPE_INVALID", f"{label} must be non-empty and at most {maximum} bytes")
    return value.strip()


def company_code(value: Any, *, label: str = "company_code") -> str:
    if type(value) is not str or _ENTITY_ID_RE.fullmatch(value) is None:
        fail("I1_SHAPE_INVALID", f"{label} is not a canonical entity id")
    return value


def canonical_codes(
    values: Sequence[Any], *, label: str, maximum: int = MAX_REASON_CODES
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        fail("I1_SHAPE_INVALID", f"{label} must be a sequence")
    rows: list[str] = []
    for index, value in enumerate(values):
        if type(value) is not str or _CODE_RE.fullmatch(value) is None:
            fail("I1_SHAPE_INVALID", f"{label}[{index}] is not a canonical code")
        rows.append(value)
    if len(rows) > maximum or len(rows) != len(set(rows)):
        fail("I1_SHAPE_INVALID", f"{label} cardinality is invalid")
    return sorted(rows, key=lambda item: item.encode("ascii"))


def canonical_content_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != CONTENT_REF_FIELDS:
        fail("I1_SHAPE_INVALID", f"{label} must be a four-field content reference")
    artifact_id = value["artifact_id"]
    artifact_version = value["artifact_version"]
    if type(artifact_id) is not str or not artifact_id:
        fail("I1_SHAPE_INVALID", f"{label}.artifact_id is required")
    if type(artifact_version) is not str or not artifact_version:
        fail("I1_SHAPE_INVALID", f"{label}.artifact_version is required")
    try:
        byte_sha = sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        semantic_sha = sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256")
    except IntelligenceContractError as exc:
        fail("I1_SHAPE_INVALID", str(exc))
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha,
        "semantic_sha256": semantic_sha,
    }


def sorted_content_refs(
    values: Sequence[Mapping[str, Any]], *, label: str, maximum: int | None = None
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        fail("I1_SHAPE_INVALID", f"{label} must be a sequence")
    rows = [
        canonical_content_ref(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    keys = [
        (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        )
        for row in rows
    ]
    if maximum is not None and len(rows) > maximum:
        fail("I1_SHAPE_INVALID", f"{label} exceeds its maximum cardinality")
    if len(keys) != len(set(keys)) or len({row["artifact_id"] for row in rows}) != len(rows):
        fail("I1_SHAPE_INVALID", f"{label} contains duplicate IDs or references")
    return sorted(
        rows,
        key=lambda row: tuple(
            row[field].encode("ascii")
            for field in ("artifact_id", "artifact_version", "byte_sha256", "semantic_sha256")
        ),
    )


def canonical_exact_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    try:
        return exact_ref(value, label=label)
    except IntelligenceContractError as exc:
        fail("I1_SHAPE_INVALID", str(exc))


def sorted_exact_source_refs(
    values: Sequence[Mapping[str, Any]], *, label: str, maximum: int | None = None
) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        fail("I1_SHAPE_INVALID", f"{label} must be a sequence")
    rows = [
        canonical_exact_ref(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    ]
    keys = [(row["relative_path"], row["byte_sha256"]) for row in rows]
    if maximum is not None and len(rows) > maximum:
        fail("I1_SHAPE_INVALID", f"{label} exceeds its maximum cardinality")
    if len(keys) != len(set(keys)):
        fail("I1_SHAPE_INVALID", f"{label} contains duplicate references")
    return sorted(
        rows,
        key=lambda row: (row["relative_path"].encode("ascii"), row["byte_sha256"].encode("ascii")),
    )


def canonical_decimal(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> str:
    try:
        parsed = decimal_value(value, label=label, minimum=minimum, maximum=maximum)
    except IntelligenceContractError as exc:
        fail("I1_SHAPE_INVALID", str(exc))
    return decimal_text(parsed)


def ensure_artifact_size(document: Mapping[str, Any]) -> None:
    try:
        size = len(canonical_resource_bytes(document))
    except Exception:
        fail("I1_SHAPE_INVALID", "artifact is not canonically serializable")
    if size > MAX_ARTIFACT_BYTES:
        fail("I1_SHAPE_INVALID", "artifact exceeds the 8 MiB limit")


def _taxonomy_values(values: Sequence[Any], *, label: str, allowed: frozenset[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        fail("I1_POLICY_INVALID", f"{label} must be a sequence")
    rows = list(values)
    if any(type(value) is not str or value not in allowed for value in rows):
        fail("I1_POLICY_INVALID", f"{label} contains a non-allowlisted value")
    if len(rows) != len(set(rows)):
        fail("I1_POLICY_INVALID", f"{label} contains duplicates")
    return sorted(rows, key=lambda value: value.encode("ascii"))


def build_decision_policy(
    *,
    created_at: Any,
    research_required_classes: Sequence[str],
    paper_required_classes: Sequence[str],
    research_required_risk_dimensions: Sequence[str],
    paper_required_risk_dimensions: Sequence[str],
    min_research_confidence: Any,
    min_paper_confidence: Any,
    min_research_posterior: Any,
    min_paper_posterior: Any,
    max_research_risk: Any,
    max_paper_risk: Any,
    hard_veto_severity: Any,
    require_r22_supported_for_research: bool,
    require_r22_supported_for_paper: bool,
    max_review_delay_seconds: int,
) -> dict[str, Any]:
    from .receipts import seal_artifact

    research_classes = _taxonomy_values(
        research_required_classes,
        label="research_required_classes",
        allowed=REQUIREMENT_CLASSES,
    )
    paper_classes = _taxonomy_values(
        paper_required_classes,
        label="paper_required_classes",
        allowed=REQUIREMENT_CLASSES,
    )
    research_dimensions = _taxonomy_values(
        research_required_risk_dimensions,
        label="research_required_risk_dimensions",
        allowed=RISK_DIMENSIONS,
    )
    paper_dimensions = _taxonomy_values(
        paper_required_risk_dimensions,
        label="paper_required_risk_dimensions",
        allowed=RISK_DIMENSIONS,
    )
    if not set(research_classes) <= set(paper_classes):
        fail("I1_POLICY_INVALID", "paper required classes must include research classes")
    if not set(research_dimensions) <= set(paper_dimensions):
        fail("I1_POLICY_INVALID", "paper risk dimensions must include research dimensions")
    low = Decimal("0")
    high = Decimal("1")
    min_research_confidence_text = canonical_decimal(
        min_research_confidence, label="min_research_confidence", minimum=low, maximum=high
    )
    min_paper_confidence_text = canonical_decimal(
        min_paper_confidence, label="min_paper_confidence", minimum=low, maximum=high
    )
    min_research_posterior_text = canonical_decimal(
        min_research_posterior, label="min_research_posterior", minimum=low, maximum=high
    )
    min_paper_posterior_text = canonical_decimal(
        min_paper_posterior, label="min_paper_posterior", minimum=low, maximum=high
    )
    max_research_risk_text = canonical_decimal(
        max_research_risk, label="max_research_risk", minimum=low, maximum=high
    )
    max_paper_risk_text = canonical_decimal(
        max_paper_risk, label="max_paper_risk", minimum=low, maximum=high
    )
    hard_veto_severity_text = canonical_decimal(
        hard_veto_severity, label="hard_veto_severity", minimum=low, maximum=high
    )
    if Decimal(min_paper_confidence_text) < Decimal(min_research_confidence_text):
        fail("I1_POLICY_INVALID", "paper confidence cannot be weaker than research")
    if Decimal(min_paper_posterior_text) < Decimal(min_research_posterior_text):
        fail("I1_POLICY_INVALID", "paper posterior cannot be weaker than research")
    if Decimal(max_paper_risk_text) > Decimal(max_research_risk_text):
        fail("I1_POLICY_INVALID", "paper risk limit cannot be weaker than research")
    if (
        type(require_r22_supported_for_research) is not bool
        or type(require_r22_supported_for_paper) is not bool
    ):
        fail("I1_POLICY_INVALID", "R2.2 tier gates must be booleans")
    if (
        type(max_review_delay_seconds) is not int
        or type(max_review_delay_seconds) is bool
        or not 1 <= max_review_delay_seconds <= 31_536_000
    ):
        fail("I1_POLICY_INVALID", "max_review_delay_seconds is outside 1..31536000")
    return seal_artifact(
        version=POLICY_VERSION,
        identity_field="policy_id",
        timestamp_value=canonical_timestamp(
            created_at, label="created_at", code="I1_POLICY_INVALID"
        ),
        payload={
            "hard_veto_severity": hard_veto_severity_text,
            "max_paper_risk": max_paper_risk_text,
            "max_research_risk": max_research_risk_text,
            "max_review_delay_seconds": max_review_delay_seconds,
            "min_paper_confidence": min_paper_confidence_text,
            "min_paper_posterior": min_paper_posterior_text,
            "min_research_confidence": min_research_confidence_text,
            "min_research_posterior": min_research_posterior_text,
            "paper_required_classes": paper_classes,
            "paper_required_risk_dimensions": paper_dimensions,
            "require_r22_supported_for_paper": require_r22_supported_for_paper,
            "require_r22_supported_for_research": require_r22_supported_for_research,
            "research_required_classes": research_classes,
            "research_required_risk_dimensions": research_dimensions,
        },
    )


POLICY_PAYLOAD_FIELDS: Final = frozenset(
    {
        "hard_veto_severity",
        "max_paper_risk",
        "max_research_risk",
        "max_review_delay_seconds",
        "min_paper_confidence",
        "min_paper_posterior",
        "min_research_confidence",
        "min_research_posterior",
        "paper_required_classes",
        "paper_required_risk_dimensions",
        "require_r22_supported_for_paper",
        "require_r22_supported_for_research",
        "research_required_classes",
        "research_required_risk_dimensions",
    }
)


def validate_decision_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    from .receipts import validate_closed_artifact

    row = validate_closed_artifact(
        document,
        version=POLICY_VERSION,
        identity_field="policy_id",
        payload_fields=set(POLICY_PAYLOAD_FIELDS),
    )
    try:
        expected = build_decision_policy(
            created_at=row["timestamp"],
            research_required_classes=row["research_required_classes"],
            paper_required_classes=row["paper_required_classes"],
            research_required_risk_dimensions=row["research_required_risk_dimensions"],
            paper_required_risk_dimensions=row["paper_required_risk_dimensions"],
            min_research_confidence=row["min_research_confidence"],
            min_paper_confidence=row["min_paper_confidence"],
            min_research_posterior=row["min_research_posterior"],
            min_paper_posterior=row["min_paper_posterior"],
            max_research_risk=row["max_research_risk"],
            max_paper_risk=row["max_paper_risk"],
            hard_veto_severity=row["hard_veto_severity"],
            require_r22_supported_for_research=row["require_r22_supported_for_research"],
            require_r22_supported_for_paper=row["require_r22_supported_for_paper"],
            max_review_delay_seconds=row["max_review_delay_seconds"],
        )
    except KeyError as exc:
        fail("I1_SHAPE_INVALID", f"decision policy is missing {exc.args[0]}")
    if expected != row:
        fail("I1_REPLAY_MISMATCH", "decision policy replay mismatch")
    return row


def build_context_note(
    *,
    kind: str,
    company_code: Any,
    text: Any,
    observed_at: Any,
    available_at: Any,
    source_ref: Mapping[str, Any],
) -> dict[str, Any]:
    from .receipts import seal_artifact

    if kind not in CONTEXT_NOTE_KINDS:
        fail("I1_SHAPE_INVALID", "context note kind is not allowlisted")
    observed = canonical_timestamp(observed_at, label="observed_at")
    available = canonical_timestamp(available_at, label="available_at")
    if observed > available:
        fail("I1_SHAPE_INVALID", "context note cannot be available before observation")
    reference = canonical_exact_ref(source_ref, label="source_ref")
    if reference["cutoff"] > available:
        fail("I1_FUTURE_INPUT", "context note source cutoff exceeds available_at")
    return seal_artifact(
        version=CONTEXT_NOTE_VERSION,
        identity_field="note_id",
        timestamp_value=available,
        payload={
            "available_at": available,
            "company_code": globals()["company_code"](company_code),
            "kind": kind,
            "observed_at": observed,
            "source_ref": reference,
            "text": bounded_text(text, label="text"),
        },
    )


CONTEXT_NOTE_PAYLOAD_FIELDS: Final = frozenset(
    {"available_at", "company_code", "kind", "observed_at", "source_ref", "text"}
)


def validate_context_note(
    document: Mapping[str, Any], *, as_of: Any, authorized_source_refs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    from .receipts import validate_closed_artifact

    cutoff = canonical_timestamp(as_of, label="as_of")
    row = validate_closed_artifact(
        document,
        version=CONTEXT_NOTE_VERSION,
        identity_field="note_id",
        payload_fields=set(CONTEXT_NOTE_PAYLOAD_FIELDS),
    )
    try:
        expected = build_context_note(
            kind=row["kind"],
            company_code=row["company_code"],
            text=row["text"],
            observed_at=row["observed_at"],
            available_at=row["available_at"],
            source_ref=row["source_ref"],
        )
    except KeyError as exc:
        fail("I1_SHAPE_INVALID", f"context note is missing {exc.args[0]}")
    if expected != row:
        fail("I1_REPLAY_MISMATCH", "context note replay mismatch")
    if row["available_at"] > cutoff:
        fail("I1_FUTURE_INPUT", "context note is not available at as_of")
    authorized = sorted_exact_source_refs(authorized_source_refs, label="authorized_source_refs")
    if row["source_ref"] not in authorized:
        fail("I1_REF_MISMATCH", "context note source is outside the authorized closure")
    return row


__all__ = [
    "ALLOWED_ERROR_CODES",
    "CONTENT_REF_FIELDS",
    "CONTEXT_NOTE_KINDS",
    "CONTEXT_NOTE_PAYLOAD_FIELDS",
    "CONTEXT_NOTE_VERSION",
    "CONTEXT_VERSION",
    "DECISION_PROTOCOL",
    "DECISION_RECEIPT_VERSION",
    "DECISION_STATES",
    "DISCIPLINE_ENTRY_VERSION",
    "DecisionContractError",
    "MAX_AI_DRAFTS",
    "MAX_ARTIFACT_BYTES",
    "MAX_ASSESSMENTS_PER_DIMENSION",
    "MAX_BLOCKER_CODES",
    "MAX_CONTEXT_NOTES",
    "MAX_EVIDENCE_REFS",
    "MAX_REASON_CODES",
    "MAX_TEXT_BYTES",
    "MEMO_VERSION",
    "PAPER_INTAKE_PROPOSAL_VERSION",
    "POLICY_PAYLOAD_FIELDS",
    "POLICY_VERSION",
    "R22_HYPOTHESIS_STATUSES",
    "REQUIREMENT_CLASSES",
    "RISK_ASSESSMENT_KINDS",
    "RISK_DIMENSIONS",
    "RISK_RECEIPT_VERSION",
    "bounded_text",
    "build_context_note",
    "build_decision_policy",
    "canonical_codes",
    "canonical_content_ref",
    "canonical_decimal",
    "canonical_exact_ref",
    "canonical_timestamp",
    "company_code",
    "ensure_artifact_size",
    "fail",
    "sorted_content_refs",
    "sorted_exact_source_refs",
    "validate_context_note",
    "validate_decision_policy",
]
