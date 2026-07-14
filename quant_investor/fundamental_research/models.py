"""Strict offline contracts for Codex-assisted fundamental research."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import hashlib
import json
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "fundamental-research.v1"
Symbol = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    ),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


def compute_base_score_sha256(base_score: float) -> str:
    payload = (
        json.dumps(
            {"base_score": base_score},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SourceTier(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    INELIGIBLE = "ineligible"


class SourceEligibilityPolicyV1(StrictModel):
    """Local taxonomy; response-declared source tiers are never authoritative."""

    primary_hostnames: set[str] = Field(
        default_factory=lambda: {
            "cninfo.com.cn",
            "sse.com.cn",
            "szse.cn",
            "csrc.gov.cn",
            "bse.cn",
            "gov.cn",
        }
    )
    secondary_hostnames: set[str] = Field(default_factory=set)
    primary_document_kinds: set[str] = Field(
        default_factory=lambda: {
            "annual_report",
            "interim_report",
            "quarterly_report",
            "exchange_filing",
            "regulatory_filing",
            "company_ir",
            "government_release",
            "industry_association_release",
        }
    )

    @staticmethod
    def _matches(hostname: str, configured: set[str]) -> bool:
        return any(hostname == item or hostname.endswith(f".{item}") for item in configured)

    def classify(self, source: "SourceRecordV1") -> SourceTier:
        hostname = (urlparse(source.canonical_url).hostname or "").casefold()
        if self._matches(hostname, self.primary_hostnames):
            if source.document_kind in self.primary_document_kinds:
                return SourceTier.PRIMARY
            return SourceTier.SECONDARY
        if self._matches(hostname, self.secondary_hostnames):
            return SourceTier.SECONDARY
        return SourceTier.INELIGIBLE

    def authority_key(self, source: "SourceRecordV1") -> str:
        """Return the configured authority, never a model-chosen subdomain."""

        hostname = (urlparse(source.canonical_url).hostname or "").casefold()
        matches = [
            item
            for item in self.primary_hostnames | self.secondary_hostnames
            if hostname == item or hostname.endswith(f".{item}")
        ]
        return max(matches, key=len) if matches else ""


def compute_source_policy_sha256(policy: SourceEligibilityPolicyV1) -> str:
    payload = {
        "primary_hostnames": sorted(policy.primary_hostnames),
        "secondary_hostnames": sorted(policy.secondary_hostnames),
        "primary_document_kinds": sorted(policy.primary_document_kinds),
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ClaimKind(str, Enum):
    FACT = "fact"
    JUDGMENT = "judgment"
    UNKNOWN = "unknown"


class Dimension(str, Enum):
    FINANCIAL_QUALITY = "financial_quality"
    BUSINESS_ECONOMICS = "business_economics"
    INDUSTRY_VALUE_CHAIN = "industry_value_chain"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"
    MANAGEMENT_CAPITAL_ALLOCATION = "management_capital_allocation"
    VALUATION_SCENARIOS = "valuation_scenarios"


class DimensionSignal(str, Enum):
    STRONG_NEGATIVE = "strong_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    STRONG_POSITIVE = "strong_positive"
    UNKNOWN = "unknown"


class SourceRecordV1(StrictModel):
    source_id: Identifier
    publisher: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]
    document_kind: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
    ]
    canonical_url: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)
    ]
    published_at: datetime
    first_available_at: datetime
    retrieved_at: datetime
    source_tier: SourceTier
    content_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    locator: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
    evidence_extract: Annotated[str, StringConstraints(min_length=1, max_length=2000)]

    @field_validator("canonical_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
        ):
            raise ValueError("canonical_url must be an uncredentialed http(s) URL")
        return value

    @field_validator("evidence_extract")
    @classmethod
    def reject_executable_markup(cls, value: str) -> str:
        lowered = value.casefold()
        if any(token in lowered for token in ("<script", "<iframe", "javascript:")):
            raise ValueError("executable markup is forbidden in evidence extracts")
        return value

    @model_validator(mode="after")
    def validate_times(self) -> "SourceRecordV1":
        values = (self.published_at, self.first_available_at, self.retrieved_at)
        if any(value.tzinfo is None or value.utcoffset() is None for value in values):
            raise ValueError("source timestamps must be timezone-aware")
        if self.first_available_at < self.published_at:
            raise ValueError("first_available_at cannot precede published_at")
        if self.retrieved_at < self.first_available_at:
            raise ValueError("retrieved_at cannot precede first_available_at")
        return self


class ClaimV1(StrictModel):
    claim_id: Identifier
    kind: ClaimKind
    dimension: Dimension
    statement: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    direction: Literal["negative", "neutral", "positive"]
    materiality: float = Field(ge=0.0, le=1.0)
    supporting_source_ids: list[Identifier] = Field(default_factory=list, max_length=10)
    counter_source_ids: list[Identifier] = Field(default_factory=list, max_length=10)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    confidence_rationale: Annotated[str, StringConstraints(max_length=1000)] = ""

    @model_validator(mode="after")
    def validate_claim(self) -> "ClaimV1":
        for value in (self.valid_from, self.valid_until):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("claim validity timestamps must be timezone-aware")
        if self.valid_from and self.valid_until and self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")
        if self.kind == ClaimKind.UNKNOWN and (
            self.supporting_source_ids or self.counter_source_ids
        ):
            raise ValueError("unknown claims cannot cite evidence")
        if self.kind == ClaimKind.UNKNOWN and self.direction != "neutral":
            raise ValueError("unknown claims must be directionally neutral")
        if self.kind != ClaimKind.UNKNOWN and not self.supporting_source_ids:
            raise ValueError("fact and judgment claims require supporting sources")
        if set(self.supporting_source_ids) & set(self.counter_source_ids):
            raise ValueError("a source cannot both support and counter the same claim")
        return self


class DimensionAssessmentV1(StrictModel):
    dimension: Dimension
    signal: DimensionSignal
    claim_ids: list[Identifier] = Field(default_factory=list, max_length=50)
    counter_evidence_summary: Annotated[str, StringConstraints(max_length=1000)] = ""
    unknowns: list[Annotated[str, StringConstraints(max_length=500)]] = Field(
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def validate_signal(self) -> "DimensionAssessmentV1":
        if self.signal == DimensionSignal.UNKNOWN and self.claim_ids:
            raise ValueError("unknown dimensions cannot cite scoring claims")
        if self.signal != DimensionSignal.UNKNOWN and not self.claim_ids:
            raise ValueError("non-unknown dimensions require claim_ids")
        return self


class FundamentalResearchDossierV1(StrictModel):
    schema_version: Literal["fundamental-research.v1"] = "fundamental-research.v1"
    dossier_id: Identifier
    request_id: Identifier
    symbol: Symbol
    company_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]
    market: Literal["CN"] = "CN"
    decision_cutoff: datetime
    produced_at: datetime
    model_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ]
    prompt_version: Identifier
    sources: list[SourceRecordV1] = Field(max_length=25)
    claims: list[ClaimV1] = Field(max_length=150)
    dimensions: list[DimensionAssessmentV1] = Field(min_length=6, max_length=6)
    bull_case: list[Identifier] = Field(default_factory=list, max_length=30)
    bear_case: list[Identifier] = Field(default_factory=list, max_length=30)
    key_risks: list[Identifier] = Field(default_factory=list, max_length=30)
    catalysts: list[Identifier] = Field(default_factory=list, max_length=30)
    unknowns: list[Annotated[str, StringConstraints(max_length=500)]] = Field(
        default_factory=list, max_length=30
    )

    @model_validator(mode="after")
    def validate_graph(self) -> "FundamentalResearchDossierV1":
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (self.decision_cutoff, self.produced_at)
        ):
            raise ValueError("dossier timestamps must be timezone-aware")
        source_ids = [item.source_id for item in self.sources]
        claim_ids = [item.claim_id for item in self.claims]
        if len(source_ids) != len(set(source_ids)) or len(claim_ids) != len(set(claim_ids)):
            raise ValueError("source_id and claim_id must be unique")
        if {item.dimension for item in self.dimensions} != set(Dimension):
            raise ValueError("dimensions must contain every canonical dimension exactly once")
        source_set, claim_set = set(source_ids), set(claim_ids)
        for claim in self.claims:
            if not set(claim.supporting_source_ids + claim.counter_source_ids) <= source_set:
                raise ValueError(f"claim {claim.claim_id} references unknown sources")
        for assessment in self.dimensions:
            if not set(assessment.claim_ids) <= claim_set:
                raise ValueError(
                    f"dimension {assessment.dimension.value} references unknown claims"
                )
            if any(
                next(item for item in self.claims if item.claim_id == claim_id).dimension
                != assessment.dimension
                for claim_id in assessment.claim_ids
            ):
                raise ValueError(
                    f"dimension {assessment.dimension.value} references cross-dimension claims"
                )
        for label, ids in (
            ("bull_case", self.bull_case),
            ("bear_case", self.bear_case),
            ("key_risks", self.key_risks),
            ("catalysts", self.catalysts),
        ):
            if not set(ids) <= claim_set:
                raise ValueError(f"{label} references unknown claims")
        return self


class ResearchBudgetV1(StrictModel):
    max_minutes: Literal[60] = 60
    max_searches: Literal[20] = 20
    max_documents: Literal[25] = 25


class LocalFundamentalContextV1(StrictModel):
    """Sanitized deterministic context; external research cannot replace it."""

    industry: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ] = "UNCONFIRMED"
    industry_status: Literal["confirmed", "unconfirmed"] = "unconfirmed"
    peer_symbols: list[Symbol] = Field(default_factory=list, max_length=30)
    peer_set_status: Literal["confirmed", "unconfirmed"] = "unconfirmed"
    base_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    available_modules: list[Identifier] = Field(default_factory=list, max_length=20)
    missing_modules: list[Identifier] = Field(default_factory=list, max_length=20)
    valuation_price: float | None = Field(default=None, gt=0.0)
    valuation_price_as_of: date | None = None

    @model_validator(mode="after")
    def validate_local_context(self) -> "LocalFundamentalContextV1":
        if self.industry_status == "unconfirmed" and self.industry != "UNCONFIRMED":
            raise ValueError("unconfirmed industry must use UNCONFIRMED")
        if self.peer_set_status == "unconfirmed" and self.peer_symbols:
            raise ValueError("unconfirmed peer set must be empty")
        if self.peer_set_status == "confirmed" and not self.peer_symbols:
            raise ValueError("confirmed peer set cannot be empty")
        if set(self.available_modules) & set(self.missing_modules):
            raise ValueError("available and missing modules cannot overlap")
        if (self.valuation_price is None) != (self.valuation_price_as_of is None):
            raise ValueError("valuation price and as-of date must be present together")
        return self


class FundamentalResearchRequestV1(StrictModel):
    schema_version: Literal["fundamental-research.v1"] = "fundamental-research.v1"
    request_id: Identifier
    run_id: Identifier
    symbol: Symbol
    company_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]
    market: Literal["CN"] = "CN"
    decision_cutoff: datetime
    created_at: datetime
    expires_at: datetime
    base_score: float = Field(ge=-1.0, le=1.0)
    base_score_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    git_sha: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,64}$")]
    data_generation: Identifier
    selection_reasons: list[Annotated[str, StringConstraints(max_length=256)]] = Field(
        min_length=1, max_length=10
    )
    prompt_version: Identifier
    policy_version: Identifier
    source_policy_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    budget: ResearchBudgetV1 = Field(default_factory=ResearchBudgetV1)
    local_context: LocalFundamentalContextV1 = Field(default_factory=LocalFundamentalContextV1)

    @model_validator(mode="after")
    def validate_times(self) -> "FundamentalResearchRequestV1":
        values = (self.decision_cutoff, self.created_at, self.expires_at)
        if any(value.tzinfo is None or value.utcoffset() is None for value in values):
            raise ValueError("request timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must follow created_at")
        if self.created_at < self.decision_cutoff:
            raise ValueError("created_at cannot precede decision_cutoff")
        if (self.expires_at - self.created_at).total_seconds() > 30 * 86400:
            raise ValueError("request TTL cannot exceed 30 calendar days")
        if self.base_score_sha256 != compute_base_score_sha256(self.base_score):
            raise ValueError("base_score_sha256 does not match canonical base_score")
        return self


class FundamentalResearchResponseV1(StrictModel):
    schema_version: Literal["fundamental-research.v1"] = "fundamental-research.v1"
    request_id: Identifier
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    dossier: FundamentalResearchDossierV1


class DimensionContributionV1(StrictModel):
    dimension: Dimension
    signal: DimensionSignal
    qualified: bool
    weight: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(ge=-1.0, le=1.0)
    blockers: list[str] = Field(default_factory=list)


class FundamentalOverlayV1(StrictModel):
    schema_version: Literal["fundamental-research.v1"] = "fundamental-research.v1"
    request_id: Identifier
    dossier_id: Identifier
    symbol: Symbol
    base_score: float = Field(ge=-1.0, le=1.0)
    computed_delta: float = Field(ge=-0.1, le=0.1)
    adjusted_score: float = Field(ge=-1.0, le=1.0)
    eligible: bool
    contributions: list[DimensionContributionV1] = Field(min_length=6, max_length=6)
    blockers: list[str] = Field(default_factory=list)


class JobState(str, Enum):
    PREPARED = "PREPARED"
    EXPORTED = "EXPORTED"
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


class ApplicationState(str, Enum):
    SHADOW_EVALUATED = "SHADOW_EVALUATED"
    LIMITED_APPLIED = "LIMITED_APPLIED"
    PRODUCTION_APPLIED = "PRODUCTION_APPLIED"
    SKIPPED = "SKIPPED"


class JobEventV1(StrictModel):
    event_id: Identifier
    request_id: Identifier
    state: JobState
    occurred_at: datetime
    reason: Annotated[str, StringConstraints(max_length=500)] = ""
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    response_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    dossier_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    overlay_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None

    @field_validator("occurred_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value


class ApplicationEventV1(StrictModel):
    event_id: Identifier
    request_id: Identifier
    dossier_id: Identifier
    run_key: Identifier
    run_cutoff: datetime
    state: ApplicationState
    occurred_at: datetime
    mode: Literal["off", "shadow", "limited", "production"]
    base_score: float = Field(ge=-1.0, le=1.0)
    computed_delta: float = Field(ge=-0.1, le=0.1)
    adjusted_score: float = Field(ge=-1.0, le=1.0)

    @field_validator("occurred_at", "run_cutoff")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("application timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_mode_state(self) -> "ApplicationEventV1":
        expected = {
            ApplicationState.SHADOW_EVALUATED: "shadow",
            ApplicationState.LIMITED_APPLIED: "limited",
            ApplicationState.PRODUCTION_APPLIED: "production",
        }.get(self.state)
        if expected is not None and self.mode != expected:
            raise ValueError(f"{self.state.value} requires mode={expected}")
        return self
