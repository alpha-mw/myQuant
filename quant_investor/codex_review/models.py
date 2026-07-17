"""Strict schemas for the offline v16 Codex review handoff."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
Symbol = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
BoundedText = Annotated[str, StringConstraints(max_length=20_000)]
Rationale = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
    )


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class ReviewState(str, Enum):
    S1_PREPARED = "S1_PREPARED"
    S1_EXPORTED = "S1_EXPORTED"
    S1_RECEIVED = "S1_RECEIVED"
    S1_VALIDATED = "S1_VALIDATED"
    MENU_SEALED = "MENU_SEALED"
    S2_PREPARED = "S2_PREPARED"
    S2_EXPORTED = "S2_EXPORTED"
    S2_RECEIVED = "S2_RECEIVED"
    S2_VALIDATED = "S2_VALIDATED"
    CAPITAL_MAPPED = "CAPITAL_MAPPED"
    AWAITING_HUMAN_AUTH = "AWAITING_HUMAN_AUTH"
    AUTHORIZED = "AUTHORIZED"
    BLOCKED = "BLOCKED"
    EXPIRED = "EXPIRED"


class EvidenceAnnotations(StrictModel):
    """Text-only branch annotations; scores/confidence/likelihood are not fields."""

    quant: BoundedText = ""
    fundamental: BoundedText = ""
    macro: BoundedText = ""


class CandidatePacket(StrictModel):
    symbol: Symbol
    annotations: EvidenceAnnotations


class Stage1Payload(StrictModel):
    schema_version: Literal["codex-review-stage1-payload.v1"] = "codex-review-stage1-payload.v1"
    candidates: list[CandidatePacket] = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_symbols(self) -> "Stage1Payload":
        symbols = [item.symbol for item in self.candidates]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Stage1 candidate symbols must be unique")
        return self


class PITFactRowModel(StrictModel):
    symbol: Symbol
    stratum: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    eligibility_receipt_sha256: Sha256
    formal_quant_score: float = Field(ge=-1.0, le=1.0)
    quant_facts: dict[str, object]
    fundamental_facts: dict[str, object]
    macro_facts: dict[str, object]

    @model_validator(mode="after")
    def validate_fact_groups(self) -> "PITFactRowModel":
        if not self.quant_facts or not self.fundamental_facts or not self.macro_facts:
            raise ValueError("Q/F/M fact groups must all be non-empty")
        return self


class Stage1FactPackageModel(StrictModel):
    schema_version: Literal["v16.codex-stage1.request.v1"]
    target_definition: Literal["CN_20D_NET_EXCESS_VS_CSI300_GT_0"]
    market: Literal["CN"]
    cutoff_at: str
    expires_at: str
    pit_pointer_sha256: Sha256
    rows: list[PITFactRowModel] = Field(min_length=1)
    funnel_symbols: list[Symbol] = Field(min_length=1, max_length=500)
    universe_symbol_set_sha256: Sha256
    funnel_symbol_set_sha256: Sha256
    stratum_counts: dict[str, int]
    payload_sha256: Sha256


class RequestBindings(StrictModel):
    run_id: Identifier
    git_sha: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,64}$")]
    config_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    config_sha256: Sha256
    prompt_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    prompt_sha256: Sha256
    model_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
    ]
    model_sha256: Sha256
    pit_pointer_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    pit_pointer_sha256: Sha256
    symbol_set: list[Symbol] = Field(min_length=1, max_length=600)
    symbol_set_sha256: Sha256
    predecessor_sha256: Sha256
    decision_cutoff_at: datetime
    expires_at: datetime

    @field_validator("decision_cutoff_at", "expires_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime, info):
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_common(self) -> "RequestBindings":
        if self.expires_at <= self.decision_cutoff_at:
            raise ValueError("expires_at must be after decision_cutoff_at")
        if len(self.symbol_set) != len(set(self.symbol_set)):
            raise ValueError("symbol_set must be unique")
        return self


class Stage1Request(RequestBindings):
    schema_version: Literal["codex-review-stage1-request.v1"] = "codex-review-stage1-request.v1"
    stage: Literal[1] = 1
    fact_package: Stage1FactPackageModel
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_candidate_set(self) -> "Stage1Request":
        if self.fact_package.funnel_symbols != self.symbol_set:
            raise ValueError("Stage1 Funnel symbols do not match symbol_set")
        return self


class SupplementalCandidate(StrictModel):
    symbol: Symbol
    retrieval_reason: Rationale


class RetrievalEvidence(StrictModel):
    symbol: Symbol
    branch: Literal["quant", "fundamental", "macro"]
    supporting_fact_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    contradicting_fact_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    conflict_note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)] = ""


class Stage1Verdict(StrictModel):
    symbol: Symbol
    raw_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_fact_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    contradicting_fact_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    rationale: Rationale


class ResponseBindings(RequestBindings):
    stage: Literal[1, 2]
    request_sha256: Sha256


class Stage1Response(ResponseBindings):
    schema_version: Literal["codex-review-stage1-response.v1"] = "codex-review-stage1-response.v1"
    stage: Literal[1] = 1
    supplemental_candidates: list[SupplementalCandidate] = Field(
        default_factory=list,
        max_length=100,
    )
    retrieval_evidence: list[RetrievalEvidence] = Field(
        default_factory=list,
        max_length=1800,
    )
    llm_verdicts: list[Stage1Verdict] = Field(min_length=1, max_length=600)
    response_sha256: Sha256

    @model_validator(mode="after")
    def validate_unique_lists(self) -> "Stage1Response":
        for label, symbols in (
            (
                "supplemental_candidates",
                [item.symbol for item in self.supplemental_candidates],
            ),
            ("llm_verdicts", [item.symbol for item in self.llm_verdicts]),
        ):
            if len(symbols) != len(set(symbols)):
                raise ValueError(f"{label} symbols must be unique")
        evidence_keys = [(item.symbol, item.branch) for item in self.retrieval_evidence]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("retrieval_evidence (symbol, branch) pairs must be unique")
        return self


class MenuSeal(StrictModel):
    schema_version: Literal["codex-review-menu.v1"] = "codex-review-menu.v1"
    run_id: Identifier
    stage1_response_sha256: Sha256
    symbols: list[Symbol] = Field(min_length=1, max_length=50)
    items: list["MenuEntry"] = Field(min_length=1, max_length=50)
    existing_weights: dict[Symbol, float]
    sealed_at: datetime
    menu_sha256: Sha256

    @field_validator("sealed_at")
    @classmethod
    def validate_sealed_at(cls, value: datetime):
        return _require_aware(value, "sealed_at")

    @model_validator(mode="after")
    def validate_symbols(self) -> "MenuSeal":
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError("menu symbols must be unique")
        if [item.symbol for item in self.items] != self.symbols:
            raise ValueError("menu items must preserve the sealed symbol order")
        if any(value < 0.0 or value > 1.0 for value in self.existing_weights.values()):
            raise ValueError("existing_weights must be within [0, 1]")
        missing_holdings = sorted(
            symbol
            for symbol, weight in self.existing_weights.items()
            if weight > 1e-6 and symbol not in set(self.symbols)
        )
        if missing_holdings:
            raise ValueError(f"current holdings absent from sealed menu: {missing_holdings}")
        if set(self.existing_weights) != set(self.symbols):
            raise ValueError("existing_weights must cover the complete sealed menu")
        if any(
            abs(item.existing_weight - self.existing_weights[item.symbol]) > 1e-6
            for item in self.items
        ):
            raise ValueError("menu item existing_weight binding mismatch")
        return self


class RiskAdvisory(StrictModel):
    severity: Literal["low", "medium", "high", "extreme"]
    flags: list[Annotated[str, StringConstraints(min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=100
    )
    scenarios: list[Annotated[str, StringConstraints(min_length=1, max_length=2000)]] = Field(
        default_factory=list, max_length=100
    )
    suggestions: list[Annotated[str, StringConstraints(min_length=1, max_length=2000)]] = Field(
        default_factory=list, max_length=100
    )
    rationale: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
    ]


class MenuBranchEvidence(StrictModel):
    """Formal calibrated evidence exposed to Stage 2 without retrieval mixing."""

    branch: Literal["quant", "fundamental", "macro", "llm"]
    raw_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=100)


class MenuEntry(StrictModel):
    symbol: Symbol
    posterior_win_rate: float = Field(ge=0.0, le=1.0)
    posterior_expected_alpha: float
    posterior_edge_after_costs: float | None
    branch_evidence: list[MenuBranchEvidence] = Field(min_length=4, max_length=4)
    retrieval_advisory: list[RetrievalEvidence] = Field(default_factory=list, max_length=300)
    risk_advisory: RiskAdvisory
    existing_weight: float = Field(ge=0.0, le=1.0)
    reference_price: float = Field(gt=0.0)
    existing_shares: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_advisory_binding(self) -> "MenuEntry":
        if [item.branch for item in self.branch_evidence] != [
            "quant",
            "fundamental",
            "macro",
            "llm",
        ]:
            raise ValueError("menu branch_evidence must preserve exact Q/F/M/LLM order")
        if any(item.symbol != self.symbol for item in self.retrieval_advisory):
            raise ValueError("retrieval advisory contains symbol drift")
        return self


class Stage2Request(RequestBindings):
    schema_version: Literal["codex-review-stage2-request.v1"] = "codex-review-stage2-request.v1"
    stage: Literal[2] = 2
    menu_sha256: Sha256
    existing_weights: dict[Symbol, float]
    menu: list[MenuEntry] = Field(min_length=1, max_length=50)
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_menu(self) -> "Stage2Request":
        if [item.symbol for item in self.menu] != self.symbol_set:
            raise ValueError("Stage2 menu does not match symbol_set")
        return self


class PortfolioAction(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"
    SELL = "SELL"


class Stage2Verdict(StrictModel):
    symbol: Symbol
    action: PortfolioAction
    selected_for_portfolio: bool
    target_weight: float = Field(ge=0.0, le=1.0)
    rationale: Rationale
    severe_risks: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    ] = Field(default_factory=list, max_length=20)
    risk_acceptance_rationale: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=8_000)
    ] = ""

    @model_validator(mode="after")
    def validate_selection(self) -> "Stage2Verdict":
        if self.action in {PortfolioAction.AVOID, PortfolioAction.SELL} and (
            self.selected_for_portfolio or self.target_weight != 0.0
        ):
            raise ValueError("AVOID/SELL must be unselected with zero target_weight")
        if self.action == PortfolioAction.BUY and (
            not self.selected_for_portfolio or self.target_weight <= 0.0
        ):
            raise ValueError("BUY requires positive selected target_weight")
        if (
            self.action == PortfolioAction.BUY
            and self.severe_risks
            and not self.risk_acceptance_rationale
        ):
            raise ValueError("BUY with severe risks requires risk_acceptance_rationale")
        return self


class Stage2Response(ResponseBindings):
    schema_version: Literal["codex-review-stage2-response.v1"] = "codex-review-stage2-response.v1"
    stage: Literal[2] = 2
    menu_sha256: Sha256
    verdicts: list[Stage2Verdict] = Field(min_length=1, max_length=50)
    cash_ratio: float = Field(ge=0.0, le=1.0)
    response_sha256: Sha256

    @model_validator(mode="after")
    def validate_portfolio(self) -> "Stage2Response":
        symbols = [item.symbol for item in self.verdicts]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Stage2 verdict symbols must be unique")
        return self


class CapitalPosition(StrictModel):
    symbol: Symbol
    target_weight: float = Field(gt=0.0, le=1.0)
    capital_amount: float = Field(gt=0.0)
    reference_price: float = Field(gt=0.0)
    raw_target_shares: float = Field(gt=0.0)
    target_shares: float = Field(gt=0.0)


class CapitalMap(StrictModel):
    schema_version: Literal["codex-review-capital-map.v1"] = "codex-review-capital-map.v1"
    run_id: Identifier
    stage2_response_sha256: Sha256
    mapped_at: datetime
    total_capital: float = Field(gt=0.0)
    positions: list[CapitalPosition] = Field(default_factory=list, max_length=12)
    cash_ratio: float = Field(ge=0.0, le=1.0)
    cash_amount: float = Field(ge=0.0)
    capital_map_sha256: Sha256

    @field_validator("mapped_at")
    @classmethod
    def validate_mapped_at(cls, value: datetime):
        return _require_aware(value, "mapped_at")


class AuthorizationDecision(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    BLOCKED = "BLOCKED"


class HumanAuthorization(StrictModel):
    schema_version: Literal["codex-review-human-authorization.v1"] = (
        "codex-review-human-authorization.v1"
    )
    run_id: Identifier
    stage2_response_sha256: Sha256
    capital_map_sha256: Sha256
    decision: AuthorizationDecision
    authorized_by: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
    ]
    authorized_at: datetime
    expires_at: datetime
    rationale: Rationale
    receipt_sha256: Sha256

    @field_validator("authorized_at", "expires_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime, info):
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> "HumanAuthorization":
        if self.expires_at <= self.authorized_at:
            raise ValueError("authorization expires_at must follow authorized_at")
        if self.authorized_by.casefold() in {
            "codex",
            "llm",
            "model",
            "automation",
            "system",
        }:
            raise ValueError("authorization must identify a human authorizer")
        return self


class RunState(StrictModel):
    schema_version: Literal["codex-review-run-state.v1"] = "codex-review-run-state.v1"
    run_id: Identifier
    state: ReviewState
    revision: int = Field(ge=1)
    updated_at: datetime
    repo_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    git_sha: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,64}$")]
    config_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    config_sha256: Sha256
    prompt_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    prompt_sha256: Sha256
    model_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    model_sha256: Sha256
    pit_pointer_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    pit_pointer_sha256: Sha256
    decision_cutoff_at: datetime
    expires_at: datetime
    stage1_request_path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    stage1_request_sha256: Sha256
    stage1_response_path: str = ""
    stage1_response_sha256: str = ""
    final_symbol_set: list[Symbol] = Field(default_factory=list, max_length=600)
    final_symbol_set_sha256: str = ""
    menu_path: str = ""
    menu_sha256: str = ""
    stage2_request_path: str = ""
    stage2_request_sha256: str = ""
    stage2_response_path: str = ""
    stage2_response_sha256: str = ""
    capital_map_path: str = ""
    capital_map_sha256: str = ""
    authorization_path: str = ""
    authorization_sha256: str = ""
    blockers: list[Annotated[str, StringConstraints(min_length=1, max_length=512)]] = Field(
        default_factory=list, max_length=50
    )

    @field_validator("updated_at", "decision_cutoff_at", "expires_at")
    @classmethod
    def validate_aware_datetime(cls, value: datetime, info):
        return _require_aware(value, info.field_name)

    @field_validator(
        "stage1_response_sha256",
        "final_symbol_set_sha256",
        "menu_sha256",
        "stage2_request_sha256",
        "stage2_response_sha256",
        "capital_map_sha256",
        "authorization_sha256",
    )
    @classmethod
    def validate_optional_sha256(cls, value: str, info):
        if value and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError(f"{info.field_name} must be empty or a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_final_symbol_set(self) -> "RunState":
        if len(self.final_symbol_set) != len(set(self.final_symbol_set)):
            raise ValueError("final_symbol_set must be unique")
        return self
