"""Recomputable full-union posterior evidence for the disconnected v16 lane."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from quant_investor.agent_protocol import AgentStatus
from quant_investor.agents.fundamental_agent import BundleFundamentalDataLayer
from quant_investor.agents.macro_agent import MacroAgent
from quant_investor.bayesian.v16.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.bayesian.v16.likelihood import SignalLikelihoodMapper
from quant_investor.bayesian.v16.posterior import BayesianPosteriorEngine, CostComponents
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.codex_review.models import Stage1Request, Stage1Response
from quant_investor.codex_review.storage import (
    REQUEST_MAX_BYTES,
    RESPONSE_MAX_BYTES,
    canonical_json_bytes as review_canonical_json_bytes,
    parse_strict_json_bytes,
    sha256_bytes as review_sha256_bytes,
)
from quant_investor.codex_review.workflow import symbol_set_sha256
from quant_investor.v16.candidate_pipeline import (
    CandidateUnion,
    FormalBranchEvidence,
    FourBranchEvidence,
    LLMBranchVerdict,
    PosteriorMenuItem,
    RetrievalEvidence,
    build_candidate_union,
    build_posterior_menu,
    seal_four_branch_evidence,
    validate_stage1_review,
)
from quant_investor.fundamental_branch import FundamentalBranch

from .contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    decode_f64,
    encode_f64,
    seal_semantic,
    sha256_bytes,
    validate_semantic_seal,
)
from .runtime_identity import MODEL_BUNDLE_SCHEMA
from .target import CostVector
from .posterior_runtime import (
    BASE_RATE_TRAINING_SCHEMA,
    BOOTSTRAP_OFFSETS_SCHEMA,
    BOOTSTRAP_TRAINING_SCHEMA,
    LIKELIHOOD_TRAINING_SCHEMA,
    RETURN_MODEL_PARAMETERS_SCHEMA,
    RETURN_MODEL_TRAINING_SCHEMA,
    PosteriorRuntimeArtifacts,
    PosteriorRuntimeBundle,
)

FULL_UNION_POSTERIOR_SCHEMA = "v16.full-union-posterior-evidence.v2"
FORMAL_BRANCH_PREDICTION_SCHEMA = "v16.formal-branch-prediction.v2"
POSTERIOR_COST_INPUT_SCHEMA = "v16.posterior-cost-input.v2"
POSTERIOR_COST_MODEL_SCHEMA = "v16.posterior-cost-model.v2"

_PRIVATE_ROOT_POLICY = "v16.private-evidence-root.v2"
_STAGE1_REQUEST_SCHEMA = "codex-review-stage1-request.v1"
_STAGE1_RESPONSE_SCHEMA = "codex-review-stage1-response.v1"
_BINDING_FIELDS = (
    "run_id",
    "stage",
    "git_sha",
    "config_path",
    "config_sha256",
    "prompt_path",
    "prompt_sha256",
    "model_id",
    "model_sha256",
    "pit_pointer_path",
    "pit_pointer_sha256",
    "predecessor_sha256",
    "decision_cutoff_at",
    "expires_at",
    "request_sha256",
)


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not text or text != text.strip() or len(text) > 128:
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    if any(character not in allowed for character in text):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _symbol(value: Any) -> str:
    text = str(value or "")
    if (
        not text
        or text != text.strip().upper()
        or len(text) > 32
        or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in text)
    ):
        raise EvidenceV2Error("posterior symbol must be normalized")
    return text


def _private_ref(reference: EvidenceRef, *, schema: str | None = None) -> EvidenceRef:
    if not isinstance(reference, EvidenceRef):
        raise EvidenceV2Error("posterior evidence requires an EvidenceRef")
    if reference.root_policy != _PRIVATE_ROOT_POLICY:
        raise EvidenceV2Error("posterior evidence references must use the private root")
    if schema is not None and reference.artifact_schema != schema:
        raise EvidenceV2Error(f"posterior evidence reference schema mismatch: expected {schema}")
    return reference


def _ordered_refs(value: Sequence[EvidenceRef], *, label: str) -> tuple[EvidenceRef, ...]:
    refs = tuple(value)
    if not refs:
        raise EvidenceV2Error(f"{label} must bind at least one source reference")
    if any(reference.root_policy == "" for reference in refs):
        raise EvidenceV2Error(f"{label} contains an invalid source reference")
    keys = [(reference.absolute_path, reference.byte_sha256) for reference in refs]
    if len(keys) != len(set(keys)):
        raise EvidenceV2Error(f"{label} source references must be unique")
    return tuple(sorted(refs, key=lambda item: (item.absolute_path, item.byte_sha256)))


def _bound_canonical(value: Any, *, label: str) -> BoundCanonicalArtifact:
    if not isinstance(value, BoundCanonicalArtifact):
        raise EvidenceV2Error(f"{label} must be an actual BoundCanonicalArtifact")
    return value


@dataclass(frozen=True)
class BoundReviewArtifact:
    """Native-float Codex review JSON bound to an EvidenceRef."""

    reference: EvidenceRef
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvidenceRef):
            raise EvidenceV2Error("bound Codex review artifact requires an EvidenceRef")
        if not isinstance(self.payload, bytes):
            raise EvidenceV2Error("bound Codex review artifact payload must be bytes")
        if not self.payload or sha256_bytes(self.payload) != self.reference.byte_sha256:
            raise EvidenceV2Error("bound Codex review artifact byte SHA mismatch")


@dataclass(frozen=True)
class ValidatedStage1Review:
    request: Stage1Request
    response: Stage1Response
    candidate_union: CandidateUnion
    llm_by_symbol: Mapping[str, LLMBranchVerdict]
    retrieval_by_symbol: Mapping[str, tuple[RetrievalEvidence, ...]]


def _verify_review_model_seal(model: Stage1Request | Stage1Response, field: str) -> str:
    payload = model.model_dump(mode="json")
    supplied = str(payload.pop(field, "")).lower()
    expected = review_sha256_bytes(review_canonical_json_bytes(payload))
    if supplied != expected:
        raise EvidenceV2Error(f"Codex review {field} mismatch")
    return supplied


def _validate_stage1_fact_package(request: Stage1Request) -> None:
    package = request.fact_package.model_dump(mode="json")
    supplied = str(package.pop("payload_sha256", ""))
    expected = hashlib.sha256(
        json.dumps(
            package,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if supplied != expected:
        raise EvidenceV2Error("Stage1 fact package payload SHA mismatch")
    row_symbols = [item.symbol for item in request.fact_package.rows]
    if row_symbols != sorted(set(row_symbols)):
        raise EvidenceV2Error("Stage1 fact package symbols are not unique and sorted")
    if request.fact_package.universe_symbol_set_sha256 != symbol_set_sha256(row_symbols):
        raise EvidenceV2Error("Stage1 fact package universe SHA mismatch")
    if request.fact_package.funnel_symbol_set_sha256 != symbol_set_sha256(
        request.fact_package.funnel_symbols
    ):
        raise EvidenceV2Error("Stage1 fact package funnel SHA mismatch")
    expected_strata: dict[str, int] = {}
    for row in request.fact_package.rows:
        expected_strata[row.stratum] = expected_strata.get(row.stratum, 0) + 1
    if request.fact_package.stratum_counts != dict(sorted(expected_strata.items())):
        raise EvidenceV2Error("Stage1 fact package stratum counts mismatch")


def _parse_stage1_artifact(
    artifact: BoundReviewArtifact,
    *,
    model_type: type[Stage1Request] | type[Stage1Response],
    max_bytes: int,
    digest_field: str,
) -> Stage1Request | Stage1Response:
    _private_ref(artifact.reference, schema=model_type.model_fields["schema_version"].default)
    try:
        value = parse_strict_json_bytes(artifact.payload, max_bytes=max_bytes)
    except ValueError as exc:
        raise EvidenceV2Error(str(exc)) from exc
    if review_canonical_json_bytes(value) != artifact.payload:
        raise EvidenceV2Error("Codex review artifact bytes are not canonical")
    try:
        model = model_type.model_validate(value)
    except ValidationError as exc:
        raise EvidenceV2Error(str(exc)) from exc
    digest = _verify_review_model_seal(model, digest_field)
    if artifact.reference.semantic_sha256 != digest:
        raise EvidenceV2Error("Codex review artifact semantic SHA does not bind its model seal")
    return model


@dataclass(frozen=True)
class Stage1ReviewBinding:
    request: BoundReviewArtifact
    response: BoundReviewArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundReviewArtifact) or not isinstance(
            self.response,
            BoundReviewArtifact,
        ):
            raise EvidenceV2Error("Stage1 review binding requires actual bound review artifacts")

    def read(self) -> ValidatedStage1Review:
        request = _parse_stage1_artifact(
            self.request,
            model_type=Stage1Request,
            max_bytes=REQUEST_MAX_BYTES,
            digest_field="request_sha256",
        )
        response = _parse_stage1_artifact(
            self.response,
            model_type=Stage1Response,
            max_bytes=RESPONSE_MAX_BYTES,
            digest_field="response_sha256",
        )
        if not isinstance(request, Stage1Request) or not isinstance(response, Stage1Response):
            raise EvidenceV2Error("Codex Stage1 artifact type drift")
        _validate_stage1_fact_package(request)
        for field in _BINDING_FIELDS:
            if getattr(request, field) != getattr(response, field):
                raise EvidenceV2Error(f"Stage1 response binding mismatch: {field}")
        if response.model_sha256 != review_sha256_bytes(response.model_id.encode("utf-8")):
            raise EvidenceV2Error("Stage1 response model SHA mismatch")
        if response.symbol_set_sha256 != symbol_set_sha256(response.symbol_set):
            raise EvidenceV2Error("Stage1 response symbol-set SHA mismatch")
        candidate_union = build_candidate_union(
            request.symbol_set,
            [item.symbol for item in response.supplemental_candidates],
        )
        supplemental = [item.symbol for item in response.supplemental_candidates]
        if set(supplemental) & set(request.symbol_set):
            raise EvidenceV2Error("Stage1 supplemental candidates overlap the Funnel")
        universe_symbols = {item.symbol for item in request.fact_package.rows}
        if not set(supplemental).issubset(universe_symbols):
            raise EvidenceV2Error("Stage1 supplemental candidates escape the fact package")
        if response.symbol_set != list(candidate_union.symbols):
            raise EvidenceV2Error("Stage1 response is not the exact sealed candidate union")
        llm_verdicts = [
            LLMBranchVerdict(
                symbol=item.symbol,
                raw_score=item.raw_score,
                confidence=item.confidence,
                supporting_fact_ids=tuple(item.supporting_fact_ids),
                contradicting_fact_ids=tuple(item.contradicting_fact_ids),
                rationale=item.rationale,
            )
            for item in response.llm_verdicts
        ]
        retrieval = [
            RetrievalEvidence(
                symbol=item.symbol,
                branch=item.branch,
                supporting_fact_ids=tuple(item.supporting_fact_ids),
                contradicting_fact_ids=tuple(item.contradicting_fact_ids),
                conflict_note=item.conflict_note or None,
            )
            for item in response.retrieval_evidence
        ]
        required_retrieval = {
            (symbol, branch)
            for symbol in candidate_union.symbols
            for branch in CANONICAL_BRANCH_ORDER[:3]
        }
        actual_retrieval = {(item.symbol, item.branch) for item in retrieval}
        if actual_retrieval != required_retrieval:
            raise EvidenceV2Error("Stage1 retrieval annotations do not cover the exact Q/F/M union")
        validate_stage1_review(
            candidate_union,
            llm_verdicts=llm_verdicts,
            retrieval_evidence=retrieval,
        )
        retrieval_by_symbol = {
            symbol: tuple(item for item in retrieval if item.symbol == symbol)
            for symbol in candidate_union.symbols
        }
        return ValidatedStage1Review(
            request=request,
            response=response,
            candidate_union=candidate_union,
            llm_by_symbol={item.symbol: item for item in llm_verdicts},
            retrieval_by_symbol=retrieval_by_symbol,
        )


def _fact_evidence_id(branch: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(review_canonical_json_bytes(dict(value))).hexdigest()
    return f"{branch}:{digest}"


def _finite_branch_value(value: Any, *, label: str, probability: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceV2Error(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise EvidenceV2Error(f"{label} is not finite")
    lower, upper = (0.0, 1.0) if probability else (-1.0, 1.0)
    if not lower <= number <= upper:
        raise EvidenceV2Error(f"{label} is outside [{lower}, {upper}]")
    return number


def _replay_stage1_formal_evidence(
    stage1: ValidatedStage1Review,
) -> tuple[FormalBranchEvidence, ...]:
    symbols = stage1.candidate_union.symbols
    rows = {item.symbol: item for item in stage1.request.fact_package.rows}
    if not set(symbols).issubset(rows):
        raise EvidenceV2Error("formal replay symbols escape the Stage1 fact package")

    records: list[FormalBranchEvidence] = []
    fundamental_facts = {symbol: dict(rows[symbol].fundamental_facts) for symbol in symbols}
    macro_facts = [dict(rows[symbol].macro_facts) for symbol in symbols]
    macro_hashes = {
        hashlib.sha256(review_canonical_json_bytes(value)).hexdigest() for value in macro_facts
    }
    if len(macro_hashes) != 1:
        raise EvidenceV2Error("Stage1 symbols do not bind one common Macro fact snapshot")

    for symbol in symbols:
        row = rows[symbol]
        quant_facts = dict(row.quant_facts)
        quant_score = _finite_branch_value(
            quant_facts.get("formal_score"),
            label=f"{symbol} Quant formal_score",
        )
        quant_confidence = _finite_branch_value(
            quant_facts.get("formal_confidence"),
            label=f"{symbol} Quant formal_confidence",
            probability=True,
        )
        if quant_score != float(row.formal_quant_score):
            raise EvidenceV2Error("Stage1 Quant score projection mismatch")
        records.append(
            FormalBranchEvidence(
                symbol=symbol,
                branch="quant",
                raw_score=quant_score,
                confidence=quant_confidence,
                evidence_ids=(_fact_evidence_id("quant", quant_facts),),
            )
        )

    bundle = UnifiedDataBundle(
        market="CN",
        symbols=list(symbols),
        fundamentals=fundamental_facts,
        macro_data=macro_facts[0],
        metadata={"end_date": stage1.request.decision_cutoff_at.date().isoformat()},
    )
    fundamental_result = FundamentalBranch(
        data_layer=BundleFundamentalDataLayer(fundamental_facts),
        stock_pool=list(symbols),
        enable_document_semantics=False,
    ).run(bundle)
    if fundamental_result.success is not True or set(fundamental_result.symbol_scores) != set(
        symbols
    ):
        raise EvidenceV2Error("Fundamental deterministic replay is incomplete")
    fundamental_confidence = _finite_branch_value(
        fundamental_result.final_confidence,
        label="Fundamental replay confidence",
        probability=True,
    )
    for symbol in symbols:
        records.append(
            FormalBranchEvidence(
                symbol=symbol,
                branch="fundamental",
                raw_score=_finite_branch_value(
                    fundamental_result.symbol_scores[symbol],
                    label=f"{symbol} Fundamental replay score",
                ),
                confidence=fundamental_confidence,
                evidence_ids=(_fact_evidence_id("fundamental", fundamental_facts[symbol]),),
            )
        )

    macro_verdict = MacroAgent().run({"market_snapshot": macro_facts[0]})
    if macro_verdict.status != AgentStatus.SUCCESS:
        raise EvidenceV2Error("Macro deterministic replay is degraded")
    macro_score = _finite_branch_value(
        macro_verdict.final_score,
        label="Macro replay score",
    )
    macro_confidence = _finite_branch_value(
        macro_verdict.final_confidence,
        label="Macro replay confidence",
        probability=True,
    )
    macro_id = _fact_evidence_id("macro", macro_facts[0])
    for symbol in symbols:
        records.append(
            FormalBranchEvidence(
                symbol=symbol,
                branch="macro",
                raw_score=macro_score,
                confidence=macro_confidence,
                evidence_ids=(macro_id,),
            )
        )

    for symbol in symbols:
        verdict = stage1.llm_by_symbol[symbol]
        evidence_ids = verdict.supporting_fact_ids + verdict.contradicting_fact_ids
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise EvidenceV2Error("Stage1 LLM verdict evidence IDs are empty or duplicated")
        records.append(
            FormalBranchEvidence(
                symbol=symbol,
                branch="llm",
                raw_score=verdict.raw_score,
                confidence=verdict.confidence,
                evidence_ids=evidence_ids,
            )
        )

    by_key = {(item.symbol, item.branch): item for item in records}
    return tuple(
        by_key[(symbol, branch)] for symbol in symbols for branch in CANONICAL_BRANCH_ORDER
    )


def replay_stage1_formal_evidence(
    stage1_binding: Stage1ReviewBinding,
) -> tuple[FormalBranchEvidence, ...]:
    """Recompute Q/F/M locally and bind LLM exactly to the Stage1 response."""

    return _replay_stage1_formal_evidence(stage1_binding.read())


def build_formal_branch_prediction(
    *,
    protocol_attempt_id: str,
    symbol: str,
    branch: str,
    raw_score: float,
    confidence: float,
    evidence_ids: Sequence[str],
    stage1_request_ref: EvidenceRef,
    stage1_response_ref: EvidenceRef,
    model_bundle_ref: EvidenceRef,
    source_input_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    record = FormalBranchEvidence(
        symbol=_symbol(symbol),
        branch=str(branch),
        raw_score=float(raw_score),
        confidence=float(confidence),
        evidence_ids=tuple(str(item) for item in evidence_ids),
    )
    if len(record.evidence_ids) != len(set(record.evidence_ids)):
        raise EvidenceV2Error("formal branch evidence IDs must be unique")
    _private_ref(stage1_request_ref, schema=_STAGE1_REQUEST_SCHEMA)
    _private_ref(stage1_response_ref, schema=_STAGE1_RESPONSE_SCHEMA)
    _private_ref(model_bundle_ref, schema=MODEL_BUNDLE_SCHEMA)
    sources = _ordered_refs(source_input_refs, label="formal branch prediction")
    return seal_semantic(
        {
            "schema_version": FORMAL_BRANCH_PREDICTION_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "symbol": record.symbol,
            "branch": record.branch,
            "raw_score": encode_f64(record.raw_score),
            "confidence": encode_f64(record.confidence),
            "evidence_ids": list(record.evidence_ids),
            "stage1_request_ref": stage1_request_ref.to_dict(),
            "stage1_response_ref": stage1_response_ref.to_dict(),
            "model_bundle_ref": model_bundle_ref.to_dict(),
            "source_input_refs": [reference.to_dict() for reference in sources],
            "retrieval_used_in_scoring": False,
            "risk_advisor_used_in_scoring": False,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_formal_branch_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "symbol",
        "branch",
        "raw_score",
        "confidence",
        "evidence_ids",
        "stage1_request_ref",
        "stage1_response_ref",
        "model_bundle_ref",
        "source_input_refs",
        "retrieval_used_in_scoring",
        "risk_advisor_used_in_scoring",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != FORMAL_BRANCH_PREDICTION_SCHEMA:
        raise EvidenceV2Error("formal branch prediction envelope mismatch")
    if not isinstance(payload["evidence_ids"], list) or not isinstance(
        payload["source_input_refs"], list
    ):
        raise EvidenceV2Error("formal branch prediction lists are invalid")
    rebuilt = build_formal_branch_prediction(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        symbol=str(payload["symbol"]),
        branch=str(payload["branch"]),
        raw_score=decode_f64(payload["raw_score"], label="raw_score"),
        confidence=decode_f64(payload["confidence"], label="confidence"),
        evidence_ids=[str(item) for item in payload["evidence_ids"]],
        stage1_request_ref=EvidenceRef.from_dict(payload["stage1_request_ref"]),
        stage1_response_ref=EvidenceRef.from_dict(payload["stage1_response_ref"]),
        model_bundle_ref=EvidenceRef.from_dict(payload["model_bundle_ref"]),
        source_input_refs=[EvidenceRef.from_dict(item) for item in payload["source_input_refs"]],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("formal branch prediction is not canonical")
    return payload


def build_posterior_cost_model(
    *,
    protocol_attempt_id: str,
    model_id: str,
    costs: CostVector,
    source_input_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    if not isinstance(costs, CostVector):
        raise EvidenceV2Error("posterior cost model requires the eight-component vector")
    sources = _ordered_refs(source_input_refs, label="posterior cost model")
    if any(reference.root_policy != _PRIVATE_ROOT_POLICY for reference in sources):
        raise EvidenceV2Error("posterior cost model sources must use the private root")
    return seal_semantic(
        {
            "schema_version": POSTERIOR_COST_MODEL_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "model_id": _safe_id(model_id, label="cost model_id"),
            "method": "fixed-round-trip-rate.v1",
            "costs": costs.to_rows(),
            "source_input_refs": [reference.to_dict() for reference in sources],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_posterior_cost_model(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "model_id",
        "method",
        "costs",
        "source_input_refs",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != POSTERIOR_COST_MODEL_SCHEMA:
        raise EvidenceV2Error("posterior cost model envelope mismatch")
    if payload["method"] != "fixed-round-trip-rate.v1":
        raise EvidenceV2Error("posterior cost model method mismatch")
    if not isinstance(payload["costs"], list):
        raise EvidenceV2Error("posterior cost model costs must be a list")
    if not isinstance(payload["source_input_refs"], list):
        raise EvidenceV2Error("posterior cost model source refs must be a list")
    rebuilt = build_posterior_cost_model(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        model_id=str(payload["model_id"]),
        costs=CostVector.from_rows(payload["costs"]),
        source_input_refs=[EvidenceRef.from_dict(item) for item in payload["source_input_refs"]],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("posterior cost model is not canonical")
    return payload


def build_posterior_cost_input(
    *,
    protocol_attempt_id: str,
    symbol: str,
    cost_model: BoundCanonicalArtifact,
    stage1_request_ref: EvidenceRef,
) -> dict[str, Any]:
    _bound_canonical(cost_model, label="posterior cost model")
    _private_ref(cost_model.reference, schema=POSTERIOR_COST_MODEL_SCHEMA)
    _private_ref(stage1_request_ref, schema=_STAGE1_REQUEST_SCHEMA)
    model = validate_posterior_cost_model(cost_model.read())
    attempt = _safe_id(protocol_attempt_id, label="protocol_attempt_id")
    if model["protocol_attempt_id"] != attempt:
        raise EvidenceV2Error("posterior cost model crosses protocol attempts")
    costs = CostVector.from_rows(model["costs"])
    fee = math.fsum(costs.values[:5])
    slippage = math.fsum(costs.values[5:7])
    market_impact = costs.values[7]
    return seal_semantic(
        {
            "schema_version": POSTERIOR_COST_INPUT_SCHEMA,
            "protocol_attempt_id": attempt,
            "symbol": _symbol(symbol),
            "costs": costs.to_rows(),
            "fee": encode_f64(fee),
            "slippage": encode_f64(slippage),
            "market_impact": encode_f64(market_impact),
            "cost_model_ref": cost_model.reference.to_dict(),
            "stage1_request_ref": stage1_request_ref.to_dict(),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_posterior_cost_input(
    value: Mapping[str, Any],
    *,
    cost_model: BoundCanonicalArtifact,
    stage1_request_ref: EvidenceRef,
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "symbol",
        "costs",
        "fee",
        "slippage",
        "market_impact",
        "cost_model_ref",
        "stage1_request_ref",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != POSTERIOR_COST_INPUT_SCHEMA:
        raise EvidenceV2Error("posterior cost input envelope mismatch")
    rebuilt = build_posterior_cost_input(
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        symbol=str(payload["symbol"]),
        cost_model=cost_model,
        stage1_request_ref=stage1_request_ref,
    )
    if rebuilt != payload:
        raise EvidenceV2Error("posterior cost input is not canonical")
    return payload


def _formal_inputs(
    *,
    protocol_attempt_id: str,
    stage1_binding: Stage1ReviewBinding,
    stage1: ValidatedStage1Review,
    runtime: PosteriorRuntimeBundle,
    artifacts: Sequence[BoundCanonicalArtifact],
) -> tuple[tuple[FourBranchEvidence, ...], dict[tuple[str, str], EvidenceRef]]:
    expected_count = len(stage1.candidate_union.symbols) * len(CANONICAL_BRANCH_ORDER)
    if len(artifacts) != expected_count:
        raise EvidenceV2Error("formal branch artifacts do not cover the exact full union")
    replayed = {(item.symbol, item.branch): item for item in _replay_stage1_formal_evidence(stage1)}
    records: list[FormalBranchEvidence] = []
    source_refs: dict[tuple[str, str], EvidenceRef] = {}
    for artifact in artifacts:
        _bound_canonical(artifact, label="formal branch prediction")
        _private_ref(artifact.reference, schema=FORMAL_BRANCH_PREDICTION_SCHEMA)
        payload = validate_formal_branch_prediction(artifact.read())
        symbol = str(payload["symbol"])
        branch = str(payload["branch"])
        key = (symbol, branch)
        if key in source_refs:
            raise EvidenceV2Error("duplicate formal branch prediction artifact")
        if (
            payload["protocol_attempt_id"] != protocol_attempt_id
            or payload["stage1_request_ref"] != stage1_binding.request.reference.to_dict()
            or payload["stage1_response_ref"] != stage1_binding.response.reference.to_dict()
            or payload["model_bundle_ref"] != runtime.model_refs[branch].to_dict()
        ):
            raise EvidenceV2Error("formal branch prediction lineage mismatch")
        record = FormalBranchEvidence(
            symbol=symbol,
            branch=branch,
            raw_score=decode_f64(payload["raw_score"], label="raw_score"),
            confidence=decode_f64(payload["confidence"], label="confidence"),
            evidence_ids=tuple(str(item) for item in payload["evidence_ids"]),
        )
        expected_record = replayed.get(key)
        if expected_record is None or record != expected_record:
            raise EvidenceV2Error(
                f"{branch} formal evidence drifts from deterministic Stage1 replay"
            )
        expected_sources = _ordered_refs(
            (
                (stage1_binding.request.reference, stage1_binding.response.reference)
                if branch == "llm"
                else (stage1_binding.request.reference,)
            ),
            label="formal replay source",
        )
        if payload["source_input_refs"] != [reference.to_dict() for reference in expected_sources]:
            raise EvidenceV2Error("formal branch prediction source refs drift from replay")
        records.append(record)
        source_refs[key] = artifact.reference
    try:
        sealed = seal_four_branch_evidence(stage1.candidate_union, records)
    except ValueError as exc:
        raise EvidenceV2Error(str(exc)) from exc
    return sealed, source_refs


def _cost_inputs(
    *,
    protocol_attempt_id: str,
    symbols: Sequence[str],
    stage1_request_ref: EvidenceRef,
    cost_model: BoundCanonicalArtifact,
    artifacts: Sequence[BoundCanonicalArtifact],
) -> tuple[dict[str, CostComponents], dict[str, EvidenceRef]]:
    if len(artifacts) != len(symbols):
        raise EvidenceV2Error("posterior cost artifacts do not cover the exact full union")
    costs: dict[str, CostComponents] = {}
    refs: dict[str, EvidenceRef] = {}
    for artifact in artifacts:
        _bound_canonical(artifact, label="posterior cost input")
        _private_ref(artifact.reference, schema=POSTERIOR_COST_INPUT_SCHEMA)
        payload = validate_posterior_cost_input(
            artifact.read(),
            cost_model=cost_model,
            stage1_request_ref=stage1_request_ref,
        )
        symbol = str(payload["symbol"])
        if symbol in costs or payload["protocol_attempt_id"] != protocol_attempt_id:
            raise EvidenceV2Error("posterior cost artifact lineage is duplicated or invalid")
        costs[symbol] = CostComponents(
            fee=decode_f64(payload["fee"], label="fee"),
            slippage=decode_f64(payload["slippage"], label="slippage"),
            market_impact=decode_f64(payload["market_impact"], label="market_impact"),
        )
        refs[symbol] = artifact.reference
    if set(costs) != set(symbols):
        raise EvidenceV2Error("posterior costs drift from the full union symbol set")
    return costs, refs


def _retrieval_projection(items: Sequence[RetrievalEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "branch": item.branch,
            "supporting_fact_ids": list(item.supporting_fact_ids),
            "contradicting_fact_ids": list(item.contradicting_fact_ids),
            "conflict_note": item.conflict_note,
        }
        for item in items
    ]


def _compute_full_union_payload(
    *,
    protocol_attempt_id: str,
    stage1_binding: Stage1ReviewBinding,
    stage1: ValidatedStage1Review,
    runtime: PosteriorRuntimeBundle,
    formal_evidence: Sequence[FourBranchEvidence],
    formal_refs: Mapping[tuple[str, str], EvidenceRef],
    costs_by_symbol: Mapping[str, CostComponents],
    cost_refs: Mapping[str, EvidenceRef],
    cost_model_ref: EvidenceRef,
) -> dict[str, Any]:
    mapper = SignalLikelihoodMapper(
        calibration_store=runtime.calibration_store,
        correlation_matrix=runtime.correlations,
    )
    engine = BayesianPosteriorEngine(
        return_calibration_model=runtime.return_calibration,
        bootstrap_artifact=runtime.bootstrap_artifact,
    )
    rows: list[dict[str, Any]] = []
    menu_items: list[PosteriorMenuItem] = []
    for evidence in formal_evidence:
        likelihoods = mapper.compute_from_sealed_evidence(evidence)
        posterior = engine.compute_posterior(
            runtime.prior,
            likelihoods,
            symbol=evidence.symbol,
            costs=costs_by_symbol[evidence.symbol],
        )
        if posterior.posterior_edge_after_costs is None:
            raise EvidenceV2Error("full-union posterior requires complete cost evidence")
        menu_items.append(
            PosteriorMenuItem(
                symbol=evidence.symbol,
                posterior_win_rate=posterior.posterior_win_rate,
                posterior_expected_alpha=posterior.posterior_expected_alpha,
                posterior_edge_after_costs=posterior.posterior_edge_after_costs,
            )
        )
        calibrated = dict(likelihoods.as_list())
        rows.append(
            {
                "symbol": evidence.symbol,
                "prior_probability": encode_f64(runtime.prior.base_rate),
                "posterior_win_rate": encode_f64(posterior.posterior_win_rate),
                "posterior_expected_alpha": encode_f64(posterior.posterior_expected_alpha),
                "posterior_edge_after_costs": encode_f64(posterior.posterior_edge_after_costs),
                "posterior_win_rate_interval_90": [
                    encode_f64(value) for value in posterior.posterior_win_rate_interval_90
                ],
                "posterior_expected_alpha_interval_90": [
                    encode_f64(value) for value in posterior.posterior_expected_alpha_interval_90
                ],
                "raw_evidence_increment": encode_f64(posterior.raw_evidence_increment),
                "correlation_adjusted_evidence_increment": encode_f64(
                    posterior.correlation_adjusted_evidence_increment
                ),
                "correlation_vif": encode_f64(posterior.correlation_vif),
                "correlation_vif_shrink": encode_f64(posterior.correlation_vif_shrink),
                "cost_input_ref": cost_refs[evidence.symbol].to_dict(),
                "branch_evidence": [
                    {
                        "branch": branch.branch,
                        "raw_score": encode_f64(branch.raw_score),
                        "confidence": encode_f64(branch.confidence),
                        "calibrated_probability": encode_f64(calibrated[branch.branch]),
                        "evidence_ids": list(branch.evidence_ids),
                        "source_ref": formal_refs[(evidence.symbol, branch.branch)].to_dict(),
                        "model_bundle_ref": runtime.model_refs[branch.branch].to_dict(),
                    }
                    for branch in evidence.branches
                ],
                "retrieval_advisory": _retrieval_projection(
                    stage1.retrieval_by_symbol[evidence.symbol]
                ),
            }
        )
    ranked = build_posterior_menu(menu_items, menu_limit=len(menu_items))
    rank_by_symbol = {item.symbol: index + 1 for index, item in enumerate(ranked)}
    for row in rows:
        row["rank"] = rank_by_symbol[str(row["symbol"])]
    return {
        "schema_version": FULL_UNION_POSTERIOR_SCHEMA,
        "protocol_attempt_id": _safe_id(
            protocol_attempt_id,
            label="protocol_attempt_id",
        ),
        "run_id": stage1.request.run_id,
        "stage1_request_ref": stage1_binding.request.reference.to_dict(),
        "stage1_response_ref": stage1_binding.response.reference.to_dict(),
        "runtime_refs": runtime.refs_projection(),
        "cost_model_ref": cost_model_ref.to_dict(),
        "branch_order": list(CANONICAL_BRANCH_ORDER),
        "branch_weights": {branch: encode_f64(0.25) for branch in CANONICAL_BRANCH_ORDER},
        "candidate_symbols": list(stage1.candidate_union.symbols),
        "ranked_symbols": [item.symbol for item in ranked],
        "menu_symbols": [item.symbol for item in ranked[:50]],
        "posteriors": rows,
        "retrieval_used_in_scoring": False,
        "risk_advisor_used_in_scoring": False,
        "activation_candidate": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
        "blockers": [
            "evidence_v2_disconnected_from_authorizing_consumers",
            "global_attempt_registry_authority_not_integrated",
        ],
    }


def build_full_union_posterior_evidence(
    *,
    protocol_attempt_id: str,
    stage1_binding: Stage1ReviewBinding,
    runtime_artifacts: PosteriorRuntimeArtifacts,
    cost_model: BoundCanonicalArtifact,
    formal_branch_artifacts: Sequence[BoundCanonicalArtifact],
    cost_artifacts: Sequence[BoundCanonicalArtifact],
) -> dict[str, Any]:
    if not isinstance(stage1_binding, Stage1ReviewBinding):
        raise EvidenceV2Error("posterior producer requires an actual Stage1ReviewBinding")
    if not isinstance(runtime_artifacts, PosteriorRuntimeArtifacts):
        raise EvidenceV2Error("posterior producer requires actual runtime artifacts")
    _bound_canonical(cost_model, label="posterior cost model")
    if isinstance(formal_branch_artifacts, (str, bytes)) or not isinstance(
        formal_branch_artifacts,
        Sequence,
    ):
        raise EvidenceV2Error("formal branch artifacts must be a sequence")
    if isinstance(cost_artifacts, (str, bytes)) or not isinstance(cost_artifacts, Sequence):
        raise EvidenceV2Error("posterior cost artifacts must be a sequence")
    attempt = _safe_id(protocol_attempt_id, label="protocol_attempt_id")
    runtime = PosteriorRuntimeBundle(artifacts=runtime_artifacts)
    if runtime.protocol_attempt_id != attempt:
        raise EvidenceV2Error("posterior runtime crosses protocol attempts")
    stage1 = stage1_binding.read()
    llm_provider = runtime.model_payloads["llm"]["llm_provider_build"]
    if not isinstance(llm_provider, Mapping) or llm_provider.get("model_id") != (
        stage1.response.model_id
    ):
        raise EvidenceV2Error("Stage1 response model differs from frozen LLM bundle")
    formal_evidence, formal_refs = _formal_inputs(
        protocol_attempt_id=attempt,
        stage1_binding=stage1_binding,
        stage1=stage1,
        runtime=runtime,
        artifacts=formal_branch_artifacts,
    )
    costs, cost_refs = _cost_inputs(
        protocol_attempt_id=attempt,
        symbols=stage1.candidate_union.symbols,
        stage1_request_ref=stage1_binding.request.reference,
        cost_model=cost_model,
        artifacts=cost_artifacts,
    )
    return seal_semantic(
        _compute_full_union_payload(
            protocol_attempt_id=attempt,
            stage1_binding=stage1_binding,
            stage1=stage1,
            runtime=runtime,
            formal_evidence=formal_evidence,
            formal_refs=formal_refs,
            costs_by_symbol=costs,
            cost_refs=cost_refs,
            cost_model_ref=cost_model.reference,
        )
    )


def validate_full_union_posterior_evidence(
    value: Mapping[str, Any],
    *,
    stage1_binding: Stage1ReviewBinding,
    runtime_artifacts: PosteriorRuntimeArtifacts,
    cost_model: BoundCanonicalArtifact,
    formal_branch_artifacts: Sequence[BoundCanonicalArtifact],
    cost_artifacts: Sequence[BoundCanonicalArtifact],
) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    if payload.get("schema_version") != FULL_UNION_POSTERIOR_SCHEMA:
        raise EvidenceV2Error("full-union posterior evidence schema mismatch")
    for field in (
        "retrieval_used_in_scoring",
        "risk_advisor_used_in_scoring",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
    ):
        if payload.get(field) is not False:
            raise EvidenceV2Error("full-union posterior evidence must be nonauthorizing")
    recomputed = build_full_union_posterior_evidence(
        protocol_attempt_id=str(payload.get("protocol_attempt_id", "")),
        stage1_binding=stage1_binding,
        runtime_artifacts=runtime_artifacts,
        cost_model=cost_model,
        formal_branch_artifacts=formal_branch_artifacts,
        cost_artifacts=cost_artifacts,
    )
    if recomputed != payload:
        raise EvidenceV2Error("full-union posterior differs from deterministic recomputation")
    return payload


__all__ = [
    "BASE_RATE_TRAINING_SCHEMA",
    "BOOTSTRAP_OFFSETS_SCHEMA",
    "BOOTSTRAP_TRAINING_SCHEMA",
    "BoundReviewArtifact",
    "FORMAL_BRANCH_PREDICTION_SCHEMA",
    "FULL_UNION_POSTERIOR_SCHEMA",
    "LIKELIHOOD_TRAINING_SCHEMA",
    "POSTERIOR_COST_INPUT_SCHEMA",
    "PosteriorRuntimeBundle",
    "RETURN_MODEL_PARAMETERS_SCHEMA",
    "RETURN_MODEL_TRAINING_SCHEMA",
    "Stage1ReviewBinding",
    "build_formal_branch_prediction",
    "build_full_union_posterior_evidence",
    "build_posterior_cost_input",
    "replay_stage1_formal_evidence",
    "validate_formal_branch_prediction",
    "validate_full_union_posterior_evidence",
    "validate_posterior_cost_input",
]
