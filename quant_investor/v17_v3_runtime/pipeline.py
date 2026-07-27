"""Typed, admitted-source-only orchestration for V17 protocol v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext
import hashlib
from io import BytesIO
import math
from typing import Any, Mapping, cast

from quant_investor.v17_v3_contract.canonical import canonical_bytes
from quant_investor.v17_v3_contract.identities import (
    IdentityContractError,
    require_sha256,
)
from quant_investor.v17_v3_contract.namespace import (
    formal_run_path,
    shadow_run_path,
)
from quant_investor.v17_v3_contract.resources import load_packaged_json
from quant_investor.v17_v3_contract.validators import (
    validate_branch_same_pool_binding,
)

from .artifacts import (
    RuntimeArtifact,
    runtime_artifact,
    seal_typed_artifact,
)
from .authority import PROTOCOL_VERSION, authority_envelope
from .redaction import assert_public_envelope_safe, redact_public
from .sources import AdmittedSources, SourceAdmissionError
from .storage import (
    PRIVATE_RUNS_ROOT,
)

SHADOW_MODE = "shadow"
FORMAL_RESEARCH_MODE = "formal-research"
MODES = frozenset({SHADOW_MODE, FORMAL_RESEARCH_MODE})


class PipelineOrchestrationError(ValueError):
    """The admitted closure cannot satisfy the fixed V3 pipeline."""

    exit_code = 2


@dataclass(frozen=True)
class PipelineRequest:
    mode: str
    admitted_sources: AdmittedSources
    active_receipt_sha256: str | None = None


@dataclass(frozen=True)
class PipelineTerminal:
    state: str
    publishable: bool
    rank_complete: bool
    portfolio_complete: bool
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    mode: str
    run_id: str
    strategy_id: str
    cutoff: str
    locator_byte_sha256: str
    calibration_label: str
    factor_baseline_mode: str | None
    portfolio_basis: str | None
    allocation_policy_sha256: str | None
    overlay_stages: tuple[Mapping[str, Any], ...]
    terminal: PipelineTerminal
    preselection: Any | None
    quant_branch: Any | None
    fundamental_branch: Any | None
    fusion: Any | None
    deep: tuple[Any, ...]
    base_targets: Mapping[str, Any]
    overlay: Any | None
    final_targets: Mapping[str, Any]
    artifacts: tuple[RuntimeArtifact, ...] = ()
    terminal_artifact: RuntimeArtifact | None = None

    def private_core_wire(self) -> dict[str, Any]:
        """Return private canonical content; never emit this on stdout or logs."""

        if self.terminal_artifact is not None:
            return dict(self.terminal_artifact.document)
        return _wire(
            {
                "mode": self.mode,
                "run_id": self.run_id,
                "strategy_id": self.strategy_id,
                "cutoff": self.cutoff,
                "locator_byte_sha256": self.locator_byte_sha256,
                "calibration_label": self.calibration_label,
                "factor_baseline_mode": self.factor_baseline_mode,
                "portfolio_basis": self.portfolio_basis,
                "allocation_policy_sha256": self.allocation_policy_sha256,
                "overlay_stages": self.overlay_stages,
                "terminal": self.terminal,
                "preselection": self.preselection,
                "quant_branch": self.quant_branch,
                "fundamental_branch": self.fundamental_branch,
                "fusion": self.fusion,
                "deep": self.deep,
                "base_targets": self.base_targets,
                "overlay": self.overlay,
                "final_targets": self.final_targets,
                **authority_envelope(
                    formal_research_active=(
                        self.mode == FORMAL_RESEARCH_MODE and self.terminal.publishable
                    )
                ),
            }
        )

    @property
    def core_sha256(self) -> str:
        if self.terminal_artifact is not None:
            return self.terminal_artifact.byte_sha256
        return hashlib.sha256(canonical_bytes(self.private_core_wire())).hexdigest()

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.pipeline-public-result.v1",
            "mode": self.mode,
            "status": self.terminal.state,
            "publishable": self.terminal.publishable,
            "rank_complete": self.terminal.rank_complete,
            "portfolio_complete": self.terminal.portfolio_complete,
            "blockers": redact_public(list(self.terminal.blockers)),
            "locator_byte_sha256": self.locator_byte_sha256,
            "core_sha256": self.core_sha256,
            "preselection_count": _count_selected(self.preselection),
            "fusion_count": _count_selected(self.fusion),
            "deep_evaluation_count": len(self.deep),
            "final_target_count": len(self.final_targets),
            "calibration": self.calibration_label,
            "factor_baseline_mode": self.factor_baseline_mode,
            "portfolio_basis": self.portfolio_basis,
            "allocation_policy_sha256": self.allocation_policy_sha256,
            "overlay_stages": [dict(stage) for stage in self.overlay_stages],
            **authority_envelope(),
        }
        assert_public_envelope_safe(payload)
        return payload


def _wire(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineOrchestrationError("pipeline core contains a non-finite float")
        return value
    if is_dataclass(value):
        return _wire(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {
            str(key): _wire(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    raise PipelineOrchestrationError(
        f"pipeline core contains unsupported type: {type(value).__name__}"
    )


def _decimal_text(value: Any) -> str:
    """Render numeric algorithm output as canonical fixed-point text."""

    return format(value, "f") if isinstance(value, Decimal) else str(value)


def _algorithms() -> Any:
    try:
        from quant_investor.v17_v3_runtime import algorithms
    except ImportError as exc:
        raise PipelineOrchestrationError("V3 algorithm package is unavailable") from exc
    required = (
        "run_quant_preselection",
        "validate_branch_output",
        "fuse_branches",
        "evaluate_deep_research",
        "validate_monotonic_overlay",
    )
    missing = tuple(name for name in required if not callable(getattr(algorithms, name, None)))
    if missing:
        raise PipelineOrchestrationError("V3 algorithm exports are incomplete")
    return algorithms


def _role(admitted: AdmittedSources, *names: str, required: bool = True) -> Any:
    for name in names:
        if name in admitted.documents:
            document = admitted.materialize(name)
            if isinstance(document, Mapping) and "payload" in document:
                return document["payload"]
            return document
    if required:
        raise SourceAdmissionError("required admitted pipeline role is unavailable")
    return None


def _role_document(
    admitted: AdmittedSources,
    *names: str,
    required: bool = True,
) -> Mapping[str, Any] | None:
    for name in names:
        document = admitted.documents.get(name)
        if isinstance(document, Mapping):
            return document
    if required:
        raise SourceAdmissionError("required admitted pipeline artifact is unavailable")
    return None


def _run_id(admitted: AdmittedSources) -> str:
    for role in (
        "quant_preselection_inputs",
        "deep_research_inputs",
        "permissions",
    ):
        document = admitted.documents.get(role)
        if isinstance(document, Mapping) and type(document.get("run_id")) is str:
            return str(document["run_id"])
    return admitted.locator_id


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineOrchestrationError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise PipelineOrchestrationError(f"{label} must be an array")
    return tuple(value)


def _count_selected(value: Any) -> int:
    selected = getattr(value, "selected_symbols", ())
    if isinstance(selected, (list, tuple)):
        return len(selected)
    if isinstance(value, Mapping):
        raw = value.get("selected_symbols", ())
        return len(raw) if isinstance(raw, (list, tuple)) else 0
    return 0


def _pipeline_profile(
    admitted: AdmittedSources,
) -> tuple[str | None, str | None]:
    inputs = _role_document(
        admitted,
        "quant_preselection_inputs",
        required=False,
    )
    permissions = _role_document(
        admitted,
        "permissions",
        "pretrade_permissions",
        required=False,
    )
    factor_baseline_mode: str | None = None
    portfolio_basis: str | None = None
    if inputs is not None:
        factor_baseline_mode = inputs.get("factor_baseline_mode")
        if factor_baseline_mode is None and isinstance(inputs.get("payload"), Mapping):
            factor_baseline_mode = inputs["payload"].get("factor_baseline_mode")
        if factor_baseline_mode is not None and type(factor_baseline_mode) is not str:
            raise PipelineOrchestrationError("factor baseline mode is invalid")
    if permissions is not None:
        portfolio_basis = permissions.get("portfolio_basis")
        if portfolio_basis is not None and type(portfolio_basis) is not str:
            raise PipelineOrchestrationError("portfolio basis is invalid")
    return factor_baseline_mode, portfolio_basis


def _hard_stop(
    request: PipelineRequest,
    *,
    state: str,
    blocker: str,
    calibration_label: str = "UNAVAILABLE",
) -> PipelineResult:
    factor_baseline_mode, portfolio_basis = _pipeline_profile(request.admitted_sources)
    return PipelineResult(
        mode=request.mode,
        run_id=_run_id(request.admitted_sources),
        strategy_id=request.admitted_sources.strategy_id,
        cutoff=request.admitted_sources.cutoff,
        locator_byte_sha256=request.admitted_sources.locator_byte_sha256,
        calibration_label=calibration_label,
        factor_baseline_mode=factor_baseline_mode,
        portfolio_basis=portfolio_basis,
        allocation_policy_sha256=None,
        overlay_stages=(),
        terminal=PipelineTerminal(
            state=state,
            publishable=False,
            rank_complete=False,
            portfolio_complete=False,
            blockers=(blocker,),
        ),
        preselection=None,
        quant_branch=None,
        fundamental_branch=None,
        fusion=None,
        deep=(),
        base_targets={},
        overlay=None,
        final_targets={},
    )


def _promotion_weight(
    request: PipelineRequest,
    admitted: AdmittedSources,
) -> tuple[float, str]:
    receipt = _role(
        admitted,
        "fusion_promotion_receipt",
        "promotion_receipt",
        required=False,
    )
    if receipt is None:
        if request.mode == SHADOW_MODE:
            return 0.50, "UNCALIBRATED_50_50"
        raise PipelineOrchestrationError("formal research requires a promotion receipt")
    receipt_mapping = _mapping(receipt, label="fusion promotion receipt")
    state = receipt_mapping.get("state", receipt_mapping.get("status"))
    if state not in {"PROMOTED", "ACTIVE"}:
        if request.mode == SHADOW_MODE:
            return 0.50, "UNCalibrated 50/50"
        raise PipelineOrchestrationError("formal promotion receipt is not active")
    weight = receipt_mapping.get(
        "active_formal_research_weight",
        receipt_mapping.get("quant_weight"),
    )
    if isinstance(weight, bool):
        raise PipelineOrchestrationError("promotion weight must be numeric")
    try:
        normalized = float(weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineOrchestrationError("promotion weight must be numeric") from exc
    return normalized, "CALIBRATED_PROMOTED"


def _preselect(algorithms: Any, admitted: AdmittedSources) -> tuple[Any, Mapping[str, Any]]:
    wrapper = _role_document(
        admitted,
        "quant_preselection_inputs",
        "quant_preselection",
    )
    assert wrapper is not None
    inputs = _mapping(wrapper.get("payload"), label="quant preselection inputs")
    observations = inputs.get("observations")
    factor_contract = inputs.get("factor_contract")
    branch_inventory = inputs.get(
        "quant_branch_inventory",
        inputs.get("branch_inventory"),
    )
    if observations is None or factor_contract is None or branch_inventory is None:
        raise PipelineOrchestrationError("quant preselection inputs are incomplete")
    policy_sha = inputs.get("policy_sha256")
    try:
        require_sha256(policy_sha, label="preselection policy SHA-256")
    except IdentityContractError as exc:
        raise PipelineOrchestrationError(str(exc)) from exc
    normalized_observations = []
    for row in _sequence(observations, label="preselection observations"):
        mapping = _mapping(row, label="preselection observation")
        factor_values = mapping.get("factor_values")
        if not isinstance(factor_values, list):
            raise PipelineOrchestrationError("preselection factor values must be an array")
        normalized_observations.append(
            {
                **mapping,
                "factor_values": {
                    str(value["factor_id"]): value["value"]
                    for value in factor_values
                    if isinstance(value, Mapping)
                    and type(value.get("factor_id")) is str
                    and "value" in value
                },
            }
        )
    return (
        algorithms.run_quant_preselection(
            tuple(normalized_observations),
            factor_contract=_sequence(factor_contract, label="factor contract"),
            branch_inventory=_sequence(
                branch_inventory,
                label="Quant branch inventory",
            ),
            top_n=500,
        ),
        {
            **inputs,
            "factor_baseline_mode": wrapper.get("factor_baseline_mode"),
            "factor_baseline_ref": wrapper.get("factor_baseline_ref"),
        },
    )


def _current_source_locator_artifact(admitted: AdmittedSources) -> RuntimeArtifact:
    locator = _role_document(admitted, "source_locator")
    assert locator is not None
    raw = admitted.raw_objects.get("source_locator")
    if type(raw) is not bytes:
        raise PipelineOrchestrationError("admitted source locator bytes are unavailable")
    artifact = runtime_artifact(
        relative_path=admitted.locator_path,
        document=locator,
    )
    if artifact.raw != raw or artifact.byte_sha256 != admitted.locator_byte_sha256:
        raise PipelineOrchestrationError("admitted source locator identity drift")
    return artifact


def _source_locator_artifact(admitted: AdmittedSources) -> RuntimeArtifact:
    locator = _role_document(
        admitted,
        "preselection_source_locator",
        "source_locator",
    )
    assert locator is not None
    raw_role = (
        "preselection_source_locator"
        if "preselection_source_locator" in admitted.raw_objects
        else "source_locator"
    )
    raw = admitted.raw_objects.get(raw_role)
    if type(raw) is not bytes:
        raise PipelineOrchestrationError("preselection locator bytes are unavailable")
    current_locator = _role_document(admitted, "source_locator")
    lineage_ref = (
        None if current_locator is None else current_locator.get("preselection_locator_ref")
    )
    relative_path = (
        str(lineage_ref["relative_path"])
        if isinstance(lineage_ref, Mapping)
        else admitted.locator_path
    )
    return runtime_artifact(relative_path=relative_path, document=locator)


def _initial_pool_artifact(
    admitted: AdmittedSources,
    *,
    preselection: Any,
    preselection_inputs: Mapping[str, Any],
) -> RuntimeArtifact:
    wrapper = _role_document(
        admitted,
        "quant_preselection_inputs",
        "quant_preselection",
    )
    assert wrapper is not None
    run_id = wrapper.get("run_id")
    if type(run_id) is not str:
        raise PipelineOrchestrationError("preselection artifact has no run_id")
    locator = _source_locator_artifact(admitted)
    manifest = _role_document(
        admitted,
        "preselection_source_manifest",
        "source_manifest",
    )
    assert manifest is not None
    raw_manifest_ref = (
        manifest.get("parent_raw_manifest_ref")
        if manifest.get("closure_kind") == "DERIVED_CLOSURE"
        else locator.document.get("source_manifest_ref")
    )
    if not isinstance(raw_manifest_ref, Mapping):
        raise PipelineOrchestrationError("raw source manifest binding is unavailable")
    dispositions = [
        {
            "reasons": list(item.reasons),
            "score": None if item.score is None else _decimal_text(item.score),
            "selected": item.selected,
            "status": item.status,
            "symbol": item.symbol,
        }
        for item in getattr(preselection, "dispositions", ())
    ]
    coverage = getattr(preselection, "factor_coverage", {})
    if not isinstance(coverage, Mapping):
        raise PipelineOrchestrationError("preselection factor coverage is invalid")
    factor_baseline_mode = preselection_inputs.get("factor_baseline_mode")
    factor_baseline_ref = preselection_inputs.get("factor_baseline_ref")
    if factor_baseline_mode not in {
        "PROVISIONAL_RESEARCH",
        "FACTOR_V4_PRODUCTION",
    } or not isinstance(factor_baseline_ref, Mapping):
        raise PipelineOrchestrationError("preselection factor baseline binding is invalid")
    document = seal_typed_artifact(
        {
            "version": "myquant.v17.v3.initial-pool-output.v1",
            "protocol_version": PROTOCOL_VERSION,
            "output_id": f"initial-pool-{run_id}",
            "run_id": run_id,
            "strategy_id": admitted.strategy_id,
            "cutoff": admitted.cutoff,
            "created_at": admitted.cutoff,
            "state": "PRESELECT_COMPLETE",
            "status": getattr(preselection, "status"),
            "history_required": getattr(preselection, "history_required"),
            "ordered_domain": list(getattr(preselection, "ordered_domain", ())),
            "ready_domain": list(getattr(preselection, "ready_domain", ())),
            "selected_symbols": list(getattr(preselection, "selected_symbols", ())),
            "dispositions": dispositions,
            "factor_coverage": [
                {"factor_id": name, "coverage": _decimal_text(coverage[name])}
                for name in sorted(coverage)
            ],
            "pool_count": len(getattr(preselection, "selected_symbols", ())),
            "pool_symbol_order_sha256": hashlib.sha256(
                canonical_bytes(list(getattr(preselection, "selected_symbols", ())))
            ).hexdigest(),
            "policy_sha256": str(preselection_inputs["policy_sha256"]),
            "factor_baseline_mode": factor_baseline_mode,
            "factor_baseline_ref": dict(factor_baseline_ref),
            "source_locator_ref": locator.reference,
            "raw_source_manifest_ref": dict(raw_manifest_ref),
            "blockers": list(getattr(preselection, "blockers", ())),
            "authority": authority_envelope(),
        }
    )
    pool_reference = next(
        (
            reference.artifact_ref
            for reference in admitted.references
            if reference.role == "initial_pool_output"
        ),
        None,
    )
    relative_path = (
        str(pool_reference["relative_path"])
        if isinstance(pool_reference, Mapping)
        else str(PRIVATE_RUNS_ROOT / run_id / "initial_pool.json")
    )
    replay = runtime_artifact(
        relative_path=relative_path,
        document=document,
    )
    admitted_pool = admitted.documents.get("initial_pool_output")
    admitted_pool_raw = admitted.raw_objects.get("initial_pool_output")
    if admitted_pool is not None and (
        not isinstance(admitted_pool, Mapping)
        or type(admitted_pool_raw) is not bytes
        or replay.document != dict(admitted_pool)
        or replay.raw != admitted_pool_raw
        or not isinstance(pool_reference, Mapping)
        or replay.reference != dict(pool_reference)
    ):
        raise PipelineOrchestrationError("initial-pool deterministic replay drift")
    return replay


def build_initial_pool_artifact(admitted: AdmittedSources) -> RuntimeArtifact:
    """Replay PRESELECT inputs and return the immutable typed pool artifact."""

    manifest = _role_document(admitted, "source_manifest")
    locator = _role_document(admitted, "source_locator")
    if (
        manifest is None
        or locator is None
        or manifest.get("phase") not in {"PRESELECT", "SHADOW_CURRENT_PRESELECT"}
        or locator.get("preselection_locator_ref") is not None
    ):
        raise PipelineOrchestrationError(
            "initial-pool build requires an exact preselection locator"
        )
    algorithms = _algorithms()
    preselection, inputs = _preselect(algorithms, admitted)
    artifact = _initial_pool_artifact(
        admitted,
        preselection=preselection,
        preselection_inputs=inputs,
    )
    if artifact.document["status"] != "READY":
        raise PipelineOrchestrationError("preselection did not produce a READY initial pool")
    return artifact


def _algorithm_branch(
    document: Mapping[str, Any],
    *,
    internal_bindings: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "branch": document["branch"],
        "ordered_domain": document["ordered_domain"],
        "bindings": dict(internal_bindings),
        "records": document["records"],
    }


def _branch_and_fusion(
    algorithms: Any,
    admitted: AdmittedSources,
    *,
    preselection: Any,
    preselection_inputs: Mapping[str, Any],
    initial_pool: RuntimeArtifact,
    quant_weight: float,
) -> tuple[Any, Any, Any]:
    pool = tuple(getattr(preselection, "selected_symbols", ()))
    if not pool:
        raise PipelineOrchestrationError("preselection produced no organic pool")
    quant_raw = _role_document(
        admitted,
        "quant_branch_output",
        "quant_branch",
    )
    fundamental_raw = _role_document(
        admitted,
        "fundamental_branch_output",
        "fundamental_branch",
    )
    assert quant_raw is not None and fundamental_raw is not None
    locator = _source_locator_artifact(admitted)
    strong_bindings = {
        "source_locator_ref": locator.reference,
        "initial_pool_ref": initial_pool.reference,
        "initial_pool_count": len(pool),
        "initial_pool_symbol_order_sha256": initial_pool.document["pool_symbol_order_sha256"],
    }
    quant_policy_sha = str(
        load_packaged_json("resources/quant_branch_policy.v1.json")["semantic_sha256"]
    )
    fundamental_policy_sha = str(
        load_packaged_json("resources/fundamental_branch_policy.v1.json")["semantic_sha256"]
    )
    validate_branch_same_pool_binding(
        quant_raw,
        expected_bindings={
            **strong_bindings,
            "policy_sha256": quant_policy_sha,
        },
    )
    validate_branch_same_pool_binding(
        fundamental_raw,
        expected_bindings={
            **strong_bindings,
            "policy_sha256": fundamental_policy_sha,
        },
    )
    internal_bindings = {
        "source_locator_path": admitted.locator_path,
        "source_locator_byte_sha256": admitted.locator_byte_sha256,
        "cutoff": admitted.cutoff,
        "pool_byte_sha256": initial_pool.byte_sha256,
        "pool_semantic_sha256": str(initial_pool.document["semantic_sha256"]),
        "pool_count": str(len(pool)),
        "pool_symbol_order_sha256": hashlib.sha256(canonical_bytes(list(pool))).hexdigest(),
    }
    quant = algorithms.validate_branch_output(
        _algorithm_branch(quant_raw, internal_bindings=internal_bindings),
        ordered_pool=pool,
        expected_bindings=internal_bindings,
    )
    fundamental = algorithms.validate_branch_output(
        _algorithm_branch(
            fundamental_raw,
            internal_bindings=internal_bindings,
        ),
        ordered_pool=pool,
        expected_bindings=internal_bindings,
    )
    fusion = algorithms.fuse_branches(
        quant,
        fundamental,
        ordered_pool=pool,
        quant_weight=quant_weight,
        quant_bindings=quant.bindings,
        fundamental_bindings=fundamental.bindings,
        top_n=24,
    )
    return quant, fundamental, fusion


def _fusion_artifact(
    admitted: AdmittedSources,
    *,
    fusion: Any,
) -> RuntimeArtifact:
    wrapper = _role_document(admitted, "quant_preselection_inputs")
    promotion = _role_document(
        admitted,
        "fusion_promotion_receipt",
        required=False,
    )
    if wrapper is None:
        raise PipelineOrchestrationError("typed fusion artifact has no run binding")
    run_id = wrapper.get("run_id")
    promoted = promotion is not None and promotion.get("status") == "PROMOTED"
    calibration_refs = (
        promotion.get("calibration_receipt_refs", ()) if promoted and promotion is not None else ()
    )
    if type(run_id) is not str or not isinstance(
        calibration_refs,
        (list, tuple),
    ):
        raise PipelineOrchestrationError("fusion artifact bindings are incomplete")
    promotion_ref = (
        dict(admitted.reference_for_role("fusion_promotion_receipt")) if promoted else None
    )
    dispositions = [
        {
            "fundamental_percentile": (
                None
                if item.fundamental_percentile is None
                else _decimal_text(item.fundamental_percentile)
            ),
            "fusion_score": (
                None if item.fusion_score is None else _decimal_text(item.fusion_score)
            ),
            "quant_percentile": (
                None if item.quant_percentile is None else _decimal_text(item.quant_percentile)
            ),
            "reason": item.reason,
            "selected": item.selected,
            "status": item.status,
            "symbol": item.symbol,
        }
        for item in getattr(fusion, "dispositions", ())
    ]
    document = seal_typed_artifact(
        {
            "version": "myquant.v17.v3.fusion-output.v1",
            "protocol_version": PROTOCOL_VERSION,
            "output_id": f"fusion-{run_id}",
            "run_id": run_id,
            "strategy_id": admitted.strategy_id,
            "cutoff": admitted.cutoff,
            "created_at": admitted.cutoff,
            "state": "FUSION_COMPLETE",
            "status": getattr(fusion, "status"),
            "quant_branch_ref": dict(admitted.reference_for_role("quant_branch_output")),
            "fundamental_branch_ref": dict(
                admitted.reference_for_role("fundamental_branch_output")
            ),
            "calibration_receipt_refs": list(calibration_refs),
            "promotion_receipt_ref": promotion_ref,
            "calibration_label": ("CALIBRATED_PROMOTED" if promoted else "UNCALIBRATED_50_50"),
            "quant_weight": _decimal_text(getattr(fusion, "quant_weight")),
            "fundamental_weight": _decimal_text(getattr(fusion, "fundamental_weight")),
            "ordered_domain": list(getattr(fusion, "ordered_domain", ())),
            "common_ready_domain": list(getattr(fusion, "common_ready_domain", ())),
            "selected_symbols": list(getattr(fusion, "selected_symbols", ())),
            "dispositions": dispositions,
            "blockers": list(getattr(fusion, "blockers", ())),
            "authority": authority_envelope(),
        }
    )
    return runtime_artifact(
        relative_path=PRIVATE_RUNS_ROOT / run_id / "fusion_output.json",
        document=document,
    )


def _deep_and_baseline(
    algorithms: Any,
    admitted: AdmittedSources,
    *,
    fusion: Any,
) -> tuple[
    tuple[tuple[str, Any], ...],
    dict[str, Any],
    tuple[str, ...],
    str,
]:
    selected = tuple(getattr(fusion, "selected_symbols", ()))
    policy = load_packaged_json("resources/portfolio_allocation_policy.v1.json")
    if (
        policy.get("allocation_method") != "EQUAL_WEIGHT"
        or policy.get("gross_weight") != "0.72"
        or policy.get("max_weight_per_symbol") != "0.03"
        or policy.get("precision_decimal_places") != 8
        or policy.get("rounding_mode") != "ROUND_DOWN"
        or policy.get("residual_cash_policy") != "KEEP_AS_CASH_NO_REDISTRIBUTION"
    ):
        raise PipelineOrchestrationError("portfolio allocation policy is invalid")
    if not selected:
        raise PipelineOrchestrationError("portfolio allocation domain is empty")
    selected_set = frozenset(selected)
    with localcontext() as context:
        context.prec = 34
        equal_weight = (Decimal(str(policy["gross_weight"])) / Decimal(len(selected))).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
    expected_base_target = min(
        Decimal(str(policy["max_weight_per_symbol"])),
        equal_weight,
    )
    allocation_policy_sha256 = str(policy["semantic_sha256"])
    inputs = _role(admitted, "deep_research", "deep_research_inputs", required=False)
    by_symbol: dict[str, Mapping[str, Any]] = {}
    if inputs is not None:
        for row in _sequence(inputs, label="deep research inputs"):
            mapping = _mapping(row, label="deep research row")
            symbol = mapping.get("symbol")
            if type(symbol) is not str or symbol in by_symbol:
                raise PipelineOrchestrationError("deep research symbol identity is invalid")
            by_symbol[symbol] = mapping
    try:
        deep_evidence_ref = admitted.reference_for_role("deep_evidence")
    except SourceAdmissionError:
        deep_evidence_ref = None
    review_only = tuple(
        sorted(
            symbol
            for symbol, row in by_symbol.items()
            if row.get("lane") == "REVIEW_ONLY_HOLDING"
            and row.get("held") is True
            and symbol not in selected_set
        )
    )
    if any(
        row.get("lane") == "REVIEW_ONLY_HOLDING" and symbol in selected_set
        for symbol, row in by_symbol.items()
    ):
        raise PipelineOrchestrationError("review-only holding entered the fusion Top24")
    selection_rows = {
        symbol for symbol, row in by_symbol.items() if row.get("lane") == "SELECTION_POOL"
    }
    missing_selection_rows = selected_set.difference(selection_rows)
    if missing_selection_rows:
        return (
            (),
            {},
            ("deep_top24_row_missing",),
            allocation_policy_sha256,
        )
    if selection_rows != selected_set:
        raise PipelineOrchestrationError(
            "deep selection-pool rows do not exactly match fusion Top24"
        )
    decisions: list[tuple[str, Any]] = []
    baseline: dict[str, Any] = {}
    blockers: list[str] = []
    for symbol in (*selected, *review_only):
        row = by_symbol.get(symbol)
        assert row is not None
        if symbol in selected_set:
            try:
                declared_base = Decimal(str(row.get("base_target")))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise PipelineOrchestrationError(
                    "deep base target is not decimal-compatible"
                ) from exc
            if declared_base != expected_base_target:
                return (
                    (),
                    {},
                    ("base_target_allocation_policy_mismatch",),
                    allocation_policy_sha256,
                )
        if row.get("available") is True:
            evidence_refs = row.get("evidence_refs")
            if (
                deep_evidence_ref is None
                or not isinstance(evidence_refs, list)
                or not evidence_refs
                or any(
                    not isinstance(reference, Mapping) or dict(reference) != dict(deep_evidence_ref)
                    for reference in evidence_refs
                )
            ):
                raise PipelineOrchestrationError(
                    "available deep research lacks exact admitted evidence"
                )
        decision = algorithms.evaluate_deep_research(
            held=row.get("held"),
            current_target=row.get("current_target"),
            base_target=row.get("base_target"),
            available=row.get("available"),
            signal=row.get("signal"),
            veto_buy=row.get("veto_buy", False),
        )
        decisions.append((symbol, decision))
        baseline[symbol] = decision.target
        if (
            row.get("available") is False
            and row.get("held") is False
            and (
                decision.status != "BUY_VETO"
                or decision.buy_veto is not True
                or decision.target != Decimal("0")
            )
        ):
            raise PipelineOrchestrationError(
                "unavailable unheld deep row violates BUY_VETO truth table"
            )
        if symbol in review_only and decision.target > decision.current_target:
            raise PipelineOrchestrationError("review-only deep target received a positive delta")
    return (
        tuple(decisions),
        baseline,
        tuple(dict.fromkeys(blockers)),
        allocation_policy_sha256,
    )


def _deep_artifact(
    admitted: AdmittedSources,
    *,
    fusion_artifact: RuntimeArtifact,
    decisions: tuple[tuple[str, Any], ...],
) -> RuntimeArtifact:
    wrapper = _role_document(admitted, "deep_research_inputs")
    if wrapper is None or type(wrapper.get("run_id")) is not str:
        raise PipelineOrchestrationError("deep input artifact binding is incomplete")
    rows = wrapper.get("payload")
    if not isinstance(rows, list):
        raise PipelineOrchestrationError("deep input payload is invalid")
    by_symbol = {
        row["symbol"]: row
        for row in rows
        if isinstance(row, Mapping) and type(row.get("symbol")) is str
    }
    results: list[dict[str, Any]] = []
    for symbol, decision in decisions:
        if symbol not in by_symbol:
            raise PipelineOrchestrationError("deep decision identity is unavailable")
        row = by_symbol[symbol]
        signal = row.get("signal")
        results.append(
            {
                "symbol": symbol,
                "lane": str(row.get("lane")),
                "held": decision.held,
                "available": decision.available,
                "buy_veto": decision.buy_veto,
                "locked": decision.locked,
                "signal": (None if signal is None else _decimal_text(decision.signal)),
                "penalty": _decimal_text(decision.penalty),
                "base_target": _decimal_text(decision.base_target),
                "raw_adjusted_target": _decimal_text(decision.raw_adjusted_target),
                "target": _decimal_text(decision.target),
                "current_target": _decimal_text(decision.current_target),
                "status": decision.status,
                "blockers": list(decision.blockers),
                "evidence_refs": list(row.get("evidence_refs", ())),
            }
        )
    policy = load_packaged_json("resources/deep_policy.v1.json")
    document = seal_typed_artifact(
        {
            "version": "myquant.v17.v3.deep-output.v1",
            "protocol_version": PROTOCOL_VERSION,
            "output_id": f"deep-{wrapper['run_id']}",
            "run_id": wrapper["run_id"],
            "strategy_id": admitted.strategy_id,
            "cutoff": admitted.cutoff,
            "created_at": admitted.cutoff,
            "state": "DEEP_COMPLETE",
            "fusion_output_ref": fusion_artifact.reference,
            "policy_sha256": policy["semantic_sha256"],
            "results": sorted(results, key=lambda row: row["symbol"]),
            "authority": authority_envelope(),
        }
    )
    return runtime_artifact(
        relative_path=PRIVATE_RUNS_ROOT / str(wrapper["run_id"]) / "deep_output.json",
        document=document,
    )


def _apply_permissions(
    admitted: AdmittedSources,
    targets: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    raw = _role(admitted, "permissions", "pretrade_permissions", required=False)
    if raw is None:
        return {}, ("permissions_unavailable",)
    permissions: Mapping[str, Any]
    if isinstance(raw, list):
        permissions = {
            row["symbol"]: row
            for row in raw
            if isinstance(row, Mapping) and type(row.get("symbol")) is str
        }
    else:
        permissions = _mapping(raw, label="permissions")
    permissions_document = _role_document(
        admitted,
        "permissions",
        "pretrade_permissions",
    )
    assert permissions_document is not None
    portfolio_basis = permissions_document.get("portfolio_basis")
    if portfolio_basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
        invalid_rows = [
            row
            for row in permissions.values()
            if (
                not isinstance(row, Mapping)
                or row.get("lane") != "SELECTION_POOL"
                or row.get("held") is not False
                or Decimal(str(row.get("current_target", "0"))) != Decimal("0")
            )
        ]
        if invalid_rows:
            raise PipelineOrchestrationError(
                "model-only permissions contain private-holdings state"
            )
        if set(permissions) != set(targets):
            raise PipelineOrchestrationError("model-only permissions do not exactly cover Top24")
        return (
            {
                symbol: (target if permissions[symbol].get("can_buy") is True else Decimal("0"))
                for symbol, target in targets.items()
            },
            (),
        )
    if portfolio_basis != "HOLDINGS_AWARE":
        raise PipelineOrchestrationError("portfolio basis is invalid")
    allowed: dict[str, Any] = {}
    blockers: list[str] = []
    for symbol, target in targets.items():
        permission = permissions.get(symbol)
        if not isinstance(permission, Mapping):
            blockers.append("permission_missing")
            continue
        held = permission.get("held") is True
        can_buy = permission.get("can_buy") is True
        if permission.get("lane") == "REVIEW_ONLY_HOLDING":
            current = Decimal(str(permission.get("current_target", "0")))
            if Decimal(str(target)) > current:
                blockers.append("review_only_positive_delta_forbidden")
                continue
        if not held and not can_buy and Decimal(str(target)) > 0:
            blockers.append("positive_target_not_permitted")
            continue
        allowed[symbol] = target
    return allowed, tuple(dict.fromkeys(blockers))


def _portfolio_artifact(
    admitted: AdmittedSources,
    *,
    fusion: Any,
    fusion_artifact: RuntimeArtifact,
    deep_artifact: RuntimeArtifact,
    final_targets: Mapping[str, Any],
    blockers: tuple[str, ...],
    allocation_policy_sha256: str,
    overlay_stages: tuple[Mapping[str, Any], ...],
) -> tuple[RuntimeArtifact, dict[str, Any]]:
    permissions = _role_document(admitted, "permissions", "pretrade_permissions")
    if permissions is None or type(permissions.get("run_id")) is not str:
        raise PipelineOrchestrationError("portfolio permissions binding is incomplete")
    rows = permissions.get("payload")
    holdings_ref = permissions.get("holdings_snapshot_ref")
    portfolio_basis = permissions.get("portfolio_basis")
    if not isinstance(rows, list) or portfolio_basis not in {
        "MODEL_ONLY_NO_PRIVATE_HOLDINGS",
        "HOLDINGS_AWARE",
    }:
        raise PipelineOrchestrationError("portfolio permissions payload is invalid")
    try:
        admitted_calendar_ref = admitted.reference_for_role("cn_open_day_calendar")
    except SourceAdmissionError as exc:
        raise PipelineOrchestrationError("portfolio calendar exact binding is unavailable") from exc
    if dict(admitted_calendar_ref) != dict(permissions.get("canonical_calendar_ref", {})):
        raise PipelineOrchestrationError("calendar exact binding drift")
    sessions = _calendar_sessions(admitted.materialize("cn_open_day_calendar"))
    decision = permissions.get("decision_session")
    if type(decision) is not str or decision not in sessions:
        raise PipelineOrchestrationError("portfolio decision session cannot be established")
    if portfolio_basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
        if (
            holdings_ref is not None
            or permissions.get("holdings_snapshot_as_of_session") is not None
            or permissions.get("holdings_snapshot_age_sessions") is not None
            or "holdings_snapshot" in admitted.documents
        ):
            raise PipelineOrchestrationError("model-only portfolio carries private holdings")
    else:
        if not isinstance(holdings_ref, Mapping):
            raise PipelineOrchestrationError("holdings-aware portfolio has no snapshot reference")
        try:
            admitted_holdings_ref = admitted.reference_for_role("holdings_snapshot")
        except SourceAdmissionError as exc:
            raise PipelineOrchestrationError(
                "holdings snapshot exact binding is unavailable"
            ) from exc
        as_of = permissions.get("holdings_snapshot_as_of_session")
        if dict(admitted_holdings_ref) != dict(holdings_ref):
            raise PipelineOrchestrationError("holdings snapshot exact binding drift")
        if type(as_of) is not str or as_of not in sessions:
            raise PipelineOrchestrationError("portfolio session freshness cannot be established")
        age = sessions.index(decision) - sessions.index(as_of)
        if age < 0 or age > 1 or age != permissions.get("holdings_snapshot_age_sessions"):
            raise PipelineOrchestrationError("holdings snapshot session age mismatch")
        holdings = admitted.materialize("holdings_snapshot")
        if not isinstance(holdings, Mapping) or any(
            holdings.get(key) != expected
            for key, expected in (
                ("role", "holdings_snapshot"),
                ("strategy_id", admitted.strategy_id),
                ("as_of_session", as_of),
            )
        ):
            raise PipelineOrchestrationError("holdings snapshot typed scope mismatch")
        available_at = holdings.get("available_at")
        if type(available_at) is not str or available_at > admitted.cutoff:
            raise PipelineOrchestrationError("holdings snapshot availability exceeds cutoff")
    permission_by_symbol = {
        row["symbol"]: row
        for row in rows
        if isinstance(row, Mapping) and type(row.get("symbol")) is str
    }
    targets = dict(final_targets)
    selected = tuple(getattr(fusion, "selected_symbols", ()))
    selected_set = frozenset(selected)
    review_only: list[str] = []
    for symbol, row in permission_by_symbol.items():
        if row.get("lane") != "REVIEW_ONLY_HOLDING":
            continue
        if symbol in selected_set:
            raise PipelineOrchestrationError("review-only holding entered the fusion Top24")
        current = Decimal(str(row.get("current_target", "0")))
        proposed = Decimal(str(targets.get(symbol, current)))
        if proposed > current:
            raise PipelineOrchestrationError("review-only holding received a positive target delta")
        targets[symbol] = proposed
        review_only.append(symbol)
    if portfolio_basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS" and (
        review_only
        or set(permission_by_symbol) != selected_set
        or any(
            row.get("lane") != "SELECTION_POOL"
            or row.get("held") is not False
            or Decimal(str(row.get("current_target", "0"))) != Decimal("0")
            for row in permission_by_symbol.values()
        )
    ):
        raise PipelineOrchestrationError("model-only portfolio permission truth table is invalid")
    target_rows: list[dict[str, str]] = []
    gross = Decimal("0")
    for symbol in sorted(targets):
        row = permission_by_symbol.get(symbol)
        if row is None:
            raise PipelineOrchestrationError("portfolio target has no permission")
        current = Decimal(str(row.get("current_target", "0")))
        final = Decimal(str(targets[symbol]))
        gross += final
        target_rows.append(
            {
                "symbol": symbol,
                "lane": str(row.get("lane")),
                "current_target": _decimal_text(current),
                "final_target": _decimal_text(final),
            }
        )
    if gross > Decimal("1"):
        blockers = (*blockers, "gross_weight_exceeds_one")
    cash = max(Decimal("0"), Decimal("1") - gross)
    status = "COMPLETE" if target_rows and not blockers else "INFEASIBLE"
    if status != "COMPLETE":
        targets = {}
        target_rows = []
        selected = ()
        review_only = []
        gross = Decimal("0")
        cash = Decimal("1")
    document = seal_typed_artifact(
        {
            "version": "myquant.v17.v3.portfolio-output.v1",
            "protocol_version": PROTOCOL_VERSION,
            "output_id": f"portfolio-{permissions['run_id']}",
            "run_id": permissions["run_id"],
            "strategy_id": admitted.strategy_id,
            "cutoff": admitted.cutoff,
            "created_at": admitted.cutoff,
            "status": status,
            "factor_baseline_mode": _pipeline_profile(admitted)[0],
            "factor_baseline_ref": dict(
                _role_document(
                    admitted,
                    "quant_preselection_inputs",
                )["factor_baseline_ref"]
            ),
            "portfolio_basis": portfolio_basis,
            "allocation_policy_sha256": allocation_policy_sha256,
            "overlay_stages": [dict(stage) for stage in overlay_stages],
            "selection_pool_symbols": list(selected),
            "review_only_holdings": sorted(review_only),
            "targets": target_rows,
            "gross_weight": _decimal_text(gross),
            "cash_weight": _decimal_text(cash),
            "blockers": list(dict.fromkeys(blockers)),
            "fusion_output_ref": fusion_artifact.reference,
            "deep_output_ref": deep_artifact.reference,
            "holdings_snapshot_ref": (None if holdings_ref is None else dict(holdings_ref)),
            "permissions_ref": dict(admitted.reference_for_role("permissions")),
            "authority": authority_envelope(),
        }
    )
    artifact = runtime_artifact(
        relative_path=(PRIVATE_RUNS_ROOT / str(permissions["run_id"]) / "portfolio_output.json"),
        document=document,
    )
    return artifact, targets


def _calendar_sessions(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        rows: Any = value.get("sessions")
        if rows is None and isinstance(value.get("payload"), Mapping):
            rows = value["payload"].get("sessions")
        if isinstance(rows, list) and all(type(item) is str for item in rows):
            sessions = list(rows)
        else:
            raise PipelineOrchestrationError("canonical calendar JSON has no sessions")
    elif type(value) is bytes:
        try:
            import pyarrow.parquet as parquet

            table = parquet.read_table(BytesIO(value))
            name = next(
                candidate
                for candidate in ("trade_date", "session", "date")
                if candidate in table.column_names
            )
            sessions = [
                item.isoformat() if hasattr(item, "isoformat") else str(item)
                for item in table[name].to_pylist()
            ]
        except (ImportError, OSError, RuntimeError, StopIteration, ValueError) as exc:
            raise PipelineOrchestrationError("canonical calendar parquet is unreadable") from exc
    else:
        raise PipelineOrchestrationError("canonical calendar is unavailable")
    if sessions != sorted(set(sessions)):
        raise PipelineOrchestrationError("canonical calendar sessions are not strictly ordered")
    return sessions


def _overlay(
    algorithms: Any,
    admitted: AdmittedSources,
    *,
    baseline: Mapping[str, Any],
) -> tuple[
    Any,
    dict[str, Any],
    tuple[str, ...],
    tuple[Mapping[str, Any], ...],
]:
    current = dict(baseline)
    validations: list[Any] = []
    stages: list[Mapping[str, Any]] = []
    for stage, role in (
        ("MACRO", "macro_overlay"),
        ("MARKOV", "markov_overlay"),
    ):
        raw = _role(admitted, role, required=False)
        if raw is None:
            stages.append(
                {
                    "stage": stage,
                    "status": "UNAVAILABLE_NO_OP",
                    "overlay_ref": None,
                }
            )
            continue
        mapping = _mapping(raw, label=role)
        proposed = mapping.get("target_weights", mapping.get("targets"))
        if isinstance(proposed, list):
            proposed = {
                row["symbol"]: row["target"]
                for row in proposed
                if isinstance(row, Mapping) and type(row.get("symbol")) is str and "target" in row
            }
        validation = algorithms.validate_monotonic_overlay(
            baseline_targets=current,
            post_targets=_mapping(proposed, label=f"{role}.targets"),
        )
        validations.append(validation)
        if not validation.valid:
            raise PipelineOrchestrationError("portfolio overlay violates monotonicity")
        stages.append(
            {
                "stage": stage,
                "status": "APPLIED",
                "overlay_ref": dict(admitted.reference_for_role(role)),
            }
        )
        current = dict(validation.post_targets)
    return tuple(validations), current, (), tuple(stages)


def _terminal_artifact(
    request: PipelineRequest,
    *,
    run_id: str,
    state: str,
    portfolio_status: str,
    artifacts: tuple[RuntimeArtifact, ...],
    source_refs: tuple[Mapping[str, Any], ...],
    factor_baseline_mode: str,
    factor_baseline_ref: Mapping[str, Any],
    portfolio_basis: str | None,
    allocation_policy_sha256: str,
    overlay_stages: tuple[Mapping[str, Any], ...],
) -> RuntimeArtifact:
    is_shadow = request.mode == SHADOW_MODE
    version = (
        "myquant.v17.v3.shadow-output.v1"
        if is_shadow
        else "myquant.v17.v3.formal-research-output.v1"
    )
    prefix = "shadow" if is_shadow else "formal"
    root = (
        shadow_run_path(
            strategy_id=request.admitted_sources.strategy_id,
            run_id=run_id,
        )
        if is_shadow
        else formal_run_path(
            strategy_id=request.admitted_sources.strategy_id,
            run_id=run_id,
        )
    )
    refs = [artifact.reference for artifact in artifacts]
    refs.extend(dict(reference) for reference in source_refs)
    refs.sort(key=lambda ref: (ref["relative_path"], ref["byte_sha256"]))
    terminal_payload: dict[str, Any] = {
        "version": version,
        "protocol_version": PROTOCOL_VERSION,
        "output_id": f"{prefix}-{run_id}",
        "run_id": run_id,
        "strategy_id": request.admitted_sources.strategy_id,
        "cutoff": request.admitted_sources.cutoff,
        "created_at": request.admitted_sources.cutoff,
        "terminal_state": state,
        "portfolio_status": portfolio_status,
        "factor_baseline_mode": factor_baseline_mode,
        "factor_baseline_ref": dict(factor_baseline_ref),
        "portfolio_basis": portfolio_basis,
        "analyze_locator_ref": _current_source_locator_artifact(request.admitted_sources).reference,
        "portfolio_output_ref": (
            artifacts[-1].reference
            if portfolio_status in {"COMPLETE", "INFEASIBLE"} and artifacts
            else None
        ),
        "artifact_refs": refs,
        "authority": authority_envelope(),
    }
    if is_shadow:
        terminal_payload["allocation_policy_sha256"] = allocation_policy_sha256
        terminal_payload["overlay_stages"] = [dict(stage) for stage in overlay_stages]
    document = seal_typed_artifact(terminal_payload)
    filename = "shadow_output.json" if is_shadow else "formal_output.json"
    return runtime_artifact(
        relative_path=root / filename,
        document=document,
    )


def run_pipeline(request: PipelineRequest) -> PipelineResult:
    """Run the fixed pipeline with no free-form input or execution surface."""

    if not isinstance(request, PipelineRequest):
        raise TypeError("request must be PipelineRequest")
    if request.mode not in MODES:
        raise PipelineOrchestrationError("analysis mode is invalid")
    if not isinstance(request.admitted_sources, AdmittedSources):
        raise PipelineOrchestrationError("analysis requires admitted sources")
    manifest = _role_document(request.admitted_sources, "source_manifest")
    assert manifest is not None
    phase = manifest.get("phase")
    factor_baseline_mode, portfolio_basis = _pipeline_profile(request.admitted_sources)
    if type(phase) is str and phase.startswith("SHADOW_CURRENT_") and request.mode != SHADOW_MODE:
        return _hard_stop(
            request,
            state="HARD_STOP_INVALID_EVIDENCE",
            blocker="shadow_current_phase_requires_shadow_mode",
        )
    if request.mode == FORMAL_RESEARCH_MODE and factor_baseline_mode == "PROVISIONAL_RESEARCH":
        return _hard_stop(
            request,
            state="HARD_STOP_INVALID_EVIDENCE",
            blocker="formal_rejects_provisional_factor_baseline",
        )
    if request.mode == FORMAL_RESEARCH_MODE and portfolio_basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
        return _hard_stop(
            request,
            state="HARD_STOP_INVALID_EVIDENCE",
            blocker="formal_rejects_model_only_portfolio",
        )
    try:
        algorithms = _algorithms()
        quant_weight, calibration_label = _promotion_weight(
            request,
            request.admitted_sources,
        )
        preselection, preselection_inputs = _preselect(
            algorithms,
            request.admitted_sources,
        )
        if getattr(preselection, "status", None) != "READY":
            return _hard_stop(
                request,
                state="HARD_STOP_INVALID_EVIDENCE",
                blocker="quant_preselection_unavailable",
                calibration_label=calibration_label,
            )
        initial_pool = _initial_pool_artifact(
            request.admitted_sources,
            preselection=preselection,
            preselection_inputs=preselection_inputs,
        )
        quant, fundamental, fusion = _branch_and_fusion(
            algorithms,
            request.admitted_sources,
            preselection=preselection,
            preselection_inputs=preselection_inputs,
            initial_pool=initial_pool,
            quant_weight=quant_weight,
        )
        if getattr(fusion, "status", None) != "READY" or not tuple(
            getattr(fusion, "selected_symbols", ())
        ):
            return _hard_stop(
                request,
                state="HARD_STOP_INVALID_EVIDENCE",
                blocker="fusion_top24_incomplete",
                calibration_label=calibration_label,
            )
        (
            deep,
            baseline,
            deep_blockers,
            allocation_policy_sha256,
        ) = _deep_and_baseline(
            algorithms,
            request.admitted_sources,
            fusion=fusion,
        )
        if deep_blockers:
            return _hard_stop(
                request,
                state="HARD_STOP_INVALID_EVIDENCE",
                blocker=deep_blockers[0],
                calibration_label=calibration_label,
            )
        fusion_artifact = _fusion_artifact(
            request.admitted_sources,
            fusion=fusion,
        )
        deep_artifact = _deep_artifact(
            request.admitted_sources,
            fusion_artifact=fusion_artifact,
            decisions=deep,
        )
        has_permissions = "permissions" in request.admitted_sources.documents
        has_holdings = "holdings_snapshot" in request.admitted_sources.documents
        portfolio_artifact: RuntimeArtifact | None
        overlay_stages: tuple[Mapping[str, Any], ...] = (
            {
                "stage": "MACRO",
                "status": "UNAVAILABLE_NO_OP",
                "overlay_ref": None,
            },
            {
                "stage": "MARKOV",
                "status": "UNAVAILABLE_NO_OP",
                "overlay_ref": None,
            },
        )
        if not has_permissions and not has_holdings:
            permitted_baseline = {}
            overlay = None
            final_targets = {}
            blockers = tuple(dict.fromkeys((*deep_blockers, "portfolio_inputs_unavailable")))
            portfolio_artifact = None
            portfolio_complete = False
        else:
            if not has_permissions:
                raise PipelineOrchestrationError("holdings without permissions are forbidden")
            if portfolio_basis == "HOLDINGS_AWARE" and not has_holdings:
                raise PipelineOrchestrationError("holdings-aware portfolio requires exact holdings")
            if portfolio_basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS" and has_holdings:
                raise PipelineOrchestrationError("model-only portfolio forbids private holdings")
            permitted_baseline, permission_blockers = _apply_permissions(
                request.admitted_sources,
                baseline,
            )
            (
                overlay,
                overlaid,
                overlay_blockers,
                overlay_stages,
            ) = _overlay(
                algorithms,
                request.admitted_sources,
                baseline=permitted_baseline,
            )
            final_targets, final_permission_blockers = _apply_permissions(
                request.admitted_sources,
                overlaid,
            )
            blockers = tuple(
                dict.fromkeys(
                    (
                        *deep_blockers,
                        *permission_blockers,
                        *overlay_blockers,
                        *final_permission_blockers,
                    )
                )
            )
            portfolio_artifact, final_targets = _portfolio_artifact(
                request.admitted_sources,
                fusion=fusion,
                fusion_artifact=fusion_artifact,
                deep_artifact=deep_artifact,
                final_targets=final_targets,
                blockers=blockers,
                allocation_policy_sha256=allocation_policy_sha256,
                overlay_stages=overlay_stages,
            )
            portfolio_complete = portfolio_artifact.document["status"] == "COMPLETE"
        portfolio_status = (
            "NOT_REQUESTED"
            if portfolio_artifact is None
            else str(portfolio_artifact.document["status"])
        )
        if request.mode == SHADOW_MODE:
            terminal_state = {
                "COMPLETE": "SHADOW_COMPLETE",
                "INFEASIBLE": "SHADOW_PORTFOLIO_INFEASIBLE",
                "NOT_REQUESTED": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            }[portfolio_status]
            publishable = False
        else:
            terminal_state = {
                "COMPLETE": "FORMAL_RESEARCH_COMPLETE",
                "INFEASIBLE": "FORMAL_PORTFOLIO_INFEASIBLE",
                "NOT_REQUESTED": "FORMAL_RANK_COMPLETE_NO_PORTFOLIO",
            }[portfolio_status]
            publishable = False
        run_id = _run_id(request.admitted_sources)
        stage_artifacts = (
            initial_pool,
            fusion_artifact,
            deep_artifact,
            *((portfolio_artifact,) if portfolio_artifact is not None else ()),
        )
        source_ref_list: list[Mapping[str, Any]] = [
            _current_source_locator_artifact(request.admitted_sources).reference,
            request.admitted_sources.reference_for_role("quant_branch_output"),
            request.admitted_sources.reference_for_role("fundamental_branch_output"),
        ]
        for optional_role in (
            "factor_governance_readiness",
            "provisional_factor_baseline",
            "fusion_calibration",
        ):
            try:
                source_ref_list.append(request.admitted_sources.reference_for_role(optional_role))
            except SourceAdmissionError:
                pass
        source_refs = tuple(source_ref_list)
        terminal_artifact = _terminal_artifact(
            request,
            run_id=run_id,
            state=terminal_state,
            portfolio_status=portfolio_status,
            artifacts=stage_artifacts,
            source_refs=source_refs,
            factor_baseline_mode=str(factor_baseline_mode),
            factor_baseline_ref=_mapping(
                preselection_inputs.get("factor_baseline_ref"),
                label="factor baseline reference",
            ),
            portfolio_basis=portfolio_basis,
            allocation_policy_sha256=allocation_policy_sha256,
            overlay_stages=overlay_stages,
        )
        return PipelineResult(
            mode=request.mode,
            run_id=run_id,
            strategy_id=request.admitted_sources.strategy_id,
            cutoff=request.admitted_sources.cutoff,
            locator_byte_sha256=request.admitted_sources.locator_byte_sha256,
            calibration_label=calibration_label,
            factor_baseline_mode=factor_baseline_mode,
            portfolio_basis=portfolio_basis,
            allocation_policy_sha256=allocation_policy_sha256,
            overlay_stages=overlay_stages,
            terminal=PipelineTerminal(
                state=terminal_state,
                publishable=publishable,
                rank_complete=True,
                portfolio_complete=portfolio_complete,
                blockers=blockers,
            ),
            preselection=preselection,
            quant_branch=quant,
            fundamental_branch=fundamental,
            fusion=fusion,
            deep=deep,
            base_targets=permitted_baseline,
            overlay=overlay,
            final_targets=final_targets,
            artifacts=(*stage_artifacts, terminal_artifact),
            terminal_artifact=terminal_artifact,
        )
    except (SourceAdmissionError, KeyError, TypeError, ValueError) as exc:
        return _hard_stop(
            request,
            state="HARD_STOP_INVALID_EVIDENCE",
            blocker=f"invalid_admitted_pipeline_evidence:{type(exc).__name__}",
        )


__all__ = [
    "FORMAL_RESEARCH_MODE",
    "MODES",
    "PipelineOrchestrationError",
    "PipelineRequest",
    "PipelineResult",
    "PipelineTerminal",
    "SHADOW_MODE",
    "build_initial_pool_artifact",
    "run_pipeline",
]
