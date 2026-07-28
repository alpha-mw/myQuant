"""Offline application boundary for the additive V17 protocol-v3 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from quant_investor.v17_v3_contract import (
    PROTOCOL_VERSION,
    verify_package,
)
from quant_investor.v17_v3_contract.policy import source_role_matrix
from quant_investor.v17_v3_contract.resources import (
    PACKAGE_MANIFEST_SHA256,
    load_packaged_json,
)

from .activation import (
    ActivationPublisher,
    CurrentFormalResult,
    PublicationOutcome,
)
from .artifacts import (
    artifact_reference,
    load_typed_artifact,
    replace_typed_cas,
    runtime_artifact,
    seal_typed_artifact,
    write_typed_exact_once,
)
from .authority import DELIVERY_STATUS, authority_envelope
from .pipeline import (
    FORMAL_RESEARCH_MODE,
    PipelineRequest,
    PipelineResult,
    build_initial_pool_artifact,
    run_pipeline,
)
from .redaction import assert_public_envelope_safe
from .sources import AdmittedSources, admit_source_locator
from .storage import SecureStore
from .storage import EMPTY_SHA256, SHADOW_RESULTS_ROOT


class RuntimeServiceError(ValueError):
    """A public runtime request failed its offline service boundary."""

    exit_code = 2


@dataclass(frozen=True)
class RuntimeReadiness:
    status: str
    packaged_asset_count: int
    source_phase_count: int

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.runtime-readiness.v1",
            "status": self.status,
            "runtime_usable": True,
            "provider_calls": False,
            "llm_calls": False,
            "packaged_asset_count": self.packaged_asset_count,
            "source_phase_count": self.source_phase_count,
            **authority_envelope(),
        }
        assert_public_envelope_safe(payload)
        return payload


@dataclass(frozen=True)
class AnalysisOutcome:
    result: PipelineResult
    publication: PublicationOutcome | None

    def to_public_wire(self) -> dict[str, Any]:
        payload = self.result.to_public_wire()
        payload["publication"] = (
            None if self.publication is None else self.publication.to_public_wire()
        )
        assert_public_envelope_safe(payload)
        return payload


@dataclass(frozen=True)
class CalibrationOutcome:
    status: str
    promoted: bool
    active_weight: str | None
    bootstrap_matrix_sha256: str | None
    blocker_count: int
    promotion_id: str | None = None
    promotion_path: str | None = None
    promotion_sha256: str | None = None

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.calibration-outcome.v1",
            "status": self.status,
            "promoted": self.promoted,
            "active_weight": self.active_weight,
            "bootstrap_matrix_sha256": self.bootstrap_matrix_sha256,
            "blocker_count": self.blocker_count,
            "promotion_id": self.promotion_id,
            "promotion_path": self.promotion_path,
            "promotion_sha256": self.promotion_sha256,
            **authority_envelope(),
        }
        assert_public_envelope_safe(payload)
        return payload


@dataclass(frozen=True)
class InitialPoolOutcome:
    status: str
    output_id: str
    relative_path: str
    byte_sha256: str
    pool_count: int

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.initial-pool-outcome.v1",
            "status": self.status,
            "output_id": self.output_id,
            "relative_path": self.relative_path,
            "byte_sha256": self.byte_sha256,
            "pool_count": self.pool_count,
            **authority_envelope(),
        }
        assert_public_envelope_safe(payload)
        return payload


def verify_runtime() -> RuntimeReadiness:
    """Verify package resources without creating roots or invoking providers."""

    assets = verify_package()
    phases = source_role_matrix()
    return RuntimeReadiness(
        status=DELIVERY_STATUS,
        packaged_asset_count=len(assets),
        source_phase_count=len(phases),
    )


def admitted_sources(
    *,
    workspace_root: Path,
    locator_path: str,
    expected_locator_sha256: str,
) -> tuple[SecureStore, AdmittedSources]:
    store = SecureStore(workspace_root)
    admission = admit_source_locator(
        store,
        locator_path=locator_path,
        expected_locator_sha256=expected_locator_sha256,
    )
    return store, admission


def analyze(
    *,
    workspace_root: Path,
    mode: str,
    locator_path: str,
    expected_locator_sha256: str,
) -> AnalysisOutcome:
    """Analyze only from an exact admitted locator; there is no caller-array API."""

    store, admission = admitted_sources(
        workspace_root=workspace_root,
        locator_path=locator_path,
        expected_locator_sha256=expected_locator_sha256,
    )
    result = run_pipeline(
        PipelineRequest(
            mode=mode,
            admitted_sources=admission,
        )
    )
    if result.terminal_artifact is not None:
        for artifact in result.artifacts:
            write_typed_exact_once(store, artifact)
        if mode != FORMAL_RESEARCH_MODE:
            _publish_shadow_latest(store, result)
    return AnalysisOutcome(result, None)


def build_initial_pool(
    *,
    workspace_root: Path,
    locator_path: str,
    expected_locator_sha256: str,
) -> InitialPoolOutcome:
    """Materialize the immutable PRESELECT artifact for later staged lineage."""

    store, admission = admitted_sources(
        workspace_root=workspace_root,
        locator_path=locator_path,
        expected_locator_sha256=expected_locator_sha256,
    )
    artifact = build_initial_pool_artifact(admission)
    write_typed_exact_once(store, artifact)
    return InitialPoolOutcome(
        status="PRESELECT_COMPLETE",
        output_id=str(artifact.document["output_id"]),
        relative_path=str(artifact.relative_path),
        byte_sha256=artifact.byte_sha256,
        pool_count=int(artifact.document["pool_count"]),
    )


def _publish_shadow_latest(
    store: SecureStore,
    result: PipelineResult,
) -> None:
    terminal = result.terminal_artifact
    if terminal is None:
        return
    latest_path = SHADOW_RESULTS_ROOT / "strategies" / result.strategy_id / "_latest.json"
    lock_path = SHADOW_RESULTS_ROOT / "strategies" / result.strategy_id / ".latest.lock"
    with store.locked(lock_path):
        observed = store.read_optional(latest_path)
        expected = EMPTY_SHA256
        if observed is not None:
            current = load_typed_artifact(
                observed.data,
                label="shadow latest",
                expected_version="myquant.v17.v3.shadow-latest.v1",
            )
            if str(current["cutoff"]) > result.cutoff:
                return
            expected = observed.byte_sha256
        latest = seal_typed_artifact(
            {
                "version": "myquant.v17.v3.shadow-latest.v1",
                "protocol_version": PROTOCOL_VERSION,
                "strategy_id": result.strategy_id,
                "cutoff": result.cutoff,
                "updated_at": result.cutoff,
                "shadow_output_ref": terminal.reference,
                "authority": authority_envelope(),
            }
        )
        replace_typed_cas(
            store,
            runtime_artifact(
                relative_path=latest_path,
                document=latest,
            ),
            expected_sha256=expected,
        )


def calibrate(
    *,
    workspace_root: Path,
    locator_path: str,
    expected_locator_sha256: str,
) -> CalibrationOutcome:
    """Run fusion calibration over one admitted offline role closure."""

    store, admission = admitted_sources(
        workspace_root=workspace_root,
        locator_path=locator_path,
        expected_locator_sha256=expected_locator_sha256,
    )
    role = admission.materialize("fusion_calibration")
    if isinstance(role, bytes):
        raise RuntimeServiceError("fusion calibration role is not materializable JSON")
    payload = role.get("payload", role) if isinstance(role, dict) else role
    if not isinstance(payload, dict):
        raise RuntimeServiceError("fusion calibration payload must be an object")
    from quant_investor.v17_v3_runtime.algorithms import (
        CalibrationError,
        CalibrationMonth,
        calibrate_fusion,
    )

    months: list[CalibrationMonth] = []
    raw_months = payload.get("months")
    if not isinstance(raw_months, list):
        raise RuntimeServiceError("fusion calibration months must be an array")
    for index, raw_month in enumerate(raw_months):
        if not isinstance(raw_month, Mapping):
            raise RuntimeServiceError("fusion calibration month must be an object")
        origin = raw_month.get("origin")
        if type(origin) is not str:
            raise RuntimeServiceError("fusion calibration origin must be an ISO date")
        expected_cutoff = f"{origin}T07:00:00Z"
        quant = _read_exact_artifact_ref(
            store,
            raw_month.get("quant_branch_ref"),
            expected_version="myquant.v17.v3.branch-output.v1",
            expected_strategy_id=admission.strategy_id,
            expected_cutoff=expected_cutoff,
            label=f"calibration month {index} Quant branch",
        )
        fundamental = _read_exact_artifact_ref(
            store,
            raw_month.get("fundamental_branch_ref"),
            expected_version="myquant.v17.v3.branch-output.v1",
            expected_strategy_id=admission.strategy_id,
            expected_cutoff=expected_cutoff,
            label=f"calibration month {index} Fundamental branch",
        )
        ordered_pool = tuple(raw_month.get("ordered_pool", ()))
        if (
            quant.get("branch") != "quant"
            or fundamental.get("branch") != "fundamental"
            or tuple(quant.get("ordered_domain", ())) != ordered_pool
            or tuple(fundamental.get("ordered_domain", ())) != ordered_pool
        ):
            raise RuntimeServiceError("calibration branch identity or same-pool binding mismatch")
        months.append(
            CalibrationMonth(
                origin=origin,
                label_252_end_session=raw_month.get("label_252_end_session"),
                ordered_pool=ordered_pool,
                quant_branch=quant,
                fundamental_branch=fundamental,
                forward_return_60=_return_mapping(
                    raw_month.get("forward_return_60"),
                    label=f"calibration month {index} 60-session return",
                ),
                forward_return_252=_return_mapping(
                    raw_month.get("forward_return_252"),
                    label=f"calibration month {index} 252-session return",
                ),
                label_252_mature=raw_month.get("label_252_mature"),
            )
        )
    try:
        result = calibrate_fusion(
            months,
            canonical_sessions=payload.get("canonical_sessions"),
            scheduled_origins=payload.get("scheduled_origins"),
            active_cutoff=payload.get("active_cutoff"),
        )
    except CalibrationError as exc:
        raise RuntimeServiceError("fusion calibration failed closed") from exc
    promotion = _persist_calibration_result(
        store,
        admission,
        result=result,
    )
    promoted = promotion.document["status"] == "PROMOTED"
    return CalibrationOutcome(
        status="FUSION_CALIBRATED" if promoted else "PROMOTION_REJECTED",
        promoted=promoted,
        active_weight=(format(result.active_weight, ".2f") if promoted else None),
        bootstrap_matrix_sha256=(result.bootstrap_matrix_sha256),
        blocker_count=len(result.blockers),
        promotion_id=str(promotion.document["promotion_id"]),
        promotion_path=str(promotion.relative_path),
        promotion_sha256=promotion.byte_sha256,
    )


def _persist_calibration_result(
    store: SecureStore,
    admission: AdmittedSources,
    *,
    result: Any,
):
    wrapper = admission.materialize("fusion_calibration")
    if not isinstance(wrapper, Mapping):
        raise RuntimeServiceError("fusion calibration wrapper must be an object")
    run_id = wrapper.get("run_id")
    if type(run_id) is not str:
        raise RuntimeServiceError("fusion calibration wrapper has no run_id")
    input_ref = dict(admission.reference_for_role("fusion_calibration"))
    locator_artifact = runtime_artifact(
        relative_path=admission.locator_path,
        document=admission.documents["source_locator"],
    )
    evidence_refs = sorted(
        (
            dict(admission.reference_for_role("initial_pool_output")),
            dict(admission.reference_for_role("quant_branch_output")),
            dict(admission.reference_for_role("fundamental_branch_output")),
            input_ref,
        ),
        key=lambda reference: (
            reference["relative_path"],
            reference["byte_sha256"],
        ),
    )
    # Quant timing and Fundamental forward are independent 1260/2520-session
    # gates.  They must be exact, already-admitted receipts; fusion calibration
    # is not allowed to manufacture their acceptance.
    quant_receipt, quant_ref = _admitted_calibration_receipt(
        store,
        admission,
        role="quant_timing_calibration",
        expected_kind="QUANT_TIMING",
    )
    fundamental_receipt, fundamental_ref = _admitted_calibration_receipt(
        store,
        admission,
        role="fundamental_forward_calibration",
        expected_kind="FUNDAMENTAL_FORWARD",
    )
    fusion_receipt = runtime_artifact(
        relative_path=("data/private/v17_v3_runs/" f"{run_id}/calibration/fusion_promotion.json"),
        document=seal_typed_artifact(
            {
                "version": ("myquant.v17.v3.fusion-calibration-receipt.v1"),
                "protocol_version": PROTOCOL_VERSION,
                "calibration_id": f"fusion-promotion-{run_id}",
                "calibration_kind": "FUSION_PROMOTION",
                "strategy_id": admission.strategy_id,
                "cutoff": admission.cutoff,
                "created_at": admission.cutoff,
                "observation_end_at": admission.cutoff,
                "accepted": bool(result.promoted),
                "evidence_refs": sorted(
                    [input_ref, locator_artifact.reference],
                    key=lambda reference: (
                        reference["relative_path"],
                        reference["byte_sha256"],
                    ),
                ),
                "authority": authority_envelope(),
            }
        ),
    )
    # Fixed policy order, not filesystem or caller order.
    calibration_refs = [
        quant_ref,
        fundamental_ref,
        fusion_receipt.reference,
    ]
    accepted = bool(
        quant_receipt["accepted"] and fundamental_receipt["accepted"] and result.promoted
    )
    folds = [
        {
            "fold_index": fold.index,
            "training_origins": [value.isoformat() for value in fold.training_origins],
            "oos_origins": [value.isoformat() for value in fold.oos_origins],
            "selected_quant_weight": format(
                fold.selected_weight,
                ".2f",
            ),
        }
        for fold in result.folds
    ]
    outer = [origin for fold in folds for origin in fold["oos_origins"]]
    active = list(wrapper["payload"]["scheduled_origins"][-60:])
    common = {
        "version": "myquant.v17.v3.fusion-promotion-receipt.v1",
        "protocol_version": PROTOCOL_VERSION,
        "promotion_id": f"fusion-promotion-{run_id}",
        "strategy_id": admission.strategy_id,
        "cutoff": admission.cutoff,
        "created_at": admission.cutoff,
        "observation_end_at": admission.cutoff,
        "bootstrap_matrix_sha256": (result.bootstrap_matrix_sha256),
        "calibration_receipt_refs": calibration_refs,
        "evidence_refs": evidence_refs,
        "contract_package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "preselector_policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
            "semantic_sha256"
        ],
        "quant_branch_policy_sha256": load_packaged_json("resources/quant_branch_policy.v1.json")[
            "semantic_sha256"
        ],
        "fundamental_branch_policy_sha256": load_packaged_json(
            "resources/fundamental_branch_policy.v1.json"
        )["semantic_sha256"],
        "fusion_policy_sha256": load_packaged_json("resources/fusion_policy.v1.json")[
            "semantic_sha256"
        ],
        "evidence_bound": result.evidence_bound,
        "effective_outer_blocks": result.effective_outer_blocks,
        "active_refit_origins": active,
        "outer_oos_origins": outer,
        "fold_inventory": folds,
        "oos_mean_hit60": format(result.oos_mean_hit60, "f"),
        "oos_mean_q25_252": format(result.oos_mean_q25_252, "f"),
        "oos_p5_hit60": format(result.oos_p5_hit60, "f"),
        "oos_p5_q25_252": format(result.oos_p5_q25_252, "f"),
        "authority": authority_envelope(),
    }
    if accepted:
        promotion_payload = {
            **common,
            "status": "PROMOTED",
            "accepted": True,
            "active_formal_research_weight": format(
                result.active_weight,
                ".2f",
            ),
            "active_refit_origins": active,
            "outer_oos_origins": outer,
            "fold_inventory": folds,
        }
    else:
        reasons = sorted(
            set(
                (
                    *result.blockers,
                    *(() if quant_receipt["accepted"] else ("quant_timing_calibration_rejected",)),
                    *(
                        ()
                        if fundamental_receipt["accepted"]
                        else ("fundamental_forward_calibration_rejected",)
                    ),
                )
                or ("calibration_threshold_not_met",)
            )
        )
        promotion_payload = {
            **common,
            "status": "PROMOTION_REJECTED",
            "accepted": False,
            "evaluated_quant_weight": format(result.active_weight, ".2f"),
            "rejection_reasons": reasons,
        }
    promotion = runtime_artifact(
        relative_path=("data/private/v17_v3_runs/" f"{run_id}/fusion_promotion_receipt.json"),
        document=seal_typed_artifact(promotion_payload),
    )
    # No authority-bearing promotion bytes are written until the entire input
    # closure and the resulting promotion artifact have validated.
    write_typed_exact_once(store, fusion_receipt)
    write_typed_exact_once(store, promotion)
    return promotion


def _admitted_calibration_receipt(
    store: SecureStore,
    admission: AdmittedSources,
    *,
    role: str,
    expected_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = dict(admission.reference_for_role(role))
    receipt = _read_exact_artifact_ref(
        store,
        reference,
        expected_version="myquant.v17.v3.fusion-calibration-receipt.v1",
        expected_strategy_id=admission.strategy_id,
        expected_cutoff=admission.cutoff,
        label=f"{expected_kind} calibration receipt",
    )
    if receipt.get("calibration_kind") != expected_kind:
        raise RuntimeServiceError("admitted calibration receipt kind mismatch")
    if type(receipt.get("accepted")) is not bool:
        raise RuntimeServiceError("admitted calibration receipt has no decision")
    return receipt, reference


def _read_exact_artifact_ref(
    store: SecureStore,
    reference: Any,
    *,
    expected_version: str,
    expected_strategy_id: str,
    expected_cutoff: str,
    label: str,
) -> dict[str, Any]:
    """Resolve one seven-field reference to exact typed bytes."""

    if not isinstance(reference, Mapping):
        raise RuntimeServiceError(f"{label} reference must be an object")
    path = reference.get("relative_path")
    byte_sha256 = reference.get("byte_sha256")
    if type(path) is not str or type(byte_sha256) is not str:
        raise RuntimeServiceError(f"{label} reference is incomplete")
    raw = store.read(PurePosixPath(path), byte_sha256)
    document = load_typed_artifact(
        raw,
        label=label,
        expected_version=expected_version,
    )
    if (
        document.get("strategy_id") != expected_strategy_id
        or document.get("cutoff") != expected_cutoff
    ):
        raise RuntimeServiceError(f"{label} strategy/cutoff binding mismatch")
    observed = artifact_reference(
        relative_path=path,
        document=document,
        raw=raw,
    )
    if observed != dict(reference):
        raise RuntimeServiceError(f"{label} exact reference mismatch")
    return document


def _return_mapping(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, list):
        raise RuntimeServiceError(f"{label} must be an array")
    result: dict[str, str] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise RuntimeServiceError(f"{label} row must be an object")
        symbol = row.get("symbol")
        observed = row.get("value")
        if type(symbol) is not str or type(observed) is not str or symbol in result:
            raise RuntimeServiceError(f"{label} row is invalid")
        result[symbol] = observed
    return result


def status(
    *,
    workspace_root: Path,
    strategy_id: str,
) -> CurrentFormalResult:
    """Read and revalidate the current formal result without any write."""

    return ActivationPublisher(SecureStore(workspace_root)).current_active(strategy_id)


__all__ = [
    "AnalysisOutcome",
    "CalibrationOutcome",
    "InitialPoolOutcome",
    "RuntimeReadiness",
    "RuntimeServiceError",
    "admitted_sources",
    "analyze",
    "build_initial_pool",
    "calibrate",
    "status",
    "verify_runtime",
]
