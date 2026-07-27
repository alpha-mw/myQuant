from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from quant_investor.research_runtime_control import (
    ACTIVE_POINTER_VERSION,
    EMPTY_SHA256,
    RUN_VERSION,
    SELECTOR_VERSION,
    ControlArtifact,
    ControlCASMismatch,
    ControlStorageSecurityError,
    ControlStore,
    ResearchRuntimeControl,
    RuntimeControlError,
    RuntimeControlThirdState,
    SimulatedTransitionCrash,
    authority_ceiling,
)
from quant_investor.research_runtime_control.canonical import (
    CanonicalControlError,
    canonical_bytes,
    decode,
    decode_reference,
    encode,
    seal,
    sha256,
)
from quant_investor.research_runtime_control.control import (
    SELECTOR_LOCK,
    SELECTOR_PATH,
    V15ActiveRunPublisher,
)

STRATEGY = "cn-stock"
T0 = "2026-07-25T07:00:00Z"
T1 = "2026-07-26T07:00:00Z"
T2 = "2026-07-27T07:00:00Z"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    tmp_path.chmod(0o700)
    return tmp_path


def _owner_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    current = path
    while current.name != "results":
        current.chmod(0o700)
        current = current.parent


def _owner_file(path: Path, raw: bytes) -> None:
    _owner_directory(path.parent)
    path.write_bytes(raw)
    path.chmod(0o600)


def _artifact(
    relative_path: str,
    payload: dict[str, object],
    *,
    strategy_id: str = STRATEGY,
    cutoff: str,
) -> ControlArtifact:
    try:
        document = seal(payload)
        raw = encode(document)
    except CanonicalControlError:
        from quant_investor.v17_v4_contract import (
            canonical_resource_bytes,
            seal_semantic,
        )

        document = seal_semantic(dict(payload))
        raw = canonical_resource_bytes(document)
        document = decode_reference(raw)
    return ControlArtifact(
        relative_path,
        document,
        raw,
        sha256(raw),
        strategy_id,
        cutoff,
    )


def _placeholder_ref(
    artifact_id: str,
    version: str,
    path: str,
    *,
    cutoff: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(f"bytes:{artifact_id}".encode()).hexdigest(),
        "cutoff": cutoff,
        "relative_path": path,
        "semantic_sha256": hashlib.sha256(
            f"semantic:{artifact_id}".encode()
        ).hexdigest(),
        "strategy_id": STRATEGY,
    }


def _v4_authority() -> dict[str, bool]:
    return {
        "broker": False,
        "execution": False,
        "formal_research_publication": True,
        "order": False,
        "research_runtime_default": False,
        "trade": False,
    }


def _ordered_refs(*refs: dict[str, str]) -> list[dict[str, str]]:
    return sorted(
        refs,
        key=lambda row: (
            row["relative_path"],
            row["byte_sha256"],
            row["artifact_id"],
        ),
    )


def _seed_v4(
    control: ResearchRuntimeControl,
    workspace: Path,
    *,
    run_id: str,
    cutoff: str,
) -> tuple[
    ControlArtifact,
    ControlArtifact,
    ControlArtifact,
    list[dict[str, str]],
]:
    _, v15_target, v15_pointer, _ = control.resolve_current(
        strategy_id=STRATEGY
    )
    target = control.install_protocol_target(
        protocol_id="myquant.v17.v4",
        strategy_id=STRATEGY,
        cutoff=cutoff,
    )
    run = _artifact(
        (
            "results/v17_v4_formal_research/strategies/"
            f"{STRATEGY}/runs/{run_id}/run.json"
        ),
        {
            "version": "myquant.v17.v4.formal-output.v1",
            "output_id": run_id,
            "strategy_id": STRATEGY,
            "cutoff": cutoff,
            "terminal_state": "PUBLISHED_RESEARCH_ONLY",
            "protocol_version": "myquant.v17.v4",
            "authority": _v4_authority(),
            "evidence_refs": [
                _placeholder_ref(
                    f"formal-evidence-{run_id}",
                    "myquant.v17.v4.portfolio-output.v1",
                    (
                        "data/private/v17_v4_runs/"
                        f"{run_id}/portfolio-output.json"
                    ),
                    cutoff=cutoff,
                )
            ],
        },
        cutoff=cutoff,
    )
    receipt = _artifact(
        (
            "results/v17_v4_formal_research/strategies/"
            f"{STRATEGY}/receipts/{run_id}.json"
        ),
        {
            "version": "myquant.v17.v4.formal-activation-receipt.v1",
            "receipt_id": f"receipt-{run_id}",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "cutoff": cutoff,
            "status": "FORMAL_ACTIVATED",
            "from_state": "V15_DEFAULT",
            "to_state": "FORMAL_ACTIVE",
            "recorded_at": cutoff,
            "authority": _v4_authority(),
            "expected_pointer_sha256": "EMPTY",
            "observed_pointer_sha256": "EMPTY",
            "proposed_pointer_sha256": "f" * 64,
            "post_readback_sha256": "f" * 64,
            "formal_output_ref": run.reference,
            **{
                field: _placeholder_ref(
                    field.removesuffix("_ref"),
                    version,
                    (
                        "data/private/v17_v4_runs/"
                        f"{run_id}/{field}.json"
                    ),
                    cutoff=cutoff,
                )
                for field, version in {
                    "source_locator_ref": "myquant.v17.v4.source-locator.v1",
                    "quant_calibration_receipt_ref": (
                        "myquant.v17.v4.fusion-calibration-receipt.v1"
                    ),
                    "fundamental_calibration_receipt_ref": (
                        "myquant.v17.v4.fusion-calibration-receipt.v1"
                    ),
                    "fusion_promotion_receipt_ref": (
                        "myquant.v17.v4.fusion-promotion-receipt.v1"
                    ),
                    "deep_bundle_ref": (
                        "myquant.v17.v4.deep-evidence-bundle.v1"
                    ),
                    "holdings_snapshot_ref": (
                        "myquant.v17.v4.holdings-snapshot.v1"
                    ),
                    "risk_policy_ref": (
                        "myquant.v17.v4.portfolio-risk-policy.v1"
                    ),
                    "macro_overlay_ref": (
                        "myquant.v17.v4.portfolio-overlay.v1"
                    ),
                    "markov_overlay_ref": (
                        "myquant.v17.v4.portfolio-overlay.v1"
                    ),
                    "factor_control_active_set_ref": (
                        "factor-governance-production-control."
                        "active-set-pointer.v1"
                    ),
                    "factor_control_activation_receipt_ref": (
                        "factor-governance-production-control."
                        "activation-receipt.v1"
                    ),
                }.items()
            },
        },
        cutoff=cutoff,
    )
    receipt_payload = dict(receipt.document)
    receipt_payload["evidence_refs"] = _ordered_refs(
        *[
            value
            for key, value in receipt_payload.items()
            if key.endswith("_ref") and isinstance(value, dict)
        ]
    )
    receipt = _artifact(
        receipt.relative_path,
        {
            key: value
            for key, value in receipt_payload.items()
            if key != "semantic_sha256"
        },
        cutoff=cutoff,
    )
    pointer = _artifact(
        (
            "results/v17_v4_formal_research/strategies/"
            f"{STRATEGY}/_active.json"
        ),
        {
            "version": "myquant.v17.v4.formal-active-pointer.v1",
            "authority": {
                "broker": False,
                "execution": False,
                "formal_research_publication": True,
                "order": False,
                "research_runtime_default": False,
                "trade": False,
            },
            "formal_output_ref": run.reference,
            "pointer_id": f"active-{run_id}",
            "protocol_version": "myquant.v17.v4",
            "receipt_ref": receipt.reference,
            "state": "FORMAL_ACTIVE",
            "strategy_id": STRATEGY,
            "cutoff": cutoff,
            "updated_at": cutoff,
        },
        cutoff=cutoff,
    )
    public_refs = [
        _placeholder_ref(
            f"public-{index}-{run_id}",
            f"myquant.v17.v4.public-surface-receipt-{index}.v1",
            f"data/private/v17_v4_runs/{run_id}/public-{index}.json",
            cutoff=cutoff,
        )
        for index in range(4)
    ]
    validation_refs = [
        _placeholder_ref(
            f"validation-{index}-{run_id}",
            f"myquant.v17.v4.validation-receipt-{index}.v1",
            f"data/private/v17_v4_runs/{run_id}/validation-{index}.json",
            cutoff=cutoff,
        )
        for index in range(5)
    ]
    rollback_ref = _placeholder_ref(
        f"rollback-{run_id}",
        "myquant.research-runtime.rollback-receipt.v1",
        (
            "results/research_runtime_control/rollback_receipts/"
            f"rollback-{run_id}.json"
        ),
        cutoff=cutoff,
    )
    bootstrap_ref = _placeholder_ref(
        f"bootstrap-{run_id}",
        "myquant.research-runtime.route-bootstrap-receipt.v1",
        (
            "results/research_runtime_control/bootstrap_receipts/"
            f"bootstrap-{run_id}.json"
        ),
        cutoff=cutoff,
    )
    eligibility_receipt = _artifact(
        (
            "results/v17_v4_canary/strategies/"
            f"{STRATEGY}/receipts/eligibility-{run_id}.json"
        ),
        {
            "version": "myquant.v17.v4.default-eligibility-receipt.v1",
            "receipt_id": f"eligibility-{run_id}",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "authority": _v4_authority(),
            "from_state": "FORMAL_ACTIVE",
            "to_state": "DEFAULT_ELIGIBLE",
            "status": "DEFAULT_ELIGIBLE",
            "recorded_at": cutoff,
            "expected_pointer_sha256": "EMPTY",
            "observed_pointer_sha256": "EMPTY",
            "proposed_pointer_sha256": "e" * 64,
            "post_readback_sha256": "e" * 64,
            "formal_active_pointer_ref": pointer.reference,
            "selector_bootstrap_receipt_ref": bootstrap_ref,
            "rollback_drill_receipt_ref": rollback_ref,
            "public_surface_receipt_refs": _ordered_refs(*public_refs),
            "validation_receipt_refs": _ordered_refs(*validation_refs),
            "evidence_refs": _ordered_refs(
                pointer.reference,
                bootstrap_ref,
                rollback_ref,
                *public_refs,
                *validation_refs,
            ),
        },
        cutoff=cutoff,
    )
    eligible_pointer = _artifact(
        (
            "results/v17_v4_canary/strategies/"
            f"{STRATEGY}/_eligible.json"
        ),
        {
            "version": "myquant.v17.v4.default-eligible-pointer.v1",
            "pointer_id": f"eligible-{run_id}",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "authority": _v4_authority(),
            "state": "DEFAULT_ELIGIBLE",
            "formal_active_pointer_ref": pointer.reference,
            "eligibility_receipt_ref": eligibility_receipt.reference,
            "updated_at": cutoff,
        },
        cutoff=cutoff,
    )
    policy_ref = _placeholder_ref(
        f"policy-{run_id}",
        "myquant.v17.v4.historical-canary-policy.v1",
        (
            "results/v17_v4_canary/strategies/"
            f"{STRATEGY}/policies/{run_id}.json"
        ),
        cutoff=cutoff,
    )
    comparison_refs = [
        _placeholder_ref(
            f"comparison-{index}-{run_id}",
            "myquant.v17.v4.dual-run-comparison.v1",
            (
                "results/v17_v4_canary/strategies/"
                f"{STRATEGY}/runs/{run_id}/{index}.json"
            ),
            cutoff=cutoff,
        )
        for index in range(5)
    ]
    paired_run_ids = [f"paired-{index}-{run_id}" for index in range(5)]
    canary_receipt = _artifact(
        (
            "results/v17_v4_canary/strategies/"
            f"{STRATEGY}/receipts/canary-{run_id}.json"
        ),
        {
            "version": "myquant.v17.v4.canary-receipt.v1",
            "receipt_id": f"canary-{run_id}",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "authority": _v4_authority(),
            "from_state": "CANARY",
            "to_state": "CANARY",
            "status": "CANARY_COMPLETED",
            "recorded_at": cutoff,
            "expected_pointer_sha256": "EMPTY",
            "observed_pointer_sha256": "EMPTY",
            "proposed_pointer_sha256": "d" * 64,
            "post_readback_sha256": "d" * 64,
            "eligibility_pointer_ref": eligible_pointer.reference,
            "historical_canary_policy_ref": policy_ref,
            "v15_protocol_target_ref": v15_target.reference,
            "v15_active_run_pointer_ref": v15_pointer.reference,
            "session_window": {
                "start_session": "2026-07-20",
                "end_session": "2026-07-24",
                "required_session_count": 5,
            },
            "paired_run_ids": paired_run_ids,
            "comparison_refs": comparison_refs,
            "completed_sessions": [
                "2026-07-20",
                "2026-07-21",
                "2026-07-22",
                "2026-07-23",
                "2026-07-24",
            ],
            "threshold_results": [
                {
                    "threshold_id": "all-gates",
                    "observed": "1",
                    "status": "PASS",
                }
            ],
            "side_effect_counters": {
                "active_run_cas_mismatch_count": 0,
                "analysis_time_provider_call_count": 0,
                "broker_call_count": 0,
                "canary_pointer_cas_mismatch_count": 0,
                "data_pointer_cas_mismatch_count": 0,
                "eligibility_pointer_cas_mismatch_count": 0,
                "execution_call_count": 0,
                "factor_pointer_cas_mismatch_count": 0,
                "formal_pointer_cas_mismatch_count": 0,
                "llm_control_call_count": 0,
                "order_call_count": 0,
                "protocol_target_cas_mismatch_count": 0,
                "selector_cas_mismatch_count": 0,
                "trade_call_count": 0,
            },
            "evidence_refs": _ordered_refs(
                eligible_pointer.reference,
                policy_ref,
                v15_target.reference,
                v15_pointer.reference,
                *comparison_refs,
            ),
        },
        cutoff=cutoff,
    )
    canary_pointer = _artifact(
        (
            "results/v17_v4_canary/strategies/"
            f"{STRATEGY}/_current.json"
        ),
        {
            "version": "myquant.v17.v4.canary-pointer.v1",
            "pointer_id": f"canary-pointer-{run_id}",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "authority": _v4_authority(),
            "state": "CANARY",
            "eligibility_pointer_ref": eligible_pointer.reference,
            "canary_receipt_ref": canary_receipt.reference,
            "paired_run_ids": paired_run_ids,
            "updated_at": cutoff,
        },
        cutoff=cutoff,
    )
    _owner_file(workspace / run.relative_path, run.raw)
    _owner_file(workspace / receipt.relative_path, receipt.raw)
    _owner_file(workspace / pointer.relative_path, pointer.raw)
    for artifact in (
        eligibility_receipt,
        eligible_pointer,
        canary_receipt,
        canary_pointer,
    ):
        _owner_file(workspace / artifact.relative_path, artifact.raw)
    _owner_file(
        workspace
        / "results/v17_v4_formal_research/strategies"
        / STRATEGY
        / ".active.lock",
        b"",
    )
    return (
        target,
        pointer,
        run,
        [
            pointer.reference,
            eligible_pointer.reference,
            canary_pointer.reference,
        ],
    )


def _bootstrap(
    control: ResearchRuntimeControl,
    *,
    cutoff: str = T0,
    intent_id: str = "bootstrap-1",
    receipt_id: str = "bootstrap-receipt-1",
    **kwargs: object,
):
    return control.bootstrap_v15(
        strategy_id=STRATEGY,
        run_id=f"v15-{intent_id}",
        cutoff=cutoff,
        recorded_at=cutoff,
        intent_id=intent_id,
        receipt_id=receipt_id,
        **kwargs,
    )


def test_bootstrap_cutover_and_rollback_use_current_v15_standby(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    selector, v15_target, v15_pointer, first_v15_run = (
        control.resolve_current(strategy_id=STRATEGY)
    )

    assert bootstrap.outcome == "BOOTSTRAP_SUCCEEDED"
    assert selector.document["status"] == "V15_DEFAULT"
    assert set(v15_target.document["authority_ceiling"].values()) == {False}
    assert set(first_v15_run.document["authority_ceiling"].values()) == {
        False
    }
    assert control.lock_trace == [
        str(V15ActiveRunPublisher.lock_path(STRATEGY)),
        str(SELECTOR_LOCK),
    ]
    assert set(v15_target.reference) == {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }

    v4_target, v4_pointer, v4_run, v4_evidence = _seed_v4(
        control,
        workspace,
        run_id="v4-run-1",
        cutoff=T1,
    )
    external_before = {
        v4_pointer.relative_path: (
            workspace / v4_pointer.relative_path
        ).read_bytes(),
        v4_run.relative_path: (workspace / v4_run.relative_path).read_bytes(),
    }
    cutover = control.cutover(
        strategy_id=STRATEGY,
        v4_protocol_target_ref=v4_target.reference,
        expected_v4_active_pointer_sha256=v4_pointer.byte_sha256,
        expected_v4_run_ref=v4_run.reference,
        expected_selector_sha256=selector.byte_sha256,
        recorded_at=T1,
        intent_id="cutover-1",
        receipt_id="cutover-receipt-1",
        required_evidence_refs=[
            bootstrap.receipt.reference,
            *v4_evidence,
        ],
    )
    assert cutover.outcome == "CUTOVER_SUCCEEDED"
    assert control.lock_trace == [
        (
            "results/v17_v4_formal_research/strategies/"
            f"{STRATEGY}/.active.lock"
        ),
        str(SELECTOR_LOCK),
    ]
    assert control.resolve_current(strategy_id=STRATEGY)[3].reference == (
        v4_run.reference
    )

    second_v15_run = control.create_v15_run(
        run_id="v15-standby-current",
        strategy_id=STRATEGY,
        cutoff=T2,
        evidence_refs=[],
    )
    current_v15_pointer = control.v15_publisher.publish(
        strategy_id=STRATEGY,
        run_ref=second_v15_run.reference,
        cutoff=T2,
        updated_at=T2,
        expected_pointer_sha256=v15_pointer.byte_sha256,
    )
    rollback = control.rollback_to_v15(
        strategy_id=STRATEGY,
        v15_protocol_target_ref=v15_target.reference,
        expected_v15_active_pointer_sha256=(
            current_v15_pointer.byte_sha256
        ),
        expected_v15_run_ref=second_v15_run.reference,
        expected_selector_sha256=cutover.selector_sha256,
        successful_cutover_receipt_ref=cutover.receipt.reference,
        recorded_at=T2,
        intent_id="rollback-1",
        receipt_id="rollback-receipt-1",
    )

    assert rollback.outcome == "ROLLBACK_SUCCEEDED"
    assert control.lock_trace == [
        str(V15ActiveRunPublisher.lock_path(STRATEGY)),
        str(SELECTOR_LOCK),
    ]
    restored = control.resolve_current(strategy_id=STRATEGY)
    assert restored[0].document["status"] == "ROLLED_BACK_TO_V15"
    assert restored[3].reference == second_v15_run.reference
    assert restored[3].reference != first_v15_run.reference
    for relative_path, raw in external_before.items():
        assert (workspace / relative_path).read_bytes() == raw


def test_crash_recovery_exact_proposed_and_exact_old(
    workspace: Path,
) -> None:
    proposed_workspace = workspace / "proposed"
    proposed_workspace.mkdir(mode=0o700)
    proposed = ResearchRuntimeControl(proposed_workspace)
    with pytest.raises(SimulatedTransitionCrash):
        _bootstrap(
            proposed,
            intent_id="bootstrap-proposed",
            receipt_id="unused",
            crash_after_cas=True,
        )
    first_pointer = proposed.resolve_current(strategy_id=STRATEGY)[2]
    newer_run = proposed.create_v15_run(
        run_id="v15-after-bootstrap-crash",
        strategy_id=STRATEGY,
        cutoff=T1,
        evidence_refs=[],
    )
    proposed.v15_publisher.publish(
        strategy_id=STRATEGY,
        run_ref=newer_run.reference,
        cutoff=T1,
        updated_at=T1,
        expected_pointer_sha256=first_pointer.byte_sha256,
    )
    recovered = proposed.recover(
        transition="BOOTSTRAP",
        intent_id="bootstrap-proposed",
        receipt_id="bootstrap-recovered",
        recorded_at=T1,
    )
    assert recovered.outcome == "BOOTSTRAP_RECOVERED"
    assert recovered.selector_changed is False
    assert recovered.receipt.document[
        "observed_target_active_pointer_sha256"
    ] != recovered.receipt.document[
        "expected_target_active_pointer_sha256"
    ]

    old_workspace = workspace / "old"
    old_workspace.mkdir(mode=0o700)
    old = ResearchRuntimeControl(old_workspace)
    bootstrap = _bootstrap(old, intent_id="bootstrap-old")
    selector = old.current_selector()
    target, pointer, run, evidence = _seed_v4(
        old,
        old_workspace,
        run_id="v4-old",
        cutoff=T1,
    )
    with pytest.raises(SimulatedTransitionCrash):
        old.cutover(
            strategy_id=STRATEGY,
            v4_protocol_target_ref=target.reference,
            expected_v4_active_pointer_sha256=pointer.byte_sha256,
            expected_v4_run_ref=run.reference,
            expected_selector_sha256=selector.byte_sha256,
            recorded_at=T1,
            intent_id="cutover-old",
            receipt_id="unused",
            required_evidence_refs=[
                bootstrap.receipt.reference,
                *evidence,
            ],
            crash_before_cas=True,
        )
    aborted = old.recover(
        transition="CUTOVER",
        intent_id="cutover-old",
        receipt_id="cutover-aborted",
        recorded_at=T2,
    )
    assert aborted.outcome == "CUTOVER_ABORTED"
    assert old.current_selector().byte_sha256 == selector.byte_sha256


def test_recovery_third_selector_state_hard_stops(workspace: Path) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    old_selector = control.current_selector()
    target, pointer, run, evidence = _seed_v4(
        control,
        workspace,
        run_id="v4-third",
        cutoff=T1,
    )
    with pytest.raises(SimulatedTransitionCrash):
        control.cutover(
            strategy_id=STRATEGY,
            v4_protocol_target_ref=target.reference,
            expected_v4_active_pointer_sha256=pointer.byte_sha256,
            expected_v4_run_ref=run.reference,
            expected_selector_sha256=old_selector.byte_sha256,
            recorded_at=T1,
            intent_id="cutover-third",
            receipt_id="unused",
            required_evidence_refs=[
                bootstrap.receipt.reference,
                *evidence,
            ],
            crash_before_cas=True,
        )
    third = _artifact(
        str(SELECTOR_PATH),
        {
            "version": SELECTOR_VERSION,
            "selector_id": "default-protocol-selector",
            "status": "V15_DEFAULT",
            "protocol_target_ref": old_selector.document[
                "protocol_target_ref"
            ],
            "transition_intent_ref": {
                "version": (
                    "myquant.research-runtime.route-transition-intent.v1"
                ),
                "intent_id": "unrelated",
                "relative_path": (
                    "results/research_runtime_control/"
                    "bootstrap_intents/unrelated.json"
                ),
            },
            "updated_at": T2,
        },
        cutoff=T0,
    )
    control.store.replace_cas(
        SELECTOR_PATH,
        old_selector.byte_sha256,
        third.raw,
    )
    with pytest.raises(RuntimeControlThirdState):
        control.recover(
            transition="CUTOVER",
            intent_id="cutover-third",
            receipt_id="must-not-exist",
            recorded_at=T2,
        )
    assert not (
        workspace
        / "results/research_runtime_control/cutover_receipts/"
        "must-not-exist.json"
    ).exists()


def test_target_pointer_toctou_blocks_without_selector_write(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    selector = control.current_selector()
    target, expected_pointer, expected_run, expected_evidence = _seed_v4(
        control,
        workspace,
        run_id="v4-before",
        cutoff=T1,
    )
    _, replacement_pointer, _, _ = _seed_v4(
        control,
        workspace,
        run_id="v4-after",
        cutoff=T2,
    )
    pointer_path = workspace / expected_pointer.relative_path
    _owner_file(pointer_path, expected_pointer.raw)

    def replace_before_locks() -> None:
        _owner_file(pointer_path, replacement_pointer.raw)

    blocked = control.cutover(
        strategy_id=STRATEGY,
        v4_protocol_target_ref=target.reference,
        expected_v4_active_pointer_sha256=expected_pointer.byte_sha256,
        expected_v4_run_ref=expected_run.reference,
        expected_selector_sha256=selector.byte_sha256,
        recorded_at=T1,
        intent_id="cutover-toctou",
        receipt_id="cutover-toctou-receipt",
        required_evidence_refs=[
            bootstrap.receipt.reference,
            *expected_evidence,
        ],
        before_lock_hook=replace_before_locks,
    )
    assert blocked.outcome == "CUTOVER_CAS_BLOCKED"
    assert blocked.selector_changed is False
    assert control.current_selector().byte_sha256 == selector.byte_sha256
    assert blocked.receipt.document[
        "observed_target_active_pointer_sha256"
    ] == replacement_pointer.byte_sha256


def test_cutover_rejects_missing_eligibility_and_canary_evidence(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    selector = control.current_selector()
    target, pointer, run, _ = _seed_v4(
        control,
        workspace,
        run_id="v4-missing-evidence",
        cutoff=T1,
    )

    with pytest.raises(
        RuntimeControlError,
        match="requires formal, eligibility, and canary pointers",
    ):
        control.cutover(
            strategy_id=STRATEGY,
            v4_protocol_target_ref=target.reference,
            expected_v4_active_pointer_sha256=pointer.byte_sha256,
            expected_v4_run_ref=run.reference,
            expected_selector_sha256=selector.byte_sha256,
            recorded_at=T1,
            intent_id="cutover-missing-evidence",
            receipt_id="cutover-missing-evidence-receipt",
            required_evidence_refs=[bootstrap.receipt.reference],
        )
    assert control.current_selector().byte_sha256 == selector.byte_sha256


def test_cutover_rejects_wrong_formal_activation_receipt_version(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    selector = control.current_selector()
    target, pointer, run, evidence = _seed_v4(
        control,
        workspace,
        run_id="v4-wrong-receipt",
        cutoff=T1,
    )
    from quant_investor.v17_v4_contract import (
        canonical_resource_bytes,
        seal_semantic,
    )

    bogus_path = (
        "results/v17_v4_formal_research/strategies/"
        f"{STRATEGY}/receipts/bogus.json"
    )
    bogus_document = seal_semantic(
        {
            "version": "myquant.v17.v4.not-activation.v1",
            "receipt_id": "bogus-receipt",
            "strategy_id": STRATEGY,
            "cutoff": T1,
            "status": "FORMAL_ACTIVATED",
        }
    )
    bogus_raw = canonical_resource_bytes(bogus_document)
    bogus_ref = {
        "artifact_id": "bogus-receipt",
        "artifact_version": "myquant.v17.v4.not-activation.v1",
        "byte_sha256": hashlib.sha256(bogus_raw).hexdigest(),
        "cutoff": T1,
        "relative_path": bogus_path,
        "semantic_sha256": bogus_document["semantic_sha256"],
        "strategy_id": STRATEGY,
    }
    bad_pointer_payload = {
        key: value
        for key, value in pointer.document.items()
        if key != "semantic_sha256"
    }
    bad_pointer_payload["receipt_ref"] = bogus_ref
    bad_pointer = _artifact(
        pointer.relative_path,
        bad_pointer_payload,
        cutoff=T1,
    )
    _owner_file(workspace / bogus_path, bogus_raw)
    _owner_file(workspace / bad_pointer.relative_path, bad_pointer.raw)
    bad_evidence = [
        bad_pointer.reference
        if row["artifact_version"]
        == "myquant.v17.v4.formal-active-pointer.v1"
        else row
        for row in evidence
    ]

    with pytest.raises(RuntimeControlError, match="selector evidence is invalid"):
        control.cutover(
            strategy_id=STRATEGY,
            v4_protocol_target_ref=target.reference,
            expected_v4_active_pointer_sha256=bad_pointer.byte_sha256,
            expected_v4_run_ref=run.reference,
            expected_selector_sha256=selector.byte_sha256,
            recorded_at=T1,
            intent_id="cutover-wrong-receipt",
            receipt_id="cutover-wrong-receipt-receipt",
            required_evidence_refs=[
                bootstrap.receipt.reference,
                *bad_evidence,
            ],
        )
    assert control.current_selector().byte_sha256 == selector.byte_sha256


def test_cutover_rejects_activation_receipt_output_mismatch(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    selector = control.current_selector()
    target, pointer, run, evidence = _seed_v4(
        control,
        workspace,
        run_id="v4-output-mismatch",
        cutoff=T1,
    )
    receipt_ref = pointer.document["receipt_ref"]
    receipt_document = decode_reference(
        (workspace / receipt_ref["relative_path"]).read_bytes()
    )
    other_output_ref = _placeholder_ref(
        "other-formal-output",
        "myquant.v17.v4.formal-output.v1",
        (
            "results/v17_v4_formal_research/strategies/"
            f"{STRATEGY}/runs/other/formal.json"
        ),
        cutoff=T1,
    )
    bad_receipt_payload = {
        key: value
        for key, value in receipt_document.items()
        if key != "semantic_sha256"
    }
    bad_receipt_payload["formal_output_ref"] = other_output_ref
    bad_receipt_payload["evidence_refs"] = _ordered_refs(
        *[
            other_output_ref
            if row["artifact_id"] == run.reference["artifact_id"]
            else row
            for row in receipt_document["evidence_refs"]
        ]
    )
    bad_receipt = _artifact(
        receipt_ref["relative_path"],
        bad_receipt_payload,
        cutoff=T1,
    )
    bad_pointer_payload = {
        key: value
        for key, value in pointer.document.items()
        if key != "semantic_sha256"
    }
    bad_pointer_payload["receipt_ref"] = bad_receipt.reference
    bad_pointer = _artifact(
        pointer.relative_path,
        bad_pointer_payload,
        cutoff=T1,
    )
    _owner_file(workspace / bad_receipt.relative_path, bad_receipt.raw)
    _owner_file(workspace / bad_pointer.relative_path, bad_pointer.raw)
    bad_evidence = [
        bad_pointer.reference
        if row["artifact_version"]
        == "myquant.v17.v4.formal-active-pointer.v1"
        else row
        for row in evidence
    ]

    with pytest.raises(
        RuntimeControlError,
        match="does not bind the immutable output",
    ):
        control.cutover(
            strategy_id=STRATEGY,
            v4_protocol_target_ref=target.reference,
            expected_v4_active_pointer_sha256=bad_pointer.byte_sha256,
            expected_v4_run_ref=run.reference,
            expected_selector_sha256=selector.byte_sha256,
            recorded_at=T1,
            intent_id="cutover-output-mismatch",
            receipt_id="cutover-output-mismatch-receipt",
            required_evidence_refs=[
                bootstrap.receipt.reference,
                *bad_evidence,
            ],
        )
    assert control.current_selector().byte_sha256 == selector.byte_sha256


def test_cutover_rejects_unregistered_v4_formal_output(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    bootstrap = _bootstrap(control)
    selector = control.current_selector()
    target, pointer, _, evidence = _seed_v4(
        control,
        workspace,
        run_id="v4-closed-output",
        cutoff=T1,
    )
    from quant_investor.v17_v4_contract import (
        canonical_resource_bytes,
        seal_semantic,
    )

    evil_path = (
        "results/v17_v4_formal_research/strategies/"
        f"{STRATEGY}/runs/evil/formal.json"
    )
    evil_document = seal_semantic(
        {
            "authority": _v4_authority(),
            "cutoff": T1,
            "evidence_refs": [
                _placeholder_ref(
                    "evil-evidence",
                    "myquant.v17.v4.portfolio-output.v1",
                    "data/private/v17_v4_runs/evil/output.json",
                    cutoff=T1,
                )
            ],
            "output_id": "evil-output",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "terminal_state": "PUBLISHED_RESEARCH_ONLY",
            "version": "myquant.v17.v4.evil-output.v1",
        }
    )
    evil_raw = canonical_resource_bytes(evil_document)
    evil_ref = {
        "artifact_id": "evil-output",
        "artifact_version": "myquant.v17.v4.evil-output.v1",
        "byte_sha256": hashlib.sha256(evil_raw).hexdigest(),
        "cutoff": T1,
        "relative_path": evil_path,
        "semantic_sha256": evil_document["semantic_sha256"],
        "strategy_id": STRATEGY,
    }
    _owner_file(workspace / evil_path, evil_raw)

    with pytest.raises(RuntimeControlError, match="selector evidence is invalid"):
        control.cutover(
            strategy_id=STRATEGY,
            v4_protocol_target_ref=target.reference,
            expected_v4_active_pointer_sha256=pointer.byte_sha256,
            expected_v4_run_ref=evil_ref,
            expected_selector_sha256=selector.byte_sha256,
            recorded_at=T1,
            intent_id="cutover-evil-output",
            receipt_id="cutover-evil-output-receipt",
            required_evidence_refs=[
                bootstrap.receipt.reference,
                *evidence,
            ],
        )
    assert control.current_selector().byte_sha256 == selector.byte_sha256


def test_storage_cas_and_security_invariants(workspace: Path) -> None:
    store = ControlStore(workspace)
    store.initialize()
    pointer = "results/research_runtime_control/test-pointer.json"
    store.replace_cas(pointer, EMPTY_SHA256, b"old")
    with pytest.raises(ControlCASMismatch):
        store.replace_cas(pointer, "f" * 64, b"new")
    assert store.read(pointer) == b"old"

    with pytest.raises(ControlStorageSecurityError):
        store.write_exact_once(
            "results/research_runtime_control/../escape.json",
            b"x",
        )

    root = workspace / "results/research_runtime_control"
    real = root / "real"
    real.mkdir(mode=0o700)
    (root / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(ControlStorageSecurityError):
        store.read("results/research_runtime_control/alias/file.json")

    hardlink_source = "results/research_runtime_control/hardlink.json"
    store.write_exact_once(hardlink_source, b"one")
    os.link(
        workspace / hardlink_source,
        root / "hardlink-copy.json",
    )
    with pytest.raises(ControlStorageSecurityError):
        store.read(hardlink_source)

    (root / "Case").mkdir(mode=0o700)
    with pytest.raises(ControlStorageSecurityError):
        store.write_exact_once(
            "results/research_runtime_control/case/file.json",
            b"x",
        )

    insecure = "results/research_runtime_control/insecure.json"
    store.write_exact_once(insecure, b"x")
    (workspace / insecure).chmod(0o644)
    with pytest.raises(ControlStorageSecurityError):
        store.read(insecure)


def test_v15_publisher_mutates_only_its_pointer_namespace(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = ResearchRuntimeControl(workspace)
    run = control.create_v15_run(
        run_id="v15-publisher-scope",
        strategy_id=STRATEGY,
        cutoff=T0,
        evidence_refs=[],
    )
    writes: list[tuple[str, str]] = []
    exact_once = control.store.write_exact_once
    replace_cas = control.store.replace_cas

    def record_exact(path: object, raw: bytes):
        writes.append(("exact", str(path)))
        return exact_once(path, raw)

    def record_cas(path: object, expected: str, raw: bytes):
        writes.append(("cas", str(path)))
        return replace_cas(path, expected, raw)

    monkeypatch.setattr(control.store, "write_exact_once", record_exact)
    monkeypatch.setattr(control.store, "replace_cas", record_cas)
    control.v15_publisher.publish(
        strategy_id=STRATEGY,
        run_ref=run.reference,
        cutoff=T0,
        updated_at=T0,
        expected_pointer_sha256=EMPTY_SHA256,
    )
    assert writes == [
        (
            "exact",
            str(V15ActiveRunPublisher.lock_path(STRATEGY)),
        ),
        (
            "cas",
            str(V15ActiveRunPublisher.pointer_path(STRATEGY)),
        ),
    ]


def test_artifacts_are_canonical_owner_only_and_authority_free(
    workspace: Path,
) -> None:
    control = ResearchRuntimeControl(workspace)
    _bootstrap(control)
    root = workspace / "results/research_runtime_control"
    for path in root.rglob("*"):
        mode = path.stat().st_mode & 0o777
        assert mode == (0o700 if path.is_dir() else 0o600)
        if path.is_file() and path.stat().st_size:
            raw = path.read_bytes()
            document = decode(raw)
            assert hashlib.sha256(raw).hexdigest()
            if "authority_ceiling" in document:
                assert document["authority_ceiling"] == {
                    "broker": False,
                    "execution": False,
                    "order": False,
                    "trade": False,
                }
