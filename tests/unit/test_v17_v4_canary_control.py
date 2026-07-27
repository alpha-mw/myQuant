from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_contract import seal_semantic
from quant_investor.v17_v4_runtime.canary_control import (
    CanaryCrash,
    CanaryService,
    build_canary_transition_intent,
    build_dual_run_comparison,
    build_historical_canary_policy,
    evaluate_operational_canary,
)
from quant_investor.v17_v4_runtime.formal_activation import artifact_ref
from quant_investor.v17_v4_runtime.source_storage import EMPTY_SHA256
from tests.unit.test_v17_v4_contract import _comparison

STRATEGY = "quant-first"
CUTOFF = "2026-07-27T08:00:00Z"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(
    artifact_id: str,
    version: str,
    path: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": _sha(f"bytes:{artifact_id}"),
        "cutoff": CUTOFF,
        "relative_path": path,
        "semantic_sha256": _sha(f"semantic:{artifact_id}"),
        "strategy_id": STRATEGY,
    }


def _month_session(index: int) -> str:
    year = 2021 + index // 12
    month = index % 12 + 1
    return date(year, month, 15).isoformat()


def _historical_pairs() -> list[
    tuple[dict[str, Any], dict[str, str]]
]:
    pairs: list[tuple[dict[str, Any], dict[str, str]]] = []
    for index in range(60):
        comparison = dict(_comparison(index=index))
        comparison["decision_session"] = _month_session(index)
        comparison.pop("semantic_sha256")
        comparison = seal_semantic(comparison)
        reference = artifact_ref(
            comparison,
            relative_path=(
                "results/v17_v4_canary/strategies/"
                f"{STRATEGY}/historical/{index:02d}.json"
            ),
        )
        pairs.append((comparison, reference))
    return pairs


def _operational_comparisons() -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for index, session in enumerate(
        (
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        )
    ):
        comparison = dict(_comparison(index=index))
        comparison["decision_session"] = session
        comparison["stage"] = "OPERATIONAL_CANARY"
        comparison.pop("semantic_sha256")
        comparisons.append(seal_semantic(comparison))
    return comparisons


def test_historical_policy_computes_exact_sixty_origin_bands() -> None:
    policy = build_historical_canary_policy(
        policy_id="historical-policy-1",
        strategy_id=STRATEGY,
        created_at=CUTOFF,
        comparison_pairs=_historical_pairs(),
    )

    assert policy["origin_count"] == 60
    assert policy["minimum_bands"] == {
        "rank_overlap": "0.75",
        "v15_top12_recall_in_v4_top24": "0.9",
    }
    assert policy["maximum_bands"] == {
        "cash_exposure_difference": "0.01",
        "cluster_exposure_difference": "0.02",
        "gross_exposure_difference": "0.01",
        "industry_exposure_difference": "0.02",
        "l1_portfolio_distance": "0.1",
        "max_common_name_target_difference": "0.01",
        "turnover_difference": "0.03",
    }


def test_comparison_builder_derives_exact_byte_comparability() -> None:
    source = _comparison(index=0)
    built = build_dual_run_comparison(
        comparison_id="built-comparison-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        decision_session="2026-07-27",
        stage="OPERATIONAL_CANARY",
        v15_run_ref=source["v15_run_ref"],
        v4_run_ref=source["v4_run_ref"],
        comparison_inputs=source["comparison_inputs"],
        latency_seconds=source["latency_seconds"],
        metrics=source["metrics"],
        risk_invariants=source["risk_invariants"],
        side_effect_counters=source["side_effect_counters"],
    )
    assert built["classification"] == "COMPARABLE"
    assert built["differing_refs"] == []

    mismatched = _comparison(comparable=False, index=1)
    built_mismatch = build_dual_run_comparison(
        comparison_id="built-comparison-2",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        decision_session="2026-07-28",
        stage="OPERATIONAL_CANARY",
        v15_run_ref=mismatched["v15_run_ref"],
        v4_run_ref=mismatched["v4_run_ref"],
        comparison_inputs=mismatched["comparison_inputs"],
        latency_seconds=mismatched["latency_seconds"],
        metrics=mismatched["metrics"],
        risk_invariants=mismatched["risk_invariants"],
        side_effect_counters=mismatched["side_effect_counters"],
    )
    assert built_mismatch["classification"] == "NON_COMPARABLE"
    assert len(built_mismatch["differing_refs"]) == 2


def test_operational_evaluation_is_five_of_five_and_zero_side_effects() -> None:
    policy = build_historical_canary_policy(
        policy_id="historical-policy-1",
        strategy_id=STRATEGY,
        created_at=CUTOFF,
        comparison_pairs=_historical_pairs(),
    )
    thresholds, counters = evaluate_operational_canary(
        policy,
        _operational_comparisons(),
    )

    assert {row["status"] for row in thresholds} == {"PASS"}
    assert len(counters) == 14
    assert set(counters.values()) == {0}


def test_replay_policy_and_forward_comparison_publish_only_to_canary_root(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    service = CanaryService(tmp_path, repo_root=tmp_path)
    pairs = _historical_pairs()
    policy, policy_ref = service.publish_historical_policy(
        policy_id="historical-policy-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        comparisons=[document for document, _reference in pairs],
    )
    operational_ref = service.publish_operational_comparison(
        _operational_comparisons()[0]
    )

    assert policy_ref["artifact_id"] == policy["policy_id"]
    assert operational_ref["artifact_id"] == "comparison-00"
    assert (
        tmp_path / str(policy_ref["relative_path"])
    ).is_file()
    assert (
        tmp_path / str(operational_ref["relative_path"])
    ).is_file()
    assert not (tmp_path / "results/research_runtime_control").exists()


def _start_intent(expected: str = EMPTY_SHA256) -> dict[str, Any]:
    return build_canary_transition_intent(
        intent_id="canary-start-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        transition="START",
        expected_pointer_sha256=expected,
        eligibility_pointer_ref=_ref(
            "eligibility-pointer-1",
            "myquant.v17.v4.default-eligible-pointer.v1",
            (
                "results/v17_v4_formal_research/strategies/"
                f"{STRATEGY}/eligibility/_active.json"
            ),
        ),
        historical_canary_policy_ref=_ref(
            "historical-policy-1",
            "myquant.v17.v4.historical-canary-policy.v1",
            (
                "results/v17_v4_canary/strategies/"
                f"{STRATEGY}/policies/historical-policy-1.json"
            ),
        ),
        v15_protocol_target_ref=_ref(
            "v15-target-1",
            "myquant.research-runtime.protocol-target.v1",
            "results/research_runtime_control/protocol_targets/v15/target.json",
        ),
        v15_active_run_pointer_ref=_ref(
            "v15-active-1",
            "myquant.research-runtime.active-run-pointer.v1",
            (
                "results/research_runtime_control/"
                f"active_runs/v15/{STRATEGY}.json"
            ),
        ),
        session_window={
            "start_session": "2026-07-27",
            "end_session": "2026-07-31",
            "required_session_count": 5,
        },
        paired_run_ids=["paired-run-1"],
    )


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CanaryService:
    tmp_path.chmod(0o700)
    service = CanaryService(tmp_path, repo_root=tmp_path)
    monkeypatch.setattr(service, "_revalidate_intent", lambda _intent: None)
    return service


def _complete_intent(expected: str) -> dict[str, Any]:
    base = _start_intent()
    comparison_refs = [
        _ref(
            f"operational-comparison-{index}",
            "myquant.v17.v4.dual-run-comparison.v1",
            (
                "results/v17_v4_canary/strategies/"
                f"{STRATEGY}/operational/{index}/comparison.json"
            ),
        )
        for index in range(5)
    ]
    counters = {
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
    }
    return build_canary_transition_intent(
        intent_id="canary-complete-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        transition="COMPLETE",
        expected_pointer_sha256=expected,
        eligibility_pointer_ref=base["eligibility_pointer_ref"],
        historical_canary_policy_ref=base[
            "historical_canary_policy_ref"
        ],
        v15_protocol_target_ref=base["v15_protocol_target_ref"],
        v15_active_run_pointer_ref=base[
            "v15_active_run_pointer_ref"
        ],
        session_window=base["session_window"],
        paired_run_ids=[f"paired-run-{index}" for index in range(1, 6)],
        comparison_refs=comparison_refs,
        completed_sessions=[
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        ],
        threshold_results=[
            {
                "observed": "1",
                "status": "PASS",
                "threshold_id": threshold_id,
            }
            for threshold_id in (
                "five-of-five",
                "historical-bands",
                "latency",
                "risk-invariants",
                "side-effects",
            )
        ],
        side_effect_counters=counters,
    )


def test_canary_start_is_crash_safe_and_never_writes_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    result = service.transition(_start_intent())
    state = service.resolve(STRATEGY)

    assert result.status == state.status == "CANARY_STARTED"
    assert state.pointer is not None
    assert state.pointer["state"] == "PENDING_COMPLETION"
    assert state.pointer["authority"]["research_runtime_default"] is False
    assert not (tmp_path / "results/research_runtime_control").exists()


def test_canary_complete_requires_and_replaces_exact_started_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    started = service.transition(_start_intent())
    completed = service.transition(
        _complete_intent(str(started.pointer_ref["byte_sha256"]))
    )
    state = service.resolve(STRATEGY)

    assert completed.status == state.status == "CANARY_COMPLETED"
    assert state.intent is not None
    assert state.intent["transition"] == "COMPLETE"
    assert completed.pointer_ref != started.pointer_ref


@pytest.mark.parametrize("boundary", ["intent", "cas", "readback", "completion"])
def test_canary_start_recovers_every_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    service = _service(tmp_path, monkeypatch)
    with pytest.raises(CanaryCrash):
        service.transition(_start_intent(), crash_after=boundary)

    state = service.resolve(STRATEGY)
    if boundary == "intent":
        assert state.status == "DEFAULT_ELIGIBLE"
    elif boundary in {"cas", "readback"}:
        assert state.status == "PENDING_COMPLETION"
    else:
        assert state.status == "CANARY_STARTED"

    recovered = service.transition(_start_intent())
    assert recovered.status == "CANARY_STARTED"
    assert recovered.recovered is (boundary != "intent")
