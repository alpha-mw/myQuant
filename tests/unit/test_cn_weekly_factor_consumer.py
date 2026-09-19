"""Exact consumer integration; production generation validation has its own suite."""

import hashlib
import json
from datetime import datetime, timedelta

import pytest

import quant_investor.contracts as contracts
import quant_investor.contracts.core as contract_core
import quant_investor.intelligence._common as common
from quant_investor.factors.production_observation import build_factor_production_observation
from quant_investor.intelligence.storage import (
    DailyResearchPoolStore,
    approved_phase_a_policy,
    publish_phase_a_policy,
)
from scripts.cn_weekly_review_v2 import factor_coverage
from tests.unit.test_unified_daily_intelligence_storage import _rank
from tests.unit.test_unified_factor_production_observation import _inputs
from tests.unit.test_unified_factor_production_rollover import _maintenance_attempt


@pytest.fixture
def consumer_case(tmp_path, monkeypatch):
    # This fixture models original same-day publication, not a present-day backfill.
    monkeypatch.setattr(
        "quant_investor.intelligence.storage._pool_generated_at",
        lambda: "2026-08-24T13:30:00Z",
    )
    # Maintenance, observation, policy, rank and every pool leaf use real validators.
    # Only the separate Factor generation contract is abstracted to a fixed envelope.
    original = contracts.validate_artifact

    def generation_fixture_only(value, **kwargs):
        if isinstance(value, dict) and value.get("kind") == "factor.production_generation":
            return value
        return original(value, **kwargs)

    monkeypatch.setattr(contracts, "validate_artifact", generation_fixture_only)
    monkeypatch.setattr(contract_core, "validate_artifact", generation_fixture_only)
    monkeypatch.setattr(common, "validate_artifact", generation_fixture_only)
    workspace, maintenance, maintenance_sha = _maintenance_attempt(tmp_path)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = contracts.canonical_json_bytes(value)
        path.write_bytes(raw)
        path.chmod(0o600)
        return hashlib.sha256(raw).hexdigest()

    # Retarget only this new test's temporary fixture, leaving producer tests intact.
    target = "20260824"
    market_path = workspace / "data/parquet/cn/_latest.json"
    market = json.loads(market_path.read_bytes())
    market["latest_complete_trade_date"] = target
    market_sha = write(market_path, market)
    history_path = maintenance.parent / "history.json"
    history = json.loads(history_path.read_bytes())
    history.update(target_trade_date=target, effective_trade_date=target)
    history["canonical"]["latest_sha256"] = market_sha
    history_sha = write(history_path, history)
    receipt = json.loads(maintenance.read_bytes())
    state_path = maintenance.parent / "state.json"
    state = json.loads(state_path.read_bytes())
    state["target_date"] = target
    receipt["state_ref"]["sha256"] = write(state_path, state)
    # Produce a complete synthetic Aug24 Calendar through its native owner.
    # Editing an Aug20 receipt's target does not retarget its raw wire evidence.
    from test_daily_evidence_requested_session import capture

    calendar = capture("2026-08-24T20:20:00+08:00")
    raw_path = maintenance.parent / "close-session.raw.json"
    raw_path.write_bytes(calendar.raw_response_bytes)
    close = {
        **calendar.receipt,
        "raw_response_path": str(raw_path),
        "raw_response_sha256": hashlib.sha256(calendar.raw_response_bytes).hexdigest(),
    }
    close_path = maintenance.parent / "close-session-receipt.json"
    receipt["close_session_receipt_ref"]["sha256"] = write(close_path, close)
    receipt["target_date"] = target
    receipt["stage_results"][1]["evidence"]["pointer_sha256"] = market_sha
    receipt["stage_results"][2]["evidence"]["audit_sha256"] = history_sha
    maintenance_sha = write(maintenance, receipt)

    generation_id = "factor-production-generation-" + "1" * 64
    generation = {
        "artifact_id": generation_id,
        "kind": "factor.production_generation",
        "contract_sha256": "f" * 64,
        "semantic_sha256": "e" * 64,
        "created_at": "2026-08-24T12:30:00Z",
        "payload": {
            "factor_production_generation_id": generation_id,
            "as_of": "20260824",
            "low_signal_sha256": "b" * 64,
            "w80_signal_sha256": "d" * 64,
        },
    }
    generation_sha = write(
        workspace / "results/factors/generations" / generation_id / "generation.json", generation
    )
    pointer = {
        "authority_scope": "FACTOR_PRODUCTION",
        "factor_generation_id": generation_id,
        "factor_generation_sha256": generation_sha,
        "previous_pointer_sha256": "EMPTY",
        "activated_at": "2026-08-24T12:30:00Z",
        "os_actor": "uid:501",
    }
    pointer_sha = write(workspace / "results/factors/_active.json", pointer)
    inputs = _inputs()
    inputs.update(
        signal_date="20260824",
        factor_generation_id=generation_id,
        factor_generation_sha256=generation_sha,
        factor_pointer_sha256=pointer_sha,
    )
    observations = []
    for factor in inputs["factor_rows"]:
        observation = build_factor_production_observation(
            inputs=inputs, factor_row=factor, registered_at="2026-08-24T13:00:00Z"
        )
        write(
            workspace
            / "results/factors/observations/2026/08/24"
            / (factor["factor_alias"] + ".json"),
            observation,
        )
        observations.append(common.artifact_ref(observation))
    policy_result = publish_phase_a_policy(workspace)
    policy = approved_phase_a_policy()
    template = _rank(policy, signal_date="20260824")
    fields = {
        k: v
        for k, v in template["payload"].items()
        if k not in {"rank_id", "authority", "production", "research_only", "run_state"}
    }
    fields.update(
        factor_pointer_sha256=pointer_sha,
        factor_generation_ref=common.artifact_ref(generation),
        observation_refs=sorted(observations, key=lambda r: (r["kind"], r["artifact_id"])),
        common_symbol_count=inputs["factor_rows"][0]["symbol_count"],
        common_symbol_set_sha256=inputs["factor_rows"][0]["signal_symbol_set_sha256"],
    )
    identity = common.business_identity(
        kind="factor_research_rank",
        identity_inputs={"factor_pointer_sha256": pointer_sha, "policy_id": policy["artifact_id"]},
    )
    rank = common.build_artifact(
        kind="factor_research_rank",
        identity_field="rank_id",
        identity=identity,
        fields=fields,
        created_at="2026-08-24T13:30:00Z",
    )
    published = DailyResearchPoolStore(workspace).publish(
        rank=rank,
        expected_policy_sha256=policy_result["policy_sha256"],
        before_publish=lambda: None,
    )
    return (
        workspace,
        [{"path": str(maintenance), "sha256": maintenance_sha}],
        workspace / published["pool_root"],
        workspace / policy_result["policy_path"],
    )


def test_exact_maintenance_observations_policy_rank_and_pool_replay(consumer_case):
    workspace, refs, _, _ = consumer_case
    result = factor_coverage(workspace, ["2026-08-24"], refs)
    assert result["status"] == "FRESH", result
    row = result["dates"][0]
    assert row["maintenance_verified"] and row["observations_verified"] and row["top100_verified"]
    assert result["complete_same_day_dates"] == ["2026-08-24"]
    assert result["continuous_unattended_operation"] == "NOT_PROVEN"


@pytest.mark.parametrize("target", ["policy", "rank", "leaf", "receipt"])
def test_tampered_component_never_counts_complete(consumer_case, target):
    workspace, refs, pool, policy = consumer_case
    if target == "receipt":
        refs[0]["sha256"] = "0" * 64
    else:
        path = (
            policy
            if target == "policy"
            else pool
            / ("factor_research_rank.json" if target == "rank" else "selected_symbols.json")
        )
        value = json.loads(path.read_bytes())
        stamp = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00")) + timedelta(
            seconds=1
        )
        resealed = contracts.seal_artifact(
            value["kind"], value["payload"], created_at=stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        path.write_bytes(contracts.canonical_json_bytes(resealed))
    result = factor_coverage(workspace, ["2026-08-24"], refs)
    assert result["status"] == "BLOCKED", result
    assert not result.get("complete_same_day_dates")
