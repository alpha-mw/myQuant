"""Pure policy/custody comparisons; these are not native admission proofs."""

import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.core_timing import recorded_core_timing
from quant_investor.operations.daily_timing import recorded_daily_timing
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256
from quant_investor.operations.prospective_timing import (
    classify_daily_evidence,
    CORE_NODES,
    SOURCE_NODES,
)
from test_daily_evidence_daily_timing import context as journal_context


def context():
    data = journal_context()
    data["trade_date"] = "20260908"
    for row in data["nodes"].values():
        row["terminal"]["recovered"] = False
    obs_refs = {}
    for alias, node in [("LOW", "low_observation"), ("W80", "w80_observation")]:
        ref = {"path": alias + ".json", "sha256": "d" * 64}
        obs_refs[alias] = ref
        data["nodes"][node]["terminal"]["output_refs"] = {alias: ref}
    custody = recorded_daily_timing(**data)
    pointer = {"path": "factor-pointer.json", "sha256": "e" * 64}
    core = recorded_core_timing(
        pointer={"activated_at": "2026-09-08T07:00:00Z"},
        pointer_ref=pointer,
        generation={"created_at": "2020-01-01T00:00:00Z"},
        observations=[
            {
                "payload": {
                    "factor_alias": alias,
                    "registered_at": "2026-09-08T07:00:30Z",
                    "signal_date": "20260908",
                }
            }
            for alias in obs_refs
        ],
        observation_refs=obs_refs,
        terminal_refs={n: data["terminal_refs"][n] for n in CORE_NODES},
        terminals={n: data["nodes"][n]["terminal"] for n in CORE_NODES},
    )
    policy = dict(
        schema_version="cn-daily-prediction-policy.v1",
        trade_date="20260908",
        graph_sha256=GRAPH_SHA256,
        session_rule="CN_SSE_SZSE_OPEN",
        prediction_deadline="2026-09-08T07:01:00Z",
    )
    ref = {
        "path": "retained-policy.json",
        "sha256": hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
    }
    return dict(
        trade_date="20260908",
        handoff={
            "schema_version": "cn-daily-maintenance-handoff.v2",
            "trade_date": "20260908",
            "factor_pointer_ref": pointer,
            "prospective_policy_ref": ref,
            "sealed_at": "2026-09-08T07:00:50Z",
        },
        recipe={"policy_refs": {"prospective": ref}, "retrospective_ref": None},
        policy=policy,
        core_timing=core,
        node_custody=custody,
        source_times={n: [] for n in SOURCE_NODES},
        synthetic=False,
    )


def test_equality_is_contemporaneous_but_not_native_admission():
    value = classify_daily_evidence(**context())
    assert value["classification"] == "CONTEMPORANEOUS" and value["prospective"] is True
    assert value["native_replay_required"] is True
    assert set(value["proven_available_at"].values()) == {"2026-09-08T07:01:00Z"}


@pytest.mark.parametrize(
    "stamp,classification,bound",
    [
        ("2026-09-08T07:00:59.999999Z", "CONTEMPORANEOUS", "2026-09-08T07:01:00Z"),
        ("2026-09-08T07:01:00.000001Z", "LATE_REGISTERED", "2026-09-08T07:01:01Z"),
    ],
)
def test_fractional_source_bound_never_floors_across_deadline(stamp, classification, bound):
    args = context()
    args["source_times"]["fundamental"] = [
        {
            "source_ref": {"path": "source.json", "sha256": "f" * 64},
            "declared_available_at": stamp,
        }
    ]
    result = classify_daily_evidence(**args)
    assert result["classification"] == classification
    assert result["proven_available_at"]["fundamental"] == bound
    assert args["source_times"]["fundamental"][0]["declared_available_at"] == stamp


def test_historical_handoff_never_gains_prospective_status_without_recipe_marker():
    args = context()
    args["handoff"]["schema_version"] = "cn-daily-maintenance-handoff.v3"
    assert args["recipe"]["retrospective_ref"] is None
    result = classify_daily_evidence(**args)
    assert result["classification"] == "RETROSPECTIVE_RECOMPUTE"
    assert result["prospective"] is False


def test_late_portfolio_is_retrospective_even_if_delivery_meets_policy_deadline():
    args = context()
    args["portfolio_late"] = True
    result = classify_daily_evidence(**args)
    assert result["classification"] == "RETROSPECTIVE_RECOMPUTE"
    assert result["prospective"] is False
    args["portfolio_late"] = "false"
    with pytest.raises(ContractError):
        classify_daily_evidence(**args)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("policy_late", "LATE_REGISTERED"),
        ("source_late", "LATE_REGISTERED"),
        ("recovered", "UNKNOWN_LEGACY"),
        ("missing", "UNKNOWN_LEGACY"),
        ("legacy", "UNKNOWN_LEGACY"),
        ("synthetic", "RETROSPECTIVE_RECOMPUTE"),
        ("retrospective", "RETROSPECTIVE_RECOMPUTE"),
    ],
)
def test_noncontemporaneous_evidence_never_qualifies(case, expected):
    args = context()
    if case == "policy_late":
        args["handoff"]["sealed_at"] = "2026-09-08T07:01:01Z"
    elif case == "source_late":
        args["source_times"]["fundamental"] = [
            {
                "source_ref": {"path": "source.json", "sha256": "f" * 64},
                "declared_available_at": "2026-09-08T07:01:01Z",
            }
        ]
    elif case == "recovered":
        args["node_custody"]["nodes"]["store"]["recovered_at"] = "2026-09-08T07:01:00Z"
        args["node_custody"]["recovered_unknown"] = True
    elif case == "missing":
        args["policy"] = None
        args["handoff"]["prospective_policy_ref"] = None
        args["recipe"]["policy_refs"]["prospective"] = None
    elif case == "legacy":
        args["handoff"]["schema_version"] = "cn-daily-maintenance-handoff.v1"
    elif case == "synthetic":
        args["synthetic"] = True
    else:
        args["recipe"]["retrospective_ref"] = {"path": "retrospective.json", "sha256": "a" * 64}
    value = classify_daily_evidence(**args)
    assert value["classification"] == expected and value["prospective"] is False


@pytest.mark.parametrize(
    "case", ["policy_sha", "node_missing", "core_ref", "recovery_flag", "source_duplicate"]
)
def test_conflicting_evidence_rejects_instead_of_downgrading(case):
    args = context()
    if case == "policy_sha":
        args["policy"]["prediction_deadline"] = "2026-09-08T08:00:00Z"
    elif case == "node_missing":
        args["node_custody"]["nodes"].pop("macro")
    elif case == "core_ref":
        args["core_timing"]["node_custody"]["factor"]["terminal_ref"] = {
            "path": "wrong.json",
            "sha256": "a" * 64,
        }
    elif case == "recovery_flag":
        args["node_custody"]["recovered_unknown"] = True
    else:
        row = {"source_ref": {"path": "s.json", "sha256": "a" * 64}, "declared_available_at": None}
        args["source_times"]["macro"] = [row, row]
    with pytest.raises(ContractError):
        classify_daily_evidence(**args)
