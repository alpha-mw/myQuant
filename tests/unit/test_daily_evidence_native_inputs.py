"""Native input decoding is read-only and preserves original Store preimages."""

import hashlib
import pytest
from _native_daily_store_fixture import NativeStoreFixture, write
from test_daily_evidence_dashboard_adapter import ref
from scripts.daily_production_store_adapter import prepare_store_plan
from scripts.daily_native_inputs import load_native_inputs
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError


def context(root):
    fixture = NativeStoreFixture(root)
    args = fixture.advance("2026-08-24")
    plan = prepare_store_plan(args)
    write(root / "fixtures/release.json", {"synthetic": True})
    value = {
        "schema_version": "cn-daily-native-inputs.v1",
        "trade_date": "20260824",
        "factor_pointer_sha256": "a" * 64,
        "release_ref": ref(root, "fixtures/release.json"),
        "research_request_ref": ref(root, "fixtures/release.json"),
        "store_plan_ref": {"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
        "store_policy_ref": {"path": args["policy_path"], "sha256": args["policy_sha"]},
        "retrospective_ref": None,
        "calendar_ref": ref(root, str(fixture.calendar_path.relative_to(root))),
        "market_snapshot_ref": ref(root, "data/parquet/cn/_snapshots/synthetic-20260824.json"),
        "benchmark_ref": ref(root, "portfolio_dashboard/inputs/cn_index_benchmark.csv"),
        "risk_free_ref": ref(root, "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
        "previous_trade_date": "20260821",
        "adjustment_market_refs": {},
        "publish_current_dashboard": False,
    }
    return value, args


def put(root, value):
    raw = canonical_json_bytes(value)
    path = root / "fixtures/native-inputs.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}


def test_native_input_load_preserves_preimages_and_writes_nothing(tmp_path):
    value, args = context(tmp_path)
    selected = put(tmp_path, value)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    day, inputs = load_native_inputs(workspace=str(tmp_path), input_ref=selected)
    assert day == "20260824"
    assert inputs.store_arguments == args
    assert inputs.event_pointer_sha256 == args["expected_event_pointer_sha"]
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ContractError, match="DOCUMENT_SHA_MISMATCH"):
        load_native_inputs(workspace=str(tmp_path), input_ref={**selected, "sha256": "0" * 64})


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        ("command", "arbitrary", "SCHEMA_INVALID"),
        ("publish_current_dashboard", 1, "MODE_INVALID"),
        ("trade_date", "20260823", "DATE_NOT_AUTHORIZED"),
        ("previous_trade_date", "20260820", "PREVIOUS_CALENDAR_MISMATCH"),
        (
            "store_policy_ref",
            {"path": "fixtures/policy.json", "sha256": "b" * 64},
            "SOURCE_BINDING_MISMATCH",
        ),
    ],
)
def test_bad_native_input_rejected(tmp_path, field, replacement, match):
    value, _ = context(tmp_path)
    value[field] = replacement
    with pytest.raises(ContractError, match=match):
        load_native_inputs(workspace=str(tmp_path), input_ref=put(tmp_path, value))


def test_run_entry_cannot_treat_missing_factor_as_completed(tmp_path):
    from scripts.daily_native_inputs import run_native_input

    value, args = context(tmp_path)
    selected = put(tmp_path, value)
    pointer = args["record_root"] / "_record_store/current.v1.json"
    before = pointer.read_bytes()
    result = run_native_input(workspace=str(tmp_path), input_ref=selected, resume=True)
    assert result["status"] in {"FAILED", "BLOCKED", "PARTIAL"}
    assert result["completion_ref"] is None
    assert result["nodes"]["calendar"]["state"] == "FAILED"
    assert result["nodes"]["store"]["state"] == "SKIPPED"
    assert pointer.read_bytes() == before


def test_v2_optional_failure_preserves_eod_input_and_v1_shape(tmp_path):
    from quant_investor.market.next_session_failure import publish_next_session_failure

    value, args = context(tmp_path)
    failure = publish_next_session_failure(
        workspace=str(tmp_path),
        eod_trade_date="20260824",
        phase="ACQUISITION",
        failure_code="ACQUISITION_FAILED",
    )
    v2 = {
        **value,
        "schema_version": "cn-daily-native-inputs.v2",
        "next_session_calendar_proof_ref": None,
        "next_session_calendar_failure_ref": failure,
    }
    _, loaded = load_native_inputs(workspace=str(tmp_path), input_ref=put(tmp_path, v2))
    assert loaded.store_arguments == args
    assert loaded.next_session_calendar_failure_ref == failure
    assert loaded.next_session_calendar_proof_ref is None
    with pytest.raises(ContractError, match="SCHEMA_INVALID"):
        load_native_inputs(
            workspace=str(tmp_path),
            input_ref=put(tmp_path, {**v2, "schema_version": "cn-daily-native-inputs.v1"}),
        )
    with pytest.raises(ContractError, match="SCHEMA_INVALID"):
        load_native_inputs(
            workspace=str(tmp_path), input_ref=put(tmp_path, {**v2, "schema_version": "unknown"})
        )
    _, legacy = load_native_inputs(workspace=str(tmp_path), input_ref=put(tmp_path, value))
    assert legacy.next_session_calendar_failure_ref is None


def test_v2_nonnull_bad_proof_cannot_downgrade_to_absence(tmp_path):
    value, _ = context(tmp_path)
    value.update(
        schema_version="cn-daily-native-inputs.v2",
        next_session_calendar_proof_ref={"path": "missing.json", "sha256": "a" * 64},
        next_session_calendar_failure_ref=None,
    )
    with pytest.raises(ContractError):
        load_native_inputs(workspace=str(tmp_path), input_ref=put(tmp_path, value))
