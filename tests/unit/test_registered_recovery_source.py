"""Configured original-source custody; native Store plan admission is caller-owned."""

import pytest

from test_registered_automatic_origin import fixture, active, publish
from _daily_preparation_fixture import snapshot
from quant_investor.operations.automatic_origin import read_automatic_origin
from quant_investor.operations.catchup_binding import BindingSources
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.registered_recovery_source import verify_configured_origin
from quant_investor.operations.source_slot_contract import paths
from quant_investor.system.errors import SystemStorageError
from scripts.registered_daily_event_sources import read_declaration


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "writer",
        "before_preparation",
        "missing_locator",
        "missing_capture",
        "missing_commitment",
    ],
)
def test_configured_recovery_keeps_source_calendar_distinct_from_core_calendar(
    tmp_path, monkeypatch, fault
):
    case = fixture(tmp_path, monkeypatch)
    with active(case):
        ref = publish(tmp_path, case)
    context = read_automatic_origin(workspace=str(tmp_path), reference=ref, synthetic=True)
    recipe = case["derived"]["recipe"]
    declaration = read_declaration(
        workspace=str(tmp_path), declaration_ref=recipe["registered_event_declaration_ref"]
    )["declaration"]
    # Minimal plan view isolates this helper; inspect_registered_recovery first
    # validates the complete native C1 plan/proof in its separate native test.
    plan = {
        "registered_event_declaration_ref": recipe["registered_event_declaration_ref"],
        "decision_baseline_pointer_ref": declaration["baseline_store_pointer_ref"],
        "transaction_planned_at": "2026-08-25T13:20:00Z",
        "requested_target": "2026-08-25",
        "preimages": {
            "calendar_receipt_sha256": "0" * 64,
            **{
                key: recipe["store_preimages"][name]["sha256"]
                for name, key in (
                    ("store_pointer_ref", "store_pointer_sha256"),
                    ("event_pointer_ref", "event_pointer_sha256"),
                    ("benchmark_pointer_ref", "benchmark_pointer_sha256"),
                )
            },
        },
    }
    selected = paths(case["config_ref"], "20260825", 2)
    if fault == "writer":
        plan["preimages"]["store_pointer_sha256"] = "a" * 64
    elif fault == "before_preparation":
        plan["transaction_planned_at"] = "2026-08-25T12:19:59Z"
    elif fault is not None:
        key = {
            "missing_locator": "locator",
            "missing_capture": "capture",
            "missing_commitment": "commitment",
        }[fault]
        (tmp_path / selected[key]).unlink()
    source = BindingSources(str(tmp_path))
    before = snapshot(tmp_path)
    if fault is None:
        result = verify_configured_origin(source=source, origin=context, native_plan=plan)
        assert result["preparation_commitment_ref"]["path"] == selected["commitment"]
        assert result["prepared_at"] == "2026-08-25T12:20:00Z"
        assert (
            plan["preimages"]["calendar_receipt_sha256"]
            != context["resolution"]["resolution"]["calendar_ref"]["sha256"]
        )
        result["source_context"].recheck()
        source.recheck()
    else:
        with pytest.raises((ContractError, SystemStorageError)):
            verify_configured_origin(source=source, origin=context, native_plan=plan)
    assert snapshot(tmp_path) == before
