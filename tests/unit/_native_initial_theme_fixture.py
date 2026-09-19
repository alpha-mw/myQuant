"""Installed actual core binding and Theme acquisition; source-complete EOD not asserted."""

from datetime import datetime, timezone
import json
from unittest.mock import patch


def probe_initial_theme(root, request_ref, handoff_ref, *, source_complete=False):
    from quant_investor.operations.theme_core_binding import bind_theme_acquisition
    from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
    from quant_investor.operations.theme_handoff_readback import read_theme_handoff
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.operations import theme_capture_stage, theme_acquisition
    from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
    from scripts.daily_materialization import materialize_daily_inputs
    from test_tushare_theme_capture_stable import FakeClient

    workspace = root / "factor-workspace"
    recovered = read_maintenance_handoff(workspace=str(workspace), handoff_ref=handoff_ref)
    handoff = recovered["handoff"]
    assert handoff["request_ref"]["sha256"] == request_ref["sha256"]
    binding = bind_theme_acquisition(
        workspace=str(workspace),
        request_ref=handoff["request_ref"],
        core_handoff_ref=handoff["core_handoff_ref"],
    )
    companies = binding["company_keyset"]
    assert len(companies) == 100
    assert binding["core_completed_at"] <= "2026-08-27T13:00:00Z"
    theme = TECHNOLOGY_THEME_IDS[0].split(":", 1)[1]
    rows = {("dc_index", "ALL"): [(theme, "20260827", "synthetic", "概念板块", "1")]}
    rows.update(
        {
            ("dc_member", company): [("20260827", theme, company, "synthetic")]
            for company in companies
        }
    )
    client = FakeClient(rows)

    class CaptureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 27, 13, 10, tzinfo=timezone.utc)

    print("START installed native Theme acquisition for actual Top100", flush=True)
    with (
        patch.object(theme_capture_stage, "datetime", CaptureClock),
        patch.object(theme_acquisition, "datetime", CaptureClock),
        patch.object(
            theme_capture_stage.native, "OfficialTushareHttpsClient", lambda **kwargs: client
        ),
    ):
        materialized = materialize_daily_inputs(
            workspace=str(workspace), handoff_ref=handoff_ref, _execute_theme=True
        )
    assert len(client.calls) == 101
    record = json.loads((workspace / materialized["materialization_ref"]["path"]).read_bytes())
    assert record["schema_version"] == "cn-daily-materialization.v2"
    journal = DailyJournal(str(workspace), "20260827")
    checked = read_theme_handoff(
        journal=journal,
        request_ref=handoff["request_ref"],
        core_handoff_ref=handoff["core_handoff_ref"],
        handoff_ref=record["theme_source_handoff_ref"],
    )
    assert checked["binding"] == binding
    result = {
        "synthetic": True,
        "full_daily_closure": False,
        "actual_native_core_binding": True,
        "company_count": len(companies),
        "company_set_sha256": binding["identity"]["company_set_sha256"],
        "provider_calls": len(client.calls),
        "theme_handoff_ref": record["theme_source_handoff_ref"],
        "materialization": materialized,
        "limitations": [
            "Industry/Exposure/Fundamental/Macro not source-complete",
            "synthetic transport and schedule",
        ],
    }
    if source_complete:
        from scripts.daily_completion import run_materialized_native_input

        closure = run_materialized_native_input(
            workspace=str(workspace),
            input_ref=materialized["native_inputs_ref"],
            resume=True,
            synthetic=True,
        )
        (root / "initial-source-complete-status.json").write_text(
            json.dumps(closure, indent=2) + "\n"
        )
        if closure["status"] != "COMPLETE" or closure["completion_ref"] is None:
            raise AssertionError("initial source-complete resumed EOD incomplete")
        result.update(
            full_daily_closure=True,
            completion_ref=closure["completion_ref"],
            limitations=[
                "synthetic transport and schedule",
                "intentional auxiliary interruption then resumed closure",
            ],
        )
    (root / "initial-native-theme-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS installed actual native Theme binding/materialization", flush=True)
    return result
