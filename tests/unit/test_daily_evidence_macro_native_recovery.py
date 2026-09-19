"""Default maintenance adopts/replays native Macro preparation without provider rerun."""

from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import pytest

from quant_investor.market.daily_components import (
    build_default_components,
    registered_component_apis,
)
from quant_investor.market.daily_macro_layout import select_macro_layout
from quant_investor.macro import maintenance_transaction as native
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_shared_macro_fixture import build_shared_macro
from test_cn_daily_components import _context


@pytest.mark.parametrize(
    "preparation_mode,crash_phase",
    [("adopt", phase) for phase in [None, *native.PHASES]] + [("fresh", None)],
)
def test_default_component_adopts_native_preparation_then_replays_terminal(
    tmp_path, monkeypatch, preparation_mode, crash_phase
):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    fixture = NativeFactorInputs(tmp_path / "inputs", count=100)
    for offset in range(5):
        args = fixture.day(offset)
        day = args["as_of"]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        strict_market_from_factor_inputs(
            tmp_path,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
        if offset == 3:
            build_shared_macro(
                tmp_path,
                iso,
                calendar_through="2026-08-28" if preparation_mode == "fresh" else None,
            )
    market_path = tmp_path / "data/parquet/cn/_latest.json"
    market = json.loads(market_path.read_bytes())
    manifest = Path(market["manifest_path"])
    scope = tmp_path / "data/cn_universe/cn_index_components.json"
    pit_path = tmp_path / "data/parquet/cn/reference/stock_basic_membership_latest.json"
    prior = (
        {
            "stage": "MARKET",
            "evidence": {
                "pointer_path": str(market_path),
                "pointer_sha256": sha(market_path),
                "snapshot_manifest_path": str(manifest),
                "snapshot_manifest_sha256": sha(manifest),
            },
        },
        {
            "stage": "PIT",
            "evidence": {
                "scope_path": str(scope),
                "scope_sha256": sha(scope),
                "pit_binding": {
                    "discovery_pointer_path": str(pit_path),
                    "discovery_pointer_sha256": sha(pit_path),
                },
            },
        },
    )
    ctx = replace(_context(tmp_path, mode="execute", prior=prior), target_date=day)
    layout = select_macro_layout(ctx)
    if preparation_mode == "adopt":
        built = build_shared_macro(
            tmp_path, iso, transaction_identity=layout.transaction_id, prepare_only=True
        )
        assert Path(built["prepared"]["prepared_path"]) == layout.prepared_path
        assert select_macro_layout(ctx).state == "PREPARED"
    else:
        assert select_macro_layout(ctx).state == "FRESH"
    calls = []
    provider_calls = []
    apis = registered_component_apis()

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared adoption attempted provider preparation")

    def prepare(**kwargs):
        assert preparation_mode == "fresh"
        calls.append("prepare")

        def fetcher(url, issuer):
            provider_calls.append((url, issuer))
            return ("synthetic-official-coverage-" + issuer).encode(), iso + "T13:00:00Z"

        return apis.macro_prepare(
            **kwargs,
            fetcher=fetcher,
        )

    class SimulatedCrash(BaseException):
        pass

    def fail_at(phase):
        if phase == crash_phase:
            raise SimulatedCrash(phase)

    def commit(**kwargs):
        calls.append("commit")
        return apis.macro_commit(**kwargs, failure_injector=fail_at)

    def recover(**kwargs):
        calls.append("recover-forward" if kwargs["execute_forward"] else "recover-read")
        return apis.macro_recover(**kwargs)

    class SyntheticClock(datetime):
        tick = 0

        @classmethod
        def now(cls, tz=None):
            cls.tick += 1
            value = datetime.fromisoformat(iso + "T13:10:00+00:00") + timedelta(seconds=cls.tick)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(native, "datetime", SyntheticClock)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    components = build_default_components(
        workspace_root=tmp_path,
        apis=replace(apis, macro_prepare=prepare, macro_commit=commit, macro_recover=recover),
    )
    if crash_phase:
        with pytest.raises(SimulatedCrash, match=crash_phase):
            components.macro_release(ctx)
        assert select_macro_layout(ctx).state == "JOURNALED"
        assert not (ctx.attempt_root / "macro-research-source.json").exists()
        recovery_attempt = tmp_path / "recovery-attempt"
        recovery_attempt.mkdir(mode=0o700)
        ctx = replace(ctx, attempt_root=recovery_attempt, attempt_slot="recovered")
    first = components.macro_release(ctx)
    assert first["status"] == "READY", first
    assert first["evidence"].get("research_source_ref"), first
    assert first["write_performed"] is (crash_phase != "TERMINAL")
    next_attempt = tmp_path / "next-attempt"
    next_attempt.mkdir(mode=0o700)
    replayed = components.macro_release(
        replace(ctx, attempt_root=next_attempt, attempt_slot="next")
    )
    assert replayed["status"] == "READY", replayed
    assert replayed["evidence"].get("research_source_ref"), replayed
    assert replayed["write_performed"] is False
    expected_calls = (["prepare"] if preparation_mode == "fresh" else []) + ["commit"]
    if crash_phase:
        expected_calls.append("recover-read")
        if crash_phase != "TERMINAL":
            expected_calls.append("recover-forward")
    expected_calls.append("recover-read")
    assert calls == expected_calls
    assert [issuer for _, issuer in provider_calls] == (
        ["nbs_official", "pbc_official"] if preparation_mode == "fresh" else []
    )

    # Exact native stage descriptor through materialization; core controls remain
    # explicit fixtures, while Macro generation/transaction/readback stays native.
    from quant_investor.market.daily_maintenance import _stage_record
    from quant_investor.operations.research_materialization import publish_research_input
    from test_daily_evidence_research_materialization import context as controls
    from quant_investor.cli import unified
    from functools import partial

    journal, recovered, put = controls(tmp_path, trade_date=day)
    stage = {"stage": "MACRO_RELEASE", **first}
    recorded = _stage_record(ctx.attempt_root, "MACRO_RELEASE", stage)
    stage_ref = {
        "path": str(Path(recorded["path"]).relative_to(tmp_path)),
        "sha256": recorded["sha256"],
    }
    stage_doc = json.loads((tmp_path / stage_ref["path"]).read_bytes())
    recovered["recipe"]["research_sources"]["macro"] = {
        "mode": "MAINTENANCE_STAGE",
        "source_ref": None,
    }
    recovered["handoff"]["recipe_ref"] = put("recipe.json", recovered["recipe"])
    recovered["handoff"]["maintenance_core_ref"] = put(
        str(ctx.attempt_root.relative_to(tmp_path) / "core-completion.json"), {"fixture": "core"}
    )
    recovered["handoff_ref"] = put(recovered["handoff_ref"]["path"], recovered["handoff"])
    auxiliary = {
        "stages": {"macro": {"state": "RECORDED", "ref": stage_ref, "document": stage_doc}}
    }
    with journal.locked():
        materialized = publish_research_input(
            journal=journal, recovered=recovered, auxiliary=auxiliary
        )
    request = json.loads((tmp_path / materialized["research_request_ref"]["path"]).read_bytes())
    _, _, projection = unified._daily_macro_evidence(
        request["company_evidence"]["macro_risk"],
        partial(unified._daily_source_file, tmp_path),
        tmp_path,
        {"payload": {"signal_date": day}},
        {"as_of": iso + "T13:30:00Z"},
    )
    assert projection["target_date"] == day
    checked = publish_research_input(
        journal=journal, recovered=recovered, auxiliary=auxiliary, verify_only=True
    )
    assert checked["research_request_ref"] == materialized["research_request_ref"]
