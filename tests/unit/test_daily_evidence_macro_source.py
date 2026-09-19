"""Native Macro transaction evidence to source descriptor and original consumer."""

from functools import partial
import hashlib
import json
from types import SimpleNamespace

from quant_investor.cli import unified
from quant_investor.market.daily_macro_layout import MacroLayout
from quant_investor.market.daily_macro_source import attach_macro_source
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_shared_macro_fixture import build_shared_macro


def test_native_macro_source_publication_and_current_head_rejection(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source handoff attempted external network")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    fixture = NativeFactorInputs(tmp_path / "inputs", count=100)
    for offset in range(4):
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
    built = build_shared_macro(tmp_path, "2026-08-27")
    terminal = tmp_path / built["closure"]["journal_refs"][-1]["path"]
    identity = built["closure"]["transaction_id"]
    layout = MacroLayout(
        identity,
        terminal.parents[3],
        tmp_path / built["closure"]["prepared_ref"]["path"],
        terminal.parent.parent,
        identity,
        "JOURNALED",
    )
    attempt = tmp_path / "source-attempt"
    attempt.mkdir(mode=0o700)
    context = SimpleNamespace(
        mode="execute", workspace_root=tmp_path, attempt_root=attempt, target_date="20260827"
    )
    healthy = {"status": "READY", "write_performed": False, "blockers": [], "evidence": {}}
    result = attach_macro_source(context, layout, healthy)
    ref = result["evidence"]["research_source_ref"]
    raw = (tmp_path / ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    descriptor = json.loads(raw)
    assert descriptor["source"] == built["closure_ref"]
    _, _, projection = unified._daily_macro_evidence(
        descriptor,
        partial(unified._daily_source_file, tmp_path),
        tmp_path,
        {"payload": {"signal_date": "20260827"}},
        {"as_of": "2026-08-27T13:30:00Z"},
    )
    assert projection["target_date"] == "20260827"
    # An advanced/tampered head cannot turn historical health into a current source.
    market = tmp_path / "data/parquet/cn/_latest.json"
    market.write_bytes(market.read_bytes() + b"\n")
    context.attempt_root = tmp_path / "changed-head-attempt"
    context.attempt_root.mkdir(mode=0o700)
    failed = attach_macro_source(context, layout, healthy)
    assert failed["status"] == "READY" and failed["blockers"] == []
    assert "research_source_ref" not in failed["evidence"]
    assert failed["evidence"]["research_source_blockers"]
    assert list(context.attempt_root.iterdir()) == []
    assert not (tmp_path / "data/private/cn_daily_maintenance/MACRO_WRITE_VETO.json").exists()


def test_shadow_source_does_not_access_workspace(tmp_path):
    ctx = SimpleNamespace(mode="shadow")
    healthy = {"status": "READY", "evidence": {}}
    assert attach_macro_source(ctx, None, healthy) is healthy


def test_no_action_without_terminal_keeps_health_and_creates_nothing(tmp_path):
    from quant_investor.market.daily_macro_layout import select_macro_layout

    ctx = SimpleNamespace(
        mode="execute",
        workspace_root=tmp_path,
        run_root=tmp_path,
        target_date="20260827",
        attempt_root=tmp_path,
    )
    layout = select_macro_layout(ctx)
    healthy = {"status": "NO_ACTION", "blockers": [], "write_performed": False, "evidence": {}}
    result = attach_macro_source(ctx, layout, healthy)
    assert result["status"] == "NO_ACTION" and result["blockers"] == []
    assert "research_source_ref" not in result["evidence"]
    assert result["evidence"]["research_source_blockers"]
    assert list(tmp_path.iterdir()) == []
