"""Default Market adapter historical routing; native capture has its separate suite."""

from dataclasses import replace
import json

import pytest

from quant_investor.market import daily_components as components
from test_daily_evidence_historical_core import context
from test_cn_daily_components import _apis
from test_unified_factor_production_rollover import _write


@pytest.mark.parametrize("head", ["20260818", "20260821", "rebind", "20260820", "20260819"])
def test_historical_market_head_and_authority_routing(tmp_path, monkeypatch, head):
    ctx, rows, _ = context(tmp_path)
    pit = rows[0]
    pit["evidence"].update(
        reason_sets={},
        classification_evidence={},
        scope_path="fixture-scope",
        scope_sha256="a" * 64,
    )
    ctx = replace(ctx, prior_stage_results=(pit,))
    path = ctx.workspace_root / "data/parquet/cn/_latest.json"
    pointer = json.loads(path.read_bytes())
    pointer["latest_complete_trade_date"] = "20260820" if head == "rebind" else head
    if head == "rebind":
        pointer["coverage"]["pit_generation_id"] = "different-pit"
    _write(path, pointer)
    calls = []

    class ReachedCapture(RuntimeError):
        pass

    def provider():
        calls.append("provider")
        return object()

    def capture(**kwargs):
        calls.append(kwargs)
        raise ReachedCapture()

    monkeypatch.setattr(
        components, "_scope_reference", lambda *args: ("fixture-scope", "a" * 64, [])
    )
    monkeypatch.setattr(
        components, "_sealed_nontrading_evidence", lambda **kw: (kw["provider"], [], None)
    )
    native = components.build_default_components(
        workspace_root=ctx.workspace_root,
        apis=_apis(provider_factory=provider, market_capture=capture),
    )
    if head == "20260820":
        assert native.market(ctx)["status"] == "NO_ACTION"
        assert calls == []
    elif head == "20260819":
        # The real component boundary converts producer exceptions to a blocked
        # stage. The deliberate capture stop exposes its exact native arguments.
        assert native.market(ctx)["status"] == "BLOCKED"
        assert calls[0] == "provider"
        captured = calls[1]
        assert str(captured["target_authority_path"]) == ctx.historical_session_ref["path"]
        assert captured["expected_target_authority_sha256"] == ctx.historical_session_ref["sha256"]
        assert captured["target_trade_dates"] == ["20260820"]
        assert captured["parent_latest_complete_trade_date"] == "20260819"
        assert captured["same_target_rebind"] is False
    else:
        assert native.market(ctx)["status"] == "BLOCKED"
        assert calls == []
