from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "print_pipeline_state",
        ROOT / "scripts" / "print_pipeline_state.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pipeline_state_uses_effective_ledger_for_exposure(tmp_path):
    mod = _load_module()
    record_root = tmp_path / "records"
    run_dir = record_root / "20260707_1046"
    run_dir.mkdir(parents=True)
    (run_dir / "pnl_summary.csv").write_text(
        "record_time,total_value_after,cash_after,market_value_after\n"
        "2026-07-07,100,75,25\n",
        encoding="utf-8",
    )
    (run_dir / "ledger_after_manual_switch.csv").write_text(
        "symbol,name,shares,current_value,market_weight,stage_stop_price\n"
        "AAA.SZ,Alpha,100,25,1,8\n",
        encoding="utf-8",
    )
    (run_dir / "holdings_review.csv").write_text(
        "symbol,name,shares_before,current_value,market_weight,stage_stop_price,recommended_action\n"
        "AAA.SZ,Alpha,100,25,0.625,8,继续持有\n"
        "BBB.SZ,Beta,100,15,0.375,7,继续持有\n",
        encoding="utf-8",
    )
    (run_dir / "candidate_pool.csv").write_text("", encoding="utf-8")
    (run_dir / "market_snapshot.json").write_text(
        json.dumps({"candidate_generation_status": "empty"}, ensure_ascii=False),
        encoding="utf-8",
    )

    state = mod.build_state(record_root=record_root, regime_history=tmp_path / "missing_regime.jsonl")

    assert state["actual_total_exposure"] == 0.25
    assert state["exposure_components"]["holdings_review_exposure"] == 0.40
    assert state["exposure_components"]["difference_symbols_review_not_effective"] == ["BBB.SZ"]
    assert state["holdings"][0]["recommended_action"] == "继续持有"
    rendered = mod.render_state(state)
    assert "source=ledger_after_manual_switch.csv numerator=25.0 denominator=100.0" in rendered
