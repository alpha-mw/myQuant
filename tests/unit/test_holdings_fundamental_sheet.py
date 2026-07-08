from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_holdings_fundamental_sheet",
        ROOT / "scripts" / "build_holdings_fundamental_sheet.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_record(root: Path) -> None:
    run_dir = root / "20260708_0933"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "002008.SZ",
                "name": "大族激光",
                "shares": 100,
                "avg_cost": 10.0,
                "current_price": 11.0,
                "current_value": 1100.0,
                "market_weight": 0.6,
            },
            {
                "symbol": "002384.SZ",
                "name": "东山精密",
                "shares": 200,
                "avg_cost": 20.0,
                "current_price": 21.0,
                "current_value": 4200.0,
                "market_weight": 0.4,
            },
        ]
    ).to_csv(run_dir / "ledger_after_manual_switch.csv", index=False, encoding="utf-8-sig")
    (run_dir / "manual_execution_manifest.json").write_text(
        json.dumps(
            {
                "status": "no_action_carry_forward",
                "execution_status": "no_action_carry_forward",
                "recorded_at": "2026-07-08 09:33:32 CST",
                "next_ledger_path": "ledger_after_manual_switch.csv",
                "manual_execution_mode": "paper_only_local_manual_no_broker",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_fundamentals(root: Path) -> None:
    indicator = root / "table=fina_indicator"
    indicator.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "002008.SZ",
                "end_date": "20260331",
                "ann_date": "20260430",
                "tr_yoy": 12.3,
                "netprofit_yoy": -4.5,
            },
            {
                "ts_code": "002384.SZ",
                "end_date": "20260331",
                "ann_date": "20260429",
                "tr_yoy": 21.0,
                "netprofit_yoy": 33.0,
            },
        ]
    ).to_parquet(indicator / "part.parquet", index=False)
    income = root / "table=income"
    income.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "002008.SZ",
                "end_date": "20260331",
                "ann_date": "20260430",
                "n_income": 1000.0,
            },
            {
                "ts_code": "002384.SZ",
                "end_date": "20260331",
                "ann_date": "20260429",
                "n_income": 2000.0,
            },
        ]
    ).to_parquet(income / "part.parquet", index=False)


def _write_disclosure(root: Path) -> None:
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "002384.SZ",
                "ann_date": "20260701",
                "end_date": "20260630",
                "pre_date": "20260718",
                "actual_date": "",
            }
        ]
    ).to_parquet(root / "part.parquet", index=False)


def test_build_sheet_outputs_fields_verdicts_and_high_scrutiny(tmp_path):
    mod = _load_module()
    record_root = tmp_path / "records"
    fundamentals_root = tmp_path / "fundamental_raw"
    disclosure_root = tmp_path / "disclosure_date"
    output_root = tmp_path / "audit"
    _write_record(record_root)
    _write_fundamentals(fundamentals_root)
    _write_disclosure(disclosure_root)

    payload = mod.build_holdings_fundamental_sheet(
        record_root=record_root,
        output_root=output_root,
        fundamentals_root=fundamentals_root,
        disclosure_root=disclosure_root,
        as_of="20260708",
        write=True,
    )

    rows = {row["symbol"]: row for row in payload["rows"]}
    assert rows["002008.SZ"]["revenue_yoy"] == 12.3
    assert rows["002008.SZ"]["net_profit_yoy"] == -4.5
    assert rows["002008.SZ"]["h1_disclosure_date"] == mod.DISCLOSURE_PENDING
    assert rows["002008.SZ"]["industry_verdict"] == "看不清"
    assert rows["002008.SZ"]["high_scrutiny_earnings_risk"] is True
    assert rows["002384.SZ"]["h1_disclosure_date"] == "2026-07-18"
    assert rows["002384.SZ"]["industry_verdict"] == "真订单真产能"
    assert rows["002384.SZ"]["high_scrutiny_earnings_risk"] is False
    assert payload["pending_disclosure_symbols"] == ["002008.SZ"]
    assert payload["high_scrutiny_symbols"] == ["002008.SZ"]

    out_dir = output_root / "20260708" / "fundamentals"
    verdicts = json.loads((out_dir / "industry_verdicts.json").read_text(encoding="utf-8"))
    verdict_map = {row["symbol"]: row["verdict"] for row in verdicts}
    assert verdict_map["002008.SZ"] == "看不清"
    assert verdict_map["002384.SZ"] == "真订单真产能"
    assert "high_scrutiny_symbols: 002008.SZ" in (out_dir / "index.md").read_text(encoding="utf-8")
    assert (out_dir / "002008.SZ.md").exists()


def test_missing_disclosure_root_degrades_to_pending(tmp_path):
    mod = _load_module()
    record_root = tmp_path / "records"
    fundamentals_root = tmp_path / "fundamental_raw"
    _write_record(record_root)
    _write_fundamentals(fundamentals_root)

    payload = mod.build_holdings_fundamental_sheet(
        record_root=record_root,
        output_root=tmp_path / "audit",
        fundamentals_root=fundamentals_root,
        disclosure_root=tmp_path / "missing_disclosure",
        as_of="20260708",
        write=False,
    )

    rows = {row["symbol"]: row for row in payload["rows"]}
    assert rows["002008.SZ"]["h1_disclosure_date"] == mod.DISCLOSURE_PENDING
    assert rows["002384.SZ"]["h1_disclosure_date"] == mod.DISCLOSURE_PENDING
    assert payload["pending_disclosure_symbols"] == ["002008.SZ", "002384.SZ"]
