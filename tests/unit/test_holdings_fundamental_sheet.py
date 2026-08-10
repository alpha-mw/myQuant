from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from quant_investor.strategy_records.store import bootstrap_catalog


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


def _write_record(root: Path) -> tuple[Path, dict[str, str]]:
    run_dir = root / "CN" / "aggressive_tech_manufacturing" / "20260708_0933"
    run_dir.mkdir(parents=True)
    ledger_path = run_dir / "ledger_after_manual_switch.parquet"
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
    ).to_parquet(ledger_path, index=False)
    manifest_path = run_dir / "manual_execution_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "no_action_carry_forward",
                "execution_status": "no_action_carry_forward",
                "recorded_at": "2026-07-08 09:33:32 CST",
                "next_ledger_path": "ledger_after_manual_switch.parquet",
                "manual_execution_mode": "paper_only_local_manual_no_broker",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    strategy_root = root / "CN" / "aggressive_tech_manufacturing"
    relative_manifest = manifest_path.relative_to(strategy_root).as_posix()
    relative_ledger = ledger_path.relative_to(strategy_root).as_posix()
    return run_dir, {
        "manual_manifest_path": relative_manifest,
        "manual_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "ledger_path": relative_ledger,
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    }


def _install_catalog(mod, monkeypatch, root: Path, run_dir: Path, closure: dict[str, str]) -> None:
    strategy_root = root / "CN" / "aggressive_tech_manufacturing"
    monkeypatch.setattr(
        mod,
        "load_registered_catalog",
        lambda selected_root: (
            {"active_closure": dict(closure)},
            {"records": []},
        )
        if Path(selected_root) == strategy_root.absolute()
        else None,
    )
    monkeypatch.setattr(
        mod,
        "resolve_active_record_dirs",
        lambda selected_root: (run_dir.absolute(), run_dir.absolute()),
    )


def test_baseline_reads_real_registered_active_catalog(tmp_path):
    mod = _load_module()
    record_root = tmp_path / "records"
    strategy_root = record_root / "CN" / "aggressive_tech_manufacturing"
    run_dir, closure = _write_record(record_root)
    pnl_path = run_dir / "pnl_summary.csv"
    pnl_path.write_text("total_value_after\n1000000\n", encoding="utf-8")
    record = {
        "record_id": run_dir.name,
        "relative_path": run_dir.name,
        "state": "ONLINE",
        "evidence_status": "HASH_VERIFIED",
        "pnl_path": pnl_path.relative_to(strategy_root).as_posix(),
        "pnl_sha256": hashlib.sha256(pnl_path.read_bytes()).hexdigest(),
        "financial_state_sha256": "1" * 64,
        **closure,
    }
    bootstrap_catalog(
        strategy_root,
        records=[record],
        active_record_id=run_dir.name,
        generation_id="holdings-reader-test",
        published_at="2026-08-10T00:00:00Z",
    )

    baseline = mod.load_latest_holding_baseline(record_root)

    assert baseline.manifest_path == (run_dir / "manual_execution_manifest.json")
    assert baseline.ledger_path == (run_dir / "ledger_after_manual_switch.parquet")
    assert baseline.ledger["symbol"].tolist() == ["002008.SZ", "002384.SZ"]


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


def test_build_sheet_outputs_fields_verdicts_and_high_scrutiny(tmp_path, monkeypatch):
    mod = _load_module()
    record_root = tmp_path / "records"
    fundamentals_root = tmp_path / "fundamental_raw"
    disclosure_root = tmp_path / "disclosure_date"
    output_root = tmp_path / "audit"
    run_dir, closure = _write_record(record_root)
    _install_catalog(mod, monkeypatch, record_root, run_dir, closure)
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


def test_missing_disclosure_root_degrades_to_pending(tmp_path, monkeypatch):
    mod = _load_module()
    record_root = tmp_path / "records"
    fundamentals_root = tmp_path / "fundamental_raw"
    run_dir, closure = _write_record(record_root)
    _install_catalog(mod, monkeypatch, record_root, run_dir, closure)
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


def test_active_ledger_requires_declared_sha_and_fails_closed(tmp_path, monkeypatch):
    mod = _load_module()
    record_root = tmp_path / "records"
    run_dir, closure = _write_record(record_root)
    closure["ledger_sha256"] = "0" * 64
    _install_catalog(mod, monkeypatch, record_root, run_dir, closure)

    with pytest.raises(mod.RecordStoreError, match="ledger sha256 mismatch"):
        mod.load_latest_holding_baseline(record_root)


def test_active_ledger_has_no_csv_fallback(tmp_path, monkeypatch):
    mod = _load_module()
    record_root = tmp_path / "records"
    run_dir, closure = _write_record(record_root)
    csv_path = run_dir / "ledger_after_manual_switch.csv"
    csv_path.write_text("symbol,shares,avg_cost\n002008.SZ,100,10\n", encoding="utf-8")
    strategy_root = record_root / "CN" / "aggressive_tech_manufacturing"
    closure["ledger_path"] = csv_path.relative_to(strategy_root).as_posix()
    closure["ledger_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _install_catalog(mod, monkeypatch, record_root, run_dir, closure)

    with pytest.raises(
        mod.RecordStoreError,
        match="active holding ledger must be ledger_after_manual_switch.parquet",
    ):
        mod.load_latest_holding_baseline(record_root)


def test_active_ledger_rejects_symlink_even_when_target_sha_matches(
    tmp_path,
    monkeypatch,
):
    mod = _load_module()
    record_root = tmp_path / "records"
    run_dir, closure = _write_record(record_root)
    ledger_path = run_dir / "ledger_after_manual_switch.parquet"
    outside_path = tmp_path / "outside-ledger.parquet"
    ledger_path.replace(outside_path)
    ledger_path.symlink_to(outside_path)
    closure["ledger_sha256"] = hashlib.sha256(outside_path.read_bytes()).hexdigest()
    _install_catalog(mod, monkeypatch, record_root, run_dir, closure)

    with pytest.raises(mod.RecordStoreError, match="ledger path contains a symlink"):
        mod.load_latest_holding_baseline(record_root)


def test_active_ledger_rejects_record_root_escape(tmp_path, monkeypatch):
    mod = _load_module()
    record_root = tmp_path / "records"
    run_dir, closure = _write_record(record_root)
    closure["ledger_path"] = "../outside-ledger.parquet"
    _install_catalog(mod, monkeypatch, record_root, run_dir, closure)

    with pytest.raises(
        mod.RecordStoreError,
        match="ledger path must be record-root relative",
    ):
        mod.load_latest_holding_baseline(record_root)
