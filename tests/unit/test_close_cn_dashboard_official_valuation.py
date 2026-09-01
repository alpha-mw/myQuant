from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import close_cn_dashboard_official_valuation as valuation  # noqa: E402
from cn_dashboard_common import (  # noqa: E402
    DashboardInputError,
    canonical_json_bytes,
    validate_record,
)

STOCKS = (
    "002008.SZ",
    "002384.SZ",
    "002463.SZ",
    "002916.SZ",
    "601899.SH",
    "605358.SH",
    "688183.SH",
)
INDICES = valuation.INDEX_CODES
TRADE_DATE = "20260821"
SHA = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, dict, Path, dict]:
    project = tmp_path
    record_root = project / "results" / "records"
    source_dir = record_root / "20260820_1321"
    source_dir.mkdir(parents=True)
    source_rows: list[dict[str, object]] = []
    for index, symbol in enumerate(STOCKS, start=1):
        shares = index * 100
        price = 10.0 + index
        cost_basis = shares * (price - 1.0)
        source_rows.append(
            {
                "symbol": symbol,
                "name": f"company-{index}",
                "shares": shares,
                "avg_cost": price - 1.0,
                "cost_basis": cost_basis,
                "current_price": price,
                "current_value": shares * price,
                "unrealized_pnl": shares,
                "unrealized_pnl_pct": shares / cost_basis,
                "equity_sleeve_weight": 0.1,
                "market_weight": 0.1,
                "nav_weight": 0.05,
            }
        )
    source_ledger = pd.DataFrame(source_rows)
    ledger_path = source_dir / "ledger_after_manual_switch.parquet"
    source_ledger.to_parquet(ledger_path, index=False)
    ledger_sha = _sha(ledger_path)
    market_value = float(source_ledger["current_value"].sum())
    cash = 1_100_000.0
    total_value = cash + market_value
    financial_state = {
        "capital_cny": valuation.CAPITAL_CNY,
        "cash_after": cash,
        "market_value_after": market_value,
        "total_value_after": total_value,
        "portfolio_pnl_after": total_value - valuation.CAPITAL_CNY,
        "portfolio_return_after": total_value / valuation.CAPITAL_CNY - 1.0,
        "ledger_sha256": ledger_sha,
    }
    financial_state_sha = hashlib.sha256(canonical_json_bytes(financial_state)).hexdigest()
    manual = {
        "schema_version": "cn_aggressive_manual_execution.v3",
        "capital_cny": valuation.CAPITAL_CNY,
        "cash_after": cash,
        "market_value_after": market_value,
        "total_value_after": total_value,
        "portfolio_pnl_after": total_value - valuation.CAPITAL_CNY,
        "portfolio_return_after": total_value / valuation.CAPITAL_CNY - 1.0,
        "effective_manual_ledger_path": "ledger_after_manual_switch.parquet",
        "next_ledger_sha256": ledger_sha,
        "ledger_after_manual_switch_parquet_sha256": ledger_sha,
        "no_trade_performed": False,
        "execution_status": "owner_declared_manual_execution_applied",
        "applied_local_trades": [],
        "applied_owner_declared_trades": [
            {"symbol": "601899.SH", "shares": 5000, "trade_date": "20260819"}
        ],
        "rejected_or_pending_trades": [],
        "funding_events": [],
        "net_external_flow": 0.0,
        "excluded_external_flow": 0.0,
        "gross_trade_value": 100.0,
        "financial_state": financial_state,
        "financial_state_sha256": financial_state_sha,
    }
    manual_path = source_dir / "manual_execution_manifest.json"
    _write_json(manual_path, manual)
    pnl_path = source_dir / "pnl_summary.csv"
    _write_csv(
        pnl_path,
        [
            {
                "cash_after": cash,
                "market_value_after": market_value,
                "total_value_after": total_value,
                "portfolio_pnl_after": total_value - valuation.CAPITAL_CNY,
            }
        ],
    )
    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": source_dir.name,
        "action_taken_today": False,
        "trade_count": 1,
        "order_count": 0,
        "fill_count": 1,
        "manual_execution": manual,
    }
    manifest_path = source_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    closure = {
        "record_id": source_dir.name,
        "relative_path": source_dir.name,
        "manifest_path": f"{source_dir.name}/manifest.json",
        "manifest_sha256": _sha(manifest_path),
        "manual_manifest_path": f"{source_dir.name}/manual_execution_manifest.json",
        "manual_manifest_sha256": _sha(manual_path),
        "ledger_path": f"{source_dir.name}/ledger_after_manual_switch.parquet",
        "ledger_sha256": ledger_sha,
        "pnl_path": f"{source_dir.name}/pnl_summary.csv",
        "pnl_sha256": _sha(pnl_path),
        "financial_state_sha256": financial_state_sha,
    }

    snapshot_id = "20260821T000000Z"
    market_root = project / "data" / "parquet" / "cn"
    serving_root = market_root / "_snapshots" / snapshot_id / "serving" / "bars"
    snapshot_manifest_path = market_root / "_snapshots" / f"{snapshot_id}.json"
    stock_evidence: list[dict[str, object]] = []
    for index, symbol in enumerate(STOCKS, start=1):
        serving_path = serving_root / f"symbol={symbol}" / "bars.parquet"
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"symbol": symbol, "trade_date": "20260820", "close": 19.0 + index},
                {"symbol": symbol, "trade_date": TRADE_DATE, "close": 20.0 + index},
            ]
        ).to_parquet(serving_path, index=False)
        stock_evidence.append(
            {
                "symbol": symbol,
                "trade_date": TRADE_DATE,
                "close": 20.0 + index,
                "serving_parquet_path": serving_path.relative_to(project).as_posix(),
                "serving_parquet_sha256": _sha(serving_path),
            }
        )
    benchmark_path = project / "portfolio_dashboard" / "inputs" / "cn_index_benchmark.csv"
    benchmark_rows = [
        {
            "date": "2026-08-21",
            "ts_code": code,
            "close": 1000.0 + index,
            "source_system": "local.strict.capture",
            "value_date": "2026-08-21",
            "coverage": "exact_close",
        }
        for index, code in enumerate(INDICES)
    ]
    _write_csv(benchmark_path, benchmark_rows)
    market_pointer_path = market_root / "_latest.json"
    pointer = {
        "snapshot_id": snapshot_id,
        "latest_complete_trade_date": TRADE_DATE,
        "latest_trade_date": TRADE_DATE,
        "manifest_path": snapshot_manifest_path.relative_to(project).as_posix(),
    }
    _write_json(market_pointer_path, pointer)
    snapshot_manifest = {
        "market": "CN",
        "snapshot_id": snapshot_id,
        "latest_complete_trade_date": TRADE_DATE,
        "latest_trade_date": TRADE_DATE,
        "derived_serving_root": serving_root.relative_to(project).as_posix(),
    }
    _write_json(snapshot_manifest_path, snapshot_manifest)

    evidence = {
        "schema_version": valuation.EVIDENCE_SCHEMA,
        "market": "CN",
        "trade_date": TRADE_DATE,
        "market_pointer_path": market_pointer_path.relative_to(project).as_posix(),
        "market_pointer_sha256": _sha(market_pointer_path),
        "snapshot_manifest_path": snapshot_manifest_path.relative_to(project).as_posix(),
        "snapshot_manifest_sha256": _sha(snapshot_manifest_path),
        "snapshot_id": snapshot_id,
        "latest_complete_trade_date": TRADE_DATE,
        "benchmark_input_path": benchmark_path.relative_to(project).as_posix(),
        "benchmark_input_sha256": _sha(benchmark_path),
        "stocks": stock_evidence,
        "indices": [
            {
                "ts_code": code,
                "trade_date": TRADE_DATE,
                "close": 1000.0 + index,
                "benchmark_input_path": benchmark_path.relative_to(project).as_posix(),
                "benchmark_input_sha256": _sha(benchmark_path),
            }
            for index, code in enumerate(INDICES)
        ],
    }
    return project, source_dir, closure, market_pointer_path, evidence


def _build(
    tmp_path: Path,
    *,
    record_id: str = "20260821_1830",
    recorded_at_iso: str = "2026-08-21T18:30:00+08:00",
    receipt_created_at: str = "2026-08-21T18:00:00+08:00",
    publication_class: str = valuation.ORDINARY_PUBLICATION_CLASS,
    expected_valuation_date: str = "2026-08-21",
    expected_publication_date: str = "2026-08-21",
    publication_delay_reason: str = valuation.ORDINARY_PUBLICATION_REASON,
) -> tuple[Path, Path, dict, dict]:
    project, source_dir, closure, market_pointer_path, evidence = _source_fixture(tmp_path)
    staging_dir = project / "results" / "records" / record_id
    staging_dir.mkdir()
    summary = valuation.build_record(
        staging_dir=staging_dir,
        record_root=project / "results" / "records",
        source_dir=source_dir,
        registered_closure=closure,
        record_id=staging_dir.name,
        trade_date=TRADE_DATE,
        recorded_at_iso=recorded_at_iso,
        evidence=evidence,
        project_root=project,
        expected_market_pointer_sha256=_sha(market_pointer_path),
        source_pointer_sha256=SHA,
        source_catalog_generation_id="g20260821T000000-test",
        source_catalog_sha256=SHA,
        continuity_receipt_id="automation-20260821-daily-review-v1",
        continuity_receipt_sha256=SHA,
        continuity_receipt_created_at=receipt_created_at,
        continuity_checkpoint_digest=valuation.content_sha256(closure),
        evidence_input_sha256=hashlib.sha256(canonical_json_bytes(evidence) + b"\n").hexdigest(),
        publication_class=publication_class,
        expected_valuation_date=expected_valuation_date,
        expected_publication_date=expected_publication_date,
        publication_delay_reason=publication_delay_reason,
    )
    return project, staging_dir, summary, evidence


def test_build_record_uses_dynamic_seven_symbols_and_exact_five_files(
    tmp_path: Path,
) -> None:
    project, staging_dir, summary, evidence = _build(tmp_path)
    assert {path.name for path in staging_dir.iterdir()} == valuation.EXPECTED_RECORD_FILES
    assert summary["source_record"] == "20260820_1321"
    manifest = json.loads((staging_dir / "manifest.json").read_text())
    manual = json.loads((staging_dir / "manual_execution_manifest.json").read_text())
    assert manifest["data_snapshot"]["snapshot_id"] == evidence["snapshot_id"]
    assert manual["official_valuation"] is True
    assert manual["no_trade_performed"] is True
    assert manual["provider_quote_called"] is False
    assert manual["no_provider_quote_called"] is True
    assert manual["trade_count"] == manual["order_count"] == manual["fill_count"] == 0
    assert manual["source_pointer_sha256"] == SHA
    assert manual["source_catalog_generation_id"] == "g20260821T000000-test"
    assert manual["continuity_receipt_id"] == "automation-20260821-daily-review-v1"
    assert "tushare" not in json.dumps(manifest, ensure_ascii=False).lower()
    source = pd.read_parquet(
        project / "results" / "records" / "20260820_1321" / "ledger_after_manual_switch.parquet"
    ).sort_values("symbol")
    candidate = pd.read_parquet(staging_dir / "ledger_after_manual_switch.parquet").sort_values(
        "symbol"
    )
    assert candidate["symbol"].tolist() == list(source["symbol"])
    for column in ("shares", "avg_cost", "cost_basis"):
        assert candidate[column].tolist() == source[column].tolist()
    assert manual["cash_after"] == 1_100_000.0
    assert manual["cash_after"] + manual["market_value_after"] == pytest.approx(
        manual["total_value_after"]
    )


def test_offline_build_never_calls_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        valuation,
        "fetch_tushare_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider called")),
    )
    _build(tmp_path)


def test_built_record_passes_dashboard_record_validation(tmp_path: Path) -> None:
    project, staging_dir, _summary, _evidence = _build(tmp_path)
    closed = validate_record(
        staging_dir,
        project / "results" / "records",
        project,
    )
    assert closed["official_valuation"] is True
    assert closed["data_date"] == "2026-08-21"
    assert len(closed["positions"]) == 7


def test_batch_validation_can_resolve_an_immutable_staged_source(tmp_path: Path) -> None:
    project, staging_dir, _summary, _evidence = _build(tmp_path)
    record_root = project / "results" / "records"
    source_record = "20260820_1321"
    staged_source = project / "transaction" / source_record
    staged_source.parent.mkdir()
    (record_root / source_record).rename(staged_source)

    with pytest.raises(DashboardInputError, match="manifest_source_record_missing"):
        validate_record(staging_dir, record_root, project)

    closed = validate_record(
        staging_dir,
        record_root,
        project,
        source_record_dirs={source_record: staged_source},
    )
    assert closed["data_date"] == "2026-08-21"

    with pytest.raises(DashboardInputError, match="manifest_source_record_invalid"):
        validate_record(
            staging_dir,
            record_root,
            project,
            source_record_dirs={source_record: staged_source.parent},
        )


def test_late_publication_contract_is_identical_and_historical_only(
    tmp_path: Path,
) -> None:
    project, staging_dir, _summary, _evidence = _build(
        tmp_path,
        record_id="20260822_0930",
        recorded_at_iso="2026-08-22T09:30:47+08:00",
        receipt_created_at="2026-08-21T13:27:37Z",
        publication_class=valuation.LATE_PUBLICATION_CLASS,
        expected_valuation_date="2026-08-21",
        expected_publication_date="2026-08-22",
        publication_delay_reason=valuation.LATE_PUBLICATION_REASON,
    )
    manifest = json.loads((staging_dir / "manifest.json").read_text())
    manual = json.loads((staging_dir / "manual_execution_manifest.json").read_text())
    delay = manifest["publication_delay"]
    assert delay == manual["publication_delay"]
    assert delay["schema_id"] == "myquant.strategy_record_publication_delay.v1"
    assert delay["publication_class"] == "LATE_OFFICIAL_VALUATION_PUBLICATION"
    assert delay["expected_valuation_date"] == delay["evidence_date"] == "2026-08-21"
    assert delay["expected_publication_date"] == "2026-08-22"
    assert delay["source_record"] == "20260820_1321"
    assert delay["publication_delay_reason"] == "SHARED_CHECKOUT_SAFETY_GATE_DELAY"
    assert delay["historical_holdings_storage_authority"] is True
    assert delay["v17_mainline_authority"] is False
    assert delay["broker_order_trade_authority"] is False
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert '"sealed_at"' not in serialized
    assert '"published_at"' not in serialized
    closed = validate_record(
        staging_dir,
        project / "results" / "records",
        project,
    )
    assert closed["data_date"] == "2026-08-21"


def test_batch_publication_uses_real_second_plus_ordinal_without_loosening_late_contract(
    tmp_path: Path,
) -> None:
    project, staging_dir, _summary, _evidence = _build(
        tmp_path,
        record_id="20260901_090000-b01",
        recorded_at_iso="2026-09-01T09:00:00+08:00",
        receipt_created_at="2026-08-21T13:27:37Z",
        publication_class=valuation.BATCH_PUBLICATION_CLASS,
        expected_valuation_date="2026-08-21",
        expected_publication_date="2026-09-01",
        publication_delay_reason=valuation.BATCH_PUBLICATION_REASON,
    )
    manifest = json.loads((staging_dir / "manifest.json").read_text())
    manual = json.loads((staging_dir / "manual_execution_manifest.json").read_text())

    assert manifest["publication_class"] == valuation.BATCH_PUBLICATION_CLASS
    assert manual["publication_class"] == valuation.BATCH_PUBLICATION_CLASS
    assert "publication_delay" not in manifest
    assert "publication_delay" not in manual
    assert (
        validate_record(staging_dir, project / "results" / "records", project)["data_date"]
        == "2026-08-21"
    )


def test_batch_publication_rejects_synthetic_future_ordinal_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch record_id"):
        _build(
            tmp_path,
            record_id="20260901_090001-b01",
            recorded_at_iso="2026-09-01T09:00:00+08:00",
            receipt_created_at="2026-08-21T13:27:37Z",
            publication_class=valuation.BATCH_PUBLICATION_CLASS,
            expected_valuation_date="2026-08-21",
            expected_publication_date="2026-09-01",
            publication_delay_reason=valuation.BATCH_PUBLICATION_REASON,
        )


def _publication_kwargs() -> dict[str, object]:
    return {
        "publication_class": valuation.LATE_PUBLICATION_CLASS,
        "expected_valuation_date": "2026-08-21",
        "expected_publication_date": "2026-08-22",
        "publication_delay_reason": valuation.LATE_PUBLICATION_REASON,
        "trade_date": "20260821",
        "evidence_trade_date": "20260821",
        "source_record": "20260820_1321",
        "record_id": "20260822_0930",
        "recorded_at_iso": "2026-08-22T09:30:47+08:00",
        "receipt_id": "automation-20260821-daily-review-v1",
        "receipt_sha256": SHA,
        "receipt_created_at": "2026-08-21T13:27:37Z",
        "checkpoint_digest": SHA,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "expected_publication_date": "2026-08-21",
                "recorded_at_iso": "2026-08-21T09:30:47+08:00",
                "record_id": "20260821_0930",
                "receipt_created_at": "2026-08-21T00:00:00Z",
            },
            "fixed identity",
        ),
        (
            {
                "expected_publication_date": "2026-08-23",
                "recorded_at_iso": "2026-08-23T09:30:47+08:00",
                "record_id": "20260823_0930",
            },
            "fixed identity",
        ),
        (
            {
                "publication_class": valuation.ORDINARY_PUBLICATION_CLASS,
                "publication_delay_reason": valuation.ORDINARY_PUBLICATION_REASON,
            },
            "ordinary publication must be same-day",
        ),
        ({"recorded_at_iso": "2026-08-22T09:30:47"}, "timezone-aware"),
        ({"record_id": "20260822_0931"}, "Shanghai minute"),
        (
            {"receipt_created_at": "2026-08-22T10:00:00+08:00"},
            "receipt is later",
        ),
    ],
)
def test_publication_timing_rejects_invalid_paths(updates: dict[str, object], message: str) -> None:
    values = _publication_kwargs()
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        valuation._publication_contract(**values)


def test_strict_evidence_rejects_wrong_sha_date_inventory_and_nonpositive(
    tmp_path: Path,
) -> None:
    project, _source, _closure, market_pointer, evidence = _source_fixture(tmp_path)
    kwargs = {
        "project_root": project,
        "expected_symbols": set(STOCKS),
        "expected_trade_date": TRADE_DATE,
        "expected_market_pointer_sha256": _sha(market_pointer),
    }
    bad_sha = dict(evidence)
    bad_sha["market_pointer_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="market pointer SHA mismatch"):
        valuation.validate_strict_market_close_evidence(bad_sha, **kwargs)
    bad_date = dict(evidence)
    bad_date["latest_complete_trade_date"] = "20260820"
    with pytest.raises(ValueError, match="snapshot identity"):
        valuation.validate_strict_market_close_evidence(bad_date, **kwargs)
    bad_rows = dict(evidence)
    bad_rows["stocks"] = list(evidence["stocks"]) + [dict(evidence["stocks"][0])]
    with pytest.raises(ValueError, match="stock row count"):
        valuation.validate_strict_market_close_evidence(bad_rows, **kwargs)
    bad_close = dict(evidence)
    bad_close["stocks"] = [dict(row) for row in evidence["stocks"]]
    bad_close["stocks"][0]["close"] = 0
    with pytest.raises(ValueError, match="not positive"):
        valuation.validate_strict_market_close_evidence(bad_close, **kwargs)


def test_batch_evidence_accepts_newer_market_head_with_exact_historical_rows(
    tmp_path: Path,
) -> None:
    project, _source, _closure, market_pointer, evidence = _source_fixture(tmp_path)
    pointer = json.loads(market_pointer.read_text())
    manifest_path = project / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    pointer["latest_complete_trade_date"] = "20260824"
    pointer["latest_trade_date"] = "20260824"
    manifest["latest_complete_trade_date"] = "20260824"
    manifest["latest_trade_date"] = "20260824"
    _write_json(manifest_path, manifest)
    _write_json(market_pointer, pointer)
    evidence["latest_complete_trade_date"] = "20260824"
    evidence["market_pointer_sha256"] = _sha(market_pointer)
    evidence["snapshot_manifest_sha256"] = _sha(manifest_path)
    kwargs = {
        "project_root": project,
        "expected_symbols": set(STOCKS),
        "expected_trade_date": TRADE_DATE,
        "expected_market_pointer_sha256": _sha(market_pointer),
    }

    with pytest.raises(ValueError, match="snapshot identity"):
        valuation.validate_strict_market_close_evidence(evidence, **kwargs)
    validated = valuation.validate_strict_market_close_evidence(
        evidence,
        **kwargs,
        allow_historical_market_head=True,
    )
    assert validated["latest_complete_trade_date"] == TRADE_DATE
    assert validated["market_pointer_sha256"] == _sha(market_pointer)


def test_registered_source_rejects_pointer_and_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project, source_dir, closure, _pointer, _evidence = _source_fixture(tmp_path)
    record_root = source_dir.parent
    store = record_root / "_record_store"
    store.mkdir()
    pointer_path = store / "current.v1.json"
    pointer_path.write_bytes(b"registered-pointer\n")
    pointer_sha = _sha(pointer_path)
    pointer = {"active_record_id": source_dir.name, "active_closure": closure}
    monkeypatch.setattr(valuation, "load_registered_catalog", lambda _root: (pointer, {}))
    monkeypatch.setattr(
        valuation, "resolve_active_record_dirs", lambda *_args, **_kwargs: [source_dir]
    )
    with pytest.raises(ValueError, match="expected pointer SHA drift"):
        valuation.resolve_registered_source(record_root=record_root, expected_pointer_sha="0" * 64)
    resolved, observed_closure = valuation.resolve_registered_source(
        record_root=record_root, expected_pointer_sha=pointer_sha
    )
    assert resolved == source_dir.resolve()
    assert observed_closure == closure


def test_strict_evidence_schema_is_valid_json() -> None:
    schema_path = (
        ROOT
        / "portfolio_dashboard"
        / "schema"
        / "cn_dashboard_strict_market_close_evidence.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$id"] == valuation.EVIDENCE_SCHEMA
    assert schema["properties"]["stocks"]["maxItems"] == 7
    assert schema["properties"]["indices"]["maxItems"] == 3
