from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_investor.strategy_records.store import (
    content_sha256 as store_content_sha256,
)
from scripts import cn_dashboard_v2 as v2

SYMBOL = "000001.SZ"
RECORD_ID = "20260814_1200"
GENERATION_DATE = date(2026, 8, 18)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


@dataclass
class _Result:
    frame: pd.DataFrame
    issues: list[Any]


class _Reader:
    def __init__(
        self,
        *,
        root: Path,
        exact_close: float | None = 110.0,
        suspended: bool = False,
    ) -> None:
        self.root = root
        self.exact_close = exact_close
        self.suspended = suspended
        self.pointer = root / "data/parquet/cn/_latest.json"
        self.manifest = root / "data/parquet/cn/_snapshots/snapshot.json"
        self.serving = (
            root
            / "data/parquet/cn/_snapshots/snapshot/serving/bars"
            / f"symbol={SYMBOL}/bars.parquet"
        )
        _write(self.serving, b"fixture-serving-parquet")
        snapshot = self.snapshot()
        pointer = {
            "snapshot_id": snapshot["snapshot_id"],
            "latest_complete_trade_date": snapshot["latest_complete_trade_date"],
            "latest_trade_date": snapshot["latest_trade_date"],
            "manifest_path": self.manifest.relative_to(root).as_posix(),
            "coverage": snapshot["coverage"],
        }
        manifest = {
            "snapshot_id": snapshot["snapshot_id"],
            "latest_complete_trade_date": snapshot["latest_complete_trade_date"],
            "latest_trade_date": snapshot["latest_trade_date"],
            "coverage": snapshot["coverage"],
        }
        _write(
            self.pointer,
            (json.dumps(pointer, sort_keys=True) + "\n").encode(),
        )
        _write(
            self.manifest,
            (json.dumps(manifest, sort_keys=True) + "\n").encode(),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "healthy": True,
            "mode_policy": "strict",
            "snapshot_id": "snapshot",
            "latest_complete_trade_date": "20260817",
            "latest_trade_date": "20260817",
            "latest_pointer_path": str(self.pointer),
            "manifest_path": str(self.manifest),
            "coverage": {
                "coverage_schema_version": "cn-full-a-coverage.v4",
                "complete": True,
                "classification_sets_disjoint": True,
                "latest_complete_trade_date": "20260817",
                "suspended_symbols": [SYMBOL] if self.suspended else [],
                "true_missing_symbols": [],
            },
        }

    def resolve_symbol_path(self, symbol: str, **_: Any) -> Path:
        assert symbol == SYMBOL
        return self.serving

    def read_symbol_frame(
        self,
        symbol: str,
        *,
        start_date: str = "",
        end_date: str = "",
        **_: Any,
    ) -> _Result:
        assert symbol == SYMBOL
        if start_date == "20260817":
            rows = (
                []
                if self.exact_close is None
                else [
                    {
                        "symbol": SYMBOL,
                        "trade_date": "20260817",
                        "close": self.exact_close,
                    }
                ]
            )
        else:
            rows = [
                {
                    "symbol": SYMBOL,
                    "trade_date": "20260814",
                    "close": 105.0,
                }
            ]
        return _Result(pd.DataFrame(rows), [])


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_receipt: bool = True,
    exact_close: float | None = 110.0,
    suspended: bool = False,
) -> tuple[dict[str, Any], Path, Path, _Reader, dict[str, Any]]:
    root = tmp_path
    record_root = root / "results/strategy_records/CN/aggressive_tech_manufacturing"
    record_dir = record_root / RECORD_ID
    record_dir.mkdir(parents=True)
    ledger = pd.DataFrame(
        [
            {
                "symbol": SYMBOL,
                "name": "示例科技",
                "shares": 1000.0,
                "avg_cost": 100.0,
                "cost_basis": 100000.0,
            }
        ]
    )
    ledger_path = record_dir / "ledger_after_manual_switch.parquet"
    ledger.to_parquet(ledger_path, index=False)
    other_files = {
        "manifest": record_dir / "manifest.json",
        "manual_manifest": record_dir / "manual_execution_manifest.json",
        "pnl": record_dir / "pnl_summary.csv",
    }
    _write(other_files["manifest"], b'{"manifest":1}\n')
    _write(other_files["manual_manifest"], b'{"manual":1}\n')
    _write(other_files["pnl"], b"metric,value\nnav,1000000\n")
    closure = {
        "record_id": RECORD_ID,
        "relative_path": RECORD_ID,
        "inventory_sha256": "1" * 64,
        "total_bytes": 1,
        "file_count": 4,
        "manifest_path": f"{RECORD_ID}/manifest.json",
        "manifest_sha256": _sha(other_files["manifest"].read_bytes()),
        "manual_manifest_path": (f"{RECORD_ID}/manual_execution_manifest.json"),
        "manual_manifest_sha256": _sha(other_files["manual_manifest"].read_bytes()),
        "ledger_path": (f"{RECORD_ID}/ledger_after_manual_switch.parquet"),
        "ledger_sha256": _sha(ledger_path.read_bytes()),
        "pnl_path": f"{RECORD_ID}/pnl_summary.csv",
        "pnl_sha256": _sha(other_files["pnl"].read_bytes()),
        "financial_state_sha256": "2" * 64,
    }
    receipt = {
        "schema_id": "myquant.strategy_record_no_action_receipt.v1",
        "receipt_id": "automation-20260818-daily-review-v1",
        "created_at": "2026-08-18T01:30:00Z",
        "status": "NO_ACTION",
        "reason": "daily-review-no-change",
        "active_record_id": RECORD_ID,
        "active_checkpoint": closure,
        "payload_copied": False,
        "v17_mainline_authority": False,
        "broker_order_trade_authority": False,
    }
    receipt["content_sha256"] = store_content_sha256(receipt)
    catalog_relative = "_record_store/catalogs/g-fixture/catalog.v3.json"
    catalog = {
        "schema_id": "myquant.strategy_record_catalog.v3",
        "generation_id": "g-fixture",
        "active_record_id": RECORD_ID,
        "records": [
            {
                "record_id": RECORD_ID,
                "sealed_at": "2026-08-14T04:00:00Z",
            }
        ],
        "lineage_index": [
            {
                "record_id": RECORD_ID,
                "publication_class": "OFFICIAL_FINANCIAL_STATE",
                "valuation_date": "2026-08-14",
            }
        ],
        "receipts": [receipt] if include_receipt else [],
    }
    catalog_path = record_root / catalog_relative
    _write(
        catalog_path,
        (json.dumps(catalog, sort_keys=True) + "\n").encode("utf-8"),
    )
    pointer = {
        "generation_id": "g-fixture",
        "catalog_path": catalog_relative,
        "catalog_sha256": _sha(catalog_path.read_bytes()),
        "active_record_id": RECORD_ID,
        "active_closure": closure,
    }
    pointer_path = record_root / "_record_store/current.v1.json"
    _write(
        pointer_path,
        (json.dumps(pointer, sort_keys=True) + "\n").encode("utf-8"),
    )
    monkeypatch.setattr(v2, "load_registered_catalog", lambda _: (pointer, catalog))
    v1_bundle = {
        "schema_version": "cn_aggressive_dashboard.v1",
        "generated_at": "2026-08-18T09:00:00+08:00",
        "status": "PARTIAL",
        "market": "CN",
        "strategy_label": "aggressive_tech_manufacturing",
        "read_only": True,
        "blockers": [],
        "warnings": ["legacy_history_caveat"],
        "latest_valid_record": RECORD_ID,
        "latest_data_date": "2026-08-14",
        "current_evidence": {
            "manifest_sha256": closure["manifest_sha256"],
            "manual_manifest_sha256": closure["manual_manifest_sha256"],
            "ledger_sha256": closure["ledger_sha256"],
            "pnl_sha256": closure["pnl_sha256"],
            "financial_state_sha256": closure["financial_state_sha256"],
        },
        "positions": [
            {
                "symbol": SYMBOL,
                "name": "示例科技",
                "shares": 1000.0,
                "avg_cost": 100.0,
                "cost_basis": 100000.0,
            }
        ],
        "portfolio": {
            "cash": 900000.0,
            "performance_initial_capital": 1000000.0,
            "excluded_external_flow": 0.0,
            "adjusted_total_value": 1000000.0,
            "return_method": ("initial_capital_return_excluding_external_flows"),
            "performance_end_date": "2026-08-14",
            "performance_points": [
                {"adjusted_total_value": 1000000.0},
                {"adjusted_total_value": 1000000.0},
            ],
        },
        "history": {
            "evidence_status": "CANONICAL_PERFORMANCE_CLOSURE",
            "rejected_record_count": 0,
        },
        "benchmarks": [
            {"id": "CSI300", "end_date": "2026-08-14"},
            {"id": "STAR50", "end_date": "2026-08-14"},
            {"id": "CHINEXT", "end_date": "2026-08-14"},
        ],
    }
    v1_bundle["content_sha256"] = v2.content_sha256(v1_bundle)
    v1_path = root / "portfolio_dashboard/private/generated/" "cn_aggressive_dashboard.v1.json"
    _write(
        v1_path,
        (json.dumps(v1_bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    reader = _Reader(root=root, exact_close=exact_close, suspended=suspended)
    return v1_bundle, v1_path, record_root, reader, catalog


def _build(
    v1_bundle: dict[str, Any],
    v1_path: Path,
    record_root: Path,
    reader: _Reader,
    **kwargs: Any,
) -> dict[str, Any]:
    return v2.build_v2_bundle(
        project_root=v1_path.parents[3],
        v1_bundle=v1_bundle,
        v1_json_path=v1_path,
        record_root=record_root,
        generation_local_date=GENERATION_DATE,
        generated_at="2026-08-18T10:00:00+08:00",
        publication_attempt_id="attempt-1",
        market_reader=reader,
        **kwargs,
    )


def test_daily_receipt_builds_updated_view_only_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    bundle = _build(*inputs[:4])

    assert bundle["integrity"] == {"status": "VERIFIED"}
    assert bundle["continuity_authority"]["status"] == "NO_ACTION_BOUND"
    assert bundle["continuity_authority"]["holdings_valid_through"] == ("2026-08-18")
    assert bundle["freshness"] == {
        "status": "UPDATED",
        "scope": "DAILY_SYNC_LATEST_VERIFIED_LOCAL_CLOSE",
        "mark_as_of": "2026-08-17",
        "generated_at": "2026-08-18T10:00:00+08:00",
        "valid_through": "2026-08-18T23:59:59+08:00",
        "source_kind": "STRICT_CN_EOD_CLOSE",
        "reason": "CURRENT_DAILY_RECEIPT_AND_LATEST_LOCAL_CLOSE",
    }
    assert bundle["research_mark"]["positions"][0]["price"] == 110.0
    assert bundle["research_mark"]["portfolio"] == {
        "cash": 900000.0,
        "market_value": 110000.0,
        "nav": 1010000.0,
        "unrealized_pnl": 10000.0,
        "cash_weight": pytest.approx(900000.0 / 1010000.0),
        "gross_exposure": pytest.approx(110000.0 / 1010000.0),
    }
    performance = bundle["research_mark"]["current_absolute_performance"]
    assert performance["cumulative_return"] == pytest.approx(0.01)
    assert performance["continuity_interval_return"] == pytest.approx(0.01)
    assert performance["authority"] == ("VIEW_ONLY_NO_STORE_OR_PERFORMANCE_AUTHORITY")
    assert bundle["completeness"]["benchmark_relative"] == ("AS_OF_PRIOR_DATE")
    assert bundle["completeness"]["benchmark_as_of"] == "2026-08-14"
    assert v2.validate_v2_shape(bundle) == []
    assert v2.verify_v2_source_refs(bundle, tmp_path) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("point_date", "2026-01-01"),
        ("anchor_date", "2026-01-01"),
        ("continuity_interval_return", 123.456),
        ("max_drawdown", 42.0),
    ],
)
def test_current_performance_math_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    bundle = _build(*inputs[:4])
    bundle["research_mark"]["current_absolute_performance"][field] = value
    bundle["content_sha256"] = v2.content_sha256(bundle)

    assert "current_absolute_performance_identity_invalid" in (v2.validate_v2_shape(bundle))


def test_missing_daily_receipt_is_stale_not_false_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch, include_receipt=False)
    bundle = _build(*inputs[:4])

    assert bundle["continuity_authority"]["status"] == "UNCONFIRMED"
    assert bundle["continuity_authority"]["holdings_valid_through"] == ("2026-08-14")
    assert bundle["freshness"]["status"] == "STALE"
    assert bundle["freshness"]["reason"] == ("DAILY_CONTINUITY_RECEIPT_MISSING")
    assert bundle["completeness"]["current_holdings"] == "STALE"
    assert bundle["completeness"]["current_absolute_performance"] == ("STALE")
    assert v2.validate_v2_shape(bundle) == []


def test_market_pointer_change_during_mark_blocks_mixed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    reader = inputs[3]
    original = reader.read_symbol_frame

    def drifting_read(*args: Any, **kwargs: Any) -> _Result:
        result = original(*args, **kwargs)
        reader.pointer.write_text('{"snapshot_id":"advanced"}\n', encoding="utf-8")
        return result

    reader.read_symbol_frame = drifting_read  # type: ignore[method-assign]
    with pytest.raises(v2.DashboardV2Error, match="market_pointer_changed_during_mark"):
        _build(*inputs[:4])


@pytest.mark.parametrize(
    ("attempt_id", "generated_at", "message"),
    [
        ("bad id", "2026-08-18T10:00:00+08:00", "publication_attempt_id"),
        ("attempt-1", "2026-08-18T02:00:00Z", "generated_at"),
    ],
)
def test_v2_python_grammar_matches_browser_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_id: str,
    generated_at: str,
    message: str,
) -> None:
    v1_bundle, v1_path, record_root, reader, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(v2.DashboardV2Error, match=message):
        v2.build_v2_bundle(
            project_root=v1_path.parents[3],
            v1_bundle=v1_bundle,
            v1_json_path=v1_path,
            record_root=record_root,
            generation_local_date=GENERATION_DATE,
            generated_at=generated_at,
            publication_attempt_id=attempt_id,
            market_reader=reader,
        )


def test_exact_financial_publication_can_refresh_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch, include_receipt=False)
    catalog = inputs[4]
    catalog["lineage_index"][0]["valuation_date"] = "2026-08-18"
    catalog["records"][0]["sealed_at"] = "2026-08-18T01:00:00Z"
    record_root = inputs[2]
    catalog_path = record_root / "_record_store/catalogs/g-fixture/catalog.v3.json"
    catalog_path.write_text(json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8")
    pointer_path = record_root / "_record_store/current.v1.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["catalog_sha256"] = _sha(catalog_path.read_bytes())
    pointer_path.write_text(json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(v2, "load_registered_catalog", lambda _: (pointer, catalog))

    bundle = _build(*inputs[:4])
    continuity = bundle["continuity_authority"]
    assert continuity["status"] == "FINANCIAL_STATE_PUBLICATION"
    assert continuity["financial_state_changed"] is True
    assert continuity["receipt_id"] is None
    assert bundle["freshness"]["status"] == "UPDATED"
    assert bundle["freshness"]["reason"] == ("CURRENT_FINANCIAL_PUBLICATION_AND_LATEST_LOCAL_CLOSE")


def test_missing_non_suspended_close_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch, exact_close=None, suspended=False)
    with pytest.raises(
        v2.DashboardV2Error,
        match="market_exact_close_missing_non_suspended",
    ):
        _build(*inputs[:4])


def test_bound_suspension_set_allows_prior_close_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch, exact_close=None, suspended=True)
    bundle = _build(*inputs[:4])
    position = bundle["research_mark"]["positions"][0]

    assert position["price"] == 105.0
    assert position["price_date"] == "2026-08-14"
    assert position["price_evidence_status"] == ("BOUND_SUSPENSION_CARRY_FORWARD")


def test_staged_v1_override_binds_final_path_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    v1_bundle, v1_path, record_root, reader = inputs[:4]
    raw = v1_path.read_bytes()
    v1_path.unlink()

    bundle = _build(
        v1_bundle,
        v1_path,
        record_root,
        reader,
        v1_json_bytes_override=raw,
    )
    assert bundle["canonical_v1_ref"]["path"].endswith("cn_aggressive_dashboard.v1.json")
    assert v2.verify_v2_source_refs(bundle, tmp_path, v1_bytes_override=raw) == []


def test_source_ref_and_self_hash_tampering_are_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    bundle = _build(*inputs[:4])
    reader = inputs[3]
    reader.serving.write_bytes(b"tampered")
    assert any(
        "source_ref_sha256_mismatch" in error
        for error in v2.verify_v2_source_refs(bundle, tmp_path)
    )

    bundle["research_mark"]["portfolio"]["cash"] += 1.0
    errors = v2.validate_v2_shape(bundle)
    assert "research_mark_portfolio_identity_invalid" in errors
    assert "content_sha256_invalid" in errors


def test_fixed_initial_capital_method_and_zero_flow_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    v1_bundle = inputs[0]
    v1_bundle["portfolio"]["return_method"] = "flow_neutral_unit_nav"
    v1_bundle["content_sha256"] = v2.content_sha256(v1_bundle)
    inputs[1].write_text(json.dumps(v1_bundle, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(v2.DashboardV2Error, match="canonical_v1_return_method_invalid"):
        _build(*inputs[:4])


def test_json_schema_matches_built_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build(*_fixture(tmp_path, monkeypatch)[:4])
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "portfolio_dashboard/schema/cn_aggressive_dashboard.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == (bundle["schema_version"])
    assert set(schema["required"]) == set(bundle)
    assert (
        schema["$defs"]["freshness"]["properties"]["source_kind"]["const"]
        == bundle["freshness"]["source_kind"]
    )
    assert (
        schema["$defs"]["currentAbsolutePerformance"]["properties"]["authority"]["const"]
        == bundle["research_mark"]["current_absolute_performance"]["authority"]
    )


def test_validator_returns_errors_for_bad_numeric_types_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build(*_fixture(tmp_path, monkeypatch)[:4])
    bundle["research_mark"]["positions"][0]["shares"] = None
    bundle["research_mark"]["positions"].append("not-a-position")
    errors = v2.validate_v2_shape(bundle)
    assert "research_mark_position_shape_invalid" in errors
    assert any(error.startswith("research_mark_position_identity_invalid") for error in errors)
