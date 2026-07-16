from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_investor.market.branch_readiness as branch_readiness
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.market.branch_readiness import (
    DEFAULT_READINESS_ROOT,
    STATUS_BLOCK,
    STATUS_PASS,
    STATUS_WARN,
    assess_branch_data_readiness,
    assess_quant_readiness,
    load_fundamental_records,
    write_branch_readiness_report,
)
from quant_investor.market.fundamental_generation import (
    FundamentalGenerationError,
    load_fundamental_pointer,
    publish_fundamental_generation,
)
from tests.helpers.macro_fixture import bind_macro_generation


def _price_frame(symbol: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol, symbol],
            "trade_date": ["20240509", "20240510"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.0],
            "close": [10.1, 10.4],
            "volume": [1000.0, 1200.0],
            "amount": [10_000.0, 12_500.0],
        }
    )


def _write_fundamental_daily(root, symbols=("000001.SZ",), *, partial: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": "20240510",
                "availability_date": "2024-04-30",
                "source": "tushare_fina_indicator;forecast",
                "source_priority": "tushare_primary",
                "fin_roe": None if partial else 0.13,
                "fin_roa": None if partial else 0.06,
                "fin_debt_to_assets": None if partial else 0.42,
                "fin_net_profit_yoy": None if partial else 0.18,
                "fin_ocf_to_profit": None if partial else 1.12,
                "fin_fcf_to_profit": None if partial else 0.88,
                "fcf_to_price": 0.04,
                "forecast_revision": 0.05,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "part.parquet", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps(
            {
                "provider_status": "tushare_primary",
                "source_priority": "tushare_primary",
                "storage_backend": "parquet_canonical",
            }
        ),
        encoding="utf-8",
    )


def _write_macro(root):
    bind_macro_generation(
        root,
        generation_id="fixture",
        row={
            "trade_date": "2024-05-10",
            "macro_score": 0.2,
            "liquidity_score": 0.4,
            "volatility_percentile": 45.0,
            "policy_signal": "neutral",
            "source": "tushare_primary",
            "source_priority": "tushare_primary",
            "pit_status": "market_point_in_time",
            "fetched_at": "2024-05-10T08:00:00+00:00",
        },
    )


def test_quant_missing_required_amount_hard_blocks():
    frame = _price_frame().drop(columns=["amount"])

    readiness = assess_quant_readiness(
        frames={"000001.SZ": frame}, symbols=["000001.SZ"], as_of="20240510"
    )

    assert readiness.status == STATUS_BLOCK
    assert "amount" in readiness.missing_fields
    assert readiness.affected_symbols == ["000001.SZ"]


def test_quant_readiness_requires_exact_as_of_terminal_date():
    future = _price_frame().assign(trade_date=["20240510", "20240511"])
    future_readiness = assess_quant_readiness(
        frames={"000001.SZ": future},
        symbols=["000001.SZ"],
        as_of="20240510",
    )
    assert future_readiness.status == STATUS_BLOCK
    assert future_readiness.freshness_status == "stale"
    assert "freshness" in future_readiness.missing_fields

    missing_date = _price_frame().drop(columns=["trade_date"])
    missing_readiness = assess_quant_readiness(
        frames={"000001.SZ": missing_date},
        symbols=["000001.SZ"],
        as_of="20240510",
    )
    assert missing_readiness.status == STATUS_BLOCK
    assert missing_readiness.freshness_status == "stale"
    assert "freshness" in missing_readiness.missing_fields


def test_three_branch_readiness_blocks_only_missing_surviving_branch_data(tmp_path):
    fundamental_root = tmp_path / "cn_fundamental"
    macro_root = tmp_path / "cn_macro"
    _write_fundamental_daily(fundamental_root, symbols=("000001.SZ",))
    _write_macro(macro_root)
    frames = {
        "000001.SZ": _price_frame("000001.SZ"),
        "000002.SZ": _price_frame("000002.SZ"),
    }

    report = assess_branch_data_readiness(
        frames=frames,
        candidate_symbols=["000001.SZ", "000002.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        macro_root=macro_root,
        run_id="fixture",
    )

    assert tuple(report.readiness) == CANONICAL_BRANCH_ORDER
    assert report.readiness["quant"].status == STATUS_PASS
    assert report.readiness["fundamental"].status == STATUS_BLOCK
    assert report.readiness["macro"].status == STATUS_PASS
    assert report.blocked_symbols == ["000002.SZ"]
    assert report.investable_universe == ["000001.SZ"]
    assert set(report.branch_data) == {"fundamentals", "macro_data"}


def test_three_branch_readiness_reports_invalid_fundamental_generation_as_blocked(
    tmp_path,
    monkeypatch,
):
    def _invalid_generation(*_args, **_kwargs):
        raise FundamentalGenerationError(
            "fundamental primary provenance envelope missing"
        )

    monkeypatch.setattr(
        branch_readiness,
        "load_fundamental_records",
        _invalid_generation,
    )
    macro_root = tmp_path / "cn_macro"
    _write_macro(macro_root)

    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame("000001.SZ")},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=tmp_path / "invalid_fundamental",
        macro_root=macro_root,
        run_id="invalid-fundamental-generation",
    )

    fundamental = report.readiness["fundamental"]
    assert fundamental.status == STATUS_BLOCK
    assert fundamental.blockers == ["fundamental_generation_invalid"]
    assert fundamental.provider_status == "blocked_invalid_generation"
    assert fundamental.metadata["manifest"]["read_error"] == (
        "fundamental primary provenance envelope missing"
    )
    assert report.blocked_symbols == ["000001.SZ"]
    assert report.investable_universe == []


def test_retired_readiness_root_argument_is_rejected(tmp_path):
    with pytest.raises(TypeError):
        assess_branch_data_readiness(
            frames={"000001.SZ": _price_frame()},
            intelligence_root=tmp_path / "retired",  # type: ignore[call-arg]
        )


def test_unified_data_bundle_rejects_retired_intelligence_inputs():
    with pytest.raises(TypeError):
        UnifiedDataBundle(event_data={})  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        UnifiedDataBundle(sentiment_data={})  # type: ignore[call-arg]


def test_branch_readiness_quant_scope_uses_after_funnel_candidates(tmp_path):
    fundamental_root = tmp_path / "cn_fundamental"
    macro_root = tmp_path / "cn_macro"
    _write_fundamental_daily(fundamental_root)
    _write_macro(macro_root)
    frames = {
        "000001.SZ": _price_frame("000001.SZ"),
        "000002.SZ": _price_frame("000002.SZ").drop(columns=["amount"]),
    }

    report = assess_branch_data_readiness(
        frames=frames,
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        macro_root=macro_root,
        run_id="fixture",
    )

    assert report.readiness["quant"].status == STATUS_PASS
    assert report.blocked_symbols == []
    assert report.quantifiable_universe == ["000001.SZ"]
    assert report.investable_universe == ["000001.SZ"]


def test_branch_readiness_loads_default_canonical_parquet(tmp_path, monkeypatch):
    parquet_root = tmp_path / "data" / "parquet" / "cn"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(branch_readiness, "DEFAULT_FUNDAMENTAL_ROOT", parquet_root / "fundamental_daily")
    monkeypatch.setattr(branch_readiness, "DEFAULT_MACRO_ROOT", parquet_root / "macro_daily")
    _write_fundamental_daily(parquet_root / "fundamental_daily")
    _write_macro(parquet_root / "macro_daily")

    def _fail_read_csv(*_args, **_kwargs):
        raise AssertionError("branch_readiness must not read CSV marts")

    monkeypatch.setattr(branch_readiness.pd, "read_csv", _fail_read_csv)
    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame()},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=parquet_root / "fundamental_daily",
        macro_root=parquet_root / "macro_daily",
        run_id="fixture",
    )

    assert report.blocked_symbols == []
    assert report.readiness["fundamental"].status == STATUS_PASS
    assert report.readiness["macro"].status == STATUS_PASS
    assert report.readiness["fundamental"].metadata["manifest"]["storage_backend"] == "parquet_canonical"


def test_branch_readiness_reuses_pinned_macro_without_second_canonical_read(
    tmp_path,
    monkeypatch,
):
    fundamental_root = tmp_path / "cn_fundamental"
    _write_fundamental_daily(fundamental_root)
    macro_record = {
        "trade_date": "2024-05-10",
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2024-05-10T08:00:00+00:00",
    }
    macro_manifest = {
        "generation_id": "pinned-generation",
        "parquet_sha256": "a" * 64,
        "generation_manifest_sha256": "b" * 64,
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "provider_status": "verified_provider_snapshot",
        "production_eligible": True,
    }

    monkeypatch.setattr(
        branch_readiness,
        "load_macro_record",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("pinned Macro must not be read again")
        ),
    )

    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame()},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        pinned_macro_record=macro_record,
        pinned_macro_manifest=macro_manifest,
        run_id="pinned-macro",
    )

    readiness = report.readiness["macro"]
    assert readiness.status == STATUS_PASS
    assert report.branch_data["macro_data"] == macro_record
    assert readiness.metadata["canonical_identity"] == {
        "generation_id": "pinned-generation",
        "parquet_sha256": "a" * 64,
        "generation_manifest_sha256": "b" * 64,
    }


def test_branch_readiness_prefers_generation_pointer_over_stale_legacy_table(tmp_path):
    fundamental_root = tmp_path / "cn"
    legacy_root = fundamental_root / "fundamental_daily"
    _write_fundamental_daily(legacy_root)
    generation_daily = pd.read_parquet(legacy_root / "part.parquet").assign(
        fin_roe=0.27,
        source="offline_generation_fixture",
        source_priority="manual_offline_snapshot",
    )
    publish_fundamental_generation(
        root=fundamental_root,
        run_id="new-generation",
        tables={
            "fundamental_period": pd.DataFrame(columns=["ts_code"]),
            "fundamental_daily": generation_daily,
            "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
        },
        metadata={
            "run_id": "new-generation",
            "source_priority": "manual_offline_snapshot",
            "storage_backend": "parquet_canonical_generation",
        },
    )

    records, manifest = load_fundamental_records(
        ["000001.SZ"], as_of="20240510", root=fundamental_root
    )

    assert records["000001.SZ"]["fin_roe"] == 0.27
    assert (
        records["000001.SZ"]["fundamental_generation_id"]
        == "new-generation"
    )
    assert manifest["generation_id"] == "new-generation"
    assert manifest["storage_backend"] == "parquet_canonical_generation"


def test_fundamental_generation_rejects_table_tamper_after_cached_pointer(
    tmp_path,
):
    fundamental_root = tmp_path / "cn"
    daily = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "availability_date": "2024-04-30",
                "source": "offline_generation_fixture",
                "source_priority": "manual_offline_snapshot",
                "fin_roe": 0.13,
            }
        ]
    )
    table_paths, _pointer = publish_fundamental_generation(
        root=fundamental_root,
        run_id="hash-bound-generation",
        tables={
            "fundamental_period": pd.DataFrame(columns=["ts_code"]),
            "fundamental_daily": daily,
            "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
        },
        metadata={
            "run_id": "hash-bound-generation",
            "storage_backend": "parquet_canonical_generation",
        },
    )

    assert load_fundamental_pointer(fundamental_root) is not None
    daily.assign(fin_roe=0.99).to_parquet(
        table_paths["fundamental_daily"],
        index=False,
    )

    with pytest.raises(
        FundamentalGenerationError,
        match="fundamental table hash mismatch",
    ):
        load_fundamental_records(
            ["000001.SZ"],
            as_of="20240510",
            root=fundamental_root,
        )


def test_fundamental_pointer_cache_does_not_expose_mutable_nested_state(
    tmp_path,
):
    fundamental_root = tmp_path / "cn"
    publish_fundamental_generation(
        root=fundamental_root,
        run_id="immutable-cache",
        tables={
            "fundamental_period": pd.DataFrame(columns=["ts_code"]),
            "fundamental_daily": pd.DataFrame(columns=["ts_code"]),
            "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
        },
        metadata={"run_id": "immutable-cache"},
    )

    first = load_fundamental_pointer(fundamental_root)
    assert first is not None
    first["manifest"]["generation_id"] = "mutated-by-caller"

    second = load_fundamental_pointer(fundamental_root)
    assert second is not None
    assert second["manifest"]["generation_id"] == "immutable-cache"


def test_fundamental_generation_rejects_root_ancestor_symlink(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    real_root = real_parent / "cn"
    tables = {
        table_name: pd.DataFrame(columns=["ts_code"])
        for table_name in (
            "fundamental_period",
            "fundamental_daily",
            "fundamental_quarantine",
        )
    }
    publish_fundamental_generation(
        root=real_root,
        run_id="g1",
        tables=tables,
        metadata={},
    )
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        FundamentalGenerationError,
        match="fundamental root ancestor symlink rejected",
    ):
        load_fundamental_pointer(alias / "cn")
    with pytest.raises(
        FundamentalGenerationError,
        match="fundamental root ancestor symlink rejected",
    ):
        publish_fundamental_generation(
            root=alias / "new-cn",
            run_id="g2",
            tables=tables,
            metadata={},
        )


def test_fundamental_partial_record_warns_without_blocking(tmp_path):
    fundamental_root = tmp_path / "cn_fundamental"
    macro_root = tmp_path / "cn_macro"
    _write_fundamental_daily(fundamental_root, partial=True)
    _write_macro(macro_root)

    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame()},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        macro_root=macro_root,
        run_id="fixture",
    )

    assert report.readiness["fundamental"].status == STATUS_WARN
    assert report.readiness["fundamental"].affected_symbols == []
    assert report.readiness["fundamental"].metadata["partial_symbols"] == ["000001.SZ"]
    assert report.blocked_symbols == []


def test_macro_missing_blocks_three_branch_fusion(tmp_path):
    fundamental_root = tmp_path / "cn_fundamental"
    _write_fundamental_daily(fundamental_root)

    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame()},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        macro_root=tmp_path / "missing_macro",
        run_id="fixture",
    )

    assert report.readiness["macro"].status == STATUS_BLOCK
    assert report.blocked_symbols == ["000001.SZ"]
    assert report.investable_universe == []


def test_branch_readiness_writes_standard_artifacts(tmp_path):
    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame()},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=tmp_path / "missing_fundamental",
        macro_root=tmp_path / "missing_macro",
        run_id="fixture",
    )

    artifacts = write_branch_readiness_report(report, output_dir=tmp_path / "reports")

    assert set(artifacts) == {"json", "md", "csv"}
    payload = json.loads((tmp_path / "reports" / "fixture.json").read_text())
    assert tuple(payload["readiness"]) == CANONICAL_BRANCH_ORDER
    assert payload["readiness"]["quant"]["status"] == STATUS_PASS


def test_branch_readiness_default_is_v15_and_frozen_v13_root_is_rejected(
    tmp_path: Path,
):
    assert DEFAULT_READINESS_ROOT == Path("reports/v15/branch_readiness")

    report = assess_branch_data_readiness(
        frames={},
        candidate_symbols=[],
        fundamental_root=tmp_path / "missing_fundamental",
        macro_root=tmp_path / "missing_macro",
        run_id="fixture",
    )
    with pytest.raises(ValueError, match="frozen v13 retirement evidence"):
        write_branch_readiness_report(
            report,
            output_dir=branch_readiness._REPOSITORY_ROOT
            / "reports"
            / "branch_readiness",
        )
