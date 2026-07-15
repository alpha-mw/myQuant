from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import pytest

import quant_investor.market.fundamental_generation as fundamental_generation
import quant_investor.market.fundamental_mart as fundamental_mart
from quant_investor.agents.fundamental_agent import FundamentalAgent
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.factors.pit_fundamentals import (
    build_fundamental_metric_matrices,
    load_fundamental_pit_series,
)
from quant_investor.market.fundamental_mart import (
    DERIVED_DAILY_FIELDS,
    FundamentalReadinessError,
    _fetch_tushare_tables,
    _resolve_symbols_from_parquet_universe,
    run_cn_fundamental_maintenance,
    write_fundamental_mart,
)
from quant_investor.market.fundamental_generation import (
    FundamentalGenerationError,
    load_fundamental_pointer,
    publish_fundamental_generation,
)
from quant_investor.market.branch_readiness import load_fundamental_records
from quant_investor.market.dag.assembly import (
    _aggregate_branch_summaries,
    _build_branch_results,
)


def _raw_tables() -> dict[str, pd.DataFrame]:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    fina_rows = []
    income_rows = []
    balance_rows = []
    cashflow_rows = []
    daily_rows = []
    forecast_rows = []
    for idx, symbol in enumerate(symbols, start=1):
        sector = ["bank", "industrial", "healthcare"][idx - 1]
        for end_date, ann_date, profit, ocf, capex, roe in (
            ("20221231", "20230428", 80.0 + idx, 100.0 + idx, 10.0, 8.0 + idx),
            ("20231231", "20240430", 100.0 + idx, 130.0 + idx, 20.0, 12.0 + idx),
        ):
            fina_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "roe_dt": roe,
                    "roa": 5.0 + idx,
                    "debt_to_assets": 45.0 + idx,
                    "netprofit_yoy": "",
                    "ocf_to_profit": "",
                }
            )
            income_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "n_income_attr_p": profit,
                }
            )
            balance_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "total_liab": 400.0 + idx,
                    "total_assets": 1000.0 + idx,
                }
            )
            cashflow_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "n_cashflow_act": ocf,
                    "c_pay_acq_const_fiolta": capex,
                }
            )
        for trade_date in ("20240429", "20240430", "20240502", "20240510"):
            daily_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "total_mv": 100000.0 * idx,
                    "sector": sector,
                }
            )
        forecast_rows.append(
            {
                "ts_code": symbol,
                "ann_date": "20240429",
                "end_date": "20240630",
                "type": "预增",
                "p_change_min": 5.0 + idx,
                "p_change_max": 15.0 + idx,
                "summary": "fixture forecast",
                "change_reason": "fixture",
            }
        )
    fina_rows.append(
        {
            "ts_code": "000001.SZ",
            "end_date": "20231231",
            "ann_date": "20240510",
            "f_ann_date": "20240510",
            "roe_dt": 20.0,
            "roa": 7.0,
            "debt_to_assets": 40.0,
        }
    )
    fina_rows.append(
        {
            "ts_code": "000004.SZ",
            "end_date": "20231231",
            "roe_dt": 10.0,
        }
    )
    return {
        "fina_indicator": pd.DataFrame(fina_rows),
        "income": pd.DataFrame(income_rows),
        "balancesheet": pd.DataFrame(balance_rows),
        "cashflow": pd.DataFrame(cashflow_rows),
        "daily_basic": pd.DataFrame(daily_rows),
        "forecast": pd.DataFrame(forecast_rows),
    }


def _write_parquet_market_data(root, symbols):
    data_root = root / "data"
    parquet_root = data_root / "parquet" / "cn"
    bars_root = parquet_root / "bars"
    serving_root = data_root / "parquet_serving" / "cn" / "bars"
    manifest_path = parquet_root / "_snapshots" / "fixture.json"
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": "20240510",
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "vol": 1000.0,
                "amount": 10000.0,
            }
        )
        serving_dir = serving_root / f"symbol={symbol}"
        serving_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([rows[-1]]).to_parquet(serving_dir / "bars.parquet", index=False)
    (bars_root / "year=2024").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(bars_root / "year=2024" / "part.parquet", index=False)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
    (parquet_root / "_latest.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "snapshot_id": "fixture",
                "latest_complete_trade_date": "20240510",
                "latest_trade_date": "20240510",
                "table_root": str(bars_root),
                "derived_serving_root": str(serving_root),
                "manifest_path": str(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return data_root


def test_fundamental_mart_pit_join_readiness_and_quarantine(tmp_path):
    artifacts, readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )

    daily = pd.read_parquet(artifacts.fundamental_daily_path)
    symbol_daily = daily[daily["ts_code"] == "000001.SZ"].set_index("trade_date")
    assert symbol_daily.loc["2024-04-29", "fin_roe"] == 0.09
    assert symbol_daily.loc["2024-04-30", "fin_roe"] == 0.13
    assert symbol_daily.loc["2024-05-02", "fin_roe"] == 0.13
    assert symbol_daily.loc["2024-05-10", "fin_roe"] == 0.20
    assert symbol_daily.loc["2024-04-30", "fin_net_profit_yoy"] > 0.0
    assert symbol_daily.loc["2024-04-30", "fin_fcf_to_profit"] > 0.0
    assert symbol_daily.loc["2024-04-30", "fcf_to_price"] > 0.0
    assert symbol_daily.loc["2024-04-30", "forecast_revision"] == 0.11
    assert readiness["gate2_passed"] is True
    assert readiness["coverage_rate"] >= 0.60
    quarantine = pd.read_parquet(artifacts.quarantine_path)
    assert "missing_ts_code_end_date_or_announcement_date" in set(quarantine["quarantine_reason"])
    assert artifacts.readiness_json_path.exists()
    assert json.loads(artifacts.readiness_json_path.read_text())["gate2_passed"] is True


def test_vectorized_daily_pit_join_matches_per_symbol_reference() -> None:
    raw = _raw_tables()
    raw["daily_basic"] = pd.concat(
        [
            raw["daily_basic"],
            pd.DataFrame(
                [
                    {
                        "ts_code": "999999.SZ",
                        "trade_date": "20240510",
                        "total_mv": 1.0,
                        "sector": "out-of-period-scope",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    timestamp = "2024-05-10T12:00:00Z"
    period, _quarantine = fundamental_mart.derive_fundamental_period(
        raw,
        run_id="vectorized-reference",
        source="live_tushare",
        derivation_timestamp=timestamp,
    )
    tie_seed = period[period["ts_code"] == "000001.SZ"].iloc[0]
    tie_rows = []
    for ordinal in range(32):
        row = tie_seed.copy()
        row["end_date"] = pd.Timestamp("2016-03-31") + pd.offsets.QuarterEnd(ordinal)
        row["availability_date"] = pd.Timestamp("2024-04-30")
        row["fin_roe"] = float(ordinal)
        row["free_cashflow"] = float(ordinal + 1)
        tie_rows.append(row)
    period = pd.concat([period, pd.DataFrame(tie_rows)], ignore_index=True)

    daily = fundamental_mart._normalize_daily_basic(
        raw["daily_basic"],
        run_id="vectorized-reference",
        source="live_tushare",
        sector_map=None,
        derivation_timestamp=timestamp,
    )
    forecast = fundamental_mart._normalize_forecast(
        raw["forecast"],
        run_id="vectorized-reference",
        source="live_tushare",
        derivation_timestamp=timestamp,
    )
    period_work = period.copy()
    period_work["availability_date"] = pd.to_datetime(
        period_work["availability_date"], errors="coerce"
    )
    period_groups = {
        str(symbol): group.sort_values("availability_date")
        for symbol, group in period_work.groupby("ts_code", sort=False)
    }
    forecast_groups = {
        str(symbol): group.sort_values("availability_date")
        for symbol, group in forecast.groupby("ts_code", sort=False)
    }
    outputs: list[pd.DataFrame] = []
    for symbol, daily_group in daily.groupby("ts_code", sort=True):
        period_group = period_groups.get(str(symbol))
        if period_group is None or period_group.empty:
            continue
        joined = pd.merge_asof(
            daily_group.sort_values("trade_date"),
            period_group,
            left_on="trade_date",
            right_on="availability_date",
            direction="backward",
            suffixes=("", "_period"),
        )
        forecast_group = forecast_groups.get(str(symbol))
        if forecast_group is not None and not forecast_group.empty:
            joined = pd.merge_asof(
                joined.sort_values("trade_date"),
                forecast_group.drop(columns=["ts_code"]),
                left_on="trade_date",
                right_on="availability_date",
                direction="backward",
                suffixes=("", "_forecast"),
            )
        joined["ts_code"] = symbol
        outputs.append(joined)
    reference = pd.concat(outputs, ignore_index=True)
    reference["fcf_to_price"] = pd.to_numeric(
        reference.get("free_cashflow"), errors="coerce"
    ).div(
        pd.to_numeric(reference.get("total_mv_rmb"), errors="coerce").where(
            pd.to_numeric(reference.get("total_mv_rmb"), errors="coerce") > 0
        )
    )
    reference["size_bucket"] = fundamental_mart._size_buckets(reference)
    keep = [
        "ts_code",
        "trade_date",
        "end_date",
        "availability_date",
        "source_version",
        "source",
        "fetched_at",
        "sector",
        "size_bucket",
        "total_mv_rmb",
        *fundamental_mart.DERIVED_DAILY_FIELDS,
        "forecast_end_date",
        "forecast_ann_date",
        "forecast_type",
        "forecast_summary",
        "forecast_change_reason",
        "forecast_source",
        "forecast_fetched_at",
        "forecast_ingest_run_id",
    ]
    reference = reference[
        [column for column in keep if column in reference.columns]
    ].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    actual = fundamental_mart.build_fundamental_daily(
        period,
        raw["daily_basic"],
        raw["forecast"],
        run_id="vectorized-reference",
        source="live_tushare",
        derivation_timestamp=timestamp,
    )

    fundamental_mart.assert_frame_semantics_equal(
        reference,
        actual,
        label="vectorized daily PIT join",
    )
    assert "999999.SZ" not in set(actual["ts_code"])


def test_default_local_mart_is_offline_and_likelihood_neutral(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    _artifacts, readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="offline-default",
        write_raw_snapshots=False,
    )
    records, manifest = load_fundamental_records(
        ["000001.SZ"],
        as_of="20240510",
        root=data_root,
    )
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=["000001.SZ"],
        symbol_data={
            "000001.SZ": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-05-10"]),
                    "close": [10.0],
                }
            )
        },
        fundamentals=records,
        metadata={
            "branch_data_readiness": {
                "readiness": {
                    "fundamental": {
                        "status": "block",
                        "pit_status": "point_in_time",
                        "source_priority": readiness["source_priority"],
                        "metadata": {"manifest": manifest},
                    }
                }
            }
        },
    )
    verdict = FundamentalAgent().run(
        {"data_bundle": bundle, "stock_pool": ["000001.SZ"]}
    )
    summaries = _aggregate_branch_summaries(
        {"000001.SZ": {"fundamental": verdict}}
    )
    branch_results = _build_branch_results(
        {"000001.SZ": {"fundamental": verdict}},
        summaries,
    )
    likelihoods = SignalLikelihoodMapper().compute_likelihoods(
        branch_results=branch_results,
        symbol="000001.SZ",
        candidate_symbols={"000001.SZ"},
    )

    assert readiness["provider_status"] == "manual_offline_snapshot"
    assert readiness["source_priority"] == "manual_offline_snapshot"
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert likelihoods.fundamental_likelihood == 0.50
    assert "fundamental" not in likelihoods.metadata["evidence_sources"]


def test_source_name_alone_cannot_claim_tushare_primary(tmp_path):
    with pytest.raises(
        ValueError,
        match="Tushare source requires verified provider provenance",
    ):
        write_fundamental_mart(
            _raw_tables(),
            data_root=tmp_path / "clean" / "cn_fundamental",
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="forged-default-source",
            source="tushare",
            write_raw_snapshots=False,
        )


def test_verified_local_evidence_cannot_mint_primary_generation(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    evidence_path = tmp_path / "tushare_readiness.json"
    evidence_path.write_text(
        json.dumps(
            {
                "provider_status": "live_tushare_partial",
                "provider_manifest": {"provider": "tushare"},
            }
        ),
        encoding="utf-8",
    )
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="internal live Tushare attestation"):
        write_fundamental_mart(
            _raw_tables(),
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="verified-source",
            source="canonical_raw_offline_rebuild",
            provider_manifest={
                "source_priority": "tushare_primary",
                "source_provenance": "verified_local_tushare_refresh_manifests",
                "provenance_evidence": [
                    {
                        "path": str(evidence_path),
                        "sha256": evidence_sha256,
                    }
                ],
            },
        )

    assert load_fundamental_pointer(data_root) is None


def _live_provider_manifest(
    tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    outcomes: list[dict[str, object]] = []
    for table in fundamental_mart.SOURCE_TABLES:
        rows = len(tables.get(table, pd.DataFrame()))
        outcomes.append(
            {
                "schema_version": fundamental_mart.FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
                "symbol": "000001.SZ",
                "table": table,
                "status": "success" if rows else "empty",
                "rows_received": rows,
                "rows": rows,
                "rows_hard_invalid": 0,
                "rows_filtered_future": 0,
                "rows_filtered_missing_availability": 0,
                "rows_filtered_core_values": 0,
                "rows_deduplicated": 0,
                "rows_discarded_request_malformed": 0,
                "rows_hard_invalid_schema": 0,
                "rows_hard_invalid_symbol": 0,
                "rows_hard_invalid_availability_date": 0,
                "rows_hard_invalid_end_date": 0,
                "rows_hard_invalid_end_after_availability": 0,
                "rows_hard_invalid_core_numeric": 0,
            }
        )
    succeeded = sum(outcome["status"] == "success" for outcome in outcomes)
    empty = len(outcomes) - succeeded
    return {
        "schema_version": fundamental_mart.FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
        "provider": "tushare",
        "provider_status": "live_tushare",
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit",
        "tables": list(fundamental_mart.SOURCE_TABLES),
        "raw_row_counts": {
            table: len(tables.get(table, pd.DataFrame()))
            for table in fundamental_mart.SOURCE_TABLES
        },
        "raw_table_fingerprints": {
            table: fundamental_mart.frame_fingerprint(
                tables.get(table, pd.DataFrame())
            )
            for table in fundamental_mart.SOURCE_TABLES
        },
        "requests_attempted": len(outcomes),
        "requests_succeeded_with_rows": succeeded,
        "requests_empty": empty,
        "requests_failed": 0,
        "symbol_table_outcomes": outcomes,
        "request_outcome_accounting_sha256": (
            fundamental_mart.canonical_json_sha256(outcomes)
        ),
        "endpoint_audit": {
            "schema_version": fundamental_mart.FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA,
        },
    }


def test_live_attestation_binds_primary_generation_to_current_raw_tables(
    tmp_path,
):
    tables = _raw_tables()
    provider_manifest = _live_provider_manifest(tables)
    attestation = fundamental_mart._issue_live_tushare_attestation(
        "live_tushare",
        provider_manifest,
        tables,
    )
    data_root = tmp_path / "clean" / "cn_fundamental"

    write_fundamental_mart(
        tables,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="live-source",
        source="live_tushare",
        provider_manifest=provider_manifest,
        write_raw_snapshots=False,
        _live_tushare_attestation=attestation,
    )

    pointer = load_fundamental_pointer(data_root)
    assert pointer is not None
    assert pointer["metadata"]["source_priority"] == "tushare_primary"
    assert pointer["metadata"]["source_provenance"] == "live_tushare_explicit"
    assert pointer["primary_provenance_verified"] is True
    assert pointer["primary_provenance"]["schema_version"] == (
        fundamental_generation.PRIMARY_PROVENANCE_SCHEMA_VERSION
    )
    assert set(pointer["primary_provenance"]["raw_table_fingerprints"]) == set(
        fundamental_mart.SOURCE_TABLES
    )


def test_live_attestation_rejects_raw_table_tampering(tmp_path):
    tables = _raw_tables()
    provider_manifest = _live_provider_manifest(tables)
    attestation = fundamental_mart._issue_live_tushare_attestation(
        "live_tushare",
        provider_manifest,
        tables,
    )
    tampered = {name: frame.copy(deep=True) for name, frame in tables.items()}
    tampered["daily_basic"].loc[0, "total_mv"] = 999999999.0

    with pytest.raises(ValueError, match="internal live Tushare attestation"):
        write_fundamental_mart(
            tampered,
            data_root=tmp_path / "clean" / "cn_fundamental",
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="tampered-live-source",
            source="live_tushare",
            provider_manifest=provider_manifest,
            write_raw_snapshots=False,
            _live_tushare_attestation=attestation,
        )


def test_primary_generation_cannot_publish_when_readiness_gate_fails(
    tmp_path,
):
    tables = _raw_tables()
    tables["daily_basic"] = tables["daily_basic"].assign(sector="one-sector")
    provider_manifest = _live_provider_manifest(tables)
    attestation = fundamental_mart._issue_live_tushare_attestation(
        "live_tushare",
        provider_manifest,
        tables,
    )
    data_root = tmp_path / "clean" / "cn_fundamental"

    with pytest.raises(
        FundamentalGenerationError,
        match="passed readiness gate",
    ):
        write_fundamental_mart(
            tables,
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="failed-live-readiness",
            source="live_tushare",
            provider_manifest=provider_manifest,
            write_raw_snapshots=False,
            publish_on_gate_failure=True,
            _live_tushare_attestation=attestation,
        )

    assert load_fundamental_pointer(data_root) is None


def test_public_generation_publisher_rejects_forged_primary(tmp_path):
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240510",
                    "source_priority": "tushare_primary",
                    "fin_roe": 0.99,
                }
            ]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }

    with pytest.raises(
        FundamentalGenerationError,
        match="internal primary capability",
    ):
        publish_fundamental_generation(
            root=tmp_path / "cn",
            run_id="forged-primary",
            tables=tables,
            metadata={
                "run_id": "forged-primary",
                "provider_status": "live_tushare",
                "source_priority": "tushare_primary",
                "source_provenance": "live_tushare_explicit",
            },
        )

    assert load_fundamental_pointer(tmp_path / "cn") is None


def test_public_generation_publisher_allows_explicit_offline_generation(
    tmp_path,
):
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240510",
                    "source_priority": "manual_offline_snapshot",
                    "fin_roe": 0.13,
                }
            ]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }

    _paths, pointer = publish_fundamental_generation(
        root=tmp_path / "cn",
        run_id="offline-generation",
        tables=tables,
        metadata={
            "run_id": "offline-generation",
            "source_priority": "manual_offline_snapshot",
        },
    )

    assert pointer["metadata"]["source_priority"] == "manual_offline_snapshot"


def test_primary_generation_capability_is_bound_to_exact_tables(tmp_path):
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    provider_manifest = {"provider": "tushare", "run_id": "bound-primary"}
    raw_tables = _raw_tables()
    metadata = {
        "run_id": "bound-primary",
        "provider_status": "live_tushare",
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit",
        "provider_manifest": provider_manifest,
        "gate2_passed": True,
    }
    capability = (
        fundamental_generation._issue_primary_generation_attestation(
            tables=tables,
            metadata=metadata,
            source="live_tushare",
            provider_manifest_sha256=(
                fundamental_mart._canonical_mapping_sha256(provider_manifest)
            ),
            raw_table_fingerprints=(
                fundamental_mart._raw_table_fingerprints(raw_tables)
            ),
        )
    )
    tampered = {name: frame.copy(deep=True) for name, frame in tables.items()}
    tampered["fundamental_daily"].loc[0, "ts_code"] = "000002.SZ"

    with pytest.raises(
        FundamentalGenerationError,
        match="internal primary capability",
    ):
        publish_fundamental_generation(
            root=tmp_path / "cn",
            run_id="bound-primary",
            tables=tampered,
            metadata=metadata,
            _primary_attestation=capability,
        )


def test_legacy_primary_pointer_without_durable_provenance_is_rejected(
    tmp_path,
):
    root = tmp_path / "cn"
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(columns=["ts_code"]),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    _paths, pointer = publish_fundamental_generation(
        root=root,
        run_id="legacy-primary",
        tables=tables,
        metadata={
            "run_id": "legacy-primary",
            "source_priority": "manual_offline_snapshot",
        },
    )
    pointer_path = root / fundamental_generation.FUNDAMENTAL_POINTER_FILENAME
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = root / pointer["manifest_path"]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointer_payload["metadata"]["source_priority"] = "tushare_primary"
    manifest_payload["metadata"]["source_priority"] = "tushare_primary"
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(
        FundamentalGenerationError,
        match="primary provenance envelope missing",
    ):
        load_fundamental_pointer(root)


def test_offline_generation_rejects_unverified_tushare_priority(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"

    with pytest.raises(ValueError, match="tushare_primary requires"):
        write_fundamental_mart(
            _raw_tables(),
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="forged-source",
            source="canonical_raw_offline_rebuild",
            provider_manifest={
                "source_priority": "tushare_primary",
                "source_provenance": "unverified_caller_claim",
            },
        )

    assert load_fundamental_pointer(data_root) is None


def _only_symbol(
    tables: dict[str, pd.DataFrame],
    symbol: str,
) -> dict[str, pd.DataFrame]:
    return {
        name: (
            frame[frame["ts_code"] == symbol].copy()
            if "ts_code" in frame.columns
            else frame.copy()
        )
        for name, frame in tables.items()
    }


def test_partial_refresh_preserves_prior_symbols_in_new_generation(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    first, _ = write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="generation-one",
        source="first_source",
    )
    second, readiness = write_fundamental_mart(
        _only_symbol(_raw_tables(), "000001.SZ"),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="generation-two",
        source="second_source",
    )

    daily = pd.read_parquet(second.fundamental_daily_path)
    pointer = load_fundamental_pointer(data_root)
    assert set(daily["ts_code"]) == {
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    }
    assert pointer is not None
    assert pointer["generation_id"] == "generation-two"
    assert first.fundamental_daily_path.exists()
    assert readiness["merge"]["fundamental_daily"][
        "retained_existing_rows"
    ] > 0


def test_primary_refresh_cannot_upgrade_retained_offline_parent_rows(
    tmp_path,
):
    data_root = tmp_path / "clean" / "cn_fundamental"
    write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="offline-parent",
        write_raw_snapshots=False,
    )
    live_tables = _only_symbol(_raw_tables(), "000001.SZ")
    provider_manifest = _live_provider_manifest(live_tables)
    attestation = fundamental_mart._issue_live_tushare_attestation(
        "live_tushare",
        provider_manifest,
        live_tables,
    )

    with pytest.raises(
        ValueError,
        match="retain rows from an unverified parent",
    ):
        write_fundamental_mart(
            live_tables,
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="live-child",
            source="live_tushare",
            provider_manifest=provider_manifest,
            write_raw_snapshots=False,
            _live_tushare_attestation=attestation,
        )

    pointer = load_fundamental_pointer(data_root)
    assert pointer is not None
    assert pointer["generation_id"] == "offline-parent"
    assert pointer["primary_provenance_verified"] is False


def test_primary_refresh_can_retain_rows_from_verified_primary_parent(
    tmp_path,
):
    data_root = tmp_path / "clean" / "cn_fundamental"
    parent_tables = _raw_tables()
    parent_manifest = _live_provider_manifest(parent_tables)
    parent_attestation = fundamental_mart._issue_live_tushare_attestation(
        "live_tushare",
        parent_manifest,
        parent_tables,
    )
    write_fundamental_mart(
        parent_tables,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="live-parent",
        source="live_tushare",
        provider_manifest=parent_manifest,
        write_raw_snapshots=False,
        _live_tushare_attestation=parent_attestation,
    )
    child_tables = _only_symbol(_raw_tables(), "000001.SZ")
    child_manifest = _live_provider_manifest(child_tables)
    child_attestation = fundamental_mart._issue_live_tushare_attestation(
        "live_tushare",
        child_manifest,
        child_tables,
    )

    _artifacts, readiness = write_fundamental_mart(
        child_tables,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="live-child",
        source="live_tushare",
        provider_manifest=child_manifest,
        write_raw_snapshots=False,
        _live_tushare_attestation=child_attestation,
    )

    pointer = load_fundamental_pointer(data_root)
    assert readiness["merge"]["fundamental_daily"]["retained_existing_rows"] > 0
    assert pointer is not None
    assert pointer["generation_id"] == "live-child"
    assert pointer["primary_provenance_verified"] is True
    assert pointer["manifest"]["metadata"]["parent_generation_id"] == (
        "live-parent"
    )


def test_partial_exact_key_does_not_replace_more_complete_pit_row(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="complete-generation",
        source="complete_source",
    )
    partial = _only_symbol(_raw_tables(), "000001.SZ")
    partial["fina_indicator"] = partial["fina_indicator"].assign(
        roa=None,
        debt_to_assets=None,
    )
    artifacts, _ = write_fundamental_mart(
        partial,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="partial-generation",
        source="partial_source",
    )

    period = pd.read_parquet(artifacts.fundamental_period_path)
    row = period[
        (period["ts_code"] == "000001.SZ")
        & (period["end_date"] == "20231231")
        & (
            pd.to_datetime(period["availability_date"])
            == pd.Timestamp("2024-04-30")
        )
    ].iloc[-1]
    assert row["fin_roa"] == 0.06
    assert row["fin_debt_to_assets"] > 0.0
    assert row["source"] == "complete_source"


def test_complete_exact_key_refresh_wins_on_quality_tie(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="old-generation",
        source="old_source",
    )
    refreshed = _only_symbol(_raw_tables(), "000001.SZ")
    target = (
        (refreshed["fina_indicator"]["end_date"] == "20231231")
        & (refreshed["fina_indicator"]["ann_date"] == "20240430")
    )
    refreshed["fina_indicator"].loc[target, "roe_dt"] = 18.0
    artifacts, _ = write_fundamental_mart(
        refreshed,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="new-generation",
        source="new_source",
    )

    period = pd.read_parquet(artifacts.fundamental_period_path)
    row = period[
        (period["ts_code"] == "000001.SZ")
        & (period["end_date"] == "20231231")
        & (
            pd.to_datetime(period["availability_date"])
            == pd.Timestamp("2024-04-30")
        )
    ].iloc[-1]
    assert row["fin_roe"] == 0.18
    assert row["source"] == "new_source"


def test_generation_pointer_failure_cleans_final_directory_for_retry(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cn"
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    original_write = fundamental_generation._atomic_write_json
    pointer_failed = False

    def fail_pointer_once(path, payload):
        nonlocal pointer_failed
        if (
            path.name == fundamental_generation.FUNDAMENTAL_POINTER_FILENAME
            and not pointer_failed
        ):
            pointer_failed = True
            raise OSError("simulated pointer write failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        fundamental_generation,
        "_atomic_write_json",
        fail_pointer_once,
    )
    with pytest.raises(OSError, match="simulated pointer write failure"):
        publish_fundamental_generation(
            root=root,
            run_id="retryable-generation",
            tables=tables,
            metadata={"run_id": "retryable-generation"},
        )
    assert not (
        root
        / fundamental_generation.FUNDAMENTAL_GENERATIONS_DIRNAME
        / "retryable-generation"
    ).exists()

    monkeypatch.setattr(
        fundamental_generation,
        "_atomic_write_json",
        original_write,
    )
    paths, pointer = publish_fundamental_generation(
        root=root,
        run_id="retryable-generation",
        tables=tables,
        metadata={"run_id": "retryable-generation"},
    )

    assert pointer["generation_id"] == "retryable-generation"
    assert all(path.exists() for path in paths.values())


def test_generation_pointer_semantic_readback_rejects_disk_forgery(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cn"
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    original_write = fundamental_generation._atomic_write_json

    def forge_pointer_on_disk(path, payload):
        if path.name == fundamental_generation.FUNDAMENTAL_POINTER_FILENAME:
            payload = {**dict(payload), "generation_id": "disk-forged"}
        return original_write(path, payload)

    monkeypatch.setattr(
        fundamental_generation,
        "_atomic_write_json",
        forge_pointer_on_disk,
    )
    with pytest.raises(
        FundamentalGenerationError,
        match="pointer semantic readback mismatch",
    ):
        publish_fundamental_generation(
            root=root,
            run_id="expected-generation",
            tables=tables,
            metadata={"run_id": "expected-generation"},
        )

    assert not (
        root / fundamental_generation.FUNDAMENTAL_POINTER_FILENAME
    ).exists()
    assert not (
        root
        / fundamental_generation.FUNDAMENTAL_GENERATIONS_DIRNAME
        / "expected-generation"
    ).exists()


def test_generation_pointer_readback_failure_restores_predecessor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cn"
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    publish_fundamental_generation(
        root=root,
        run_id="predecessor",
        tables=tables,
        metadata={"run_id": "predecessor"},
    )
    pointer_path = root / fundamental_generation.FUNDAMENTAL_POINTER_FILENAME
    predecessor_bytes = pointer_path.read_bytes()
    predecessor_sha256 = fundamental_generation.pointer_sha256(root)
    original_write = fundamental_generation._atomic_write_json

    def forge_pointer_on_disk(path, payload):
        if path.name == fundamental_generation.FUNDAMENTAL_POINTER_FILENAME:
            payload = {**dict(payload), "generation_id": "disk-forged"}
        return original_write(path, payload)

    monkeypatch.setattr(
        fundamental_generation,
        "_atomic_write_json",
        forge_pointer_on_disk,
    )
    with pytest.raises(
        FundamentalGenerationError,
        match="pointer semantic readback mismatch",
    ):
        publish_fundamental_generation(
            root=root,
            run_id="replacement",
            tables=tables,
            metadata={"run_id": "replacement"},
            expected_pointer_sha256=predecessor_sha256,
        )

    assert pointer_path.read_bytes() == predecessor_bytes
    assert load_fundamental_pointer(root)["generation_id"] == "predecessor"
    assert not (
        root
        / fundamental_generation.FUNDAMENTAL_GENERATIONS_DIRNAME
        / "replacement"
    ).exists()


def test_primary_metadata_mutation_during_streaming_publish_fails_closed(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cn"
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    provider_manifest = {"provider": "tushare", "run_id": "metadata-race"}
    raw_tables = _raw_tables()
    metadata = {
        "run_id": "metadata-race",
        "provider_status": "live_tushare",
        "source_priority": "tushare_primary",
        "source_provenance": "live_tushare_explicit",
        "provider_manifest": provider_manifest,
        "gate2_passed": True,
    }
    capability = fundamental_generation._issue_primary_generation_attestation(
        tables=tables,
        metadata=metadata,
        source="live_tushare",
        provider_manifest_sha256=(
            fundamental_mart._canonical_mapping_sha256(provider_manifest)
        ),
        raw_table_fingerprints=(
            fundamental_mart._raw_table_fingerprints(raw_tables)
        ),
    )
    monkeypatch.setattr(
        fundamental_generation,
        "FUNDAMENTAL_STREAMING_READBACK_MIN_ROWS",
        1,
    )
    original_streaming = fundamental_generation._streaming_parquet_table_evidence
    mutated = False

    def mutate_metadata_during_readback(*args, **kwargs):
        nonlocal mutated
        evidence = original_streaming(*args, **kwargs)
        if not mutated:
            metadata["merge"] = {"tampered_during_publish": True}
            mutated = True
        return evidence

    monkeypatch.setattr(
        fundamental_generation,
        "_streaming_parquet_table_evidence",
        mutate_metadata_during_readback,
    )
    with pytest.raises(
        FundamentalGenerationError,
        match="metadata changed after attestation",
    ):
        publish_fundamental_generation(
            root=root,
            run_id="metadata-race",
            tables=tables,
            metadata=metadata,
            _primary_attestation=capability,
        )

    assert mutated is True
    assert not (
        root / fundamental_generation.FUNDAMENTAL_POINTER_FILENAME
    ).exists()


def test_offline_canonical_rebuild_can_skip_duplicate_raw_csv_snapshots(
    tmp_path,
):
    raw_snapshot_root = tmp_path / "snapshots" / "fundamental"
    artifacts, readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "cn",
        raw_snapshot_root=raw_snapshot_root,
        reports_root=tmp_path / "reports",
        run_id="offline-rebuild",
        source="canonical_raw_offline_rebuild",
        write_raw_snapshots=False,
    )

    assert artifacts.fundamental_daily_path.exists()
    assert readiness["raw_snapshot_written"] is False
    assert not list(raw_snapshot_root.rglob("*.csv"))


def test_disjoint_symbol_recovery_uses_append_merge_path(monkeypatch):
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "end_date": 20231231,
                "availability_date": pd.Timestamp("2024-04-30").date(),
                "fin_roe": 0.10,
            }
        ]
    )
    incoming = pd.DataFrame(
        [
            {
                "ts_code": "000002.SZ",
                "trade_date": "20240510",
                "end_date": "20231231",
                "availability_date": pd.Timestamp("2024-04-30"),
                "fin_roe": 0.20,
            }
        ]
    )

    def fail_slow_key_path(*_args, **_kwargs):
        raise AssertionError("disjoint recovery must not normalize every row")

    monkeypatch.setattr(
        fundamental_mart,
        "_merge_key_values",
        fail_slow_key_path,
    )
    merged, stats = fundamental_mart._merge_fundamental_table(
        existing,
        incoming,
        key_fields=("ts_code", "trade_date", "end_date"),
        quality_fields=("fin_roe",),
    )

    assert set(merged["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert pd.api.types.is_integer_dtype(merged["end_date"])
    assert all(isinstance(value, date) for value in merged["availability_date"])
    assert stats["merge_path"] == "disjoint_symbol_append"


def test_empty_refresh_retains_existing_without_rowwise_key_normalization(
    monkeypatch,
):
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "fin_roe": 0.10,
            }
        ]
    )

    def fail_slow_key_path(*_args, **_kwargs):
        raise AssertionError("empty refresh must not normalize existing rows")

    monkeypatch.setattr(
        fundamental_mart,
        "_merge_key_values",
        fail_slow_key_path,
    )
    merged, stats = fundamental_mart._merge_fundamental_table(
        existing,
        pd.DataFrame(),
        key_fields=("ts_code", "trade_date"),
        quality_fields=("fin_roe",),
    )

    pd.testing.assert_frame_equal(merged, existing)
    assert stats["merge_path"] == "retain_existing_no_incoming"


def test_quarantine_merge_aligns_existing_canonical_schema():
    existing = pd.DataFrame(
        [{"ts_code": "000001.SZ", "end_date": 20231231, "reason": "old"}]
    )
    incoming = pd.DataFrame(
        [{"ts_code": "000002.SZ", "end_date": "20240331", "reason": "new"}]
    )

    merged, _stats = fundamental_mart._merge_quarantine_table(
        existing,
        incoming,
    )

    assert pd.api.types.is_integer_dtype(merged["end_date"])


def test_quarantine_merge_counts_retained_rows_with_duplicate_incoming():
    existing = pd.DataFrame(
        [{"ts_code": "000001.SZ", "reason": "offline-parent"}]
    )
    incoming = pd.DataFrame(
        [
            {"ts_code": "000002.SZ", "reason": "live-child"},
            {"ts_code": "000002.SZ", "reason": "live-child"},
        ]
    )

    merged, stats = fundamental_mart._merge_quarantine_table(
        existing,
        incoming,
    )

    assert len(merged) == 2
    assert stats["retained_existing_rows"] == 1
    assert stats["accepted_incoming_rows"] == 1


def test_readiness_symbol_coverage_is_capped_and_reports_scope_surplus():
    period = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "availability_date": "2024-04-30"},
            {"ts_code": "000002.SZ", "availability_date": "2024-04-30"},
        ]
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                **{field: 1.0 for field in DERIVED_DAILY_FIELDS},
                "sector": "bank",
                "size_bucket": "large",
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20240510",
                **{field: 1.0 for field in DERIVED_DAILY_FIELDS},
                "sector": "health",
                "size_bucket": "small",
            },
        ]
    )

    readiness = fundamental_mart.build_readiness_payload(
        daily,
        period,
        pd.DataFrame(),
        run_id="scope-surplus",
        expected_symbol_count=1,
    )

    assert readiness["symbol_coverage_rate"] == 1.0
    assert readiness["symbol_scope_surplus_count"] == 1


def test_readiness_requires_explicit_expected_scope_when_requested():
    readiness = fundamental_mart.build_readiness_payload(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        run_id="missing-scope",
        require_expected_symbol_scope=True,
    )

    assert readiness["gate2_passed"] is False
    assert readiness["symbol_coverage_rate"] == 0.0
    assert "expected_symbol_scope_missing" in readiness["blockers"]


def test_legacy_fetched_at_fallback_disabled_for_production(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "period": "20231231",
                "metric_name": "operating_cashflow",
                "value": 100.0,
                "fetched_at": "2024-05-10",
            }
        ]
    ).to_parquet(metadata_dir / "fundamental_series.parquet", index=False)

    production = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
        allow_legacy_fallback=False,
    )
    diagnostic = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
        allow_legacy_fallback=True,
    )

    assert production.empty
    assert not diagnostic.empty
    assert diagnostic["source"].str.contains("fetched_at_fallback").any()


def test_metric_matrices_read_canonical_mart_without_legacy(tmp_path):
    artifacts, _readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )
    dates = pd.to_datetime(["2024-04-29", "2024-04-30"])
    matrices, diagnostics = build_fundamental_metric_matrices(
        dates,
        ["000001.SZ"],
        metrics=DERIVED_DAILY_FIELDS,
        mart_root=artifacts.data_root,
        allow_legacy_fallback=False,
    )

    assert diagnostics["legacy_fallback_allowed"] is False
    assert matrices["fin_roe"].loc[pd.Timestamp("2024-04-29"), "000001.SZ"] == 0.09
    assert matrices["fin_roe"].loc[pd.Timestamp("2024-04-30"), "000001.SZ"] == 0.13
    assert matrices["fcf_to_price"].loc[pd.Timestamp("2024-04-30"), "000001.SZ"] > 0.0


def test_daily_basic_uses_local_stock_list_sector(tmp_path, monkeypatch):
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry": "银行"},
            {"ts_code": "000002.SZ", "industry": "全国地产"},
            {"ts_code": "000003.SZ", "industry": "医疗服务"},
        ]
    ).to_parquet(metadata_root / "stock_list.parquet", index=False)
    monkeypatch.setattr(fundamental_mart, "DEFAULT_METADATA_ROOT", metadata_root)
    tables = _raw_tables()
    tables["daily_basic"] = tables["daily_basic"].drop(columns=["sector"])

    artifacts, _readiness = write_fundamental_mart(
        tables,
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )

    daily = pd.read_parquet(artifacts.fundamental_daily_path)
    sector_by_symbol = daily.groupby("ts_code")["sector"].first().to_dict()
    assert sector_by_symbol["000001.SZ"] == "银行"
    assert sector_by_symbol["000002.SZ"] == "全国地产"


def test_fundamental_maintenance_offline_input_writes_expected_artifacts(
    tmp_path,
    monkeypatch,
):
    market_data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    monkeypatch.setattr(
        fundamental_mart,
        "DEFAULT_MARKET_DATA_ROOT",
        market_data_root,
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for table, frame in _raw_tables().items():
        frame.to_parquet(raw_dir / f"{table}.parquet", index=False)

    result = run_cn_fundamental_maintenance(
        market="CN",
        universes="full_a",
        years=5,
        as_of="20240510",
        raw_input_dir=raw_dir,
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
    )

    assert result["provider_status"] == "offline_input"
    assert result["readiness"]["gate2_passed"] is True
    assert result["readiness"]["expected_symbol_count"] == 3
    assert result["readiness"]["raw_row_counts"]["fina_indicator"] > 0
    assert (tmp_path / "clean" / "cn_fundamental" / "latest_manifest.json").exists()
    manifest = json.loads((tmp_path / "clean" / "cn_fundamental" / "latest_manifest.json").read_text())
    assert manifest["raw_row_counts"]["daily_basic"] > 0


def test_live_maintenance_wires_internal_primary_attestation(
    tmp_path,
    monkeypatch,
):
    tables = _raw_tables()
    provider_manifest = _live_provider_manifest(tables)
    monkeypatch.setattr(
        fundamental_mart,
        "_resolve_symbols_from_parquet_universe",
        lambda *_args, **_kwargs: ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    monkeypatch.setattr(
        fundamental_mart,
        "_fetch_tushare_tables",
        lambda *_args, **_kwargs: (tables, provider_manifest),
    )
    data_root = tmp_path / "clean" / "cn_fundamental"

    result = run_cn_fundamental_maintenance(
        market="CN",
        universes="full_a",
        years=5,
        as_of="20240510",
        allow_live=True,
        pro=object(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
    )

    pointer = load_fundamental_pointer(data_root)
    assert result["provider_status"] == "live_tushare"
    assert result["readiness"]["source_priority"] == "tushare_primary"
    assert pointer is not None
    assert pointer["metadata"]["source_priority"] == "tushare_primary"


def test_authoritative_full_rebuild_writes_verified_isolated_generation(
    tmp_path,
    monkeypatch,
):
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    market_data_root = _write_parquet_market_data(tmp_path, symbols)
    components_path = market_data_root / "cn_universe" / "cn_index_components.json"
    components_path.parent.mkdir(parents=True, exist_ok=True)
    components_path.write_text(
        json.dumps({"full_a": symbols}),
        encoding="utf-8",
    )
    scope_sha = hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest()
    membership_path = tmp_path / "stock_basic_membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "list_date": "20240429",
                "effective_from": "20240429",
                "effective_to": "",
                "delist_date": "",
                "industry": f"sector-{index % 3}",
            }
            for index, symbol in enumerate(symbols)
        ]
    ).to_parquet(membership_path, index=False)
    market_pointer = tmp_path / "market_latest.json"
    market_pointer.write_text(
        json.dumps(
                {
                    "snapshot_id": "fixture-20240510",
                    "latest_complete_trade_date": "20240510",
                    "table_root": str(
                        (market_data_root / "parquet" / "cn" / "bars").resolve()
                    ),
                    "coverage": {
                    "expected_scope_count": len(symbols),
                    "expected_scope_sha256": scope_sha,
                    "pit_membership_path": str(membership_path.resolve()),
                    "pit_membership_sha256": hashlib.sha256(
                        membership_path.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    monkeypatch.setattr(fundamental_mart, "DEFAULT_MARKET_DATA_ROOT", market_data_root)
    monkeypatch.setattr(fundamental_mart, "DEFAULT_FUNDAMENTAL_ROOT", canonical_root)
    monkeypatch.setattr(
        fundamental_mart,
        "_merge_fundamental_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authoritative isolated replace must not merge/copy")
        ),
    )
    monkeypatch.setattr(
        fundamental_mart,
        "_merge_quarantine_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authoritative isolated replace must not merge/copy")
        ),
    )
    raw = _raw_tables()
    original_pointer_loader = fundamental_mart.load_fundamental_pointer

    def reject_post_publish_full_load(root):
        if (fundamental_mart._resolve_data_base(root) / "_fundamental_latest.json").exists():
            raise AssertionError(
                "maintenance must reuse the verified published pointer instead of "
                "full-loading tables while source frames are resident"
            )
        return original_pointer_loader(root)

    monkeypatch.setattr(
        fundamental_mart,
        "load_fundamental_pointer",
        reject_post_publish_full_load,
    )

    class _Provider:
        def __getattr__(self, table):
            if table not in fundamental_mart.SOURCE_TABLES:
                raise AttributeError(table)

            def fetch(**kwargs):
                frame = raw[table]
                result = frame[frame["ts_code"] == kwargs["ts_code"]].copy()
                for column in fundamental_mart.SOURCE_REQUIRED_COLUMNS[table]:
                    if column not in result.columns:
                        result[column] = "0" if column == "update_flag" else pd.NA
                if table == "income":
                    result["n_income"] = result["n_income_attr_p"]
                elif table == "cashflow":
                    result["free_cashflow"] = (
                        result["n_cashflow_act"] - result["c_pay_acq_const_fiolta"]
                    )
                elif table == "daily_basic":
                    result["circ_mv"] = result["total_mv"]
                return result

            return fetch

    staging_root = tmp_path / "staging"
    result = run_cn_fundamental_maintenance(
        market="CN",
        universes="full_a",
        years=5,
        as_of="20240510",
        workers=1,
        data_root=staging_root,
        raw_snapshot_root=tmp_path / "snapshots",
        reports_root=tmp_path / "reports",
        allow_live=True,
        pro=_Provider(),
        run_id="fixture-primary-rebuild",
        authoritative_full_rebuild=True,
        canonical_scope_path=components_path,
        canonical_market_pointer_path=market_pointer,
        canonical_membership_path=membership_path,
        checkpoint_root=tmp_path / "checkpoint",
        requests_per_second=0,
        retry_backoff_seconds=0,
    )

    pointer = load_fundamental_pointer(staging_root)
    assert result["generation_id"] == "fixture-primary-rebuild"
    assert result["primary_provenance_verified"] is True
    assert result["readiness"]["gate2_passed"] is True
    assert {
        stats["merge_path"]
        for stats in result["readiness"]["merge"].values()
        if isinstance(stats, dict)
    } == {"authoritative_isolated_replace"}
    assert pointer["manifest"]["metadata"]["provider_manifest"][
        "authoritative_full_rebuild"
    ] is True
    before_generation = pointer["generation_id"]
    with pytest.raises(
        ValueError,
        match="staging pointer already exists|data root must be empty",
    ):
        run_cn_fundamental_maintenance(
            market="CN",
            universes="full_a",
            years=5,
            as_of="20240510",
            workers=1,
            data_root=staging_root,
            raw_snapshot_root=tmp_path / "snapshots-second",
            reports_root=tmp_path / "reports-second",
            allow_live=True,
            pro=_Provider(),
            run_id="fixture-primary-rebuild-second",
            authoritative_full_rebuild=True,
            canonical_scope_path=components_path,
            canonical_market_pointer_path=market_pointer,
            canonical_membership_path=membership_path,
            checkpoint_root=tmp_path / "checkpoint-second",
            requests_per_second=0,
            retry_backoff_seconds=0,
        )
    assert load_fundamental_pointer(staging_root)["generation_id"] == before_generation


def test_v3_rederive_uses_exact_membership_sector_map_and_fixed_timestamp(
    tmp_path,
    monkeypatch,
):
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    raw = {
        table: frame[frame["ts_code"].isin(symbols)].reset_index(drop=True)
        for table, frame in _raw_tables().items()
    }
    membership_path = tmp_path / "membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "effective_from": "20200101",
                "effective_to": "",
                "industry": f"bound-sector-{index}",
            }
            for index, symbol in enumerate(symbols)
        ]
    ).to_parquet(membership_path, index=False)
    membership_bytes = membership_path.read_bytes()
    monkeypatch.setattr(
        fundamental_mart,
        "_load_sector_map",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambient sector metadata must not be read")
        ),
    )

    first, evidence = fundamental_mart.rederive_fundamental_tables_v3(
        raw,
        membership_bytes=membership_bytes,
        membership_sha256=hashlib.sha256(membership_bytes).hexdigest(),
        as_of="20240510",
        symbols=symbols,
        non_blocking_absent_symbols=[],
        run_id="v3-rederive",
        source="live_tushare",
        derivation_timestamp="2024-05-10T12:00:00Z",
    )
    second, second_evidence = fundamental_mart.rederive_fundamental_tables_v3(
        raw,
        membership_bytes=membership_bytes,
        membership_sha256=hashlib.sha256(membership_bytes).hexdigest(),
        as_of="20240510",
        symbols=symbols,
        non_blocking_absent_symbols=[],
        run_id="v3-rederive",
        source="live_tushare",
        derivation_timestamp="2024-05-10T12:00:00Z",
    )

    assert set(first["fundamental_daily"]["sector"]) == {
        "bound-sector-0",
        "bound-sector-1",
        "bound-sector-2",
    }
    assert evidence["sector_map_sha256"] == second_evidence["sector_map_sha256"]
    assert evidence["output_frame_fingerprints"] == second_evidence[
        "output_frame_fingerprints"
    ]
    for table in fundamental_mart.FUNDAMENTAL_TABLES:
        fundamental_mart.assert_frame_semantics_equal(
            first[table],
            second[table],
            label=table,
        )


def test_v3_rederive_expired_membership_requires_bound_exception(tmp_path):
    symbol = "000001.SZ"
    raw = _only_symbol(_raw_tables(), symbol)
    membership_path = tmp_path / "expired-membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "effective_from": "20200101",
                "effective_to": "20240501",
                "industry": "expired-sector",
            }
        ]
    ).to_parquet(membership_path, index=False)
    membership_bytes = membership_path.read_bytes()
    kwargs = {
        "membership_bytes": membership_bytes,
        "membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        "as_of": "20240510",
        "symbols": [symbol],
        "run_id": "v3-expired",
        "source": "live_tushare",
        "derivation_timestamp": "2024-05-10T12:00:00Z",
    }

    with pytest.raises(ValueError, match="no active interval"):
        fundamental_mart.rederive_fundamental_tables_v3(
            raw,
            non_blocking_absent_symbols=[],
            **kwargs,
        )
    _tables, evidence = fundamental_mart.rederive_fundamental_tables_v3(
        raw,
        non_blocking_absent_symbols=[symbol],
        **kwargs,
    )
    assert evidence["expired_fallback_symbol_count"] == 1


def test_offline_partial_scope_fails_closed_without_publishing_pointer(
    tmp_path,
    monkeypatch,
):
    market_data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    monkeypatch.setattr(
        fundamental_mart,
        "DEFAULT_MARKET_DATA_ROOT",
        market_data_root,
    )
    reports_root = tmp_path / "reports" / "fundamental_readiness"
    data_root = tmp_path / "clean" / "cn_fundamental"

    with pytest.raises(FundamentalReadinessError) as exc_info:
        run_cn_fundamental_maintenance(
            market="CN",
            universes="full_a",
            as_of="20240510",
            raw_tables=_only_symbol(_raw_tables(), "000001.SZ"),
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=reports_root,
        )

    readiness = exc_info.value.readiness
    assert readiness["gate2_passed"] is False
    assert readiness["symbol_coverage_rate"] == pytest.approx(1 / 3)
    assert "symbol_coverage_below_95pct" in readiness["blockers"]
    assert list(reports_root.glob("cn_fundamental_20240510_*.json"))
    assert load_fundamental_pointer(data_root) is None


def test_full_a_universe_resolves_from_parquet_serving_inventory(tmp_path):
    data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
    )

    symbols = _resolve_symbols_from_parquet_universe(data_root, ["full_a"])

    assert symbols == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]


def test_full_a_universe_intersects_canonical_components_with_serving(tmp_path):
    data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
    )
    components_path = data_root / "cn_universe" / "cn_index_components.json"
    components_path.parent.mkdir(parents=True)
    components_path.write_text(
        json.dumps(
            {
                "full_a": ["000001.SZ", "000002.SZ", "000003.SZ"],
            }
        ),
        encoding="utf-8",
    )

    symbols = _resolve_symbols_from_parquet_universe(data_root, ["full_a"])

    assert symbols == ["000001.SZ", "000002.SZ", "000003.SZ"]


def test_live_fetch_records_partial_provider_errors(monkeypatch):
    class _FakePro:
        def __init__(self):
            self.calls = []

        def fina_indicator(self, ts_code, start_date="", end_date="", fields=""):
            self.calls.append(("fina_indicator", ts_code, start_date, end_date))
            if ts_code == "000002.SZ":
                raise RuntimeError("quota limited")
            return pd.DataFrame(
                [
                    {
                        "ts_code": ts_code,
                        "end_date": "20231231",
                        "ann_date": "20240430",
                        "f_ann_date": "20240430",
                        "roe_dt": 10.0,
                        "roe": 10.0,
                        "roa": 5.0,
                        "debt_to_assets": 40.0,
                        "netprofit_yoy": 1.0,
                        "ocf_to_profit": 1.0,
                        "update_flag": "0",
                    }
                ]
            )

        def income(self, **kwargs):
            self.calls.append(("income", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def balancesheet(self, **kwargs):
            self.calls.append(("balancesheet", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def cashflow(self, **kwargs):
            self.calls.append(("cashflow", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def daily_basic(self, **kwargs):
            self.calls.append(("daily_basic", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def forecast(self, **kwargs):
            self.calls.append(("forecast", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

    monkeypatch.setattr("quant_investor.market.fundamental_mart.time.sleep", lambda _seconds: None)
    pro = _FakePro()

    tables, manifest = _fetch_tushare_tables(
        ["000001.SZ", "000002.SZ"],
        years=5,
        as_of="20240510",
        workers=4,
        pro=pro,
    )

    assert len(tables["fina_indicator"]) == 1
    assert manifest["requests_failed"] == 1
    assert manifest["errors"][0]["symbol"] == "000002.SZ"
    assert manifest["raw_row_counts"]["fina_indicator"] == 1
    first_start_by_table = {}
    for table, _symbol, start_date, _end_date in pro.calls:
        first_start_by_table.setdefault(table, start_date)
    assert first_start_by_table["daily_basic"] == "20190510"
    assert first_start_by_table["fina_indicator"] == "20170510"
    assert first_start_by_table["income"] == "20170510"
    assert first_start_by_table["forecast"] == "20170510"
    assert manifest["daily_start_date"] == "20190510"
    assert manifest["financial_start_date"] == "20170510"
