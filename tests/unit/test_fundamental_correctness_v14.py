from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import quant_investor.market.fundamental_mart as fundamental_mart
from quant_investor.agent_protocol import AgentStatus, BranchVerdict
from quant_investor.agents.fundamental_agent import (
    FundamentalAgent,
    _BundleFundamentalDataLayer,
)
from quant_investor.branch_contracts import FundamentalSnapshot, UnifiedDataBundle
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.fundamental_branch import FundamentalBranch
from quant_investor.fundamental_components import financial_quality_analyzer
from quant_investor.market.dag.assembly import (
    _aggregate_branch_summaries,
    _build_branch_results,
)
from quant_investor.market.fundamental_generation import load_fundamental_pointer


def _verified_live_raw_tables() -> dict[str, pd.DataFrame]:
    symbols = (
        ("000001.SZ", "bank", 1),
        ("000002.SZ", "industrial", 2),
        ("000003.SZ", "healthcare", 3),
    )
    periods = (
        ("20221231", "20230428", 80.0, 100.0, 10.0, 8.0),
        ("20231231", "20240430", 100.0, 130.0, 20.0, 12.0),
    )
    fina_rows: list[dict] = []
    income_rows: list[dict] = []
    balance_rows: list[dict] = []
    cashflow_rows: list[dict] = []
    daily_rows: list[dict] = []
    forecast_rows: list[dict] = []
    for symbol, sector, index in symbols:
        for end_date, ann_date, profit, cashflow, capex, roe in periods:
            common = {
                "ts_code": symbol,
                "end_date": end_date,
                "ann_date": ann_date,
                "f_ann_date": ann_date,
            }
            fina_rows.append(
                {
                    **common,
                    "roe_dt": roe + index,
                    "roa": 5.0 + index,
                    "debt_to_assets": 45.0 + index,
                }
            )
            income_rows.append(
                {**common, "n_income_attr_p": profit + index}
            )
            balance_rows.append(
                {
                    **common,
                    "total_liab": 400.0 + index,
                    "total_assets": 1000.0 + index,
                }
            )
            cashflow_rows.append(
                {
                    **common,
                    "n_cashflow_act": cashflow + index,
                    "c_pay_acq_const_fiolta": capex,
                }
            )
        for trade_date in ("20240429", "20240430", "20240502", "20240510"):
            daily_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "total_mv": 100000.0 * index,
                    "sector": sector,
                }
            )
        forecast_rows.append(
            {
                "ts_code": symbol,
                "ann_date": "20240429",
                "end_date": "20240630",
                "type": "预增",
                "p_change_min": 5.0 + index,
                "p_change_max": 15.0 + index,
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


def _write_canonical_pointer(
    tmp_path: Path,
    generation_id: str,
    *,
    source_priority: str = "tushare_primary",
) -> Path:
    root = tmp_path / generation_id
    if source_priority == "tushare_primary":
        raw_tables = _verified_live_raw_tables()
        outcomes = [
            {"symbol": symbol, "table": table, "status": "rows"}
            for symbol in ("000001.SZ", "000002.SZ", "000003.SZ")
            for table in fundamental_mart.SOURCE_TABLES
        ]
        provider_manifest = {
            "provider": "tushare",
            "provider_status": "live_tushare",
            "source_priority": "tushare_primary",
            "source_provenance": "live_tushare_explicit",
            "tables": list(fundamental_mart.SOURCE_TABLES),
            "raw_row_counts": {
                table: len(raw_tables[table])
                for table in fundamental_mart.SOURCE_TABLES
            },
            "requests_attempted": len(outcomes),
            "requests_succeeded_with_rows": len(outcomes),
            "requests_empty": 0,
            "requests_failed": 0,
            "symbol_table_outcomes": outcomes,
        }
        attestation = fundamental_mart._issue_live_tushare_attestation(
            "live_tushare",
            provider_manifest,
            raw_tables,
        )
        _artifacts, readiness = fundamental_mart.write_fundamental_mart(
            raw_tables,
            data_root=root,
            raw_snapshot_root=root.parent / f".{generation_id}-raw",
            reports_root=root.parent / f".{generation_id}-readiness",
            run_id=generation_id,
            source="live_tushare",
            provider_manifest=provider_manifest,
            write_raw_snapshots=False,
            publish_on_gate_failure=False,
            _live_tushare_attestation=attestation,
        )
        assert readiness["gate2_passed"] is True
        return root / "_fundamental_latest.json"
    generation_root = root / "_fundamental_generations" / generation_id
    generation_root.mkdir(parents=True)
    table_manifest: dict[str, dict[str, str]] = {}
    table_paths: dict[str, str] = {}
    for table_name in (
        "fundamental_period",
        "fundamental_daily",
        "fundamental_quarantine",
    ):
        table_path = generation_root / f"{table_name}.parquet"
        table_path.write_bytes(f"{generation_id}:{table_name}".encode("utf-8"))
        table_manifest[table_name] = {
            "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()
        }
        table_paths[table_name] = str(table_path.relative_to(root))
    manifest_path = generation_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cn-fundamental-generation.v1",
                "generation_id": generation_id,
                "status": "OK",
                "tables": table_manifest,
                "metadata": {"source_priority": source_priority},
            }
        ),
        encoding="utf-8",
    )
    pointer_path = root / "_fundamental_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "cn-fundamental-pointer.v1",
                "status": "OK",
                "generation_id": generation_id,
                "manifest_path": str(manifest_path.relative_to(root)),
                "tables": table_paths,
                "metadata": {
                    "storage_backend": "parquet_canonical_generation",
                    "source_priority": source_priority,
                },
            }
        ),
        encoding="utf-8",
    )
    return pointer_path


def _canonical_readiness(
    pointer_path: Path,
    generation_id: str,
    *,
    status: str = "pass",
    source_priority: str = "tushare_primary",
) -> dict:
    return {
        "branch_data_readiness": {
            "readiness": {
                "fundamental": {
                    "status": status,
                    "pit_status": "point_in_time",
                    "source_priority": source_priority,
                    "metadata": {
                        "manifest": {
                            "generation_id": generation_id,
                            "pointer_path": str(pointer_path.resolve()),
                            "storage_backend": "parquet_canonical_generation",
                            "source_priority": source_priority,
                        }
                    },
                }
            }
        }
    }


def _bundle(
    *,
    price_date: str = "2026-03-26",
    metadata: dict | None = None,
    fundamentals: dict | None = None,
) -> UnifiedDataBundle:
    default_fundamentals = {
        "trade_date": "2026-03-26",
        "availability_date": "2026-03-20",
        "source_version": "2026-03-20",
        "source": "tushare_fina_indicator",
        "source_priority": "tushare_primary",
        "fin_roe": 0.18,
    }
    if fundamentals is None and metadata:
        generation_id = str(
            metadata.get("branch_data_readiness", {})
            .get("readiness", {})
            .get("fundamental", {})
            .get("metadata", {})
            .get("manifest", {})
            .get("generation_id", "")
        ).strip()
        if generation_id:
            default_fundamentals["fundamental_generation_id"] = generation_id
    return UnifiedDataBundle(
        market="CN",
        symbols=["000001.SZ"],
        symbol_data={
            "000001.SZ": pd.DataFrame(
                {"date": pd.to_datetime([price_date]), "close": [10.0]}
            )
        },
        fundamentals={
            "000001.SZ": (
                fundamentals
                if fundamentals is not None
                else default_fundamentals
            )
        },
        metadata=metadata or {},
    )


def test_bundle_adapter_preserves_missing_fields_in_availability_mask():
    layer = _BundleFundamentalDataLayer(
        {
            "000001.SZ": {
                "trade_date": "2026-03-26",
                "availability_date": "2026-03-20",
                "fin_roe": 0.18,
                "fin_net_profit_yoy": -0.10,
            }
        }
    )

    snapshot = layer.get_point_in_time_fundamental_snapshot(
        "000001.SZ", "2026-03-26"
    )

    assert snapshot.data_quality["available_fields"] == ["roe", "profit_growth"]
    assert "gross_margin" in snapshot.data_quality["missing_fields"]
    assert financial_quality_analyzer(snapshot).score == pytest.approx(0.10)


def test_explicit_zero_is_scored_but_missing_zero_is_not():
    missing = FundamentalSnapshot(
        symbol="000001.SZ",
        available=True,
        roe=0.18,
        data_quality={"available_fields": ["roe"]},
    )
    explicit_zero = FundamentalSnapshot(
        symbol="000001.SZ",
        available=True,
        roe=0.18,
        gross_margin=0.0,
        data_quality={"available_fields": ["roe", "gross_margin"]},
    )

    assert financial_quality_analyzer(missing).score == 0.30
    assert financial_quality_analyzer(explicit_zero).score == pytest.approx(0.20)


def test_bundle_forecast_does_not_invent_analyst_coverage():
    layer = _BundleFundamentalDataLayer(
        {
            "000001.SZ": {
                "trade_date": "2026-03-26",
                "availability_date": "2026-03-20",
                "forecast_ingest_run_id": "forecast-generation-2",
                "forecast_revision": 0.04,
            }
        }
    )

    snapshot = layer.get_earnings_forecast_snapshot("000001.SZ", "2026-03-26")

    assert snapshot.coverage_count == 0
    assert snapshot.data_quality["forecast_kind"] == "corporate_guidance"
    assert snapshot.data_quality["available_fields"] == ["forecast_revision"]


def test_partial_bundle_fundamental_verdict_is_degraded(tmp_path):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    bundle = _bundle(
        metadata=_canonical_readiness(
            pointer_path, "fundamental-generation-7"
        ),
        fundamentals={
            "trade_date": "2026-03-26",
            "availability_date": "2026-03-20",
            "source_version": "2026-03-20",
            "source": "tushare_fina_indicator",
            "source_priority": "tushare_primary",
            "fin_roe": 0.18,
            "forecast_revision": 0.04,
            "fundamental_generation_id": "fundamental-generation-7",
        },
    )

    verdict = FundamentalAgent().run(
        {"data_bundle": bundle, "stock_pool": ["000001.SZ"]}
    )

    assert verdict.status == AgentStatus.DEGRADED
    assert verdict.metadata["degraded_reason"] == "fundamental_evidence_incomplete"
    assert verdict.metadata["horizon_days"] == 30
    assert verdict.metadata["structured_signals"]["quality_breakdown"]
    assert verdict.metadata["structured_signals"]["module_confidences"][
        "financial_quality"
    ] == pytest.approx(1 / 6, abs=1e-4)
    assert verdict.metadata["structured_signals"]["module_coverages"][
        "financial_quality"
    ] == "partial"
    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": "fundamental-generation-7"
    }
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "confirmed"
    }
    assert verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]["canonical_generation_id"] == "fundamental-generation-7"


def test_fundamental_generation_comes_from_canonical_pointer_not_price_date(
    tmp_path,
):
    pointer_paths = {
        generation_id: _write_canonical_pointer(tmp_path, generation_id)
        for generation_id in (
            "fundamental-generation-7",
            "fundamental-generation-8",
        )
    }

    def generation(price_date: str, generation_id: str) -> str:
        verdict = FundamentalAgent().run(
            {
                "data_bundle": _bundle(
                    price_date=price_date,
                    metadata=_canonical_readiness(
                        pointer_paths[generation_id], generation_id
                    ),
                ),
                "stock_pool": ["000001.SZ"],
            }
        )
        return verdict.metadata["fundamental_data_generation_by_symbol"][
            "000001.SZ"
        ]

    assert generation("2026-03-26", "fundamental-generation-7") == generation(
        "2026-03-27", "fundamental-generation-7"
    )
    assert generation(
        "2026-03-27", "fundamental-generation-8"
    ) != generation("2026-03-27", "fundamental-generation-7")


def test_missing_canonical_generation_is_unconfirmed_and_fails_closed():
    verdict = FundamentalAgent().run(
        {"data_bundle": _bundle(), "stock_pool": ["000001.SZ"]}
    )

    assert verdict.status == AgentStatus.DEGRADED
    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert verdict.metadata["fundamental_data_generation_evidence"]["status"] == (
        "UNCONFIRMED"
    )
    assert "fundamental_generation_UNCONFIRMED" in verdict.metadata[
        "degraded_reason"
    ]
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
    assert verdict.final_score > 0.0
    assert likelihoods.fundamental_likelihood == 0.50


def test_pre_v14_primary_pointer_without_durable_provenance_is_unconfirmed(
    tmp_path,
):
    generation_id = "pre-v14-primary"
    pointer_path = _write_canonical_pointer(
        tmp_path,
        generation_id,
        source_priority="manual_offline_snapshot",
    )
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = pointer_path.parent / pointer_payload["manifest_path"]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointer_payload["metadata"]["source_priority"] = "tushare_primary"
    manifest_payload["metadata"]["source_priority"] = "tushare_primary"
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(pointer_path, generation_id),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source_version": "2026-03-20",
                    "source": "tushare_fina_indicator",
                    "source_priority": "tushare_primary",
                    "fundamental_generation_id": generation_id,
                    "fin_roe": 0.18,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
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

    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert "canonical_fundamental_pointer_unverified" in verdict.metadata[
        "fundamental_data_generation_evidence"
    ]["blockers"]
    assert likelihoods.fundamental_likelihood == 0.50
    assert "fundamental" not in likelihoods.metadata["evidence_sources"]


@pytest.mark.parametrize(
    ("source_priority", "source"),
    [
        ("public_structured_fallback", "akshare_structured"),
        ("manual_offline_snapshot", "offline_fundamental_snapshot"),
    ],
)
def test_nonprimary_generation_cannot_contribute_fundamental_likelihood(
    tmp_path,
    source_priority,
    source,
):
    generation_id = "nonprimary-generation"
    pointer_path = _write_canonical_pointer(
        tmp_path,
        generation_id,
        source_priority=source_priority,
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path,
                    generation_id,
                    source_priority=source_priority,
                ),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source_version": "2026-03-20",
                    "source": source,
                    "source_priority": source_priority,
                    "fundamental_generation_id": generation_id,
                    "fin_roe": 0.18,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
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

    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert likelihoods.fundamental_likelihood == 0.50
    assert "fundamental" not in likelihoods.metadata["evidence_sources"]


def test_blocked_fundamental_readiness_cannot_confirm_generation(tmp_path):
    generation_id = "blocked-readiness-generation"
    pointer_path = _write_canonical_pointer(tmp_path, generation_id)
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path,
                    generation_id,
                    status="block",
                )
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert "fundamental_readiness_not_eligible" in verdict.metadata[
        "fundamental_data_generation_evidence"
    ]["blockers"]


def test_nonprimary_row_cannot_borrow_primary_generation_lineage(tmp_path):
    generation_id = "primary-generation"
    pointer_path = _write_canonical_pointer(tmp_path, generation_id)
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(pointer_path, generation_id),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source_version": "2026-03-20",
                    "source": "offline_fundamental_snapshot",
                    "source_priority": "manual_offline_snapshot",
                    "fundamental_generation_id": generation_id,
                    "fin_roe": 0.18,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    evidence = verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert (
        "financial_quality:source_priority_not_tushare_primary"
        in evidence["blockers"]
    )


def test_pointer_without_symbol_pit_lineage_is_unconfirmed(tmp_path):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path, "fundamental-generation-7"
                ),
                fundamentals={"trade_date": "2026-03-26", "fin_roe": 0.18},
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]["blockers"][0] == "symbol_pit_lineage_unconfirmed"


def test_symbol_row_without_generation_id_cannot_confirm_pointer_lineage(
    tmp_path,
):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path, "fundamental-generation-7"
                ),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source_version": "2026-03-20",
                    "source": "tushare_fina_indicator",
                    "source_priority": "tushare_primary",
                    "fin_roe": 0.18,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    evidence = verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }
    assert (
        "financial_quality:canonical_generation_id_missing"
        in evidence["blockers"]
    )


def test_nonexistent_pointer_cannot_confirm_generation(tmp_path):
    missing_pointer = tmp_path / "missing" / "_fundamental_latest.json"
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    missing_pointer, "fundamental-generation-7"
                )
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    assert "canonical_fundamental_pointer_unverified" in verdict.metadata[
        "fundamental_data_generation_evidence"
    ]["blockers"]


def test_relative_canonical_pointer_is_bound_inside_runtime_root(
    tmp_path,
    monkeypatch,
):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    pointer_path = _write_canonical_pointer(
        runtime_root / "data" / "parquet" / "cn",
        "fundamental-generation-7",
    )
    readiness = _canonical_readiness(
        pointer_path,
        "fundamental-generation-7",
    )
    readiness["branch_data_readiness"]["readiness"]["fundamental"][
        "metadata"
    ]["manifest"]["pointer_path"] = str(pointer_path.relative_to(runtime_root))
    monkeypatch.chdir(runtime_root)
    loaded_pointer = load_fundamental_pointer(
        pointer_path.parent.relative_to(runtime_root)
    )

    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(metadata=readiness),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": "fundamental-generation-7"
    }
    evidence = verdict.metadata["fundamental_data_generation_evidence"]
    assert evidence["pointer_bound"] is True
    assert evidence["resolved_pointer_path"] == str(pointer_path.resolve())
    assert loaded_pointer is not None
    assert Path(loaded_pointer["pointer_path"]).is_absolute()
    assert Path(loaded_pointer["pointer_path"]).resolve() == pointer_path.resolve()


@pytest.mark.parametrize(
    "escape_kind",
    ["parent", "relative_symlink", "absolute_symlink"],
)
def test_canonical_pointer_rejects_escape_and_symlink(
    tmp_path,
    monkeypatch,
    escape_kind,
):
    runtime_root = tmp_path / "runtime"
    external_root = tmp_path / "external"
    runtime_root.mkdir()
    pointer_path = _write_canonical_pointer(
        external_root,
        "fundamental-generation-7",
    )
    if escape_kind == "parent":
        readiness_path = str(pointer_path.relative_to(tmp_path))
        readiness_path = f"../{readiness_path}"
    else:
        (runtime_root / "linked-data").symlink_to(
            external_root,
            target_is_directory=True,
        )
        linked_pointer = (
            runtime_root
            / "linked-data"
            / pointer_path.relative_to(external_root)
        )
        readiness_path = str(
            linked_pointer
            if escape_kind == "absolute_symlink"
            else linked_pointer.relative_to(runtime_root)
        )
    readiness = _canonical_readiness(
        pointer_path,
        "fundamental-generation-7",
    )
    readiness["branch_data_readiness"]["readiness"]["fundamental"][
        "metadata"
    ]["manifest"]["pointer_path"] = readiness_path
    monkeypatch.chdir(runtime_root)

    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(metadata=readiness),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    assert "canonical_fundamental_pointer_unverified" in verdict.metadata[
        "fundamental_data_generation_evidence"
    ]["blockers"]


def test_future_pit_row_is_blocked_and_generation_is_unconfirmed(tmp_path):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    fundamentals = {
        "trade_date": "2026-03-26",
        "availability_date": "2026-03-30",
        "source_version": "2026-03-30",
        "fin_roe": 0.18,
    }
    layer = _BundleFundamentalDataLayer({"000001.SZ": fundamentals})

    snapshot = layer.get_point_in_time_fundamental_snapshot(
        "000001.SZ", "2026-03-26"
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path, "fundamental-generation-7"
                ),
                fundamentals=fundamentals,
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert snapshot.available is False
    assert snapshot.data_quality["pit_status"] == "blocked"
    assert "pit_publish_time_after_as_of" in snapshot.data_quality["pit_blockers"]
    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    assert verdict.metadata["fundamental_data_generation_status_by_symbol"] == {
        "000001.SZ": "UNCONFIRMED"
    }


def test_valid_forecast_cannot_mask_future_fundamental_lineage(tmp_path):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path, "fundamental-generation-7"
                ),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-30",
                    "source_version": "2026-03-30",
                    "fin_roe": 0.18,
                    "forecast_ann_date": "2026-03-20",
                    "forecast_ingest_run_id": "forecast-generation-2",
                    "forecast_revision": 0.04,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    evidence = verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]
    assert evidence["status"] == "UNCONFIRMED"
    assert "financial_quality:pit_publish_time_after_as_of" in evidence["blockers"]


def test_unconfirmed_source_lineage_cannot_bind_canonical_generation(tmp_path):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path, "fundamental-generation-7"
                ),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source_version": "2026-03-20",
                    "source": "UNCONFIRMED",
                    "fin_roe": 0.18,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    evidence = verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]
    assert "financial_quality:source_lineage_unconfirmed" in evidence["blockers"]


@pytest.mark.parametrize(
    ("lineage_fields", "expected_blocker"),
    [
        (
            {"source_version": "2026-03-30"},
            "financial_quality:revision_id_after_as_of",
        ),
        (
            {
                "source_version": "2026-03-20",
                "fundamental_generation_id": "different-generation",
            },
            "financial_quality:canonical_generation_mismatch",
        ),
    ],
)
def test_future_or_mismatched_symbol_lineage_is_unconfirmed(
    tmp_path,
    lineage_fields,
    expected_blocker,
):
    pointer_path = _write_canonical_pointer(
        tmp_path, "fundamental-generation-7"
    )
    verdict = FundamentalAgent().run(
        {
            "data_bundle": _bundle(
                metadata=_canonical_readiness(
                    pointer_path, "fundamental-generation-7"
                ),
                fundamentals={
                    "trade_date": "2026-03-26",
                    "availability_date": "2026-03-20",
                    "source": "local_fundamental_mart",
                    "fin_roe": 0.18,
                    **lineage_fields,
                },
            ),
            "stock_pool": ["000001.SZ"],
        }
    )

    assert verdict.metadata["fundamental_data_generation_by_symbol"] == {
        "000001.SZ": ""
    }
    evidence = verdict.metadata["fundamental_data_generation_evidence"][
        "symbol_pit_evidence"
    ]["000001.SZ"]
    assert expected_blocker in evidence["blockers"]


def test_assembly_preserves_degraded_status_and_fundamental_metadata():
    verdict = BranchVerdict(
        agent_name="FundamentalAgent",
        thesis="partial evidence",
        status=AgentStatus.DEGRADED,
        final_score=0.1,
        final_confidence=0.4,
        metadata={
            "branch_name": "fundamental",
            "reliability": 0.45,
            "structured_signals": {"x": 1},
            "fundamental_data_generation_by_symbol": {
                "000001.SZ": "fundamental-generation-7"
            },
            "fundamental_data_generation_status_by_symbol": {
                "000001.SZ": "confirmed"
            },
            "fundamental_data_generation_evidence": {
                "status": "confirmed"
            },
        },
    )
    research = {"000001.SZ": {"fundamental": verdict}}

    summaries = _aggregate_branch_summaries(research)
    results = _build_branch_results(research, summaries)

    assert summaries["fundamental"].status == AgentStatus.DEGRADED
    assert summaries["fundamental"].metadata["degraded_symbols"] == ["000001.SZ"]
    assert results["fundamental"].metadata["degraded_reason"] == (
        "symbol_research_degraded"
    )
    assert results["fundamental"].metadata["reliability"] == 0.45
    assert results["fundamental"].signals[
        "structured_signals_by_symbol"
    ]["000001.SZ"] == {"x": 1}
    assert results["fundamental"].metadata[
        "fundamental_data_generation_by_symbol"
    ] == {"000001.SZ": "fundamental-generation-7"}
    assert results["fundamental"].metadata[
        "fundamental_data_generation_status_by_symbol"
    ] == {"000001.SZ": "confirmed"}
    assert results["fundamental"].metadata[
        "fundamental_data_generation_evidence_by_symbol"
    ]["000001.SZ"] == {"status": "confirmed"}
    assert results["fundamental"].metadata[
        "fundamental_data_generation_evidence"
    ] == {"000001.SZ": {"status": "confirmed"}}


def test_symbol_conclusion_includes_primary_negative_driver():
    branch = object.__new__(FundamentalBranch)

    conclusion = branch._build_symbol_conclusion(
        symbol="000001.SZ",
        available_modules=["financial_quality"],
        missing_modules=["valuation"],
        support_points=["财务质量: ROE 处于较优区间。"],
        drag_points=["财务质量: 盈利增速转弱。"],
    )

    assert "主要风险为财务质量: 盈利增速转弱" in conclusion
