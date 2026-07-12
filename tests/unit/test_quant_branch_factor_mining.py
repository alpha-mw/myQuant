from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.retest_aquant_alpha_mix_8gate as retest
import scripts.daily_factor_mining_automation as daily
import scripts.mine_quant_branch_factors as mining
from quant_investor.factors.governance import (
    FactorAdmissionDecision,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.registry_store import (
    FactorRegistryConflictError,
    FactorRegistryMalformedError,
)
from quant_investor.factors.runtime import MinedFactorRegistry
from scripts.mine_quant_branch_factors import (
    DEFAULT_DIVERSITY_POLICY,
    MiningCandidate,
    apply_candidate_diversity_governance,
    apply_production_family_governance,
    apply_production_candidate_registry_updates,
    build_candidate_catalog,
    candidate_primitive_lineage,
    candidate_maturity_context,
    compute_formulaic_signal,
)
from scripts.retest_aquant_alpha_mix_8gate import (
    RetestContext,
    compute_existing_composite,
)


def _context() -> RetestContext:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    columns = ["000001.SZ", "000002.SZ"]
    matrix = pd.DataFrame(1.0, index=dates, columns=columns)
    return RetestContext(
        frames={},
        universe_by_symbol={symbol: "fixture" for symbol in columns},
        adj_close=matrix,
        volume=matrix,
        amount=matrix,
        forward_return=matrix,
        rebalance_dates=list(dates),
        biweekly_dates=list(dates),
        existing_composite=None,
        existing_blocker="",
    )


def test_candidate_maturity_context_uses_signal_coverage_only():
    context = _context()
    signal = pd.DataFrame(
        [
            [None, None],
            [1.0, None],
            [2.0, None],
            [3.0, 4.0],
        ],
        index=context.adj_close.index,
        columns=context.adj_close.columns,
    )

    matured, start = candidate_maturity_context(
        context,
        signal,
        base_start="2024-01-02",
        min_signal_coverage=0.60,
    )

    assert start == "2024-01-05"
    assert list(matured.adj_close.index) == [pd.Timestamp("2024-01-05")]


def test_full_a_coverage_uses_sector_and_size_not_single_universe_label():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    symbols = ["A", "B", "C", "D"]
    signal = pd.DataFrame(1.0, index=dates, columns=symbols)

    metrics = retest._coverage_metrics(
        signal,
        dates,
        {"A": "bank", "B": "bank", "C": "health", "D": "health"},
        {"A": "large", "B": "large", "C": "small", "D": "small"},
    )

    assert metrics["max_sector_coverage_share"] == 0.5
    assert metrics["max_size_bucket_coverage_share"] == 0.5


def test_exposure_neutralization_only_computes_requested_dates():
    context = _context()
    context.sector_by_symbol = {
        "000001.SZ": "bank",
        "000002.SZ": "health",
    }
    context.size_bucket_by_symbol = {
        "000001.SZ": "large",
        "000002.SZ": "small",
    }
    requested = [context.adj_close.index[-1]]

    neutral = retest._neutralize_by_exposure(
        context.adj_close,
        context,
        requested,
    )

    assert list(neutral.index) == requested


def test_strict_exposure_loader_uses_bounded_share_reconstruction(tmp_path):
    root = tmp_path / "cn"
    stock_path = root / "dag_core_raw" / "table=stock_basic" / "part.parquet"
    daily_path = root / "daily_basic" / "part.parquet"
    ext_path = (
        root / "dag_core_raw" / "table=daily_basic_ext" / "part.parquet"
    )
    stock_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    ext_path.parent.mkdir(parents=True)
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    pd.DataFrame(
        {
            "ts_code": symbols,
            "industry": ["bank", "bank", "health", "health"],
        }
    ).to_parquet(stock_path, index=False)
    pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": int(date.strftime("%Y%m%d")),
                "total_mv": float((index + 1) * 100),
            }
            for date in pd.to_datetime(["2024-01-02", "2024-01-03"])
            for index, symbol in enumerate(symbols[:3])
        ]
    ).to_parquet(daily_path, index=False)
    pd.DataFrame(
        {
            "ts_code": symbols,
            "trade_date": [20240105] * 4,
            "total_share": [10.0] * 4,
            "total_mv": [100.0, 200.0, 300.0, 400.0],
            "close": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_parquet(ext_path, index=False)

    def catalog_row(path, logical_table, latest_date):
        return {
            "status": "ok",
            "logical_table": logical_table,
            "row_count": len(pd.read_parquet(path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "snapshot_id": f"{logical_table}-snapshot",
            "latest_date": latest_date,
        }

    (root / "_catalog.json").write_text(
        json.dumps(
            {
                "tables": {
                    "dag_core_raw/stock_basic": catalog_row(
                        stock_path,
                        "dag_core_raw/stock_basic",
                        "20240105",
                    ),
                    "daily_basic": catalog_row(
                        daily_path,
                        "daily_basic",
                        "20240103",
                    ),
                    "dag_core_raw/daily_basic_ext": catalog_row(
                        ext_path,
                        "dag_core_raw/daily_basic_ext",
                        "20240105",
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    close = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]],
        index=dates,
        columns=symbols,
    )

    _sectors, _sizes, by_date, metadata = (
        retest.load_fundamental_exposure_maps(
            mart_root=root,
            symbols=symbols,
            as_of=pd.Timestamp("2024-01-05"),
            evaluation_dates=list(dates),
            close_by_date=close,
        )
    )

    assert metadata["status"] == "ready"
    assert metadata["catalog_validated"] is True
    assert metadata["pit_size_pair_coverage_ratio"] == pytest.approx(0.75)
    assert metadata["reconstructed_size_pair_ratio"] == pytest.approx(0.25)
    assert by_date["000004.SZ"].notna().all()


def test_auto_analysis_start_uses_point_in_time_observable_universe():
    dates = pd.date_range("2021-01-04", periods=4, freq="B")
    prices = pd.DataFrame(
        {
            "old_a": [1.0, 1.1, 1.2, 1.3],
            "old_b": [2.0, 2.1, 2.2, 2.3],
            "new_ipo": [None, None, 3.0, 3.1],
        },
        index=dates,
    )
    context = RetestContext(
        frames={},
        universe_by_symbol={column: "full_a" for column in prices},
        adj_close=prices,
        volume=prices,
        amount=prices,
        forward_return=prices,
        rebalance_dates=list(dates),
        biweekly_dates=list(dates),
        existing_composite=None,
    )

    start = mining._auto_analysis_start_date(context, 0.95)

    assert start == dates[0]


def test_compute_formulaic_signal_rank_blends_primitives():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    columns = ["000001.SZ", "000002.SZ"]
    left = pd.DataFrame([[1.0, 2.0], [4.0, 3.0]], index=dates, columns=columns)
    right = pd.DataFrame(
        [[9.0, 8.0], [5.0, 6.0]],
        index=dates,
        columns=columns,
    )
    candidate = MiningCandidate(
        name="formula_fixture",
        family="fixture",
        category="formulaic_research",
        implementation="research_formula:rank_blend",
        description="fixture",
        params={"left": "left", "right": "right", "left_weight": 0.25},
    )

    actual = compute_formulaic_signal(
        candidate,
        {"left": left, "right": right},
    )
    expected = left.rank(axis=1, pct=True).mul(0.25) + right.rank(
        axis=1,
        pct=True,
    ).mul(0.75)

    pd.testing.assert_frame_equal(actual, expected)


def test_lineage_normalizes_residual_formula_and_fundamental_windows():
    formula = MiningCandidate(
        name="formula_mom120_np_yoy_resid_w30",
        family="momentum_fundamental_residual",
        category="formulaic_research",
        implementation="research_formula:rank_blend",
        description="fixture",
        params={
            "left": "momentum_120",
            "right": "fin_net_profit_yoy_resid_existing",
            "left_weight": 0.30,
        },
    )
    direct = MiningCandidate(
        name="fund_fin_net_profit_yoy_60d",
        family="fin_net_profit_yoy",
        category="growth",
        implementation="aquant_expression:fund_fin_net_profit_yoy_60d",
        description="fixture",
        expression="cs_rank(ts_mean(fin_net_profit_yoy, 60))",
        window=60,
    )

    formula_lineage = candidate_primitive_lineage(formula)
    direct_lineage = candidate_primitive_lineage(direct)

    assert formula_lineage["dominant_primitives"] == ["fin_net_profit_yoy"]
    assert set(formula_lineage["primitive_lineage"]) == {
        "fin_net_profit_yoy",
        "price_momentum",
    }
    assert direct_lineage["dominant_primitives"] == ["fin_net_profit_yoy"]
    assert direct_lineage["lineage_extraction_status"] == "complete"


def test_lineage_uses_composite_weights_not_window_lengths():
    candidate = next(
        item
        for item in build_candidate_catalog((5, 20, 60, 120))
        if item.name == "pv_blend_volstab19x2_mom90_amihud5_w70"
    )

    lineage = candidate_primitive_lineage(candidate)

    assert lineage["dominant_primitives"] == ["volume"]
    assert lineage["primitive_contributions"] == pytest.approx(
        {
            "amihud_illiquidity": 0.12,
            "price_momentum": 0.18,
            "volume": 0.70,
        }
    )


def test_diversity_governance_collapses_current_six_to_runtime_champion():
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    columns = [f"{index:06d}.SZ" for index in range(30)]
    base = pd.DataFrame(
        [range(30), range(1, 31), range(2, 32), range(3, 33)],
        index=dates,
        columns=columns,
        dtype=float,
    )
    candidates = [
        MiningCandidate(
            name=f"formula_mom120_np_yoy_resid_w{weight}",
            family="momentum_fundamental_residual",
            category="formulaic_research",
            implementation="research_formula:rank_blend",
            description="fixture",
            params={
                "left": "momentum_120",
                "right": "fin_net_profit_yoy_resid_existing",
                "left_weight": weight / 100.0,
            },
        )
        for weight in (30, 25, 20)
    ]
    candidates.extend(
        [
            MiningCandidate(
                name=name,
                family="fin_net_profit_yoy",
                category="growth",
                implementation=f"aquant_expression:{name}",
                description="fixture",
                expression=expression,
                window=window,
            )
            for name, expression, window in (
                (
                    "fund_fin_net_profit_yoy",
                    "cs_rank(fin_net_profit_yoy)",
                    None,
                ),
                (
                    "fund_fin_net_profit_yoy_60d",
                    "cs_rank(ts_mean(fin_net_profit_yoy, 60))",
                    60,
                ),
                (
                    "fund_fin_net_profit_yoy_20d",
                    "cs_rank(ts_mean(fin_net_profit_yoy, 20))",
                    20,
                ),
            )
        ]
    )
    icirs = [1.10, 1.08, 1.06, 0.86, 0.85, 0.84]
    results = []
    for candidate, icir in zip(candidates, icirs):
        item = _qualified_result(candidate.name)
        item.update(
            family=candidate.family,
            category=candidate.category,
            implementation=candidate.implementation,
            expression=candidate.expression,
            window=candidate.window,
            params=dict(candidate.params or {}),
        )
        item["metrics"]["icir"] = icir
        item["metrics"]["mean_rankic"] = icir / 20.0
        results.append(item)

    audit = apply_candidate_diversity_governance(
        results,
        candidates_by_name={item.name: item for item in candidates},
        signals_by_name={item.name: base for item in candidates},
        rebalance_dates=list(dates),
    )

    assert audit["raw_qualified_count"] == 6
    assert audit["runtime_eligible_count"] == 3
    assert audit["selected_champions"] == ["fund_fin_net_profit_yoy"]
    direct = next(
        item for item in results if item["name"] == "fund_fin_net_profit_yoy"
    )
    assert direct["diversity_selection"]["final_registry_write_eligible"]
    skipped = [
        item for item in results if item["name"] != "fund_fin_net_profit_yoy"
    ]
    assert all(
        not item["diversity_selection"]["final_registry_write_eligible"]
        for item in skipped
    )


def test_diversity_governance_fails_closed_for_missing_pairwise_evidence():
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    columns = [f"{index:06d}.SZ" for index in range(30)]
    matrix = pd.DataFrame(
        [range(30), range(1, 31)], index=dates, columns=columns, dtype=float
    )
    candidates = [
        MiningCandidate(
            name="mom",
            family="momentum",
            category="momentum",
            implementation="price_volume:pv_momentum_20d",
            description="fixture",
            window=20,
        ),
        MiningCandidate(
            name="liquidity",
            family="high_dollar_volume",
            category="capacity",
            implementation="price_volume:pv_high_dollar_volume_20d",
            description="fixture",
            window=20,
        ),
    ]
    results = [_qualified_result(item.name) for item in candidates]
    for result, candidate in zip(results, candidates):
        result.update(
            family=candidate.family,
            category=candidate.category,
            implementation=candidate.implementation,
        )

    audit = apply_candidate_diversity_governance(
        results,
        candidates_by_name={item.name: item for item in candidates},
        signals_by_name={item.name: matrix for item in candidates},
        rebalance_dates=list(dates),
    )

    assert audit["correlation_champion_count"] == 0
    pairs = {
        tuple(sorted(pair)) for pair in audit["incomplete_required_pairs"]
    }
    assert pairs == {
        ("liquidity", "mom")
    }
    assert all(
        item["diversity_selection"]["status"] == "evidence_missing"
        for item in results
    )


def test_compute_existing_composite_supports_promoted_blend_factor():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    columns = ["000001.SZ", "000002.SZ", "000003.SZ"]
    adj_close = pd.DataFrame(
        {
            "000001.SZ": range(100, 220),
            "000002.SZ": range(80, 200),
            "000003.SZ": range(120, 240),
        },
        index=dates,
        dtype=float,
    )
    volume = pd.DataFrame(
        {
            "000001.SZ": range(1000, 1120),
            "000002.SZ": range(1200, 1320),
            "000003.SZ": range(900, 1020),
        },
        index=dates,
        dtype=float,
    )
    amount = adj_close.mul(volume)
    record = FactorRecord(
        name="pv_blend_volstab19x2_mom90_amihud5_w75",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:pv_blend_volstab19x2_mom90_amihud5_w75",
        weight=0.05,
        gate_results=[GateResult.from_dict(row) for row in _gate_rows()],
    )

    composite, blocker = compute_existing_composite(
        MinedFactorRegistry.from_records([record]),
        adj_close,
        volume,
        amount,
    )

    assert blocker == ""
    assert composite is not None
    assert composite.index.equals(adj_close.index)
    assert set(composite.columns) == set(columns)


def test_load_daily_frames_uses_parquet_reader_when_backend_strict(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range("2026-01-02", periods=2, freq="B")

    class FakeMarketDataReader:
        def __init__(self, *, market, data_root, mode_policy):
            assert market == "CN"
            assert data_root == tmp_path / "missing_clean"
            assert mode_policy == "strict"

        def list_symbols(self, *, category):
            return {
                "hs300": ["000001.SZ"],
                "zz500": ["600000.SH"],
            }.get(category, [])

        def read_symbol_frames(self, symbols, **_kwargs):
            return {
                symbol: SimpleNamespace(
                    frame=pd.DataFrame(
                        {
                            "ts_code": [symbol, symbol],
                            "trade_date": dates.strftime("%Y%m%d"),
                            "close": [10.0, 11.0],
                            "vol": [100.0, 110.0],
                            "amount": [1000.0, 1210.0],
                        }
                    )
                )
                for symbol in symbols
            }

    monkeypatch.setenv("MYQUANT_MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MYQUANT_MARKET_DATA_MODE_POLICY", "strict")
    monkeypatch.setattr(
        retest,
        "MarketDataReader",
        FakeMarketDataReader,
        raising=False,
    )

    frames, universe_by_symbol = retest.load_daily_frames(
        tmp_path / "missing_clean",
        ("hs300", "zz500"),
    )

    assert sorted(frames) == ["000001.SZ", "600000.SH"]
    assert universe_by_symbol == {
        "000001.SZ": "hs300",
        "600000.SH": "zz500",
    }
    assert set(frames["000001.SZ"].columns) >= {
        "symbol",
        "trade_date",
        "close",
        "vol",
        "amount",
    }


def _gate_rows() -> list[dict[str, object]]:
    return [
        {
            "gate_id": gate_id,
            "gate_key": f"gate_{gate_id}",
            "title": f"Gate {gate_id}",
            "passed": True,
            "reasons": [],
            "metrics": {},
            "severity": "info",
        }
        for gate_id in range(1, 9)
    ]


def _qualified_result(name: str = "daily_fixture_factor") -> dict[str, object]:
    return {
        "name": name,
        "family": "momentum",
        "category": "momentum",
        "implementation": "price_volume:pv_momentum_20d",
        "expression": "",
        "window": 20,
        "params": {},
        "description": "fixture daily mining factor",
        "effective_analysis_start_date": "2024-01-02",
        "decision": FactorAdmissionDecision.PRODUCTION_CANDIDATE.value,
        "target_state": FactorLifecycleState.PRODUCTION_CANDIDATE.value,
        "gates_passed": 8,
        "passed_gate_ids": list(range(1, 9)),
        "failed_gate_ids": [],
        "gate_results": _gate_rows(),
        "metrics": {
            "horizon_days": 30,
            "mean_rankic": 0.035,
            "icir": 0.85,
            "positive_ic_ratio": 0.62,
            "top_bottom_spread": 0.04,
            "master_return_delta": 0.012,
        },
        "blockers": [],
        "summary": "passes fixture gates",
        "primitive_lineage": ["price_momentum"],
        "primitive_contributions": {"price_momentum": 1.0},
        "dominant_primitives": ["price_momentum"],
        "lineage_extraction_status": "complete",
        "runtime_write_eligible": True,
        "runtime_write_blockers": [],
        "diversity_selection": {
            "policy_version": DEFAULT_DIVERSITY_POLICY.version,
            "policy_hash": DEFAULT_DIVERSITY_POLICY.policy_hash,
            "status": "champion_not_applicable_single_candidate",
            "skip_reason": "",
            "redundancy_stage": "",
            "family_champion": name,
            "lineage_component_id": "lineage-001",
            "lineage_champion": name,
            "correlation_cluster_id": "correlation-001",
            "cluster_champion": name,
            "max_abs_candidate_corr": None,
            "valid_corr_date_count": 0,
            "final_registry_write_eligible": True,
        },
    }


def _write_registry(path, registry: MinedFactorRegistry) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": registry.schema_version,
                "metadata": registry.metadata,
                "factors": [record.to_dict() for record in registry.factors],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_mining_publishes_restricted_window_exposure_evidence(
    monkeypatch,
    tmp_path,
):
    full_context = _context()
    full_context.exposure_metadata = {
        "status": "ready",
        "window": "full_history",
    }
    restricted_evidence = {
        "status": "blocked",
        "window": "resolved_analysis_window",
        "coverage_ratio": 0.50,
    }
    monkeypatch.setattr(
        mining,
        "build_context",
        lambda **_kwargs: full_context,
    )
    monkeypatch.setattr(
        mining,
        "load_fundamental_exposure_maps",
        lambda **_kwargs: ({}, {}, pd.DataFrame(), restricted_evidence),
    )
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    output_dir = tmp_path / "mining"
    args = mining.parse_args(
        [
            "--no-price-volume",
            "--no-fundamental",
            "--no-formulaic",
            "--analysis-start-date",
            "2024-01-03",
            "--registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "restricted-exposure-fixture",
        ]
    )

    payload = mining.run_mining(args)
    readback = json.loads(
        (output_dir / "quant_branch_factor_mining_results.json").read_text()
    )

    assert payload["resolved_analysis_start_date"] == "2024-01-03"
    assert payload["factor_exposure_evidence"] == restricted_evidence
    assert readback["factor_exposure_evidence"] == restricted_evidence


def _market_evidence(symbol_count: int = 3) -> dict[str, object]:
    return {
        "backend": "parquet",
        "mode_policy": "strict",
        "pointer_status": "OK",
        "snapshot_id": "fixture-snapshot",
        "coverage_complete": True,
        "expected_symbol_count": symbol_count,
        "loaded_symbol_count": symbol_count,
        "table_root_exists": True,
        "serving_root_exists": True,
        "manifest_exists": True,
        "factor_exposure_evidence": {
            "status": "ready",
            "source": "strict_parquet_hybrid_market_cap_exposure",
            "coverage_ratio": 0.96,
            "catalog_validated": True,
            "size_policy": (
                "same_trade_date_total_mv_then_asof_total_share_times_close"
            ),
            "evaluation_date_coverage_ratio": 1.0,
            "min_cross_section_coverage_ratio": 0.99,
            "combined_size_pair_coverage_ratio": 0.99,
            "pit_size_pair_coverage_ratio": 0.72,
            "reconstructed_size_pair_ratio": 0.27,
            "share_reference_covers_evaluation_end": True,
            "sector_count": 100,
            "size_bucket_count": 3,
        },
    }


def _write_mining_report(
    path: Path,
    *,
    run_id: str,
    results: list[dict[str, object]],
    universes: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "universes": universes or ["full_a"],
                "results": results,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_candidate_catalog_covers_daily_mining_dimensions():
    candidates = build_candidate_catalog((5, 20, 60, 120))
    families = {candidate.family for candidate in candidates}
    categories = {candidate.category for candidate in candidates}

    assert len(candidates) >= 120
    assert {
        "momentum",
        "short_reversal",
        "volume_stability",
        "low_dollar_volume",
        "high_dollar_volume",
        "amihud_illiquidity",
        "volatility_penalty",
        "downside_volatility",
        "price_efficiency",
        "dollar_volume_growth",
        "fin_roe",
        "momentum_liquidity",
        "fundamental_quality_value",
    }.issubset(families)
    assert {
        "momentum",
        "reversal",
        "trading_activity",
        "liquidity",
        "capacity",
        "risk",
        "trend_quality",
        "formulaic_research",
    }.issubset(categories)


def test_registry_write_adds_zero_weight_production_candidate(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    source_notes = [{"title": "fixture source", "url": "https://example.com"}]

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[_qualified_result()],
        run_timestamp="2026-06-07T04:30:00",
        run_id="daily_factor_mining_fixture",
        report_path="reports/factor_governance/daily_mining/fixture.json",
        owner="test owner",
        source_notes=source_notes,
        horizon_days=30,
        max_candidates=5,
        journal_path=tmp_path / "reports" / "registry_mutation.json",
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    assert manifest["status"] == "updated"
    assert manifest["registry_mutation_manifest"]["status"] == "applied"
    assert (
        manifest["registry_mutation_manifest"]["changed_metadata_count"] == 8
    )
    assert Path(manifest["registry_mutation_manifest_path"]).exists()
    assert manifest["before_registry_sha256"]
    assert manifest["after_registry_sha256"]
    assert manifest["before_registry_sha256"] != manifest[
        "after_registry_sha256"
    ]
    assert manifest["written_factors"] == ["daily_fixture_factor"]
    assert registry.selectable_factors() == []
    assert registry.non_selectable_reasons() == {
        "daily_fixture_factor": "state=production_candidate"
    }
    record = registry.factors[0]
    assert record.state == FactorLifecycleState.PRODUCTION_CANDIDATE
    assert record.weight == 0.0
    assert record.admission_decision == (
        FactorAdmissionDecision.PRODUCTION_CANDIDATE
    )
    assert record.metadata["manual_promotion_required"] is True
    assert record.metadata["runtime_effect"] == (
        "none_until_manual_production_factor_promotion"
    )
    assert record.metadata["source_notes"] == source_notes


def test_registry_write_fails_closed_for_malformed_registry(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    registry_path.write_text("{bad\n", encoding="utf-8")

    with pytest.raises(FactorRegistryMalformedError):
        apply_production_candidate_registry_updates(
            registry_path=registry_path,
            qualified_results=[_qualified_result()],
            run_timestamp="2026-06-07T04:30:00",
            run_id="daily_factor_mining_malformed_fixture",
            report_path="reports/factor_governance/daily_mining/fixture.json",
            owner="test owner",
            journal_path=tmp_path / "reports" / "registry_mutation.json",
            write=True,
        )

    assert registry_path.read_text(encoding="utf-8") == "{bad\n"


def test_registry_write_propagates_cas_conflict_without_mutation(
    monkeypatch,
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()

    def raise_conflict(*_args, **_kwargs):
        raise FactorRegistryConflictError("fixture mining CAS conflict")

    monkeypatch.setattr(mining, "apply_factor_record_patch", raise_conflict)
    with pytest.raises(FactorRegistryConflictError, match="mining CAS"):
        apply_production_candidate_registry_updates(
            registry_path=registry_path,
            qualified_results=[_qualified_result()],
            run_timestamp="2026-06-07T04:30:00",
            run_id="daily_factor_mining_conflict_fixture",
            report_path="reports/factor_governance/daily_mining/fixture.json",
            owner="test owner",
            journal_path=tmp_path / "reports" / "registry_mutation.json",
            write=True,
        )

    assert registry_path.read_bytes() == before


def test_registry_write_does_not_override_existing_production_factor(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    existing = FactorRecord(
        name="daily_fixture_factor",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:pv_momentum_20d",
        weight=0.05,
        gate_results=[
            GateResult.from_dict(row) for row in _gate_rows()
        ],
        admission_decision=FactorAdmissionDecision.PRODUCTION_CANDIDATE,
    )
    _write_registry(
        registry_path,
        MinedFactorRegistry.from_records([existing]),
    )

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[_qualified_result()],
        run_timestamp="2026-06-07T04:30:00",
        run_id="daily_factor_mining_fixture",
        report_path="reports/factor_governance/daily_mining/fixture.json",
        owner="test owner",
        horizon_days=30,
        max_candidates=5,
        journal_path=tmp_path / "reports" / "registry_mutation.json",
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    record = registry.factors[0]
    assert manifest["status"] == "no_registry_changes"
    assert manifest["skipped_factors"] == [
        {
            "name": "daily_fixture_factor",
            "reason": "existing_production_factor",
        }
    ]
    assert record.state == FactorLifecycleState.PRODUCTION_FACTOR
    assert record.weight == 0.05


@pytest.mark.parametrize("gate_case", ["missing", "failed"])
def test_registry_writer_rejects_incomplete_or_failed_gate_evidence(
    tmp_path,
    gate_case,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()
    result = _qualified_result()
    if gate_case == "missing":
        result["gate_results"] = result["gate_results"][:7]
        expected_reason = "gate_ids_not_exactly_1_to_8"
    else:
        result["gate_results"][3]["passed"] = False
        expected_reason = "gate_4_not_passed"

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[result],
        run_timestamp="2026-06-07T04:30:00",
        run_id=f"daily_factor_mining_{gate_case}_gate_fixture",
        report_path="reports/factor_governance/daily_mining/fixture.json",
        owner="test owner",
        journal_path=tmp_path / "reports" / "registry_mutation.json",
        write=True,
    )

    assert manifest["status"] == "no_registry_changes"
    assert manifest["skipped_factors"] == [
        {"name": "daily_fixture_factor", "reason": expected_reason}
    ]
    assert registry_path.read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing", "diversity_selection_missing"),
        ("hash", "diversity_policy_hash_mismatch"),
        ("runtime", "runtime_write_ineligible"),
    ],
)
def test_registry_writer_rejects_missing_or_forged_diversity_evidence(
    tmp_path, mutation, expected_reason
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()
    result = _qualified_result()
    if mutation == "missing":
        result.pop("diversity_selection")
    elif mutation == "hash":
        result["diversity_selection"]["policy_hash"] = "forged"
    else:
        result["runtime_write_eligible"] = False

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[result],
        run_timestamp="2026-07-12T10:00:00",
        run_id=f"diversity-{mutation}",
        report_path="reports/factor_governance/diversity.json",
        owner="test owner",
        journal_path=tmp_path / "registry_mutation.json",
        write=True,
    )

    assert manifest["status"] == "no_registry_changes"
    assert manifest["skipped_factors"][0]["reason"] == expected_reason
    assert registry_path.read_bytes() == before


def test_registry_writer_enumerates_post_diversity_max_candidate_skip(
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    first = _qualified_result("first")
    second = _qualified_result("second")
    second["family"] = "capacity"
    second["dominant_primitives"] = ["traded_amount"]
    second["primitive_lineage"] = ["traded_amount"]
    second["primitive_contributions"] = {"traded_amount": 1.0}
    second["diversity_selection"].update(
        family_champion="second",
        lineage_component_id="lineage-002",
        lineage_champion="second",
        correlation_cluster_id="correlation-002",
        cluster_champion="second",
    )

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[first, second],
        run_timestamp="2026-07-12T10:00:00",
        run_id="diversity-max-cap",
        report_path="reports/factor_governance/diversity.json",
        owner="test owner",
        max_candidates=1,
        journal_path=tmp_path / "reports" / "registry_mutation.json",
        write=True,
    )

    assert manifest["selected_champions"] == ["first"]
    assert manifest["skipped_factors"] == [
        {"name": "second", "reason": "max_registry_candidates"}
    ]


def test_registry_writer_rejects_entire_batch_for_duplicate_champions(
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()
    first = _qualified_result("first")
    second = _qualified_result("second")

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[first, second],
        run_timestamp="2026-07-12T10:00:00",
        run_id="duplicate-champions",
        report_path="reports/factor_governance/diversity.json",
        owner="test owner",
        journal_path=tmp_path / "reports" / "registry_mutation.json",
        write=True,
    )

    assert manifest["status"] == "no_registry_changes"
    assert manifest["fail_closed_reason"] == (
        "diversity_batch_validation_failed"
    )
    assert registry_path.read_bytes() == before


def test_production_family_governance_keeps_only_current_8gate_champion(
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    records = []
    for name in ("w70", "w75", "short"):
        records.append(
            FactorRecord(
                name=name,
                state=FactorLifecycleState.PRODUCTION_FACTOR,
                implementation=f"price_volume:{name}",
                weight=0.05,
                gate_results=[
                    GateResult.from_dict(row) for row in _gate_rows()
                ],
            )
        )
    _write_registry(registry_path, MinedFactorRegistry.from_records(records))
    champion = _qualified_result("w70")
    champion["family"] = "blend"
    redundant = _qualified_result("w75")
    redundant["family"] = "blend"
    redundant["decision"] = FactorAdmissionDecision.PAPER_FACTOR.value
    redundant["gate_results"][7]["passed"] = False
    unrelated = _qualified_result("short")
    unrelated["family"] = "short_reversal"
    unrelated["decision"] = FactorAdmissionDecision.WATCHLIST.value
    unrelated["gate_results"][2]["passed"] = False
    results = [champion, redundant, unrelated]
    report_path = _write_mining_report(
        tmp_path / "reports" / "mining.json",
        run_id="direct-family-governance",
        results=results,
    )

    manifest = apply_production_family_governance(
        registry_path=registry_path,
        results=results,
        run_timestamp="2026-07-12T12:00:00+08:00",
        run_id="direct-family-governance",
        report_path=str(report_path),
        journal_path=tmp_path / "reports" / "family_governance.json",
        universes=["full_a"],
        market_evidence=_market_evidence(),
        owner="test owner",
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    assert manifest["status"] == "updated"
    assert manifest["kept_factors"] == ["w70"]
    assert [item.name for item in registry.selectable_factors()] == ["w70"]
    by_name = {item.name: item for item in registry.factors}
    assert by_name["w75"].state == FactorLifecycleState.DEPRECATED
    assert by_name["w75"].weight == 0.0
    assert by_name["w75"].deprecated_reason == "current_8gate_not_passed"
    assert by_name["short"].state == FactorLifecycleState.DEPRECATED
    assert by_name["w70"].metadata["governance_family"] == "blend"


def test_production_family_governance_fails_closed_without_champion(tmp_path):
    registry_path = tmp_path / "mined_factors.json"
    record = FactorRecord(
        name="failing",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:failing",
        weight=0.05,
        gate_results=[GateResult.from_dict(row) for row in _gate_rows()],
    )
    _write_registry(
        registry_path,
        MinedFactorRegistry.from_records([record]),
    )
    before = registry_path.read_bytes()
    result = _qualified_result("failing")
    result["decision"] = FactorAdmissionDecision.WATCHLIST.value
    result["gate_results"][2]["passed"] = False
    report_path = _write_mining_report(
        tmp_path / "reports" / "no_champion.json",
        run_id="no-champion",
        results=[result],
    )

    manifest = apply_production_family_governance(
        registry_path=registry_path,
        results=[result],
        run_timestamp="2026-07-12T12:00:00+08:00",
        run_id="no-champion",
        report_path=str(report_path),
        journal_path=tmp_path / "reports" / "family_governance.json",
        universes=["full_a"],
        market_evidence=_market_evidence(1),
        owner="test owner",
        write=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["fail_closed_reason"] == (
        "no_current_8gate_family_champion"
    )
    assert registry_path.read_bytes() == before


def test_production_family_governance_promotes_new_champion_in_one_transaction(
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    result = _qualified_result("new_weekly_champion")
    report_path = _write_mining_report(
        tmp_path / "reports" / "promotion.json",
        run_id="weekly-promotion",
        results=[result],
    )
    journal_path = tmp_path / "reports" / "promotion_wal.json"

    manifest = apply_production_family_governance(
        registry_path=registry_path,
        results=[result],
        run_timestamp="2026-07-12T12:00:00+08:00",
        run_id="weekly-promotion",
        report_path=str(report_path),
        journal_path=journal_path,
        universes=["full_a"],
        market_evidence=_market_evidence(5502),
        owner="test owner",
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    assert manifest["status"] == "updated"
    assert manifest["promoted_factors"] == ["new_weekly_champion"]
    assert manifest["kept_factors"] == []
    assert manifest["active_after_count"] == 1
    assert registry.metadata["production_factor_count"] == 1
    record = registry.selectable_factors()[0]
    assert record.name == "new_weekly_champion"
    assert record.weight == 0.05
    assert record.metadata["manual_promotion_required"] is False
    assert journal_path.exists()


def test_production_family_governance_rejects_subset_evidence_byte_identical(
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()
    result = _qualified_result("subset_champion")
    report_path = _write_mining_report(
        tmp_path / "reports" / "subset.json",
        run_id="subset-run",
        results=[result],
        universes=["hs300", "zz500", "zz1000"],
    )

    manifest = apply_production_family_governance(
        registry_path=registry_path,
        results=[result],
        run_timestamp="2026-07-12T12:00:00+08:00",
        run_id="subset-run",
        report_path=str(report_path),
        journal_path=tmp_path / "reports" / "subset_wal.json",
        universes=["hs300", "zz500", "zz1000"],
        market_evidence=_market_evidence(1800),
        owner="test owner",
        write=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["fail_closed_reason"] == (
        "production_universe_not_exact_full_a"
    )
    assert registry_path.read_bytes() == before


def test_daily_wrapper_success_path_writes_durable_registry_journal(
    monkeypatch,
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    fixed_now = datetime(
        2026,
        7,
        11,
        9,
        30,
        tzinfo=daily.SHANGHAI_TZ,
    )
    monkeypatch.setattr(daily, "_now_shanghai", lambda: fixed_now)
    monkeypatch.setattr(
        daily,
        "latest_download_report",
        lambda _path: (None, {}),
    )

    def fake_run_mining(args):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = [_qualified_result()]
        _write_mining_report(
            output_dir / "quant_branch_factor_mining_results.json",
            run_id="wrapper-success-fixture",
            results=results,
        )
        return {
            "output_dir": str(output_dir),
            "results": results,
            "candidate_count": 1,
            "qualified_count": 1,
            "loaded_symbol_count": 5502,
            "factor_exposure_evidence": _market_evidence(5502)[
                "factor_exposure_evidence"
            ],
            "conclusion": "manual_production_factor_review_candidate",
        }

    monkeypatch.setattr(daily, "run_mining", fake_run_mining)
    monkeypatch.setattr(
        daily,
        "strict_full_a_market_evidence",
        lambda **_kwargs: _market_evidence(5502),
    )
    args = daily.parse_args(
        [
            "--output-root",
            str(tmp_path / "reports"),
            "--registry-path",
            str(registry_path),
            "--run-id",
            "wrapper-success-fixture",
            "--strict-positive-evidence",
        ]
    )

    payload = daily.run_daily_automation(args)

    manifest = payload["registry_update_manifest"]
    journal_path = Path(manifest["registry_mutation_manifest_path"])
    assert payload["success_gate_passed"] is True
    assert manifest["status"] == "updated"
    assert journal_path.parent == Path(payload["mining_output_dir"])
    assert journal_path.exists()
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["status"] == "applied"
    registry = MinedFactorRegistry.load(registry_path)
    assert registry.factors[0].weight == 0.05
    assert registry.factors[0].state == (
        FactorLifecycleState.PRODUCTION_FACTOR
    )


def test_daily_wrapper_keeps_incumbent_when_no_new_challenger_passes(
    monkeypatch,
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    incumbent = FactorRecord(
        name="incumbent",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation="price_volume:incumbent",
        weight=0.05,
        gate_results=[GateResult.from_dict(row) for row in _gate_rows()],
    )
    _write_registry(
        registry_path,
        MinedFactorRegistry.from_records([incumbent]),
    )
    before = registry_path.read_bytes()
    fixed_now = datetime(
        2026,
        7,
        18,
        4,
        30,
        tzinfo=daily.SHANGHAI_TZ,
    )
    monkeypatch.setattr(daily, "_now_shanghai", lambda: fixed_now)
    monkeypatch.setattr(
        daily,
        "latest_download_report",
        lambda _path: (None, {}),
    )

    def fake_run_mining(args):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = _qualified_result("challenger")
        result["decision"] = FactorAdmissionDecision.WATCHLIST.value
        result["gate_results"][2]["passed"] = False
        result["failed_gate_ids"] = [3]
        result["passed_gate_ids"] = [1, 2, 4, 5, 6, 7, 8]
        result["gates_passed"] = 7
        _write_mining_report(
            output_dir / "quant_branch_factor_mining_results.json",
            run_id="incumbent-carry-forward",
            results=[result],
        )
        return {
            "output_dir": str(output_dir),
            "results": [result],
            "candidate_count": 1,
            "qualified_count": 0,
            "loaded_symbol_count": 5502,
            "factor_exposure_evidence": _market_evidence(5502)[
                "factor_exposure_evidence"
            ],
            "conclusion": "no_candidate_passed_myquant_8gate",
        }

    monkeypatch.setattr(daily, "run_mining", fake_run_mining)
    monkeypatch.setattr(
        daily,
        "strict_full_a_market_evidence",
        lambda **_kwargs: _market_evidence(5502),
    )
    args = daily.parse_args(
        [
            "--output-root",
            str(tmp_path / "reports"),
            "--registry-path",
            str(registry_path),
            "--run-id",
            "incumbent-carry-forward",
            "--strict-positive-evidence",
        ]
    )

    payload = daily.run_daily_automation(args)

    manifest = payload["registry_update_manifest"]
    assert payload["success_gate_passed"] is True
    assert payload["candidate_promotion_gate_passed"] is False
    assert payload["incumbent_carry_forward"] is True
    assert manifest["status"] == "no_registry_changes"
    assert manifest["kept_factors"] == ["incumbent"]
    assert registry_path.read_bytes() == before

    blocked_evidence = _market_evidence(5502)
    blocked_evidence["pointer_status"] = "BROKEN"
    monkeypatch.setattr(
        daily,
        "strict_full_a_market_evidence",
        lambda **_kwargs: blocked_evidence,
    )
    blocked_args = daily.parse_args(
        [
            "--output-root",
            str(tmp_path / "blocked-reports"),
            "--registry-path",
            str(registry_path),
            "--run-id",
            "incumbent-carry-forward-blocked",
            "--strict-positive-evidence",
        ]
    )

    blocked_payload = daily.run_daily_automation(blocked_args)

    assert blocked_payload["success_gate_passed"] is False
    assert blocked_payload["incumbent_carry_forward"] is False
    assert blocked_payload["fail_closed_reason"] == (
        "parquet_latest_pointer_not_ok"
    )
    assert registry_path.read_bytes() == before
