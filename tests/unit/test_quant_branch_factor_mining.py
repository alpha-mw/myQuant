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
from quant_investor.factors.governance_protocol_v2 import (
    CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
)
from quant_investor.factors.registry_store import (
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
    ext_path = root / "dag_core_raw" / "table=daily_basic_ext" / "part.parquet"
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


def test_legacy_8gate_metrics_do_not_forge_data_oos_or_full_chain_evidence():
    dates = pd.date_range("2023-01-31", periods=36, freq="ME")
    symbols = [f"{index:06d}.SZ" for index in range(30)]
    signal = pd.DataFrame(
        [range(30) for _ in dates],
        index=dates,
        columns=symbols,
        dtype=float,
    )
    forward = signal.div(1000.0).add(0.001)
    context = RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in symbols},
        adj_close=signal.add(100.0),
        volume=signal.add(1000.0),
        amount=signal.add(1_000_000.0),
        forward_return=forward,
        rebalance_dates=list(dates),
        biweekly_dates=list(dates),
        existing_composite=signal.mul(0.25),
        existing_blocker="",
    )

    metrics = retest.candidate_metrics(
        signal=signal,
        context=context,
        decision_cost_bps=1.0,
        incremental_sleeve=0.03,
    )
    review = retest.evaluate_with_myquant_gate("fixture", metrics)

    assert metrics["no_future_leakage"] is False
    assert metrics["rankic_direction_stable"] is True
    assert metrics["walk_forward_purged"] is False
    assert metrics["full_control_chain_evaluated"] is False
    assert metrics["master_return_delta"] is None
    assert metrics["diagnostic_linear_overlay_return_delta"] is not None
    assert {1, 7, 8}.issubset(set(retest._failed_gate_ids(review)))


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
            "family_fdr_method": "benjamini_hochberg_by_family",
            "family_fdr_q_value": 0.08,
            "family_fdr_passed": True,
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


def _ready_exposure_evidence() -> dict[str, object]:
    return {
        "status": "ready",
        "source": "strict_parquet_hybrid_market_cap_exposure",
        "coverage_ratio": 1.0,
        "catalog_validated": True,
        "size_policy": (
            "same_trade_date_total_mv_then_asof_total_share_times_close"
        ),
        "evaluation_date_coverage_ratio": 1.0,
        "min_cross_section_coverage_ratio": 1.0,
        "combined_size_pair_coverage_ratio": 1.0,
        "pit_size_pair_coverage_ratio": 0.75,
        "reconstructed_size_pair_ratio": 0.25,
        "share_reference_covers_evaluation_end": True,
        "sector_count": 2,
        "size_bucket_count": 3,
    }


def _ready_full_a_market_evidence() -> dict[str, object]:
    return {
        "backend": "parquet",
        "mode_policy": "strict",
        "pointer_status": "OK",
        "snapshot_id": "fixture-snapshot",
        "coverage_complete": True,
        "expected_symbol_count": 3,
        "loaded_symbol_count": 3,
        "table_root_exists": True,
        "serving_root_exists": True,
        "manifest_exists": True,
    }


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


def test_mining_attaches_family_scoped_bh_fdr_evidence_before_qualification():
    first = _qualified_result("first")
    second = _qualified_result("second")
    first["metrics"]["rank_ic_p_value"] = 0.01
    second["metrics"]["rank_ic_p_value"] = 0.20

    mining._set_family_fdr([first, second])

    assert first["metrics"]["family_fdr_method"] == (
        "benjamini_hochberg_by_family"
    )
    assert first["metrics"]["family_fdr_passed"] is True
    assert second["metrics"]["family_fdr_passed"] is False
    assert first["metrics"]["family_test_count"] == 2


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


@pytest.mark.parametrize("write", [False, True])
def test_direct_candidate_registry_writer_is_retired_and_never_mutates(
    tmp_path,
    write,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()
    wal_path = tmp_path / "reports" / "registry_mutation.json"

    manifest = apply_production_candidate_registry_updates(
        registry_path=registry_path,
        qualified_results=[_qualified_result()],
        run_timestamp="2026-07-12T10:00:00",
        run_id="direct-writer-retired",
        report_path="reports/factor_governance/mining.json",
        owner="test owner",
        journal_path=wal_path,
        write=write,
    )

    assert manifest["status"] == ("blocked" if write else "report_only")
    assert manifest["before_registry_sha256"] == manifest[
        "after_registry_sha256"
    ]
    assert registry_path.read_bytes() == before
    assert not wal_path.exists()
    if write:
        assert manifest["fail_closed_reason"] == (
            "direct_candidate_registry_write_retired_use_"
            "factor_governance_protocol_v2"
        )
        assert manifest["skipped_factors"] == [
            {
                "name": "daily_fixture_factor",
                "reason": "direct_candidate_registry_write_retired",
            }
        ]


def test_retired_direct_candidate_write_cli_flag_exits_nonzero(monkeypatch):
    def fail_if_called(_args):
        pytest.fail("retired direct writer must be rejected before mining")

    monkeypatch.setattr(mining, "run_mining", fail_if_called)

    assert mining.main(["--write-production-candidates"]) == 2


def test_report_only_cli_exits_zero_without_claiming_registry_apply(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        mining,
        "run_mining",
        lambda _args: {
            "output_dir": "reports/factor_governance/fixture",
            "conclusion": "manual_production_factor_review_candidate",
            "candidate_count": 1,
            "qualified_count": 1,
            "qualified_factors": ["fixture"],
            "registry_update_manifest": {"status": "report_only"},
        },
    )

    assert mining.main([]) == 0
    output = capsys.readouterr().out
    assert '"registry_update_status": "report_only"' in output
    assert '"registry_update_status": "applied"' not in output


def test_retired_bulk_family_governance_cannot_deprecate_unrelated_factors(
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

    manifest = apply_production_family_governance(
        registry_path=registry_path,
        results=[champion, redundant, unrelated],
        run_timestamp="2026-07-12T12:00:00+08:00",
        run_id="direct-family-governance",
        report_path="reports/factor_governance/mining.json",
        journal_path=tmp_path / "reports" / "family_governance.json",
        write=True,
    )

    registry = MinedFactorRegistry.load(registry_path)
    assert manifest["status"] == "blocked"
    assert manifest["changed_record_names"] == []
    assert manifest["before_registry_sha256"] == manifest["after_registry_sha256"]
    assert [item.name for item in registry.selectable_factors()] == [
        "w70",
        "w75",
        "short",
    ]
    by_name = {item.name: item for item in registry.factors}
    assert by_name["w75"].state == FactorLifecycleState.PRODUCTION_FACTOR
    assert by_name["w75"].weight == 0.05
    assert by_name["short"].state == FactorLifecycleState.PRODUCTION_FACTOR


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

    manifest = apply_production_family_governance(
        registry_path=registry_path,
        results=[result],
        run_timestamp="2026-07-12T12:00:00+08:00",
        run_id="no-champion",
        report_path="reports/factor_governance/mining.json",
        journal_path=tmp_path / "reports" / "family_governance.json",
        write=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["fail_closed_reason"] == (
        "bulk_family_reconciliation_retired_use_factor_governance_protocol_v2"
    )
    assert registry_path.read_bytes() == before


def test_weekly_wrapper_defaults_to_report_only_and_keeps_registry_sha(
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
        return {
            "output_dir": str(output_dir),
            "results": [_qualified_result()],
            "candidate_count": 1,
            "qualified_count": 1,
            "conclusion": "manual_production_factor_review_candidate",
            "loaded_symbol_count": 3,
            "factor_exposure_evidence": _ready_exposure_evidence(),
        }

    monkeypatch.setattr(daily, "run_mining", fake_run_mining)
    monkeypatch.setattr(
        daily,
        "strict_full_a_market_evidence",
        lambda **_kwargs: _ready_full_a_market_evidence(),
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
    assert payload["success_gate_passed"] is True
    assert manifest["status"] == "report_only"
    assert manifest["before_registry_sha256"] == manifest["after_registry_sha256"]
    assert manifest["changed_record_names"] == []
    assert payload["registry_write_requested"] is False
    assert payload["run_mode"] == "report_only"
    assert payload["factor_protocol"]["apply_requested"] is False
    assert payload["factor_protocol"]["transition_applied"] is False
    report_only_mining = json.loads(
        (Path(payload["mining_output_dir"]) / "quant_branch_factor_mining_results.json")
        .read_text(encoding="utf-8")
    )
    assert report_only_mining["factor_protocol"] == payload["factor_protocol"]
    registry = MinedFactorRegistry.load(registry_path)
    assert registry.factors == []


def test_weekly_strict_mode_fails_when_market_evidence_is_blocked(
    monkeypatch,
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    before = registry_path.read_bytes()
    monkeypatch.setattr(
        daily,
        "latest_download_report",
        lambda _path: (None, {}),
    )

    def fake_run_mining(args):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "output_dir": str(output_dir),
            "results": [_qualified_result()],
            "candidate_count": 1,
            "qualified_count": 1,
            "conclusion": "manual_production_factor_review_candidate",
            "loaded_symbol_count": 3,
            "factor_exposure_evidence": _ready_exposure_evidence(),
        }

    blocked_market = _ready_full_a_market_evidence()
    blocked_market["pointer_status"] = "MISSING"
    monkeypatch.setattr(daily, "run_mining", fake_run_mining)
    monkeypatch.setattr(
        daily,
        "strict_full_a_market_evidence",
        lambda **_kwargs: dict(blocked_market),
    )
    argv = [
        "--output-root",
        str(tmp_path / "reports"),
        "--registry-path",
        str(registry_path),
        "--run-id",
        "wrapper-market-blocked",
        "--strict-positive-evidence",
    ]

    payload = daily.run_daily_automation(daily.parse_args(argv))

    assert payload["success_gate_passed"] is False
    assert payload["fail_closed_reason"] == "parquet_latest_pointer_not_ok"
    assert payload["registry_update_manifest"]["status"] == "report_only"
    assert payload["registry_update_manifest"]["before_registry_sha256"] == (
        payload["registry_update_manifest"]["after_registry_sha256"]
    )
    assert registry_path.read_bytes() == before
    assert daily.main(argv) == 2
    assert registry_path.read_bytes() == before


def test_weekly_wrapper_apply_requires_protocol_hash_and_canonical_evidence(tmp_path):
    expected_hash = daily.protocol_hash()
    with pytest.raises(SystemExit):
        daily.parse_args(["--apply-governed-transitions"])
    with pytest.raises(SystemExit):
        daily.parse_args(
            [
                "--apply-governed-transitions",
                "--protocol-version",
                "v2",
                "--expected-protocol-hash",
                "wrong",
                "--governed-evidence-json",
                str(tmp_path / "plan.json"),
            ]
        )
    args = daily.parse_args(
        [
            "--apply-governed-transitions",
            "--protocol-version",
            "v2",
            "--expected-protocol-hash",
            expected_hash,
            "--governed-evidence-json",
            str(tmp_path / "plan.json"),
        ]
    )
    assert args.apply_governed_transitions is True


def test_weekly_wrapper_routes_normalized_evidence_to_blocked_apply_path(
    monkeypatch,
    tmp_path,
):
    registry_path = tmp_path / "mined_factors.json"
    _write_registry(registry_path, MinedFactorRegistry())
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}\n", encoding="utf-8")
    fixed_now = datetime(
        2026,
        7,
        31,
        17,
        0,
        tzinfo=daily.SHANGHAI_TZ,
    )
    monkeypatch.setattr(daily, "_now_shanghai", lambda: fixed_now)
    monkeypatch.setattr(daily, "latest_download_report", lambda _path: (None, {}))

    def fake_run_mining(args):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "output_dir": str(output_dir),
            "results": [],
            "candidate_count": 0,
            "qualified_count": 0,
            "conclusion": "no_candidate_passed_myquant_8gate",
            "loaded_symbol_count": 3,
            "factor_exposure_evidence": _ready_exposure_evidence(),
        }

    sentinel_plan = object()
    calls = {}

    def fake_apply(path, plan, *, expected_protocol_hash, valid_trading_days, write):
        calls.update(
            path=path,
            plan=plan,
            expected_protocol_hash=expected_protocol_hash,
            valid_trading_days=valid_trading_days,
            write=write,
        )
        return {
            "status": "blocked",
            "blockers": [CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER],
            "before_registry_sha256": "before",
            "after_registry_sha256": "before",
            "inverse_wal_path": "",
            "changed_record_names": [],
            "canonical_replay_producer_control": {
                "production_apply_eligible": False,
                "blocker": CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER,
            },
        }

    monkeypatch.setattr(daily, "run_mining", fake_run_mining)
    monkeypatch.setattr(
        daily,
        "strict_full_a_market_evidence",
        lambda **_kwargs: _ready_full_a_market_evidence(),
    )
    sentinel_evidence = object()
    monkeypatch.setattr(
        daily,
        "load_governance_replay_evidence",
        lambda _path: sentinel_evidence,
    )
    monkeypatch.setattr(
        daily,
        "build_registry_mutation_plan_from_evidence",
        lambda **kwargs: (
            sentinel_plan,
            ["2026-07-31"],
        )
        if kwargs["evidence"] is sentinel_evidence
        else pytest.fail("canonical evidence was not routed to plan builder"),
    )
    monkeypatch.setattr(daily, "apply_governed_transition", fake_apply)
    args = daily.parse_args(
        [
            "--output-root",
            str(tmp_path / "reports"),
            "--registry-path",
            str(registry_path),
            "--run-id",
            "wrapper-apply-fixture",
            "--apply-governed-transitions",
            "--protocol-version",
            "v2",
            "--expected-protocol-hash",
            daily.protocol_hash(),
            "--governed-evidence-json",
            str(evidence_path),
        ]
    )

    payload = daily.run_daily_automation(args)

    assert calls == {
        "path": str(registry_path),
        "plan": sentinel_plan,
        "expected_protocol_hash": daily.protocol_hash(),
        "valid_trading_days": ["2026-07-31"],
        "write": True,
    }
    assert payload["factor_protocol"]["status"] == "blocked"
    assert payload["factor_protocol"]["transition_applied"] is False
    assert payload["factor_protocol"]["inverse_wal_path"] == ""
    assert CANONICAL_FULL_CHAIN_PRODUCER_BLOCKER in payload[
        "factor_protocol"
    ]["blockers"]
    mining_summary = json.loads(
        (Path(payload["mining_output_dir"]) / "quant_branch_factor_mining_results.json")
        .read_text(encoding="utf-8")
    )
    assert mining_summary["registry_write"] is False
    assert mining_summary["factor_protocol"] == payload["factor_protocol"]


def test_weekly_wrapper_apply_requested_but_protocol_blocked_exits_nonzero(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        daily,
        "run_daily_automation",
        lambda _args: {
            "summary_report_path": str(tmp_path / "summary.md"),
            "success_gate_passed": True,
            "run_mode": "governed_apply",
            "fail_closed_reason": "",
            "candidate_count": 1,
            "evidence_counts": {
                "positive_evidence_count": 1,
                "positive_candidate_count": 1,
                "diverse_positive_champion_count": 1,
                "qualified_count": 1,
            },
            "registry_update_manifest": {"status": "blocked"},
            "production_family_governance_manifest": {
                "status": "blocked"
            },
            "factor_protocol": {
                "status": "blocked",
                "transition_applied": False,
                "blockers": ["monthly_transition_budget_exhausted"],
            },
        },
    )

    exit_code = daily.main(
        [
            "--apply-governed-transitions",
            "--protocol-version",
            "v2",
            "--expected-protocol-hash",
            daily.protocol_hash(),
            "--governed-evidence-json",
            str(tmp_path / "canonical-evidence.json"),
        ]
    )

    assert exit_code == 2
