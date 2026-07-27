from __future__ import annotations

import ast
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from quant_investor.v17_v3_runtime.algorithms import (
    CANDIDATE_QUANT_WEIGHTS,
    BranchFusionError,
    BranchOutput,
    BranchRecord,
    CalibrationError,
    CalibrationMonth,
    FactorIdentity,
    FactorInventoryConflict,
    FactorSpec,
    MonthlyFusionMetric,
    SymbolObservation,
    bootstrap_matrix_header_bytes,
    bootstrap_matrix_sha256,
    calibrate_fusion,
    canonical_decimal_string,
    circular_moving_block_bootstrap_matrix,
    evaluate_deep_research,
    fuse_branches,
    normalize_decimal,
    run_quant_preselection,
    schedule_month_end_origins,
    select_fusion_weight,
    validate_branch_output,
    validate_disjoint_factor_inventories,
    validate_monotonic_overlay,
)

ALGORITHM_ROOT = (
    Path(__file__).resolve().parents[2] / "quant_investor" / "v17_v3_runtime" / "algorithms"
)


def test_v3_algorithm_package_has_no_v17_v2_import() -> None:
    for path in ALGORITHM_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "v17_v2" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all("v17_v2" not in alias.name for alias in node.names)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.0000000000005", Decimal("1.000000000000")),
        ("1.0000000000015", Decimal("1.000000000002")),
        ("-0.0000000000004", Decimal("0.000000000000")),
    ],
)
def test_decimal_normalization_is_half_even_at_1e_12(
    value: str,
    expected: Decimal,
) -> None:
    assert normalize_decimal(value) == expected


def test_decimal_wire_is_fixed_point_canonical_and_rejects_invalid_values() -> None:
    assert canonical_decimal_string("1000.230000000000") == "1000.23"
    assert canonical_decimal_string("-0") == "0"
    assert canonical_decimal_string("1e100") == "1" + "0" * 100
    for invalid in (True, "NaN", "Infinity", float("-inf")):
        with pytest.raises(ValueError):
            canonical_decimal_string(invalid)  # type: ignore[arg-type]


def _factor(
    *,
    weight: Decimal = Decimal("1"),
    lookback: int = 150,
    minimum_coverage: Decimal = Decimal("0.60"),
) -> FactorSpec:
    return FactorSpec(
        name="factor-a",
        definition_hash="sha256:preselector-factor-a",
        family="quality",
        lineage="factor-a.v1",
        weight=weight,
        lookback=lookback,
        minimum_coverage=minimum_coverage,
    )


def _inventory(
    *,
    definition_hash: str = "sha256:downstream-quant-factor",
    family: str = "timing",
    lineage: str = "quant-factor.v1",
) -> tuple[FactorIdentity, ...]:
    return (
        FactorIdentity(
            name="quant-factor",
            definition_hash=definition_hash,
            family=family,
            lineage=lineage,
        ),
    )


def _observation(
    symbol: str,
    value: object,
    *,
    history: int = 150,
    liquid: bool = True,
) -> SymbolObservation:
    return SymbolObservation(
        symbol=symbol,
        factor_values={"factor-a": value},
        history_count=history,
        liquid=liquid,
    )


def test_quant_preselector_ties_missing_and_input_order_replay() -> None:
    observations = (
        _observation("000002.SZ", "4.0000000000004"),
        _observation("000001.SZ", "4"),
        _observation("000003.SZ", "1"),
        _observation("000004.SZ", None),
        _observation("000005.SZ", "9", liquid=False),
    )
    result = run_quant_preselection(
        observations,
        factor_contract=(_factor(),),
        branch_inventory=_inventory(),
        top_n=1,
    )
    assert result.status == "READY"
    assert result.history_required == 150
    assert result.selected_symbols == ("000001.SZ",)
    assert dict(result.factor_coverage) == {"factor-a": Decimal("0.750000000000")}
    missing = next(row for row in result.dispositions if row.symbol == "000004.SZ")
    assert missing.status == "UNAVAILABLE"
    assert missing.reasons == ("factor_missing_or_nonfinite:factor-a",)

    replay = run_quant_preselection(
        tuple(reversed(observations)),
        factor_contract=(_factor(),),
        branch_inventory=_inventory(),
        top_n=1,
    )
    assert replay.selected_symbols == result.selected_symbols
    assert dict(replay.scores) == dict(result.scores)


def test_quant_preselector_coverage_and_inventory_conflict_fail_closed() -> None:
    observations = tuple(
        _observation(f"{index:06d}.SZ", value)
        for index, value in enumerate(("1", "2", None, None), start=1)
    )
    blocked = run_quant_preselection(
        observations,
        factor_contract=(_factor(),),
        branch_inventory=_inventory(),
    )
    assert blocked.status == "UNAVAILABLE"
    assert blocked.selected_symbols == ()
    assert any(
        blocker.startswith("factor_coverage_below_threshold") for blocker in blocked.blockers
    )


@pytest.mark.parametrize(
    ("inventory", "dimension"),
    [
        (_inventory(definition_hash="sha256:preselector-factor-a"), "definition_hash"),
        (_inventory(family="quality"), "family"),
        (_inventory(lineage="factor-a.v1"), "lineage"),
    ],
)
def test_preselector_and_quant_inventory_conflicts_hard_stop(
    inventory: tuple[FactorIdentity, ...],
    dimension: str,
) -> None:
    with pytest.raises(FactorInventoryConflict, match=dimension):
        validate_disjoint_factor_inventories((_factor(),), inventory)


def test_preselector_and_quant_inventory_must_both_be_nonempty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_disjoint_factor_inventories((_factor(),), ())


def _branch(
    name: str,
    pool: tuple[str, ...],
    scores: Mapping[str, object],
    *,
    unavailable: frozenset[str] = frozenset(),
    binding: str = "snapshot-1",
) -> BranchOutput:
    return BranchOutput(
        branch=name,
        ordered_domain=pool,
        bindings={"snapshot": binding},
        records=tuple(
            BranchRecord(
                symbol=symbol,
                status="UNAVAILABLE" if symbol in unavailable else "READY",
                score=None if symbol in unavailable else scores[symbol],
                reason="source_missing" if symbol in unavailable else None,
            )
            for symbol in pool
        ),
    )


def test_branch_validation_common_domain_ties_and_top24_no_backfill() -> None:
    pool = tuple(f"{index:06d}.SZ" for index in range(1, 26))
    scores = {symbol: "1" for symbol in pool}
    quant = _branch("quant", pool, scores)
    fundamental = _branch(
        "fundamental",
        pool,
        scores,
        unavailable=frozenset({pool[-1]}),
    )
    assert (
        validate_branch_output(
            quant,
            ordered_pool=pool,
            expected_bindings={"snapshot": "snapshot-1"},
        ).ordered_domain
        == pool
    )
    result = fuse_branches(
        quant,
        fundamental,
        ordered_pool=pool,
        quant_weight="0.50",
    )
    assert result.status == "READY"
    assert result.selected_symbols == pool[:-1]
    assert len(result.dispositions) == len(pool)
    assert result.dispositions[0].fusion_score == Decimal("0.500000000000")
    assert result.dispositions[-1].status == "UNAVAILABLE"

    insufficient = fuse_branches(
        quant,
        _branch(
            "fundamental",
            pool,
            scores,
            unavailable=frozenset({pool[-1], pool[-2]}),
        ),
        ordered_pool=pool,
        quant_weight="0.50",
    )
    assert insufficient.status == "UNAVAILABLE"
    assert len(insufficient.selected_symbols) == 23
    assert insufficient.blockers == ("common_ready_below_top_n:23:24",)


def test_branch_decimal_rank_tie_is_decided_after_1e_12_normalization() -> None:
    pool = ("000002.SZ", "000001.SZ")
    quant = _branch(
        "quant",
        pool,
        {pool[0]: "1.0000000000004", pool[1]: "1"},
    )
    fundamental = _branch("fundamental", pool, {symbol: "1" for symbol in pool})
    result = fuse_branches(
        quant,
        fundamental,
        ordered_pool=pool,
        quant_weight="0.25",
        top_n=1,
    )
    assert result.selected_symbols == ("000001.SZ",)
    assert {item.fusion_score for item in result.dispositions} == {Decimal("0.500000000000")}
    with pytest.raises(BranchFusionError, match="bindings"):
        validate_branch_output(
            quant,
            ordered_pool=pool,
            expected_bindings={"snapshot": "different"},
        )


def test_single_name_percentile_is_one() -> None:
    pool = ("000001.SZ",)
    quant = _branch("quant", pool, {pool[0]: "10"})
    fundamental = _branch("fundamental", pool, {pool[0]: "-3"})
    result = fuse_branches(
        quant,
        fundamental,
        ordered_pool=pool,
        quant_weight="0.50",
        top_n=1,
    )
    assert result.dispositions[0].quant_percentile == Decimal("1.000000000000")
    assert result.dispositions[0].fusion_score == Decimal("1.000000000000")


def test_shanghai_month_end_schedule_has_no_off_by_one_or_skipped_month() -> None:
    scheduled = schedule_month_end_origins(("2026-01-29", "2026-01-30", "2026-02-02", "2026-02-27"))
    assert tuple(item.session.isoformat() for item in scheduled) == (
        "2026-01-30",
        "2026-02-27",
    )
    assert scheduled[0].origin_at.isoformat() == "2026-01-30T15:00:00+08:00"
    with pytest.raises(CalibrationError, match="scheduled month skipped"):
        schedule_month_end_origins(("2026-01-30", "2026-03-31"))


def test_bootstrap_matrix_identity_is_frozen() -> None:
    matrix = circular_moving_block_bootstrap_matrix()
    assert matrix.shape == (10_000, 60)
    assert matrix.dtype == np.dtype("<i8")
    assert matrix.flags.c_contiguous
    assert matrix[0].tolist() == [
        *range(46, 58),
        *range(47, 59),
        *range(15, 27),
        *range(20, 32),
        56,
        57,
        58,
        59,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert bootstrap_matrix_header_bytes() == (
        b'{"dtype":"<i8","order":"C","shape":[10000,60],'
        b'"version":"myquant.v17.v3.bootstrap-index-matrix.v1"}'
    )
    assert bootstrap_matrix_sha256(matrix) == (
        "7a87680d07f0803ecb60c9cdb634dcbc06306a974890df7bd90820f548303b4c"
    )


def _direct_metrics(*, constant: bool) -> tuple[dict[Decimal, MonthlyFusionMetric], ...]:
    rows: list[dict[Decimal, MonthlyFusionMetric]] = []
    origin = date(2020, 1, 31)
    for month_index in range(60):
        monthly: dict[Decimal, MonthlyFusionMetric] = {}
        for weight_index, weight in enumerate(CANDIDATE_QUANT_WEIGHTS):
            hit = (
                Decimal("0.60")
                if constant
                else Decimal("0.55") + Decimal((month_index + weight_index) % 4) / Decimal("100")
            )
            q25 = (
                Decimal("0.02")
                if constant
                else Decimal("0.01")
                + Decimal((month_index * 3 + weight_index) % 7) / Decimal("1000")
            )
            monthly[weight] = MonthlyFusionMetric(origin, weight, hit, q25)
        rows.append(monthly)
    return tuple(rows)


def test_calibration_zero_variance_all_invalid_is_hard_stop() -> None:
    with pytest.raises(CalibrationError, match="all_fusion_weights_invalid"):
        select_fusion_weight(_direct_metrics(constant=True))


def _canonical_weekday_sessions(start: date, count: int) -> tuple[date, ...]:
    sessions: list[date] = []
    cursor = start
    while len(sessions) < count:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return tuple(sessions)


def _full_calibration_fixture() -> tuple[
    tuple[CalibrationMonth, ...],
    tuple[date, ...],
    date,
]:
    sessions = _canonical_weekday_sessions(date(2010, 1, 1), 3_800)
    origins = schedule_month_end_origins(sessions)
    selected_origins = origins[:138]
    session_index = {session: index for index, session in enumerate(sessions)}
    pool = tuple(f"{index:06d}.SZ" for index in range(1, 31))
    months: list[CalibrationMonth] = []
    for month_index, origin in enumerate(selected_origins):
        quant_scores = {symbol: Decimal(symbol[:6]) for symbol in pool}
        fundamental_scores = {
            symbol: Decimal(((int(symbol[:6]) * 7 + month_index) % 31) + 1) for symbol in pool
        }
        forward60 = {
            symbol: (
                Decimal("-0.01") if (int(symbol[:6]) + month_index) % 6 == 0 else Decimal("0.02")
            )
            for symbol in pool
        }
        forward252 = {
            symbol: Decimal(((int(symbol[:6]) * 3 + month_index) % 19) - 4) / Decimal("1000")
            for symbol in pool
        }
        label_end = sessions[session_index[origin.session] + 252]
        months.append(
            CalibrationMonth(
                origin=origin.session,
                label_252_end_session=label_end,
                ordered_pool=pool,
                quant_branch=_branch(
                    "quant",
                    pool,
                    quant_scores,
                    binding=f"q-{month_index}",
                ),
                fundamental_branch=_branch(
                    "fundamental",
                    pool,
                    fundamental_scores,
                    binding=f"q-{month_index}",
                ),
                forward_return_60=forward60,
                forward_return_252=forward252,
                label_252_mature=True,
            )
        )
    cutoff = sessions[session_index[selected_origins[-1].session] + 253]
    return tuple(months), sessions, cutoff


def test_calibration_outer_windows_are_leakage_free_and_active_refits() -> None:
    months, sessions, cutoff = _full_calibration_fixture()
    result = calibrate_fusion(
        months,
        canonical_sessions=sessions,
        active_cutoff=cutoff,
    )
    assert len(result.folds) == 5
    assert all(len(fold.training_origins) == 60 for fold in result.folds)
    assert all(len(fold.oos_origins) == 12 for fold in result.folds)
    label_end = {
        date.fromisoformat(str(month.origin)): date.fromisoformat(str(month.label_252_end_session))
        for month in months
    }
    for fold in result.folds:
        assert all(label_end[origin] < fold.oos_origins[0] for origin in fold.training_origins)
    assert result.active_weight in CANDIDATE_QUANT_WEIGHTS
    assert result.bootstrap_matrix_sha256 == bootstrap_matrix_sha256()


def test_calibration_rejects_neighbor_month_label_leakage_without_day_approximation() -> None:
    months, sessions, cutoff = _full_calibration_fixture()
    broken = list(months)
    target = broken[60]
    origin_index = sessions.index(date.fromisoformat(str(target.origin)))
    broken[60] = CalibrationMonth(
        origin=target.origin,
        label_252_end_session=sessions[origin_index + 251],
        ordered_pool=target.ordered_pool,
        quant_branch=target.quant_branch,
        fundamental_branch=target.fundamental_branch,
        forward_return_60=target.forward_return_60,
        forward_return_252=target.forward_return_252,
        label_252_mature=True,
    )
    with pytest.raises(CalibrationError, match="offset_invalid"):
        calibrate_fusion(
            broken,
            canonical_sessions=sessions,
            active_cutoff=cutoff,
        )


def test_deep_truth_table_holding_floor_lock_veto_and_no_positive_adjustment() -> None:
    locked = evaluate_deep_research(
        held=True,
        current_target="0.19",
        base_target="0.20",
        available=False,
    )
    assert locked.status == "LOCK"
    assert locked.target == Decimal("0.190000000000")
    assert locked.locked

    vetoed = evaluate_deep_research(
        held=False,
        current_target="0",
        base_target="0.20",
        available=True,
        signal="-1",
        veto_buy=True,
    )
    assert vetoed.status == "BUY_VETO"
    assert vetoed.target == 0

    floored = evaluate_deep_research(
        held=True,
        current_target="0.19",
        base_target="0.20",
        available=True,
        signal="-2",
    )
    assert floored.penalty == Decimal("0.100000000000")
    assert floored.raw_adjusted_target == Decimal("0.180000000000")
    assert floored.target == Decimal("0.190000000000")

    red_flagged_holding = evaluate_deep_research(
        held=True,
        current_target="0.19",
        base_target="0.25",
        available=True,
        signal="0",
        veto_buy=True,
    )
    assert red_flagged_holding.status == "LOCK"
    assert red_flagged_holding.locked
    assert red_flagged_holding.target == Decimal("0.190000000000")

    positive = evaluate_deep_research(
        held=False,
        current_target="0",
        base_target="0.20",
        available=True,
        signal="1",
    )
    assert positive.penalty == 0
    assert positive.target == Decimal("0.200000000000")


def test_overlay_is_subset_shrink_only_without_renormalization() -> None:
    valid = validate_monotonic_overlay(
        {"000001.SZ": "0.40", "000002.SZ": "0.20"},
        {"000001.SZ": "0.30"},
    )
    assert valid.valid
    assert valid.post_targets == {"000001.SZ": Decimal("0.300000000000")}
    assert valid.baseline_gross == Decimal("0.600000000000")
    assert valid.post_gross == Decimal("0.300000000000")
    assert valid.cash_delta == Decimal("0.300000000000")

    invalid = validate_monotonic_overlay(
        {"000001.SZ": "0.40"},
        {"000001.SZ": "0.41", "000002.SZ": "0.01"},
    )
    assert not invalid.valid
    assert "post_target_exceeds_baseline:000001.SZ" in invalid.blockers
    assert "post_symbol_not_in_baseline:000002.SZ" in invalid.blockers
    assert "post_gross_exceeds_baseline" in invalid.blockers
