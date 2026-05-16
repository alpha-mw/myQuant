from __future__ import annotations

import copy
import json

import pytest

from quant_investor.factors.library import FactorLibraryAuditReport, FactorLibraryPolicy
from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_FAMILY_MOMENTUM,
    FACTOR_STATUS_PRODUCTION,
    FactorDefinition,
    FactorLibraryEntry,
    ProductionFactorLibrary,
)
from quant_investor.factors.shadow_scoring import (
    SHADOW_COMPARISON_STATUS_OK,
    SHADOW_COMPARISON_STATUS_WARN,
    SHADOW_SCORE_STATUS_AUDIT_BLOCKED,
    SHADOW_SCORE_STATUS_INSUFFICIENT_DATA,
    SHADOW_SCORE_STATUS_LIBRARY_MISSING,
    SHADOW_SCORE_STATUS_MISSING_DATE,
    SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX,
    SHADOW_SCORE_STATUS_MISSING_SYMBOL,
    SHADOW_SCORE_STATUS_OK,
    SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
    ShadowCandidateScore,
    ShadowScoringComparisonReport,
    ShadowScoringConfig,
    build_factor_matrix_lookup,
    build_shadow_candidate_scores,
    build_shadow_scoring_comparison_report,
    build_shadow_scoring_dashboard_payload,
    extract_factor_value_for_symbol,
    rank_normalize_factor_values,
    render_shadow_scoring_comparison_markdown,
    resolve_factor_expected_direction,
    select_usable_production_factors,
)


def _entry(factor_id: str, factor_version: str = "v1") -> FactorLibraryEntry:
    return FactorLibraryEntry(
        factor_id=factor_id,
        factor_version=factor_version,
        status=FACTOR_STATUS_PRODUCTION,
        admission_decision_id=f"decision-{factor_id}",
        validation_report_id=f"validation-{factor_id}",
        production_since="2026-04-27",
    )


def _library(factor_ids: list[str]) -> ProductionFactorLibrary:
    entries = [_entry(factor_id) for factor_id in factor_ids]
    return ProductionFactorLibrary(
        library_id=f"library-{len(entries)}",
        generated_at="2026-04-27T00:00:00",
        entries=entries,
    )


def _definition(
    factor_id: str,
    *,
    expected_direction: float = 1.0,
) -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id,
        factor_name=f"Factor {factor_id}",
        factor_family=FACTOR_FAMILY_MOMENTUM,
        status=FACTOR_STATUS_PRODUCTION,
        version="v1",
        expression="close / delay(close, 20) - 1",
        input_fields=["close", "trade_date"],
        data_sources=["fixture"],
        universe="CN",
        benchmark="CSI300",
        expected_direction=expected_direction,
        rebalance_frequency="weekly",
        lookback_window=20,
        delay_days=1,
        execution_price="next_open",
        economic_rationale="Fixture rationale.",
        owner="research",
        created_at="2026-04-27",
    )


def _matrix(
    factor_id: str,
    values: list[list[float | None]],
    *,
    dates: list[str] | None = None,
    matrix_id: str | None = None,
    symbols: list[str] | None = None,
    expected_direction: float | None = None,
) -> FactorMatrix:
    resolved_symbols = symbols or ["AAA", "BBB", "CCC"]
    resolved_dates = dates or ["2026-04-26", "2026-04-27"]
    total = len(resolved_symbols) * len(resolved_dates)
    missing = sum(1 for row in values for value in row if value is None)
    metadata = {}
    if expected_direction is not None:
        metadata["expected_direction"] = expected_direction
    return FactorMatrix(
        matrix_id=matrix_id or f"matrix-{factor_id}",
        factor_id=factor_id,
        factor_version="v1",
        expression=f"{factor_id}_expr",
        symbols=resolved_symbols,
        dates=resolved_dates,
        values=values,
        coverage_ratio=(total - missing) / total,
        missing_ratio=missing / total,
        metadata=metadata,
    )


def _audit(blocked_factor_ids: list[str]) -> FactorLibraryAuditReport:
    return FactorLibraryAuditReport(
        report_id="audit-shadow-scoring",
        generated_at="2026-04-27T00:00:00",
        policy=FactorLibraryPolicy(require_incremental_review=False),
        library_id="library",
        production_factor_count=2,
        blocked_factor_ids=blocked_factor_ids,
        verdict="pass",
    )


def _candidates_without_ranks() -> list[dict[str, object]]:
    return [
        {"symbol": "AAA", "name": "Alpha", "official_score": 0.90},
        {"symbol": "BBB", "name": "Beta", "official_score": 0.80},
        {"symbol": "CCC", "name": "Gamma", "official_score": 0.70},
    ]


def _complete_report() -> ShadowScoringComparisonReport:
    config = ShadowScoringConfig(
        config_id="config-shadow-test",
        as_of="2026-04-27",
        top_n=2,
        max_rank_delta_warning=10,
        min_factor_coverage_ratio=0.50,
    )
    return build_shadow_scoring_comparison_report(
        candidates=_candidates_without_ranks(),
        library=_library(["momentum", "reversal"]),
        factor_matrices=[
            _matrix("momentum", [[0.10, 0.20], [0.10, 0.10], [0.10, 0.30]]),
            _matrix("reversal", [[4.0, 5.0], [4.0, 1.0], [4.0, 3.0]]),
        ],
        definitions=[
            _definition("momentum", expected_direction=1.0),
            _definition("reversal", expected_direction=-1.0),
        ],
        config=config,
        generated_at="2026-04-27T12:00:00",
        metadata={"fixture": True},
    )


def test_extract_factor_value_for_symbol_uses_latest_date_before_as_of() -> None:
    matrix = _matrix(
        "momentum",
        [[1.0, 2.0], [None, 4.0]],
        dates=["2026-04-25", "2026-04-27"],
        symbols=["AAA", "BBB"],
    )

    assert extract_factor_value_for_symbol(matrix, symbol="AAA", as_of="2026-04-26") == (
        1.0,
        SHADOW_SCORE_STATUS_OK,
    )
    assert extract_factor_value_for_symbol(matrix, symbol="ZZZ", as_of="2026-04-26") == (
        None,
        SHADOW_SCORE_STATUS_MISSING_SYMBOL,
    )
    assert extract_factor_value_for_symbol(matrix, symbol="AAA", as_of="2026-04-24") == (
        None,
        SHADOW_SCORE_STATUS_MISSING_DATE,
    )
    assert extract_factor_value_for_symbol(matrix, symbol="BBB", as_of="2026-04-26") == (
        None,
        SHADOW_SCORE_STATUS_INSUFFICIENT_DATA,
    )


def test_rank_normalize_factor_values_handles_direction_and_ties() -> None:
    positive = rank_normalize_factor_values(
        {"BBB": 2.0, "AAA": 2.0, "CCC": 1.0, "DDD": None},
        expected_direction=1.0,
    )
    negative = rank_normalize_factor_values(
        {"BBB": 2.0, "AAA": 2.0, "CCC": 1.0, "DDD": None},
        expected_direction=-1.0,
    )

    assert positive["AAA"] == (1.0, 1)
    assert positive["BBB"] == (0.5, 2)
    assert positive["CCC"] == (0.0, 3)
    assert positive["DDD"] == (None, None)
    assert negative["CCC"] == (1.0, 1)
    assert negative["AAA"] == (0.5, 2)
    assert rank_normalize_factor_values({"AAA": 5.0})["AAA"] == (1.0, 1)


def test_build_factor_matrix_lookup_prefers_latest_then_deterministic_id() -> None:
    older = _matrix(
        "momentum",
        [[1.0], [2.0]],
        dates=["2026-04-25"],
        symbols=["AAA", "BBB"],
        matrix_id="matrix-z",
    )
    latest_z = _matrix(
        "momentum",
        [[1.0], [2.0]],
        dates=["2026-04-27"],
        symbols=["AAA", "BBB"],
        matrix_id="matrix-z",
    )
    latest_a = _matrix(
        "momentum",
        [[1.0], [2.0]],
        dates=["2026-04-27"],
        symbols=["AAA", "BBB"],
        matrix_id="matrix-a",
    )

    lookup = build_factor_matrix_lookup([older, latest_z, latest_a])

    assert lookup[("momentum", "v1")].matrix_id == "matrix-a"


def test_select_usable_production_factors_excludes_audit_blocked_factor() -> None:
    library = _library(["allowed", "blocked"])
    audit = _audit(["blocked"])

    usable = select_usable_production_factors(
        library=library,
        audit_report=audit,
        include_blocked_factors=False,
    )
    all_factors = select_usable_production_factors(
        library=library,
        audit_report=audit,
        include_blocked_factors=True,
    )

    assert [entry.factor_id for entry in usable] == ["allowed"]
    assert [entry.factor_id for entry in all_factors] == ["allowed", "blocked"]


def test_resolve_factor_expected_direction_prefers_definition_then_matrix_metadata() -> None:
    matrix = _matrix(
        "momentum",
        [[1.0], [2.0]],
        dates=["2026-04-27"],
        symbols=["AAA", "BBB"],
        expected_direction=-1.0,
    )

    assert resolve_factor_expected_direction(
        factor_id="momentum",
        factor_version="v1",
        definitions=[_definition("momentum", expected_direction=1.0)],
        matrix=matrix,
    ) == 1.0
    assert resolve_factor_expected_direction(
        factor_id="momentum",
        factor_version="v1",
        definitions=[],
        matrix=matrix,
    ) == -1.0


def test_build_shadow_candidate_scores_ranks_and_preserves_inputs() -> None:
    candidates = _candidates_without_ranks()
    original = copy.deepcopy(candidates)
    config = ShadowScoringConfig(config_id="config", as_of="2026-04-27")

    scores = build_shadow_candidate_scores(
        candidates=candidates,
        library=_library(["momentum", "reversal"]),
        factor_matrices=[
            _matrix("momentum", [[0.10, 0.20], [0.10, 0.10], [0.10, 0.30]]),
            _matrix("reversal", [[4.0, 5.0], [4.0, 1.0], [4.0, 3.0]]),
        ],
        definitions=[
            _definition("momentum", expected_direction=1.0),
            _definition("reversal", expected_direction=-1.0),
        ],
        config=config,
    )

    by_symbol = {score.symbol: score for score in scores}
    assert candidates == original
    assert [score.symbol for score in scores] == ["AAA", "BBB", "CCC"]
    assert by_symbol["AAA"].official_rank == 1
    assert by_symbol["CCC"].shadow_factor_rank == 1
    assert by_symbol["BBB"].shadow_factor_rank == 2
    assert by_symbol["AAA"].shadow_factor_rank == 3
    assert by_symbol["CCC"].rank_delta == 2
    assert by_symbol["AAA"].rank_delta == -2
    assert by_symbol["CCC"].shadow_factor_score == pytest.approx(0.75)
    assert by_symbol["AAA"].score_delta == pytest.approx(0.25 - 0.90)


def test_missing_matrix_and_missing_library_warn_without_raising() -> None:
    candidates = [{"symbol": "AAA", "official_score": 0.5}]
    config = ShadowScoringConfig(config_id="config", as_of="2026-04-27")

    missing_matrix_scores = build_shadow_candidate_scores(
        candidates=candidates,
        library=_library(["missing"]),
        factor_matrices=[],
        config=config,
    )
    missing_library_scores = build_shadow_candidate_scores(
        candidates=candidates,
        library=None,
        factor_matrices=[],
        config=config,
    )

    assert missing_matrix_scores[0].warning_codes == [
        SHADOW_SCORE_STATUS_INSUFFICIENT_DATA,
        SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX,
    ]
    assert missing_library_scores[0].warning_codes == [
        SHADOW_SCORE_STATUS_INSUFFICIENT_DATA,
        SHADOW_SCORE_STATUS_LIBRARY_MISSING,
    ]
    assert missing_matrix_scores[0].shadow_factor_score is None
    assert missing_library_scores[0].shadow_factor_score is None


def test_audit_blocked_factor_warning_is_shadow_only() -> None:
    config = ShadowScoringConfig(config_id="config", as_of="2026-04-27")

    scores = build_shadow_candidate_scores(
        candidates=[{"symbol": "AAA", "official_score": 0.5}],
        library=_library(["allowed", "blocked"]),
        factor_matrices=[
            _matrix("allowed", [[1.0]], dates=["2026-04-27"], symbols=["AAA"]),
            _matrix("blocked", [[9.0]], dates=["2026-04-27"], symbols=["AAA"]),
        ],
        audit_report=_audit(["blocked"]),
        config=config,
    )

    assert SHADOW_SCORE_STATUS_AUDIT_BLOCKED in scores[0].warning_codes
    assert [factor.factor_id for factor in scores[0].factor_scores] == ["allowed"]


def test_comparison_report_overlap_deltas_round_trip_and_renderers() -> None:
    report = _complete_report()
    round_trip = ShadowScoringComparisonReport.from_dict(report.to_dict())
    markdown = render_shadow_scoring_comparison_markdown(report)
    dashboard = build_shadow_scoring_dashboard_payload(report)

    assert report.status == SHADOW_COMPARISON_STATUS_OK
    assert report.official_top_symbols == ["AAA", "BBB"]
    assert report.shadow_top_symbols == ["CCC", "BBB"]
    assert report.overlap_top_symbols == ["BBB"]
    assert report.overlap_ratio == pytest.approx(0.5)
    assert report.largest_positive_rank_deltas[0]["symbol"] == "CCC"
    assert report.largest_negative_rank_deltas[0]["symbol"] == "AAA"
    assert round_trip.to_dict() == report.to_dict()
    assert ShadowCandidateScore.from_dict(report.candidate_scores[0].to_dict()).to_dict() == (
        report.candidate_scores[0].to_dict()
    )
    assert SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE in markdown
    assert dashboard["status"] == SHADOW_COMPARISON_STATUS_OK
    json.dumps(dashboard, ensure_ascii=False, sort_keys=True, allow_nan=False)


def test_low_coverage_creates_warn_status() -> None:
    config = ShadowScoringConfig(
        config_id="config-low-coverage",
        as_of="2026-04-27",
        min_factor_coverage_ratio=0.90,
    )

    report = build_shadow_scoring_comparison_report(
        candidates=_candidates_without_ranks(),
        library=_library(["momentum", "missing"]),
        factor_matrices=[
            _matrix("momentum", [[0.10, 0.20], [0.10, 0.10], [0.10, 0.30]]),
        ],
        config=config,
        generated_at="2026-04-27T12:00:00",
    )

    assert report.status == SHADOW_COMPARISON_STATUS_WARN
    assert SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX in report.warning_codes
    assert "low_factor_coverage" in report.warning_codes
