from __future__ import annotations

import json

import pytest

from quant_investor.factors.shadow_scoring import (
    SHADOW_COMPARISON_STATUS_OK,
    SHADOW_SCORE_STATUS_OK,
    ShadowCandidateScore,
    ShadowFactorScore,
    ShadowScoringComparisonReport,
    ShadowScoringConfig,
    build_shadow_scoring_dashboard_payload,
    render_shadow_scoring_comparison_markdown,
)
from quant_investor.factors.store import FactorShadowScoringStore


def _factor_score() -> ShadowFactorScore:
    return ShadowFactorScore(
        factor_id="momentum",
        factor_version="v1",
        symbol="AAA",
        as_of="2026-04-27",
        raw_value=0.2,
        normalized_score=1.0,
        rank=1,
        coverage_status=SHADOW_SCORE_STATUS_OK,
    )


def _candidate_score() -> ShadowCandidateScore:
    return ShadowCandidateScore(
        symbol="AAA",
        name="Alpha",
        as_of="2026-04-27",
        official_score=0.8,
        official_rank=1,
        shadow_factor_score=1.0,
        shadow_factor_rank=1,
        rank_delta=0,
        score_delta=0.2,
        factor_count=1,
        covered_factor_count=1,
        factor_coverage_ratio=1.0,
        factor_scores=[_factor_score()],
    )


def _report() -> ShadowScoringComparisonReport:
    candidate = _candidate_score()
    return ShadowScoringComparisonReport(
        report_id="shadow-report-store-fixture",
        generated_at="2026-04-27T12:00:00",
        as_of="2026-04-27",
        config=ShadowScoringConfig(config_id="config-store", as_of="2026-04-27", top_n=1),
        production_factor_count=1,
        used_factor_count=1,
        candidate_count=1,
        scored_candidate_count=1,
        average_factor_coverage_ratio=1.0,
        official_top_symbols=["AAA"],
        shadow_top_symbols=["AAA"],
        overlap_top_symbols=["AAA"],
        overlap_ratio=1.0,
        largest_positive_rank_deltas=[],
        largest_negative_rank_deltas=[],
        warning_codes=[],
        status=SHADOW_COMPARISON_STATUS_OK,
        candidate_scores=[candidate],
        metadata={"fixture": True, "non_runtime_impact": True},
    )


def test_append_and_read_factor_scores(tmp_path) -> None:
    store = FactorShadowScoringStore(tmp_path / "shadow")
    score = _factor_score()

    count = store.append_factor_scores([score])

    assert count == 1
    assert store.read_factor_scores()[0].to_dict() == score.to_dict()


def test_append_and_read_candidate_scores(tmp_path) -> None:
    store = FactorShadowScoringStore(tmp_path / "shadow")
    score = _candidate_score()

    count = store.append_candidate_scores([score])

    assert count == 1
    assert store.read_candidate_scores()[0].to_dict() == score.to_dict()


def test_append_and_read_comparison_report(tmp_path) -> None:
    store = FactorShadowScoringStore(tmp_path / "shadow")
    report = _report()

    store.append_comparison_report(report)

    assert store.read_comparison_reports()[0].to_dict() == report.to_dict()
    assert store.get_comparison_report_ids() == {report.report_id}


def test_duplicate_report_id_raises(tmp_path) -> None:
    store = FactorShadowScoringStore(tmp_path / "shadow")
    report = _report()
    store.append_comparison_report(report)

    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_comparison_report(report)


def test_save_and_load_markdown_and_dashboard(tmp_path) -> None:
    store = FactorShadowScoringStore(tmp_path / "shadow")
    report = _report()
    markdown = render_shadow_scoring_comparison_markdown(report)
    dashboard = build_shadow_scoring_dashboard_payload(report)

    markdown_path = store.save_markdown(markdown)
    dashboard_path = store.save_dashboard_payload(dashboard)

    assert markdown_path == store.comparison_markdown_path
    assert dashboard_path == store.dashboard_payload_path
    assert store.load_markdown() == markdown
    assert store.load_dashboard_payload()["status"] == SHADOW_COMPARISON_STATUS_OK
    json.dumps(store.load_dashboard_payload(), ensure_ascii=False, sort_keys=True)


def test_malformed_json_raises_clear_value_error(tmp_path) -> None:
    store = FactorShadowScoringStore(tmp_path / "shadow")
    store.factor_scores_path.parent.mkdir(parents=True, exist_ok=True)
    store.factor_scores_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_factor_scores()


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "shadow"
    store = FactorShadowScoringStore(root)

    assert not root.exists()
    store.append_factor_scores([_factor_score()])

    assert root.exists()
    assert store.factor_scores_path.exists()
