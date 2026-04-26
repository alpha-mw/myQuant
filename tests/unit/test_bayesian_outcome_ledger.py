from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from quant_investor.agent_protocol import GlobalContext
from quant_investor.bayesian.calibration import CalibrationStore
from quant_investor.bayesian.outcome_ledger import (
    OUTCOME_STATUS_RESOLVED,
    OutcomeLedgerStore,
    build_prediction_record,
    build_prediction_records,
    extract_branch_confidences,
    extract_branch_scores,
    make_deterministic_run_id,
    make_outcome_id,
    make_prediction_id,
)
from quant_investor.bayesian.types import LikelihoodSet, PosteriorResult, PriorSet
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import OUTCOME_LEDGER_SCHEMA_VERSION


@dataclass
class BranchStub:
    final_score: float
    final_confidence: float
    symbol_scores: dict[str, float] = field(default_factory=dict)


def _context() -> GlobalContext:
    return GlobalContext(
        market="CN",
        universe_key="full_a",
        rebalance_date="2026-04-26",
        latest_trade_date="2026-04-25",
        universe_hash="hash123",
        macro_regime="趋势上涨",
    )


def _posterior(symbol: str = "000001.SZ", rank: int = 3) -> PosteriorResult:
    return PosteriorResult(
        symbol=symbol,
        company_name="平安银行",
        prior=PriorSet(composite_prior=0.56, market_prior=0.55),
        likelihoods=LikelihoodSet(
            kline_likelihood=0.61,
            quant_likelihood=0.67,
            fundamental_likelihood=0.58,
            intelligence_likelihood=0.52,
        ),
        posterior_win_rate=0.66,
        posterior_expected_alpha=0.04,
        posterior_confidence=0.72,
        posterior_action_score=0.63,
        posterior_edge_after_costs=0.031,
        posterior_capacity_penalty=0.01,
        rank=rank,
        coverage_discount=0.02,
        fallback_penalty=0.03,
        correlation_discount=0.04,
        data_quality_penalty=0.05,
        regime_adjustment=0.06,
        evidence_sources=["quant", "kline"],
        action_threshold_used=0.58,
        metadata={"momentum_strength": 0.81},
    )


def _branch_results() -> dict[str, BranchStub]:
    return {
        "quant": BranchStub(final_score=0.20, final_confidence=0.70, symbol_scores={"000001.SZ": 0.31}),
        "kline": BranchStub(final_score=0.10, final_confidence=0.60, symbol_scores={"000001.SZ": 0.42}),
        "intelligence": BranchStub(final_score=0.05, final_confidence=0.50),
        "noncanonical": BranchStub(final_score=0.99, final_confidence=0.99, symbol_scores={"000001.SZ": 0.99}),
    }


def test_deterministic_ids_are_stable() -> None:
    run_args = {
        "market": "CN",
        "universe_key": "full_a",
        "rebalance_date": "2026-04-26",
        "universe_hash": "hash123",
    }
    run_id = make_deterministic_run_id(**run_args)

    assert make_deterministic_run_id(**run_args) == run_id
    assert make_prediction_id(run_id=run_id, symbol="000001.SZ", horizon_days=20) == make_prediction_id(
        run_id=run_id,
        symbol="000001.SZ",
        horizon_days=20,
    )
    prediction_id = make_prediction_id(run_id=run_id, symbol="000001.SZ", horizon_days=20)
    assert make_outcome_id(
        prediction_id=prediction_id,
        resolution_date="2026-05-26",
        status=OUTCOME_STATUS_RESOLVED,
    ) == make_outcome_id(
        prediction_id=prediction_id,
        resolution_date="2026-05-26",
        status=OUTCOME_STATUS_RESOLVED,
    )


def test_prediction_builder_round_trip_captures_posterior_and_branches() -> None:
    record = build_prediction_record(_posterior(), _context(), _branch_results())
    payload = record.to_dict()
    round_trip = type(record).from_dict(payload)

    assert round_trip.schema_version == OUTCOME_LEDGER_SCHEMA_VERSION
    assert round_trip.horizon_label == "20D"
    assert round_trip.rank == 3
    assert round_trip.prior["composite_prior"] == pytest.approx(0.56)
    assert round_trip.likelihoods["quant_likelihood"] == pytest.approx(0.67)
    assert round_trip.branch_scores["quant"] == pytest.approx(0.31)
    assert round_trip.branch_scores["kline"] == pytest.approx(0.42)
    assert round_trip.branch_scores["intelligence"] == pytest.approx(0.05)
    assert round_trip.branch_confidences["quant"] == pytest.approx(0.70)
    assert round_trip.posterior_win_rate == pytest.approx(0.66)
    assert round_trip.posterior_expected_alpha == pytest.approx(0.04)
    assert round_trip.posterior_edge_after_costs == pytest.approx(0.031)
    assert round_trip.evidence_sources == ["quant", "kline"]
    assert round_trip.metadata["outcome_ledger_schema_version"] == OUTCOME_LEDGER_SCHEMA_VERSION


def test_store_round_trip_reads_prediction_values(tmp_path: Path) -> None:
    store = OutcomeLedgerStore(tmp_path)
    record = build_prediction_record(_posterior(), _context(), _branch_results())

    store.append_prediction(record)

    records = store.read_predictions()
    assert len(records) == 1
    assert records[0].prediction_id == record.prediction_id
    assert records[0].symbol == "000001.SZ"


def test_duplicate_prediction_append_raises(tmp_path: Path) -> None:
    store = OutcomeLedgerStore(tmp_path)
    record = build_prediction_record(_posterior(), _context(), _branch_results())
    store.append_prediction(record)

    with pytest.raises(ValueError, match="Duplicate prediction_id"):
        store.append_prediction(record)


def test_batch_builder_rejects_duplicate_prediction_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate prediction_id"):
        build_prediction_records([_posterior(), _posterior()], _context(), _branch_results())


def test_outcome_resolution_computes_excess_return(tmp_path: Path) -> None:
    store = OutcomeLedgerStore(tmp_path)
    record = build_prediction_record(_posterior(), _context(), _branch_results())
    store.append_prediction(record)

    outcome = store.resolve_prediction(
        record.prediction_id,
        resolution_date="2026-05-26",
        realized_return=0.08,
        benchmark_return=0.03,
        entry_price=10.0,
        exit_price=10.8,
    )

    assert outcome.status == OUTCOME_STATUS_RESOLVED
    assert outcome.excess_return == pytest.approx(0.05)
    outcomes = store.read_outcomes()
    assert len(outcomes) == 1
    assert outcomes[0].prediction_id == record.prediction_id
    assert outcomes[0].excess_return == pytest.approx(0.05)


def test_unresolved_iterator_filters_after_resolution(tmp_path: Path) -> None:
    store = OutcomeLedgerStore(tmp_path)
    record = build_prediction_record(_posterior(), _context(), _branch_results())
    store.append_prediction(record)

    assert [item.prediction_id for item in store.iter_unresolved_predictions(horizon_days=20)] == [record.prediction_id]

    store.resolve_prediction(
        record.prediction_id,
        resolution_date="2026-05-26",
        realized_return=0.01,
    )

    assert store.iter_unresolved_predictions() == []


def test_canonical_branch_defaults_and_noncanonical_exclusion() -> None:
    branch_results = {"quant": _branch_results()["quant"], "noncanonical": _branch_results()["noncanonical"]}

    scores = extract_branch_scores(branch_results, "000001.SZ")
    confidences = extract_branch_confidences(branch_results, "000001.SZ")

    assert list(scores) == list(CANONICAL_BRANCH_ORDER)
    assert "noncanonical" not in scores
    assert scores["quant"] == pytest.approx(0.31)
    assert scores["macro"] == pytest.approx(0.0)
    assert confidences["quant"] == pytest.approx(0.70)
    assert confidences["macro"] == pytest.approx(0.0)


def test_malformed_json_raises_clear_value_error(tmp_path: Path) -> None:
    store = OutcomeLedgerStore(tmp_path)
    store.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    store.predictions_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_predictions()


def test_calibration_store_legacy_outcome_file_still_created(tmp_path: Path) -> None:
    store = CalibrationStore(str(tmp_path / "bayesian_calibration.json"))

    store.record_outcome(
        symbol="000001.SZ",
        branch_name="quant",
        predicted_score=0.25,
        realized_return=0.03,
        run_date="2026-04-26",
    )

    legacy_path = tmp_path / "bayesian_outcomes.jsonl"
    assert legacy_path.exists()
    payload = json.loads(legacy_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["symbol"] == "000001.SZ"
    assert payload["branch"] == "quant"
    assert payload["bucket"] == "positive"
