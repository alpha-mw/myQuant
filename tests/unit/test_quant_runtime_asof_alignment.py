from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

import quant_investor.agents.quant_agent as quant_agent_module
import quant_investor.factors.runtime as runtime_module
from quant_investor.agents.quant_agent import QuantAgent
from quant_investor.branch_contracts import UnifiedDataBundle
from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import (
    MinedFactorRegistry,
    MinedFactorScorer,
    ProductionEvaluationContext,
    RuntimeFactorScore,
    _mint_production_evaluation_context,
    production_evaluation_context_sha256,
    production_symbol_set_sha256,
    validate_production_evaluation_context,
)
from quant_investor.market.dag.context import (
    _build_production_evaluation_context,
    _researchable_frame_subset,
)
from quant_investor.market.pit_universe import PITUniverseRecord
from quant_investor.market.read_result import MarketDataReadResult


AS_OF = "20260106"
PIT_OBSERVED_AT = "2026-01-06T00:00:00Z"


def _pit_row(symbol: str, *, source_run_id: str) -> dict[str, object]:
    return PITUniverseRecord(
        symbol=symbol,
        source_list_status="L",
        list_date="20200101",
        effective_from="20200101",
        observed_at=PIT_OBSERVED_AT,
        source="tushare.stock_basic",
        source_run_id=source_run_id,
        raw_payload_hash=f"fixture-{symbol}",
        membership_quality="ok",
    ).to_dict()


def _frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for index in range(20):
        symbol = f"S{index:02d}"
        frames[symbol] = pd.DataFrame(
            {
                "ts_code": [symbol] * 6,
                "trade_date": pd.date_range("2026-01-01", periods=6),
                "adj_close": np.linspace(10.0 + index, 11.0 + index, 6),
                "vol": np.linspace(100.0, 120.0 + index, 6),
                "amount": np.linspace(1_000.0 + index, 1_200.0 + index, 6),
            }
        )
    return frames


def _context(
    frames: dict[str, pd.DataFrame],
    tmp_path: Path,
    *,
    market: str = "CN",
) -> ProductionEvaluationContext:
    pit_status = "verified" if market == "CN" else "not_applicable"
    artifact_paths: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    for name in ("snapshot_pointer", "snapshot_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "snapshot_id": "snapshot-20260106",
                    "latest_complete_trade_date": AS_OF,
                }
            ),
            encoding="utf-8",
        )
        artifact_paths[name] = str(path.resolve())
        artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    calendar_path = tmp_path / "open_day_calendar.json"
    calendar_path.write_text(
        json.dumps(
            {
                "schema_version": "market-open-days.v1",
                "market": market,
                "open_dates": [AS_OF],
            }
        ),
        encoding="utf-8",
    )
    artifact_paths["open_day_calendar"] = str(calendar_path.resolve())
    artifact_hashes["open_day_calendar"] = hashlib.sha256(
        calendar_path.read_bytes()
    ).hexdigest()
    if market == "CN":
        pit_canonical = tmp_path / "pit_canonical.parquet"
        pd.DataFrame(
            [
                _pit_row(symbol, source_run_id="pit-snapshot-20260106")
                for symbol in frames
            ]
        ).to_parquet(pit_canonical, index=False)
        pit_manifest = tmp_path / "pit_manifest.json"
        pit_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "cn_pit_universe_manifest.v1",
                    "membership_schema_version": "cn_pit_universe.v1",
                    "source": "tushare.stock_basic",
                    "source_run_id": "pit-snapshot-20260106",
                    "observed_at": PIT_OBSERVED_AT,
                    "canonical_path": str(pit_canonical.resolve()),
                    "row_count": len(frames),
                }
            ),
            encoding="utf-8",
        )
        artifact_paths.update(
            {
                "pit_manifest": str(pit_manifest.resolve()),
                "pit_canonical": str(pit_canonical.resolve()),
            }
        )
        artifact_hashes.update(
            {
                "pit_manifest": hashlib.sha256(pit_manifest.read_bytes()).hexdigest(),
                "pit_canonical": hashlib.sha256(pit_canonical.read_bytes()).hexdigest(),
            }
        )
    return _mint_production_evaluation_context(
        evaluation_as_of=AS_OF,
        market=market,
        universe_key="full_a" if market == "CN" else "sp500",
        universe_sha256=production_symbol_set_sha256(list(frames)),
        snapshot_id="snapshot-20260106",
        latest_complete_trade_date=AS_OF,
        pit_membership_status=pit_status,
        pit_membership_as_of=AS_OF if pit_status == "verified" else "",
        pit_membership_proof_sha256="a" * 64 if pit_status == "verified" else "",
        pit_membership_not_applicable_reason=(
            "market_not_cn" if pit_status == "not_applicable" else ""
        ),
        open_day_proof_sha256=artifact_hashes["open_day_calendar"],
        read_result_provenance_sha256="c" * 64,
        verified_artifact_paths=artifact_paths,
        verified_artifact_sha256s=artifact_hashes,
    )


def _record() -> FactorRecord:
    return FactorRecord(
        name="pv_low_dollar_volume_5d",
        version="v1",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        category="liquidity",
        implementation="price_volume:pv_low_dollar_volume_5d",
        weight=0.05,
        direction=1.0,
        gate_results=[
            GateResult(
                gate_id=index,
                gate_key=f"gate_{index}",
                title=f"Gate {index}",
                passed=True,
            )
            for index in range(1, 9)
        ],
        metadata={
            "factor_family": "liquidity",
            "dominant_primitive_cluster": "dollar_volume",
        },
    )


def _ready_scorer(monkeypatch: pytest.MonkeyPatch) -> MinedFactorScorer:
    record = _record()
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records([record]))
    monkeypatch.setattr(
        scorer,
        "_runtime_contract",
        lambda: (
            [record],
            {
                "status": "ready",
                "factor_mode": "governed_mined_factors",
                "confidence_multiplier": 1.0,
                "production_eligible": True,
                "blockers": [],
                "factor_runtime_contracts": {
                    record.name: {
                        "required_columns": ["trade_date", "amount"],
                        "lookback_rows": 5,
                        "gate2_min_coverage_rate": 1.0,
                        "min_cross_section": 20,
                    }
                },
            },
        ),
    )
    return scorer


def test_production_evaluation_context_is_frozen_and_canonically_hashed(
    tmp_path: Path,
) -> None:
    frames = _frames()
    context = _context(frames, tmp_path)

    assert context.context_sha256 == production_evaluation_context_sha256(context)
    assert context.to_metadata()["context_sha256"] == context.context_sha256
    assert replace(context).context_sha256 == context.context_sha256
    with pytest.raises(FrozenInstanceError):
        context.snapshot_id = "forged"  # type: ignore[misc]


def test_production_scorer_requires_explicit_evaluation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _ready_scorer(monkeypatch)

    result = scorer.score(_frames())

    assert result.governance_status == "governance_blocked"
    assert "production_evaluation_context_missing" in result.runtime_blockers


def test_self_reported_unsealed_context_cannot_enable_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _frames()
    context = ProductionEvaluationContext(
        evaluation_as_of=AS_OF,
        market="CN",
        universe_key="full_a",
        universe_sha256=production_symbol_set_sha256(list(frames)),
        snapshot_id="snapshot-20260106",
        latest_complete_trade_date=AS_OF,
        pit_membership_status="verified",
        pit_membership_as_of=AS_OF,
        pit_membership_proof_sha256="a" * 64,
        pit_membership_not_applicable_reason="",
        open_day_proof_sha256="b" * 64,
        read_result_provenance_sha256="c" * 64,
    )

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )

    assert result.governance_status == "governance_blocked"
    assert "production_evaluation_context_not_readback_verified" in result.runtime_blockers


@pytest.mark.parametrize(
    ("corruption", "expected_blocker"),
    [
        ("artifact_shape", "production_verified_artifact_set_invalid"),
        ("scalar_type", "production_evaluation_context_field_type_invalid"),
    ],
)
def test_malformed_public_evaluation_context_blocks_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    expected_blocker: str,
) -> None:
    frames = _frames()
    kwargs = {
        "evaluation_as_of": AS_OF,
        "market": "CN",
        "universe_key": "full_a",
        "universe_sha256": production_symbol_set_sha256(list(frames)),
        "snapshot_id": "snapshot-20260106",
        "latest_complete_trade_date": AS_OF,
        "pit_membership_status": "verified",
        "pit_membership_as_of": AS_OF,
        "pit_membership_proof_sha256": "a" * 64,
        "pit_membership_not_applicable_reason": "",
        "open_day_proof_sha256": "b" * 64,
        "read_result_provenance_sha256": "c" * 64,
    }
    if corruption == "artifact_shape":
        kwargs["verified_artifact_paths"] = ("bad",)
        kwargs["verified_artifact_sha256s"] = ("bad",)
    else:
        kwargs["market"] = 7
    context = ProductionEvaluationContext(**kwargs)  # type: ignore[arg-type]

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )

    assert result.governance_status == "governance_blocked"
    assert "production_evaluation_context_not_readback_verified" in (
        result.runtime_blockers
    )
    assert expected_blocker in result.runtime_blockers


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("future_head", "production_frame_date_order_invalid"),
        ("future_tail", "production_frame_future_row"),
        ("stale_terminal", "production_frame_terminal_date_mismatch"),
        ("duplicate_date", "production_frame_duplicate_trade_date"),
        ("duplicate_head", "production_frame_duplicate_trade_date"),
        ("unordered_no_future", "production_frame_date_order_invalid"),
        ("intraday", "production_frame_trade_date_not_daily"),
        ("timezone", "production_frame_trade_date_timezone_aware"),
        ("unparseable", "production_frame_trade_date_unparseable"),
        ("invalid_calendar", "production_frame_trade_date_unparseable"),
        ("symbol_missing", "production_frame_symbol_column_missing"),
        ("symbol_mismatch", "production_frame_symbol_mismatch"),
        ("dual_symbol_conflict", "production_frame_symbol_mismatch"),
    ],
)
def test_production_scorer_rejects_temporal_or_symbol_misalignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    blocker: str,
) -> None:
    frames = _frames()
    context = _context(frames, tmp_path)
    symbol = next(iter(frames))
    frame = frames[symbol]
    if mutation == "future_head":
        frames[symbol] = pd.concat(
            [
                frame.iloc[[0]].assign(trade_date=pd.Timestamp("2026-01-07")),
                frame,
            ],
            ignore_index=True,
        )
    elif mutation == "future_tail":
        frames[symbol] = pd.concat(
            [frame, frame.iloc[[-1]].assign(trade_date=pd.Timestamp("2026-01-07"))],
            ignore_index=True,
        )
    elif mutation == "stale_terminal":
        frames[symbol] = frame.iloc[:-1].copy()
    elif mutation == "duplicate_date":
        frames[symbol] = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    elif mutation == "duplicate_head":
        frames[symbol] = pd.concat([frame.iloc[[-1]], frame], ignore_index=True)
    elif mutation == "unordered_no_future":
        frames[symbol] = pd.concat([frame.iloc[[-1]], frame.iloc[:-1]], ignore_index=True)
    elif mutation == "intraday":
        frames[symbol] = frame.copy()
        frames[symbol].loc[frame.index[-1], "trade_date"] = pd.Timestamp(
            "2026-01-06 09:30:00"
        )
    elif mutation == "timezone":
        frames[symbol] = frame.copy()
        frames[symbol]["trade_date"] = frames[symbol]["trade_date"].dt.tz_localize(
            "Asia/Shanghai"
        )
    elif mutation == "unparseable":
        frames[symbol] = frame.copy()
        frames[symbol]["trade_date"] = frames[symbol]["trade_date"].astype(object)
        frames[symbol].loc[frame.index[-1], "trade_date"] = "not-a-date"
    elif mutation == "invalid_calendar":
        frames[symbol] = frame.copy()
        frames[symbol]["trade_date"] = frames[symbol]["trade_date"].astype(object)
        frames[symbol].loc[frame.index[-1], "trade_date"] = "99999999"
    elif mutation == "symbol_missing":
        frames[symbol] = frame.drop(columns=["ts_code"])
    elif mutation == "symbol_mismatch":
        frames[symbol] = frame.copy()
        frames[symbol].loc[frame.index[-1], "ts_code"] = "FOREIGN"
    elif mutation == "dual_symbol_conflict":
        frames[symbol] = frame.copy()
        frames[symbol]["symbol"] = symbol
        frames[symbol].loc[frame.index[-1], "symbol"] = "FOREIGN"

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )

    assert result.governance_status == "governance_blocked"
    assert any(blocker in item for item in result.runtime_blockers)


def test_production_frame_validation_meets_full_a_throughput_budget() -> None:
    frame_count = 1_200
    rows_per_frame = 280
    dates = pd.date_range(end="2026-01-06", periods=rows_per_frame, freq="B")
    frames = {
        (symbol := f"B{index:04d}"): pd.DataFrame(
            {
                "ts_code": np.full(rows_per_frame, symbol, dtype=object),
                "trade_date": dates,
            }
        )
        for index in range(frame_count)
    }
    context = ProductionEvaluationContext(
        evaluation_as_of=AS_OF,
        market="CN",
        universe_key="full_a",
        universe_sha256="a" * 64,
        snapshot_id="benchmark",
        latest_complete_trade_date=AS_OF,
        pit_membership_status="verified",
        pit_membership_as_of=AS_OF,
        pit_membership_proof_sha256="b" * 64,
        pit_membership_not_applicable_reason="",
        open_day_proof_sha256="c" * 64,
        read_result_provenance_sha256="d" * 64,
    )
    symbols = list(frames)

    assert runtime_module._validate_production_frames(
        frames,
        symbols=symbols,
        context=context,
    ) is None
    timings: list[float] = []
    for _ in range(3):
        started = perf_counter()
        assert runtime_module._validate_production_frames(
            frames,
            symbols=symbols,
            context=context,
        ) is None
        timings.append(perf_counter() - started)

    median_seconds = median(timings)
    rows_per_second = frame_count * rows_per_frame / median_seconds
    assert median_seconds < 3.5
    assert rows_per_second >= 100_000


@pytest.mark.parametrize(
    ("changed_context", "blocker"),
    [
        (lambda context: replace(context, universe_sha256="c" * 64), "universe"),
        (
            lambda context: replace(
                context,
                latest_complete_trade_date="20260107",
            ),
            "latest_complete_trade_date",
        ),
        (
            lambda context: replace(
                context,
                pit_membership_proof_sha256="",
            ),
            "pit_membership_proof",
        ),
        (
            lambda context: replace(context, open_day_proof_sha256=""),
            "open_day_proof",
        ),
    ],
)
def test_production_scorer_rejects_missing_or_mismatched_context_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_context,
    blocker: str,
) -> None:
    frames = _frames()
    context = changed_context(_context(frames, tmp_path))

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )

    assert result.governance_status == "governance_blocked"
    assert any(blocker in item for item in result.runtime_blockers)


def test_valid_context_is_bound_into_score_metadata_and_output_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    context = _context(frames, tmp_path)

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )
    metadata = result.to_metadata()

    assert result.governance_status == "ready"
    assert metadata["production_evaluation_context"] == context.to_metadata()
    assert metadata["production_evaluation_context_sha256"] == context.context_sha256
    assert len(metadata["production_output_attestation_sha256"]) == 64


def test_internal_history_gap_is_allowed_when_lookback_and_terminal_align(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = {
        symbol: frame.drop(index=frame.index[2]).reset_index(drop=True)
        for symbol, frame in _frames().items()
    }
    context = _context(frames, tmp_path)

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )

    assert result.governance_status == "ready"


@pytest.mark.parametrize("artifact_name", ["snapshot_manifest", "pit_canonical"])
def test_verified_artifact_bytes_drift_blocks_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_name: str,
) -> None:
    frames = _frames()
    context = _context(frames, tmp_path)
    artifact_path = Path(dict(context.verified_artifact_paths)[artifact_name])
    artifact_path.write_text('{"forged":true}', encoding="utf-8")

    result = _ready_scorer(monkeypatch).score(
        frames,
        evaluation_context=context,
    )

    assert result.governance_status == "governance_blocked"
    assert any("artifact_bytes_drift" in item for item in result.runtime_blockers)


def test_validator_rejects_reused_resolved_artifact_path(tmp_path: Path) -> None:
    frames = _frames()
    original = _context(frames, tmp_path)
    artifact_paths = dict(original.verified_artifact_paths)
    artifact_hashes = dict(original.verified_artifact_sha256s)
    artifact_paths["open_day_calendar"] = artifact_paths["snapshot_pointer"]
    artifact_hashes["open_day_calendar"] = artifact_hashes["snapshot_pointer"]
    reused = _mint_production_evaluation_context(
        evaluation_as_of=original.evaluation_as_of,
        market=original.market,
        universe_key=original.universe_key,
        universe_sha256=original.universe_sha256,
        snapshot_id=original.snapshot_id,
        latest_complete_trade_date=original.latest_complete_trade_date,
        pit_membership_status=original.pit_membership_status,
        pit_membership_as_of=original.pit_membership_as_of,
        pit_membership_proof_sha256=original.pit_membership_proof_sha256,
        pit_membership_not_applicable_reason=(
            original.pit_membership_not_applicable_reason
        ),
        open_day_proof_sha256=artifact_hashes["open_day_calendar"],
        read_result_provenance_sha256=(
            original.read_result_provenance_sha256
        ),
        verified_artifact_paths=artifact_paths,
        verified_artifact_sha256s=artifact_hashes,
    )

    blockers = validate_production_evaluation_context(
        reused,
        expected_symbols=list(frames),
    )

    assert "production_verified_artifact_path_reused" in blockers


def _dag_snapshots(
    frames: dict[str, pd.DataFrame],
    tmp_path: Path,
    *,
    market: str = "CN",
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, MarketDataReadResult],
]:
    symbols = list(frames)
    pointer_path = tmp_path / "_latest.json"
    manifest_path = tmp_path / "snapshot.json"
    table_root = tmp_path / "parquet" / market.lower() / "bars"
    serving_root = tmp_path / "parquet_serving" / market.lower() / "bars"
    table_root.mkdir(parents=True)
    serving_root.mkdir(parents=True)
    pointer_payload = {
        "snapshot_id": "snapshot-20260106",
        "status": "OK",
        "blockers": [],
        "latest_complete_trade_date": AS_OF,
        "latest_trade_date": AS_OF,
        "manifest_path": str(manifest_path.resolve()),
        "table_root": str(table_root.resolve()),
        "derived_serving_root": str(serving_root.resolve()),
    }
    manifest_payload = {
        **pointer_payload,
        "market": market,
        "readback_validated": True,
    }
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    calendar_path = tmp_path / "open_day_calendar.json"
    calendar_path.write_text(
        json.dumps(
            {
                "schema_version": "market-open-days.v1",
                "market": market,
                "open_dates": [AS_OF],
            }
        ),
        encoding="utf-8",
    )
    gate = {
        "status": "ok",
        "healthy": True,
        "blockers": [],
        "snapshot_id": "snapshot-20260106",
        "latest_complete_trade_date": AS_OF,
        "latest_pointer_path": str(pointer_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "table_root": str(table_root.resolve()),
        "serving_root": str(serving_root.resolve()),
    }
    reader_snapshot: dict[str, object] = {
        **gate,
        "market": market,
        "storage_layer": "canonical+serving",
        "resolution_strategy": "strict_parquet_serving",
    }
    scoped: dict[str, object] = {
        "market": market,
        "universe_key": "full_a" if market == "CN" else "sp500",
        "local_latest_trade_date": AS_OF,
        "strict_parquet_gate": dict(gate),
        "open_day_calendar": {"path": str(calendar_path.resolve())},
    }
    if market == "CN":
        pit_manifest_path = tmp_path / "pit_manifest.json"
        pit_canonical_path = tmp_path / "pit_canonical.parquet"
        pit_rows = [
            _pit_row(symbol, source_run_id="pit-snapshot-20260106")
            for symbol in symbols
        ]
        pd.DataFrame(pit_rows).to_parquet(pit_canonical_path, index=False)
        pit_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "cn_pit_universe_manifest.v1",
                    "membership_schema_version": "cn_pit_universe.v1",
                    "source": "tushare.stock_basic",
                    "source_run_id": "pit-snapshot-20260106",
                    "observed_at": PIT_OBSERVED_AT,
                    "canonical_path": str(pit_canonical_path.resolve()),
                    "row_count": len(pit_rows),
                }
            ),
            encoding="utf-8",
        )
        scoped["pit_universe"] = {
            "enabled": True,
            "required": True,
            "status": "applied",
            "as_of": AS_OF,
            "snapshot_id": "pit-snapshot-20260106",
            "manifest_path": str(pit_manifest_path.resolve()),
            "canonical_path": str(pit_canonical_path.resolve()),
            "coverage_ratio": 1.0,
            "missing_count": 0,
            "statuses": {
                symbol: {
                    "symbol": symbol,
                    "date": AS_OF,
                    "in_universe": True,
                    "research_eligible": True,
                    "tradable": True,
                    "reason": "listed",
                    "list_date": "20200101",
                    "delist_date": "",
                    "source_list_status": "L",
                    "observed_at": PIT_OBSERVED_AT,
                    "membership_quality": "ok",
                }
                for symbol in symbols
            },
        }
    read_results = {
        symbol: MarketDataReadResult(
            frame=frames[symbol],
            path=str(table_root.resolve()),
            symbol=symbol,
            universe_key="full_a" if market == "CN" else "sp500",
            resolver_trace={
                **reader_snapshot,
                "resolution_strategy": "strict_parquet_canonical_batch",
            },
            metadata={
                "snapshot_id": "snapshot-20260106",
                "latest_complete_trade_date": AS_OF,
                "storage_layer": "canonical_batch",
                "resolution_strategy": "strict_parquet_canonical_batch",
                "resolved": True,
            },
        )
        for symbol in symbols
    }
    return reader_snapshot, scoped, read_results


def test_dag_context_requires_dual_source_snapshot_and_complete_cn_pit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert blockers == []
    assert context is not None
    assert context.universe_sha256 == production_symbol_set_sha256(list(frames))
    assert context.pit_membership_status == "verified"

    drifted_reader = {**reader_snapshot, "snapshot_id": "different"}
    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=drifted_reader,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )
    assert context is None
    assert "production_snapshot_id_mismatch" in blockers

    pit_date_drift = json.loads(json.dumps(scoped))
    first_symbol = next(iter(frames))
    pit_date_drift["pit_universe"]["statuses"][first_symbol]["date"] = "20260105"
    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=pit_date_drift,
        read_results=read_results,
    )
    assert context is None
    assert f"production_cn_pit_status_mismatch:{first_symbol}" in blockers

    disabled_pit = {**scoped, "pit_universe": {"enabled": False}}
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        False,
    )
    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=disabled_pit,
        read_results=read_results,
    )
    assert context is None
    assert "production_cn_pit_membership_disabled" in blockers


def test_dag_context_rejects_snapshot_root_or_read_path_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    tampered_gate = json.loads(json.dumps(scoped))
    tampered_gate["strict_parquet_gate"]["table_root"] = str(
        (tmp_path / "other-bars").resolve()
    )

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=tampered_gate,
        read_results=read_results,
    )

    assert context is None
    assert "production_snapshot_table_root_mismatch" in blockers

    first_symbol = next(iter(frames))
    read_results[first_symbol].path = str((tmp_path / "forged").resolve())
    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert f"production_read_result_path_mismatch:{first_symbol}" in blockers


@pytest.mark.parametrize(
    ("artifact", "field", "value", "expected_blocker"),
    [
        ("pointer", "status", "BLOCKED", "production_snapshot_pointer_status_invalid"),
        ("pointer", "blockers", ["x"], "production_snapshot_pointer_blockers_invalid"),
        (
            "manifest",
            "readback_validated",
            False,
            "production_snapshot_manifest_not_readback_validated",
        ),
        (
            "manifest",
            "market",
            "US",
            "production_snapshot_manifest_market_mismatch",
        ),
        (
            "manifest",
            "table_root",
            "/forged/table",
            "production_snapshot_table_root_mismatch",
        ),
        (
            "manifest",
            "derived_serving_root",
            "/forged/serving",
            "production_snapshot_serving_root_mismatch",
        ),
        (
            "manifest",
            "manifest_path",
            "/forged/manifest.json",
            "production_snapshot_manifest_path_mismatch",
        ),
    ],
)
def test_dag_context_rejects_same_id_date_semantic_artifact_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact: str,
    field: str,
    value: object,
    expected_blocker: str,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    path_key = "latest_pointer_path" if artifact == "pointer" else "manifest_path"
    artifact_path = Path(str(reader_snapshot[path_key]))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload[field] = value
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert expected_blocker in blockers


def test_non_cn_context_uses_explicit_pit_not_applicable(tmp_path: Path) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(
        frames,
        tmp_path,
        market="US",
    )

    context, blockers = _build_production_evaluation_context(
        market="US",
        universe_key="sp500",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert blockers == []
    assert context is not None
    assert context.pit_membership_status == "not_applicable"
    assert context.pit_membership_not_applicable_reason == "market_not_cn"


def test_dag_context_accepts_serving_reader_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    serving_root = Path(
        str(scoped["strict_parquet_gate"]["serving_root"])
    )
    for symbol, read_result in read_results.items():
        read_result.path = str(
            serving_root / f"symbol={symbol}" / "bars.parquet"
        )
        read_result.metadata.update(
            {
                "storage_layer": "serving",
                "resolved": True,
            }
        )
        read_result.metadata.pop("resolution_strategy", None)
        read_result.resolver_trace["resolution_strategy"] = (
            "strict_parquet_serving"
        )

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert blockers == []
    assert context is not None


def test_dag_context_rejects_as_of_absent_from_independent_open_day_calendar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    sunday = "20260104"
    snapshot_payload = {
        "snapshot_id": "snapshot-20260106",
        "latest_complete_trade_date": sunday,
    }
    for key in ("latest_pointer_path", "manifest_path"):
        Path(str(reader_snapshot[key])).write_text(
            json.dumps(snapshot_payload),
            encoding="utf-8",
        )
    reader_snapshot["latest_complete_trade_date"] = sunday
    scoped["local_latest_trade_date"] = sunday
    scoped["strict_parquet_gate"]["latest_complete_trade_date"] = sunday
    scoped["pit_universe"]["as_of"] = sunday
    for status in scoped["pit_universe"]["statuses"].values():
        status["date"] = sunday
    for read_result in read_results.values():
        read_result.metadata["latest_complete_trade_date"] = sunday

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert "production_evaluation_as_of_not_open_day" in blockers


def test_dag_context_rejects_snapshot_pointer_reused_as_open_day_calendar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    pointer_path = Path(str(reader_snapshot["latest_pointer_path"]))
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer_payload.update(
        {
            "schema_version": "market-open-days.v1",
            "market": "CN",
            "open_dates": [AS_OF],
        }
    )
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf-8")
    scoped["open_day_calendar"] = {"path": str(pointer_path)}

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert "production_open_day_calendar_not_independent" in blockers


def test_dag_context_rejects_non_string_open_day_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    calendar_path = Path(str(scoped["open_day_calendar"]["path"]))
    calendar_payload = json.loads(calendar_path.read_text(encoding="utf-8"))
    calendar_payload["open_dates"] = [{"date": AS_OF}]
    calendar_path.write_text(json.dumps(calendar_payload), encoding="utf-8")

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert "production_open_day_calendar_dates_invalid" in blockers


def test_dag_context_recomputes_pit_status_from_canonical_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    first_symbol = next(iter(frames))
    canonical_path = Path(scoped["pit_universe"]["canonical_path"])
    canonical = pd.read_parquet(canonical_path)
    canonical.loc[canonical["symbol"] == first_symbol, "source_list_status"] = "D"
    canonical.loc[canonical["symbol"] == first_symbol, "delist_date"] = "20260105"
    canonical.loc[canonical["symbol"] == first_symbol, "effective_to"] = "20260105"
    canonical.to_parquet(canonical_path, index=False)

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert f"production_cn_pit_canonical_status_mismatch:{first_symbol}" in blockers


@pytest.mark.parametrize(
    ("corruption", "expected_blocker"),
    [
        ("missing_schema", "production_cn_pit_canonical_schema_mismatch"),
        ("wrong_schema", "production_cn_pit_canonical_schema_mismatch"),
        ("mixed_source_run", "production_cn_pit_canonical_source_run_mismatch"),
        ("unknown_list_status", "production_cn_pit_canonical_list_status_invalid"),
        ("forged_manifest_source", "production_cn_pit_manifest_source_invalid"),
        ("forged_source", "production_cn_pit_canonical_source_mismatch"),
        ("observed_at_mismatch", "production_cn_pit_canonical_observed_at_mismatch"),
        ("missing_field", "production_cn_pit_canonical_columns_mismatch"),
    ],
)
def test_dag_context_rejects_invalid_pit_canonical_record_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
    expected_blocker: str,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    canonical_path = Path(scoped["pit_universe"]["canonical_path"])
    canonical = pd.read_parquet(canonical_path)
    first_row = canonical.index[0]
    if corruption == "missing_schema":
        canonical = canonical.drop(columns=["schema_version"], errors="ignore")
    elif corruption == "wrong_schema":
        canonical["schema_version"] = "cn_pit_universe.v1"
        canonical.loc[first_row, "schema_version"] = "cn_pit_universe.v0"
    elif corruption == "mixed_source_run":
        canonical.loc[first_row, "source_run_id"] = "other-run"
    elif corruption == "unknown_list_status":
        canonical.loc[first_row, "source_list_status"] = "X"
    elif corruption == "forged_manifest_source":
        manifest_path = Path(scoped["pit_universe"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"] = "forged.provider"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "forged_source":
        canonical["source"] = "tushare.stock_basic"
        canonical.loc[first_row, "source"] = "forged.provider"
    elif corruption == "observed_at_mismatch":
        canonical.loc[first_row, "observed_at"] = "2026-01-05T00:00:00Z"
    else:
        canonical = canonical.drop(columns=["raw_payload_hash"], errors="ignore")
    canonical.to_parquet(canonical_path, index=False)

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert expected_blocker in blockers


def test_dag_context_malformed_pit_statuses_returns_stable_blocker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    scoped["pit_universe"]["statuses"] = "bad"

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert "production_cn_pit_statuses_invalid" in blockers


@pytest.mark.parametrize(
    ("corruption", "expected_blocker"),
    [
        ("gate_mapping", "production_snapshot_gate_metadata_invalid"),
        ("calendar_mapping", "production_open_day_calendar_metadata_invalid"),
        ("calendar_json", "production_open_day_calendar_json_invalid"),
        ("read_metadata", "production_read_result_metadata_invalid:S00"),
        ("pit_mapping", "production_cn_pit_metadata_invalid"),
        ("pit_manifest_json", "production_cn_pit_manifest_mismatch"),
        ("pit_canonical_parquet", "production_cn_pit_canonical_parquet_invalid"),
    ],
)
def test_dag_context_malformed_inputs_fail_closed_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
    expected_blocker: str,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    if corruption == "gate_mapping":
        scoped["strict_parquet_gate"] = "bad"
    elif corruption == "calendar_mapping":
        scoped["open_day_calendar"] = "bad"
    elif corruption == "calendar_json":
        Path(str(scoped["open_day_calendar"]["path"])).write_text(
            "{",
            encoding="utf-8",
        )
    elif corruption == "read_metadata":
        read_results["S00"].metadata = "bad"  # type: ignore[assignment]
    elif corruption == "pit_mapping":
        scoped["pit_universe"] = "bad"
    elif corruption == "pit_manifest_json":
        Path(str(scoped["pit_universe"]["manifest_path"])).write_text(
            "{",
            encoding="utf-8",
        )
    else:
        Path(str(scoped["pit_universe"]["canonical_path"])).write_bytes(b"bad")

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert expected_blocker in blockers


def test_dag_context_invalid_read_result_returns_stable_blocker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    reader_snapshot, scoped, read_results = _dag_snapshots(frames, tmp_path)
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "quant_investor.market.dag.context.config.PIT_UNIVERSE_REQUIRED",
        True,
    )
    read_results["S00"] = "bad"  # type: ignore[assignment]

    context, blockers = _build_production_evaluation_context(
        market="CN",
        universe_key="full_a",
        symbols=list(frames),
        reader_snapshot=reader_snapshot,
        scoped_data_snapshot=scoped,
        read_results=read_results,
    )

    assert context is None
    assert "production_read_result_invalid:S00" in blockers


def test_researchable_frame_subset_excludes_quarantined_symbols() -> None:
    frames = _frames()
    researchable = list(frames)[:-2]

    selected = _researchable_frame_subset(frames, researchable)

    assert list(selected) == researchable
    assert set(selected).isdisjoint(set(frames) - set(researchable))


def test_quant_agent_explicitly_threads_verified_evaluation_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = _frames()
    context = _context(frames, tmp_path)
    captured: list[ProductionEvaluationContext | None] = []

    def fake_score(runtime_frames, *, evaluation_context=None):
        captured.append(evaluation_context)
        return RuntimeFactorScore(
            symbol_scores={symbol: 0.25 for symbol in runtime_frames},
            factor_count=1,
            factors_used=["fixture"],
            factor_weights={"fixture": 1.0},
            factor_coverages={"fixture": 1.0},
            governance_status="ready",
            factor_mode="governed_mined_factors",
            confidence_multiplier=1.0,
            production_eligible=True,
        )

    monkeypatch.setattr(quant_agent_module, "score_with_mined_factors", fake_score)
    monkeypatch.setattr(
        quant_agent_module,
        "production_runtime_score_is_ready",
        lambda _score, **kwargs: (
            kwargs.get("expected_evaluation_context") is context
        ),
    )
    bundle = UnifiedDataBundle(
        market="CN",
        symbols=list(frames),
        symbol_data=frames,
    )

    blocked = QuantAgent().run({"data_bundle": bundle})
    ready = QuantAgent().run(
        {
            "data_bundle": bundle,
            "production_evaluation_context": context,
        }
    )

    assert captured == [None, context]
    assert blocked.final_confidence == 0.0
    assert ready.final_confidence > 0.0
    assert ready.metadata["factor_mode"] == "governed_mined_factors"
