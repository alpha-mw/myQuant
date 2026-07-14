from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_investor.agents.quant_agent as quant_agent_module
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
)
from quant_investor.market.dag.context import (
    _build_production_evaluation_context,
    _researchable_frame_subset,
)
from quant_investor.market.read_result import MarketDataReadResult


AS_OF = "20260106"


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
    if market == "CN":
        pit_manifest = tmp_path / "pit_manifest.json"
        pit_manifest.write_text(
            json.dumps({"source_run_id": "pit-snapshot-20260106"}),
            encoding="utf-8",
        )
        pit_canonical = tmp_path / "pit_canonical.parquet"
        pit_canonical.write_bytes(b"verified-pit-fixture")
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
        open_day_proof_sha256=artifact_hashes["snapshot_manifest"],
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
    snapshot_payload = {
        "snapshot_id": "snapshot-20260106",
        "latest_complete_trade_date": AS_OF,
    }
    pointer_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    gate = {
        "healthy": True,
        "snapshot_id": "snapshot-20260106",
        "latest_complete_trade_date": AS_OF,
        "latest_pointer_path": str(pointer_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }
    reader_snapshot: dict[str, object] = dict(gate)
    scoped: dict[str, object] = {
        "market": market,
        "universe_key": "full_a" if market == "CN" else "sp500",
        "local_latest_trade_date": AS_OF,
        "strict_parquet_gate": dict(gate),
    }
    if market == "CN":
        pit_manifest_path = tmp_path / "pit_manifest.json"
        pit_canonical_path = tmp_path / "pit_canonical.parquet"
        pit_manifest_path.write_text(
            json.dumps({"source_run_id": "pit-snapshot-20260106"}),
            encoding="utf-8",
        )
        pit_canonical_path.write_bytes(b"pit-canonical-fixture")
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
                }
                for symbol in symbols
            },
        }
    read_results = {
        symbol: MarketDataReadResult(
            frame=frames[symbol],
            path=str(tmp_path / f"{symbol}.parquet"),
            symbol=symbol,
            universe_key="full_a" if market == "CN" else "sp500",
            metadata={
                "snapshot_id": "snapshot-20260106",
                "latest_complete_trade_date": AS_OF,
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
