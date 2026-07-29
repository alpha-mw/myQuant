from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.v17_v4_contract import seal_semantic, validate_artifact
from quant_investor.v17_v4_runtime.research_quant import (
    RESEARCH_FACTOR_NAMES,
    RESEARCH_QUANT_BRANCH_VERSION,
    ResearchQuantError,
    STOPPED_FACTOR_NAMES,
    build_research_quant_branch,
    compile_research_quant_branch,
)
from quant_investor.v17_v4_runtime.source_storage import GovernedStore

CUTOFF = "2026-07-28T07:00:00Z"
SESSION = "2026-07-28"
STRATEGY_ID = "quant-first"
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _pool() -> list[str]:
    return [f"{index:06d}.SZ" for index in range(1, 25)]


def _ref(
    *,
    artifact_id: str,
    artifact_version: str,
    byte_sha256: str = "1" * 64,
    relative_path: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha256,
        "cutoff": CUTOFF,
        "relative_path": relative_path,
        "semantic_sha256": byte_sha256,
        "strategy_id": STRATEGY_ID,
    }


def _initial_pool() -> dict[str, object]:
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "cutoff": CUTOFF,
            "ordered_pool": _pool(),
            "origin": SESSION,
            "output_id": "initial-pool-research-quant",
            "preselect_locator_ref": _ref(
                artifact_id="locator-research-quant",
                artifact_version="myquant.v17.v4.preselect-locator.v1",
                relative_path=("data/private/v17_v4_runs/research-quant/" "preselect_locator.json"),
            ),
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.initial-pool-output.v1",
        }
    )


def _market_frame(*, shock_first_symbol: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range(end=SESSION, periods=270)
    symbols = [f"{index:06d}.SZ" for index in range(1, 31)]
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols, start=1):
        steps = np.arange(len(dates), dtype=float)
        returns = (
            0.0004
            + 0.004 * np.sin(steps / (5.0 + symbol_index / 10.0))
            + 0.002 * np.cos(steps / (11.0 + symbol_index / 7.0))
            + symbol_index * 0.000002
        )
        if shock_first_symbol and symbol_index == 1:
            returns[-40:] = np.where(
                np.arange(40) % 3 == 0,
                0.08,
                -0.015,
            )
        prices = 10.0 * np.cumprod(1.0 + returns)
        rows.extend(
            {
                "adj_close": float(price),
                "available_at": CUTOFF,
                "symbol": symbol,
                "trade_date": session.date().isoformat(),
            }
            for session, price in zip(dates, prices, strict=True)
        )
    return pd.DataFrame(rows)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _build(*, shock_first_symbol: bool = False) -> dict[str, object]:
    initial = _initial_pool()
    initial_ref = _ref(
        artifact_id=str(initial["output_id"]),
        artifact_version=str(initial["version"]),
        byte_sha256="2" * 64,
        relative_path=("data/private/v17_v4_runs/research-quant/initial_pool.json"),
    )
    market_raw = _parquet_bytes(_market_frame(shock_first_symbol=shock_first_symbol))
    import hashlib

    market_sha = hashlib.sha256(market_raw).hexdigest()
    return build_research_quant_branch(
        initial_pool=initial,
        initial_pool_ref=initial_ref,
        market_slice_ref=_ref(
            artifact_id="research-quant-market-slice",
            artifact_version=("myquant.v17.v4.dataset.quant-factor-input.v1"),
            byte_sha256=market_sha,
            relative_path=("data/private/v17_v4_sources/research-quant/market.parquet"),
        ),
        market_slice_raw=market_raw,
        output_id="research-quant-branch",
    )


def test_research_quant_branch_uses_exact_incubator_trio_in_scores() -> None:
    branch = _build()
    validate_artifact(branch)
    assert branch["version"] == RESEARCH_QUANT_BRANCH_VERSION
    assert branch["factor_names"] == list(RESEARCH_FACTOR_NAMES)
    assert set(branch["factor_names"]).isdisjoint(STOPPED_FACTOR_NAMES)
    assert branch["shadow_only"] is True
    assert branch["formal_activation_eligible"] is False
    assert branch["canary_evidence_eligible"] is False
    for factor_row, score_row in zip(
        branch["factor_rows"],
        branch["score_rows"],
        strict=True,
    ):
        factor_values = [Decimal(value["value"]) for value in factor_row["factor_values"]]
        assert Decimal(score_row["score"]) == (
            sum(factor_values, Decimal("0")) / Decimal("3")
        ).quantize(Decimal("0.0000000000000001"))


def test_factor_input_change_changes_quant_score_and_cross_sectional_rank() -> None:
    baseline = _build()
    shocked = _build(shock_first_symbol=True)
    baseline_scores = {row["symbol"]: Decimal(row["score"]) for row in baseline["score_rows"]}
    shocked_scores = {row["symbol"]: Decimal(row["score"]) for row in shocked["score_rows"]}
    symbol = _pool()[0]
    assert shocked_scores[symbol] != baseline_scores[symbol]
    baseline_order = sorted(
        baseline_scores,
        key=lambda item: (-baseline_scores[item], item),
    )
    shocked_order = sorted(
        shocked_scores,
        key=lambda item: (-shocked_scores[item], item),
    )
    assert shocked_order != baseline_order


def test_quant_compile_writes_exact_replayable_branch(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    store = GovernedStore(root)
    store.initialize()
    initial = _initial_pool()
    from quant_investor.v17_v4_contract import canonical_resource_bytes
    import hashlib

    initial_raw = canonical_resource_bytes(initial)
    initial_path = "data/private/v17_v4_runs/research-quant/initial_pool.json"
    store.write_exact_once(initial_path, initial_raw)
    market_raw = _parquet_bytes(_market_frame())
    market_path = "data/private/v17_v4_sources/research-quant/market.parquet"
    store.write_exact_once(market_path, market_raw)
    result = compile_research_quant_branch(
        str(root),
        run_id="research-quant",
        output_id="research-quant-branch",
        initial_pool_path=initial_path,
        initial_pool_sha256=hashlib.sha256(initial_raw).hexdigest(),
        market_slice_path=market_path,
        market_slice_sha256=hashlib.sha256(market_raw).hexdigest(),
    )
    assert result["created"] is True
    assert result["factor_names"] == list(RESEARCH_FACTOR_NAMES)
    assert result["formal_activation_eligible"] is False


def test_quant_compile_fails_closed_on_missing_factor_history() -> None:
    frame = _market_frame()
    frame = frame.loc[~(frame["symbol"].eq(_pool()[0]) & frame["trade_date"].lt("2026-03-01"))]
    initial = _initial_pool()
    with pytest.raises(ResearchQuantError, match="factor_unavailable"):
        build_research_quant_branch(
            initial_pool=initial,
            initial_pool_ref=_ref(
                artifact_id=str(initial["output_id"]),
                artifact_version=str(initial["version"]),
                byte_sha256="2" * 64,
                relative_path=("data/private/v17_v4_runs/research-quant/" "initial_pool.json"),
            ),
            market_slice_ref=_ref(
                artifact_id="short-market",
                artifact_version=("myquant.v17.v4.dataset.quant-factor-input.v1"),
                relative_path=("data/private/v17_v4_sources/research-quant/" "short.parquet"),
            ),
            market_slice_raw=_parquet_bytes(frame),
            output_id="research-quant-short",
        )
