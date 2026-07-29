"""Compile and replay the mandatory research-factor V17 v4 Quant branch."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
import hashlib
from io import BytesIO
import math
from pathlib import PurePosixPath
import re
from typing import Any, Final

import pandas as pd

from quant_investor.factors.governance_literature_incubator_v4 import (
    INCUBATOR_VERSION,
    candidate_catalog_v4,
    evaluate_candidate_v4,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v4_contract.identities import require_sha256
from quant_investor.v17_v4_contract.schema_validation import artifact_identity_field

from .source_storage import ExactReferenceReader, GovernedStore

RESEARCH_QUANT_BRANCH_VERSION: Final = "myquant.v17.v4.research-quant-branch-output.v1"
INITIAL_POOL_VERSION: Final = "myquant.v17.v4.initial-pool-output.v1"
MARKET_SLICE_VERSION: Final = "myquant.v17.v4.dataset.quant-factor-input.v1"
RESEARCH_FACTOR_NAMES: Final = (
    "cn_fip_continuous_direction_12m",
    "cn_low_market_adjusted_tail_asymmetry_252d",
    "cn_low_total_skewness_20d",
)
STOPPED_FACTOR_NAMES: Final = (
    "cn_52_week_high_momentum_12m",
    "cn_low_left_tail_var1_250d",
    "cn_low_max_return_20d",
)
REQUIRED_MARKET_COLUMNS: Final = (
    "adj_close",
    "available_at",
    "symbol",
    "trade_date",
)
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
_PATH_ID_RE: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", re.ASCII)
_CN_SYMBOL_RE: Final = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)
_POLICY = {
    "aggregation": ("arithmetic_mean_of_three_full_cross_sectional_ranks_" "quantized_16dp"),
    "factor_mode": "LITERATURE_INCUBATOR_RESEARCH",
    "factor_names": list(RESEARCH_FACTOR_NAMES),
    "incubator_version": INCUBATOR_VERSION,
    "missing_value_behavior": "FAIL_CLOSED",
    "production_authority": False,
    "stopped_factor_names": list(STOPPED_FACTOR_NAMES),
}
RESEARCH_FACTOR_POLICY_SHA256: Final = hashlib.sha256(canonical_bytes(_POLICY)).hexdigest()
_CATALOG_BY_NAME = {str(row["name"]): row for row in candidate_catalog_v4()}
RESEARCH_FACTOR_DEFINITION_SHA256: Final = hashlib.sha256(
    canonical_bytes([_CATALOG_BY_NAME[name] for name in RESEARCH_FACTOR_NAMES])
).hexdigest()


class ResearchQuantError(RuntimeError):
    """Raised when the research Quant branch cannot be proven."""

    exit_code = 2


def _blocked(reason: str) -> ResearchQuantError:
    return ResearchQuantError(f"V17_V4_RESEARCH_QUANT_BLOCKED:{reason}")


def _path_id(value: str, *, label: str) -> str:
    if type(value) is not str or _PATH_ID_RE.fullmatch(value) is None:
        raise _blocked(f"{label}_path")
    return value


def _artifact_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(document)
    identity_field = artifact_identity_field(str(document["version"]))
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _decimal_text(value: float) -> str:
    if not math.isfinite(value):
        raise _blocked("nonfinite_factor_value")
    return format(value, ".17g")


def _market_frame(raw: bytes, *, cutoff: str, origin: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(BytesIO(raw))
    except Exception as exc:
        raise _blocked("market_slice_parquet") from exc
    if (
        set(frame.columns) != set(REQUIRED_MARKET_COLUMNS)
        or frame.empty
        or frame[["symbol", "trade_date"]].duplicated().any()
    ):
        raise _blocked("market_slice_shape")
    result = frame.loc[:, list(REQUIRED_MARKET_COLUMNS)].copy()
    if (
        not result["symbol"]
        .map(lambda value: type(value) is str and _CN_SYMBOL_RE.fullmatch(value) is not None)
        .all()
    ):
        raise _blocked("market_slice_symbol")
    try:
        sessions = result["trade_date"].map(
            lambda value: date.fromisoformat(value).isoformat() if type(value) is str else None
        )
    except ValueError as exc:
        raise _blocked("market_slice_session") from exc
    if sessions.isna().any() or not sessions.eq(result["trade_date"]).all():
        raise _blocked("market_slice_session")
    availability = pd.to_datetime(
        result["available_at"],
        utc=True,
        errors="coerce",
    )
    cutoff_value = pd.Timestamp(cutoff)
    if (
        availability.isna().any()
        or availability.gt(cutoff_value).any()
        or sessions.gt(origin).any()
    ):
        raise _blocked("market_slice_pit")
    prices = pd.to_numeric(result["adj_close"], errors="coerce")
    if prices.isna().any() or not prices.map(math.isfinite).all() or prices.le(0.0).any():
        raise _blocked("market_slice_price")
    result["adj_close"] = prices.astype(float)
    return result.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _factor_values(
    frame: pd.DataFrame,
    *,
    pool: list[str],
    origin: str,
) -> dict[str, dict[str, float]]:
    wide = frame.pivot(
        index="trade_date",
        columns="symbol",
        values="adj_close",
    ).sort_index()
    wide.index = pd.DatetimeIndex(
        pd.to_datetime(wide.index, format="%Y-%m-%d"),
        name="trade_date",
    )
    origin_timestamp = pd.Timestamp(origin)
    if origin_timestamp not in wide.index or not set(pool).issubset(wide.columns):
        raise _blocked("market_slice_pool_coverage")
    if any(frame.loc[frame["symbol"].eq(symbol), "trade_date"].max() != origin for symbol in pool):
        raise _blocked("market_slice_pool_stale")
    mask = wide.notna()
    by_factor: dict[str, dict[str, float]] = {}
    for factor_name in RESEARCH_FACTOR_NAMES:
        signal = evaluate_candidate_v4(
            name=factor_name,
            inputs={"adj_close": wide},
            pit_mask=mask,
        )
        values = signal.loc[origin_timestamp].reindex(pool)
        if values.isna().any() or not values.map(math.isfinite).all():
            raise _blocked(f"factor_unavailable_{factor_name}")
        by_factor[factor_name] = {symbol: float(values.loc[symbol]) for symbol in pool}
    return by_factor


def build_research_quant_branch(
    *,
    initial_pool: Mapping[str, Any],
    initial_pool_ref: Mapping[str, Any],
    market_slice_ref: Mapping[str, Any],
    market_slice_raw: bytes,
    output_id: str,
) -> dict[str, Any]:
    """Build one shadow-only Quant branch whose scores use the research trio."""

    pool = list(initial_pool["ordered_pool"])
    frame = _market_frame(
        market_slice_raw,
        cutoff=str(initial_pool["cutoff"]),
        origin=str(initial_pool["origin"]),
    )
    values = _factor_values(
        frame,
        pool=pool,
        origin=str(initial_pool["origin"]),
    )
    value_text = {
        symbol: {name: _decimal_text(values[name][symbol]) for name in RESEARCH_FACTOR_NAMES}
        for symbol in pool
    }
    score_text = {
        symbol: format(
            (
                sum(
                    (Decimal(value_text[symbol][name]) for name in RESEARCH_FACTOR_NAMES),
                    Decimal("0"),
                )
                / Decimal(len(RESEARCH_FACTOR_NAMES))
            ).quantize(Decimal("0.0000000000000001")),
            "f",
        )
        for symbol in pool
    }
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "branch_kind": "QUANT",
            "canary_evidence_eligible": False,
            "cutoff": str(initial_pool["cutoff"]),
            "factor_definition_sha256": RESEARCH_FACTOR_DEFINITION_SHA256,
            "factor_mode": "LITERATURE_INCUBATOR_RESEARCH",
            "factor_names": list(RESEARCH_FACTOR_NAMES),
            "factor_policy_sha256": RESEARCH_FACTOR_POLICY_SHA256,
            "factor_rows": [
                {
                    "factor_values": [
                        {
                            "factor_name": name,
                            "value": value_text[symbol][name],
                        }
                        for name in RESEARCH_FACTOR_NAMES
                    ],
                    "symbol": symbol,
                }
                for symbol in pool
            ],
            "formal_activation_eligible": False,
            "incubator_version": INCUBATOR_VERSION,
            "initial_pool_ref": dict(initial_pool_ref),
            "market_slice_ref": dict(market_slice_ref),
            "origin": str(initial_pool["origin"]),
            "output_id": output_id,
            "protocol_version": PROTOCOL_VERSION,
            "score_rows": [
                {
                    "score": score_text[symbol],
                    "symbol": symbol,
                }
                for symbol in pool
            ],
            "shadow_only": True,
            "strategy_id": str(initial_pool["strategy_id"]),
            "version": RESEARCH_QUANT_BRANCH_VERSION,
        }
    )
    validate_artifact(document)
    return document


def compile_research_quant_branch(
    workspace_root: str,
    *,
    run_id: str,
    output_id: str,
    initial_pool_path: str,
    initial_pool_sha256: str,
    market_slice_path: str,
    market_slice_sha256: str,
) -> dict[str, Any]:
    """Compile, replay, and exact-once write one research Quant branch."""

    run = _path_id(run_id, label="run_id")
    reader = ExactReferenceReader(workspace_root)
    initial_raw = reader.read(
        initial_pool_path,
        require_sha256(initial_pool_sha256, label="initial_pool_sha256"),
    )
    validated = load_canonical_artifact(
        initial_raw,
        expected_version=INITIAL_POOL_VERSION,
    )
    initial = load_canonical_resource(initial_raw, label=INITIAL_POOL_VERSION)
    if type(initial) is not dict or validated.payload != initial:
        raise _blocked("initial_pool_readback")
    initial_ref = _artifact_ref(initial, relative_path=initial_pool_path)
    market_sha = require_sha256(
        market_slice_sha256,
        label="market_slice_sha256",
    )
    market_raw = reader.read(market_slice_path, market_sha)
    market_ref = {
        "artifact_id": f"{run}-quant-market-slice",
        "artifact_version": MARKET_SLICE_VERSION,
        "byte_sha256": market_sha,
        "cutoff": str(initial["cutoff"]),
        "relative_path": str(PurePosixPath(market_slice_path)),
        "semantic_sha256": market_sha,
        "strategy_id": str(initial["strategy_id"]),
    }
    document = build_research_quant_branch(
        initial_pool=initial,
        initial_pool_ref=initial_ref,
        market_slice_ref=market_ref,
        market_slice_raw=market_raw,
        output_id=output_id,
    )
    replayed = build_research_quant_branch(
        initial_pool=initial,
        initial_pool_ref=initial_ref,
        market_slice_ref=market_ref,
        market_slice_raw=market_raw,
        output_id=output_id,
    )
    if replayed != document:
        raise _blocked("nondeterministic_replay")
    path = f"data/private/v17_v4_runs/{run}/research_quant_branch.json"
    writer = GovernedStore(workspace_root)
    writer.initialize()
    result = writer.write_exact_once(path, canonical_resource_bytes(document))
    return {
        "branch_ref": _artifact_ref(document, relative_path=path),
        "created": result.created,
        "factor_names": list(RESEARCH_FACTOR_NAMES),
        "formal_activation_eligible": False,
        "shadow_only": True,
    }


def revalidate_research_quant_branch(
    branch: Mapping[str, Any],
    *,
    initial_pool: Mapping[str, Any],
    initial_pool_ref: Mapping[str, Any],
    reader: ExactReferenceReader,
) -> None:
    """Recompute one research branch from its exact market-slice bytes."""

    raw = reader.read(
        str(branch["market_slice_ref"]["relative_path"]),
        str(branch["market_slice_ref"]["byte_sha256"]),
    )
    replayed = build_research_quant_branch(
        initial_pool=initial_pool,
        initial_pool_ref=initial_pool_ref,
        market_slice_ref=branch["market_slice_ref"],
        market_slice_raw=raw,
        output_id=str(branch["output_id"]),
    )
    if replayed != dict(branch):
        raise _blocked("branch_replay_mismatch")


__all__ = [
    "RESEARCH_FACTOR_DEFINITION_SHA256",
    "RESEARCH_FACTOR_NAMES",
    "RESEARCH_FACTOR_POLICY_SHA256",
    "RESEARCH_QUANT_BRANCH_VERSION",
    "ResearchQuantError",
    "STOPPED_FACTOR_NAMES",
    "build_research_quant_branch",
    "compile_research_quant_branch",
    "revalidate_research_quant_branch",
]
