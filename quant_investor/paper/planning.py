"""Read-only planning inputs for the Paper account's own positions.

After the first fill the Paper account and the manual strategy ledger hold
different things, so the decision inputs must come from the account itself:

- positions, shares, settled lots and cost: the registered Paper account
- thresholds: the sealed trailing-anchor policy applied by
  `strategy_records.research_risk.calculate_position_risk`, with strict closes
  read from the published CN snapshot
- owner stops: the sealed owner-stop policy

Nothing here writes; the writer remains the only mutation surface.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Any, Mapping

TRAILING_POLICY_RELATIVE = (
    "results/policies/risk/aggressive_tech_manufacturing/trailing-anchor.v1/"
    "owner-trailing-anchor-policy-20260901-v1.json"
)
STOP_POLICY_RELATIVE = (
    "results/policies/risk/aggressive_tech_manufacturing/initial-risk-stop.v1/"
    "owner-stop-policy-20260828-v1.json"
)
CALENDAR_ROOT = "data/parquet/cn/macro_release_calendar"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def trailing_anchors(workspace: Path) -> dict[str, Mapping[str, Any]]:
    policy = _read_json(workspace / TRAILING_POLICY_RELATIVE)
    return {row["symbol"]: row for row in policy.get("anchors", [])}


def owner_stops(workspace: Path) -> dict[str, Any]:
    """Confirmed owner stops plus the policy's own effective date."""

    policy = _read_json(workspace / STOP_POLICY_RELATIVE)
    return {
        "effective_from": str(policy["effective_from"])[:10].replace("-", ""),
        "stops": {
            row["symbol"]: row
            for row in policy.get("stops", [])
            if row.get("initial_stop_state") == "CONFIRMED"
        },
    }


def session_dates(workspace: Path, *, as_of: str) -> list[str]:
    """Strict session calendar up to and including `as_of`."""

    import hashlib

    from quant_investor.macro.release_calendar import load_release_calendar

    root = Path(CALENDAR_ROOT)
    raw = (workspace / root / "_latest.json").read_bytes()
    calendar = load_release_calendar(
        canonical_root=workspace / root,
        expected_pointer_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return [
        str(day).replace("-", "")
        for day in calendar.open_dates
        if str(day).replace("-", "") <= as_of
    ]


def technology_candidates(
    *, workspace: Path, trade_date: str, universe_sha256: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank the strategy's technology universe by the sealed factor formula.

    The sealed research pool ranks the whole market, so its members are not a
    strategy universe. This ranks only the symbols captured in
    `paper-technology-universe.v1.json` (the policy's DC technology themes) using
    the same cross-sectional percentile algorithm and the same policy weights the
    sealed pool uses, then returns them best-first.
    """

    import glob
    import hashlib

    from quant_investor.intelligence._common import decimal_text
    from quant_investor.intelligence.daily import _signal_percentiles

    evidence_path = (
        workspace / "data/private/paper_evidence" / trade_date / "paper-technology-universe.v1.json"
    )
    evidence_raw = evidence_path.read_bytes()
    if universe_sha256 is not None:
        if hashlib.sha256(evidence_raw).hexdigest() != universe_sha256:
            raise SystemExit("technology universe evidence SHA differs")
    universe = json.loads(evidence_raw)
    if universe.get("trade_date") != trade_date:
        raise SystemExit("technology universe trade date differs")

    generations = sorted(
        path
        for path in glob.glob(
            str(workspace / "results/factors/objects/factor.production_generation/*.json")
        )
    )
    signal_values = None
    generation_ref = None
    for path in reversed(generations):
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        payload = value.get("payload", {})
        if payload.get("as_of") == trade_date and "signal_values" in payload:
            signal_values = payload["signal_values"]
            generation_ref = {
                "path": str(Path(path).relative_to(workspace)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            break
    if signal_values is None or generation_ref is None:
        raise SystemExit(f"no factor generation for {trade_date}")

    policy = json.loads(
        (workspace / "results/policies/research/aggressive_tech_manufacturing/v2.json").read_text()
    )["payload"]
    aliases = {"LOW": "pv_low_dollar_volume_5d", "W80": "pv_blend_volstab19x2_mom90_amihud5_w80"}
    weights = {row["factor_alias"]: Decimal(row["weight"]) for row in policy["factor_rows"]}
    # The sealed rank rounds every per-factor percentile through decimal_text
    # before the weighted sum; skipping that step changes the 12th decimal.
    percentiles = {
        alias: {
            symbol: Decimal(decimal_text(value))
            for symbol, value in _signal_percentiles(
                {
                    symbol: Decimal.from_float(float.fromhex(raw))
                    for symbol, raw in signal_values[factor_id].items()
                }
            ).items()
        }
        for alias, factor_id in aliases.items()
    }
    members = universe["symbols"]
    rows = []
    for symbol in members:
        if symbol not in percentiles["LOW"] or symbol not in percentiles["W80"]:
            continue
        combined = sum(
            (percentiles[alias][symbol] * weights[alias] for alias in weights), Decimal("0")
        )
        rows.append(
            {
                "symbol": symbol,
                "combined_percentile": decimal_text(combined),
                "technology_theme_ids": [f"TUSHARE_DC:{code}" for code in members[symbol]],
            }
        )
    rows.sort(key=lambda row: (-Decimal(row["combined_percentile"]), row["symbol"]))
    return rows, {
        "universe_ref": {
            "path": str(evidence_path.relative_to(workspace)),
            "sha256": hashlib.sha256(evidence_raw).hexdigest(),
        },
        "generation_ref": generation_ref,
    }


def position_views(
    *,
    workspace: Path,
    account: Mapping[str, Any],
    as_of: str,
    dates: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One calculator row per Paper position, in the rules engine's input shape."""

    from quant_investor.market.market_data_reader import MarketDataReader
    from quant_investor.strategy_records.research_risk import (
        ResearchRiskError,
        calculate_position_risk,
    )

    anchors = trailing_anchors(workspace)
    stop_policy = owner_stops(workspace)
    stops = stop_policy["stops"]
    expected_dates = dates if dates is not None else session_dates(workspace, as_of=as_of)
    reader = MarketDataReader(data_root=workspace / "data", mode_policy="strict")
    cash = Decimal(str(account["state"]["cash"]))
    views: list[dict[str, Any]] = []
    market_value = Decimal("0")
    rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    for position in account["ledger"]:
        if int(position["shares"]) <= 0:
            # A fully exited position stays in the ledger as history.
            continue
        symbol = position["symbol"]
        anchor = anchors.get(symbol)
        stop_row = stops.get(symbol)
        start = str(anchor["tracking_start_date"]) if anchor else as_of
        if stop_row:
            start = min(start, stop_policy["effective_from"])
        read = reader.read_symbol_frame(symbol, start_date=start, end_date=as_of)
        if read.issues or read.frame.empty:
            raise SystemExit(f"{symbol} strict closes are unavailable: {read.issues}")
        stop_value = stop_row["initial_stop_price_cny"] if stop_row else None
        try:
            row = calculate_position_risk(
                position=position,
                anchor=anchor,
                closes=read.frame.to_dict("records"),
                expected_dates=expected_dates,
                as_of=as_of,
                lifecycle_blockers=[],
                owner_stop=stop_value,
                holdings_current=True,
                owner_stop_blockers=[],
                trailing_blockers=[],
            )
        except (ResearchRiskError, KeyError) as exc:  # pragma: no cover - defensive
            raise SystemExit(f"{symbol} risk calculation failed: {exc}") from exc
        # Valuation never depends on whether the threshold lane could be
        # calculated: take the session's own strict close for every position.
        session_rows = read.frame[read.frame["trade_date"] == as_of]
        close = Decimal(str(session_rows.iloc[0]["close"])) if len(session_rows) == 1 else None
        if close is not None:
            market_value += close * Decimal(int(position["shares"]))
        rows.append((position, row))
        views.append({"position": position, "row": row, "close": close})

    nav = cash + market_value
    for view in views:
        position, row = view["position"], view["row"]
        close = view["close"]
        symbol = position["symbol"]
        stop_row = stops.get(symbol)
        fraction = (
            (close * Decimal(int(position["shares"])) / nav).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )
            if close is not None and nav > 0
            else Decimal("0")
        )
        view.update(
            {
                "symbol": symbol,
                "name": position.get("name"),
                "shares": int(position["shares"]),
                "settled_shares": int(position["settled_shares"]),
                "avg_cost": f"{Decimal(str(position['avg_cost'])):.6f}",
                "close": None if close is None else f"{close:.2f}",
                "hard_stop": row.get("owner_stop_price")
                or (stop_row["initial_stop_price_cny"] if stop_row else None),
                "hard_stop_source": (
                    "owner-stop-policy-20260828-v1"
                    if stop_row
                    else "risk-calculator:owner_stop_price"
                ),
                "giveback_ratio": (
                    str(row["profit_giveback_ratio"])
                    if row.get("profit_giveback_ratio") is not None
                    else None
                ),
                "peak_price": None if row.get("peak_price") is None else str(row["peak_price"]),
                "review_price": row.get("moving_take_profit_review_price"),
                "reduce_price": row.get("moving_take_profit_reduce_price"),
                "deterioration_evidence": [],
                "nav_weight": float(fraction),
                "current_value": (
                    float(close * Decimal(int(position["shares"]))) if close is not None else 0.0
                ),
                "thesis_status": position.get("thesis_status"),
                "calculation_state": row.get("calculation_state"),
                "trailing_trigger": row.get("trailing_trigger"),
                "owner_stop_trigger": row.get("owner_stop_trigger"),
                "blockers": sorted(
                    {
                        *row.get("blockers", []),
                        *row.get("owner_stop_blockers", []),
                        *row.get("trailing_blockers", []),
                    }
                ),
            }
        )
    return views


__all__ = [
    "CALENDAR_ROOT",
    "technology_candidates",
    "STOP_POLICY_RELATIVE",
    "TRAILING_POLICY_RELATIVE",
    "owner_stops",
    "position_views",
    "session_dates",
    "trailing_anchors",
]
