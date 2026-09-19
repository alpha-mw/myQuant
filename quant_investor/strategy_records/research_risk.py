"""Pure research risk projection; no I/O or execution authority."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
from typing import Any, Mapping, Sequence


class ResearchRiskError(ValueError):
    """Unusable research input, never an execution fallback."""


def decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ResearchRiskError("NONFINITE_OR_INVALID_NUMBER")
    try:
        result = Decimal(str(value))
    except (ValueError, InvalidOperation) as exc:
        raise ResearchRiskError("NONFINITE_OR_INVALID_NUMBER") from exc
    if not result.is_finite():
        raise ResearchRiskError("NONFINITE_OR_INVALID_NUMBER")
    return result


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    raw = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return {**body, "content_sha256": hashlib.sha256(raw).hexdigest()}


def calculate_position_risk(
    *,
    position: Mapping[str, Any],
    anchor: Mapping[str, Any] | None,
    closes: Sequence[Mapping[str, Any]],
    expected_dates: Sequence[str],
    as_of: str,
    lifecycle_blockers: Sequence[str] = (),
    owner_stop: Any = None,
    holdings_current: bool = False,
    owner_stop_blockers: Sequence[str] = (),
    trailing_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Calculate fixed owner-policy retention thresholds over a closed session set.

    Caller verifies policy/entry/lifecycle refs. Missing inputs block this symbol.
    A valid calculation on an older financial state remains historical research.
    """
    result: dict[str, Any] = {
        "symbol": position["symbol"],
        "name": position.get("name", ""),
        "as_of": as_of,
        "tracking_start_date": None,
        "calculation_state": "UNCONFIRMED",
        "threshold_state": "NON_EXECUTABLE",
        "moving_take_profit_review_price": None,
        "moving_take_profit_reduce_price": None,
        "moving_stop_price": None,
        "peak_price": None,
        "peak_date": None,
        "strict_close": None,
        "profit_giveback_ratio": None,
        "trailing_trigger": "NOT_CONFIGURED",
        "owner_stop_price": None,
        "owner_stop_trigger": "NOT_CONFIGURED",
        "owner_review_state": "NOT_APPLICABLE",
        "actions": [],
        "executable": False,
        "investment_authority": False,
        "blockers": sorted(set([*lifecycle_blockers, *trailing_blockers, *owner_stop_blockers])),
        "trailing_blockers": sorted(set([*lifecycle_blockers, *trailing_blockers])),
        "owner_stop_blockers": sorted(set([*lifecycle_blockers, *owner_stop_blockers])),
    }
    try:
        cost = decimal(position["avg_cost"])
        shares = decimal(position["shares"])
        if cost <= 0 or shares <= 0 or shares != shares.to_integral_value():
            raise ResearchRiskError("INVALID_COST_OR_QUANTITY")
        if not anchor:
            raise ResearchRiskError("ANCHOR_NOT_CONFIGURED")
        start = str(anchor["tracking_start_date"])
        result["tracking_start_date"] = start
        dates = list(expected_dates)
        if not dates or dates != sorted(set(dates)) or dates[-1] != as_of or start > as_of:
            raise ResearchRiskError("CALENDAR_OR_ANCHOR_DATE_INVALID")
        required = [day for day in dates if start <= day <= as_of]
        selected = [row for row in closes if start <= str(row["trade_date"]) <= as_of]
        by_date = {str(row["trade_date"]): row for row in selected}
        if len(by_date) != len(selected):
            raise ResearchRiskError("DUPLICATE_CLOSE_DATE")
        if set(by_date) != set(required):
            raise ResearchRiskError("STRICT_CLOSE_SESSION_GAP")
        prices = {day: decimal(by_date[day]["close"]) for day in required}
        adjustments = {decimal(by_date[day]["adj_factor"]) for day in required}
        if any(value <= 0 for value in prices.values()) or any(x <= 0 for x in adjustments):
            raise ResearchRiskError("NONPOSITIVE_PRICE_OR_ADJUSTMENT")
        if len(adjustments) != 1:
            raise ResearchRiskError("CORPORATE_ACTION_OR_ADJUSTMENT_REVIEW_REQUIRED")
        if result["trailing_blockers"]:
            raise ResearchRiskError("TRAILING_LIFECYCLE_UNCONFIRMED")
        peak = max(prices.values())
        current = prices[as_of]
        result.update(
            peak_price=str(peak),
            peak_date=min(k for k, v in prices.items() if v == peak),
            strict_close=str(current),
            threshold_state=(
                "RESEARCH_ONLY" if holdings_current else "NON_EXECUTABLE_HOLDINGS_STALE"
            ),
        )
        if peak <= cost:
            result["calculation_state"] = "NOT_APPLICABLE_UNTIL_POSITIVE_PROFIT_PEAK"
        else:
            review = cost + Decimal("0.80") * (peak - cost)
            reduce = cost + Decimal("0.65") * (peak - cost)
            giveback = (peak - cost - max(current - cost, Decimal(0))) / (peak - cost)
            result.update(
                calculation_state="CALCULATED",
                profit_giveback_ratio=str(giveback),
                moving_take_profit_review_price=str(
                    review.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
                ),
                moving_take_profit_reduce_price=str(
                    reduce.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
                ),
            )
            result["moving_stop_price"] = result["moving_take_profit_review_price"]
            result["trailing_trigger"] = (
                "REDUCTION_REVIEW"
                if giveback >= Decimal(".35")
                else "REVIEW" if giveback >= Decimal(".20") else "CLEAR"
            )
    except (ResearchRiskError, KeyError) as exc:
        result["blockers"] = sorted(set([*result["blockers"], str(exc)]))
        result["trailing_blockers"] = sorted(set([*result["trailing_blockers"], str(exc)]))
        result["calculation_state"] = "UNCONFIRMED"
        result["threshold_state"] = "NON_EXECUTABLE"
    if owner_stop is not None:
        result["owner_stop_trigger"] = "UNCONFIRMED"
        try:
            stop = decimal(owner_stop)
            if decimal(position["shares"]) <= 0 or decimal(position["avg_cost"]) <= 0:
                raise ResearchRiskError("OWNER_STOP_POSITION_INVALID")
            latest = [row for row in closes if str(row["trade_date"]) == as_of]
            if stop <= 0 or len(latest) != 1 or owner_stop_blockers or lifecycle_blockers:
                raise ResearchRiskError("OWNER_STOP_EVIDENCE_UNCONFIRMED")
            price = decimal(latest[0]["close"])
            if price <= 0:
                raise ResearchRiskError("OWNER_STOP_PRICE_INVALID")
            result["owner_stop_price"] = str(stop)
            result["owner_stop_trigger"] = "BREACH" if price <= stop else "CLEAR"
            if price <= stop:
                result["owner_review_state"] = "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW"
        except (ResearchRiskError, KeyError) as exc:
            result["owner_stop_trigger"] = "UNCONFIRMED"
            result["owner_stop_blockers"] = sorted(set([*result["owner_stop_blockers"], str(exc)]))
            result["blockers"] = sorted(set([*result["blockers"], *result["owner_stop_blockers"]]))
    return seal(result)
