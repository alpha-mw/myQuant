"""Deterministic Morning threshold observations; all formula ownership stays native."""

import hashlib
import re
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes
from quant_investor.strategy_records.corporate_contracts import instant
from quant_investor.strategy_records.research_risk import calculate_position_risk, decimal, seal
from quant_investor.operations.daily_contract import ContractError, validate_ref, utc_stamp
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, _false_authority, _validate_day
from .corporate_reconciliation import BLOCKERS as CORPORATE_BLOCKERS

SCHEMA = "morning-owner-threshold-review.v1"
FIELDS = frozenset(
    (
        "schema_version run_date previous_trade_date "
        "previous_completion_ref eod_validated_at synthetic evidence_mode "
        "quote_capture_ref quote_raw_ref quote_requested_at "
        "decision_result_ref threshold_policy_refs policy_times "
        "source_refs rows quote_only_symbols summary_state authority "
        "content_sha256"
    ).split()
)
ROW_FIELDS = frozenset(
    (
        "symbol name position_ref policy_binding eod_risk "
        "quote_observation executable investment_authority actions"
    ).split()
)
RISK_FIELDS = frozenset(
    (
        "symbol name as_of tracking_start_date calculation_state "
        "threshold_state moving_take_profit_review_price "
        "moving_take_profit_reduce_price moving_stop_price peak_price "
        "peak_date strict_close profit_giveback_ratio trailing_trigger "
        "owner_stop_price owner_stop_trigger owner_review_state actions "
        "executable investment_authority blockers trailing_blockers "
        "owner_stop_blockers content_sha256"
    ).split()
)
AMOUNTS = (
    "moving_take_profit_review_price moving_take_profit_reduce_price "
    "moving_stop_price peak_price strict_close profit_giveback_ratio "
    "owner_stop_price"
).split()
CODES = CORPORATE_BLOCKERS | frozenset(
    (
        "TRAILING_POLICY_NOT_RECONCILED_AT_PRIOR_EOD "
        "POLICY_BASELINE_NOT_ANCESTOR POSITION_LIFECYCLE_CHANGED "
        "POSITION_COST_OR_QUANTITY_CHANGED "
        "NEW_EVENT_REQUIRES_ANCHOR_REVIEW EXACT_ENTRY_REF_MISMATCH "
        "ENTRY_REFERENCE_UNCONFIRMED OWNER_STOP_HISTORY_GAP "
        "OWNER_STOP_CORPORATE_ACTION_REVIEW OWNER_STOP_POSITION_CHANGED "
        "STRICT_CLOSE_UNAVAILABLE OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD "
        "NONFINITE_OR_INVALID_NUMBER INVALID_COST_OR_QUANTITY "
        "ANCHOR_NOT_CONFIGURED CALENDAR_OR_ANCHOR_DATE_INVALID "
        "DUPLICATE_CLOSE_DATE STRICT_CLOSE_SESSION_GAP "
        "NONPOSITIVE_PRICE_OR_ADJUSTMENT "
        "CORPORATE_ACTION_OR_ADJUSTMENT_REVIEW_REQUIRED "
        "TRAILING_LIFECYCLE_UNCONFIRMED OWNER_STOP_POSITION_INVALID "
        "OWNER_STOP_EVIDENCE_UNCONFIRMED OWNER_STOP_PRICE_INVALID "
        "OWNER_POLICY_REVALIDATION_REQUIRED"
    ).split()
)
COMPARISONS = {
    "CLEAR",
    "WARNING_NOT_BREACH",
    "NOT_APPLICABLE",
    "NOT_CONFIGURED",
    "UNCONFIRMED",
    "QUOTE_UNAVAILABLE",
}


def review_hash(value):
    return hashlib.sha256(
        canonical_json_bytes({k: v for k, v in value.items() if k != "content_sha256"})
    ).hexdigest()


def _refs(values):
    refs = {(validate_ref(r)["path"], r["sha256"]) for r in values}
    if len({p for p, _ in refs}) != len(refs):
        raise ContractError("MORNING_THRESHOLD_SOURCE_REF_CONFLICT")
    return [{"path": p, "sha256": sha} for p, sha in sorted(refs)]


def _shape(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ContractError("MORNING_THRESHOLD_REVIEW_SHAPE_INVALID")


def _amount(value, *, nullable=True):
    if value is None and nullable:
        return None
    if type(value) is not str or not re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[Ee][+-]?[0-9]+)?", value
    ):
        raise ContractError("MORNING_THRESHOLD_DECIMAL_TEXT_INVALID")
    return decimal(value)


def _codes(values):
    if (
        type(values) is not list
        or any(type(v) is not str for v in values)
        or values != sorted(set(values))
    ):
        raise ContractError("MORNING_THRESHOLD_BLOCKERS_INVALID")
    for value in values:
        if value in CODES:
            continue
        if value.startswith("LIFECYCLE_UNCONFIRMED:"):
            _validate_day(value.split(":", 1)[1])
        else:
            raise ContractError("MORNING_THRESHOLD_BLOCKER_UNKNOWN")


def _compare(price, threshold, unavailable):
    if threshold is None:
        return unavailable
    if price is None:
        return "QUOTE_UNAVAILABLE"
    return "WARNING_NOT_BREACH" if price <= decimal(threshold) else "CLEAR"


def _row(item, *, day, position_ref, policy_refs, quote):
    risk = calculate_position_risk(
        position=item["position"],
        anchor=item["anchor"],
        closes=item["closes"],
        expected_dates=item["expected_dates"],
        as_of=day,
        owner_stop=item["owner_stop"],
        holdings_current=True,
        trailing_blockers=item["trailing_blockers"],
        owner_stop_blockers=item["owner_stop_blockers"],
    )
    trailing = (
        "NOT_CONFIGURED"
        if item["anchor"] is None
        else (
            "NOT_RECONCILED_AT_PRIOR_EOD"
            if not item["same_trailing_policy"]
            else (
                "UNCONFIRMED"
                if risk["trailing_blockers"] or risk["calculation_state"] == "UNCONFIRMED"
                else "BOUND"
            )
        )
    )
    stop = (
        "NOT_CONFIGURED"
        if item["stop"] is None
        else (
            "BOUND"
            if risk["owner_stop_trigger"] in {"CLEAR", "BREACH"} and not risk["owner_stop_blockers"]
            else "UNCONFIRMED"
        )
    )
    try:
        price = decimal(quote.get("price"))
        if price <= 0:
            price = None
    except ValueError:
        price = None
    quote_trailing = risk["moving_take_profit_review_price"] if trailing == "BOUND" else None
    quote_reduce = risk["moving_take_profit_reduce_price"] if trailing == "BOUND" else None
    missing_trailing = (
        "NOT_CONFIGURED"
        if trailing == "NOT_CONFIGURED"
        else "NOT_APPLICABLE" if trailing == "BOUND" else "UNCONFIRMED"
    )
    return {
        "symbol": risk["symbol"],
        "name": risk["name"],
        "position_ref": position_ref,
        "policy_binding": {
            "trailing": {
                "state": trailing,
                "policy_ref": policy_refs["trailing"],
                "blocker_codes": risk["trailing_blockers"],
            },
            "initial_stop": {
                "state": stop,
                "policy_ref": policy_refs["initial_stop"],
                "blocker_codes": risk["owner_stop_blockers"],
                "configured_price": (
                    None if item["stop"] is None else str(item["stop"]["initial_stop_price_cny"])
                ),
            },
        },
        "eod_risk": risk,
        "quote_observation": {
            "price": None if price is None else str(price),
            "state": "UNAVAILABLE" if price is None else "AVAILABLE",
            "trailing_review_comparison": _compare(price, quote_trailing, missing_trailing),
            "trailing_reduce_comparison": _compare(price, quote_reduce, missing_trailing),
            "initial_stop_comparison": _compare(
                price,
                risk["owner_stop_price"] if stop == "BOUND" else None,
                "NOT_CONFIGURED" if stop == "NOT_CONFIGURED" else "UNCONFIRMED",
            ),
        },
        "executable": False,
        "investment_authority": False,
        "actions": [],
    }


def _summary(rows):
    complete = all(
        row["policy_binding"]["trailing"]["state"] == "BOUND"
        and row["policy_binding"]["initial_stop"]["state"] == "BOUND"
        and not row["eod_risk"]["blockers"]
        and row["quote_observation"]["state"] == "AVAILABLE"
        for row in rows
    )
    return "COMPLETE_RESEARCH_REVIEW" if complete else "PARTIAL_RESEARCH_REVIEW"


def build_threshold_review(
    *,
    sources,
    quote,
    quote_capture_ref,
    quote_raw_ref,
    decision_result_ref,
    evidence_mode,
    synthetic=None,
):
    quote_rows = {r["symbol"]: r for r in quote["quote_rows"]}
    run_date = (
        utc_stamp(quote["request_time"]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    )
    rows = [
        _row(
            row,
            day=sources.day,
            position_ref=sources.store_outputs["ledger"],
            policy_refs=sources.policy_refs,
            quote=quote_rows[row["position"]["symbol"]],
        )
        for row in sources.rows
    ]
    value = {
        "schema_version": SCHEMA,
        "run_date": run_date,
        "previous_trade_date": sources.day,
        "previous_completion_ref": sources.completion_ref,
        "eod_validated_at": sources.recorded["native_validation_completed_at"],
        "synthetic": sources.recorded["synthetic"] if synthetic is None else synthetic,
        "evidence_mode": evidence_mode,
        "quote_capture_ref": quote_capture_ref,
        "quote_raw_ref": quote_raw_ref,
        "quote_requested_at": quote["request_time"],
        "decision_result_ref": decision_result_ref,
        "threshold_policy_refs": sources.policy_refs,
        "policy_times": {
            k: {
                "effective_from": v["effective_from"],
                "owner_confirmation_recorded_at": v.get("owner_confirmation_recorded_at"),
            }
            for k, v in sources.policies.items()
        },
        "source_refs": _refs(
            [
                *sources.source_refs(),
                sources.completion_ref,
                quote_capture_ref,
                quote_raw_ref,
                decision_result_ref,
            ]
        ),
        "rows": rows,
        "quote_only_symbols": sorted(set(quote_rows) - {row["symbol"] for row in rows}),
        "summary_state": _summary(rows),
        "authority": FALSE_AUTHORITY,
    }
    value["content_sha256"] = review_hash(value)
    validate_threshold_review(value)
    return value


def _validate_risk(risk):
    _shape(risk, RISK_FIELDS)
    if (
        seal(risk) != risk
        or risk["executable"] is not False
        or risk["investment_authority"] is not False
        or risk["actions"] != []
    ):
        raise ContractError("MORNING_THRESHOLD_NATIVE_RISK_INVALID")
    enums = {
        "calculation_state": {
            "UNCONFIRMED",
            "CALCULATED",
            "NOT_APPLICABLE_UNTIL_POSITIVE_PROFIT_PEAK",
        },
        "threshold_state": {"NON_EXECUTABLE", "RESEARCH_ONLY", "NON_EXECUTABLE_HOLDINGS_STALE"},
        "trailing_trigger": {"NOT_CONFIGURED", "CLEAR", "REVIEW", "REDUCTION_REVIEW"},
        "owner_stop_trigger": {"NOT_CONFIGURED", "UNCONFIRMED", "CLEAR", "BREACH"},
        "owner_review_state": {"NOT_APPLICABLE", "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW"},
    }
    if any(risk[k] not in allowed for k, allowed in enums.items()):
        raise ContractError("MORNING_THRESHOLD_NATIVE_STATE_INVALID")
    for key in AMOUNTS:
        _amount(risk[key])
    for key in ("as_of", "tracking_start_date", "peak_date"):
        if risk[key] is not None:
            _validate_day(risk[key])
    for key in ("blockers", "trailing_blockers", "owner_stop_blockers"):
        _codes(risk[key])
    _validate_risk_values(risk)


def _validate_risk_values(risk):
    levels = [
        risk[k]
        for k in (
            "moving_take_profit_review_price",
            "moving_take_profit_reduce_price",
            "moving_stop_price",
        )
    ]
    state = risk["calculation_state"]
    if state == "CALCULATED" and (
        any(level is None for level in levels)
        or risk["profit_giveback_ratio"] is None
        or risk["moving_stop_price"] != risk["moving_take_profit_review_price"]
    ):
        raise ContractError("MORNING_THRESHOLD_CALCULATED_LEVEL_MISSING")
    if state == "NOT_APPLICABLE_UNTIL_POSITIVE_PROFIT_PEAK" and (
        any(level is not None for level in levels) or risk["profit_giveback_ratio"] is not None
    ):
        raise ContractError("MORNING_THRESHOLD_INAPPLICABLE_LEVEL_PRESENT")
    if state != "UNCONFIRMED" and (
        risk["peak_price"] is None or risk["peak_date"] is None or risk["strict_close"] is None
    ):
        raise ContractError("MORNING_THRESHOLD_WINDOW_VALUE_MISSING")
    if risk["owner_stop_trigger"] in {"CLEAR", "BREACH"}:
        if risk["owner_stop_price"] is None or _amount(risk["owner_stop_price"]) <= 0:
            raise ContractError("MORNING_THRESHOLD_CONFIRMED_STOP_MISSING")
        expected = (
            "OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW"
            if risk["owner_stop_trigger"] == "BREACH"
            else "NOT_APPLICABLE"
        )
        if risk["owner_review_state"] != expected:
            raise ContractError("MORNING_THRESHOLD_OWNER_REVIEW_STATE_INVALID")


def _validate_row(row, value):
    _shape(row, ROW_FIELDS)
    if (
        type(row["symbol"]) is not str
        or not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", row["symbol"])
        or type(row["name"]) is not str
    ):
        raise ContractError("MORNING_THRESHOLD_ROW_IDENTITY_INVALID")
    if (
        row["executable"] is not False
        or row["investment_authority"] is not False
        or row["actions"] != []
    ):
        raise ContractError("MORNING_THRESHOLD_AUTHORITY_INVALID")
    validate_ref(row["position_ref"])
    _validate_risk(row["eod_risk"])
    if (
        row["eod_risk"]["symbol"] != row["symbol"]
        or row["eod_risk"]["name"] != row["name"]
        or row["eod_risk"]["as_of"] != value["previous_trade_date"]
    ):
        raise ContractError("MORNING_THRESHOLD_RISK_IDENTITY_DIFFERS")
    _validate_binding(row, value)
    _validate_observation(row)


def _validate_binding(row, value):
    _shape(row["policy_binding"], {"trailing", "initial_stop"})
    for lane, binding in row["policy_binding"].items():
        _shape(
            binding,
            {"state", "policy_ref", "blocker_codes"}
            | ({"configured_price"} if lane == "initial_stop" else set()),
        )
        allowed = {"BOUND", "NOT_CONFIGURED", "UNCONFIRMED"} | (
            {"NOT_RECONCILED_AT_PRIOR_EOD"} if lane == "trailing" else set()
        )
        if (
            binding["state"] not in allowed
            or binding["policy_ref"] != value["threshold_policy_refs"][lane]
        ):
            raise ContractError("MORNING_THRESHOLD_POLICY_BINDING_INVALID")
        _codes(binding["blocker_codes"])
        if lane == "initial_stop":
            price = _amount(binding["configured_price"])
            if price is not None and price <= 0:
                raise ContractError("MORNING_THRESHOLD_CONFIGURED_STOP_INVALID")
    risk = row["eod_risk"]
    trailing, stop = row["policy_binding"]["trailing"], row["policy_binding"]["initial_stop"]
    if (
        trailing["blocker_codes"] != risk["trailing_blockers"]
        or stop["blocker_codes"] != risk["owner_stop_blockers"]
    ):
        raise ContractError("MORNING_THRESHOLD_BRANCH_BLOCKERS_DIFFER")
    if trailing["state"] == "BOUND" and (
        risk["trailing_blockers"]
        or risk["calculation_state"] == "UNCONFIRMED"
        or risk["threshold_state"] != "RESEARCH_ONLY"
    ):
        raise ContractError("MORNING_THRESHOLD_UNUSABLE_TRAILING_BOUND")
    if stop["state"] == "BOUND" and (
        risk["owner_stop_blockers"]
        or risk["owner_stop_trigger"] not in {"CLEAR", "BREACH"}
        or risk["owner_stop_price"] is None
        or _amount(stop["configured_price"]) != _amount(risk["owner_stop_price"])
    ):
        raise ContractError("MORNING_THRESHOLD_UNUSABLE_STOP_BOUND")
    if (
        trailing["state"] == "NOT_RECONCILED_AT_PRIOR_EOD"
        and "TRAILING_POLICY_NOT_RECONCILED_AT_PRIOR_EOD" not in trailing["blocker_codes"]
    ):
        raise ContractError("MORNING_THRESHOLD_POLICY_MISMATCH_UNDISCLOSED")


def _validate_observation(row):
    risk = row["eod_risk"]
    trailing, stop = row["policy_binding"]["trailing"], row["policy_binding"]["initial_stop"]
    observation = row["quote_observation"]
    _shape(
        observation,
        {
            "price",
            "state",
            "trailing_review_comparison",
            "trailing_reduce_comparison",
            "initial_stop_comparison",
        },
    )
    price = _amount(observation["price"])
    if observation["state"] != ("UNAVAILABLE" if price is None else "AVAILABLE") or (
        price is not None and price <= 0
    ):
        raise ContractError("MORNING_THRESHOLD_QUOTE_STATE_INVALID")
    if any(
        observation[k] not in COMPARISONS
        for k in (
            "trailing_review_comparison",
            "trailing_reduce_comparison",
            "initial_stop_comparison",
        )
    ):
        raise ContractError("MORNING_THRESHOLD_COMPARISON_INVALID")
    missing = (
        "NOT_CONFIGURED"
        if trailing["state"] == "NOT_CONFIGURED"
        else "NOT_APPLICABLE" if trailing["state"] == "BOUND" else "UNCONFIRMED"
    )
    expected = {
        "trailing_review_comparison": _compare(
            price,
            risk["moving_take_profit_review_price"] if trailing["state"] == "BOUND" else None,
            missing,
        ),
        "trailing_reduce_comparison": _compare(
            price,
            risk["moving_take_profit_reduce_price"] if trailing["state"] == "BOUND" else None,
            missing,
        ),
        "initial_stop_comparison": _compare(
            price,
            risk["owner_stop_price"] if stop["state"] == "BOUND" else None,
            "NOT_CONFIGURED" if stop["state"] == "NOT_CONFIGURED" else "UNCONFIRMED",
        ),
    }
    if any(observation[k] != v for k, v in expected.items()):
        raise ContractError("MORNING_THRESHOLD_COMPARISON_DIFFERS")


def _validate_review_identity(value):
    _shape(value, FIELDS)
    if (
        value["schema_version"] != SCHEMA
        or value["content_sha256"] != review_hash(value)
        or not _false_authority(value["authority"])
        or type(value["synthetic"]) is not bool
    ):
        raise ContractError("MORNING_THRESHOLD_REVIEW_INVALID")
    for key in ("run_date", "previous_trade_date"):
        _validate_day(value[key])
    if value["previous_trade_date"] >= value["run_date"]:
        raise ContractError("MORNING_THRESHOLD_REVIEW_DATE_INVALID")
    for key in (
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "decision_result_ref",
    ):
        validate_ref(value[key])
    if value["previous_completion_ref"]["path"] != (
        "results/operations/daily_production/CN/"
        + value["previous_trade_date"]
        + "/completion.v1.json"
    ):
        raise ContractError("MORNING_THRESHOLD_COMPLETION_PATH_INVALID")
    quote_at = utc_stamp(value["quote_requested_at"])
    eod_at = utc_stamp(value["eod_validated_at"])
    if quote_at.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != value["run_date"]:
        raise ContractError("MORNING_THRESHOLD_QUOTE_DATE_INVALID")
    if value["evidence_mode"] not in {"PRIOR_EOD_BEFORE_QUOTE", "REPLAY_ONLY"} or (
        value["evidence_mode"] == "PRIOR_EOD_BEFORE_QUOTE"
        and (value["synthetic"] or eod_at > quote_at)
    ):
        raise ContractError("MORNING_THRESHOLD_EVIDENCE_TIMING_INVALID")


def _validate_policy_times(value):
    quote_at = utc_stamp(value["quote_requested_at"])
    _shape(value["threshold_policy_refs"], {"trailing", "initial_stop"})
    _shape(value["policy_times"], {"trailing", "initial_stop"})
    for lane, ref in value["threshold_policy_refs"].items():
        validate_ref(ref)
        times = value["policy_times"][lane]
        _shape(times, {"effective_from", "owner_confirmation_recorded_at"})
        if instant(times["effective_from"]) > quote_at:
            raise ContractError("MORNING_THRESHOLD_POLICY_AFTER_QUOTE")
        if lane == "trailing" and times["owner_confirmation_recorded_at"] is not None:
            raise ContractError("MORNING_THRESHOLD_INVENTED_POLICY_CLOCK")
        if lane == "initial_stop" and instant(times["owner_confirmation_recorded_at"]) > quote_at:
            raise ContractError("MORNING_THRESHOLD_POLICY_AFTER_QUOTE")


def _validate_review_sources(value):
    if type(value["source_refs"]) is not list or value["source_refs"] != _refs(
        value["source_refs"]
    ):
        raise ContractError("MORNING_THRESHOLD_SOURCE_REFS_INVALID")
    symbols = [r["symbol"] for r in value["rows"]]
    extras = value["quote_only_symbols"]
    if (
        symbols != sorted(set(symbols))
        or type(extras) is not list
        or extras != sorted(set(extras))
        or set(extras) & set(symbols)
    ):
        raise ContractError("MORNING_THRESHOLD_SYMBOL_SET_INVALID")
    if any(type(s) is not str or not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", s) for s in extras):
        raise ContractError("MORNING_THRESHOLD_QUOTE_ONLY_SYMBOL_INVALID")
    required = [
        value[k]
        for k in (
            "previous_completion_ref",
            "quote_capture_ref",
            "quote_raw_ref",
            "decision_result_ref",
        )
    ]
    required.extend(value["threshold_policy_refs"].values())
    required.extend(row["position_ref"] for row in value["rows"])
    if any(ref not in value["source_refs"] for ref in required):
        raise ContractError("MORNING_THRESHOLD_REQUIRED_SOURCE_REF_MISSING")


def validate_threshold_review(value):
    _validate_review_identity(value)
    _validate_policy_times(value)
    if type(value["rows"]) is not list or not value["rows"]:
        raise ContractError("MORNING_THRESHOLD_HOLDINGS_EMPTY")
    for row in value["rows"]:
        _validate_row(row, value)
    _validate_review_sources(value)
    if value["summary_state"] != _summary(value["rows"]):
        raise ContractError("MORNING_THRESHOLD_REVIEW_SUMMARY_INVALID")
    return value
