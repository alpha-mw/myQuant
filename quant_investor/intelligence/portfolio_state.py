"""Source-bound pre-close book context; never changes native investment decisions."""

from decimal import Decimal
from quant_investor.contracts import validate_artifact

from quant_investor.strategy_records.accounting import _money, _unit
from quant_investor.operations.daily_contract import validate_ref
from ._common import (
    IntelligenceError,
    build_artifact,
    business_identity,
    company_code,
    timestamp,
)
from .fundamental_time import availability_instant, session_date, SHANGHAI
from .pcb_ai_hardware import physical_refs

PORTFOLIO_STATE_KIND = "research_portfolio_state"
STRATEGY_ID = "aggressive_tech_manufacturing"
REPORT_POLICY = "native-five-state-source-completeness.v1"


def portfolio_timing(*, as_of, effective_date, sealed_at, published_at, created_at):
    cutoff = availability_instant(as_of)
    seal, publication, custody = map(availability_instant, (sealed_at, published_at, created_at))
    if session_date(effective_date) >= cutoff.astimezone(SHANGHAI).date():
        raise IntelligenceError("PORTFOLIO_EFFECTIVE_DATE_NOT_PRIOR")
    if not seal <= publication <= custody:
        raise IntelligenceError("PORTFOLIO_SOURCE_CUSTODY_ORDER_INVALID")
    reasons = []
    if max(seal, publication) > cutoff:
        reasons.append("PORTFOLIO_SOURCE_NOT_AVAILABLE_AT_DECISION")
    if custody > cutoff:
        reasons.append("PORTFOLIO_CUSTODY_AFTER_DECISION")
    return {
        "timing_status": "LATE_RECORDED" if reasons else "ON_TIME",
        "prospective": False,
        "reason_codes": sorted(reasons),
    }


def normalize_positions(positions):
    rows = []
    for raw in positions:
        if set(raw) != {"symbol", "shares", "avg_cost", "cost_basis"}:
            raise IntelligenceError("PORTFOLIO_POSITION_SHAPE_INVALID")
        symbol = company_code(raw["symbol"])
        values = {}
        for field in ("shares", "avg_cost", "cost_basis"):
            if raw[field] is None or isinstance(raw[field], bool):
                raise IntelligenceError("PORTFOLIO_POSITION_VALUE_INVALID")
            value = Decimal(str(raw[field]))
            if not value.is_finite() or value < 0:
                raise IntelligenceError("PORTFOLIO_POSITION_VALUE_INVALID")
            values[field] = value
        if _money(values["cost_basis"], label="cost_basis") != _money(
            _unit(values["avg_cost"], label="avg_cost") * values["shares"],
            label="position cost identity",
        ):
            raise IntelligenceError("PORTFOLIO_COST_IDENTITY_INVALID")
        rows.append(
            {"symbol": symbol, **{key: format(value, "f") for key, value in values.items()}}
        )
    symbols = [row["symbol"] for row in rows]
    if len(symbols) != len(set(symbols)):
        raise IntelligenceError("PORTFOLIO_POSITION_DUPLICATED")
    return sorted(rows, key=lambda row: row["symbol"])


def build_portfolio_state(
    *,
    as_of,
    created_at,
    store_plan_ref,
    frozen_pointer_ref,
    catalog_ref,
    source_record_id,
    source_effective_trade_date,
    source_sealed_at,
    pointer_published_at,
    source_refs,
    positions,
    cash,
):
    stamp = timestamp(as_of, label="portfolio decision cutoff")
    custody = timestamp(created_at, label="portfolio actual custody")
    _money(cash, label="manual cash")
    amount = Decimal(str(cash))
    if type(source_record_id) is not str or not source_record_id:
        raise IntelligenceError("PORTFOLIO_SOURCE_RECORD_INVALID")
    refs = [validate_ref(ref) for ref in (store_plan_ref, frozen_pointer_ref, catalog_ref)]
    fields = {
        "as_of": stamp,
        "trade_date": availability_instant(stamp).astimezone(SHANGHAI).strftime("%Y%m%d"),
        "strategy_id": STRATEGY_ID,
        "store_plan_ref": refs[0],
        "frozen_pointer_ref": refs[1],
        "catalog_ref": refs[2],
        "source_record_id": source_record_id,
        "source_effective_trade_date": session_date(source_effective_trade_date).strftime("%Y%m%d"),
        "source_sealed_at": source_sealed_at,
        "pointer_published_at": pointer_published_at,
        "source_refs": physical_refs([*refs, *source_refs]),
        "positions": normalize_positions(positions),
        "cash": format(amount, "f"),
        **portfolio_timing(
            as_of=stamp,
            effective_date=source_effective_trade_date,
            sealed_at=source_sealed_at,
            published_at=pointer_published_at,
            created_at=custody,
        ),
    }
    return build_artifact(
        kind=PORTFOLIO_STATE_KIND,
        identity_field="portfolio_state_id",
        identity=business_identity(
            kind=PORTFOLIO_STATE_KIND,
            identity_inputs={"created_at": custody, **fields},
        ),
        created_at=custody,
        fields=fields,
    )


def validate_portfolio_state(artifact):
    value = validate_artifact(artifact, expected_kind=PORTFOLIO_STATE_KIND)
    body = value["payload"]
    inputs = {
        key: body[key]
        for key in (
            "as_of",
            "store_plan_ref",
            "frozen_pointer_ref",
            "catalog_ref",
            "source_record_id",
            "source_effective_trade_date",
            "source_sealed_at",
            "pointer_published_at",
            "source_refs",
            "positions",
            "cash",
        )
    }
    if build_portfolio_state(created_at=value["created_at"], **inputs) != value:
        raise IntelligenceError("PORTFOLIO_STATE_INTRINSIC_REPLAY_MISMATCH")
    return value
