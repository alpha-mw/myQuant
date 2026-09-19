"""Owner-declared registered financial facts; never a broker or writer authority."""

from collections.abc import Mapping
from decimal import Decimal, DecimalException
import re

from quant_investor.operations.daily_contract import validate_ref
from .event_contracts import event_date, instant
from .event_store import EVENT_DIMENSIONS, StrategyEventStoreError
from .store import StrategyRecordStoreError, content_sha256

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"
FACT_SCHEMA = "cn-registered-owner-facts.v1"
DECLARATION_SCHEMA = "cn-registered-financial-declaration.v1"
STRATEGY = "aggressive_tech_manufacturing"
NONE = "OWNER_DECLARED_NONE"
FACTS = "OWNER_DECLARED_FACTS"
FEE_STATUS = "OWNER_FEE_POLICY_APPLIED_BROKER_STATEMENT_READBACK_PENDING"
AUTHORITY = dict.fromkeys(
    (
        "store_mutation",
        "actual_holdings_mutation",
        "cash_mutation",
        "broker",
        "order",
        "execution",
        "trade",
        "daily_writer",
    ),
    False,
)
COMMON = {
    "schema_version",
    "strategy_id",
    "trade_date",
    "owner",
    "owner_declared_at",
    "baseline_store_pointer_ref",
    "writer_record_id",
    "domains",
    "evidence_level",
    "broker_statement_verified",
    "authority",
    "content_sha256",
}
FACT_FIELDS = COMMON | {"owner_fact_id", "writer_pointer_sha256"}
DECLARATION_FIELDS = COMMON | {
    "declaration_id",
    "owner_fact_ref",
    "registered_at",
    "baseline_catalog_ref",
    "baseline_record_id",
    "writer_store_pointer_ref",
    "writer_catalog_ref",
    "writer_record_refs",
    "late_event_behavior",
}
RECORD_REFS = {
    "manifest",
    "manual_manifest",
    "ledger",
    "pnl",
    "performance_manifest",
    "performance_series",
    "performance_owner_declaration",
}
TRADE_FIELDS = {
    "trade_id",
    "symbol",
    "name",
    "side",
    "shares",
    "execution_price",
    "trade_value",
    "commission_cny",
    "stamp_duty_cny",
    "transfer_fee_cny",
    "final_total_fee_cny",
    "cost_basis_cny",
    "avg_cost_cny_per_share",
    "commission_rate",
    "commission_minimum_cny",
    "commission_includes_regulatory_and_handling",
    "transfer_fee_rate",
    "fill_cost_status",
    "reported_by",
    "source_channel",
    "trade_date",
}


class RegisteredEventError(StrategyRecordStoreError):
    """Invalid owner/source evidence, with no fallback to an empty event."""


def seal(value):
    return {**value, "content_sha256": content_sha256(value)}


def ref(value):
    try:
        return validate_ref(value)
    except ValueError as exc:
        raise RegisteredEventError("REGISTERED_EVENT_REF_INVALID") from exc


def stamp(value):
    try:
        return instant(value, label="registered declaration")
    except StrategyEventStoreError as exc:
        raise RegisteredEventError("REGISTERED_EVENT_TIME_INVALID") from exc


def identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise RegisteredEventError("REGISTERED_EVENT_ID_INVALID")
    return value


def paths(writer_sha):
    ref({"path": "pointer.json", "sha256": writer_sha})
    prefix = f"{RECORD_ROOT}/_event_store/registered/{writer_sha}"
    return {
        "pointer": prefix + "/writer-pointer.v1.json",
        "declaration": prefix + "/declaration.v1.json",
    }


def validate_domains(value):
    if type(value) is not dict or set(value) != set(EVENT_DIMENSIONS):
        raise RegisteredEventError("REGISTERED_EVENT_DOMAINS_INCOMPLETE")
    for row in value.values():
        if type(row) is not dict or set(row) != {"state", "fact_refs"}:
            raise RegisteredEventError("REGISTERED_EVENT_DOMAIN_INVALID")
        if type(row["fact_refs"]) is not list:
            raise RegisteredEventError("REGISTERED_EVENT_DOMAIN_REFS_INVALID")
        refs = [ref(r) for r in row["fact_refs"]]
        keys = [(r["path"], r["sha256"]) for r in refs]
        if keys != sorted(set(keys)) or len({r[0] for r in keys}) != len(keys):
            raise RegisteredEventError("REGISTERED_EVENT_DOMAIN_REFS_INVALID")
        if not ((row["state"] == NONE and not refs) or (row["state"] == FACTS and refs)):
            raise RegisteredEventError("REGISTERED_EVENT_DOMAIN_STATE_INVALID")
    return value


def _common(value, schema, fields):
    if type(value) is not dict or set(value) != fields or value.get("schema_version") != schema:
        raise RegisteredEventError("REGISTERED_EVENT_FIELDS_INVALID")
    if value["content_sha256"] != content_sha256(value):
        raise RegisteredEventError("REGISTERED_EVENT_CONTENT_SHA_INVALID")
    if (
        value["strategy_id"] != STRATEGY
        or type(value["owner"]) is not str
        or not value["owner"]
        or value["owner"] != value["owner"].strip()
        or len(value["owner"]) > 256
        or value["evidence_level"] != "OWNER_DECLARED"
        or value["broker_statement_verified"] is not False
        or value["authority"] != AUTHORITY
        or any(type(x) is not bool for x in value["authority"].values())
    ):
        raise RegisteredEventError("REGISTERED_EVENT_SCOPE_INVALID")
    try:
        event_date(value["trade_date"])
    except StrategyEventStoreError as exc:
        raise RegisteredEventError("REGISTERED_EVENT_DATE_INVALID") from exc
    stamp(value["owner_declared_at"])
    identifier(value["writer_record_id"])
    baseline = ref(value["baseline_store_pointer_ref"])
    if (
        re.fullmatch(
            re.escape(RECORD_ROOT)
            + r"/_record_store/daily_close_transactions/"
            + r"daily-close-[0-9]{8}-[0-9a-f]{16}/committed-pointer\.v1\.json",
            baseline["path"],
        )
        is None
    ):
        raise RegisteredEventError("REGISTERED_EVENT_BASELINE_CUSTODY_INVALID")
    validate_domains(value["domains"])


def validate_fact(value):
    _common(value, FACT_SCHEMA, FACT_FIELDS)
    identifier(value["owner_fact_id"])
    paths(value["writer_pointer_sha256"])
    return value


def validate_declaration(value, *, declaration_ref=None):
    _common(value, DECLARATION_SCHEMA, DECLARATION_FIELDS)
    identifier(value["declaration_id"])
    identifier(value["baseline_record_id"])
    if stamp(value["registered_at"]) < stamp(value["owner_declared_at"]):
        raise RegisteredEventError("REGISTERED_EVENT_REGISTRATION_BEFORE_FACT")
    if value["late_event_behavior"] != "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED":
        raise RegisteredEventError("REGISTERED_EVENT_RESTATEMENT_RULE_INVALID")
    for name in ("owner_fact_ref", "baseline_catalog_ref", "writer_catalog_ref"):
        ref(value[name])
    pointer = ref(value["writer_store_pointer_ref"])
    expected = paths(pointer["sha256"])
    if pointer["path"] != expected["pointer"] or (
        declaration_ref is not None and ref(declaration_ref)["path"] != expected["declaration"]
    ):
        raise RegisteredEventError("REGISTERED_EVENT_CUSTODY_PATH_INVALID")
    if (
        type(value["writer_record_refs"]) is not dict
        or set(value["writer_record_refs"]) != RECORD_REFS
    ):
        raise RegisteredEventError("REGISTERED_EVENT_RECORD_REFS_INVALID")
    for item in value["writer_record_refs"].values():
        ref(item)
    return value


def number(value):
    if isinstance(value, bool) or value is None:
        raise RegisteredEventError("REGISTERED_EVENT_NUMBER_INVALID")
    try:
        result = Decimal(str(value))
    except (ValueError, DecimalException) as exc:
        raise RegisteredEventError("REGISTERED_EVENT_NUMBER_INVALID") from exc
    if not result.is_finite():
        raise RegisteredEventError("REGISTERED_EVENT_NUMBER_INVALID")
    return result


def close(left, right):
    if abs(left - right) > Decimal("0.01"):
        raise RegisteredEventError("REGISTERED_EVENT_FINANCIAL_BRIDGE_MISMATCH")


def positions(record):
    values = record.get("positions")
    if type(values) is not list or not values:
        raise RegisteredEventError("REGISTERED_EVENT_POSITIONS_INVALID")
    result = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise RegisteredEventError("REGISTERED_EVENT_POSITIONS_INVALID")
        symbol = row.get("symbol")
        if (
            type(symbol) is not str
            or re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", symbol) is None
            or symbol in result
        ):
            raise RegisteredEventError("REGISTERED_EVENT_POSITIONS_INVALID")
        qty, avg, cost = (number(row.get(k)) for k in ("shares", "avg_cost", "cost_basis"))
        if qty <= 0 or qty != qty.to_integral_value() or avg <= 0 or cost <= 0:
            raise RegisteredEventError("REGISTERED_EVENT_POSITIONS_INVALID")
        close(qty * avg, cost)
        result[symbol] = (qty, avg, cost)
    return result


def _trade(row, *, day, owner):
    if type(row) is not dict or set(row) != TRADE_FIELDS:
        raise RegisteredEventError("REGISTERED_EVENT_TRADE_FIELDS_UNSUPPORTED")
    identifier(row["trade_id"])
    if (
        row["side"] != "BUY"
        or row["trade_date"] != day.replace("-", "")
        or row["reported_by"] != owner
        or row["fill_cost_status"] != FEE_STATUS
        or row["commission_includes_regulatory_and_handling"] is not True
        or any(type(row[k]) is not str or not row[k].strip() for k in ("name", "source_channel"))
        or type(row["symbol"]) is not str
        or re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", row["symbol"]) is None
    ):
        raise RegisteredEventError("REGISTERED_EVENT_TRADE_PROFILE_UNSUPPORTED")
    numeric = {
        k: number(row[k])
        for k in (
            "shares",
            "execution_price",
            "trade_value",
            "commission_cny",
            "stamp_duty_cny",
            "transfer_fee_cny",
            "final_total_fee_cny",
            "cost_basis_cny",
            "avg_cost_cny_per_share",
            "commission_rate",
            "commission_minimum_cny",
            "transfer_fee_rate",
        )
    }
    if (
        any(v < 0 for v in numeric.values())
        or any(
            numeric[k] <= 0
            for k in (
                "shares",
                "execution_price",
                "trade_value",
                "cost_basis_cny",
                "avg_cost_cny_per_share",
            )
        )
        or numeric["shares"] != numeric["shares"].to_integral_value()
    ):
        raise RegisteredEventError("REGISTERED_EVENT_TRADE_NUMBER_INVALID")
    close(numeric["trade_value"], numeric["shares"] * numeric["execution_price"])
    close(
        numeric["final_total_fee_cny"],
        sum(numeric[k] for k in ("commission_cny", "stamp_duty_cny", "transfer_fee_cny")),
    )
    close(numeric["cost_basis_cny"], numeric["trade_value"] + numeric["final_total_fee_cny"])
    close(numeric["avg_cost_cny_per_share"] * numeric["shares"], numeric["cost_basis_cny"])
    return numeric


def validate_buy_transition(*, baseline, writer, manual, fact, record_refs):
    """Attribute already registered differences; never calculate a new ledger."""
    validate_fact(fact)
    if any(not isinstance(v, Mapping) for v in (baseline, writer, manual, record_refs)):
        raise RegisteredEventError("REGISTERED_EVENT_FINANCIAL_PROFILE_UNSUPPORTED")
    if (
        manual.get("schema_version") != "cn_aggressive_manual_execution.v3"
        or any(
            manual.get(k) != "owner_declared_manual_execution_applied"
            for k in ("status", "execution_status")
        )
        or writer.get("execution_kind") != "applied_effective_ledger"
        or writer.get("official_valuation") is not False
        or writer.get("record") != fact["writer_record_id"]
        or writer.get("data_date") != fact["trade_date"]
        or baseline.get("official_valuation") is not True
        or baseline.get("data_date", "") >= fact["trade_date"]
        or writer.get("source_record") != baseline.get("record")
        or any(
            manual.get(k) != []
            for k in ("applied_local_trades", "rejected_or_pending_trades", "funding_events")
        )
        or manual.get("owner_reported_external_fills") is not True
        or manual.get("no_broker_api_called") is not True
        or any(writer.get(k) is not None for k in ("funding", "funding_correction"))
        or any(manual.get(k) not in (None, []) for k in ("corporate_actions", "manual_changes"))
        or any(
            manual.get(k) is not None
            for k in (
                "corporate_action_application",
                "corporate_action_application_ref",
                "manual_funding_supplement",
            )
        )
        or any(number(manual.get(k)) != 0 for k in ("net_external_flow", "excluded_external_flow"))
        or not isinstance(manual.get("owner_declaration"), Mapping)
        or manual["owner_declaration"].get("approved_by") != fact["owner"]
    ):
        raise RegisteredEventError("REGISTERED_EVENT_FINANCIAL_PROFILE_UNSUPPORTED")
    for key in ("trade_date", "valuation_trade_date"):
        if manual.get(key) != fact["trade_date"].replace("-", ""):
            raise RegisteredEventError("REGISTERED_EVENT_TRADE_DATE_MISMATCH")
    allowed = {
        (ref(record_refs[k])["path"], ref(record_refs[k])["sha256"])
        for k in ("manifest", "manual_manifest", "ledger", "pnl")
    }
    manual_ref = ref(record_refs["manual_manifest"])
    manual_key = (manual_ref["path"], manual_ref["sha256"])
    for name, domain in fact["domains"].items():
        keys = {(ref(r)["path"], ref(r)["sha256"]) for r in domain["fact_refs"]}
        if not keys <= allowed:
            raise RegisteredEventError("REGISTERED_EVENT_FACT_REF_UNBOUND")
        if name in {"executions", "fills", "cost_basis_changes"} and (
            domain["state"] != FACTS or manual_key not in keys
        ):
            raise RegisteredEventError("REGISTERED_EVENT_TRADE_DOMAIN_UNPROVEN")
        if name in {"funding", "corporate_actions", "manual_changes"} and domain["state"] != NONE:
            raise RegisteredEventError("REGISTERED_EVENT_MIXED_DOMAIN_UNSUPPORTED")
    trades = manual.get("applied_owner_declared_trades")
    if type(trades) is not list or not trades:
        raise RegisteredEventError("REGISTERED_EVENT_TRADES_MISSING")
    before, after = positions(baseline), positions(writer)
    quantities = {s: v[0] for s, v in before.items()}
    costs = {s: v[2] for s, v in before.items()}
    seen, changed, total_cost = set(), set(), Decimal("0")
    for trade in trades:
        numeric = _trade(trade, day=fact["trade_date"], owner=fact["owner"])
        if trade["trade_id"] in seen:
            raise RegisteredEventError("REGISTERED_EVENT_TRADE_ID_DUPLICATE")
        seen.add(trade["trade_id"])
        symbol = trade["symbol"]
        changed.add(symbol)
        quantities[symbol] = quantities.get(symbol, Decimal("0")) + numeric["shares"]
        costs[symbol] = costs.get(symbol, Decimal("0")) + numeric["cost_basis_cny"]
        total_cost += numeric["cost_basis_cny"]
    if set(after) != set(quantities):
        raise RegisteredEventError("REGISTERED_EVENT_SYMBOL_DELTA_MISMATCH")
    for symbol, (qty, avg, cost) in after.items():
        if qty != quantities[symbol] or (symbol not in changed and after[symbol] != before[symbol]):
            raise RegisteredEventError("REGISTERED_EVENT_POSITION_DELTA_MISMATCH")
        close(cost, costs[symbol])
        close(avg * qty, cost)
    close(
        number(writer["accounting"]["cash_after"]),
        number(baseline["accounting"]["cash_after"]) - total_cost,
    )
    return {
        "profile": "OWNER_DECLARED_BUYS_V1",
        "trade_count": len(trades),
        "fee_evidence_level": FEE_STATUS,
    }
