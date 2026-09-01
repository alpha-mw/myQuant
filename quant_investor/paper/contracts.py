"""Canonical contracts for the independent sell-only Paper authority."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Final, Mapping

from quant_investor.contracts import canonical_json_bytes

WRITER_ID: Final = "cn-paper-risk-exit-writer.v1"
POLICY_RELATIVE_PATH: Final = (
    "results/policies/paper/aggressive_tech_manufacturing/"
    "owner-paper-risk-execution-policy-20260901-v1.json"
)
POLICY_SHA256: Final = "d3f86f3ba26556d084eebc48136864a5ba858efe75c9c9d139fb99627d746961"
PAPER_ROOT: Final = "results/paper/accounts"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_MONEY = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{4}$")
_RATIO = re.compile(r"^(?:0|1)\.[0-9]{2}$")
_REF_FIELDS = {"path", "sha256"}


class PaperError(RuntimeError):
    """Stable fail-closed Paper error."""

    exit_code = 2

    def __init__(self, code: str, detail: str = "") -> None:
        if type(code) is not str or not code.startswith("PAPER_"):
            raise ValueError("Paper error code is invalid")
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}:{detail}")


def require_exact(value: Any, fields: set[str], *, code: str, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise PaperError(code, f"{label} fields differ")
    return dict(value)


def require_sha(value: Any, *, code: str, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise PaperError(code, f"{label} is not SHA-256")
    return value


def require_identifier(value: Any, *, code: str, label: str) -> str:
    if type(value) is not str or len(value) > 80 or _IDENTIFIER.fullmatch(value) is None:
        raise PaperError(code, f"{label} is invalid")
    return value


def require_symbol(value: Any, *, code: str, label: str = "symbol") -> str:
    if type(value) is not str or _SYMBOL.fullmatch(value) is None:
        raise PaperError(code, f"{label} is invalid")
    return value


def require_date(value: Any, *, code: str, label: str) -> str:
    if type(value) is not str or len(value) != 8 or not value.isdigit():
        raise PaperError(code, f"{label} must be YYYYMMDD")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise PaperError(code, f"{label} is invalid") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise PaperError(code, f"{label} is not canonical")
    return value


def require_timestamp(value: Any, *, code: str, label: str) -> str:
    if type(value) is not str:
        raise PaperError(code, f"{label} must be UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise PaperError(code, f"{label} is invalid") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise PaperError(code, f"{label} is not canonical")
    return value


def require_money(value: Any, *, code: str, label: str, nonnegative: bool = False) -> Decimal:
    if type(value) is not str or _MONEY.fullmatch(value) is None or value == "-0.0000":
        raise PaperError(code, f"{label} must be scale-4 decimal text")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise PaperError(code, f"{label} is invalid") from exc
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        raise PaperError(code, f"{label} is invalid")
    return parsed


def require_ratio(value: Any, *, code: str, label: str) -> Decimal:
    if type(value) is not str or _RATIO.fullmatch(value) is None:
        raise PaperError(code, f"{label} must be scale-2 ratio text")
    return Decimal(value)


def validate_ref(value: Any, *, code: str, label: str) -> dict[str, str]:
    row = require_exact(value, _REF_FIELDS, code=code, label=label)
    path = row.get("path")
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise PaperError(code, f"{label}.path is invalid")
    try:
        path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PaperError(code, f"{label}.path must be ASCII") from exc
    return {"path": path, "sha256": require_sha(row.get("sha256"), code=code, label=label)}


def seal_document(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    result = {**body, "semantic_sha256": digest}
    canonical_json_bytes(result)
    return result


def parse_document(raw: bytes, *, expected_schema: str, code: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PaperError(code, "document is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise PaperError(code, "document is not canonical")
    if value.get("schema_version") != expected_schema:
        raise PaperError(code, "schema differs")
    observed = value.get("semantic_sha256")
    body = dict(value)
    body.pop("semantic_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if observed != expected:
        raise PaperError(code, "semantic SHA differs")
    return value


def writer_registration() -> dict[str, Any]:
    return seal_document(
        {
            "schema_version": "paper-writer-registration.v1",
            "writer_id": WRITER_ID,
            "writer_version": "1",
            "allowed_account_type": "PAPER",
            "allowed_actions": ["EXIT_100", "REDUCE_25", "REDUCE_50"],
            "allow_new_risk": False,
            "paper_order": True,
            "paper_fill": True,
            "paper_ledger_mutation": True,
            "broker": False,
            "real_order": False,
            "live_execution": False,
            "actual_holdings_mutation": False,
        }
    )


_REGISTRATION_FIELDS = {
    "schema_version",
    "semantic_sha256",
    "account_id",
    "account_type",
    "strategy_id",
    "currency",
    "allowed_writer_id",
    "policy_ref",
    "genesis_source_ref",
    "initial_cash",
    "initial_positions",
    "all_initial_shares_settled",
    "broker",
    "real_order",
    "actual_holdings_mutation",
}
_POSITION_FIELDS = {
    "symbol",
    "name",
    "shares",
    "settled_shares",
    "avg_cost",
    "cost_basis",
    "realized_pnl",
    "cumulative_fees",
    "acquisition_lots",
}
_LOT_FIELDS = {"shares", "acquisition_date", "settlement_date"}


def validate_registration(value: Any) -> dict[str, Any]:
    code = "PAPER_ACCOUNT_REGISTRATION_INVALID"
    row = require_exact(value, _REGISTRATION_FIELDS, code=code, label="registration")
    if row["schema_version"] != "paper-account-registration.v1":
        raise PaperError(code, "schema differs")
    account = require_identifier(row["account_id"], code=code, label="account_id")
    if (
        row["account_type"] != "PAPER"
        or row["strategy_id"] != "aggressive_tech_manufacturing"
        or row["currency"] != "CNY"
        or row["allowed_writer_id"] != WRITER_ID
        or row["all_initial_shares_settled"] is not True
        or any(
            row[field] is not False
            for field in ("broker", "real_order", "actual_holdings_mutation")
        )
    ):
        raise PaperError(code, "registration authority differs")
    policy_ref = validate_ref(row["policy_ref"], code=code, label="policy_ref")
    source_ref = validate_ref(row["genesis_source_ref"], code=code, label="genesis_source_ref")
    cash = require_money(row["initial_cash"], code=code, label="initial_cash", nonnegative=True)
    positions = row["initial_positions"]
    if type(positions) is not list:
        raise PaperError(code, "initial_positions must be list")
    normalized = []
    seen: set[str] = set()
    for index, item in enumerate(positions):
        position = require_exact(item, _POSITION_FIELDS, code=code, label=f"position[{index}]")
        symbol = require_symbol(position["symbol"], code=code)
        if symbol in seen:
            raise PaperError(code, "duplicate position")
        seen.add(symbol)
        shares = position["shares"]
        settled = position["settled_shares"]
        if type(shares) is not int or shares < 0 or type(settled) is not int or settled != shares:
            raise PaperError(code, "initial shares are not fully settled")
        lots = position["acquisition_lots"]
        if type(lots) is not list:
            raise PaperError(code, "acquisition_lots must be list")
        lot_sum = 0
        normalized_lots = []
        for lot_index, lot_value in enumerate(lots):
            lot = require_exact(lot_value, _LOT_FIELDS, code=code, label=f"lot[{lot_index}]")
            lot_shares = lot["shares"]
            if type(lot_shares) is not int or lot_shares <= 0:
                raise PaperError(code, "lot shares invalid")
            require_date(lot["acquisition_date"], code=code, label="acquisition_date")
            require_date(lot["settlement_date"], code=code, label="settlement_date")
            lot_sum += lot_shares
            normalized_lots.append(dict(lot))
        if lot_sum != settled:
            raise PaperError(code, "settled lots differ")
        normalized.append(
            {
                **position,
                "avg_cost": format(
                    require_money(
                        position["avg_cost"], code=code, label="avg_cost", nonnegative=True
                    ),
                    ".4f",
                ),
                "cost_basis": format(
                    require_money(
                        position["cost_basis"], code=code, label="cost_basis", nonnegative=True
                    ),
                    ".4f",
                ),
                "realized_pnl": format(
                    require_money(position["realized_pnl"], code=code, label="realized_pnl"),
                    ".4f",
                ),
                "cumulative_fees": format(
                    require_money(
                        position["cumulative_fees"],
                        code=code,
                        label="cumulative_fees",
                        nonnegative=True,
                    ),
                    ".4f",
                ),
                "acquisition_lots": normalized_lots,
            }
        )
    if [item["symbol"] for item in normalized] != sorted(
        seen, key=lambda text: text.encode("ascii")
    ):
        raise PaperError(code, "positions must be ASCII sorted")
    return {
        **row,
        "account_id": account,
        "policy_ref": policy_ref,
        "genesis_source_ref": source_ref,
        "initial_cash": format(cash, ".4f"),
        "initial_positions": normalized,
    }


_INTENT_FIELDS = {
    "schema_version",
    "semantic_sha256",
    "source_intent_id",
    "idempotency_key_sha256",
    "economic_action_key_sha256",
    "account_id",
    "strategy_id",
    "signal_date",
    "eligible_from_trade_date",
    "symbol",
    "action",
    "requested_ratio",
    "requested_shares",
    "reason_codes",
    "policy_ref",
    "expected_account_pointer_sha256",
    "expected_position",
    "evidence_refs",
    "broker",
    "real_order",
    "actual_holdings_mutation",
}


def validate_intent(value: Any) -> dict[str, Any]:
    code = "PAPER_INTENT_INVALID"
    row = require_exact(value, _INTENT_FIELDS, code=code, label="intent")
    if row["schema_version"] != "paper-risk-intent.v1":
        raise PaperError(code, "schema differs")
    require_identifier(row["source_intent_id"], code=code, label="source_intent_id")
    require_sha(row["idempotency_key_sha256"], code=code, label="idempotency_key_sha256")
    require_sha(row["economic_action_key_sha256"], code=code, label="economic_action_key_sha256")
    require_identifier(row["account_id"], code=code, label="account_id")
    if row["strategy_id"] != "aggressive_tech_manufacturing":
        raise PaperError(code, "strategy differs")
    require_date(row["signal_date"], code=code, label="signal_date")
    require_date(row["eligible_from_trade_date"], code=code, label="eligible_from_trade_date")
    require_symbol(row["symbol"], code=code)
    if row["action"] not in {"EXIT_100", "REDUCE_25", "REDUCE_50"}:
        raise PaperError(code, "action differs")
    ratio = require_ratio(row["requested_ratio"], code=code, label="requested_ratio")
    expected_ratio = {
        "EXIT_100": Decimal("1.00"),
        "REDUCE_25": Decimal("0.25"),
        "REDUCE_50": Decimal("0.50"),
    }
    if ratio != expected_ratio[row["action"]]:
        raise PaperError(code, "ratio differs from action")
    if type(row["requested_shares"]) is not int or row["requested_shares"] < 0:
        raise PaperError(code, "requested_shares invalid")
    reasons = row["reason_codes"]
    if type(reasons) is not list or not reasons or reasons != sorted(set(reasons)):
        raise PaperError(code, "reason_codes invalid")
    validate_ref(row["policy_ref"], code=code, label="policy_ref")
    expected_pointer = row["expected_account_pointer_sha256"]
    if expected_pointer != "EMPTY":
        require_sha(expected_pointer, code=code, label="expected pointer")
    expected = require_exact(
        row["expected_position"],
        {"shares", "settled_shares", "avg_cost"},
        code=code,
        label="expected_position",
    )
    if type(expected["shares"]) is not int or type(expected["settled_shares"]) is not int:
        raise PaperError(code, "expected shares invalid")
    require_money(expected["avg_cost"], code=code, label="expected avg_cost", nonnegative=True)
    refs = row["evidence_refs"]
    if type(refs) is not list:
        raise PaperError(code, "evidence_refs invalid")
    for index, ref in enumerate(refs):
        validate_ref(ref, code=code, label=f"evidence_refs[{index}]")
    if any(
        row[field] is not False for field in ("broker", "real_order", "actual_holdings_mutation")
    ):
        raise PaperError(code, "real authority forbidden")
    return row


_ELIGIBILITY_FIELDS = {
    "schema_version",
    "semantic_sha256",
    "account_id",
    "source_intent_ref",
    "symbol",
    "signal_date",
    "eligible_trade_date",
    "evaluated_trade_date",
    "open_price",
    "previous_close",
    "limit_up",
    "limit_down",
    "suspended",
    "corporate_action_state",
    "open_session_ordinal",
    "expiry_session_ordinal",
    "calendar_ref",
    "raw_bar_ref",
    "price_limit_ref",
    "suspension_ref",
    "corporate_action_ref",
    "evidence_status",
}


def validate_eligibility(value: Any) -> dict[str, Any]:
    code = "PAPER_ELIGIBILITY_INVALID"
    row = require_exact(value, _ELIGIBILITY_FIELDS, code=code, label="eligibility")
    if row["schema_version"] != "paper-input-eligibility.v1":
        raise PaperError(code, "schema differs")
    require_identifier(row["account_id"], code=code, label="account_id")
    validate_ref(row["source_intent_ref"], code=code, label="source_intent_ref")
    require_symbol(row["symbol"], code=code)
    for field in ("signal_date", "eligible_trade_date", "evaluated_trade_date"):
        require_date(row[field], code=code, label=field)
    for field in ("open_price", "previous_close", "limit_up", "limit_down"):
        if row[field] is not None:
            require_money(row[field], code=code, label=field, nonnegative=True)
    if type(row["suspended"]) is not bool:
        raise PaperError(code, "suspended must be bool")
    if row["corporate_action_state"] not in {"CLEAR", "PENDING"}:
        raise PaperError(code, "corporate_action_state invalid")
    for field in ("open_session_ordinal", "expiry_session_ordinal"):
        if type(row[field]) is not int or row[field] < 0:
            raise PaperError(code, f"{field} invalid")
    for field in (
        "calendar_ref",
        "raw_bar_ref",
        "price_limit_ref",
        "suspension_ref",
        "corporate_action_ref",
    ):
        if row[field] is not None:
            validate_ref(row[field], code=code, label=field)
    if row["evidence_status"] not in {"READY", "NOT_YET_AVAILABLE", "MISSING"}:
        raise PaperError(code, "evidence_status invalid")
    if row["evidence_status"] == "READY" and any(
        row[field] is None
        for field in (
            "open_price",
            "previous_close",
            "limit_up",
            "limit_down",
            "calendar_ref",
            "raw_bar_ref",
            "price_limit_ref",
            "suspension_ref",
            "corporate_action_ref",
        )
    ):
        raise PaperError(code, "READY evidence is incomplete")
    return row


def canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def utc_now_text() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "PAPER_ROOT",
    "POLICY_RELATIVE_PATH",
    "POLICY_SHA256",
    "WRITER_ID",
    "PaperError",
    "canonical_sha",
    "parse_document",
    "require_date",
    "require_exact",
    "require_identifier",
    "require_money",
    "require_sha",
    "require_symbol",
    "seal_document",
    "utc_now_text",
    "validate_eligibility",
    "validate_intent",
    "validate_ref",
    "validate_registration",
    "writer_registration",
]
