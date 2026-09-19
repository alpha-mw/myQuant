"""Exact head/selector/view contracts for dated completed-EOD Dashboard serving."""

from datetime import datetime, timezone
import hashlib
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import FALSE_AUTHORITY, _false_authority, _validate_day

POLICY = "native-eod-first.v1"
PREFIX = "portfolio_dashboard/private/generated"
HEAD_JSON = "cn_daily_completed_head.v1.json"
HEAD_JS = "cn_daily_completed_head.v1.js"
EVIDENCE_JSON = "cn_daily_dashboard_evidence.v1.json"
EVIDENCE_JS = "cn_daily_dashboard_evidence.v1.js"
SELECTOR_JSON = "cn_aggressive_dashboard_selector.v2.json"
SELECTOR_JS = "cn_aggressive_dashboard_selector.v2.js"
FINANCIAL_NAMES = (
    "cn_aggressive_dashboard.v1.json",
    "cn_aggressive_dashboard.v1.js",
    "cn_aggressive_dashboard.v2.json",
    "cn_aggressive_dashboard.v2.js",
)
SERVING_NAMES = (*FINANCIAL_NAMES, EVIDENCE_JSON, EVIDENCE_JS, SELECTOR_JSON, SELECTOR_JS)
ALL_NAMES = frozenset((*SERVING_NAMES, HEAD_JSON, HEAD_JS))
REGISTERED_VIEW_FIELDS = {
    "registered_transition_ref",
    "registered_transition",
    "registered_close_summary",
}
REGISTERED_CLOSE_FIELDS = {
    "source_profile",
    "store_plan_ref",
    "decision_baseline_pointer_ref",
    "writer_pointer_ref",
    "final_pointer_ref",
    "writer_record_id",
    "final_record_id",
    "official_valuation",
    "close_writer_trade_count",
    "close_writer_order_count",
    "close_writer_fill_count",
}
HEAD_FIELDS = {
    "schema_version",
    "trade_date",
    "completion_ref",
    "previous_head_sha256",
    "registered_at",
    "authority",
    "content_sha256",
}
SELECTOR_FIELDS = {
    "schema_version",
    "attempt_id",
    "status",
    "updated_at",
    "v2_content_sha256",
    "reason",
    "content_sha256",
    "trade_date",
    "completion_ref",
    "completed_head_sha256",
    "v1_byte_sha256",
    "v2_byte_sha256",
    "daily_evidence_sha256",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sealed(value):
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    return {**body, "content_sha256": digest(canonical_json_bytes(body))}


def instant(value):
    try:
        if type(value) is not str or value != value.strip():
            raise ValueError("timestamp text required")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone missing")
        return parsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError("DASHBOARD_TIMESTAMP_INVALID") from exc


def _shape(value, fields, schema):
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != schema
        or sealed(value) != value
    ):
        raise ContractError("DASHBOARD_SERVING_CONTRACT_INVALID:" + schema)


def completion_ref(ref, day):
    ref = validate_ref(ref)
    if ref["path"] != f"results/operations/daily_production/CN/{day}/completion.v1.json":
        raise ContractError("DASHBOARD_COMPLETION_PATH_INVALID")
    return ref


def validate_head(value):
    _shape(value, HEAD_FIELDS, "cn-daily-completed-head.v1")
    _validate_day(value["trade_date"])
    completion_ref(value["completion_ref"], value["trade_date"])
    if not _false_authority(value["authority"]) or utc_stamp(value["registered_at"]) > datetime.now(
        timezone.utc
    ):
        raise ContractError("DASHBOARD_HEAD_AUTHORITY_OR_TIME_INVALID")
    if value["previous_head_sha256"] is not None:
        validate_ref({"path": "head.json", "sha256": value["previous_head_sha256"]})
    return value


def build_head(*, day, ref, previous_sha, registered_at):
    return validate_head(
        sealed(
            {
                "schema_version": "cn-daily-completed-head.v1",
                "trade_date": day,
                "completion_ref": completion_ref(ref, day),
                "previous_head_sha256": previous_sha,
                "registered_at": registered_at,
                "authority": FALSE_AUTHORITY,
            }
        )
    )


def head_bytes(raw):
    return validate_head(parse_canonical_json_bytes(raw))


def raw_js(global_name, raw_name, raw):
    # The raw JSON text is preserved exactly for browser byte-SHA verification.
    import json

    text = raw.decode("utf-8")
    literal = json.dumps(text, ensure_ascii=True)
    return (
        f"window.{raw_name} = {literal};\nwindow.{global_name} = JSON.parse(window.{raw_name});\n"
    ).encode()


def head_js(raw):
    head_bytes(raw)
    return raw_js("MyQuantCNDailyCompletedHead", "MyQuantCNDailyCompletedHeadRaw", raw)


def validate_selector(value):
    _shape(value, SELECTOR_FIELDS, "cn_aggressive_dashboard_selector.v3")
    _validate_day(value["trade_date"])
    completion_ref(value["completion_ref"], value["trade_date"])
    if (
        value["status"] != "UPDATED"
        or value["reason"] != "native_eod_completed"
        or type(value["attempt_id"]) is not str
        or not value["attempt_id"]
    ):
        raise ContractError("DASHBOARD_SEALED_SELECTOR_INVALID")
    if instant(value["updated_at"]).utcoffset() != ZoneInfo("Asia/Shanghai").utcoffset(
        instant(value["updated_at"])
    ):
        raise ContractError("DASHBOARD_SELECTOR_TIMEZONE_INVALID")
    for name in (
        "v2_content_sha256",
        "completed_head_sha256",
        "v1_byte_sha256",
        "v2_byte_sha256",
        "daily_evidence_sha256",
    ):
        validate_ref({"path": "serving.json", "sha256": value[name]})
    return value


def build_selector(*, head, head_sha, v1_raw, v2_raw, evidence_raw, commit_at):
    v2 = __import__("json").loads(v2_raw)
    return validate_selector(
        sealed(
            {
                "schema_version": "cn_aggressive_dashboard_selector.v3",
                "attempt_id": v2["publication_attempt_id"],
                "status": "UPDATED",
                "updated_at": utc_stamp(commit_at)
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .isoformat(timespec="seconds"),
                "v2_content_sha256": v2["content_sha256"],
                "reason": "native_eod_completed",
                "trade_date": head["trade_date"],
                "completion_ref": head["completion_ref"],
                "completed_head_sha256": head_sha,
                "v1_byte_sha256": digest(v1_raw),
                "v2_byte_sha256": digest(v2_raw),
                "daily_evidence_sha256": digest(evidence_raw),
            }
        )
    )


def validate_registered_serving_fields(value, *, day, cutoff, final_pointer_ref):
    from quant_investor.contracts import validate_artifact

    if type(value) is not dict or set(value) != REGISTERED_VIEW_FIELDS:
        raise ContractError("DASHBOARD_REGISTERED_SERVING_FIELDS_INVALID")
    ref = validate_ref(value["registered_transition_ref"])
    report = validate_artifact(
        value["registered_transition"],
        expected_kind="registered_financial_transition_reconciliation",
    )
    if digest(canonical_json_bytes(report)) != ref["sha256"]:
        raise ContractError("DASHBOARD_REGISTERED_SERVING_REPORT_SHA_INVALID")
    body, summary = report["payload"], value["registered_close_summary"]
    if type(summary) is not dict or set(summary) != REGISTERED_CLOSE_FIELDS:
        raise ContractError("DASHBOARD_REGISTERED_CLOSE_SUMMARY_INVALID")
    for key in (
        "store_plan_ref",
        "decision_baseline_pointer_ref",
        "writer_pointer_ref",
        "final_pointer_ref",
    ):
        validate_ref(summary[key])
    if (
        body["trade_date"] != day
        or body["as_of"] != cutoff
        or body["evidence_level"] != "OWNER_DECLARED"
        or body["broker_statement_verified"] is not False
        or body["financial_admission_state"] != "VALIDATED_REGISTERED_TRANSITION"
        or body["risk_readiness_state"] != "OWNER_POLICY_REVALIDATION_REQUIRED"
        or summary["source_profile"] != body["source_profile"]
        or summary["source_profile"] != "OWNER_DECLARED_BUYS_V1"
        or summary["decision_baseline_pointer_ref"] != body["decision_baseline_pointer_ref"]
        or summary["writer_pointer_ref"] != body["writer_pointer_ref"]
        or summary["writer_record_id"] != body["writer_record_id"]
        or summary["final_record_id"] == summary["writer_record_id"]
        or summary["final_pointer_ref"] != validate_ref(final_pointer_ref)
        or summary["official_valuation"] is not True
        or any(
            type(summary[k]) is not int or summary[k] != 0
            for k in (
                "close_writer_trade_count",
                "close_writer_order_count",
                "close_writer_fill_count",
            )
        )
        or summary["store_plan_ref"] not in body["source_refs"]
    ):
        raise ContractError("DASHBOARD_REGISTERED_SERVING_BINDING_INVALID")
    return value
