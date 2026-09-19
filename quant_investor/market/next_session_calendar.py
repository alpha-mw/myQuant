"""Native provider-validated future Calendar projection for the EOD consumer proof."""

from datetime import datetime, timedelta
import hashlib
from typing import Any, Callable, Mapping

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import _validate_day
from .tushare_calendar_authority import (
    validate_trusted_provider_calendar_capture,
    _capture_projection,
)

HORIZON_NATURAL_DAYS = 21


def build_next_session_projection(
    *,
    eod_trade_date: str,
    captures: Mapping[str, Any],
    capability: Mapping[str, Any],
    docs_raw: bytes,
    raw_resolver: Callable[[Mapping[str, str]], bytes],
) -> dict:
    """Revalidate native raw capture bindings; this alone confers no Morning admission."""
    _validate_day(eod_trade_date)
    if set(captures) != {"SSE", "SZSE"}:
        raise ContractError("NEXT_SESSION_EXCHANGE_SET_INVALID")
    eod = datetime.strptime(eod_trade_date, "%Y%m%d").date()
    horizon = eod + timedelta(days=HORIZON_NATURAL_DAYS)
    dates = [(eod + timedelta(days=i)).isoformat() for i in range(HORIZON_NATURAL_DAYS + 1)]
    by_exchange = {}
    for exchange in ("SSE", "SZSE"):
        raw = raw_resolver(captures[exchange]["payload"]["raw_file_ref"])
        validated = validate_trusted_provider_calendar_capture(
            captures[exchange],
            raw=raw,
            capability=capability,
            docs_raw=docs_raw,
            historical=True,
        )["payload"]
        if (
            validated["exchange_id"] != exchange
            or validated["calendar_authority_conferred"] is not True
        ):
            raise ContractError("NEXT_SESSION_DIRECT_EXCHANGE_REQUIRED")
        if validated["cutoff_date"] != horizon.isoformat():
            raise ContractError("NEXT_SESSION_HORIZON_MISMATCH")
        rows = {row["date"]: row for row in _capture_projection(raw)}
        if any(day not in rows for day in dates):
            raise ContractError("NEXT_SESSION_HORIZON_INCOMPLETE")
        by_exchange[exchange] = [int(rows[day]["status"] == "OPEN") for day in dates]
    if by_exchange["SSE"] != by_exchange["SZSE"]:
        raise ContractError("NEXT_SESSION_EXCHANGE_DISAGREEMENT")
    if by_exchange["SSE"][0] != 1:
        raise ContractError("NEXT_SESSION_EOD_NOT_OPEN")
    later = [day for day, opened in zip(dates[1:], by_exchange["SSE"][1:]) if opened]
    if not later:
        raise ContractError("NEXT_SESSION_NO_LATER_OPEN")
    projection = [
        {"date": day.replace("-", ""), "sse_is_open": sse, "szse_is_open": szse}
        for day, sse, szse in zip(dates, by_exchange["SSE"], by_exchange["SZSE"])
    ]
    return {
        "eod_trade_date": eod_trade_date,
        "observed_through_date": horizon.strftime("%Y%m%d"),
        "next_open_session": later[0].replace("-", ""),
        "projection": projection,
        "projection_sha256": hashlib.sha256(canonical_json_bytes(projection)).hexdigest(),
    }


def inspect_next_session_capture(
    *,
    workspace: str,
    eod_trade_date: str,
    execution: Mapping[str, Any],
    execution_ref: Mapping[str, str],
    success: Mapping[str, Any],
    success_ref: Mapping[str, str],
) -> dict:
    """Replay published native custody at the sole permitted future-capture parent."""
    from pathlib import Path, PurePosixPath
    from quant_investor.contracts import parse_canonical_json_bytes
    from quant_investor.system.store import object_ref_for_artifact
    from .tushare_calendar_authority import (
        validate_published_trusted_provider_calendar_capture_root,
        validate_trusted_provider_calendar_capability,
        validate_calendar_authority_policy,
        validate_trusted_provider_calendar_capture_transaction,
        validate_trusted_provider_calendar_capture_execution,
    )

    _validate_day(eod_trade_date)
    parent = (
        Path(workspace).resolve(strict=True)
        / f"results/operations/daily_production/CN/{eod_trade_date}/calendar-future/captures"
    )
    files = validate_published_trusted_provider_calendar_capture_root(
        capture_parent=parent,
        capture_execution=execution,
        capture_execution_file_ref=execution_ref,
        capture_success=success,
        capture_success_file_ref=success_ref,
    )
    root_name = execution["payload"]["capture_root_name"]

    def ref(name):
        return {
            "relative_path": root_name + "/" + name,
            "byte_sha256": hashlib.sha256(files[name]).hexdigest(),
        }

    captures = {
        exchange: parse_canonical_json_bytes(files[f"capture-{exchange.lower()}.json"])
        for exchange in ("SSE", "SZSE")
    }
    raw_refs = [ref(f"response-{exchange.lower()}.raw") for exchange in ("SSE", "SZSE", "BSE")]
    capture_refs = [ref(f"capture-{exchange.lower()}.json") for exchange in ("SSE", "SZSE", "BSE")]
    raw_refs.sort(key=lambda item: item["relative_path"])
    capture_refs.sort(key=lambda item: item["relative_path"])
    shared = {
        "documentation_raw_file_ref": ref("documentation.raw"),
        "capability_file_ref": ref("capability.json"),
        "policy_file_ref": ref("policy.json"),
        "provider_raw_file_refs": raw_refs,
        "provider_capture_file_refs": capture_refs,
    }
    transaction = validate_trusted_provider_calendar_capture_transaction(
        files["capture-transaction.json"], **shared
    )
    validated_execution = validate_trusted_provider_calendar_capture_execution(
        execution,
        release_install_input_raw=files["release-install-input.json"],
        capture_transaction_file_ref=ref("capture-transaction.json"),
        historical=True,
        **shared,
    )
    capability = validate_trusted_provider_calendar_capability(
        files["capability.json"], docs_raw=files["documentation.raw"], historical=True
    )
    policy = validate_calendar_authority_policy(files["policy.json"])
    if policy["payload"]["provider_capability_ref"] != object_ref_for_artifact(capability):
        raise ContractError("NEXT_SESSION_POLICY_CAPABILITY_MISMATCH")

    def resolve(native_ref):
        name = PurePosixPath(native_ref["relative_path"]).name
        if native_ref != ref(name):
            raise ContractError("NEXT_SESSION_CAPTURE_FILE_REF_MISMATCH")
        return files[name]

    projection = build_next_session_projection(
        eod_trade_date=eod_trade_date,
        captures=captures,
        capability=capability,
        docs_raw=files["documentation.raw"],
        raw_resolver=resolve,
    )
    return {
        **projection,
        "capture_root_ref": {"capture_parent": str(parent), "capture_root_name": root_name},
        "transaction_ref": ref("capture-transaction.json"),
        "execution_ref": dict(execution_ref),
        "success_ref": dict(success_ref),
        "policy_ref": ref("policy.json"),
        "capability_ref": ref("capability.json"),
        "provider_capture_refs": capture_refs,
        "raw_refs": raw_refs,
        "execution": validated_execution,
        "success": dict(success),
        "source_limitations": transaction["payload"]["source_limitations"],
        "calendar_policy": policy,
        "consumer_admission": False,
    }
