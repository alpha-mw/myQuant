"""Pure Morning input consumption from exact prior EOD, owner policy and quote bytes."""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo
import hashlib
import os
import stat

import pandas as pd
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.strategy_records.store import regular_file_sha256
from quant_investor.operations.daily_contract import (
    ContractError,
    utc_stamp,
    EOD_NODE_IDS,
    validate_ref,
)
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.morning_contract import (
    validate_morning_request,
    request_version,
    expected_quote_symbols,
    require_next_open_session,
)
from quant_investor.market.next_session_proof import read_next_session_proof
from quant_investor.intelligence.morning import (
    validate_sina_quote_capture,
    classify_sina_quote_timing,
)
from scripts.daily_completion_replay import replay_native_completion


def _read_morning_evidence(*, workspace: str, request: dict) -> dict:
    """No producer, lock, report or receipt writer is reachable from this function."""
    values = validate_morning_request(request)
    root = Path(workspace).resolve(strict=True)
    reader = SecureSystemStorage(workspace)
    observed = {}

    def read_bytes(ref, *, store=False):
        if store:
            path = root / ref["path"]
            if (
                not ref["path"].startswith(
                    "results/strategy_records/CN/aggressive_tech_manufacturing/"
                )
                or path.resolve(strict=True) != path
            ):
                raise ContractError("MORNING_STORE_SOURCE_PATH_INVALID")
            metadata = path.stat()
            if (
                metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o600, 0o644}
                or metadata.st_size > 16 * 1024 * 1024
            ):
                raise ContractError("MORNING_STORE_SOURCE_UNSAFE")
            digest, _ = regular_file_sha256(path, label="Morning retained Store ledger")
            raw = path.read_bytes()
            if digest != hashlib.sha256(raw).hexdigest():
                raise ContractError("MORNING_STORE_SOURCE_CHANGED")
        else:
            stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
            raw, digest = stored.data, stored.byte_sha256
        if digest != ref["sha256"]:
            raise ContractError("MORNING_SOURCE_SHA_MISMATCH")
        observed[(ref["path"], ref["sha256"], store)] = raw
        return raw

    def read(ref):
        return parse_canonical_json_bytes(read_bytes(ref))

    previous = PurePosixPath(values["previous_completion_ref"]["path"]).parent.name
    recorded = _upstream_read(inspect_recorded_completion, workspace, previous, values)[
        "recorded_completion"
    ]
    calendar = read(recorded["node_terminal_refs"]["calendar"])
    future_ref = calendar["output_refs"].get("next_session_calendar_proof")
    if future_ref is None:
        raise ContractError("CALENDAR_NEXT_SESSION_UNAVAILABLE")
    future = read_next_session_proof(
        workspace=workspace, eod_trade_date=previous, publication_ref=future_ref
    )
    projection = [
        {"date": row["date"], "status": "OPEN" if row["sse_is_open"] else "CLOSED"}
        for row in future["projection"]
    ]
    require_next_open_session(
        runtime_projection=projection, previous_date=previous, run_date=values["run_date"]
    )
    if future["proof"]["next_open_session"] != values["run_date"]:
        raise ContractError("MORNING_NEXT_SESSION_PROOF_MISMATCH")
    quote = read(values["quote_capture_ref"])
    raw = read_bytes(values["quote_raw_ref"])
    quote = validate_sina_quote_capture(quote, raw=raw, run_date=values["run_date"])
    if {k: quote["raw_ref"][k] for k in ("path", "sha256")} != values["quote_raw_ref"]:
        raise ContractError("MORNING_QUOTE_RAW_REF_MISMATCH")
    shanghai = ZoneInfo("Asia/Shanghai")
    if any(
        utc_stamp(quote[k]).astimezone(shanghai).strftime("%Y%m%d") != values["run_date"]
        for k in ("request_time", "response_time")
    ):
        raise ContractError("MORNING_QUOTE_RESPONSE_DATE_MISMATCH")
    native = _upstream_read(replay_native_completion, workspace, previous, values)
    if (
        native.get("native_replay_validated") is not True
        or native.get("validated_nodes") != sorted(EOD_NODE_IDS)
        or native.get("completion_ref") != values["previous_completion_ref"]
        or native.get("trade_date") != previous
    ):
        raise ContractError("MORNING_FULL_NATIVE_REPLAY_REQUIRED")
    store = read(recorded["node_terminal_refs"]["store"])
    ledger = pd.read_parquet(BytesIO(read_bytes(store["output_refs"]["ledger"], store=True)))
    if not {"symbol", "shares"} <= set(ledger.columns):
        raise ContractError("MORNING_STORE_HOLDINGS_INVALID")
    symbols = expected_quote_symbols(
        policy=read(values["owner_policy_ref"]),
        holdings=ledger.to_dict(orient="records"),
        run_date=values["run_date"],
    )
    if [row["symbol"] for row in quote["quote_rows"]] != symbols:
        raise ContractError("MORNING_QUOTE_SCOPE_MISMATCH")
    decision = read(recorded["node_terminal_refs"]["decision"])
    if decision.get("state") != "SUCCEEDED":
        raise ContractError("MORNING_DECISION_CUSTODY_INVALID")
    utc_stamp(decision["finished_at"])

    risk_sources = None
    if request_version(values) == "v3":
        from quant_investor.operations.morning_risk_sources import MorningRiskSources

        risk_sources = MorningRiskSources(
            workspace=workspace,
            recorded=recorded,
            completion_ref=values["previous_completion_ref"],
            policy_refs=values["threshold_policy_refs"],
            quote_requested_at=quote["request_time"],
        )
        if risk_sources.store_outputs["ledger"] != store["output_refs"]["ledger"]:
            raise ContractError("MORNING_THRESHOLD_HOLDINGS_BINDING_DIFFERS")
        if (
            parse_canonical_json_bytes(read_bytes(decision["output_refs"]["result"]))
            != native["decision"]
        ):
            raise ContractError("MORNING_THRESHOLD_DECISION_SOURCE_DIFFERS")

    def recheck():
        if risk_sources is not None:
            risk_sources.recheck()
        if (
            read_next_session_proof(
                workspace=workspace, eod_trade_date=previous, publication_ref=future_ref
            )
            != future
        ):
            raise ContractError("MORNING_CALENDAR_PROOF_CHANGED_DURING_READ")
        for (path, digest, is_store), raw in list(observed.items()):
            if read_bytes({"path": path, "sha256": digest}, store=is_store) != raw:
                raise ContractError("MORNING_SOURCE_CHANGED_DURING_READ")
        if (
            inspect_recorded_completion(
                workspace=workspace,
                trade_date=previous,
                completion_ref=values["previous_completion_ref"],
            )["recorded_completion"]
            != recorded
        ):
            raise ContractError("MORNING_COMPLETION_CHANGED_DURING_READ")

    return {
        "values": values,
        "risk_sources": risk_sources,
        "decision_terminal": decision,
        "recorded": recorded,
        "native": native,
        "future_calendar": future,
        "quote": quote,
        "symbols": symbols,
        "synthetic": recorded["synthetic"] or future["synthetic"],
        "decision_completed_at": decision["finished_at"],
        "recheck": recheck,
    }


def _require_evidence_at(evidence: dict, at: datetime) -> None:
    times = [
        evidence["recorded"]["native_validation_completed_at"],
        evidence["decision_completed_at"],
        evidence["quote"]["request_time"],
        evidence["quote"]["response_time"],
        evidence["future_calendar"]["proof_sealed_at"],
    ]
    if any(utc_stamp(stamp) > at for stamp in times):
        raise ContractError("MORNING_EVIDENCE_NOT_YET_AVAILABLE")


def _require_live_provenance(evidence: dict) -> None:
    native, recorded = evidence["native"], evidence["recorded"]
    ledger = native.get("ledger")
    if (
        evidence["synthetic"]
        or native.get("synthetic") is not False
        or recorded.get("prospective_admission_state") != "LEDGER_ELIGIBLE"
        or type(ledger) is not dict
        or ledger.get("validation_scope") != "NATIVE_LEDGER_DERIVATION"
        or ledger.get("classification") != "CONTEMPORANEOUS"
        or ledger.get("prospective") is not True
        or ledger.get("synthetic") is not False
        or ledger.get("ledger_ref") != recorded.get("prospective_ledger_ref")
    ):
        raise ContractError("MORNING_LIVE_PROVENANCE_REQUIRED")
    validate_ref(ledger["ledger_ref"])
    future = evidence.get("future_calendar")
    if (
        type(future) is not dict
        or future.get("synthetic") is not False
        or future.get("live_eligible") is not True
        or type(future.get("proof")) is not dict
        or future["proof"].get("schema_version") != "cn-next-session-calendar-proof.v2"
    ):
        raise ContractError("MORNING_PRODUCTION_CALENDAR_REQUIRED")


def prepare_morning_consumer(*, workspace: str, request: dict) -> dict:
    evidence = _read_morning_evidence(workspace=workspace, request=request)
    values = evidence["values"]
    now = datetime.now(timezone.utc)
    _require_evidence_at(evidence, now)
    shanghai = ZoneInfo("Asia/Shanghai")
    if values["action"] != "REPLAY":
        _require_live_provenance(evidence)
        if now.astimezone(shanghai).strftime("%Y%m%d") != values["run_date"]:
            raise ContractError("MORNING_LIVE_DATE_MISMATCH")
        if values["action"] == "SEAL":
            raise ContractError("MORNING_V2_LIVE_ADMISSION_NOT_READY")
    evidence["recheck"]()
    stamp = datetime.now(timezone.utc)
    if stamp < now:
        raise ContractError("MORNING_CONSUMER_CLOCK_REGRESSED")
    if (
        values["action"] != "REPLAY"
        and stamp.astimezone(shanghai).strftime("%Y%m%d") != values["run_date"]
    ):
        raise ContractError("MORNING_LIVE_DATE_MISMATCH")
    previous = PurePosixPath(values["previous_completion_ref"]["path"]).parent.name
    symbols, quote, native, synthetic = (
        evidence["symbols"],
        evidence["quote"],
        evidence["native"],
        evidence["synthetic"],
    )
    version = request_version(values)
    result = {
        "schema_version": "morning-strategy-replay." + version,
        "command_status": (
            "REPLAY_VERIFIED" if values["action"] == "REPLAY" else "PREFLIGHT_COMPLETE"
        ),
        "admission": "RESEARCH_ONLY" if values["action"] == "REPLAY" else "LIVE_RESEARCH_CONSUMER",
        "run_date": values["run_date"],
        "previous_trade_date": previous,
        "previous_completion_ref": values["previous_completion_ref"],
        "expected_symbols": symbols,
        "quote_capture_ref": values["quote_capture_ref"],
        "quote_raw_ref": values["quote_raw_ref"],
        "owner_policy_ref": values["owner_policy_ref"],
        "quote_rows": quote["quote_rows"],
        "quote_timing": classify_sina_quote_timing(
            quote["request_time"], run_date=values["run_date"]
        ),
        "validated_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic": synthetic,
        "prospective_admission_state": "NOT_CLAIMED",
        "decision": native["decision"],
        "authority": FALSE_AUTHORITY,
    }

    if version == "v3":
        from quant_investor.operations.morning_report import render_morning_report

        result.update(morning_threshold_fields(evidence))
        result["report_markdown"] = render_morning_report(result).decode("utf-8")
        evidence["recheck"]()
        final_stamp = datetime.now(timezone.utc)
        if final_stamp < stamp:
            raise ContractError("MORNING_CONSUMER_CLOCK_REGRESSED")
        if (
            values["action"] != "REPLAY"
            and final_stamp.astimezone(shanghai).strftime("%Y%m%d") != values["run_date"]
        ):
            raise ContractError("MORNING_LIVE_DATE_MISMATCH")
        result["validated_at"] = final_stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _upstream_read(function, workspace, previous, values):
    try:
        return function(
            workspace=workspace,
            trade_date=previous,
            completion_ref=values["previous_completion_ref"],
        )
    except (ValueError, OSError, RuntimeError) as exc:
        if request_version(values) != "v3":
            raise
        error = ContractError("MORNING_UPSTREAM_DAG_INCOMPLETE")
        error.failed_node_ids = []
        try:
            from quant_investor.operations.daily_status import read_daily_status

            status = read_daily_status(workspace, previous)
            error.failed_node_ids = sorted(
                name
                for name, row in status["nodes"].items()
                if name in EOD_NODE_IDS and row.get("state") not in {"SUCCEEDED"}
            )
        except (ValueError, OSError, RuntimeError, KeyError):
            pass
        raise error from exc


def morning_threshold_fields(evidence):
    """Shared live/replay/history projection from already read frozen evidence."""
    from quant_investor.intelligence.morning_threshold_review import build_threshold_review

    values, quote = evidence["values"], evidence["quote"]
    mode = "REPLAY_ONLY"
    try:
        _require_live_provenance(evidence)
        if utc_stamp(evidence["recorded"]["native_validation_completed_at"]) <= utc_stamp(
            quote["request_time"]
        ):
            mode = "PRIOR_EOD_BEFORE_QUOTE"
    except ContractError:
        pass
    return {
        "threshold_policy_refs": values["threshold_policy_refs"],
        "threshold_review": build_threshold_review(
            sources=evidence["risk_sources"],
            quote=quote,
            quote_capture_ref=values["quote_capture_ref"],
            quote_raw_ref=values["quote_raw_ref"],
            decision_result_ref=evidence["decision_terminal"]["output_refs"]["result"],
            evidence_mode=mode,
            synthetic=evidence["synthetic"],
        ),
    }
