"""Deterministic morning-strategy evidence and cutover evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final, Mapping
from zoneinfo import ZoneInfo

from quant_investor.cli.input import read_exact_request
from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_authority import verify_factor_production
from quant_investor.factors.production_observation import (
    validate_factor_production_observation,
)
from quant_investor.strategy_records import load_registered_catalog

from ._common import IntelligenceError, validate_stable_artifact

MORNING_RECEIPT_SCHEMA: Final = "morning-strategy-run.v1"
CUTOVER_RECEIPT_SCHEMA: Final = "morning-strategy-cutover.v1"
EOD_EVALUATION_SCHEMA: Final = "morning-strategy-eod-evaluation.v1"
SINA_CAPTURE_SCHEMA: Final = "cn-public-quote-capture.v1"
STORE_POINTER_RELATIVE: Final = (
    "results/strategy_records/CN/aggressive_tech_manufacturing/" "_record_store/current.v1.json"
)
STORE_ROOT_RELATIVE: Final = "results/strategy_records/CN/aggressive_tech_manufacturing"
MARKET_POINTER_RELATIVE: Final = "data/parquet/cn/_latest.json"
_DATE_RE: Final = re.compile(r"^[0-9]{8}$")
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_COMPANY_RE: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_SHANGHAI: Final = ZoneInfo("Asia/Shanghai")


def _date(value: Any, *, label: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise IntelligenceError(f"{label} must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise IntelligenceError(f"{label} is invalid") from exc
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise IntelligenceError(f"{label} must be lowercase SHA-256")
    return value


def _workspace(value: str | os.PathLike[str]) -> Path:
    try:
        root = Path(value).resolve(strict=True)
        observed = os.lstat(root)
    except OSError as exc:
        raise IntelligenceError("workspace root is invalid") from exc
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise IntelligenceError("workspace root is invalid")
    return root


def _factor_closure_ready(
    factor: Mapping[str, Any], *, expected_date: str, expected_pointer_sha256: str
) -> bool:
    """Apply the direct Factor-store verification contract.

    ``verify_factor_production`` returns the verified store projection itself.
    The public CLI adds its own ``verified`` convenience field later, so the
    intelligence layer must not require that CLI-only projection key.
    """

    return (
        factor.get("factor_authority") == "ACTIVE"
        and factor.get("factor_readiness") == "READY"
        and factor.get("blockers") == []
        and factor.get("as_of") == expected_date
        and factor.get("factor_pointer_byte_sha256") == expected_pointer_sha256
    )


def _relative(value: Any, *, label: str) -> PurePosixPath:
    if type(value) is not str:
        raise IntelligenceError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise IntelligenceError(f"{label} path is invalid")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise IntelligenceError(f"{label} path is invalid") from exc
    return path


def _stable_raw(root: Path, path_value: Any, expected_sha: Any, *, label: str) -> bytes:
    relative = _relative(path_value, label=label)
    expected = _sha(expected_sha, label=f"{label} SHA")
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise IntelligenceError(f"{label} is unavailable") from exc
    if resolved != root and root not in resolved.parents:
        raise IntelligenceError(f"{label} escapes workspace")
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) & 0o022
    ):
        raise IntelligenceError(f"{label} is unsafe")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second or hashlib.sha256(first).hexdigest() != expected:
        raise IntelligenceError(f"{label} SHA differs")
    return first


def _json_ref(root: Path, path_value: Any, expected_sha: Any, *, label: str) -> dict[str, Any]:
    raw, value = read_exact_request(root, str(path_value), str(expected_sha))
    if hashlib.sha256(raw).hexdigest() != expected_sha or type(value) is not dict:
        raise IntelligenceError(f"{label} is invalid")
    return value


def _absolute_json(path_value: Any, expected_sha: Any, *, label: str) -> dict[str, Any]:
    if type(path_value) is not str or not Path(path_value).is_absolute():
        raise IntelligenceError(f"{label} absolute path is invalid")
    path = Path(path_value)
    expected = _sha(expected_sha, label=f"{label} SHA")
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) & 0o022
    ):
        raise IntelligenceError(f"{label} is unsafe")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second or hashlib.sha256(first).hexdigest() != expected:
        raise IntelligenceError(f"{label} SHA differs")
    try:
        value = json.loads(first)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntelligenceError(f"{label} is invalid JSON") from exc
    if type(value) is not dict:
        raise IntelligenceError(f"{label} must be an object")
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise IntelligenceError(f"{label} must be UTC-second timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise IntelligenceError(f"{label} is invalid") from exc
    return parsed


def validate_sina_quote_capture(
    document: Mapping[str, Any],
    *,
    raw: bytes,
    run_date: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "provider",
        "request_time",
        "response_time",
        "encoding",
        "raw_ref",
        "field_definitions",
        "symbol_mapping",
        "quote_rows",
        "reasonable",
        "broker",
        "order",
        "execution",
    }
    value = dict(document)
    if set(value) != required or value.get("schema_version") != SINA_CAPTURE_SCHEMA:
        raise IntelligenceError("Sina quote capture shape is invalid")
    if (
        value.get("provider") != "SINA"
        or value.get("encoding") != "GB18030"
        or value.get("reasonable") is not True
        or any(value.get(field) is not False for field in ("broker", "order", "execution"))
    ):
        raise IntelligenceError("Sina quote capture policy is invalid")
    if value.get("field_definitions") != {
        "amount": "provider cumulative turnover CNY",
        "price": "provider current price CNY",
        "volume": "provider cumulative shares",
    }:
        raise IntelligenceError("Sina quote field definitions differ")
    raw_ref = value.get("raw_ref")
    if (
        type(raw_ref) is not dict
        or set(raw_ref) != {"path", "sha256", "size"}
        or raw_ref.get("sha256") != hashlib.sha256(raw).hexdigest()
        or raw_ref.get("size") != len(raw)
    ):
        raise IntelligenceError("Sina raw quote binding differs")
    request_time = _timestamp(value.get("request_time"), label="quote request_time")
    response_time = _timestamp(value.get("response_time"), label="quote response_time")
    if response_time < request_time or (response_time - request_time).total_seconds() > 120:
        raise IntelligenceError("Sina quote response chronology is invalid")
    local = request_time.astimezone(_SHANGHAI)
    if local.strftime("%Y%m%d") != run_date or not (
        (local.hour, local.minute) >= (9, 30) and (local.hour, local.minute) <= (9, 46)
    ):
        raise IntelligenceError("Sina quote request is outside the 09:30-09:46 window")
    mappings = value.get("symbol_mapping")
    rows = value.get("quote_rows")
    if type(mappings) is not list or not mappings or type(rows) is not list or not rows:
        raise IntelligenceError("Sina quote capture is empty")
    expected_symbols: list[str] = []
    for row in mappings:
        if (
            type(row) is not dict
            or set(row) != {"provider_symbol", "symbol"}
            or type(row.get("symbol")) is not str
            or _COMPANY_RE.fullmatch(row["symbol"]) is None
            or type(row.get("provider_symbol")) is not str
        ):
            raise IntelligenceError("Sina symbol mapping is invalid")
        expected_symbols.append(row["symbol"])
    if expected_symbols != sorted(set(expected_symbols), key=lambda item: item.encode("ascii")):
        raise IntelligenceError("Sina symbol mapping must be unique and sorted")
    observed_symbols = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "amount",
            "high",
            "low",
            "name",
            "open",
            "previous_close",
            "price",
            "provider_date",
            "provider_time",
            "symbol",
            "volume",
        }:
            raise IntelligenceError("Sina quote row shape is invalid")
        symbol = row.get("symbol")
        if symbol not in expected_symbols:
            raise IntelligenceError("Sina quote row symbol differs")
        observed_symbols.append(symbol)
        for field in (
            "amount",
            "high",
            "low",
            "open",
            "previous_close",
            "price",
            "volume",
        ):
            try:
                number = float(str(row.get(field)))
            except (TypeError, ValueError) as exc:
                raise IntelligenceError("Sina numeric quote field is invalid") from exc
            if number < 0:
                raise IntelligenceError("Sina numeric quote field is negative")
        high = float(str(row["high"]))
        low = float(str(row["low"]))
        price = float(str(row["price"]))
        if high < low or (price > 0 and not low <= price <= high):
            raise IntelligenceError("Sina quote price range is inconsistent")
        if row.get("provider_date", "").replace("-", "") != run_date:
            raise IntelligenceError("Sina quote provider date differs")
        try:
            provider_time = datetime.strptime(str(row.get("provider_time")), "%H:%M:%S")
        except ValueError as exc:
            raise IntelligenceError("Sina quote provider time is invalid") from exc
        if not (provider_time.hour == 9 and 30 <= provider_time.minute <= 46):
            raise IntelligenceError("Sina quote provider time is outside the morning window")
    if observed_symbols != expected_symbols:
        raise IntelligenceError("Sina quote rows do not close the requested symbols")
    return value


def _observation(
    root: Path,
    *,
    path: Any,
    sha256: Any,
    alias: str,
    previous_trade_date: str,
) -> dict[str, Any]:
    value = _json_ref(root, path, sha256, label=f"{alias} observation")
    validated = validate_factor_production_observation(value)
    payload = validated["payload"]
    if (
        payload.get("factor_alias") != alias
        or payload.get("signal_date") != previous_trade_date
        or payload.get("state") != "OPEN"
        or payload.get("authority") != "NON_AUTHORIZING"
        or payload.get("planned_horizons") != [1, 5, 20, 60]
    ):
        raise IntelligenceError(f"{alias} observation state differs")
    return validated


def _morning_input_state(
    *,
    workspace_root: str | os.PathLike[str],
    request: Mapping[str, Any],
    now: datetime | None,
) -> dict[str, Any]:
    required = {
        "action",
        "automation_id",
        "run_date",
        "previous_trade_date",
        "expected_factor_pointer_sha256",
        "low_observation_path",
        "low_observation_sha256",
        "w80_observation_path",
        "w80_observation_sha256",
        "expected_store_pointer_sha256",
        "quote_capture_path",
        "quote_capture_sha256",
        "pool_manifest_path",
        "pool_manifest_sha256",
        "output_path",
        "output_sha256",
    }
    values = dict(request)
    if set(values) != required or values.get("action") not in {"PREFLIGHT", "SEAL"}:
        raise IntelligenceError("morning strategy request shape is invalid")
    if values.get("automation_id") != "automation":
        raise IntelligenceError("morning strategy automation identity differs")
    run_date = _date(values["run_date"], label="run_date")
    previous = _date(values["previous_trade_date"], label="previous_trade_date")
    if previous >= run_date:
        raise IntelligenceError("morning strategy previous date is invalid")
    observed_now = now or datetime.now(tz=_SHANGHAI)
    if observed_now.astimezone(_SHANGHAI).strftime("%Y%m%d") != run_date:
        raise IntelligenceError("morning strategy run date is not current local date")
    root = _workspace(workspace_root)
    factor_pointer_sha = _sha(
        values["expected_factor_pointer_sha256"],
        label="expected Factor pointer",
    )
    factor = verify_factor_production(root)
    core_blockers: list[str] = []
    auxiliary_blockers: list[str] = []
    if not _factor_closure_ready(
        factor,
        expected_date=previous,
        expected_pointer_sha256=factor_pointer_sha,
    ):
        core_blockers.append("FACTOR_NOT_READY_FOR_PREVIOUS_TRADE_DATE")
    low = _observation(
        root,
        path=values["low_observation_path"],
        sha256=values["low_observation_sha256"],
        alias="LOW",
        previous_trade_date=previous,
    )
    w80 = _observation(
        root,
        path=values["w80_observation_path"],
        sha256=values["w80_observation_sha256"],
        alias="W80",
        previous_trade_date=previous,
    )

    expected_store_sha = _sha(
        values["expected_store_pointer_sha256"],
        label="expected Store pointer",
    )
    store_raw = _stable_raw(
        root,
        STORE_POINTER_RELATIVE,
        expected_store_sha,
        label="Store pointer",
    )
    loaded = load_registered_catalog(root / STORE_ROOT_RELATIVE)
    if loaded is None:
        core_blockers.append("STORE_UNREGISTERED")
        store_pointer: dict[str, Any] = {}
    else:
        store_pointer, _catalog = loaded
        if canonical_json_bytes(store_pointer) != store_raw.rstrip(b"\n"):
            # Store uses newline-terminated canonical JSON; byte SHA above remains authority.
            try:
                parsed_pointer = json.loads(store_raw)
            except json.JSONDecodeError:
                parsed_pointer = None
            if parsed_pointer != store_pointer:
                core_blockers.append("STORE_POINTER_READBACK_DIFFERS")
        if not isinstance(store_pointer.get("active_closure"), dict):
            core_blockers.append("STORE_HOLDINGS_UNAVAILABLE")

    capture = _json_ref(
        root,
        values["quote_capture_path"],
        values["quote_capture_sha256"],
        label="quote capture",
    )
    raw_ref = capture.get("raw_ref") if isinstance(capture, dict) else None
    if not isinstance(raw_ref, dict):
        raise IntelligenceError("quote capture raw ref is invalid")
    raw = _stable_raw(
        root,
        raw_ref.get("path"),
        raw_ref.get("sha256"),
        label="quote raw response",
    )
    quote = validate_sina_quote_capture(capture, raw=raw, run_date=run_date)

    pool_path = values["pool_manifest_path"]
    pool_sha = values["pool_manifest_sha256"]
    pool_ref = None
    if pool_path is None and pool_sha is None:
        auxiliary_blockers.append("TOP100_UNAVAILABLE")
    elif type(pool_path) is str and type(pool_sha) is str:
        pool = validate_stable_artifact(
            _json_ref(root, pool_path, pool_sha, label="Top100 manifest"),
            expected_kind="daily_research_pool_manifest",
        )
        if pool["payload"].get("signal_date") != previous:
            auxiliary_blockers.append("TOP100_STALE")
        else:
            pool_ref = {"path": pool_path, "sha256": pool_sha}
    else:
        raise IntelligenceError("Top100 manifest arguments are inconsistent")
    return {
        "action": values["action"],
        "run_date": run_date,
        "previous_trade_date": previous,
        "factor": factor,
        "factor_pointer_sha256": factor_pointer_sha,
        "low_observation": low,
        "low_observation_sha256": values["low_observation_sha256"],
        "w80_observation": w80,
        "w80_observation_sha256": values["w80_observation_sha256"],
        "store_pointer": store_pointer,
        "store_pointer_sha256": expected_store_sha,
        "quote": quote,
        "quote_capture_ref": {
            "path": values["quote_capture_path"],
            "sha256": values["quote_capture_sha256"],
        },
        "pool_ref": pool_ref,
        "core_blockers": sorted(set(core_blockers)),
        "auxiliary_blockers": sorted(set(auxiliary_blockers)),
        "output_path": values["output_path"],
        "output_sha256": values["output_sha256"],
        "workspace_root": root,
    }


def _owner_directory(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        observed = os.lstat(path)
    except OSError as exc:
        raise IntelligenceError("morning strategy directory is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise IntelligenceError("morning strategy directory is unsafe")
    return path


def _write_exact(path: Path, raw: bytes) -> tuple[str, bool]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        observed = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise IntelligenceError("morning strategy immutable path is unsafe")
        existing = path.read_bytes()
        if existing != raw:
            raise IntelligenceError("morning strategy immutable conflict")
        return hashlib.sha256(existing).hexdigest(), False
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest(), True


def run_morning_strategy(
    *,
    workspace_root: str | os.PathLike[str],
    request: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate one morning input closure and optionally seal its machine receipt."""

    state = _morning_input_state(workspace_root=workspace_root, request=request, now=now)
    status = (
        "BLOCKED"
        if state["core_blockers"]
        else "PARTIAL" if state["auxiliary_blockers"] else "COMPLETE"
    )
    result = {
        "status": status,
        "run_date": state["run_date"],
        "previous_trade_date": state["previous_trade_date"],
        "factor_status": (
            "READY"
            if not any(blocker.startswith("FACTOR_") for blocker in state["core_blockers"])
            else "BLOCKED"
        ),
        "holdings_status": (
            "AVAILABLE"
            if not any(blocker.startswith("STORE_") for blocker in state["core_blockers"])
            else "UNAVAILABLE"
        ),
        "quote_status": "READY",
        "core_blockers": state["core_blockers"],
        "auxiliary_blockers": state["auxiliary_blockers"],
        "broker": False,
        "live_order": False,
        "actual_holdings_mutation": False,
    }
    if state["action"] == "PREFLIGHT":
        if state["output_path"] is not None or state["output_sha256"] is not None:
            raise IntelligenceError("PREFLIGHT must not bind an output")
        return {"command_status": "PREFLIGHT_COMPLETE", **result}

    run_date = state["run_date"]
    expected_output = f"results/operations/morning_strategy/CN/{run_date}/0945-strategy.md"
    if state["output_path"] != expected_output:
        raise IntelligenceError("morning strategy output path is not deterministic")
    output_sha = _sha(state["output_sha256"], label="morning strategy output SHA")
    output_raw = _stable_raw(
        state["workspace_root"],
        expected_output,
        output_sha,
        label="morning strategy output",
    )
    if not output_raw.strip():
        raise IntelligenceError("morning strategy output is empty")
    for declaration in (
        b"research_only=true",
        b"broker=false",
        b"live_order=false",
        b"actual_holdings_mutation=false",
    ):
        if declaration not in output_raw:
            raise IntelligenceError("morning strategy authority declaration is missing")
    quote = state["quote"]
    receipt = {
        "schema_version": MORNING_RECEIPT_SCHEMA,
        "run_date": run_date,
        "previous_trade_date": state["previous_trade_date"],
        "automation_id": "automation",
        "run_mode": "MORNING_STRATEGY",
        "status": status,
        "factor_pointer_sha256": state["factor_pointer_sha256"],
        "low_observation_sha256": state["low_observation_sha256"],
        "w80_observation_sha256": state["w80_observation_sha256"],
        "store_pointer_sha256": state["store_pointer_sha256"],
        "quote_provider": "SINA",
        "quote_capture_ref": state["quote_capture_ref"],
        "quote_request_time": quote["request_time"],
        "quote_response_time": quote["response_time"],
        "quote_raw_sha256": quote["raw_ref"]["sha256"],
        "pool_manifest_ref": state["pool_ref"],
        "core_blockers": state["core_blockers"],
        "auxiliary_blockers": state["auxiliary_blockers"],
        "output_path": expected_output,
        "output_sha256": output_sha,
        "broker": False,
        "live_order": False,
        "live_execution": False,
        "actual_holdings_mutation": False,
    }
    receipt_root = _owner_directory(
        state["workspace_root"] / f"results/operations/morning_strategy/CN/{run_date}"
    )
    receipt_path = receipt_root / "0945-run.v1.json"
    raw = canonical_json_bytes(receipt)
    digest, created = _write_exact(receipt_path, raw)
    return {
        "command_status": "PUBLISHED" if created else "NO_ACTION",
        **result,
        "receipt_path": str(receipt_path.relative_to(state["workspace_root"])),
        "receipt_sha256": digest,
    }


def _morning_receipt_success(value: Mapping[str, Any]) -> bool:
    run_date = value.get("run_date")
    previous = value.get("previous_trade_date")
    dates_valid = (
        type(run_date) is str
        and type(previous) is str
        and _DATE_RE.fullmatch(run_date) is not None
        and _DATE_RE.fullmatch(previous) is not None
        and previous < run_date
    )
    return (
        dates_valid
        and value.get("schema_version") == MORNING_RECEIPT_SCHEMA
        and value.get("status") in {"COMPLETE", "PARTIAL"}
        and value.get("core_blockers") == []
        and value.get("quote_provider") == "SINA"
        and type(value.get("quote_capture_ref")) is dict
        and set(value["quote_capture_ref"]) == {"path", "sha256"}
        and type(value["quote_capture_ref"].get("path")) is str
        and isinstance(value["quote_capture_ref"].get("sha256"), str)
        and _SHA_RE.fullmatch(value["quote_capture_ref"]["sha256"]) is not None
        and isinstance(value.get("quote_raw_sha256"), str)
        and _SHA_RE.fullmatch(value["quote_raw_sha256"]) is not None
        and value.get("broker") is False
        and value.get("live_order") is False
        and value.get("live_execution") is False
        and value.get("actual_holdings_mutation") is False
    )


def _return_text(close: Any, reference: Any, *, label: str) -> str:
    try:
        close_value = Decimal(str(close))
        reference_value = Decimal(str(reference))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise IntelligenceError(f"{label} price is invalid") from exc
    if not close_value.is_finite() or not reference_value.is_finite():
        raise IntelligenceError(f"{label} price is invalid")
    if close_value <= 0 or reference_value <= 0:
        raise IntelligenceError(f"{label} price must be positive")
    value = (close_value / reference_value - Decimal("1")).quantize(Decimal("0.000000000001"))
    return format(value, "f")


def evaluate_morning_strategy_eod(
    *,
    workspace_root: str | os.PathLike[str],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one successful 09:45 run against the same-day strict close.

    This inactive research evaluator creates no portfolio, order, fill or
    holdings state.  It binds the exact morning receipt, public quote capture,
    and canonical Market pointer before sealing one date-bound outcome.
    """

    required = {
        "action",
        "run_date",
        "morning_receipt_path",
        "morning_receipt_sha256",
        "quote_capture_path",
        "quote_capture_sha256",
        "expected_market_pointer_sha256",
        "benchmark_symbol",
        "output_path",
        "output_sha256",
    }
    values = dict(request)
    if set(values) != required:
        raise IntelligenceError("morning EOD evaluation request shape is invalid")
    action = values["action"]
    if action not in {"PREFLIGHT", "SEAL"}:
        raise IntelligenceError("morning EOD evaluation action is invalid")
    root = _workspace(workspace_root)
    run_date = _date(values["run_date"], label="run_date")
    morning = _json_ref(
        root,
        values["morning_receipt_path"],
        values["morning_receipt_sha256"],
        label="morning receipt",
    )
    if not _morning_receipt_success(morning) or morning.get("run_date") != run_date:
        raise IntelligenceError("morning receipt is not a successful same-day run")

    quote_raw = _stable_raw(
        root,
        values["quote_capture_path"],
        values["quote_capture_sha256"],
        label="quote capture",
    )
    try:
        quote_document = json.loads(quote_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntelligenceError("quote capture is invalid JSON") from exc
    if type(quote_document) is not dict:
        raise IntelligenceError("quote capture must be an object")
    raw_ref = quote_document.get("raw_ref")
    if type(raw_ref) is not dict:
        raise IntelligenceError("quote capture raw ref is invalid")
    provider_raw = _stable_raw(
        root,
        raw_ref.get("path"),
        raw_ref.get("sha256"),
        label="raw quote response",
    )
    quote = validate_sina_quote_capture(quote_document, raw=provider_raw, run_date=run_date)
    if morning.get("quote_raw_sha256") != raw_ref.get("sha256"):
        raise IntelligenceError("morning receipt quote binding differs")
    if morning.get("quote_capture_ref") != {
        "path": values["quote_capture_path"],
        "sha256": values["quote_capture_sha256"],
    }:
        raise IntelligenceError("morning receipt quote capture binding differs")

    expected_market_sha = _sha(
        values["expected_market_pointer_sha256"],
        label="expected Market pointer",
    )
    _stable_raw(root, MARKET_POINTER_RELATIVE, expected_market_sha, label="Market pointer")
    from quant_investor.market.market_data_reader import MarketDataReader

    reader = MarketDataReader(market="CN", data_root=root / "data", mode_policy="strict")
    gate = reader.clean_snapshot_gate(refresh=True)
    if gate.get("healthy") is not True or gate.get("latest_complete_trade_date") != run_date:
        raise IntelligenceError("strict Market close is not healthy for run_date")
    quote_rows = quote["quote_rows"]
    symbols = [row["symbol"] for row in quote_rows]
    close_frame = reader.read_cross_section(
        run_date,
        columns=["ts_code", "trade_date", "close"],
    )
    if "symbol" not in close_frame.columns or "close" not in close_frame.columns:
        raise IntelligenceError("strict Market close columns are unavailable")
    close_by_symbol = {
        str(row.symbol): row.close
        for row in close_frame.loc[close_frame["symbol"].isin(symbols)].itertuples()
    }
    _stable_raw(root, MARKET_POINTER_RELATIVE, expected_market_sha, label="Market pointer")

    outcomes: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for quote_row in quote_rows:
        symbol = quote_row["symbol"]
        close = close_by_symbol.get(symbol)
        if close is None:
            unavailable.append(symbol)
            outcomes.append(
                {
                    "symbol": symbol,
                    "state": "CLOSE_UNAVAILABLE",
                    "quote_0945": str(quote_row["price"]),
                    "close": None,
                    "return_0945_to_close": None,
                }
            )
            continue
        try:
            observed_return = _return_text(
                close,
                quote_row["price"],
                label=symbol,
            )
        except IntelligenceError:
            unavailable.append(symbol)
            outcomes.append(
                {
                    "symbol": symbol,
                    "state": "RETURN_UNAVAILABLE",
                    "quote_0945": str(quote_row["price"]),
                    "close": str(close),
                    "return_0945_to_close": None,
                }
            )
            continue
        outcomes.append(
            {
                "symbol": symbol,
                "state": "OBSERVED",
                "quote_0945": str(quote_row["price"]),
                "close": str(close),
                "return_0945_to_close": observed_return,
            }
        )

    benchmark_symbol = values["benchmark_symbol"]
    if benchmark_symbol is not None and (
        type(benchmark_symbol) is not str or _COMPANY_RE.fullmatch(benchmark_symbol) is None
    ):
        raise IntelligenceError("benchmark_symbol is invalid")
    benchmark_row = next(
        (row for row in outcomes if row["symbol"] == benchmark_symbol),
        None,
    )
    benchmark_return = (
        None
        if benchmark_row is None or benchmark_row["state"] != "OBSERVED"
        else benchmark_row["return_0945_to_close"]
    )
    relative_rows = []
    if benchmark_return is not None:
        benchmark_value = Decimal(benchmark_return)
        relative_rows = [
            {
                **row,
                "benchmark_relative_return": (
                    None
                    if row["return_0945_to_close"] is None
                    else format(
                        (Decimal(row["return_0945_to_close"]) - benchmark_value).quantize(
                            Decimal("0.000000000001")
                        ),
                        "f",
                    )
                ),
            }
            for row in outcomes
        ]
    unavailable_states = {
        row["symbol"]: row["state"] for row in outcomes if row["symbol"] in unavailable
    }
    auxiliary_blockers = [f"{unavailable_states[symbol]}:{symbol}" for symbol in unavailable]
    if benchmark_return is None:
        auxiliary_blockers.append("BENCHMARK_UNAVAILABLE")
    decision_quality = (
        "PARTIAL_AUXILIARY" if morning.get("status") == "PARTIAL" or auxiliary_blockers else "READY"
    )
    status = "PARTIAL" if decision_quality == "PARTIAL_AUXILIARY" else "COMPLETE"
    result = {
        "schema_version": EOD_EVALUATION_SCHEMA,
        "run_date": run_date,
        "strategy_run_id": values["morning_receipt_sha256"],
        "operational_success": True,
        "decision_quality": decision_quality,
        "status": status,
        "morning_receipt_ref": {
            "path": values["morning_receipt_path"],
            "sha256": values["morning_receipt_sha256"],
        },
        "quote_capture_ref": {
            "path": values["quote_capture_path"],
            "sha256": values["quote_capture_sha256"],
        },
        "market_pointer_sha256": expected_market_sha,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_return_0945_to_close": benchmark_return,
        "instrument_outcomes": relative_rows or outcomes,
        "auxiliary_blockers": sorted(set(auxiliary_blockers)),
        "broker": False,
        "live_order": False,
        "live_execution": False,
        "paper_fill": False,
        "actual_holdings_mutation": False,
    }
    if action == "PREFLIGHT":
        if values["output_path"] is not None or values["output_sha256"] is not None:
            raise IntelligenceError("PREFLIGHT must not bind an output")
        return {"command_status": "PREFLIGHT_COMPLETE", **result}

    expected_output = f"results/operations/morning_strategy/CN/{run_date}/eod-evaluation.md"
    if values["output_path"] != expected_output:
        raise IntelligenceError("morning EOD output path is not deterministic")
    output_sha = _sha(values["output_sha256"], label="morning EOD output SHA")
    output_raw = _stable_raw(root, expected_output, output_sha, label="morning EOD output")
    for declaration in (
        b"research_only=true",
        b"broker=false",
        b"live_order=false",
        b"actual_holdings_mutation=false",
    ):
        if declaration not in output_raw:
            raise IntelligenceError("morning EOD authority declaration is missing")
    result["output_path"] = expected_output
    result["output_sha256"] = output_sha
    receipt_root = _owner_directory(root / f"results/operations/morning_strategy/CN/{run_date}")
    receipt_path = receipt_root / "eod-evaluation.v1.json"
    digest, created = _write_exact(receipt_path, canonical_json_bytes(result))
    return {
        "command_status": "PUBLISHED" if created else "NO_ACTION",
        **result,
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_sha256": digest,
    }


def evaluate_morning_cutover(
    *,
    workspace_root: str | os.PathLike[str],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one 20:20 core/auxiliary cutover decision from exact refs."""

    required = {
        "target_date",
        "maintenance_receipt_path",
        "maintenance_receipt_sha256",
        "calendar_success_path",
        "calendar_success_sha256",
        "factor_rollover_status",
        "expected_factor_pointer_sha256",
        "low_observation_path",
        "low_observation_sha256",
        "w80_observation_path",
        "w80_observation_sha256",
        "expected_store_pointer_sha256",
        "expected_import_root",
        "scheduler_origin_verified",
        "current_schedule_state",
        "morning_receipts",
        "auxiliary_blockers",
    }
    values = dict(request)
    if set(values) != required:
        raise IntelligenceError("morning cutover request shape is invalid")
    target = _date(values["target_date"], label="target_date")
    root = _workspace(workspace_root)
    core_blockers: list[str] = []
    holdings_blockers: list[str] = []
    auxiliary = values["auxiliary_blockers"]
    if type(auxiliary) is not list or any(type(item) is not str or not item for item in auxiliary):
        raise IntelligenceError("auxiliary_blockers must be text list")
    auxiliary_blockers = list(auxiliary)

    maintenance = _json_ref(
        root,
        values["maintenance_receipt_path"],
        values["maintenance_receipt_sha256"],
        label="maintenance receipt",
    )
    if (
        maintenance.get("mode") != "execute"
        or maintenance.get("attempt_slot") != "2020"
        or maintenance.get("target_date") != target
        or maintenance.get("factor_input_readiness") != "READY"
        or maintenance.get("core_blockers") != []
    ):
        core_blockers.append("MAINTENANCE_CORE_NOT_READY")
    for blocker in maintenance.get("macro_blockers", []):
        auxiliary_blockers.append(f"MACRO:{blocker}")
    if maintenance.get("fundamental_integrity_status") != "READY":
        auxiliary_blockers.append("FUNDAMENTAL_PARTIAL")

    calendar = validate_stable_artifact(
        _absolute_json(
            values["calendar_success_path"],
            values["calendar_success_sha256"],
            label="Calendar capture success",
        ),
        expected_kind="system.trusted_provider_calendar_capture_success",
    )
    if (
        calendar.get("kind") != "system.trusted_provider_calendar_capture_success"
        or not isinstance(calendar.get("payload"), dict)
        or calendar["payload"].get("state") != "COMPLETE"
    ):
        core_blockers.append("CALENDAR_CAPTURE_NOT_COMPLETE")

    if values["factor_rollover_status"] not in {
        "ACTIVATED",
        "NO_ACTION",
        "RECOVERED",
        "IDEMPOTENT",
        "ROLLOVER_ACTIVATED",
    }:
        core_blockers.append("FACTOR_ROLLOVER_NOT_ALLOWED")
    expected_factor_sha = _sha(
        values["expected_factor_pointer_sha256"],
        label="expected Factor pointer",
    )
    factor = verify_factor_production(root)
    if not _factor_closure_ready(
        factor,
        expected_date=target,
        expected_pointer_sha256=expected_factor_sha,
    ):
        core_blockers.append("FACTOR_VERIFY_NOT_TARGET_READY")
    _observation(
        root,
        path=values["low_observation_path"],
        sha256=values["low_observation_sha256"],
        alias="LOW",
        previous_trade_date=target,
    )
    _observation(
        root,
        path=values["w80_observation_path"],
        sha256=values["w80_observation_sha256"],
        alias="W80",
        previous_trade_date=target,
    )

    expected_store_sha = _sha(
        values["expected_store_pointer_sha256"],
        label="expected Store pointer",
    )
    _stable_raw(root, STORE_POINTER_RELATIVE, expected_store_sha, label="Store pointer")
    loaded = load_registered_catalog(root / STORE_ROOT_RELATIVE)
    if loaded is None or not isinstance(loaded[0].get("active_closure"), dict):
        holdings_blockers.append("STORE_HOLDINGS_UNAVAILABLE")

    expected_import_root = Path(str(values["expected_import_root"] or ""))
    import quant_investor

    import_origin = Path(quant_investor.__file__).resolve()
    if (
        not expected_import_root.is_absolute()
        or expected_import_root != import_origin
        and expected_import_root not in import_origin.parents
    ):
        core_blockers.append("IMPORT_ORIGIN_MISMATCH")
    if values["scheduler_origin_verified"] is not True:
        core_blockers.append("SCHEDULER_ORIGIN_UNVERIFIED")

    receipt_refs = values["morning_receipts"]
    if type(receipt_refs) is not list or len(receipt_refs) > 2:
        raise IntelligenceError("morning_receipts must contain at most two refs")
    morning_receipts: list[dict[str, Any]] = []
    for index, reference in enumerate(receipt_refs):
        if type(reference) is not dict or set(reference) != {"path", "sha256"}:
            raise IntelligenceError("morning receipt ref shape is invalid")
        morning_receipts.append(
            _json_ref(
                root,
                reference["path"],
                reference["sha256"],
                label=f"morning receipt[{index}]",
            )
        )
    successful_receipts = sorted(
        (receipt for receipt in morning_receipts if _morning_receipt_success(receipt)),
        key=lambda receipt: str(receipt["run_date"]),
    )
    if successful_receipts and successful_receipts[-1]["run_date"] == target:
        consecutive_success_count = 1
        if (
            len(successful_receipts) == 2
            and successful_receipts[0]["run_date"] == successful_receipts[1]["previous_trade_date"]
        ):
            consecutive_success_count = 2
    else:
        consecutive_success_count = 0

    current_state = values["current_schedule_state"]
    if current_state not in {"EVENING_PRIMARY", "DUAL_RUN", "MORNING_PRIMARY"}:
        raise IntelligenceError("current_schedule_state is invalid")
    eligible = not core_blockers and not holdings_blockers
    if not eligible:
        next_state = current_state
        schedule_action = "KEEP_FALLBACK"
    elif current_state == "EVENING_PRIMARY":
        next_state = "DUAL_RUN"
        schedule_action = "ENABLE_0945_CREATE_2100_FALLBACK_KEEP_2130"
    elif current_state == "DUAL_RUN" and consecutive_success_count >= 2:
        next_state = "MORNING_PRIMARY"
        schedule_action = "PAUSE_2100_FALLBACK_PAUSE_2130"
    elif current_state == "MORNING_PRIMARY" and not any(
        receipt.get("run_date") == target and _morning_receipt_success(receipt)
        for receipt in morning_receipts
    ):
        next_state = "DUAL_RUN"
        schedule_action = "RESUME_2100_FALLBACK_KEEP_0945_RESUME_2130"
    else:
        next_state = current_state
        schedule_action = "KEEP_CURRENT_SCHEDULE"

    core_status = "COMPLETE" if not core_blockers else "BLOCKED"
    holdings_status = "COMPLETE" if not holdings_blockers else "BLOCKED"
    auxiliary_status = "COMPLETE" if not auxiliary_blockers else "PARTIAL"
    overall_status = (
        "BLOCKED"
        if core_status == "BLOCKED" or holdings_status == "BLOCKED"
        else "PARTIAL" if auxiliary_status == "PARTIAL" else "COMPLETE"
    )
    receipt = {
        "schema_version": CUTOVER_RECEIPT_SCHEMA,
        "target_date": target,
        "overall_status": overall_status,
        "core_production_status": core_status,
        "holdings_status": holdings_status,
        "auxiliary_status": auxiliary_status,
        "morning_strategy_cutover_eligible": eligible,
        "core_blockers": sorted(set(core_blockers)),
        "holdings_blockers": sorted(set(holdings_blockers)),
        "auxiliary_blockers": sorted(set(auxiliary_blockers)),
        "current_schedule_state": current_state,
        "next_schedule_state": next_state,
        "schedule_action": schedule_action,
        "consecutive_morning_success_count": consecutive_success_count,
        "maintenance_receipt_ref": {
            "path": values["maintenance_receipt_path"],
            "sha256": values["maintenance_receipt_sha256"],
        },
        "calendar_success_ref": {
            "path": values["calendar_success_path"],
            "sha256": values["calendar_success_sha256"],
        },
        "factor_pointer_sha256": expected_factor_sha,
        "store_pointer_sha256": expected_store_sha,
        "import_origin": str(import_origin),
        "broker": False,
        "live_order": False,
        "live_execution": False,
        "actual_holdings_mutation": False,
    }
    receipt_root = _owner_directory(root / f"results/operations/morning_strategy/CN/{target}")
    receipt_path = receipt_root / "2020-cutover.v1.json"
    raw = canonical_json_bytes(receipt)
    digest, created = _write_exact(receipt_path, raw)
    return {
        "command_status": "PUBLISHED" if created else "NO_ACTION",
        **receipt,
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_sha256": digest,
    }


__all__ = [
    "CUTOVER_RECEIPT_SCHEMA",
    "EOD_EVALUATION_SCHEMA",
    "MORNING_RECEIPT_SCHEMA",
    "SINA_CAPTURE_SCHEMA",
    "evaluate_morning_strategy_eod",
    "run_morning_strategy",
    "evaluate_morning_cutover",
    "validate_sina_quote_capture",
]
