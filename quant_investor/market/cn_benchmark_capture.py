"""Exact Tushare benchmark capture and native publication; no portfolio authority."""

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time

from quant_investor.operations.daily_preparation import Sources
from quant_investor.operations.journal_storage import JournalStorage
from quant_investor.system.errors import SystemSecurityError
from . import cn_benchmark_store as native
from .credential_preflight import read_project_env_token
from .tushare_transport import OfficialTushareHttpsClient, replay_tushare_response_bytes

PREFIX = "data/private/cn_benchmark_close"
SCHEMA = "myquant.cn_benchmark_acquisition_receipt.v2"
FIELDS = ("ts_code", "trade_date", "close")
TUSHARE_REQUEST_INTERVAL_SECONDS = 21.0
RECEIPT_FIELDS = {
    "schema_id",
    "generation_id",
    "captured_at",
    "start_date",
    "end_date",
    "required_dates",
    "codes",
    "expected_pointer_sha256",
    "parent_pointer_ref",
    "row_count",
    "rows_sha256",
    "source_system",
    "credential_contract",
    "credential_material_recorded",
    "broker_order_trade_authority",
    "requests",
    "content_sha256",
}


class BenchmarkCaptureError(native.CNBenchmarkStoreError):
    pass


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    if type(value) is not str or date.fromisoformat(value).isoformat() != value:
        raise BenchmarkCaptureError("BENCHMARK_DATE_INVALID")
    return date.fromisoformat(value)


def request_partition(start_date, end_date):
    start, end = _iso(start_date), _iso(end_date)
    if start > end:
        raise BenchmarkCaptureError("BENCHMARK_RANGE_INVALID")
    windows = []
    while start <= end:
        following = (
            start.replace(year=start.year + 1, month=1, day=1)
            if start.month == 12
            else start.replace(month=start.month + 1, day=1)
        )
        last = min(end, following - timedelta(days=1))
        windows.append((start.strftime("%Y%m%d"), last.strftime("%Y%m%d")))
        start = last + timedelta(days=1)
    return [
        {"ts_code": code, "start_date": first, "end_date": last}
        for code in native.REQUIRED_CODES
        for first, last in windows
    ]


def _arguments(start_date, end_date, generation_id, expected, required):
    partitions = request_partition(start_date, end_date)
    if type(generation_id) is not str or native._GENERATION.fullmatch(generation_id) is None:
        raise BenchmarkCaptureError("BENCHMARK_GENERATION_ID_INVALID")
    if type(expected) is not str or native._SHA.fullmatch(expected) is None:
        raise BenchmarkCaptureError("BENCHMARK_PREIMAGE_INVALID")
    if required is not None:
        if (
            type(required) is not list
            or not required
            or any(type(d) is not str for d in required)
            or required != sorted(set(required))
        ):
            raise BenchmarkCaptureError("BENCHMARK_REQUIRED_DATES_INVALID")
        for day in required:
            _iso(day)
            if not start_date <= day <= end_date:
                raise BenchmarkCaptureError("BENCHMARK_REQUIRED_DATE_OUTSIDE_RANGE")
    return partitions


class CaptureStorage(JournalStorage):
    @staticmethod
    def _path(value):
        path = PurePosixPath(value)
        try:
            parts = path.relative_to(PREFIX).parts
        except ValueError as exc:
            raise SystemSecurityError("benchmark capture path invalid") from exc
        if not parts or native._GENERATION.fullmatch(parts[0]) is None:
            raise SystemSecurityError("benchmark capture generation invalid")
        valid = (
            len(parts) == 2
            and parts[1] in {"capture.v1.json", "capture.v2.json", "parent-pointer.json"}
        ) or (
            len(parts) == 3
            and parts[1] == "raw"
            and re.fullmatch(r"[0-9]{4,}-[a-f0-9]{64}\.json", parts[2]) is not None
        )
        if str(path) != value or not valid:
            raise SystemSecurityError("benchmark capture path invalid")
        return path


def _ref(path, raw):
    return {"path": path, "sha256": native._sha256(raw)}


def _parent_rows(source, root, reference, *, expected, generation_id):
    if expected == native.EMPTY_POINTER_SHA256:
        if reference is not None:
            raise BenchmarkCaptureError("BENCHMARK_EMPTY_PARENT_REF_INVALID")
        return []
    expected_path = f"{PREFIX}/{generation_id}/parent-pointer.json"
    if reference != {"path": expected_path, "sha256": expected}:
        raise BenchmarkCaptureError("BENCHMARK_PARENT_REF_INVALID")
    pointer = json.loads(source.raw(reference))
    native._validate_seal(pointer, label="benchmark captured parent pointer")
    if (
        pointer.get("schema_id") != native.BENCHMARK_POINTER_SCHEMA
        or pointer.get("generation_id") == generation_id
    ):
        raise BenchmarkCaptureError("BENCHMARK_PARENT_IDENTITY_INVALID")
    parent = native.load_immutable_generation(root, pointer["generation_id"])
    manifest = parent["manifest"]
    manifest_ref = {
        "path": str(parent["manifest_path"].relative_to(root)),
        "sha256": parent["manifest_sha256"],
    }
    if (
        pointer["manifest"] != manifest_ref
        or pointer["series"] != manifest["series"]
        or any(pointer.get(k) != manifest.get(k) for k in ("start_date", "end_date", "trade_dates"))
    ):
        raise BenchmarkCaptureError("BENCHMARK_PARENT_GENERATION_MISMATCH")
    for ref in (manifest_ref, pointer["series"]):
        source.raw({"path": "data/parquet/cn/benchmarks/" + ref["path"], "sha256": ref["sha256"]})
    return parent["rows"]


def _wire_rows(raw, params):
    response = replay_tushare_response_bytes(
        raw, api_name="index_daily", expected_fields=FIELDS, strict_decimal_decode=True
    )
    if (
        response.api_name != "index_daily"
        or tuple(response.fields) != FIELDS
        or response.has_more is not False
        or response.item_count != len(response.rows)
        or response.reported_count != response.item_count
        or response.provider_reported_count not in {0, response.item_count}
    ):
        raise BenchmarkCaptureError("BENCHMARK_RESPONSE_INCOMPLETE")
    result = []
    for values in response.rows:
        if type(values) is not tuple or len(values) != 3:
            raise BenchmarkCaptureError("BENCHMARK_RESPONSE_ROW_INVALID")
        code, compact, close = values
        if (
            code != params["ts_code"]
            or type(compact) is not str
            or re.fullmatch(r"[0-9]{8}", compact) is None
        ):
            raise BenchmarkCaptureError("BENCHMARK_RESPONSE_IDENTITY_INVALID")
        day = datetime.strptime(compact, "%Y%m%d").date().isoformat()
        if not params["start_date"] <= compact <= params["end_date"] or type(close) is bool:
            raise BenchmarkCaptureError("BENCHMARK_RESPONSE_RANGE_INVALID")
        price = float(close)
        if not math.isfinite(price) or price <= 0:
            raise BenchmarkCaptureError("BENCHMARK_CLOSE_INVALID")
        result.append(
            {
                "date": day,
                "ts_code": code,
                "close": price,
                "source_system": "tushare.index_daily",
                "coverage": "exact_close",
                "value_date": day,
            }
        )
    return result


def _captured_rows(source, requests, partitions, base, required):
    if type(requests) is not list or len(requests) != len(partitions):
        raise BenchmarkCaptureError("BENCHMARK_REQUEST_PARTITION_INVALID")
    rows = []
    for index, (request, params) in enumerate(zip(requests, partitions)):
        if (
            type(request) is not dict
            or set(request) != {"api_name", "params", "fields", "raw_ref"}
            or request["api_name"] != "index_daily"
            or request["params"] != params
            or request["fields"] != list(FIELDS)
        ):
            raise BenchmarkCaptureError("BENCHMARK_REQUEST_PARTITION_INVALID")
        reference = request["raw_ref"]
        raw = source.raw(reference)
        if reference != _ref(f"{base}/raw/{index:04d}-{native._sha256(raw)}.json", raw):
            raise BenchmarkCaptureError("BENCHMARK_RAW_PATH_INVALID")
        rows.extend(_wire_rows(raw, params))
    native._normalize_rows(rows)  # Complete three-code days, positive finite and no duplicates.
    rows.sort(key=lambda row: (row["date"], row["ts_code"]))
    if required is not None and {(r["date"], r["ts_code"]) for r in rows} != {
        (d, c) for d in required for c in native.REQUIRED_CODES
    }:
        raise BenchmarkCaptureError("BENCHMARK_REQUIRED_DATES_INCOMPLETE")
    return rows


def _merge(parent, rows):
    by_key = {(r["date"], r["ts_code"]): r for r in parent}
    for row in native._normalize_rows(rows):
        key = row["date"], row["ts_code"]
        if key in by_key and by_key[key] != row:
            raise BenchmarkCaptureError("BENCHMARK_EXISTING_ROW_CONFLICT")
        by_key[key] = row
    return [by_key[k] for k in sorted(by_key)]


def _read_capture(
    source, stored, *, start_date, end_date, generation_id, expected, required, partitions
):
    receipt = json.loads(stored.data)
    if (
        type(receipt) is not dict
        or set(receipt) != RECEIPT_FIELDS
        or native.canonical_json_bytes(receipt) != stored.data
    ):
        raise BenchmarkCaptureError("BENCHMARK_CAPTURE_FIELDS_INVALID")
    native._validate_seal(receipt, label="benchmark capture")
    expected_values = {
        "schema_id": SCHEMA,
        "generation_id": generation_id,
        "start_date": start_date,
        "end_date": end_date,
        "required_dates": required,
        "codes": list(native.REQUIRED_CODES),
        "expected_pointer_sha256": expected,
        "source_system": "tushare",
        "credential_contract": "OFFICIAL_TUSHARE_TOKEN_ENV",
        "credential_material_recorded": False,
        "broker_order_trade_authority": False,
    }
    if (
        any(receipt[k] != v for k, v in expected_values.items())
        or receipt["credential_material_recorded"] is not False
        or receipt["broker_order_trade_authority"] is not False
    ):
        raise BenchmarkCaptureError("BENCHMARK_CAPTURE_ARGUMENT_CONFLICT")
    captured = datetime.strptime(receipt["captured_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    if captured.strftime("%Y-%m-%dT%H:%M:%SZ") != receipt["captured_at"] or captured > _now():
        raise BenchmarkCaptureError("BENCHMARK_CAPTURE_IN_FUTURE")
    base = f"{PREFIX}/{generation_id}"
    rows = _captured_rows(source, receipt["requests"], partitions, base, required)
    if (
        type(receipt["row_count"]) is not int
        or receipt["row_count"] != len(rows)
        or receipt["rows_sha256"] != native._sha256(native.canonical_json_bytes(rows))
    ):
        raise BenchmarkCaptureError("BENCHMARK_CAPTURE_ROWS_MISMATCH")
    parent = _parent_rows(
        source,
        Path(source.workspace) / "data/parquet/cn/benchmarks",
        receipt["parent_pointer_ref"],
        expected=expected,
        generation_id=generation_id,
    )
    source.raw(_ref(f"{base}/capture.v2.json", stored.data))
    return receipt, _merge(parent, rows)


def capture_benchmark_close(
    *,
    workspace_root,
    start_date,
    end_date,
    generation_id,
    expected_pointer_sha256,
    required_dates=None,
    client=None,
):
    """Only explicit inherited TUSHARE_TOKEN is read; this callable never edits it."""
    root = Path(workspace_root).resolve(strict=True)
    benchmark_root = root / "data/parquet/cn/benchmarks"
    expected = expected_pointer_sha256
    partitions = _arguments(start_date, end_date, generation_id, expected, required_dates)
    if expected == native.EMPTY_POINTER_SHA256 and start_date != "2026-03-17":
        raise BenchmarkCaptureError(
            "initial benchmark generation must capture the full performance history"
        )
    base = f"{PREFIX}/{generation_id}"
    storage, source = CaptureStorage(str(root)), Sources(str(root))
    if storage.read(base + "/capture.v1.json") is not None:
        raise BenchmarkCaptureError("BENCHMARK_LEGACY_CAPTURE_REQUIRES_ORIGINAL_PRODUCER")
    stored = storage.read(base + "/capture.v2.json")
    if (
        expected == native.EMPTY_POINTER_SHA256
        and storage.read(base + "/parent-pointer.json") is not None
    ):
        raise BenchmarkCaptureError("BENCHMARK_EMPTY_PARENT_FILE_FORBIDDEN")
    if stored is None:
        actual = native.pointer_sha256(benchmark_root)
        if actual != expected:
            raise native.CNBenchmarkCASMismatch("benchmark preimage changed before capture")
        parent_ref = None
        if expected == native.EMPTY_POINTER_SHA256:
            if start_date != "2026-03-17":
                raise BenchmarkCaptureError(
                    "initial benchmark generation must capture the full performance history"
                )
        else:
            pointer_ref = source.pin("data/parquet/cn/benchmarks/_latest.json")
            loaded = native.load_generation(benchmark_root)
            if pointer_ref["sha256"] != expected or loaded["pointer_sha256"] != expected:
                raise native.CNBenchmarkCASMismatch("benchmark parent changed before capture")
            raw = source.raw(pointer_ref)
            parent_ref = _ref(base + "/parent-pointer.json", raw)
            storage.write(parent_ref["path"], raw)
            _parent_rows(
                source, benchmark_root, parent_ref, expected=expected, generation_id=generation_id
            )
        source.recheck()
        selected = client or OfficialTushareHttpsClient(strict_decimal_decode=True)
        requests = []
        for index, params in enumerate(partitions):
            if index:
                time.sleep(TUSHARE_REQUEST_INTERVAL_SECONDS)
            response = selected.request(
                api_name="index_daily", params=params, expected_fields=FIELDS
            )
            raw = bytes(response.raw_body)
            _wire_rows(raw, params)
            reference = _ref(f"{base}/raw/{index:04d}-{native._sha256(raw)}.json", raw)
            storage.write(reference["path"], raw)
            requests.append(
                {
                    "api_name": "index_daily",
                    "params": params,
                    "fields": list(FIELDS),
                    "raw_ref": reference,
                }
            )
        captured = _now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = _captured_rows(source, requests, partitions, base, required_dates)
        parent = _parent_rows(
            source, benchmark_root, parent_ref, expected=expected, generation_id=generation_id
        )
        _merge(parent, rows)
        receipt = native._seal(
            {
                "schema_id": SCHEMA,
                "generation_id": generation_id,
                "captured_at": captured,
                "start_date": start_date,
                "end_date": end_date,
                "required_dates": required_dates,
                "codes": list(native.REQUIRED_CODES),
                "expected_pointer_sha256": expected,
                "parent_pointer_ref": parent_ref,
                "row_count": len(rows),
                "rows_sha256": native._sha256(native.canonical_json_bytes(rows)),
                "source_system": "tushare",
                "credential_contract": "OFFICIAL_TUSHARE_TOKEN_ENV",
                "credential_material_recorded": False,
                "broker_order_trade_authority": False,
                "requests": requests,
            }
        )
        source.recheck()
        stored = storage.write(base + "/capture.v2.json", native.canonical_json_bytes(receipt))
    # Do not retain mutable current pointer observations across our own publication.
    source = Sources(str(root))
    receipt, rows = _read_capture(
        source,
        stored,
        start_date=start_date,
        end_date=end_date,
        generation_id=generation_id,
        expected=expected,
        required=required_dates,
        partitions=partitions,
    )
    source.recheck()
    ref = _ref(base + "/capture.v2.json", stored.data)
    published = native.publish_generation_with_compatibility(
        workspace_root=root,
        rows=rows,
        generation_id=generation_id,
        captured_at=receipt["captured_at"],
        expected_pointer_sha256=expected,
        acquisition_receipt_ref=ref,
    )
    source.recheck()
    return {
        "status": "NO_ACTION" if published["no_action"] else "PUBLISHED",
        "generation_id": generation_id,
        "pointer_sha256": published["pointer_sha256"],
        "start_date": published["pointer"]["start_date"],
        "end_date": published["pointer"]["end_date"],
        "row_count": published["manifest"]["row_count"],
        "receipt_path": ref["path"],
        "receipt_sha256": ref["sha256"],
        "compatibility_repaired": published["compatibility_repaired"],
        "credential_material_recorded": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
    }


@contextmanager
def _project_credentials(workspace):
    absent = object()
    previous = os.environ.get("TUSHARE_TOKEN", absent)
    token = read_project_env_token(Path(workspace) / ".env")
    try:
        os.environ["TUSHARE_TOKEN"] = token
        yield
    finally:
        if previous is absent:
            os.environ.pop("TUSHARE_TOKEN", None)
        else:
            os.environ["TUSHARE_TOKEN"] = previous


def _legacy_eastmoney(args):
    from scripts.backfill_cn_dashboard_benchmark import pull_eastmoney_kline_rows

    root = args.workspace_root
    if CaptureStorage(str(root)).read(f"{PREFIX}/{args.generation_id}/capture.v2.json") is not None:
        raise BenchmarkCaptureError("BENCHMARK_CAPTURE_SOURCE_CONFLICT")
    benchmark_root = root / "data/parquet/cn/benchmarks"
    if native.pointer_sha256(benchmark_root) != args.expected_pointer_sha256:
        raise native.CNBenchmarkCASMismatch("benchmark pointer preimage mismatch")
    rows = pull_eastmoney_kline_rows(
        start_date=args.start_date, end_date=args.end_date, ts_codes=native.REQUIRED_CODES
    )
    rows.sort(key=lambda row: (row["date"], row["ts_code"]))
    parent = (
        []
        if args.expected_pointer_sha256 == native.EMPTY_POINTER_SHA256
        else native.load_generation(benchmark_root)["rows"]
    )
    if not parent and args.start_date != "2026-03-17":
        raise BenchmarkCaptureError(
            "initial benchmark generation must capture the full performance history"
        )
    merged = _merge(parent, rows)
    captured = _now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = native._seal(
        {
            "schema_id": "myquant.cn_benchmark_acquisition_receipt.v1",
            "generation_id": args.generation_id,
            "captured_at": captured,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "codes": list(native.REQUIRED_CODES),
            "row_count": len(rows),
            "rows_sha256": native._sha256(native.canonical_json_bytes(rows)),
            "source_system": "eastmoney",
            "credential_contract": "PUBLIC_READ_ONLY_NO_CREDENTIAL",
            "credential_material_recorded": False,
            "broker_order_trade_authority": False,
        }
    )
    raw = native.canonical_json_bytes(receipt)
    ref = _ref(f"{PREFIX}/{args.generation_id}/capture.v1.json", raw)
    CaptureStorage(str(root)).write(ref["path"], raw)
    published = native.publish_generation_with_compatibility(
        workspace_root=root,
        rows=merged,
        generation_id=args.generation_id,
        captured_at=captured,
        expected_pointer_sha256=args.expected_pointer_sha256,
        acquisition_receipt_ref=ref,
    )
    return {
        "status": "PUBLISHED",
        "generation_id": args.generation_id,
        "pointer_sha256": published["pointer_sha256"],
        "start_date": published["pointer"]["start_date"],
        "end_date": published["pointer"]["end_date"],
        "row_count": published["manifest"]["row_count"],
        "receipt_path": ref["path"],
        "receipt_sha256": ref["sha256"],
        "credential_material_recorded": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    for name in ("start-date", "end-date", "generation-id", "expected-pointer-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source", choices=("tushare", "eastmoney"), default="tushare")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.workspace_root = args.workspace_root.resolve(strict=True)
    _arguments(
        args.start_date, args.end_date, args.generation_id, args.expected_pointer_sha256, None
    )
    if not args.execute:
        observed = native.pointer_sha256(args.workspace_root / "data/parquet/cn/benchmarks")
        if observed != args.expected_pointer_sha256:
            raise native.CNBenchmarkCASMismatch("benchmark pointer preimage mismatch")
        result = {
            "status": "PLAN_ONLY",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "generation_id": args.generation_id,
            "expected_pointer_sha256": observed,
            "provider_called": False,
        }
    elif args.source == "eastmoney":
        result = _legacy_eastmoney(args)
    else:

        def execute():
            return capture_benchmark_close(
                workspace_root=args.workspace_root,
                start_date=args.start_date,
                end_date=args.end_date,
                generation_id=args.generation_id,
                expected_pointer_sha256=args.expected_pointer_sha256,
            )

        storage = CaptureStorage(str(args.workspace_root))
        if storage.read(f"{PREFIX}/{args.generation_id}/capture.v2.json") is not None:
            result = execute()
        elif storage.read(f"{PREFIX}/{args.generation_id}/capture.v1.json") is not None:
            raise BenchmarkCaptureError("BENCHMARK_LEGACY_CAPTURE_REQUIRES_ORIGINAL_PRODUCER")
        else:
            with _project_credentials(args.workspace_root):
                result = execute()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
