#!/usr/bin/env python3
"""Capture one credential-free, read-only Sina A-share quote response."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable
from urllib.request import Request, urlopen

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.intelligence.morning import (
    SINA_CAPTURE_SCHEMA,
    validate_sina_quote_capture,
)

_SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")


class SinaCaptureError(RuntimeError):
    """One controlled public-quote capture error."""


def _mapping(symbol: str) -> str:
    if _SYMBOL.fullmatch(symbol) is None:
        raise SinaCaptureError("SINA_SYMBOL_INVALID")
    suffix = symbol[-2:].lower()
    return f"{suffix}{symbol[:6]}"


def _load_request(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SinaCaptureError("SINA_REQUEST_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SinaCaptureError("SINA_REQUEST_SHA_MISMATCH")
    try:
        value = parse_canonical_json_bytes(raw, label="Sina capture request")
    except Exception as exc:
        raise SinaCaptureError("SINA_REQUEST_NOT_CANONICAL") from exc
    if type(value) is not dict or set(value) != {"run_date", "symbols"}:
        raise SinaCaptureError("SINA_REQUEST_INVALID")
    run_date = value["run_date"]
    try:
        datetime.strptime(run_date, "%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise SinaCaptureError("SINA_RUN_DATE_INVALID") from exc
    symbols = value["symbols"]
    if (
        type(symbols) is not list
        or not symbols
        or any(type(symbol) is not str or _SYMBOL.fullmatch(symbol) is None for symbol in symbols)
        or symbols != sorted(set(symbols), key=lambda item: item.encode("ascii"))
    ):
        raise SinaCaptureError("SINA_SYMBOL_SET_INVALID")
    return value


def _safe_output_root(workspace: Path, output_root: Path, *, run_date: str) -> Path:
    expected = workspace / f"data/private/cn_public_quotes/{run_date}/sina-0945"
    if output_root != expected:
        raise SinaCaptureError("SINA_OUTPUT_ROOT_NOT_DETERMINISTIC")
    private_root = workspace / "data/private"
    private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(private_root, 0o700)
    for directory in (
        private_root / "cn_public_quotes",
        private_root / "cn_public_quotes" / run_date,
    ):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    if output_root.exists():
        observed = os.lstat(output_root)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise SinaCaptureError("SINA_OUTPUT_ROOT_UNSAFE")
        return output_root
    output_root.mkdir(mode=0o700, parents=True)
    os.chmod(output_root, 0o700)
    return output_root


def _write_once(path: Path, raw: bytes) -> str:
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
            raise SinaCaptureError("SINA_CAPTURE_CONFLICT_UNSAFE")
        existing = path.read_bytes()
        if existing != raw:
            raise SinaCaptureError("SINA_CAPTURE_CONFLICT")
        return hashlib.sha256(existing).hexdigest()
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Referer": "https://finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0 myQuant research-only quote capture",
        },
        method="GET",
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed HTTPS endpoint
        if response.status != 200:
            raise SinaCaptureError("SINA_HTTP_STATUS_INVALID")
        return response.read()


def _parse(raw: bytes, mappings: list[dict[str, str]]) -> list[dict[str, str]]:
    try:
        text = raw.decode("gb18030", errors="strict")
    except UnicodeError as exc:
        raise SinaCaptureError("SINA_RESPONSE_DECODE_FAILED") from exc
    by_provider = {row["provider_symbol"]: row["symbol"] for row in mappings}
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        prefix = "var hq_str_"
        if not line.startswith(prefix) or '="' not in line or not line.endswith('";'):
            continue
        provider_symbol, payload = line[len(prefix) :].split('="', 1)
        if provider_symbol not in by_provider:
            continue
        fields = payload[:-2].split(",")
        if len(fields) < 32 or not fields[0]:
            raise SinaCaptureError("SINA_RESPONSE_FIELDS_INVALID")
        symbol = by_provider[provider_symbol]
        rows[symbol] = {
            "symbol": symbol,
            "name": fields[0],
            "open": fields[1],
            "previous_close": fields[2],
            "price": fields[3],
            "high": fields[4],
            "low": fields[5],
            "volume": fields[8],
            "amount": fields[9],
            "provider_date": fields[30],
            "provider_time": fields[31],
        }
    ordered = [rows[row["symbol"]] for row in mappings if row["symbol"] in rows]
    if [row["symbol"] for row in ordered] != [row["symbol"] for row in mappings]:
        raise SinaCaptureError("SINA_RESPONSE_SYMBOL_SET_INCOMPLETE")
    return ordered


def run(
    args: argparse.Namespace,
    *,
    fetcher: Callable[[str], bytes] = _fetch,
    now: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
) -> dict[str, Any]:
    workspace = Path(args.workspace_root).resolve(strict=True)
    request = _load_request(Path(args.request_path), args.request_sha256)
    run_date = request["run_date"]
    output_root = Path(args.output_root)
    if not args.allow_live:
        if output_root.exists():
            raise SinaCaptureError("SINA_DRY_RUN_OUTPUT_EXISTS")
        return {
            "status": "DRY_RUN_VALIDATED",
            "provider": "SINA",
            "network_attempts": 0,
            "symbol_count": len(request["symbols"]),
        }
    root = _safe_output_root(workspace, output_root, run_date=run_date)
    mappings = [
        {"provider_symbol": _mapping(symbol), "symbol": symbol} for symbol in request["symbols"]
    ]
    request_time = now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = "https://hq.sinajs.cn/list=" + ",".join(row["provider_symbol"] for row in mappings)
    raw = fetcher(url)
    response_time = now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    quote_rows = _parse(raw, mappings)
    raw_path = root / "raw.gb18030.txt"
    raw_sha = _write_once(raw_path, raw)
    raw_relative = raw_path.relative_to(workspace).as_posix()
    capture = {
        "schema_version": SINA_CAPTURE_SCHEMA,
        "provider": "SINA",
        "request_time": request_time,
        "response_time": response_time,
        "encoding": "GB18030",
        "raw_ref": {"path": raw_relative, "sha256": raw_sha, "size": len(raw)},
        "field_definitions": {
            "amount": "provider cumulative turnover CNY",
            "price": "provider current price CNY",
            "volume": "provider cumulative shares",
        },
        "symbol_mapping": mappings,
        "quote_rows": quote_rows,
        "reasonable": True,
        "broker": False,
        "order": False,
        "execution": False,
    }
    validate_sina_quote_capture(capture, raw=raw, run_date=run_date)
    capture_path = root / "capture.json"
    capture_sha = _write_once(capture_path, canonical_json_bytes(capture))
    summary = {
        "status": "CAPTURED",
        "provider": "SINA",
        "network_attempts": 1,
        "symbol_count": len(mappings),
        "capture_path": capture_path.relative_to(workspace).as_posix(),
        "capture_sha256": capture_sha,
        "raw_sha256": raw_sha,
        "broker": False,
        "order": False,
        "execution": False,
    }
    _write_once(root / "summary.json", canonical_json_bytes(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        result = run(parse_args())
    except SinaCaptureError as exc:
        print(json.dumps({"status": "QUOTE_UNAVAILABLE", "blocker": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
