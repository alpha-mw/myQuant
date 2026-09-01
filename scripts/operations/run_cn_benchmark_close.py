#!/usr/bin/env python3
"""Capture exact CN benchmark closes and publish one immutable Market generation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

from quant_investor.credential_utils import create_tushare_pro
from quant_investor.config import Config
from quant_investor.market.cn_benchmark_store import (
    REQUIRED_CODES,
    canonical_json_bytes,
    pointer_sha256,
    publish_generation,
)
from quant_investor.market.credential_preflight import read_project_env_token


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_exact(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("benchmark capture identity collision")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def _provider_rows(
    token: str | None,
    *,
    start_date: str,
    end_date: str,
    source: str,
) -> list[dict[str, Any]]:
    if source == "eastmoney":
        from backfill_cn_dashboard_benchmark import pull_eastmoney_kline_rows

        rows = pull_eastmoney_kline_rows(
            start_date=start_date,
            end_date=end_date,
            ts_codes=REQUIRED_CODES,
        )
        return sorted(rows, key=lambda row: (row["date"], row["ts_code"]))
    import tushare as ts  # type: ignore[import-not-found]

    if token is None:
        raise RuntimeError("PROJECT_ENV token is required for Tushare benchmark capture")
    pro = create_tushare_pro(ts, token, Config.TUSHARE_URL)
    if pro is None:
        raise RuntimeError("benchmark provider initialization failed")
    result: list[dict[str, Any]] = []
    for code in REQUIRED_CODES:
        frame = pro.index_daily(
            ts_code=code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        if (
            frame is None
            or frame.empty
            or not {"ts_code", "trade_date", "close"}.issubset(frame.columns)
        ):
            raise RuntimeError(f"benchmark provider response incomplete: {code}")
        for row in frame.loc[:, ["ts_code", "trade_date", "close"]].to_dict("records"):
            if str(row["ts_code"]) != code:
                raise RuntimeError("benchmark provider symbol drift")
            compact = str(row["trade_date"])
            day = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
            result.append(
                {
                    "date": day,
                    "ts_code": code,
                    "close": float(row["close"]),
                    "source_system": "tushare.index_daily",
                    "coverage": "exact_close",
                    "value_date": day,
                }
            )
    return sorted(result, key=lambda row: (row["date"], row["ts_code"]))


def _write_compatibility_csv(path: Path, new_rows: list[dict[str, Any]]) -> None:
    fields = ["date", "ts_code", "close", "source_system", "value_date", "coverage"]
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row.get("date") or ""), str(row.get("ts_code") or ""))
                if all(key):
                    existing[key] = {field: str(row.get(field) or "") for field in fields}
    for row in new_rows:
        existing[(row["date"], row["ts_code"])] = {field: str(row[field]) for field in fields}
    ordered = [existing[key] for key in sorted(existing)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--expected-pointer-sha256", required=True)
    parser.add_argument("--source", choices=("tushare", "eastmoney"), default="tushare")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve(strict=True)
    benchmark_root = workspace / "data/parquet/cn/benchmarks"
    observed = pointer_sha256(benchmark_root)
    if observed != args.expected_pointer_sha256:
        raise RuntimeError("benchmark pointer preimage mismatch")
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "generation_id": args.generation_id,
                    "expected_pointer_sha256": observed,
                    "provider_called": False,
                },
                sort_keys=True,
            )
        )
        return 0
    token = read_project_env_token(workspace / ".env") if args.source == "tushare" else None
    try:
        rows = _provider_rows(
            token,
            start_date=args.start_date,
            end_date=args.end_date,
            source=args.source,
        )
    finally:
        token = None
    expected_days = {row["date"] for row in rows if args.start_date <= row["date"] <= args.end_date}
    if not expected_days:
        raise RuntimeError("benchmark capture returned no trade dates")
    captured_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    receipt = {
        "schema_id": "myquant.cn_benchmark_acquisition_receipt.v1",
        "generation_id": args.generation_id,
        "captured_at": captured_at,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "codes": list(REQUIRED_CODES),
        "row_count": len(rows),
        "rows_sha256": _sha(canonical_json_bytes(rows)),
        "source_system": args.source,
        "credential_contract": (
            "PROJECT_ENV_V2" if args.source == "tushare" else "PUBLIC_READ_ONLY_NO_CREDENTIAL"
        ),
        "credential_material_recorded": False,
        "broker_order_trade_authority": False,
    }
    receipt["content_sha256"] = _sha(canonical_json_bytes(receipt))
    receipt_relative = f"data/private/cn_benchmark_close/{args.generation_id}/capture.v1.json"
    receipt_path = workspace / receipt_relative
    receipt_raw = canonical_json_bytes(receipt)
    _write_exact(receipt_path, receipt_raw)
    published = publish_generation(
        benchmark_root,
        rows=rows,
        generation_id=args.generation_id,
        captured_at=captured_at,
        expected_pointer_sha256=observed,
        acquisition_receipt_ref={"path": receipt_relative, "sha256": _sha(receipt_raw)},
    )
    _write_compatibility_csv(workspace / "portfolio_dashboard/inputs/cn_index_benchmark.csv", rows)
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "generation_id": args.generation_id,
                "pointer_sha256": published["pointer_sha256"],
                "start_date": published["pointer"]["start_date"],
                "end_date": published["pointer"]["end_date"],
                "row_count": published["manifest"]["row_count"],
                "receipt_path": receipt_relative,
                "receipt_sha256": _sha(receipt_raw),
                "credential_material_recorded": False,
                "broker_calls": False,
                "order_calls": False,
                "trade_calls": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
