#!/usr/bin/env python3
"""Build the Dashboard-only official China 1Y government-yield input."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_govt_bond_yield.csv"
)
SOURCE_SYSTEM = "chinabond.mof_govt_yield_curve"
SOURCE_URL = "https://yield.chinabond.com.cn/cbweb-mn/pgxh/showHistory"
QUERY_URL = "https://yield.chinabond.com.cn/cbweb-mn/pgxh/historyQuery"
CURVE_NAME = "中债国债收益率曲线"


def _fetch(start_date: str, end_date: str) -> bytes:
    payload = urllib.parse.urlencode(
        {
            "startDate": start_date,
            "endDate": end_date,
            "gjqx": "1",
            "locale": "",
        }
    ).encode("ascii")
    request = urllib.request.Request(
        QUERY_URL,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "myQuant-cn-dashboard-risk-free/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _rows(raw: bytes, start_date: str, end_date: str) -> list[dict[str, str]]:
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("risk_free_response_not_valid_json") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("risk_free_response_empty")
    selected: dict[str, dict[str, str]] = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("qxmc") != CURVE_NAME:
            raise ValueError("risk_free_curve_identity_invalid")
        work_time = str(item.get("workTime") or "")[:10]
        value = item.get("oneYear")
        try:
            observed_date = date.fromisoformat(work_time)
            annual_yield = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("risk_free_observation_invalid") from exc
        if not (
            date.fromisoformat(start_date)
            <= observed_date
            <= date.fromisoformat(end_date)
        ):
            raise ValueError("risk_free_observation_outside_requested_range")
        if not (0 <= annual_yield < 100):
            raise ValueError("risk_free_yield_out_of_range")
        if work_time in selected:
            raise ValueError("risk_free_duplicate_observation")
        selected[work_time] = {
            "date": work_time,
            "tenor": "1Y",
            "annual_yield_percent": format(annual_yield, ".8g"),
            "source_system": SOURCE_SYSTEM,
            "source_url": SOURCE_URL,
        }
    return [selected[key] for key in sorted(selected)]


def _write_atomic(output: Path, rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "tenor",
                "annual_yield_percent",
                "source_system",
                "source_url",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-03-17")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise SystemExit("start_date_after_end_date")
    raw = args.input_json.read_bytes() if args.input_json else _fetch(
        start.isoformat(), end.isoformat()
    )
    rows = _rows(raw, start.isoformat(), end.isoformat())
    _write_atomic(args.output.resolve(), rows)
    print(
        json.dumps(
            {
                "written": True,
                "output": str(args.output.resolve()),
                "row_count": len(rows),
                "start_date": rows[0]["date"],
                "end_date": rows[-1]["date"],
                "source_system": SOURCE_SYSTEM,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
