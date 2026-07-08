#!/usr/bin/env python3
"""Backfill the ignored CN dashboard benchmark input from real index close sources."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_index_benchmark.csv"
DEFAULT_START_DATE = "2026-02-15"
SOURCE_SYSTEM = "tushare.index_daily"
EASTMONEY_SOURCE_SYSTEM = "eastmoney.push2his.kline"
FIELDNAMES = ["date", "ts_code", "close", "source_system", "value_date", "coverage"]
DEFAULT_TS_CODES = ("000300.SH", "000905.SH", "000852.SH", "000688.SH", "399006.SZ")
EASTMONEY_INDEX_SECIDS = {
    "000300.SH": "1.000300",
    "000905.SH": "1.000905",
    "000852.SH": "1.000852",
    "000688.SH": "1.000688",
    "399006.SZ": "0.399006",
}


def _iso_to_tushare(value: str) -> str:
    return str(value or "").replace("-", "")


def _tushare_to_iso(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def _read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _parse_close(value: Any) -> float | None:
    try:
        close = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return close if close > 0 else None


def pull_index_daily_rows(
    pro: Any,
    *,
    start_date: str,
    end_date: str,
    ts_codes: tuple[str, ...] = DEFAULT_TS_CODES,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    start_key = _iso_to_tushare(start_date)
    end_key = _iso_to_tushare(end_date)
    for ts_code in ts_codes:
        frame = pro.index_daily(ts_code=ts_code, start_date=start_key, end_date=end_key)
        if frame is None or getattr(frame, "empty", True):
            continue
        if "trade_date" not in frame.columns or "close" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            iso_date = _tushare_to_iso(row.get("trade_date"))
            close = _parse_close(row.get("close"))
            if not iso_date or close is None:
                continue
            rows.append(
                {
                    "date": iso_date,
                    "ts_code": ts_code,
                    "close": f"{close:.6f}",
                    "source_system": SOURCE_SYSTEM,
                    "value_date": iso_date,
                    "coverage": "exact_close",
                }
            )
    return sorted(rows, key=lambda item: (item["date"], item["ts_code"]))


def _fetch_json(url: str, retries: int = 3) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=12) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - live network defensive guard.
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Eastmoney kline request failed after {retries} attempts: {last_error}")


def pull_eastmoney_kline_rows(
    *,
    start_date: str,
    end_date: str,
    ts_codes: tuple[str, ...] = DEFAULT_TS_CODES,
    fetch_json: Any = _fetch_json,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    start_key = _iso_to_tushare(start_date)
    end_key = _iso_to_tushare(end_date)
    for ts_code in ts_codes:
        secid = EASTMONEY_INDEX_SECIDS.get(ts_code)
        if not secid:
            continue
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "0",
            "beg": start_key,
            "end": end_key,
        }
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
        payload = fetch_json(url)
        klines = ((payload.get("data") or {}).get("klines") or [])
        for item in klines:
            parts = str(item).split(",")
            if len(parts) < 3:
                continue
            iso_date = _tushare_to_iso(parts[0])
            close = _parse_close(parts[2])
            if not iso_date or close is None:
                continue
            rows.append(
                {
                    "date": iso_date,
                    "ts_code": ts_code,
                    "close": f"{close:.6f}",
                    "source_system": EASTMONEY_SOURCE_SYSTEM,
                    "value_date": iso_date,
                    "coverage": "exact_close",
                }
            )
    return sorted(rows, key=lambda item: (item["date"], item["ts_code"]))


def backfill_benchmark_file(
    pro: Any,
    *,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    ts_codes: tuple[str, ...] = DEFAULT_TS_CODES,
    source: str = "tushare",
    replace_existing: bool = True,
) -> dict[str, Any]:
    end_date = end_date or date.today().isoformat()
    existing_rows = _read_existing_rows(output_file)
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in existing_rows:
        key = (str(row.get("date") or "").strip(), str(row.get("ts_code") or "").strip())
        if key[0] and key[1]:
            merged[key] = {field: str(row.get(field) or "").strip() for field in FIELDNAMES}
    if source == "eastmoney":
        pulled_rows = pull_eastmoney_kline_rows(start_date=start_date, end_date=end_date, ts_codes=ts_codes)
        source_system = EASTMONEY_SOURCE_SYSTEM
    else:
        pulled_rows = pull_index_daily_rows(pro, start_date=start_date, end_date=end_date, ts_codes=ts_codes)
        source_system = SOURCE_SYSTEM
    for row in pulled_rows:
        key = (row["date"], row["ts_code"])
        if replace_existing or key not in merged:
            merged[key] = row
    output_rows = [merged[key] for key in sorted(merged)]
    _write_rows(output_file, output_rows)
    return {
        "output_file": str(output_file),
        "start_date": start_date,
        "end_date": end_date,
        "ts_codes": list(ts_codes),
        "source_system": source_system,
        "replace_existing": replace_existing,
        "existing_row_count": len(existing_rows),
        "pulled_row_count": len(pulled_rows),
        "output_row_count": len(output_rows),
    }


def _create_tushare_pro() -> Any:
    import tushare as ts  # type: ignore[import-not-found]

    from quant_investor.config import Config
    from quant_investor.credential_utils import create_tushare_pro

    pro = create_tushare_pro(ts, Config.TUSHARE_TOKEN, Config.TUSHARE_URL)
    if pro is None:
        raise SystemExit("TUSHARE_TOKEN 未设置，无法拉取 cn_index_benchmark.csv。")
    return pro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--ts-code", action="append", dest="ts_codes")
    parser.add_argument("--source", choices=["tushare", "eastmoney"], default="tushare")
    parser.add_argument(
        "--fill-only-missing",
        action="store_true",
        help="Preserve existing (date, ts_code) rows and only add missing rows.",
    )
    args = parser.parse_args()
    ts_codes = tuple(args.ts_codes) if args.ts_codes else DEFAULT_TS_CODES
    summary = backfill_benchmark_file(
        _create_tushare_pro() if args.source == "tushare" else None,
        output_file=args.output_file,
        start_date=args.start_date,
        end_date=args.end_date,
        ts_codes=ts_codes,
        source=args.source,
        replace_existing=not args.fill_only_missing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
