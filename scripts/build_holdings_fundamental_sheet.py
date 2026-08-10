#!/usr/bin/env python3
"""Build offline fundamental tracking sheets for current CN aggressive holdings.

The script is measurement-only. It reads the catalog-resolved active
``ledger_after_manual_switch.parquet`` after exact closure-hash verification,
then combines it with local Parquet fundamentals/disclosure tables and writes
ignored audit artifacts under
``results/track_record_audit/<YYYYMMDD>/fundamentals/``. It never calls online
providers and never modifies strategy records.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from quant_investor.strategy_records.store import (
    StrategyRecordStoreError,
    load_registered_catalog,
    resolve_active_record_dirs,
)

RecordStoreError = StrategyRecordStoreError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = PROJECT_ROOT / "results" / "strategy_records"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "track_record_audit"
DEFAULT_FUNDAMENTALS_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "fundamental_raw"
DEFAULT_DISCLOSURE_ROOT = PROJECT_ROOT / "data" / "parquet" / "cn" / "dag_core_raw" / "table=disclosure_date"
DEFAULT_AS_OF = date.today().strftime("%Y%m%d")
H1_END_DATE = "20260630"
DISCLOSURE_PENDING = "disclosure_date_pending_manual"
INVALID_MANUAL_LEDGER_STATUS_MARKERS = ("invalidated_price_basis_no_execution",)
HIGH_SCRUTINY_VERDICTS = {"看不清", "偏故事"}
MAXWELL_VERDICTS = {
    "002008.SZ": ("大族激光", "看不清"),
    "002384.SZ": ("东山精密", "真订单真产能"),
    "002463.SZ": ("沪电股份", "真订单真产能"),
    "002851.SZ": ("麦格米特", "看不清"),
    "605358.SH": ("立昂微", "真订单真产能"),
}
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")


@dataclass(frozen=True)
class HoldingBaseline:
    ledger: Any
    manifest: dict[str, Any]
    manifest_path: Path
    ledger_path: Path


@contextmanager
def _suppress_native_stderr():
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _manual_manifest_is_valid_baseline(manifest: dict[str, Any]) -> bool:
    status_text = " ".join(
        str(manifest.get(key) or "")
        for key in ("status", "execution_status", "price_basis", "note")
    )
    return not any(marker in status_text for marker in INVALID_MANUAL_LEDGER_STATUS_MARKERS)


def _record_store_root(record_root: Path) -> Path:
    """Keep the historical strategy-root CLI spelling as a compatibility alias."""

    root = Path(record_root).absolute()
    if root.name == "aggressive_tech_manufacturing" and root.parent.name == "CN":
        return root
    return root / "CN" / "aggressive_tech_manufacturing"


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _contained_regular_file(
    *,
    store_root: Path,
    active_dir: Path,
    relative_path: Any,
    declared_sha256: Any,
    label: str,
) -> tuple[Path, bytes]:
    raw_relative = str(relative_path or "").strip()
    raw_sha256 = str(declared_sha256 or "").strip().lower()
    if not raw_relative or Path(raw_relative).is_absolute() or ".." in Path(raw_relative).parts:
        raise RecordStoreError(f"active closure {label} path must be record-root relative")
    if len(raw_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in raw_sha256):
        raise RecordStoreError(f"active closure {label} sha256 missing or invalid")

    store_root = store_root.absolute()
    active_dir = active_dir.absolute()
    candidate = store_root / raw_relative
    try:
        active_dir.relative_to(store_root)
        candidate.relative_to(active_dir)
    except ValueError as exc:
        raise RecordStoreError(f"active closure {label} path escapes active record") from exc

    current = store_root
    if current.is_symlink():
        raise RecordStoreError("strategy-record root must not be a symlink")
    for part in candidate.relative_to(store_root).parts:
        current = current / part
        if current.is_symlink():
            raise RecordStoreError(f"active closure {label} path contains a symlink")
    try:
        resolved_root = store_root.resolve(strict=True)
        resolved_active = active_dir.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_active.relative_to(resolved_root)
        resolved_candidate.relative_to(resolved_active)
        before = candidate.stat()
        if not candidate.is_file():
            raise RecordStoreError(f"active closure {label} is not a regular file")
        raw = candidate.read_bytes()
        after = candidate.stat()
    except RecordStoreError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise RecordStoreError(
            f"active closure {label} is missing, unreadable, or unsafe"
        ) from exc
    if candidate.is_symlink() or _stat_identity(before) != _stat_identity(after):
        raise RecordStoreError(f"active closure {label} changed during read")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != raw_sha256:
        raise RecordStoreError(
            f"active closure {label} sha256 mismatch: {actual_sha256} != {raw_sha256}"
        )
    return resolved_candidate, raw


def _read_table(path: Path, *, columns: list[str] | None = None):
    import pandas as pd  # type: ignore[import-not-found]

    if path.is_dir():
        path = path / "part.parquet"
    if path.suffix.lower() != ".parquet":
        raise RuntimeError(f"strict Parquet table required: {path}")
    with _suppress_native_stderr():
        return pd.read_parquet(path, columns=columns)


def load_latest_holding_baseline(record_root: Path = DEFAULT_RECORD_ROOT) -> HoldingBaseline:
    import pandas as pd  # type: ignore[import-not-found]

    store_root = _record_store_root(record_root)
    registered = load_registered_catalog(store_root)
    if registered is None:
        raise RecordStoreError("registered strategy catalog missing")
    pointer, _catalog = registered
    closure = pointer.get("active_closure")
    if not isinstance(closure, dict):
        raise RecordStoreError("active strategy-record closure missing")
    active_dirs = resolve_active_record_dirs(store_root)
    if not active_dirs:
        raise RecordStoreError("active strategy-record directory missing")
    active_dir = active_dirs[0]
    manifest_path, manifest_raw = _contained_regular_file(
        store_root=store_root,
        active_dir=active_dir,
        relative_path=closure.get("manual_manifest_path"),
        declared_sha256=closure.get("manual_manifest_sha256"),
        label="manual manifest",
    )
    ledger_path, ledger_raw = _contained_regular_file(
        store_root=store_root,
        active_dir=active_dir,
        relative_path=closure.get("ledger_path"),
        declared_sha256=closure.get("ledger_sha256"),
        label="ledger",
    )
    if ledger_path.name != "ledger_after_manual_switch.parquet":
        raise RecordStoreError("active holding ledger must be ledger_after_manual_switch.parquet")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordStoreError("active manual execution manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or not _manual_manifest_is_valid_baseline(manifest):
        raise RecordStoreError("active manual execution manifest is not a valid baseline")
    next_ledger = str(manifest.get("next_ledger_path") or "").strip()
    if next_ledger:
        declared_ledger = Path(next_ledger)
        if declared_ledger.is_absolute() or ".." in declared_ledger.parts:
            raise RecordStoreError("manual manifest next_ledger_path is unsafe")
        if declared_ledger.name != ledger_path.name:
            raise RecordStoreError("manual manifest ledger does not match active closure")
    with _suppress_native_stderr():
        ledger = pd.read_parquet(io.BytesIO(ledger_raw))
    required = {"symbol", "shares", "avg_cost"}
    missing = required - set(ledger.columns)
    if missing:
        raise RuntimeError(
            f"effective ledger schema missing {sorted(missing)}: {ledger_path}"
        )
    ledger = ledger.copy()
    ledger["symbol"] = ledger["symbol"].map(_normalize_symbol)
    return HoldingBaseline(
        ledger=ledger,
        manifest=manifest,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
    )


def _latest_by_symbol(frame: Any, symbols: set[str], date_columns: Iterable[str]):
    if frame is None or frame.empty or "ts_code" not in frame.columns:
        return {}
    working = frame.copy()
    working["ts_code"] = working["ts_code"].map(_normalize_symbol)
    working = working[working["ts_code"].isin(symbols)].copy()
    if working.empty:
        return {}
    sort_cols = ["ts_code"] + [col for col in date_columns if col in working.columns]
    if len(sort_cols) > 1:
        for col in sort_cols[1:]:
            working[col] = working[col].astype(str)
        working = working.sort_values(sort_cols)
    return {symbol: group.iloc[-1].to_dict() for symbol, group in working.groupby("ts_code", sort=True)}


def _read_optional_parquet_table(root: Path, relative: str):
    path = root / relative
    if not path.exists():
        return None
    try:
        return _read_table(path)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _clean_date(value: Any) -> str:
    text = "".join(ch for ch in str(value or "") if ch.isdigit())
    return text[:8] if len(text) >= 8 else ""


def _iso_date(value: Any) -> str:
    text = _clean_date(value)
    if len(text) != 8:
        return ""
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _date_distance_days(as_of: str, value: str) -> int | None:
    left = _clean_date(as_of)
    right = _clean_date(value)
    if len(left) != 8 or len(right) != 8:
        return None
    try:
        return abs((datetime.strptime(right, "%Y%m%d") - datetime.strptime(left, "%Y%m%d")).days)
    except ValueError:
        return None


def load_default_industry_verdicts(as_of: str = DEFAULT_AS_OF) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "name": name,
            "verdict": verdict,
            "as_of": _iso_date(as_of),
            "source": "maxwell_manual",
            "note": "",
        }
        for symbol, (name, verdict) in sorted(MAXWELL_VERDICTS.items())
    ]


def _h1_disclosure_by_symbol(disclosure_root: Path, symbols: set[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not disclosure_root.exists():
        return {}, sorted(symbols)
    try:
        frame = _read_table(disclosure_root)
    except Exception:
        return {}, sorted(symbols)
    if frame.empty or not {"ts_code", "end_date"}.issubset(set(frame.columns)):
        return {}, sorted(symbols)
    frame = frame.copy()
    frame["ts_code"] = frame["ts_code"].map(_normalize_symbol)
    frame["end_date"] = frame["end_date"].astype(str).str.replace("-", "", regex=False)
    frame = frame[(frame["ts_code"].isin(symbols)) & (frame["end_date"] == H1_END_DATE)]
    if frame.empty:
        return {}, sorted(symbols)
    sort_cols = [col for col in ("ts_code", "actual_date", "pre_date", "ann_date") if col in frame.columns]
    frame = frame.sort_values(sort_cols)
    result: dict[str, dict[str, Any]] = {}
    for symbol, group in frame.groupby("ts_code", sort=True):
        latest = group.iloc[-1].to_dict()
        disclosure_date = _clean_date(latest.get("actual_date")) or _clean_date(latest.get("pre_date"))
        result[symbol] = {
            "disclosure_date": _iso_date(disclosure_date) if disclosure_date else DISCLOSURE_PENDING,
            "disclosure_date_source": "actual_date" if _clean_date(latest.get("actual_date")) else (
                "pre_date" if disclosure_date else "missing_local_disclosure_date"
            ),
            "h1_end_date": _iso_date(H1_END_DATE),
            "ann_date": _iso_date(latest.get("ann_date")),
        }
    pending = sorted(symbols - set(result))
    return result, pending


def build_fundamental_rows(
    holdings: Any,
    *,
    fundamentals_root: Path = DEFAULT_FUNDAMENTALS_ROOT,
    disclosure_root: Path = DEFAULT_DISCLOSURE_ROOT,
    as_of: str = DEFAULT_AS_OF,
    near_days: int = 14,
) -> dict[str, Any]:
    symbols = {_normalize_symbol(value) for value in holdings["symbol"].tolist()}
    symbols.discard("")
    indicator = _read_optional_parquet_table(fundamentals_root, "table=fina_indicator")
    income = _read_optional_parquet_table(fundamentals_root, "table=income")
    forecast = _read_optional_parquet_table(fundamentals_root, "table=forecast")
    indicator_latest = _latest_by_symbol(indicator, symbols, ("ann_date", "end_date"))
    income_latest = _latest_by_symbol(income, symbols, ("ann_date", "f_ann_date", "end_date"))
    forecast_latest = _latest_by_symbol(forecast, symbols, ("ann_date", "end_date"))
    disclosures, pending_disclosure = _h1_disclosure_by_symbol(disclosure_root, symbols)
    verdicts = {row["symbol"]: row for row in load_default_industry_verdicts(as_of)}
    rows: list[dict[str, Any]] = []
    for row in holdings.sort_values("symbol").to_dict(orient="records"):
        symbol = _normalize_symbol(row.get("symbol"))
        indicator_row = indicator_latest.get(symbol, {})
        income_row = income_latest.get(symbol, {})
        forecast_row = forecast_latest.get(symbol, {})
        disclosure = disclosures.get(
            symbol,
            {
                "disclosure_date": DISCLOSURE_PENDING,
                "disclosure_date_source": "missing_local_disclosure_date",
                "h1_end_date": _iso_date(H1_END_DATE),
                "ann_date": "",
            },
        )
        verdict = verdicts.get(
            symbol,
            {
                "symbol": symbol,
                "name": row.get("name") or "",
                "verdict": "",
                "as_of": _iso_date(as_of),
                "source": "maxwell_manual",
                "note": "no_manual_verdict",
            },
        )
        disclosure_value = disclosure["disclosure_date"]
        near = _date_distance_days(as_of, disclosure_value)
        disclosure_pending_or_near = disclosure_value == DISCLOSURE_PENDING or (
            near is not None and near <= near_days
        )
        high_scrutiny = verdict.get("verdict") in HIGH_SCRUTINY_VERDICTS and disclosure_pending_or_near
        rows.append(
            {
                "symbol": symbol,
                "name": row.get("name") or verdict.get("name") or "",
                "shares": int(_safe_float(row.get("shares")) or 0),
                "market_weight": _safe_float(row.get("market_weight")),
                "latest_period": _iso_date(indicator_row.get("end_date") or income_row.get("end_date")),
                "ann_date": _iso_date(indicator_row.get("ann_date") or income_row.get("ann_date")),
                "revenue_yoy": _safe_float(
                    indicator_row.get("tr_yoy")
                    or indicator_row.get("revenue_yoy")
                    or indicator_row.get("oper_rev_yoy")
                ),
                "net_profit_yoy": _safe_float(
                    indicator_row.get("netprofit_yoy")
                    or indicator_row.get("netprofit_yoy_dt")
                    or indicator_row.get("profit_yoy")
                ),
                "net_income": _safe_float(income_row.get("n_income") or income_row.get("n_income_attr_p")),
                "forecast_type": forecast_row.get("type", ""),
                "forecast_summary": forecast_row.get("summary", ""),
                "forecast_ann_date": _iso_date(forecast_row.get("ann_date")),
                "h1_disclosure_date": disclosure_value,
                "h1_disclosure_date_source": disclosure["disclosure_date_source"],
                "industry_verdict": verdict.get("verdict", ""),
                "industry_verdict_source": verdict.get("source", ""),
                "high_scrutiny_earnings_risk": bool(high_scrutiny),
            }
        )
    return {
        "rows": rows,
        "pending_disclosure_symbols": pending_disclosure,
        "high_scrutiny_symbols": sorted(row["symbol"] for row in rows if row["high_scrutiny_earnings_risk"]),
        "local_data_sources": {
            "fundamentals_root": str(fundamentals_root),
            "disclosure_root": str(disclosure_root),
        },
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_index(payload: dict[str, Any]) -> str:
    lines = [
        "# Holdings Fundamental Tracking",
        "",
        "产业实质判断（真订单/真产能 vs 纯故事）由 Maxwell 人工完成；本表只提供本地数据。",
        "",
        f"- as_of: {payload['as_of']}",
        f"- baseline_record: {payload['baseline_record']}",
        f"- pending_disclosure_symbols: {', '.join(payload['pending_disclosure_symbols']) or 'none'}",
        f"- high_scrutiny_symbols: {', '.join(payload['high_scrutiny_symbols']) or 'none'}",
        "",
        "| symbol | name | period | ann_date | revenue_yoy | net_profit_yoy | H1 disclosure | verdict | high_scrutiny |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {symbol} | {name} | {latest_period} | {ann_date} | {revenue_yoy} | "
            "{net_profit_yoy} | {h1_disclosure_date} | {industry_verdict} | "
            "{high_scrutiny_earnings_risk} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    return "\n".join(lines) + "\n"


def _render_symbol_page(row: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = [
        f"# {row['symbol']} {row.get('name') or ''}".rstrip(),
        "",
        "产业实质判断（真订单/真产能 vs 纯故事）由 Maxwell 人工完成；本页只提供本地数据。",
        "",
        f"- as_of: {payload['as_of']}",
        f"- latest_period: {_fmt(row.get('latest_period'))}",
        f"- ann_date: {_fmt(row.get('ann_date'))}",
        f"- revenue_yoy: {_fmt(row.get('revenue_yoy'))}",
        f"- net_profit_yoy: {_fmt(row.get('net_profit_yoy'))}",
        f"- net_income: {_fmt(row.get('net_income'))}",
        f"- forecast_type: {_fmt(row.get('forecast_type'))}",
        f"- forecast_ann_date: {_fmt(row.get('forecast_ann_date'))}",
        f"- forecast_summary: {_fmt(row.get('forecast_summary'))}",
        f"- h1_disclosure_date: {_fmt(row.get('h1_disclosure_date'))}",
        f"- h1_disclosure_date_source: {_fmt(row.get('h1_disclosure_date_source'))}",
        f"- industry_verdict: {_fmt(row.get('industry_verdict'))}",
        f"- high_scrutiny_earnings_risk: {row.get('high_scrutiny_earnings_risk')}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fundamentals.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    verdict_path = output_dir / "industry_verdicts.json"
    verdict_path.write_text(
        json.dumps(payload["industry_verdicts"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index_path = output_dir / "index.md"
    index_path.write_text(_render_index(payload), encoding="utf-8")
    for row in payload["rows"]:
        page_path = output_dir / f"{row['symbol']}.md"
        page_path.write_text(_render_symbol_page(row, payload), encoding="utf-8")
    return {
        "fundamentals_json": str(json_path),
        "industry_verdicts_json": str(verdict_path),
        "index_md": str(index_path),
    }


def build_holdings_fundamental_sheet(
    *,
    record_root: Path = DEFAULT_RECORD_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fundamentals_root: Path = DEFAULT_FUNDAMENTALS_ROOT,
    disclosure_root: Path = DEFAULT_DISCLOSURE_ROOT,
    as_of: str = DEFAULT_AS_OF,
    near_days: int = 14,
    write: bool = True,
) -> dict[str, Any]:
    baseline = load_latest_holding_baseline(record_root)
    fundamentals = build_fundamental_rows(
        baseline.ledger,
        fundamentals_root=fundamentals_root,
        disclosure_root=disclosure_root,
        as_of=as_of,
        near_days=near_days,
    )
    output_dir = output_root / _clean_date(as_of) / "fundamentals"
    payload = {
        "schema_version": "holdings_fundamental_sheet.v1",
        "as_of": _iso_date(as_of),
        "baseline_record": baseline.manifest_path.parent.name,
        "baseline_manifest_path": str(baseline.manifest_path),
        "baseline_ledger_path": str(baseline.ledger_path),
        "industry_verdicts": load_default_industry_verdicts(as_of),
        **fundamentals,
        "output_dir": str(output_dir),
    }
    if write:
        payload["written_outputs"] = write_outputs(payload, output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fundamentals-root", type=Path, default=DEFAULT_FUNDAMENTALS_ROOT)
    parser.add_argument("--disclosure-root", type=Path, default=DEFAULT_DISCLOSURE_ROOT)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--near-days", type=int, default=14)
    args = parser.parse_args()
    payload = build_holdings_fundamental_sheet(
        record_root=args.record_root,
        output_root=args.output_root,
        fundamentals_root=args.fundamentals_root,
        disclosure_root=args.disclosure_root,
        as_of=args.as_of,
        near_days=args.near_days,
        write=True,
    )
    print(json.dumps(payload["written_outputs"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
