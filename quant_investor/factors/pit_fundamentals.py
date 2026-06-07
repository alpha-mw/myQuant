"""Point-in-time fundamental helpers for governed factor research.

The helper is intentionally source-backed and CSV based.  It does not depend on
the pyc-only DataHub layer, and it keeps ``fetched_at`` as an audit field rather
than a historical availability date when source announcement dates are present.
"""

from __future__ import annotations

import json
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PIT_COLUMNS = [
    "ts_code",
    "report_period",
    "availability_date",
    "metric_name",
    "value",
    "source",
    "fetched_at",
    "raw_table",
    "raw_field",
]

DEFAULT_METADATA_DIR = Path("data/metadata")
DEFAULT_FUNDAMENTAL_MART_ROOT = Path("data/clean/cn_fundamental")
DEFAULT_PIT_SERIES_FILENAME = "fundamental_pit_series.csv"
OPERATING_CASHFLOW_METRIC = "operating_cashflow"
NET_INCOME_METRIC = "net_income"
FIN_OCF_TO_PROFIT_METRIC = "fin_ocf_to_profit"
FUNDAMENTAL_METRICS = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "fcf_to_price",
)


@dataclass
class PitCoverageDiagnostics:
    symbols_requested: list[str] = field(default_factory=list)
    symbols_with_ocf_profit: list[str] = field(default_factory=list)
    ratio_rows: int = 0
    pit_rows: int = 0
    blocker: str = ""

    @property
    def coverage_rate(self) -> float:
        if not self.symbols_requested:
            return 0.0
        return len(self.symbols_with_ocf_profit) / len(self.symbols_requested)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols_requested": list(self.symbols_requested),
            "symbols_with_ocf_profit": list(self.symbols_with_ocf_profit),
            "coverage_rate": float(self.coverage_rate),
            "ratio_rows": int(self.ratio_rows),
            "pit_rows": int(self.pit_rows),
            "blocker": self.blocker,
        }


def normalize_ts_code(symbol: object) -> str:
    """Normalize local symbols to Tushare-style ``000001.SZ`` codes."""

    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        if right in {"SH", "SZ", "BJ"}:
            return f"{left.zfill(6)}.{right}"
        if left in {"SH", "SZ", "BJ"}:
            return f"{right.zfill(6)}.{left}"
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 6:
        return text
    code = digits[-6:]
    suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{suffix}"


def _metadata_dir(path: str | Path | None) -> Path:
    return Path(path or DEFAULT_METADATA_DIR).expanduser()


def _pit_path(metadata_dir: str | Path | None = None, pit_series_path: str | Path | None = None) -> Path:
    if pit_series_path is not None:
        return Path(pit_series_path).expanduser()
    return _metadata_dir(metadata_dir) / DEFAULT_PIT_SERIES_FILENAME


def _date_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    try:
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _period_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else text


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _record(
    *,
    ts_code: object,
    report_period: object,
    availability_date: object,
    metric_name: str,
    value: object,
    source: str,
    fetched_at: object = "",
    raw_table: str,
    raw_field: str,
) -> dict[str, Any] | None:
    symbol = normalize_ts_code(ts_code)
    period = _period_text(report_period)
    available = _date_text(availability_date)
    number = _safe_float(value)
    if not symbol or not period or not available or number is None:
        return None
    return {
        "ts_code": symbol,
        "report_period": period,
        "availability_date": available,
        "metric_name": str(metric_name),
        "value": float(number),
        "source": str(source),
        "fetched_at": str(fetched_at or ""),
        "raw_table": str(raw_table),
        "raw_field": str(raw_field),
    }


def _rows_from_pit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    for column in PIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[PIT_COLUMNS].copy()
    frame["ts_code"] = frame["ts_code"].map(normalize_ts_code)
    frame["report_period"] = frame["report_period"].map(_period_text)
    frame["availability_date"] = pd.to_datetime(frame["availability_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["availability_date", "value"])
    frame = frame[(frame["ts_code"].astype(str) != "") & (frame["report_period"].astype(str) != "")]
    frame["availability_date"] = frame["availability_date"].dt.strftime("%Y-%m-%d")
    for column in ("metric_name", "source", "fetched_at", "raw_table", "raw_field"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame.to_dict("records")


def _rows_from_fundamental_mart(root: str | Path | None) -> list[dict[str, Any]]:
    base = Path(root or DEFAULT_FUNDAMENTAL_MART_ROOT).expanduser()
    period_path = base / "fundamental_period.csv"
    if not period_path.exists():
        return []
    frame = pd.read_csv(period_path)
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        for metric in FUNDAMENTAL_METRICS:
            if metric == "fcf_to_price" or metric not in frame.columns:
                continue
            row = _record(
                ts_code=item.get("ts_code"),
                report_period=item.get("end_date") or item.get("report_period"),
                availability_date=item.get("availability_date"),
                metric_name=metric,
                value=item.get(metric),
                source=str(item.get("source", "cn_fundamental_mart")),
                fetched_at=item.get("fetched_at", ""),
                raw_table="fundamental_period",
                raw_field=metric,
            )
            if row:
                rows.append(row)
    return rows


def _rows_from_fundamental_series(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    metric_map = {
        "operating_cashflow": OPERATING_CASHFLOW_METRIC,
        "net_income": NET_INCOME_METRIC,
    }
    for _, item in frame.iterrows():
        raw_metric = str(item.get("metric_name", "")).strip()
        metric = metric_map.get(raw_metric)
        if metric is None:
            continue
        fetched_at = str(item.get("fetched_at", "") or "")
        row = _record(
            ts_code=item.get("ts_code"),
            report_period=item.get("period"),
            availability_date=fetched_at,
            metric_name=metric,
            value=item.get("value"),
            source=f"{item.get('source', 'local_metadata')}:fetched_at_fallback",
            fetched_at=fetched_at,
            raw_table="fundamental_series",
            raw_field=raw_metric,
        )
        if row:
            rows.append(row)
    return rows


def _iter_raw_json_records(raw_json: object) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if raw_json is None or (isinstance(raw_json, float) and np.isnan(raw_json)):
        return
    try:
        payload = json.loads(str(raw_json))
    except Exception:
        return
    if not isinstance(payload, Mapping):
        return
    for table in ("income", "cashflow"):
        records = payload.get(table)
        if isinstance(records, Mapping):
            yield table, records
        elif isinstance(records, list):
            for item in records:
                if isinstance(item, Mapping):
                    yield table, item


def _rows_from_snapshot_raw_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, snapshot in frame.iterrows():
        for table, item in _iter_raw_json_records(snapshot.get("raw_json")):
            ts_code = item.get("ts_code") or snapshot.get("ts_code")
            period = item.get("end_date") or snapshot.get("report_period")
            available = item.get("f_ann_date") or item.get("ann_date")
            fetched_at = snapshot.get("fetched_at", "")
            if table == "cashflow":
                row = _record(
                    ts_code=ts_code,
                    report_period=period,
                    availability_date=available,
                    metric_name=OPERATING_CASHFLOW_METRIC,
                    value=item.get("n_cashflow_act"),
                    source="fundamental_snapshots.raw_json",
                    fetched_at=fetched_at,
                    raw_table="cashflow",
                    raw_field="n_cashflow_act",
                )
                if row:
                    rows.append(row)
            if table == "income":
                value = item.get("n_income")
                if value is None:
                    value = item.get("n_income_attr_p")
                row = _record(
                    ts_code=ts_code,
                    report_period=period,
                    availability_date=available,
                    metric_name=NET_INCOME_METRIC,
                    value=value,
                    source="fundamental_snapshots.raw_json",
                    fetched_at=fetched_at,
                    raw_table="income",
                    raw_field="n_income",
                )
                if row:
                    rows.append(row)
    return rows


def _rows_from_snapshot_flat(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        fetched_at = str(item.get("fetched_at", "") or "")
        for column, metric in (
            ("operating_cashflow", OPERATING_CASHFLOW_METRIC),
            ("net_income", NET_INCOME_METRIC),
        ):
            if column not in frame.columns:
                continue
            row = _record(
                ts_code=item.get("ts_code"),
                report_period=item.get("report_period"),
                availability_date=fetched_at,
                metric_name=metric,
                value=item.get(column),
                source="fundamental_snapshots:fetched_at_fallback",
                fetched_at=fetched_at,
                raw_table="fundamental_snapshots",
                raw_field=column,
            )
            if row:
                rows.append(row)
    return rows


def load_fundamental_pit_series(
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
    *,
    mart_root: str | Path | None = None,
    allow_legacy_fallback: bool = True,
) -> pd.DataFrame:
    """Load PIT rows.

    ``mart_root`` is the production path. Legacy metadata fallbacks are only
    diagnostic and can be disabled so ``fetched_at`` never becomes a production
    availability date.
    """

    base_dir = _metadata_dir(metadata_dir)
    pit_path = _pit_path(base_dir, pit_series_path)
    rows: list[dict[str, Any]] = []
    rows.extend(_rows_from_fundamental_mart(mart_root))
    rows.extend(_rows_from_pit(pit_path))
    if allow_legacy_fallback:
        rows.extend(_rows_from_snapshot_raw_json(base_dir / "fundamental_snapshots.csv"))
        rows.extend(_rows_from_snapshot_flat(base_dir / "fundamental_snapshots.csv"))
        rows.extend(_rows_from_fundamental_series(base_dir / "fundamental_series.csv"))
    if not rows:
        return pd.DataFrame(columns=PIT_COLUMNS)
    frame = pd.DataFrame(rows, columns=PIT_COLUMNS)
    frame["ts_code"] = frame["ts_code"].map(normalize_ts_code)
    frame["availability_date"] = pd.to_datetime(frame["availability_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "report_period", "availability_date", "value"])
    frame = frame.sort_values(
        ["ts_code", "metric_name", "report_period", "availability_date", "source"]
    )
    return frame.drop_duplicates(
        subset=["ts_code", "metric_name", "report_period", "availability_date", "raw_table"],
        keep="last",
    ).reset_index(drop=True)


def write_fundamental_pit_series(
    rows: Sequence[Mapping[str, Any]],
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
) -> Path:
    """Merge new PIT rows into the canonical CSV and return the written path."""

    path = _pit_path(metadata_dir, pit_series_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _rows_from_pit(path)
    normalized: list[dict[str, Any]] = []
    for item in [*existing, *list(rows)]:
        row = _record(
            ts_code=item.get("ts_code"),
            report_period=item.get("report_period"),
            availability_date=item.get("availability_date"),
            metric_name=str(item.get("metric_name", "")),
            value=item.get("value"),
            source=str(item.get("source", "")),
            fetched_at=item.get("fetched_at", ""),
            raw_table=str(item.get("raw_table", "")),
            raw_field=str(item.get("raw_field", "")),
        )
        if row:
            normalized.append(row)
    frame = pd.DataFrame(normalized, columns=PIT_COLUMNS)
    if frame.empty:
        frame = pd.DataFrame(columns=PIT_COLUMNS)
    else:
        frame = frame.sort_values(
            ["ts_code", "metric_name", "report_period", "availability_date", "source"]
        ).drop_duplicates(
            subset=[
                "ts_code",
                "report_period",
                "availability_date",
                "metric_name",
                "raw_table",
                "raw_field",
            ],
            keep="last",
        )
    frame.to_csv(path, index=False)
    return path


def _fin_ocf_to_profit_from_daily_mart(
    *,
    date_index: pd.DatetimeIndex,
    normalized_symbols: Sequence[str],
    mart_root: str | Path | None,
) -> tuple[pd.DataFrame, PitCoverageDiagnostics, bool]:
    matrix = pd.DataFrame(index=date_index, columns=normalized_symbols, dtype=float)
    diagnostics = PitCoverageDiagnostics(symbols_requested=list(normalized_symbols))
    daily_path = Path(mart_root or DEFAULT_FUNDAMENTAL_MART_ROOT).expanduser() / "fundamental_daily.csv"
    if not daily_path.exists():
        return matrix, diagnostics, False

    needed = {"ts_code", "trade_date", FIN_OCF_TO_PROFIT_METRIC}
    try:
        header = set(pd.read_csv(daily_path, nrows=0).columns)
    except Exception:
        return matrix, diagnostics, False
    if not needed.issubset(header):
        return matrix, diagnostics, False

    daily = pd.read_csv(daily_path, usecols=list(needed))
    diagnostics.pit_rows = int(len(daily))
    if daily.empty:
        diagnostics.blocker = "empty_fundamental_daily"
        return matrix, diagnostics, False

    symbol_set = set(normalized_symbols)
    daily["ts_code"] = daily["ts_code"].map(normalize_ts_code)
    daily = daily[daily["ts_code"].isin(symbol_set)].copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    daily[FIN_OCF_TO_PROFIT_METRIC] = pd.to_numeric(
        daily[FIN_OCF_TO_PROFIT_METRIC],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(subset=["ts_code", "trade_date", FIN_OCF_TO_PROFIT_METRIC])
    diagnostics.ratio_rows = int(len(daily))
    if daily.empty:
        diagnostics.blocker = "daily_mart_fin_ocf_to_profit_unavailable"
        return matrix, diagnostics, False

    wide = daily.pivot_table(
        index="trade_date",
        columns="ts_code",
        values=FIN_OCF_TO_PROFIT_METRIC,
        aggfunc="last",
    ).sort_index()
    matrix = wide.reindex(date_index).reindex(columns=normalized_symbols)
    symbols_with_ratio = [str(symbol) for symbol in matrix.columns if matrix[symbol].notna().any()]
    diagnostics.symbols_with_ocf_profit = symbols_with_ratio
    if not symbols_with_ratio:
        diagnostics.blocker = "fin_ocf_to_profit_unavailable"
        return matrix, diagnostics, False
    return matrix, diagnostics, True


def build_fin_ocf_to_profit_matrix(
    dates: Sequence[object] | pd.DatetimeIndex,
    symbols: Sequence[object],
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
    *,
    mart_root: str | Path | None = None,
    allow_legacy_fallback: bool = True,
) -> tuple[pd.DataFrame, PitCoverageDiagnostics]:
    """Build a date x symbol PIT matrix for ``fin_ocf_to_profit``.

    Values become visible only from ``availability_date`` onward and are
    forward-filled until a later available filing supersedes them.
    """

    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().unique()
    date_index = pd.DatetimeIndex(sorted(date_index))
    normalized_symbols = [normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)]
    normalized_symbols = list(dict.fromkeys(normalized_symbols))
    matrix = pd.DataFrame(index=date_index, columns=normalized_symbols, dtype=float)
    diagnostics = PitCoverageDiagnostics(symbols_requested=normalized_symbols)
    daily_matrix, daily_diagnostics, daily_used = _fin_ocf_to_profit_from_daily_mart(
        date_index=date_index,
        normalized_symbols=normalized_symbols,
        mart_root=mart_root,
    )
    if daily_used:
        return daily_matrix, daily_diagnostics
    pit = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        pit_series_path=pit_series_path,
        mart_root=mart_root,
        allow_legacy_fallback=allow_legacy_fallback,
    )
    diagnostics.pit_rows = int(len(pit))
    if pit.empty or date_index.empty or not normalized_symbols:
        diagnostics.blocker = "missing_fundamental_pit_series"
        return matrix, diagnostics

    pit = pit[pit["ts_code"].isin(normalized_symbols)].copy()
    if pit.empty:
        diagnostics.blocker = "no_requested_symbols_in_pit_series"
        return matrix, diagnostics

    ratio_frames: list[pd.DataFrame] = []
    direct = pit[pit["metric_name"] == FIN_OCF_TO_PROFIT_METRIC][
        ["ts_code", "report_period", "availability_date", "value"]
    ].copy()
    if not direct.empty:
        direct = direct.rename(columns={"value": "ratio"})
        ratio_frames.append(direct)

    base = pit[pit["metric_name"].isin([OPERATING_CASHFLOW_METRIC, NET_INCOME_METRIC])]
    if not base.empty:
        ocf = (
            base[base["metric_name"] == OPERATING_CASHFLOW_METRIC][
                ["ts_code", "report_period", "availability_date", "value"]
            ]
            .rename(columns={"availability_date": "ocf_availability_date", "value": "ocf"})
            .dropna()
            .sort_values(["ts_code", "report_period", "ocf_availability_date"])
            .drop_duplicates(["ts_code", "report_period"], keep="last")
        )
        profit = (
            base[base["metric_name"] == NET_INCOME_METRIC][
                ["ts_code", "report_period", "availability_date", "value"]
            ]
            .rename(columns={"availability_date": "profit_availability_date", "value": "profit"})
            .dropna()
            .sort_values(["ts_code", "report_period", "profit_availability_date"])
            .drop_duplicates(["ts_code", "report_period"], keep="last")
        )
        merged = ocf.merge(profit, on=["ts_code", "report_period"], how="inner")
        if not merged.empty:
            denominator = pd.to_numeric(merged["profit"], errors="coerce")
            numerator = pd.to_numeric(merged["ocf"], errors="coerce")
            derived = merged[["ts_code", "report_period"]].copy()
            derived["availability_date"] = pd.concat(
                [
                    pd.to_datetime(merged["ocf_availability_date"], errors="coerce"),
                    pd.to_datetime(merged["profit_availability_date"], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1)
            derived["ratio"] = numerator.div(denominator.where(denominator.abs() > 1e-12))
            ratio_frames.append(derived)

    if ratio_frames:
        ratio_all = pd.concat(ratio_frames, ignore_index=True)
    else:
        ratio_all = pd.DataFrame(columns=["ts_code", "report_period", "availability_date", "ratio"])
    ratio_all["availability_date"] = pd.to_datetime(ratio_all["availability_date"], errors="coerce")
    ratio_all["ratio"] = pd.to_numeric(ratio_all["ratio"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    ratio_all = ratio_all.dropna(subset=["ts_code", "availability_date", "ratio"])
    ratio_all = ratio_all.sort_values(["ts_code", "availability_date", "report_period"])
    ratio_all = ratio_all.drop_duplicates(["ts_code", "availability_date"], keep="last")
    diagnostics.ratio_rows = int(len(ratio_all))
    if not ratio_all.empty:
        wide = ratio_all.pivot_table(
            index="availability_date",
            columns="ts_code",
            values="ratio",
            aggfunc="last",
        ).sort_index()
        wide = wide.reindex(columns=normalized_symbols)
        matrix = (
            wide.reindex(date_index.union(wide.index))
            .sort_index()
            .ffill()
            .reindex(date_index)
            .reindex(columns=normalized_symbols)
        )

    symbols_with_ratio = [str(symbol) for symbol in matrix.columns if matrix[symbol].notna().any()]
    diagnostics.symbols_with_ocf_profit = symbols_with_ratio
    if not symbols_with_ratio:
        diagnostics.blocker = "fin_ocf_to_profit_unavailable"
    return matrix, diagnostics


def _daily_mart_metric_matrices(
    *,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    metrics: Sequence[str],
    mart_root: str | Path | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    root = Path(mart_root or DEFAULT_FUNDAMENTAL_MART_ROOT).expanduser()
    daily_path = root / "fundamental_daily.csv"
    matrices = {metric: pd.DataFrame(index=dates, columns=symbols, dtype=float) for metric in metrics}
    diagnostics: dict[str, Any] = {"mart_root": str(root), "daily_rows": 0, "blocker": ""}
    if not daily_path.exists():
        diagnostics["blocker"] = "missing_fundamental_daily"
        return matrices, diagnostics
    try:
        header = set(pd.read_csv(daily_path, nrows=0).columns)
    except Exception as exc:
        diagnostics["blocker"] = f"read_fundamental_daily_header_failed:{exc}"
        return matrices, diagnostics
    available_metrics = [metric for metric in metrics if metric in header]
    if not {"ts_code", "trade_date"}.issubset(header):
        diagnostics["blocker"] = "fundamental_daily_missing_key_columns"
        return matrices, diagnostics
    if not available_metrics:
        diagnostics["blocker"] = "fundamental_daily_missing_requested_metrics"
        return matrices, diagnostics
    daily = pd.read_csv(daily_path, usecols=["ts_code", "trade_date", *available_metrics])
    diagnostics["daily_rows"] = int(len(daily))
    if daily.empty:
        diagnostics["blocker"] = "empty_fundamental_daily"
        return matrices, diagnostics
    daily["ts_code"] = daily["ts_code"].map(normalize_ts_code)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    daily = daily.dropna(subset=["ts_code", "trade_date"])
    daily = daily[daily["ts_code"].isin(set(symbols))]
    for metric in available_metrics:
        metric_frame = daily[["ts_code", "trade_date", metric]].copy()
        metric_frame[metric] = pd.to_numeric(metric_frame[metric], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        metric_frame = metric_frame.dropna(subset=[metric])
        if metric_frame.empty:
            continue
        wide = metric_frame.pivot_table(
            index="trade_date",
            columns="ts_code",
            values=metric,
            aggfunc="last",
        ).sort_index()
        matrices[metric] = wide.reindex(dates).reindex(columns=symbols)
    return matrices, diagnostics


def build_fundamental_metric_matrices(
    dates: Sequence[object] | pd.DatetimeIndex,
    symbols: Sequence[object],
    metrics: Sequence[str] = FUNDAMENTAL_METRICS,
    *,
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
    mart_root: str | Path | None = None,
    allow_legacy_fallback: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build date x symbol PIT matrices for requested fundamental metrics."""

    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().unique()
    date_index = pd.DatetimeIndex(sorted(date_index))
    normalized_symbols = [normalize_ts_code(symbol) for symbol in symbols if normalize_ts_code(symbol)]
    normalized_symbols = list(dict.fromkeys(normalized_symbols))
    requested_metrics = [str(metric) for metric in metrics if str(metric)]
    matrices, daily_diag = _daily_mart_metric_matrices(
        dates=date_index,
        symbols=normalized_symbols,
        metrics=requested_metrics,
        mart_root=mart_root,
    )
    if allow_legacy_fallback:
        pit = load_fundamental_pit_series(
            metadata_dir=metadata_dir,
            pit_series_path=pit_series_path,
            mart_root=mart_root,
            allow_legacy_fallback=True,
        )
    else:
        pit = pd.DataFrame(columns=PIT_COLUMNS)
    diagnostics: dict[str, Any] = {
        "symbols_requested": normalized_symbols,
        "metrics_requested": requested_metrics,
        "daily": daily_diag,
        "pit_rows": int(len(pit)),
        "coverage_by_metric": {},
        "legacy_fallback_allowed": bool(allow_legacy_fallback),
    }
    if not date_index.empty and normalized_symbols and not pit.empty:
        pit = pit[pit["ts_code"].isin(normalized_symbols) & pit["metric_name"].isin(requested_metrics)].copy()
        for (metric, symbol), group in pit.groupby(["metric_name", "ts_code"]):
            if metric not in matrices or symbol not in matrices[metric].columns:
                continue
            existing = matrices[metric][symbol]
            group = group.sort_values(["availability_date", "report_period"])
            series = pd.Series(
                pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(group["availability_date"], errors="coerce"),
            )
            series = series.dropna()
            series = series[~series.index.duplicated(keep="last")]
            aligned = series.sort_index().reindex(date_index, method="ffill")
            matrices[metric][symbol] = existing.combine_first(aligned)
    if FIN_OCF_TO_PROFIT_METRIC in matrices:
        ratio_matrix, ratio_diag = build_fin_ocf_to_profit_matrix(
            date_index,
            normalized_symbols,
            metadata_dir=metadata_dir,
            pit_series_path=pit_series_path,
            mart_root=mart_root,
            allow_legacy_fallback=allow_legacy_fallback,
        )
        matrices[FIN_OCF_TO_PROFIT_METRIC] = matrices[FIN_OCF_TO_PROFIT_METRIC].combine_first(
            ratio_matrix
        )
        diagnostics["fin_ocf_to_profit_ratio_fallback"] = ratio_diag.to_dict()
    for metric, matrix in matrices.items():
        total = max(matrix.size, 1)
        diagnostics["coverage_by_metric"][metric] = float(matrix.notna().sum().sum() / total)
    return matrices, diagnostics


def _tushare_financial_records(
    *,
    ts_code: str,
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    fetched_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_name, frame, metric, raw_field in (
        ("income", income, NET_INCOME_METRIC, "n_income"),
        ("cashflow", cashflow, OPERATING_CASHFLOW_METRIC, "n_cashflow_act"),
    ):
        if frame is None or frame.empty:
            continue
        for _, item in frame.iterrows():
            value = item.get(raw_field)
            if table_name == "income" and value is None:
                value = item.get("n_income_attr_p")
            row = _record(
                ts_code=item.get("ts_code") or ts_code,
                report_period=item.get("end_date"),
                availability_date=item.get("f_ann_date") or item.get("ann_date"),
                metric_name=metric,
                value=value,
                source="tushare_backfill",
                fetched_at=fetched_at,
                raw_table=table_name,
                raw_field=raw_field,
            )
            if row:
                rows.append(row)
    return rows


@contextmanager
def _time_limit(seconds: float | None) -> Iterable[None]:
    if seconds is None or seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Tushare request timed out after {seconds:.1f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _call_tushare_table(pro: Any, table_name: str, timeout_seconds: float | None, **kwargs: Any) -> pd.DataFrame:
    with _time_limit(timeout_seconds):
        value = getattr(pro, table_name)(**kwargs)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def append_tushare_financial_pit_series(
    symbols: Sequence[object],
    *,
    start_date: str | int | None = None,
    end_date: str | int | None = None,
    metadata_dir: str | Path | None = None,
    pit_series_path: str | Path | None = None,
    pro: Any | None = None,
    rate_limit_per_min: int | None = None,
    request_timeout_seconds: float | None = 20.0,
    max_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Backfill PIT financial rows with Tushare ``income`` and ``cashflow``.

    Tokens are never written to disk; the returned manifest only records provider
    status, symbols, row counts, and failures.
    """

    normalized_symbols = list(dict.fromkeys(normalize_ts_code(symbol) for symbol in symbols))
    normalized_symbols = [symbol for symbol in normalized_symbols if symbol]
    fetched_at = _now_iso()
    manifest: dict[str, Any] = {
        "provider": "tushare",
        "status": "not_started",
        "symbols_requested": normalized_symbols,
        "symbols_fetched": [],
        "rows_written": 0,
        "failures": {},
        "fetched_at": fetched_at,
        "token_persisted": False,
        "request_timeout_seconds": request_timeout_seconds,
        "max_elapsed_seconds": max_elapsed_seconds,
    }
    if not normalized_symbols:
        manifest["status"] = "empty_symbol_set"
        return manifest
    if pro is None:
        try:
            import tushare as ts  # type: ignore

            from quant_investor.config import config
            from quant_investor.credential_utils import create_tushare_pro

            if not config.TUSHARE_TOKEN:
                manifest["status"] = "provider_unavailable"
                manifest["failures"]["provider"] = "missing TUSHARE_TOKEN"
                return manifest
            pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
            if rate_limit_per_min is None:
                rate_limit_per_min = int(config.TUSHARE_RATE_LIMIT_PER_MIN or 0)
        except Exception as exc:
            manifest["status"] = "provider_unavailable"
            manifest["failures"]["provider"] = str(exc)
            return manifest

    income_fields = ",".join(
        [
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "end_type",
            "n_income",
            "n_income_attr_p",
            "total_revenue",
            "update_flag",
        ]
    )
    cashflow_fields = ",".join(
        [
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "end_type",
            "n_cashflow_act",
            "net_profit",
            "free_cashflow",
            "update_flag",
        ]
    )
    sleep_seconds = 0.0
    if rate_limit_per_min and rate_limit_per_min > 0:
        sleep_seconds = max(120.0 / float(rate_limit_per_min), 0.0)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, symbol in enumerate(normalized_symbols):
        if max_elapsed_seconds is not None and time.monotonic() - started > max_elapsed_seconds:
            manifest["failures"]["provider_timeout"] = (
                f"max_elapsed_seconds exceeded after {index} symbols"
            )
            break
        try:
            income = _call_tushare_table(
                pro,
                "income",
                request_timeout_seconds,
                ts_code=symbol,
                start_date=str(start_date or ""),
                end_date=str(end_date or ""),
                fields=income_fields,
            )
            cashflow = _call_tushare_table(
                pro,
                "cashflow",
                request_timeout_seconds,
                ts_code=symbol,
                start_date=str(start_date or ""),
                end_date=str(end_date or ""),
                fields=cashflow_fields,
            )
            rows.extend(
                _tushare_financial_records(
                    ts_code=symbol,
                    income=income,
                    cashflow=cashflow,
                    fetched_at=fetched_at,
                )
            )
            manifest["symbols_fetched"].append(symbol)
        except Exception as exc:
            manifest["failures"][symbol] = str(exc)
        if sleep_seconds and index < len(normalized_symbols) - 1:
            time.sleep(sleep_seconds)

    if rows:
        write_fundamental_pit_series(rows, metadata_dir=metadata_dir, pit_series_path=pit_series_path)
    manifest["rows_written"] = len(rows)
    manifest["status"] = "ok" if rows else "no_rows"
    if manifest["failures"] and rows:
        manifest["status"] = "partial"
    elif manifest["failures"]:
        manifest["status"] = "failed"
    return manifest


__all__ = [
    "DEFAULT_PIT_SERIES_FILENAME",
    "DEFAULT_FUNDAMENTAL_MART_ROOT",
    "FUNDAMENTAL_METRICS",
    "FIN_OCF_TO_PROFIT_METRIC",
    "NET_INCOME_METRIC",
    "OPERATING_CASHFLOW_METRIC",
    "PIT_COLUMNS",
    "PitCoverageDiagnostics",
    "append_tushare_financial_pit_series",
    "build_fin_ocf_to_profit_matrix",
    "build_fundamental_metric_matrices",
    "load_fundamental_pit_series",
    "normalize_ts_code",
    "write_fundamental_pit_series",
]
