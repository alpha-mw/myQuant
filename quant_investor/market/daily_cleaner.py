"""Offline daily-bar cleaner for local CN/US market CSV snapshots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "market-daily-clean.v1"
CN_CATEGORY_PRIORITY = ("hs300", "zz500", "zz1000", "other")
US_CATEGORY_PRIORITY = ("full_us", "large_cap", "mid_cap", "small_cap")
BLOCKING_ROW_ISSUES = (
    "invalid_date",
    "missing_price",
    "missing_volume",
    "nonpositive_price",
    "negative_volume",
    "ohlc_inconsistent",
)


@dataclass(frozen=True)
class DailyCleanConfig:
    market: str
    raw_dir: Path
    clean_dir: Path
    audit_dir: Path
    latest_required_date: str | None = None
    write_clean: bool = True
    min_valid_date: str | None = None
    max_quarantine_rows_per_symbol: int = 5000


@dataclass
class FileCleanResult:
    market: str
    symbol: str
    category: str
    source_path: str
    output_path: str | None
    raw_rows: int = 0
    clean_rows: int = 0
    dropped_rows: int = 0
    duplicate_date_rows: int = 0
    latest_raw_date: str | None = None
    latest_clean_date: str | None = None
    issue_counts: dict[str, int] = field(default_factory=dict)
    quarantine_reasons: list[str] = field(default_factory=list)

    @property
    def has_quarantine_reasons(self) -> bool:
        return bool(self.quarantine_reasons)


def clean_market_daily_data(config: DailyCleanConfig) -> dict[str, Any]:
    """Clean local market CSVs into a separate clean layer plus audit files."""

    cleaner = MarketDailyCleaner(config)
    return cleaner.run()


class MarketDailyCleaner:
    def __init__(self, config: DailyCleanConfig) -> None:
        self.config = config
        self.market = config.market.upper()
        if self.market not in {"CN", "US"}:
            raise ValueError(f"Unsupported market: {config.market!r}")
        self.raw_dir = Path(config.raw_dir)
        self.clean_dir = Path(config.clean_dir)
        self.audit_dir = Path(config.audit_dir)
        self.min_valid_date = pd.Timestamp(
            config.min_valid_date or ("1990-01-01" if self.market == "CN" else "1980-01-01")
        )
        self.latest_required_ts = (
            pd.Timestamp(config.latest_required_date)
            if config.latest_required_date
            else None
        )
        self.file_results: list[FileCleanResult] = []
        self.quarantine_rows: list[dict[str, Any]] = []
        self.membership_rows: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        if self.config.write_clean:
            self.clean_dir.mkdir(parents=True, exist_ok=True)

        groups = self._discover_symbol_groups()
        for symbol, entries in sorted(groups.items()):
            canonical = self._select_canonical_entry(entries)
            self._record_membership(symbol, entries, canonical)
            self.file_results.append(self._clean_file(canonical))

        return self._write_audit(groups)

    def _discover_symbol_groups(self) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in sorted(self.raw_dir.glob("*/*.csv")):
            if path.parent.name.startswith(".") or path.parent.name.startswith("_"):
                continue
            category = path.parent.name
            symbol = path.stem.upper()
            groups[symbol].append(
                {
                    "symbol": symbol,
                    "category": category,
                    "path": path,
                }
            )
        return groups

    def _select_canonical_entry(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        priority = CN_CATEGORY_PRIORITY if self.market == "CN" else US_CATEGORY_PRIORITY
        order = {category: index for index, category in enumerate(priority)}
        return sorted(
            entries,
            key=lambda item: (order.get(str(item["category"]), len(order)), str(item["path"])),
        )[0]

    def _record_membership(
        self,
        symbol: str,
        entries: list[dict[str, Any]],
        canonical: dict[str, Any],
    ) -> None:
        categories = sorted({str(entry["category"]) for entry in entries})
        self.membership_rows.append(
            {
                "market": self.market,
                "symbol": symbol,
                "categories": "|".join(categories),
                "source_count": len(entries),
                "has_duplicate_storage": len(entries) > 1,
                "canonical_category": canonical["category"],
                "canonical_path": _posix(canonical["path"]),
                "all_source_paths": "|".join(_posix(entry["path"]) for entry in entries),
            }
        )

    def _clean_file(self, entry: dict[str, Any]) -> FileCleanResult:
        path = Path(entry["path"])
        symbol = str(entry["symbol"]).upper()
        category = str(entry["category"])
        output_path = self._output_path(symbol, category)
        result = FileCleanResult(
            market=self.market,
            symbol=symbol,
            category=category,
            source_path=_posix(path),
            output_path=_posix(output_path) if self.config.write_clean else None,
        )

        try:
            raw = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - exercised by real corrupt files.
            result.issue_counts["csv_unreadable"] = 1
            result.quarantine_reasons.append("csv_unreadable")
            result.issue_counts["exception"] = 1
            self._append_file_exception(result, exc)
            return result

        result.raw_rows = int(len(raw))
        if raw.empty:
            result.issue_counts["empty_file"] = 1
            result.quarantine_reasons.append("empty_file")
            return result

        normalized = self._normalize_frame(raw, symbol)
        result.latest_raw_date = _max_date(normalized["trade_date"])

        masks = self._build_issue_masks(normalized)
        for issue_name, mask in masks.items():
            count = int(mask.sum())
            if count:
                result.issue_counts[issue_name] = count

        blocking_mask = pd.Series(False, index=normalized.index)
        for issue_name in BLOCKING_ROW_ISSUES:
            blocking_mask = blocking_mask | masks.get(issue_name, False)

        if bool(blocking_mask.any()):
            self._append_quarantine_rows(
                symbol=symbol,
                category=category,
                path=path,
                frame=normalized.loc[blocking_mask],
                masks=masks,
            )

        cleaned = normalized.loc[~blocking_mask].copy()
        result.dropped_rows = int(blocking_mask.sum())

        if not cleaned.empty:
            duplicate_mask = cleaned["trade_date"].duplicated(keep="last")
            result.duplicate_date_rows = int(duplicate_mask.sum())
            if result.duplicate_date_rows:
                result.issue_counts["duplicate_date"] = result.duplicate_date_rows
                cleaned = cleaned.loc[~duplicate_mask].copy()
            cleaned = cleaned.sort_values("trade_date").reset_index(drop=True)

        if self.market == "CN":
            adjustment_missing = self._cn_adjustment_missing(cleaned)
            if adjustment_missing:
                result.issue_counts["missing_adjustment"] = adjustment_missing

        result.clean_rows = int(len(cleaned))
        result.latest_clean_date = _max_date(cleaned["trade_date"]) if not cleaned.empty else None
        result.quarantine_reasons = self._build_quarantine_reasons(result)

        if self.config.write_clean and not cleaned.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned[self._output_columns(cleaned)].to_csv(output_path, index=False)

        return result

    def _output_path(self, symbol: str, category: str) -> Path:
        if self.market == "CN":
            return self.clean_dir / category / f"{symbol}.csv"
        return self.clean_dir / f"{symbol}.csv"

    def _normalize_frame(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        frame = raw.copy()
        rename_map: dict[str, str] = {}
        if self.market == "CN":
            rename_map.update({"trade_date": "trade_date", "vol": "volume"})
            symbol_col = "ts_code"
        else:
            rename_map.update(
                {
                    "Date": "trade_date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Dividends": "dividends",
                    "Stock Splits": "stock_splits",
                    "Capital Gains": "capital_gains",
                    "Amount": "amount",
                }
            )
            symbol_col = "symbol"

        frame = frame.rename(columns=rename_map)
        frame = _collapse_duplicate_columns(frame)
        if "trade_date" not in frame.columns:
            frame["trade_date"] = pd.NA
        frame["trade_date"] = _parse_trade_dates(frame["trade_date"])
        frame["symbol"] = _normalize_symbol_column(
            frame.get(symbol_col),
            fallback=symbol,
            index=frame.index,
        )

        for column in self._numeric_columns(frame):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            if column not in frame.columns:
                frame[column] = pd.NA

        frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
        return frame

    def _numeric_columns(self, frame: pd.DataFrame) -> list[str]:
        candidates = [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "volume",
            "amount",
            "adj_factor",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "dividends",
            "stock_splits",
            "capital_gains",
        ]
        return [column for column in candidates if column in frame.columns]

    def _build_issue_masks(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
        price = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")

        invalid_date = dates.isna() | (dates < self.min_valid_date)
        missing_price = price.isna().any(axis=1)
        missing_volume = volume.isna()
        nonpositive_price = (price <= 0).any(axis=1).fillna(False)
        negative_volume = (volume < 0).fillna(False)
        ohlc_inconsistent = (
            (price["high"] < price[["open", "low", "close"]].max(axis=1))
            | (price["low"] > price[["open", "high", "close"]].min(axis=1))
        ).fillna(False)

        return {
            "invalid_date": invalid_date,
            "missing_price": missing_price,
            "missing_volume": missing_volume,
            "nonpositive_price": nonpositive_price,
            "negative_volume": negative_volume,
            "ohlc_inconsistent": ohlc_inconsistent & ~missing_price,
        }

    def _cn_adjustment_missing(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        missing = pd.Series(False, index=frame.index)
        for column in ("adj_factor", "adj_close"):
            if column not in frame.columns:
                return int(len(frame))
            values = pd.to_numeric(frame[column], errors="coerce")
            missing = missing | values.isna()
            if column == "adj_factor":
                missing = missing | (values <= 0)
        return int(missing.sum())

    def _build_quarantine_reasons(self, result: FileCleanResult) -> list[str]:
        reasons: list[str] = []
        if result.clean_rows == 0:
            reasons.append("no_clean_rows")
        if result.dropped_rows:
            reasons.append("raw_rows_dropped")
        if result.duplicate_date_rows:
            reasons.append("duplicate_dates_deduped")
        if result.issue_counts.get("missing_adjustment", 0):
            reasons.append("missing_adjustment")
        if self.latest_required_ts is not None:
            latest_clean = pd.to_datetime(result.latest_clean_date, errors="coerce")
            if pd.isna(latest_clean) or latest_clean < self.latest_required_ts:
                reasons.append("stale_vs_required_date")
        return sorted(set(reasons))

    def _append_quarantine_rows(
        self,
        *,
        symbol: str,
        category: str,
        path: Path,
        frame: pd.DataFrame,
        masks: dict[str, pd.Series],
    ) -> None:
        if frame.empty:
            return
        remaining = max(self.config.max_quarantine_rows_per_symbol, 0)
        for index, row in frame.iterrows():
            if remaining <= 0:
                break
            issues = [
                issue_name
                for issue_name in BLOCKING_ROW_ISSUES
                if issue_name in masks and bool(masks[issue_name].loc[index])
            ]
            self.quarantine_rows.append(
                {
                    "market": self.market,
                    "symbol": symbol,
                    "category": category,
                    "trade_date": row.get("trade_date"),
                    "issues": "|".join(issues),
                    "source_path": _posix(path),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                }
            )
            remaining -= 1

    def _append_file_exception(self, result: FileCleanResult, exc: Exception) -> None:
        self.quarantine_rows.append(
            {
                "market": self.market,
                "symbol": result.symbol,
                "category": result.category,
                "trade_date": None,
                "issues": "csv_unreadable",
                "source_path": result.source_path,
                "error": str(exc),
            }
        )

    def _output_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.market == "CN":
            preferred = [
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pre_close",
                "change",
                "pct_chg",
                "adj_factor",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
            ]
        else:
            preferred = [
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "dividends",
                "stock_splits",
                "capital_gains",
            ]
        return [column for column in preferred if column in frame.columns]

    def _write_audit(self, groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        file_audit_path = self.audit_dir / "file_audit.csv"
        quarantine_symbols_path = self.audit_dir / "quarantine_symbols.csv"
        quarantine_rows_path = self.audit_dir / "quarantine_rows.csv"
        membership_path = self.audit_dir / "membership.csv"
        manifest_path = self.audit_dir / "clean_manifest.json"
        summary_path = self.audit_dir / "summary.md"

        file_rows = [
            {
                **asdict(result),
                "issue_counts": json.dumps(result.issue_counts, ensure_ascii=False, sort_keys=True),
                "quarantine_reasons": "|".join(result.quarantine_reasons),
            }
            for result in self.file_results
        ]
        pd.DataFrame(file_rows).to_csv(file_audit_path, index=False)
        pd.DataFrame(self.membership_rows).to_csv(membership_path, index=False)
        pd.DataFrame(self.quarantine_rows).to_csv(quarantine_rows_path, index=False)

        quarantine_rows = [
            row
            for row in file_rows
            if str(row.get("quarantine_reasons") or "").strip()
        ]
        pd.DataFrame(quarantine_rows).to_csv(quarantine_symbols_path, index=False)

        latest_distribution = Counter(
            result.latest_clean_date or "missing"
            for result in self.file_results
        )
        issue_totals: Counter[str] = Counter()
        for result in self.file_results:
            issue_totals.update(result.issue_counts)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "market": self.market,
            "raw_dir": _posix(self.raw_dir),
            "clean_dir": _posix(self.clean_dir),
            "audit_dir": _posix(self.audit_dir),
            "write_clean": self.config.write_clean,
            "min_valid_date": self.min_valid_date.date().isoformat(),
            "latest_required_date": (
                self.latest_required_ts.date().isoformat()
                if self.latest_required_ts is not None
                else None
            ),
            "totals": {
                "source_file_count": sum(len(entries) for entries in groups.values()),
                "processed_file_count": len(self.file_results),
                "unique_symbol_count": len(groups),
                "duplicate_symbol_count": sum(1 for entries in groups.values() if len(entries) > 1),
                "output_file_count": sum(1 for result in self.file_results if result.clean_rows > 0),
                "raw_rows": sum(result.raw_rows for result in self.file_results),
                "clean_rows": sum(result.clean_rows for result in self.file_results),
                "dropped_rows": sum(result.dropped_rows for result in self.file_results),
                "quarantined_symbol_count": len(quarantine_rows),
                "quarantined_row_count": len(self.quarantine_rows),
            },
            "issue_totals": dict(sorted(issue_totals.items())),
            "latest_clean_date_distribution": dict(latest_distribution.most_common()),
            "artifacts": {
                "file_audit": _posix(file_audit_path),
                "quarantine_symbols": _posix(quarantine_symbols_path),
                "quarantine_rows": _posix(quarantine_rows_path),
                "membership": _posix(membership_path),
                "summary": _posix(summary_path),
            },
        }

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_path.write_text(_render_summary(manifest), encoding="utf-8")
        return manifest


def _parse_trade_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    compact_mask = text.str.match(r"^\d{8}$", na=False)
    if bool(compact_mask.any()):
        parsed.loc[compact_mask] = pd.to_datetime(
            text.loc[compact_mask],
            format="%Y%m%d",
            errors="coerce",
        )
    return parsed


def _collapse_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.is_unique:
        return frame
    collapsed = pd.DataFrame(index=frame.index)
    for column in dict.fromkeys(str(value) for value in frame.columns):
        values = frame.loc[:, frame.columns.astype(str) == column]
        if isinstance(values, pd.Series):
            collapsed[column] = values
        elif values.shape[1] == 1:
            collapsed[column] = values.iloc[:, 0]
        else:
            collapsed[column] = values.bfill(axis=1).iloc[:, 0]
    return collapsed


def _normalize_symbol_column(
    series: pd.Series | None,
    *,
    fallback: str,
    index: pd.Index,
) -> pd.Series:
    if series is None:
        return pd.Series(fallback, index=index)
    normalized = series.astype("string").str.strip().str.upper()
    normalized = normalized.mask(normalized.isna() | (normalized == ""), fallback)
    return normalized


def _max_date(series: pd.Series) -> str | None:
    dates = pd.to_datetime(series, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date().isoformat()


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


def _render_summary(manifest: dict[str, Any]) -> str:
    totals = manifest["totals"]
    issue_totals = manifest["issue_totals"]
    latest = manifest["latest_clean_date_distribution"]
    lines = [
        f"# {manifest['market']} daily clean audit",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- raw_dir: `{manifest['raw_dir']}`",
        f"- clean_dir: `{manifest['clean_dir']}`",
        f"- latest_required_date: `{manifest['latest_required_date']}`",
        f"- processed_file_count: {totals['processed_file_count']}",
        f"- unique_symbol_count: {totals['unique_symbol_count']}",
        f"- duplicate_symbol_count: {totals['duplicate_symbol_count']}",
        f"- raw_rows: {totals['raw_rows']}",
        f"- clean_rows: {totals['clean_rows']}",
        f"- dropped_rows: {totals['dropped_rows']}",
        f"- quarantined_symbol_count: {totals['quarantined_symbol_count']}",
        f"- quarantined_row_count: {totals['quarantined_row_count']}",
        "",
        "## Issue totals",
    ]
    if issue_totals:
        lines.extend(f"- {key}: {value}" for key, value in sorted(issue_totals.items()))
    else:
        lines.append("- none")
    lines.extend(["", "## Latest clean date distribution"])
    lines.extend(f"- {key}: {value}" for key, value in latest.items())
    lines.append("")
    return "\n".join(lines)


def latest_download_report_target(raw_dir: Path, market: str) -> str | None:
    """Best-effort target date from the newest local download report."""

    reports = sorted(Path(raw_dir).glob("download_report_*.json"))
    if not reports:
        return None
    try:
        payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if market.upper() == "CN":
        value = config.get("strict_trade_date") or config.get("effective_target_trade_date")
        return _compact_date_to_iso(value)
    latest_dates: list[str] = []
    categories = payload.get("categories", {}) if isinstance(payload, dict) else {}
    if isinstance(categories, dict):
        for rows in categories.values():
            if isinstance(rows, Iterable):
                for row in rows:
                    if isinstance(row, dict) and row.get("latest_date"):
                        latest_dates.append(str(row["latest_date"]))
    if latest_dates:
        return max(latest_dates)
    return None


def _compact_date_to_iso(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()
