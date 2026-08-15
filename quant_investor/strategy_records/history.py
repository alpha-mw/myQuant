"""Read-only Strategy Record Store history projections.

History is resolved exclusively through the registered catalog.  This module
never scans a strategy directory and has no compatibility path for an
unregistered record tree.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any

from .store import (
    StrategyRecordStoreError,
    catalog_history_entries,
    load_registered_catalog,
)


RecordStoreError = StrategyRecordStoreError
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_STRATEGY_BY_MARKET = {
    "CN": "aggressive_tech_manufacturing",
}
log = logging.getLogger("strategy_record_history")


def _dedupe_text(values: list[str], limit: int = 8) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
        if len(ordered) >= limit:
            break
    return ordered


def _normalized_identity(market: str, strategy: str | None) -> tuple[str, str]:
    normalized_market = str(market or "CN").strip().upper()
    normalized_strategy = str(
        strategy or DEFAULT_HISTORY_STRATEGY_BY_MARKET.get(normalized_market, "")
    ).strip()
    if not normalized_strategy or Path(normalized_strategy).name != normalized_strategy:
        raise ValueError("history_strategy must be one explicit registered strategy id")
    return normalized_market, normalized_strategy


class HistoryLoader:
    """Project recent history from one registered strategy catalog."""

    def __init__(self, strategy_records_root: Path | None = None) -> None:
        self.strategy_records_root = (
            Path(strategy_records_root)
            if strategy_records_root is not None
            else PROJECT_ROOT / "results" / "strategy_records"
        )

    def _catalog_entries(self, market: str, strategy: str) -> list[dict[str, Any]]:
        strategy_root = self.strategy_records_root / market / strategy
        if load_registered_catalog(strategy_root) is None:
            raise RecordStoreError(f"registered strategy catalog missing: {market}/{strategy}")
        entries = catalog_history_entries(strategy_root)
        if any(type(item) is not dict for item in entries):
            raise RecordStoreError("catalog history entry is invalid")
        return list(entries)

    @staticmethod
    def _normalize_catalog_entry(
        raw: dict[str, Any], market: str, strategy: str
    ) -> dict[str, Any]:
        record_id = str(raw.get("record_id") or "").strip()
        if not record_id:
            raise RecordStoreError("catalog history entry record_id missing")
        timestamp = str(raw.get("timestamp") or record_id).strip()
        date_part = str(raw.get("date") or timestamp[:8]).strip()
        summary = raw.get("summary") if type(raw.get("summary")) is dict else {}
        return {
            **raw,
            "date": date_part,
            "timestamp": timestamp,
            "strategy": strategy,
            "market": market,
            "record_id": record_id,
            "record_dir": str(raw.get("record_dir") or ""),
            "storage_state": str(raw.get("storage_state") or "UNKNOWN"),
            "evidence_status": str(raw.get("evidence_status") or "UNKNOWN"),
            "markdown_files": list(raw.get("markdown_files") or raw.get("markdown_refs") or []),
            "csv_files": list(raw.get("csv_files") or raw.get("csv_refs") or []),
            "markdown_excerpts": list(
                raw.get("markdown_excerpts") or summary.get("markdown_excerpts") or []
            ),
            "csv_summaries": list(
                raw.get("csv_summaries") or summary.get("csv_summaries") or []
            ),
            "latest_report_excerpt": str(
                raw.get("latest_report_excerpt") or summary.get("latest_report_excerpt") or ""
            ),
            "symbols": _dedupe_text(
                list(raw.get("symbols") or summary.get("symbols") or []), limit=10
            ),
            "actions": _dedupe_text(
                list(raw.get("actions") or summary.get("actions") or []), limit=10
            ),
        }

    def load_recent(
        self,
        market: str = "CN",
        max_dates: int = 5,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        market, strategy = _normalized_identity(market, strategy)
        if max_dates <= 0:
            return []
        entries = [
            self._normalize_catalog_entry(item, market, strategy)
            for item in self._catalog_entries(market, strategy)
        ]
        dates = sorted({str(item["date"]) for item in entries}, reverse=True)[:max_dates]
        return sorted(
            (item for item in entries if str(item["date"]) in dates),
            key=lambda item: str(item.get("timestamp", "")),
            reverse=True,
        )

    def build_recall_context(
        self, runs: list[dict[str, Any]], market: str = "CN"
    ) -> dict[str, Any]:
        dates = sorted(
            {str(item.get("date", "")) for item in runs if item.get("date")},
            reverse=True,
        )
        return {
            "source": "strategy_record_catalog",
            "market": str(market or "CN").upper(),
            "window_dates": dates[:5],
            "records": [
                {
                    "date": item.get("date", ""),
                    "strategy": item.get("strategy", ""),
                    "record_id": item.get("record_id", ""),
                    "record_dir": item.get("record_dir", ""),
                    "storage_state": item.get("storage_state", "UNKNOWN"),
                    "evidence_status": item.get("evidence_status", "UNKNOWN"),
                    "markdown_excerpts": item.get("markdown_excerpts", [])[:4],
                    "csv_summaries": item.get("csv_summaries", [])[:4],
                }
                for item in runs[:12]
            ],
            "recent_symbols": _dedupe_text(
                [symbol for item in runs for symbol in item.get("symbols", [])],
                limit=20,
            ),
            "recent_actions": [],
            "latest_report_excerpt": (
                str(runs[0].get("latest_report_excerpt", "") or "") if runs else ""
            ),
        }

    def load_last_report(
        self, market: str = "CN", strategy: str | None = None
    ) -> str | None:
        for item in self.load_recent(market=market, strategy=strategy, max_dates=5):
            record_dir = str(item.get("record_dir", "")).strip()
            if not record_dir:
                continue
            report_path = Path(record_dir) / "analysis_report.md"
            if report_path.is_symlink() or not report_path.is_file():
                continue
            try:
                return report_path.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning("读取已登记策略记录报告失败: %s", exc)
                return None
        return None

    @staticmethod
    def format_context_section(runs: list[dict[str, Any]]) -> str:
        if not runs:
            return "_暂无最近 5 个日期的已登记策略记录。_"
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in runs:
            grouped[str(item.get("date", ""))].append(item)
        lines: list[str] = []
        for date_key in sorted(grouped, reverse=True):
            lines.append(f"### {date_key}")
            for item in sorted(
                grouped[date_key],
                key=lambda row: str(row.get("timestamp", "")),
                reverse=True,
            ):
                lines.append(
                    f"- `{item.get('strategy', '')}` | `{item.get('record_id', '')}` | "
                    f"storage=`{item.get('storage_state', 'UNKNOWN')}` | "
                    f"evidence=`{item.get('evidence_status', 'UNKNOWN')}`"
                )
        return "\n".join(lines)


__all__ = ["HistoryLoader", "RecordStoreError"]
