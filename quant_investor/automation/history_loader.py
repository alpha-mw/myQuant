from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from quant_investor.strategy_records.store import (
    StrategyRecordStoreError,
    catalog_history_entries,
    load_registered_catalog,
)

RecordStoreError = StrategyRecordStoreError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{3,6}$")
log = logging.getLogger("daily_runner")

REGISTERED_HISTORY_STRATEGIES = frozenset({("CN", "aggressive_tech_manufacturing")})
LEGACY_HISTORY_ALLOWLIST = frozenset({("US", "simulated_portfolio_10000")})
DEFAULT_HISTORY_STRATEGY_BY_MARKET = {
    "CN": "aggressive_tech_manufacturing",
    "US": "simulated_portfolio_10000",
}


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
        raise ValueError("history_strategy must be one explicit strategy id")
    return normalized_market, normalized_strategy


class HistoryLoader:
    """Read recent strategy history through the registered record catalog.

    The sole direct-directory compatibility path is the explicit legacy
    allowlist above. Registered CN aggressive history never falls back to a
    recursive filesystem scan.
    """

    def __init__(self, strategy_records_root: Path | None = None) -> None:
        self.strategy_records_root = (
            Path(strategy_records_root)
            if strategy_records_root is not None
            else PROJECT_ROOT / "results" / "strategy_records"
        )

    def _legacy_run_dirs(self, market: str, strategy: str) -> list[Path]:
        identity = (market, strategy)
        if identity not in LEGACY_HISTORY_ALLOWLIST:
            raise RecordStoreError(
                f"strategy history is not registered or legacy-allowlisted: {market}/{strategy}"
            )
        strategy_root = self.strategy_records_root / market / strategy
        if not strategy_root.exists():
            return []
        return sorted(
            (
                path
                for path in strategy_root.iterdir()
                if path.is_dir() and not path.is_symlink() and RUN_DIR_PATTERN.fullmatch(path.name)
            ),
            key=lambda item: item.name,
            reverse=True,
        )

    def _catalog_entries(self, market: str, strategy: str) -> list[dict[str, Any]]:
        strategy_root = self.strategy_records_root / market / strategy
        registered = load_registered_catalog(strategy_root)
        if registered is None:
            raise RecordStoreError(f"registered strategy catalog missing: {market}/{strategy}")
        entries = catalog_history_entries(strategy_root)
        if any(not isinstance(item, dict) for item in entries):
            raise RecordStoreError("catalog history entry is invalid")
        return list(entries)

    def _parse_markdown_excerpt(self, path: Path) -> dict[str, Any]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            return {"file": path.name, "error": str(exc)}

        excerpt_lines: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("#", "-", "*")) or ":" in line:
                excerpt_lines.append(line[:180])
            if len(excerpt_lines) >= 6:
                break
        if not excerpt_lines:
            excerpt_lines = [line.strip()[:180] for line in lines if line.strip()][:3]
        return {"file": path.name, "excerpt_lines": excerpt_lines}

    @staticmethod
    def _parse_csv_summary(path: Path) -> dict[str, Any]:
        return {
            "file": path.name,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "note": "csv_export_not_read",
        }

    def _extract_markdown_summary(self, excerpt: dict[str, Any]) -> dict[str, Any]:
        symbols: list[str] = []
        actions: list[str] = []
        for raw_line in excerpt.get("excerpt_lines", []):
            line = str(raw_line or "").strip().lstrip("-*# ").strip()
            match = re.match(
                r"^(symbol|action)\s*[:：]\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            value = match.group(2).strip()
            if not value:
                continue
            if match.group(1).lower() == "symbol":
                symbols.append(value)
            else:
                actions.append(value.lower())
        return {
            "file": str(excerpt.get("file", "") or ""),
            "symbols": _dedupe_text(symbols, limit=10),
            "actions": _dedupe_text(actions, limit=10),
        }

    def _collect_legacy_entry(
        self,
        market: str,
        strategy: str,
        run_dir: Path,
    ) -> Optional[dict[str, Any]]:
        try:
            date_part, time_part = run_dir.name.split("_", 1)
        except ValueError:
            return None
        markdown_files: list[str] = []
        csv_files: list[str] = []
        markdown_excerpts: list[dict[str, Any]] = []
        csv_summaries: list[dict[str, Any]] = []
        latest_report_excerpt = ""

        for child in sorted(run_dir.iterdir(), key=lambda item: item.name):
            if not child.is_file() or child.is_symlink():
                continue
            if child.suffix.lower() not in {".md", ".csv"}:
                continue
            if child.suffix.lower() == ".md":
                markdown_files.append(str(child))
                excerpt = self._parse_markdown_excerpt(child)
                markdown_excerpts.append(excerpt)
                if child.name == "analysis_report.md" and not latest_report_excerpt:
                    latest_report_excerpt = "\n".join(excerpt.get("excerpt_lines", [])[:4])
            else:
                csv_files.append(str(child))
                csv_summaries.append(self._parse_csv_summary(child))

        summaries = [
            *csv_summaries,
            *(self._extract_markdown_summary(item) for item in markdown_excerpts),
        ]
        return {
            "date": date_part,
            "timestamp": f"{date_part}_{time_part.zfill(6)}",
            "strategy": strategy,
            "market": market,
            "record_id": run_dir.name,
            "record_dir": str(run_dir),
            "storage_state": "LEGACY_ONLINE",
            "evidence_status": "LEGACY_ALLOWLIST_UNHASHED",
            "markdown_files": markdown_files,
            "csv_files": csv_files,
            "markdown_excerpts": markdown_excerpts,
            "csv_summaries": csv_summaries,
            "latest_report_excerpt": latest_report_excerpt,
            "symbols": _dedupe_text(
                [symbol for summary in summaries for symbol in summary.get("symbols", [])],
                limit=10,
            ),
            "actions": _dedupe_text(
                [action for summary in summaries for action in summary.get("actions", [])],
                limit=10,
            ),
        }

    def _normalize_catalog_entry(
        self,
        raw: dict[str, Any],
        market: str,
        strategy: str,
    ) -> dict[str, Any]:
        record_id = str(raw.get("record_id") or "").strip()
        if not record_id:
            raise RecordStoreError("catalog history entry record_id missing")
        timestamp = str(raw.get("timestamp") or record_id).strip()
        date_part = str(raw.get("date") or timestamp[:8]).strip()
        summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
        markdown_excerpts = list(
            raw.get("markdown_excerpts") or summary.get("markdown_excerpts") or []
        )
        csv_summaries = list(raw.get("csv_summaries") or summary.get("csv_summaries") or [])
        markdown_files = list(raw.get("markdown_files") or raw.get("markdown_refs") or [])
        csv_files = list(raw.get("csv_files") or raw.get("csv_refs") or [])
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
            "markdown_files": markdown_files,
            "csv_files": csv_files,
            "markdown_excerpts": markdown_excerpts,
            "csv_summaries": csv_summaries,
            "latest_report_excerpt": str(
                raw.get("latest_report_excerpt") or summary.get("latest_report_excerpt") or ""
            ),
            "symbols": _dedupe_text(
                list(raw.get("symbols") or summary.get("symbols") or []),
                limit=10,
            ),
            "actions": _dedupe_text(
                list(raw.get("actions") or summary.get("actions") or []),
                limit=10,
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
        identity = (market, strategy)
        if identity in REGISTERED_HISTORY_STRATEGIES:
            entries = [
                self._normalize_catalog_entry(item, market, strategy)
                for item in self._catalog_entries(market, strategy)
            ]
        else:
            entries = [
                entry
                for run_dir in self._legacy_run_dirs(market, strategy)
                if (entry := self._collect_legacy_entry(market, strategy, run_dir)) is not None
            ]
        if not entries:
            return []

        ordered_dates = sorted({str(item["date"]) for item in entries}, reverse=True)[:max_dates]
        selected = [item for item in entries if str(item["date"]) in ordered_dates]
        return sorted(
            selected,
            key=lambda item: str(item.get("timestamp", "")),
            reverse=True,
        )

    def build_recall_context(
        self,
        runs: list[dict[str, Any]],
        market: str = "CN",
    ) -> dict[str, Any]:
        window_dates = sorted(
            {str(item.get("date", "")) for item in runs if item.get("date")},
            reverse=True,
        )
        recent_symbols = _dedupe_text(
            [symbol for item in runs for symbol in item.get("symbols", [])],
            limit=20,
        )
        recent_actions: list[dict[str, str]] = []
        for item in runs:
            for summary in [
                *item.get("csv_summaries", []),
                *(
                    self._extract_markdown_summary(excerpt)
                    for excerpt in item.get("markdown_excerpts", [])
                ),
            ]:
                actions = summary.get("actions", [])
                symbols = summary.get("symbols", [])
                if not actions and not symbols:
                    continue
                recent_actions.append(
                    {
                        "date": str(item.get("date", "") or ""),
                        "strategy": str(item.get("strategy", "") or ""),
                        "file": str(summary.get("file", "") or ""),
                        "symbol": symbols[0] if symbols else "",
                        "action": actions[0] if actions else "",
                    }
                )
                if len(recent_actions) >= 20:
                    break
            if len(recent_actions) >= 20:
                break

        return {
            "source": "strategy_record_catalog",
            "market": str(market or "CN").upper(),
            "window_dates": window_dates[:5],
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
            "recent_symbols": recent_symbols,
            "recent_actions": recent_actions,
            "latest_report_excerpt": (
                str(runs[0].get("latest_report_excerpt", "") or "") if runs else ""
            ),
        }

    def load_last_report(
        self,
        market: str = "CN",
        strategy: str | None = None,
    ) -> Optional[str]:
        for item in self.load_recent(
            market=market,
            strategy=strategy,
            max_dates=5,
        ):
            record_dir_text = str(item.get("record_dir", "")).strip()
            if not record_dir_text:
                continue
            report_path = Path(record_dir_text) / "analysis_report.md"
            if report_path.is_symlink() or not report_path.is_file():
                continue
            try:
                return report_path.read_text(encoding="utf-8")
            except Exception as exc:
                log.warning("读取最新策略记录报告失败: %s", exc)
                return None
        return None

    def format_context_section(self, runs: list[dict[str, Any]]) -> str:
        if not runs:
            return "_暂无最近 5 个日期的策略记录。_"

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in runs:
            grouped[str(item.get("date", ""))].append(item)

        lines: list[str] = []
        for date_key in sorted(grouped.keys(), reverse=True):
            lines.append(f"### {date_key}")
            for item in sorted(
                grouped[date_key],
                key=lambda row: str(row.get("timestamp", "")),
                reverse=True,
            ):
                symbols = "、".join(item.get("symbols", [])[:4]) or "无"
                actions = "、".join(item.get("actions", [])[:4]) or "无"
                excerpt_lines: list[str] = []
                for excerpt in item.get("markdown_excerpts", []):
                    excerpt_lines.extend(excerpt.get("excerpt_lines", [])[:2])
                    if len(excerpt_lines) >= 2:
                        break
                excerpt_text = " / ".join(excerpt_lines[:2]) if excerpt_lines else "暂无摘要"
                lines.append(
                    f"- `{item.get('strategy', '')}` | `{item.get('record_id', '')}` | "
                    f"storage=`{item.get('storage_state', 'UNKNOWN')}` | "
                    f"evidence=`{item.get('evidence_status', 'UNKNOWN')}` | "
                    f"{len(item.get('markdown_files', []))} md / "
                    f"{len(item.get('csv_files', []))} csv | "
                    f"symbols: {symbols} | actions: {actions}"
                )
                lines.append(f"  摘要: {excerpt_text}")
        return "\n".join(lines)
