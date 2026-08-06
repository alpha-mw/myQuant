from __future__ import annotations

import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from quant_investor.automation import daily_runner as _runner

RUN_DIR_PATTERN: re.Pattern[str] = _runner.RUN_DIR_PATTERN  # type: ignore[has-type]
_dedupe_text = _runner._dedupe_text
_strategy_records_market_root = _runner._strategy_records_market_root
log: logging.Logger = _runner.log  # type: ignore[has-type]


class HistoryLoader:
    """从 strategy_records 读取最近 5 个日期的策略记录。"""

    def _iter_run_dirs(self, market: str) -> list[Path]:
        market_root = _strategy_records_market_root(market)
        if not market_root.exists():
            return []

        run_dirs: list[Path] = []
        for path in market_root.rglob("*"):
            if path.is_dir() and RUN_DIR_PATTERN.match(path.name):
                run_dirs.append(path)
        return sorted(run_dirs, key=lambda item: item.as_posix(), reverse=True)

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

    def _parse_csv_summary(self, path: Path) -> dict[str, Any]:
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
            match = re.match(r"^(symbol|action)\s*[:：]\s*(.+)$", line, flags=re.IGNORECASE)
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

    def _collect_run_entry(self, market: str, run_dir: Path) -> Optional[dict[str, Any]]:
        market_root = _strategy_records_market_root(market)
        try:
            relative = run_dir.relative_to(market_root)
        except ValueError:
            return None
        if len(relative.parts) < 2:
            return None

        date_part, time_part = run_dir.name.split("_", 1)
        strategy = "/".join(relative.parts[:-1])
        markdown_files: list[str] = []
        csv_files: list[str] = []
        markdown_excerpts: list[dict[str, Any]] = []
        csv_summaries: list[dict[str, Any]] = []
        latest_report_excerpt = ""

        for child in sorted(run_dir.iterdir(), key=lambda item: item.name):
            if not child.is_file():
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

        return {
            "date": date_part,
            "timestamp": f"{date_part}_{time_part.zfill(6)}",
            "strategy": strategy,
            "record_dir": str(run_dir),
            "markdown_files": markdown_files,
            "csv_files": csv_files,
            "markdown_excerpts": markdown_excerpts,
            "csv_summaries": csv_summaries,
            "latest_report_excerpt": latest_report_excerpt,
            "symbols": _dedupe_text(
                [
                    symbol
                    for summary in [
                        *csv_summaries,
                        *(self._extract_markdown_summary(excerpt) for excerpt in markdown_excerpts),
                    ]
                    for symbol in summary.get("symbols", [])
                ],
                limit=10,
            ),
            "actions": _dedupe_text(
                [
                    action
                    for summary in [
                        *csv_summaries,
                        *(self._extract_markdown_summary(excerpt) for excerpt in markdown_excerpts),
                    ]
                    for action in summary.get("actions", [])
                ],
                limit=10,
            ),
        }

    def load_recent(self, market: str = "CN", max_dates: int = 5) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_dir in self._iter_run_dirs(market):
            entry = self._collect_run_entry(market, run_dir)
            if entry is not None:
                entries.append(entry)
        if not entries:
            return []

        ordered_dates = sorted({item["date"] for item in entries}, reverse=True)[:max_dates]
        selected = [item for item in entries if item["date"] in ordered_dates]
        return sorted(selected, key=lambda item: str(item.get("timestamp", "")), reverse=True)

    def build_recall_context(self, runs: list[dict[str, Any]], market: str = "CN") -> dict[str, Any]:
        window_dates = sorted({str(item.get("date", "")) for item in runs if item.get("date")}, reverse=True)
        recent_symbols = _dedupe_text(
            [symbol for item in runs for symbol in item.get("symbols", [])],
            limit=20,
        )
        recent_actions: list[dict[str, str]] = []
        for item in runs:
            for summary in [
                *item.get("csv_summaries", []),
                *(self._extract_markdown_summary(excerpt) for excerpt in item.get("markdown_excerpts", [])),
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
            "source": "strategy_records",
            "market": str(market or "CN").upper(),
            "window_dates": window_dates[:5],
            "records": [
                {
                    "date": item.get("date", ""),
                    "strategy": item.get("strategy", ""),
                    "record_dir": item.get("record_dir", ""),
                    "markdown_excerpts": item.get("markdown_excerpts", [])[:4],
                    "csv_summaries": item.get("csv_summaries", [])[:4],
                }
                for item in runs[:12]
            ],
            "recent_symbols": recent_symbols,
            "recent_actions": recent_actions,
            "latest_report_excerpt": str(runs[0].get("latest_report_excerpt", "") or "") if runs else "",
        }

    def load_last_report(self, market: str = "CN") -> Optional[str]:
        for item in self.load_recent(market=market, max_dates=5):
            report_path = Path(str(item.get("record_dir", ""))) / "analysis_report.md"
            if not report_path.exists():
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
            for item in sorted(grouped[date_key], key=lambda row: str(row.get("timestamp", "")), reverse=True):
                symbols = "、".join(item.get("symbols", [])[:4]) or "无"
                actions = "、".join(item.get("actions", [])[:4]) or "无"
                excerpt_lines: list[str] = []
                for excerpt in item.get("markdown_excerpts", []):
                    excerpt_lines.extend(excerpt.get("excerpt_lines", [])[:2])
                    if len(excerpt_lines) >= 2:
                        break
                excerpt_text = " / ".join(excerpt_lines[:2]) if excerpt_lines else "暂无摘要"
                lines.append(
                    f"- `{item.get('strategy', '')}` | `{Path(str(item.get('record_dir', ''))).name}` | "
                    f"{len(item.get('markdown_files', []))} md / {len(item.get('csv_files', []))} csv | "
                    f"symbols: {symbols} | actions: {actions}"
                )
                lines.append(f"  摘要: {excerpt_text}")
        return "\n".join(lines)
