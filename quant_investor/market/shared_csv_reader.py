#!/usr/bin/env python3
"""Shared resolver-backed CSV reader for market DAG execution."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_investor.agent_protocol import DataQualityIssue
from quant_investor.data.storage.csv_reader import CSVReadResult, infer_latest_date_from_frames, peek_latest_date, read_csv_with_diagnostics
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.config import get_market_settings


@dataclass
class SharedCSVReadResult:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    path: str = ""
    symbol: str = ""
    category: str = ""
    universe_key: str = ""
    resolver_trace: dict[str, Any] = field(default_factory=dict)
    issues: list[DataQualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame"] = self.frame.to_dict(orient="records") if isinstance(self.frame, pd.DataFrame) else []
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


class SharedCSVReader:
    """Resolver-backed CSV reader shared by completeness checks and symbol reads."""

    def __init__(
        self,
        market: str = "CN",
        data_dir: str | Path | None = None,
        resolver: CNUniverseResolver | None = None,
    ) -> None:
        settings = get_market_settings(market)
        self.market = settings.market
        self.data_dir = Path(data_dir or settings.data_dir)
        self.resolver = resolver
        if self.market == "CN" and self.resolver is None:
            self.resolver = CNUniverseResolver(data_dir=str(self.data_dir))
        self.issues: list[DataQualityIssue] = []

    def snapshot(self) -> dict[str, Any]:
        return self.resolver.snapshot() if self.resolver is not None else {}

    def physical_directories_for_full_a(self) -> list[Path]:
        if self.resolver is None:
            return []
        return self.resolver.physical_directories_for_full_a()

    def list_symbols(self, universe_key: str = "full_a", category: str | None = None) -> list[str]:
        if self.market == "CN" and str(universe_key or "").strip().lower() == "full_a":
            if self.resolver is not None:
                symbols, _ = self.resolver.collect_full_a_inventory(local_union_fallback_used=True)
                return symbols
        target_category = str(category or universe_key or "").strip().lower()
        if not target_category:
            return []
        category_dir = self.data_dir / target_category
        if not category_dir.exists():
            return []
        return sorted(path.stem for path in category_dir.glob("*.csv") if path.stem.strip())

    def resolve_symbol_path(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        for_write: bool = False,
    ) -> Path | None:
        normalized = str(symbol or "").strip()
        if not normalized:
            return None
        if self.market == "CN" and str(universe_key or "").strip().lower() == "full_a" and self.resolver is not None:
            return self.resolver.resolve_write_path(normalized, universe_key="full_a") if for_write else self.resolver.resolve_symbol_file(normalized, universe_key="full_a")
        target_category = str(category or universe_key or "").strip().lower()
        if not target_category:
            return None
        target_dir = self.data_dir / target_category
        if for_write:
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / f"{normalized}.csv"
        candidate = target_dir / f"{normalized}.csv"
        return candidate if candidate.exists() else None

    def peek_symbol_latest_date(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
    ) -> str:
        """Return the latest trade date for *symbol* without loading the full CSV.

        Uses :func:`peek_latest_date` which reads only the tail bytes of
        the file, making it ~100x faster than a full ``pd.read_csv`` call.
        Returns an 8-digit ``YYYYMMDD`` string, or ``""`` if the file is
        missing or unreadable.
        """
        path = self.resolve_symbol_path(symbol, universe_key=universe_key, category=category)
        if path is None:
            return ""
        return peek_latest_date(path)

    def read_symbol_frame(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
        category: str | None = None,
        start_date: str = "",
        end_date: str = "",
    ) -> SharedCSVReadResult:
        path = self.resolve_symbol_path(symbol, universe_key=universe_key, category=category)
        resolver_trace = self.snapshot()
        if path is None:
            issue = DataQualityIssue(
                path="",
                symbol=str(symbol or ""),
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                issue_type="missing_file",
                severity="error",
                message="symbol not resolved to an existing CSV file",
                resolver_strategy=str(resolver_trace.get("resolution_strategy", "")),
                metadata={"resolver": resolver_trace},
            )
            self.issues.append(issue)
            return SharedCSVReadResult(
                path="",
                symbol=str(symbol or ""),
                category=str(category or ""),
                universe_key=str(universe_key or ""),
                resolver_trace=resolver_trace,
                issues=[issue],
                metadata={"resolved": False},
            )

        result = read_csv_with_diagnostics(
            path,
            symbol=symbol,
            category=str(category or ""),
            universe_key=str(universe_key or ""),
            resolver_strategy=str(resolver_trace.get("resolution_strategy", "")),
            start_date=start_date,
            end_date=end_date,
        )
        self.issues.extend(result.issues)
        return SharedCSVReadResult(
            frame=result.frame,
            path=result.path,
            symbol=result.symbol,
            category=result.category,
            universe_key=result.universe_key,
            resolver_trace=resolver_trace,
            issues=list(result.issues),
            metadata=dict(result.metadata),
        )

    def read_path(
        self,
        path: str | Path,
        *,
        symbol: str = "",
        category: str = "",
        universe_key: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> SharedCSVReadResult:
        result = read_csv_with_diagnostics(
            path,
            symbol=symbol,
            category=category,
            universe_key=universe_key,
            resolver_strategy="explicit_path",
            start_date=start_date,
            end_date=end_date,
        )
        self.issues.extend(result.issues)
        return SharedCSVReadResult(
            frame=result.frame,
            path=result.path,
            symbol=result.symbol,
            category=result.category,
            universe_key=result.universe_key,
            resolver_trace=self.snapshot(),
            issues=list(result.issues),
            metadata=dict(result.metadata),
        )

    def latest_trade_date(self, universe_key: str = "full_a") -> str:
        if self.market == "CN" and str(universe_key or "").strip().lower() == "full_a":
            frames: list[pd.DataFrame] = []
            if self.resolver is not None:
                _symbols, resolved_paths = self.resolver.collect_full_a_inventory(local_union_fallback_used=True)
                for path_str in sorted(set(resolved_paths.values())):
                    resolved = self.read_path(path_str, universe_key=universe_key)
                    if not resolved.frame.empty:
                        frames.append(resolved.frame)
            else:
                for directory in self.physical_directories_for_full_a():
                    for csv_path in sorted(Path(directory).glob("*.csv")):
                        resolved = self.read_path(csv_path, universe_key=universe_key)
                        if not resolved.frame.empty:
                            frames.append(resolved.frame)
            latest = infer_latest_date_from_frames(frames)
            return latest
        frames = []
        for symbol in self.list_symbols(universe_key=universe_key):
            frame = self.read_symbol_frame(symbol, universe_key=universe_key).frame
            if not frame.empty:
                frames.append(frame)
        return infer_latest_date_from_frames(frames)
