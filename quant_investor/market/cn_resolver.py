"""
CN universe resolver.

`full_a` is a logical universe only. This module maps logical full-A-share
requests onto existing physical board/category folders.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


CN_DIRECTORY_PRIORITY: tuple[str, ...] = ("hs300", "zz500", "zz1000", "other")


@dataclass
class CNResolverTrace:
    universe_key: str = "full_a"
    directory_priority: list[str] = field(default_factory=lambda: list(CN_DIRECTORY_PRIORITY))
    physical_directories_used_for_full_a: list[str] = field(default_factory=list)
    local_union_fallback_used: bool = False
    resolved_file_paths_by_symbol: dict[str, str] = field(default_factory=dict)
    resolution_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CNUniverseResolver:
    """Deterministic CN universe resolver for logical `full_a` views."""

    def __init__(
        self,
        data_dir: str,
        directory_priority: Iterable[str] | None = None,
        directories: Mapping[str, str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.directory_priority = list(directory_priority or CN_DIRECTORY_PRIORITY)
        self.directories: dict[str, Path] = {
            name: Path(path)
            for name, path in (directories.items() if directories else {})
        }
        for name in self.directory_priority:
            self.directories.setdefault(name, self.data_dir / name)
        self.trace = CNResolverTrace(directory_priority=list(self.directory_priority))

    def _candidate_directories(self, universe_key: str) -> list[tuple[str, Path]]:
        key = str(universe_key or "").strip().lower()
        if key in {"full_a", "full_market", "all_a", "all", "full"}:
            return [
                (name, self.directories[name])
                for name in self.directory_priority
                if name in self.directories
            ]
        if key in self.directories:
            return [(key, self.directories[key])]
        return [(key, self.data_dir / key)] if key else []

    def _existing_candidate_directories(self, universe_key: str) -> list[tuple[str, Path]]:
        return [
            (name, directory)
            for name, directory in self._candidate_directories(universe_key)
            if directory.exists()
        ]

    def physical_directories_for_full_a(self) -> list[Path]:
        return [directory for _name, directory in self._existing_candidate_directories("full_a")]

    def _record_directories_used(self, directories: list[Path]) -> None:
        existing = list(self.trace.physical_directories_used_for_full_a)
        for path in directories:
            path_str = str(path)
            if path_str not in existing:
                existing.append(path_str)
        self.trace.physical_directories_used_for_full_a = existing

    def _record_resolution(
        self,
        *,
        strategy: str,
        local_union_fallback_used: bool = False,
        resolved_file_paths_by_symbol: Optional[dict[str, str]] = None,
    ) -> None:
        self.trace.resolution_strategy = strategy
        self.trace.local_union_fallback_used = bool(self.trace.local_union_fallback_used or local_union_fallback_used)
        if resolved_file_paths_by_symbol is not None:
            merged = dict(self.trace.resolved_file_paths_by_symbol)
            for symbol, path in resolved_file_paths_by_symbol.items():
                if symbol not in merged or path:
                    merged[symbol] = path
            self.trace.resolved_file_paths_by_symbol = merged
        self.trace.directory_priority = list(self.directory_priority)
        self.trace.metadata = {
            "universe_key": self.trace.universe_key,
            "physical_directories_used_for_full_a": list(self.trace.physical_directories_used_for_full_a),
        }

    def collect_full_a_inventory(self, *, local_union_fallback_used: bool = False) -> tuple[list[str], dict[str, str]]:
        symbols_by_path: dict[str, str] = {}
        used_directories: list[Path] = []
        for _name, directory in self._existing_candidate_directories("full_a"):
            if directory not in used_directories:
                used_directories.append(directory)
            for csv_file in sorted(directory.glob("*.csv")):
                symbol = csv_file.stem.strip()
                if symbol and symbol not in symbols_by_path:
                    symbols_by_path[symbol] = str(csv_file)
        self._record_directories_used(used_directories)
        self._record_resolution(
            strategy="local_union" if local_union_fallback_used else "directory_union",
            local_union_fallback_used=local_union_fallback_used,
            resolved_file_paths_by_symbol=symbols_by_path,
        )
        return sorted(symbols_by_path), symbols_by_path

    def resolve_symbol_file(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
    ) -> Optional[Path]:
        normalized_symbol = str(symbol or "").strip()
        if not normalized_symbol:
            return None

        resolved_path: Optional[Path] = None
        used_directories: list[Path] = []
        for _name, directory in self._candidate_directories(universe_key):
            if not directory.exists():
                continue
            if directory not in used_directories:
                used_directories.append(directory)
            candidate = directory / f"{normalized_symbol}.csv"
            if candidate.exists():
                resolved_path = candidate
                break

        self._record_directories_used(used_directories if universe_key == "full_a" else used_directories[:1])
        if resolved_path is not None:
            self._record_resolution(
                strategy="priority_lookup",
                resolved_file_paths_by_symbol={normalized_symbol: str(resolved_path)},
            )
        else:
            self._record_resolution(
                strategy="priority_lookup_missing",
                resolved_file_paths_by_symbol={normalized_symbol: ""},
            )
        return resolved_path

    def resolve_write_path(
        self,
        symbol: str,
        *,
        universe_key: str = "full_a",
    ) -> Path:
        resolved = self.resolve_symbol_file(symbol, universe_key=universe_key)
        if resolved is not None:
            return resolved

        normalized_symbol = str(symbol or "").strip()
        if not normalized_symbol:
            raise ValueError("symbol 不能为空")

        if str(universe_key or "").strip().lower() == "full_a":
            fallback_dir = self.directories.get("other") or self.data_dir / "other"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self._record_directories_used([fallback_dir])
            self._record_resolution(
                strategy="priority_lookup_fallback_write",
                local_union_fallback_used=True,
                resolved_file_paths_by_symbol={normalized_symbol: str(fallback_dir / f"{normalized_symbol}.csv")},
            )
            return fallback_dir / f"{normalized_symbol}.csv"

        target_dir = self.directories.get(str(universe_key).strip().lower())
        if target_dir is None:
            target_dir = self.data_dir / str(universe_key).strip().lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        self._record_directories_used([target_dir])
        self._record_resolution(
            strategy="category_write",
            resolved_file_paths_by_symbol={normalized_symbol: str(target_dir / f"{normalized_symbol}.csv")},
        )
        return target_dir / f"{normalized_symbol}.csv"

    def snapshot(self) -> dict[str, Any]:
        return self.trace.to_dict()
