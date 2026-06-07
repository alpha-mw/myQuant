"""Offline CSV store compatibility layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_investor.data.sources.base import (
    _filter_ohlcv_by_date,
    _normalize_ohlcv_frame,
)


class CSVStore:
    """Read and write local OHLCV CSV files by symbol."""

    STANDARD_COLUMNS = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_close",
    ]

    def __init__(self, base_dir: str | Path):
        self._base_dir = Path(base_dir)
        self._index: dict[str, Path] = {}

    def _ensure_index(self) -> None:
        self._index = {}
        if not self._base_dir.exists():
            return
        for path in self._base_dir.rglob("*.csv"):
            if path.name.endswith("_fundamental.csv"):
                continue
            self._index[path.stem.upper()] = path

    def resolve_path(self, symbol: str, category: str = "") -> Path:
        normalized = str(symbol).upper()
        if category:
            return self._base_dir / str(category) / f"{normalized}.csv"
        if not self._index:
            self._ensure_index()
        return self._index.get(normalized, self._base_dir / f"{normalized}.csv")

    def read(self, symbol: str, start_date: str = "", end_date: str = "") -> pd.DataFrame:
        path = self.resolve_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)
        frame = pd.read_csv(path)
        normalized = _normalize_ohlcv_frame(frame)
        if normalized.empty:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)
        return _filter_ohlcv_by_date(normalized, start_date, end_date)

    def write(
        self,
        symbol: str,
        df: pd.DataFrame,
        category: str = "",
        append: bool = False,
    ) -> Path:
        path = self.resolve_path(symbol, category)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = _normalize_ohlcv_frame(df)
        if append and path.exists():
            current = _normalize_ohlcv_frame(pd.read_csv(path))
            frame = (
                pd.concat([current, frame], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        frame.to_csv(path, index=False)
        self._index[str(symbol).upper()] = path
        return path

    def list_symbols(self) -> list[str]:
        self._ensure_index()
        return sorted(self._index)

    def symbol_exists(self, symbol: str) -> bool:
        return self.resolve_path(symbol).exists()
