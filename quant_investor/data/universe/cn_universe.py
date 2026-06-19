"""CN stock-universe helpers kept source-backed for reproducible imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

LOCAL_UNIVERSE_DIR = Path("data/cn_universe")


@dataclass
class StockUniverse:
    token: str | None = None
    local_dir: Path = LOCAL_UNIVERSE_DIR
    metadata: dict[str, Any] = field(default_factory=dict)

    def load_local_stock_list(self) -> pd.DataFrame:
        candidates = [
            self.local_dir / "stock_list.parquet",
            Path("data/metadata/stock_list.parquet"),
        ]
        for path in candidates:
            if path.exists():
                frame = pd.read_parquet(path)
                for column in ("ts_code", "symbol"):
                    if column in frame.columns:
                        frame[column] = frame[column].astype(str)
                return frame
        return pd.DataFrame(columns=["ts_code", "name", "industry", "market", "list_date"])

    def list_symbols(self) -> list[str]:
        frame = self.load_local_stock_list()
        if frame.empty:
            return []
        column = "ts_code" if "ts_code" in frame.columns else "symbol" if "symbol" in frame.columns else ""
        if not column:
            return []
        return [str(item).strip() for item in frame[column].dropna().tolist() if str(item).strip()]
