"""K线分析后端抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from branch_contracts import BranchResult


class KLineBackend(ABC):
    """所有 K线分析后端的公共接口。"""

    name: str = "base"
    reliability: float = 0.5
    horizon_days: int = 5

    @abstractmethod
    def predict(self, symbol_data: dict[str, pd.DataFrame], stock_pool: list[str]) -> BranchResult:
        """对每只股票的 OHLCV 数据执行 K线分析，返回标准 BranchResult。

        Parameters
        ----------
        symbol_data : dict[str, DataFrame]
            键为股票代码，值为含 open/high/low/close/volume 列的 DataFrame。
        stock_pool : list[str]
            需要输出分数的完整股票池（含无数据的股票）。

        Returns
        -------
        BranchResult
        """
        ...
