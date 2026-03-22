"""
统一市场回测入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.market.config import get_market_settings, normalize_categories
from quant_investor.market.analyze import get_all_local_symbols
from quant_investor.portfolio_backtest import PortfolioBacktester


def _load_market_frame(
    market: str,
    categories: list[str],
    data_dir: str | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    settings = get_market_settings(market)
    base_dir = Path(data_dir or settings.data_dir)
    frames: list[pd.DataFrame] = []
    for category in categories:
        symbols = get_all_local_symbols(category, market=settings.market, data_dir=str(base_dir))
        if sample_size:
            symbols = symbols[:sample_size]
        for symbol in symbols:
            csv_path = base_dir / category / f"{symbol}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            date_column = "trade_date" if "trade_date" in df.columns else "Date" if "Date" in df.columns else "date"
            if date_column not in df.columns:
                continue
            close_column = "close" if "close" in df.columns else "Close"
            volume_column = "vol" if "vol" in df.columns else "volume" if "volume" in df.columns else "Volume"
            normalized = pd.DataFrame(
                {
                    "date": pd.to_datetime(df[date_column], errors="coerce"),
                    "symbol": symbol,
                    "close": pd.to_numeric(df[close_column], errors="coerce"),
                    "volume": pd.to_numeric(df.get(volume_column, 0), errors="coerce"),
                }
            ).dropna(subset=["date", "close"])
            if normalized.empty:
                continue
            normalized["forward_ret_1d"] = normalized["close"].shift(-1) / normalized["close"] - 1
            normalized["factor_score"] = normalized["close"].pct_change(20).fillna(0.0)
            normalized["benchmark_return"] = 0.0
            frames.append(normalized)
    if not frames:
        raise ValueError("未找到可回测的本地市场数据，请先执行 market download。")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["forward_ret_1d"]).sort_values(["date", "symbol"])
    if frame.empty:
        raise ValueError("本地数据不足以构造回测标签，请检查时间跨度。")
    return frame


def run_market_backtest(
    market: str,
    categories: list[str] | None = None,
    data_dir: str | None = None,
    sample_size: int | None = None,
    initial_capital: float = 1_000_000,
    n_holdings: int = 10,
    rebalance_freq: str = "W",
) -> dict[str, Any]:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)
    frame = _load_market_frame(
        settings.market,
        selected_categories,
        data_dir=data_dir,
        sample_size=sample_size,
    )
    backtester = PortfolioBacktester(
        frame,
        initial_capital=initial_capital,
        n_holdings=n_holdings,
        rebalance_freq=rebalance_freq,
    )
    result = backtester.run_simple()
    target_dir = Path(settings.backtest_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{settings.market}_backtest_report.md"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(result.backtest_report)
    return {
        "result": result,
        "report_path": str(report_path),
        "categories": selected_categories,
        "rows": len(frame),
    }
