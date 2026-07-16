#!/usr/bin/env python3
"""
美股全量激进分析运行器 - 读取 Kronos 本地 CSV 数据

修改点：
1. YahooDataSource.get_ohlcv() → 优先读 Kronos 本地 CSV（标记为 real，非 synthetic）
2. get_all_local_symbols() → 从 Kronos 数据目录获取股票列表
3. risk_level = '激进'
4. capital = 100000, top_k = 8
"""

import os
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KRONOS_DATA_DIR = Path(
    os.environ.get("KRONOS_DATA_DIR", PROJECT_ROOT / "data" / "us_market_full")
).expanduser()

# ──────────────────────────────────────────────
# Step 1: 补丁 YahooDataSource.get_ohlcv()
# ──────────────────────────────────────────────
import quant_investor.enhanced_data_layer as edl

_symbol_category_cache: dict = {}

def _find_local_csv(symbol: str) -> str | None:
    """在 Kronos 各子目录中查找 CSV 文件路径。"""
    if symbol in _symbol_category_cache:
        cat = _symbol_category_cache[symbol]
        path = KRONOS_DATA_DIR / cat / f"{symbol}.csv"
        return str(path) if path.exists() else None

    for cat in ("large_cap", "mid_cap", "small_cap"):
        path = KRONOS_DATA_DIR / cat / f"{symbol}.csv"
        if path.exists():
            _symbol_category_cache[symbol] = cat
            return str(path)
    return None


def _local_csv_get_ohlcv(self, symbol: str, start_date: str,
                          end_date: str, freq: str = "1d") -> pd.DataFrame:
    """
    优先读取 Kronos 本地 CSV；若不存在则回退到原始 yfinance。
    本地 CSV 列名：Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
    """
    csv_path = _find_local_csv(symbol)
    if csv_path:
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

            # 日期过滤
            if start_date:
                sd = pd.to_datetime(str(start_date).replace("-", ""))
                df = df[df["date"] >= sd]
            if end_date:
                ed = pd.to_datetime(str(end_date).replace("-", ""))
                df = df[df["date"] <= ed]

            df = df.sort_values("date").reset_index(drop=True)
            if not df.empty:
                print(f"[LocalCSV] {symbol}: {len(df)} 行，"
                      f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
                return df
        except Exception as e:
            print(f"[LocalCSV] 读取 {symbol} 失败，回退 yfinance: {e}")

    # 回退到原始 yfinance 实现
    return _original_yahoo_get_ohlcv(self, symbol, start_date, end_date, freq)


# 保存原始方法并注入补丁
_original_yahoo_get_ohlcv = edl.YahooDataSource.get_ohlcv
edl.YahooDataSource.get_ohlcv = _local_csv_get_ohlcv

print("✅ YahooDataSource.get_ohlcv 已补丁，优先读取 Kronos 本地 CSV")

# ──────────────────────────────────────────────
# Step 2: 导入批量分析模块（已继承补丁）
# ──────────────────────────────────────────────
from quant_investor.market import analyze as batch_module

# ──────────────────────────────────────────────
# Step 3: 补丁 get_all_local_symbols() → Kronos 路径
# ──────────────────────────────────────────────
def _kronos_get_all_local_symbols(category: str):
    data_dir = KRONOS_DATA_DIR / category
    if not data_dir.exists():
        print(f"[Kronos] 目录不存在: {data_dir}")
        return []
    csv_files = [path.name for path in data_dir.iterdir() if path.suffix == ".csv"]
    symbols = [f.replace(".csv", "") for f in csv_files]
    print(f"[Kronos] {category}: {len(symbols)} 只股票")
    return symbols

batch_module.get_all_local_symbols = _kronos_get_all_local_symbols
print("✅ get_all_local_symbols 已补丁，读取 Kronos 股票列表")

# ──────────────────────────────────────────────
# Step 4: 包装 current v15 analyze_batch() → risk_level='激进'
# ──────────────────────────────────────────────
_original_analyze_batch = batch_module.analyze_batch

def _aggressive_analyze_batch(symbols, category, batch_id, **kwargs):
    """Run the current v15 producer with aggressive US settings."""

    for key in (
        "market",
        "total_capital",
        "risk_level",
        "verbose",
        "analysis_kwargs",
    ):
        kwargs.pop(key, None)
    return _original_analyze_batch(
        symbols,
        category,
        batch_id,
        market="US",
        total_capital=100_000,
        risk_level="激进",
        verbose=False,
        analysis_kwargs={
            "enable_kline": True,
            "enable_kronos": True,
            "enable_agent_layer": True,
        },
        **kwargs,
    )

batch_module.analyze_batch = _aggressive_analyze_batch
print("✅ analyze_batch 已补丁，使用激进风格 + capital=100,000")

# ──────────────────────────────────────────────
# Step 5: 确保结果目录存在
# ──────────────────────────────────────────────
results_dir = PROJECT_ROOT / "results" / "v15" / "us_analysis_full"
results_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Step 6: 运行全量分析
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 美股全量激进分析启动")
    print(f"   数据来源: {KRONOS_DATA_DIR}")
    print(f"   总资金: $100,000")
    print(f"   风格: 激进")
    print(f"   Top-K: 8")
    print("="*80 + "\n")

    all_results = {}
    for cat in ["large_cap", "mid_cap", "small_cap"]:
        results = batch_module.analyze_category_full(
            cat,
            market="US",
            batch_size=25,
            output_dir=str(results_dir),
        )
        if results:
            all_results[cat] = results

    if all_results:
        batch_module.generate_full_report(
            all_results,
            market="US",
            output_dir=str(results_dir),
            total_capital=100_000,
            top_k=8,
        )
        print("\n✅ 全量分析完成！结果保存在 results/v15/us_analysis_full/")
    else:
        print("\n⚠️ 所有类别均无结果，请检查数据路径。")
