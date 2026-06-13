"""Shared helpers for the US simulated portfolio tracker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "results" / "strategy_records" / "US" / "simulated_portfolio_10000"
)
DEFAULT_NOTES_PATH = DEFAULT_BASE_DIR / "latest_notes_payload.md"
DEFAULT_INITIAL_CASH = 7763.03
DEFAULT_HOLDINGS = [
    ("CVX", 4, 199.71),
    ("EOG", 5, 137.19),
    ("COP", 4, 125.53),
    ("AEP", 2, 125.03),
]
DEFAULT_CAPS = {
    "CVX": 6,
    "EOG": 8,
    "COP": 7,
    "AEP": 4,
}
NO_DATA_SAMPLE_LIMIT = 4
THEME_BASKETS = {
    "software": ["MSFT", "NOW", "CRM", "ORCL", "SNOW", "PANW"],
    "ai": ["NVDA", "AVGO", "PLTR", "ANET", "SMCI", "AMD"],
    "semiconductor": ["NVDA", "AMD", "AVGO", "AMAT", "LRCX", "KLAC", "TXN"],
    "energy": ["CVX", "EOG", "COP", "OXY", "DVN", "VLO", "SLB"],
    "defensive": ["AEP", "DUK", "SO", "PG", "KO", "PEP", "WMT"],
}


@dataclass
class TradeOrder:
    symbol: str
    action: str
    shares: int
    price: float
    trade_value: float
    reason: str


def parse_initial_holding(text: str) -> tuple[str, int, float]:
    symbol, shares, avg_cost = text.split(":")
    return symbol.upper(), int(shares), float(avg_cost)


def parse_cap(text: str) -> tuple[str, int]:
    symbol, max_shares = text.split(":")
    return symbol.upper(), int(max_shares)


def safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-9:
        return 0.0
    return value / base


def rank_theme_strength(themes: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        themes.items(),
        key=lambda item: (
            item[1].get("avg_20d_return", 0.0),
            item[1].get("ma20_gt_ma60_ratio", 0.0),
            item[1].get("avg_5d_return", 0.0),
        ),
        reverse=True,
    )


def theme_for_symbol(symbol: str) -> str:
    for theme, symbols in THEME_BASKETS.items():
        if symbol in symbols:
            return theme
    return "other"


def format_theme_lines(themes: dict[str, dict[str, Any]]) -> list[str]:
    ranked = rank_theme_strength(themes)
    lines = []
    for name, payload in ranked:
        lines.append(
            f"- {name}: 20日均值 {payload['avg_20d_return']:.2%}，5日均值 {payload['avg_5d_return']:.2%}，"
            f"MA20>MA60 占比 {payload['ma20_gt_ma60_ratio']:.1%}"
        )
    return lines


__all__ = [
    "DEFAULT_BASE_DIR",
    "DEFAULT_CAPS",
    "DEFAULT_HOLDINGS",
    "DEFAULT_INITIAL_CASH",
    "DEFAULT_NOTES_PATH",
    "NO_DATA_SAMPLE_LIMIT",
    "PROJECT_ROOT",
    "THEME_BASKETS",
    "TradeOrder",
    "format_theme_lines",
    "parse_cap",
    "parse_initial_holding",
    "rank_theme_strength",
    "safe_pct",
    "theme_for_symbol",
]
