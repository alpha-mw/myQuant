from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    tushare_token: Optional[str]
    start_date: str
    end_date: str
    data_dir: Path
    benchmark_code: str


@dataclass
class StrategyConfig:
    initial_weight_momentum: float
    step: float
    max_iterations: int
    top_n: int
    rebalance: str
    risk_free_rate: float
    max_position_pct: float
    daily_stop_loss: float


@dataclass
class ReportConfig:
    output_dir: Path


@dataclass
class AppConfig:
    data: DataConfig
    strategy: StrategyConfig
    report: ReportConfig


DEFAULT_CONFIG = {
    "data": {
        "tushare_token": None,
        "start_date": "20210101",
        "end_date": "20260101",
        "data_dir": "data",
        "benchmark_code": "399300.SZ",
    },
    "strategy": {
        "initial_weight_momentum": 0.5,
        "step": 0.05,
        "max_iterations": 20,
        "top_n": 50,
        "rebalance": "monthly",
        "risk_free_rate": 0.03,
        "max_position_pct": 0.10,
        "daily_stop_loss": 0.02,
    },
    "report": {
        "output_dir": "outputs",
    },
}


def load_config(path: Optional[Path] = None) -> AppConfig:
    config_data = DEFAULT_CONFIG.copy()
    if path and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
        for section, value in user_config.items():
            if section in config_data and isinstance(value, dict):
                config_data[section].update(value)
            else:
                config_data[section] = value

    data_cfg = config_data["data"]
    strategy_cfg = config_data["strategy"]
    report_cfg = config_data["report"]

    return AppConfig(
        data=DataConfig(
            tushare_token=data_cfg.get("tushare_token"),
            start_date=data_cfg.get("start_date"),
            end_date=data_cfg.get("end_date"),
            data_dir=Path(data_cfg.get("data_dir")),
            benchmark_code=data_cfg.get("benchmark_code"),
        ),
        strategy=StrategyConfig(
            initial_weight_momentum=strategy_cfg.get("initial_weight_momentum", 0.5),
            step=strategy_cfg.get("step", 0.05),
            max_iterations=strategy_cfg.get("max_iterations", 20),
            top_n=strategy_cfg.get("top_n", 50),
            rebalance=strategy_cfg.get("rebalance", "monthly"),
            risk_free_rate=strategy_cfg.get("risk_free_rate", 0.03),
            max_position_pct=strategy_cfg.get("max_position_pct", 0.1),
            daily_stop_loss=strategy_cfg.get("daily_stop_loss", 0.02),
        ),
        report=ReportConfig(output_dir=Path(report_cfg.get("output_dir"))),
    )
