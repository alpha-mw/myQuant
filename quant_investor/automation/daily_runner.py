#!/usr/bin/env python3
"""
myQuant 每日 A 股自动分析脚本

用法:
  python daily_runner.py                 # 立即运行一次完整分析
  python daily_runner.py --daemon        # 定时守护：每天定时运行分析
  python daily_runner.py --report-only   # 打印最新策略记录中的正式分析报告
  python daily_runner.py --dry-run       # 验证配置和 strategy_records 输入，不实际运行
  python daily_runner.py --skip-stage1   # 跳过 Stage 1（数据检查与下载），直接分析
  python daily_runner.py --skip-download # 跳过数据下载，直接分析
  python daily_runner.py --config PATH   # 指定配置文件路径
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

from quant_investor.config import config as runtime_config
from quant_investor.llm_provider_priority import (
    coerce_review_model_priority,
    normalize_model_name,
)

# ── 项目根目录 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{3,6}$")
DEFAULT_DAILY_REPORT_DIR = "reports/v15/daily"


def resolve_daily_report_dir(value: Any) -> Path:
    """Resolve a v15 daily output and reject the frozen v13 evidence tree."""

    raw = str(value or DEFAULT_DAILY_REPORT_DIR).strip()
    report_dir = Path(raw)
    resolved = (
        report_dir.resolve(strict=False)
        if report_dir.is_absolute()
        else (ROOT / report_dir).resolve(strict=False)
    )
    frozen = (ROOT / "reports/daily").resolve(strict=False)
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError(
            "reports/daily is frozen v13 retirement evidence; "
            "set report_dir to reports/v15/daily"
        )
    return resolved


def run_staged_maintenance(**kwargs: Any) -> dict[str, Any]:
    from quant_investor.market.staged_maintenance import run_staged_maintenance as _run

    return _run(**kwargs)


def _legacy_review_model_fields(config: dict[str, Any]) -> list[str]:
    return [
        str(config.get("agent_model", "") or ""),
        str(config.get("agent_fallback_model", "") or ""),
        str(config.get("master_model", "") or ""),
        str(config.get("master_fallback_model", "") or ""),
    ]


def _resolve_review_model_priority(config: dict[str, Any]) -> list[str]:
    return coerce_review_model_priority(
        config.get("review_model_priority", []),
        legacy_models=_legacy_review_model_fields(config),
    )


def _normalize_role_model_overrides(config: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in (
        "agent_model",
        "agent_fallback_model",
        "master_model",
        "master_fallback_model",
    ):
        value = normalize_model_name(str(config.get(key, "") or ""))
        if value:
            normalized[key] = value
    return normalized


def _bootstrap_project_venv() -> None:
    """If available, re-exec into the project .venv Python."""
    venv_root = (ROOT / ".venv").resolve()
    target = venv_root / "bin" / "python"
    if not target.exists():
        return

    try:
        current_prefix = Path(sys.prefix).resolve()
        if current_prefix == venv_root:
            return
    except Exception:
        if str(sys.prefix).startswith(str(venv_root)):
            return

    sys.stderr.write(f"[daily_runner] switching interpreter to {target}\n")
    sys.stderr.flush()
    os.execv(str(target), [str(target), *sys.argv])

# ── 日志 ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("daily_runner")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 配置加载
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """从 daily_config.py（或指定路径）加载配置。"""
    path = Path(config_path) if config_path else ROOT / "daily_config.py"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}\n请先创建 daily_config.py，参考项目文档。")

    spec = importlib.util.spec_from_file_location("_daily_cfg", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    cfg: dict[str, Any] = getattr(mod, "DAILY_CONFIG", {})
    if not cfg:
        raise ValueError(f"配置文件 {path} 中未找到 DAILY_CONFIG 字典。")

    normalized = dict(cfg)
    normalized["review_model_priority"] = _resolve_review_model_priority(cfg)

    # 默认值补全
    defaults: dict[str, Any] = {
        "market": "CN",
        "universe": "full_a",
        "risk_level": "中等",
        "total_capital": 1_000_000,
        "review_model_priority": ["deepseek-chat", "moonshot-v1-128k", "qwen3.5-plus"],
        "master_model": "moonshot-v1-128k",
        "master_fallback_model": "deepseek-reasoner",
        "master_reasoning_effort": "",
        "funnel_profile": runtime_config.FUNNEL_PROFILE,
        "funnel_max_candidates": runtime_config.FUNNEL_MAX_CANDIDATES,
        "trend_windows": list(runtime_config.FUNNEL_TREND_WINDOWS),
        "volume_spike_threshold": runtime_config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
        "breakout_distance_pct": runtime_config.FUNNEL_BREAKOUT_DISTANCE_PCT,
        "bayesian_shortlist_size": 50,
        "freshness_mode": "stable",
        "kline_backend": "heuristic",
        "top_k": 20,
        "agent_timeout": runtime_config.DEFAULT_AGENT_TIMEOUT_SECONDS,
        "master_timeout": runtime_config.DEFAULT_MASTER_TIMEOUT_SECONDS,
        "enable_agent_layer": False,
        "skip_stage1": False,
        "skip_download": False,
        "years": 3,
        "workers": 4,
        "maintenance_batch_size": 200,
        "maintenance_max_batches_per_run": 200,
        "maintenance_min_symbol_success_rate": 0.95,
        "maintenance_target_date": "auto",
        "maintenance_daily_window": True,
        "schedule_time": "17:30",
        "report_dir": DEFAULT_DAILY_REPORT_DIR,
    }
    for key, val in defaults.items():
        normalized.setdefault(key, val)
    normalized.update(_normalize_role_model_overrides(normalized))
    for key in (
        "pipeline_mode",
        "history_lookback",
        "backend_host",
        "backend_port",
    ):
        normalized.pop(key, None)
    return normalized


def _normalize_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("report_dir", DEFAULT_DAILY_REPORT_DIR)
    resolve_daily_report_dir(normalized["report_dir"])
    normalized["review_model_priority"] = _resolve_review_model_priority(normalized)
    normalized.update(_normalize_role_model_overrides(normalized))
    for key in (
        "pipeline_mode",
        "history_lookback",
        "backend_host",
        "backend_port",
    ):
        normalized.pop(key, None)
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# 2. 策略记录历史
# ══════════════════════════════════════════════════════════════════════════════

def _strategy_records_market_root(market: str) -> Path:
    return ROOT / "results" / "strategy_records" / str(market or "CN").upper()


def _dedupe_text(values: list[str], limit: int = 8) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
        if len(ordered) >= limit:
            break
    return ordered



from quant_investor.automation.analysis_runner import (
    AnalysisRunner,
    _maintenance_categories,
    _run_automation_data_update_preflight,
)
from quant_investor.automation.history_loader import HistoryLoader
from quant_investor.automation.persistence import PersistenceManager
from quant_investor.automation.report_builder import ReportBuilder


# ══════════════════════════════════════════════════════════════════════════════
# 4. 分析执行入口
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
# 5. 报告生成（8 章节）
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# 6. 持久化
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# 7. 主流程函数
# ══════════════════════════════════════════════════════════════════════════════

def run_once(
    config: dict[str, Any],
    skip_download: bool = False,
    skip_stage1: bool = False,
) -> str:
    """执行一次完整分析，返回报告路径。"""
    config = _normalize_runtime_config(config)
    if skip_stage1:
        config = {**config, "skip_stage1": True}
    if skip_download:
        config = {**config, "skip_download": True}

    history_loader = HistoryLoader()
    runner = AnalysisRunner()
    builder = ReportBuilder()
    persist = PersistenceManager()

    history = history_loader.load_recent(market=config["market"], max_dates=5)
    recall_context = history_loader.build_recall_context(history, market=config["market"])
    log.info(
        "已加载 %d 条策略记录，覆盖最近 %d 个日期",
        len(history),
        len(recall_context.get("window_dates", [])),
    )

    pipeline_result = runner.run(config, recall_context=recall_context)

    log.info("生成决策报告...")
    report_md = builder.build(pipeline_result, config, history)

    report_path = persist.save(report_md, pipeline_result, config)

    # 打印报告到控制台
    print("\n" + "=" * 80)
    print(report_md)
    print("=" * 80)

    log.info("分析完成，report_path=%s", report_path)
    return report_path


def run_daemon(config: dict[str, Any]) -> None:
    """守护模式：每天定时运行分析。"""
    try:
        schedule_dt = datetime.strptime(config["schedule_time"], "%H:%M")
        schedule_time = schedule_dt.time()
    except ValueError:
        log.error("无效的 schedule_time 格式: %s（应为 HH:MM）", config["schedule_time"])
        sys.exit(1)

    def _shutdown(signum: int, frame: Any) -> None:
        log.info("收到信号 %s，正在退出...", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("守护模式启动 | 每日 %s 自动分析", config["schedule_time"])

    last_run_date: Optional[date] = None

    while True:
        now = datetime.now()
        today = now.date()

        if now.time() >= schedule_time and today != last_run_date:
            log.info("触发每日分析（%s %s）...", today, config["schedule_time"])
            try:
                run_once(config)
                last_run_date = today
            except Exception as exc:
                log.error("每日分析失败: %s", exc, exc_info=True)
                # 失败后记录错误，但不更新 last_run_date，允许当天重试
                # 为避免频繁重试，等待 30 分钟
                log.info("30 分钟后重试...")
                time.sleep(1800)
                continue

        time.sleep(60)


def print_last_report(config: dict[str, Any]) -> None:
    """打印最后一次分析报告。"""
    loader = HistoryLoader()
    report = loader.load_last_report(config["market"])
    if not report:
        print("暂无策略记录报告。请先确认 results/strategy_records 下存在正式 run 目录。")
        return
    print(report)


def dry_run(config: dict[str, Any]) -> None:
    """验证配置和策略记录输入，不实际运行分析。"""
    config = _normalize_runtime_config(config)
    print("=== DRY RUN 模式 ===\n")

    print("✓ 配置加载成功")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n检查策略记录目录...")
    loader = HistoryLoader()
    market_root = _strategy_records_market_root(config["market"])
    print(f"  market_root: {market_root}")
    if market_root.exists():
        print("✓ 市场策略记录目录存在")
    else:
        print("✗ 市场策略记录目录不存在")

    try:
        runs = loader.load_recent(market=config["market"], max_dates=5)
        recall_context = loader.build_recall_context(runs, market=config["market"])
        print(f"✓ 最近 5 个日期共解析 {len(runs)} 条策略记录")
        print(f"  window_dates: {recall_context.get('window_dates', [])}")
        print(f"  recent_symbols: {recall_context.get('recent_symbols', [])[:10]}")
    except Exception as exc:
        print(f"✗ 策略记录解析失败: {exc}")

    print("\n检查 Python 环境...")
    try:
        from quant_investor.market.run_pipeline import run_unified_pipeline  # noqa: F401
        print("✓ quant_investor 包可导入")
    except ImportError as exc:
        print(f"✗ quant_investor 导入失败: {exc}")

    print("\n决策主线: unified_dag")
    print(f"Review 模型优先级: {' -> '.join(config.get('review_model_priority', []))}")
    try:
        from quant_investor.bayesian import BayesianPosteriorEngine, HierarchicalPriorBuilder  # noqa: F401
        from quant_investor.funnel import DeterministicFunnel  # noqa: F401
        from quant_investor.global_context import GlobalContextBuilder  # noqa: F401
        print("✓ Bayesian pipeline 模块可导入")
    except ImportError as exc:
        print(f"✗ Bayesian pipeline 导入失败: {exc}")

    print("\nDRY RUN 完成。")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="myQuant 每日 A 股分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="守护模式：每天定时分析",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="打印最新策略记录目录中的正式分析报告",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="验证配置和 strategy_records 输入，不实际运行",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="跳过 Stage 1（数据新鲜度检查与下载），直接进入分析",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="跳过 Stage 1（数据新鲜度检查与下载）的兼容别名",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过数据下载，直接分析（数据需提前准备好）",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="配置文件路径（默认: daily_config.py）",
    )
    parser.add_argument(
        "--master-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="",
        help="覆盖 Master Agent 的 reasoning 强度（默认使用配置文件中的值）",
    )
    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as exc:
        log.error("配置加载失败: %s", exc)
        sys.exit(1)

    if args.master_reasoning_effort:
        config["master_reasoning_effort"] = args.master_reasoning_effort
    if args.skip_stage1 or args.skip_data_check:
        config["skip_stage1"] = True
    config = _normalize_runtime_config(config)

    # 分支执行
    if args.dry_run:
        dry_run(config)
    elif args.report_only:
        print_last_report(config)
    elif args.daemon:
        run_daemon(config)
    else:
        # 默认：立即运行一次分析
        try:
            run_once(
                config,
                skip_download=args.skip_download,
                skip_stage1=bool(config.get("skip_stage1", False)),
            )
        except KeyboardInterrupt:
            log.info("用户中断。")
            sys.exit(0)
        except Exception as exc:
            log.error("分析失败: %s", exc, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    _bootstrap_project_venv()
    main()
