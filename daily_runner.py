#!/usr/bin/env python3
"""
myQuant 每日 A 股自动分析脚本

用法:
  python daily_runner.py                 # 立即运行一次完整分析
  python daily_runner.py --daemon        # 后台守护：启动后端 + 每天定时运行
  python daily_runner.py --backend-only  # 仅启动并守护后端（不做分析）
  python daily_runner.py --report-only   # 打印上次分析报告
  python daily_runner.py --dry-run       # 验证配置和连接，不实际运行
  python daily_runner.py --skip-stage1   # 跳过 Stage 1（数据检查与下载），直接分析
  python daily_runner.py --skip-download # 跳过数据下载，直接分析
  python daily_runner.py --config PATH   # 指定配置文件路径
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen
from urllib.error import URLError

# ── 项目根目录 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()


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

    # 默认值补全
    defaults: dict[str, Any] = {
        "market": "CN",
        "universe": "full_a",
        "risk_level": "中等",
        "total_capital": 1_000_000,
        "agent_model": "moonshot-v1-128k",
        "agent_fallback_model": "deepseek-reasoner",
        "master_model": "moonshot-v1-128k",
        "master_fallback_model": "deepseek-chat",
        "master_reasoning_effort": "",
        "pipeline_mode": "bayesian",
        "funnel_max_candidates": 400,
        "bayesian_shortlist_size": 20,
        "freshness_mode": "stable",
        "kline_backend": "heuristic",
        "top_k": 20,
        "agent_timeout": 20.0,
        "master_timeout": 45.0,
        "enable_agent_layer": True,
        "skip_stage1": False,
        "skip_download": False,
        "years": 3,
        "workers": 4,
        "schedule_time": "17:30",
        "report_dir": "reports/daily",
        "history_lookback": 5,
        "backend_host": "127.0.0.1",
        "backend_port": 8000,
    }
    for key, val in defaults.items():
        cfg.setdefault(key, val)

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# 2. 后端管理
# ══════════════════════════════════════════════════════════════════════════════

class BackendManager:
    """启动、监控并自动重启 FastAPI/uvicorn 后端。"""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._proc: Optional[subprocess.Popen] = None
        self._health_url = f"http://{host}:{port}/api/health"

    def _find_python(self) -> str:
        """优先使用项目 venv，回退到当前 Python。"""
        venv_python = ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        return sys.executable

    def start(self) -> subprocess.Popen:
        """启动 uvicorn 后端进程。"""
        python = self._find_python()
        cmd = [
            python, "-m", "uvicorn", "web.api:app",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "warning",
        ]
        log.info("启动后端: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # 等待就绪（最多 15 秒）
        for _ in range(30):
            time.sleep(0.5)
            if self.is_healthy():
                log.info("后端已就绪: http://%s:%s", self.host, self.port)
                return proc
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"后端启动失败（退出码 {proc.returncode}）:\n{output}")
        log.warning("后端启动超时，继续尝试...")
        return proc

    def is_healthy(self) -> bool:
        """检查后端健康状态。"""
        try:
            with urlopen(self._health_url, timeout=3) as resp:
                data = json.loads(resp.read())
                return bool(data.get("ok"))
        except Exception:
            return False

    def ensure_running(self, proc: Optional[subprocess.Popen]) -> subprocess.Popen:
        """若后端进程已死或不健康，则重启。"""
        if proc is None or proc.poll() is not None:
            log.warning("后端进程已停止，正在重启...")
            return self.start()
        if not self.is_healthy():
            log.warning("后端健康检查失败，尝试重启...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return self.start()
        return proc

    def stop(self, proc: Optional[subprocess.Popen]) -> None:
        if proc and proc.poll() is None:
            log.info("正在停止后端...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ══════════════════════════════════════════════════════════════════════════════
# 3. 历史记录加载
# ══════════════════════════════════════════════════════════════════════════════

class HistoryLoader:
    """从 web_runs.db 加载历史分析记录，用于报告上下文。"""

    def __init__(self) -> None:
        self._store: Any = None

    def _get_store(self) -> Any:
        if self._store is None:
            from web.services.run_history_store import RunHistoryStore
            self._store = RunHistoryStore()
            self._store.init_db()
        return self._store

    def load_recent(self, n: int, market: str = "CN") -> list[dict[str, Any]]:
        """加载最近 n 次分析记录。"""
        try:
            store = self._get_store()
            items, _ = store.get_history(page=1, per_page=n, market=market)
            return items
        except Exception as exc:
            log.warning("加载历史记录失败: %s", exc)
            return []

    def load_last_report(self, market: str = "CN") -> Optional[str]:
        """读取上次分析的 Markdown 报告。"""
        try:
            store = self._get_store()
            items, _ = store.get_history(page=1, per_page=1, market=market)
            if not items:
                return None
            run = store.get_run(items[0]["job_id"])
            if run:
                return run.get("report_markdown", "")
            return None
        except Exception as exc:
            log.warning("读取上次报告失败: %s", exc)
            return None

    def format_context_section(self, runs: list[dict[str, Any]]) -> str:
        """格式化历史摘要，作为报告中的"历史上下文"章节。"""
        if not runs:
            return "_暂无历史分析记录。_"

        lines = []
        for run in runs:
            created = run.get("created_at", "")[:16]
            status = run.get("status", "")
            stocks = run.get("stock_pool", [])
            if isinstance(stocks, str):
                try:
                    stocks = json.loads(stocks)
                except Exception:
                    stocks = []
            stock_count = len(stocks) if isinstance(stocks, list) else 0
            risk = run.get("risk_level", "")
            job_id = run.get("job_id", "")[:8]

            lines.append(
                f"- `{created}` | {status} | 分析 {stock_count} 只股票 | "
                f"风险偏好: {risk} | ID: `{job_id}`"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 分析执行
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisRunner:
    """包装 run_unified_pipeline，执行全量 A 股分析。"""

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """执行全量市场分析，返回 pipeline 结果字典。"""
        # 确保工作目录正确（pipeline 依赖相对路径）
        os.chdir(ROOT)

        from quant_investor.market.run_pipeline import run_unified_pipeline
        from quant_investor.model_roles import resolve_model_role

        branch_resolution = resolve_model_role(
            role="branch",
            primary_model=str(config.get("agent_model", "")),
            fallback_model=str(config.get("agent_fallback_model", "")),
        )
        master_resolution = resolve_model_role(
            role="master",
            primary_model=str(config.get("master_model", "")),
            fallback_model=str(config.get("master_fallback_model", "")),
        )
        config = dict(config)
        config["agent_model"] = branch_resolution.resolved_model
        config["master_model"] = master_resolution.resolved_model
        config.setdefault("universe", "full_a")
        config.setdefault("skip_stage1", bool(config.get("skip_data_check", False)))
        config["model_role_resolution"] = {
            "branch": branch_resolution.to_dict(),
            "master": master_resolution.to_dict(),
        }

        # Propagate Bayesian pipeline settings to Config via env vars
        pipeline_mode = config.get("pipeline_mode", "bayesian")
        os.environ.setdefault("PIPELINE_MODE", pipeline_mode)
        os.environ.setdefault("FUNNEL_MAX_CANDIDATES", str(config.get("funnel_max_candidates", 400)))
        os.environ.setdefault("BAYESIAN_SHORTLIST_SIZE", str(config.get("bayesian_shortlist_size", 20)))
        os.environ.setdefault("CN_FRESHNESS_MODE", config.get("freshness_mode", "stable"))

        log.info(
            "开始分析 | market=%s | universe=%s | pipeline_mode=%s | branch_model=%s%s | master_model=%s%s | master_reasoning_effort=%s | top_k=%s | skip_stage1=%s",
            config["market"],
            config.get("universe", "full_a"),
            pipeline_mode,
            config["agent_model"] or "(默认)",
            " [fallback]" if branch_resolution.fallback_used else "",
            config["master_model"] or "(默认)",
            " [fallback]" if master_resolution.fallback_used else "",
            config.get("master_reasoning_effort", "") or "(默认)",
            config["top_k"],
            bool(config.get("skip_stage1", False)),
        )
        if branch_resolution.fallback_used:
            log.warning(
                "branch model fallback activated: primary=%s fallback=%s reason=%s",
                branch_resolution.primary_model,
                branch_resolution.fallback_model,
                branch_resolution.fallback_reason,
            )
        if master_resolution.fallback_used:
            log.warning(
                "master model fallback activated: primary=%s fallback=%s reason=%s",
                master_resolution.primary_model,
                master_resolution.fallback_model,
                master_resolution.fallback_reason,
            )
        started = time.time()

        def _call_pipeline(skip_dl: bool) -> dict[str, Any]:
            return run_unified_pipeline(
                market=config["market"],
                universe=config.get("universe", "full_a"),
                mode="batch",
                skip_stage1=bool(config.get("skip_stage1", False)),
                skip_download=skip_dl,
                total_capital=config["total_capital"],
                top_k=config["top_k"],
                years=config["years"],
                workers=config["workers"],
                enable_agent_layer=config["enable_agent_layer"],
                agent_model=config["agent_model"],
                agent_fallback_model=config.get("agent_fallback_model", "deepseek-reasoner"),
                master_model=config["master_model"],
                master_fallback_model=config.get("master_fallback_model", "deepseek-chat"),
                master_reasoning_effort=config.get("master_reasoning_effort", ""),
                agent_timeout=config["agent_timeout"],
                master_timeout=config["master_timeout"],
                verbose=True,
            )

        try:
            result = _call_pipeline(skip_dl=config["skip_download"])
        except RuntimeError as exc:
            msg = str(exc)
            if "tushare" in msg and ("未安装" in msg or "not installed" in msg):
                log.warning(
                    "tushare 下载阶段失败（%s），自动切换到 skip_download=True 使用本地数据...",
                    msg,
                )
                result = _call_pipeline(skip_dl=True)
            else:
                raise

        elapsed = time.time() - started
        log.info("分析完成，耗时 %.0f 秒（%.1f 分钟）", elapsed, elapsed / 60)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. 报告生成（8 章节）
# ══════════════════════════════════════════════════════════════════════════════

_BRANCH_LABEL_MAP = {
    "kline": "K线技术",
    "quant": "量化因子",
    "fundamental": "基本面",
    "intelligence": "智能融合",
    "macro": "宏观",
}

_ACTION_EMOJI = {
    "买入": "🟢",
    "轻仓试错": "🟡",
    "观察": "⚪",
    "持有": "🔵",
    "减仓": "🟠",
    "清仓": "🔴",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_average(values: list[Any], default: float = 0.0) -> float:
    nums = [_safe_float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else default


def _confidence_label(c: float) -> str:
    if c >= 0.70:
        return "高"
    if c >= 0.45:
        return "中"
    return "低"


class ReportBuilder:
    """从 pipeline 结果构建 8 章节 Markdown 决策报告。"""

    @staticmethod
    def _display_name(item: dict[str, Any]) -> str:
        symbol = str(item.get("symbol", "")).strip()
        company_name = str(item.get("company_name") or item.get("name") or "").strip()
        return f"{symbol} {company_name}".strip() if company_name else symbol

    def build(
        self,
        pipeline_result: dict[str, Any],
        config: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        all_results: dict[str, list[dict[str, Any]]] = pipeline_result.get("analysis", {})
        reports: dict[str, Any] = pipeline_result.get("reports", {})
        timing: dict[str, Any] = pipeline_result.get("timing", {})
        download: dict[str, Any] = pipeline_result.get("download", {})
        categories: list[str] = pipeline_result.get("categories", [])
        analysis_meta: dict[str, Any] = pipeline_result.get("analysis_meta", {})

        # 聚合分支数据
        branch_summary = self._aggregate_branches(all_results)

        # 构建组合计划
        plan = self._build_plan(all_results, config)
        portfolio_plan: dict[str, Any] = plan.get("portfolio_plan", {})
        recommendations: list[dict[str, Any]] = plan.get("recommendations", [])
        market_summary: dict[str, Any] = plan.get("market_summary", {})

        # 获取 NarratorAgent 生成的报告（如有）
        report_bundle = reports.get("report_bundle")
        narrator_md: str = ""
        executive_summary: list[str] = []
        market_view: str = ""
        if report_bundle is not None:
            narrator_md = getattr(report_bundle, "markdown_report", "") or ""
            executive_summary = list(getattr(report_bundle, "executive_summary", []) or [])
            market_view = str(getattr(report_bundle, "market_view", "") or "")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_stocks = int(market_summary.get("total_stocks", 0))
        selected_count = int(portfolio_plan.get("selected_count", len(recommendations)))

        sections = [
            self._header(config, now_str, total_stocks, selected_count),
            self._history_context(history),
            self._section_data_overview(market_summary, download, categories, config),
            self._section_market_overview(branch_summary, executive_summary, market_view, portfolio_plan),
            self._section_analysis_process(timing, config),
            self._section_bayesian_decision(analysis_meta, config),
            self._section_run_context(analysis_meta, report_bundle),
            self._section_subagent_decisions(branch_summary),
            self._section_master_decisions(executive_summary, market_view, portfolio_plan, narrator_md),
            self._section_investment_recommendations(recommendations, config["market"]),
            self._section_positions_orders(recommendations, portfolio_plan, config),
            self._section_next_steps(history, config, recommendations, timing),
        ]
        return "\n\n---\n\n".join(s for s in sections if s.strip())

    # ── 聚合工具 ──────────────────────────────────────────────────────────────

    def _aggregate_branches(
        self, all_results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        """跨批次聚合各分支得分与结论。"""
        try:
            from quant_investor.market.analyze import _aggregate_branch_summary
            return _aggregate_branch_summary(all_results)
        except Exception as exc:
            log.debug("branch 聚合回退: %s", exc)
            return {}

    def _build_plan(
        self, all_results: dict[str, list[dict[str, Any]]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """构建全市场交易计划。"""
        try:
            from quant_investor.market.analyze import build_full_market_trade_plan
            return build_full_market_trade_plan(
                all_results,
                market=config["market"],
                total_capital=config["total_capital"],
                top_k=config["top_k"],
            )
        except Exception as exc:
            log.debug("trade plan 构建回退: %s", exc)
            return {"portfolio_plan": {}, "recommendations": [], "market_summary": {}}

    # ── 各章节 ───────────────────────────────────────────────────────────────

    def _header(
        self, config: dict, now_str: str, total_stocks: int, selected_count: int
    ) -> str:
        capital_str = f"{config['total_capital']:,.0f}"
        pipeline_mode = config.get("pipeline_mode", "legacy")
        pipeline_label = "Bayesian 7-Layer" if pipeline_mode == "bayesian" else "Legacy 3-Layer"
        return (
            f"# 📊 myQuant 每日 A 股分析报告\n\n"
            f"**生成时间**: {now_str}  \n"
            f"**市场**: {config['market']}  \n"
            f"**决策引擎**: {pipeline_label}  \n"
            f"**风险偏好**: {config['risk_level']}  \n"
            f"**总资金**: ¥{capital_str}  \n"
            f"**分析股票数**: {total_stocks}  \n"
            f"**精选标的数**: {selected_count}  \n"
            f"**Subagent 模型**: {config['agent_model'] or '(系统默认)'}  \n"
            f"**Master Agent 模型**: {config['master_model'] or '(系统默认)'}  \n"
            f"**Master Agent reasoning**: {config.get('master_reasoning_effort', '') or '(系统默认)'}"
        )

    def _history_context(self, history: list[dict[str, Any]]) -> str:
        loader = HistoryLoader()
        context = loader.format_context_section(history)
        return f"## 📚 历史分析上下文（最近 {len(history)} 次）\n\n{context}"

    def _section_data_overview(
        self,
        market_summary: dict,
        download: dict,
        categories: list[str],
        config: dict,
    ) -> str:
        total_stocks = int(market_summary.get("total_stocks", 0))
        total_batches = int(market_summary.get("total_batches", 0))
        generated_at = market_summary.get("generated_at", "N/A")
        download_status = download.get("status", "unknown")
        download_reason = download.get("reason", "")

        cat_lines = []
        for cat_name, cat_data in market_summary.get("categories", {}).items():
            label = cat_data.get("category_name", cat_name)
            count = cat_data.get("stock_count", 0)
            candidate = cat_data.get("candidate_count", 0)
            avg_exp = _safe_float(cat_data.get("avg_target_exposure", 0))
            cat_lines.append(
                f"| {label} | {count} | {candidate} | {avg_exp:.1%} |"
            )

        cat_table = ""
        if cat_lines:
            cat_table = (
                "\n| 板块 | 分析股票数 | 候选标的数 | 平均目标仓位 |\n"
                "|------|-----------|-----------|------------|\n"
                + "\n".join(cat_lines)
            )

        completeness_note = ""
        if isinstance(download.get("completeness_after"), dict):
            blocking = download["completeness_after"].get("blocking_incomplete_count", 0)
            if blocking:
                completeness_note = f"\n\n> ⚠️ 数据完整性：{blocking} 只股票存在阻塞性缺口，已跳过。"

        return (
            f"## § 1 数据概览\n\n"
            f"- **数据生成时间**: {generated_at}\n"
            f"- **数据下载状态**: {download_status}（{download_reason}）\n"
            f"- **分析覆盖**: {total_stocks} 只股票，共 {total_batches} 个批次\n"
            f"- **分析板块**: {', '.join(categories) if categories else '全量'}\n"
            f"{cat_table}{completeness_note}"
        )

    def _section_market_overview(
        self,
        branch_summary: dict,
        executive_summary: list[str],
        market_view: str,
        portfolio_plan: dict,
    ) -> str:
        macro = branch_summary.get("macro", {})
        macro_score = _safe_float(macro.get("score", 0))
        macro_confidence = _safe_float(macro.get("confidence", 0))
        macro_conclusion = macro.get("conclusion", "宏观数据暂未汇总。")

        exec_lines = ""
        if executive_summary:
            exec_lines = "\n**执行摘要（NarratorAgent 三句话）：**\n" + "\n".join(
                f"> {line}" for line in executive_summary
            )

        mv_line = f"\n**市场判断**: {market_view}" if market_view else ""

        style_bias = portfolio_plan.get("style_bias", "均衡")
        target_exp = _safe_float(portfolio_plan.get("target_exposure", 0))
        reliability = _safe_float(portfolio_plan.get("reliability", 0))

        branch_scores_lines = []
        for branch_key, blabel in _BRANCH_LABEL_MAP.items():
            b = branch_summary.get(branch_key, {})
            score = _safe_float(b.get("score", 0))
            conf = _safe_float(b.get("confidence", 0))
            if b:
                bar = "█" * int(abs(score) * 10) if abs(score) > 0.05 else "─"
                sign = "+" if score >= 0 else ""
                branch_scores_lines.append(
                    f"| {blabel} | {sign}{score:.3f} | {bar} | {_confidence_label(conf)}({conf:.2f}) |"
                )

        branch_table = ""
        if branch_scores_lines:
            branch_table = (
                "\n**各分支得分概览：**\n\n"
                "| 分支 | 得分 | 方向 | 可信度 |\n"
                "|------|------|------|-------|\n"
                + "\n".join(branch_scores_lines)
            )

        return (
            f"## § 2 市场概览\n\n"
            f"- **宏观评分**: {macro_score:+.3f}（可信度: {_confidence_label(macro_confidence)}）\n"
            f"- **宏观结论**: {macro_conclusion}\n"
            f"- **组合风格偏向**: {style_bias}\n"
            f"- **建议总仓位**: {target_exp:.1%}\n"
            f"- **整体可信度**: {_confidence_label(reliability)}（{reliability:.2f}）\n"
            f"{branch_table}{exec_lines}{mv_line}"
        )

    def _section_analysis_process(self, timing: dict, config: dict) -> str:
        dl_secs = _safe_float(timing.get("download_seconds", 0))
        an_secs = _safe_float(timing.get("analysis_seconds", 0))
        total_secs = _safe_float(timing.get("total_seconds", 0))

        return (
            f"## § 3 分析过程\n\n"
            f"**时间消耗：**\n"
            f"- 数据下载/检查: {dl_secs:.1f}s\n"
            f"- 分析与报告生成: {an_secs:.1f}s（{an_secs/60:.1f} 分钟）\n"
            f"- 总耗时: {total_secs:.1f}s（{total_secs/60:.1f} 分钟）\n\n"
            f"**分析配置：**\n"
            f"- K线后端: `{config['kline_backend']}`\n"
            f"- Subagent 模型: `{config['agent_model'] or '(系统默认)'}`\n"
            f"- Master Agent 模型: `{config['master_model'] or '(系统默认)'}`\n"
            f"- Master Agent reasoning: `{config.get('master_reasoning_effort', '') or '(系统默认)'}`\n"
            f"- Subagent 超时: {config['agent_timeout']}s\n"
            f"- Master Agent 超时: {config['master_timeout']}s\n"
            f"- Agent Layer 启用: {'是' if config['enable_agent_layer'] else '否'}\n\n"
            f"**决策引擎**: `{config.get('pipeline_mode', 'legacy')}`\n\n"
            + (
                f"**分析层级（Bayesian 7-Layer）：** GlobalContext → 全市场分支（K线+量化） → "
                f"漏斗压缩（{config.get('funnel_max_candidates', 400)} 候选） → 候选分支（基本面+智能融合） → "
                f"Bayesian 后验决策 → Master Discussion（Top {config.get('bayesian_shortlist_size', 20)}） → "
                f"确定性控制链 → 组合构建 → 报告生成"
                if config.get("pipeline_mode") == "bayesian"
                else
                f"**分析层级（Legacy 3-Layer）：** 数据层 → K线层 → 量化因子层 → 基本面层 → "
                f"智能融合层 → 宏观层 → 风险层 → Subagent 审查 → Master synthesis / portfolio-level judgment before deterministic risk and sizing → 组合构建 → 报告生成"
            )
        )

    def _section_run_context(self, analysis_meta: dict[str, Any], report_bundle: Any) -> str:
        model_role_metadata = analysis_meta.get("model_role_metadata")
        execution_trace = analysis_meta.get("execution_trace")
        what_if_plan = analysis_meta.get("what_if_plan")
        if not any([model_role_metadata, execution_trace, what_if_plan]) and report_bundle is not None:
            model_role_metadata = getattr(report_bundle, "model_role_metadata", None)
            execution_trace = getattr(report_bundle, "execution_trace", None)
            what_if_plan = getattr(report_bundle, "what_if_plan", None)
        if not any([model_role_metadata, execution_trace, what_if_plan]):
            return "## § 4 模型角色与执行轨迹\n\n_本次运行未记录结构化角色元数据或执行轨迹。_"

        from quant_investor.reporting.conclusion_renderer import ConclusionRenderer

        rendered = ConclusionRenderer.render_run_context(
            model_role_metadata,
            execution_trace,
            what_if_plan,
        )
        return "## § 4 模型角色与执行轨迹\n\n" + "\n".join(rendered).strip()

    def _section_bayesian_decision(self, analysis_meta: dict[str, Any], config: dict[str, Any]) -> str:
        """§ 4.5 Bayesian 决策层摘要（仅在 pipeline_mode=bayesian 时渲染）。"""
        pipeline_mode = analysis_meta.get("pipeline_mode", config.get("pipeline_mode", "legacy"))
        if pipeline_mode != "bayesian":
            return ""

        record_count = int(analysis_meta.get("bayesian_record_count", 0))
        funnel_candidates = int(analysis_meta.get("funnel_candidates_count", 0))
        funnel_excluded = int(analysis_meta.get("funnel_excluded_count", 0))
        shortlist_symbols = list(analysis_meta.get("bayesian_shortlist_symbols", []))

        lines = [
            "## § 4.5 Bayesian 决策层",
            "",
            f"- **漏斗压缩**: 全市场 → {funnel_candidates} 候选（排除 {funnel_excluded} 只）",
            f"- **后验排名**: {record_count} 只候选完成 Bayesian 后验计算",
            f"- **Master Discussion 入选**: {len(shortlist_symbols)} 只",
        ]
        if shortlist_symbols:
            lines.append(f"- **精选标的**: {', '.join(shortlist_symbols[:20])}")
        lines.append("")
        lines.append(
            "> Bayesian 后验 = 分层先验（市场/宏观/行业/交易性/数据质量）"
            " × 多分支似然（log-odds 更新）× 相关性折扣 × 降级惩罚 × 覆盖折扣"
        )
        return "\n".join(lines)

    def _section_subagent_decisions(self, branch_summary: dict) -> str:
        if not branch_summary:
            return "## § 5 Subagent 决策过程\n\n_本次运行未启用 Agent Layer 或数据不可用。_"

        branch_blocks = []
        for branch_key, blabel in _BRANCH_LABEL_MAP.items():
            b = branch_summary.get(branch_key)
            if not b:
                continue
            score = _safe_float(b.get("score", 0))
            conf = _safe_float(b.get("confidence", 0))
            conclusion = b.get("conclusion", "暂无结论。")
            support = b.get("support_drivers", [])
            drag = b.get("drag_drivers", [])
            risks = b.get("investment_risks", [])
            coverage = b.get("coverage_notes", [])

            sign = "+" if score >= 0 else ""
            support_text = "；".join(str(s) for s in support[:3]) if support else "暂无明显支撑项。"
            drag_text = "；".join(str(d) for d in drag[:3]) if drag else "暂无明显拖累项。"
            risks_text = "；".join(str(r) for r in risks[:3]) if risks else "无"
            coverage_text = "；".join(str(c) for c in coverage[:2]) if coverage else "覆盖完整。"

            branch_blocks.append(
                f"### {blabel}分支（Subagent 分析）\n\n"
                f"| 项目 | 值 |\n|------|----|\n"
                f"| 综合得分 | `{sign}{score:.4f}` |\n"
                f"| 可信度 | {_confidence_label(conf)}（{conf:.2f}） |\n"
                f"| 分支结论 | {conclusion} |\n"
                f"| 支撑因素 | {support_text} |\n"
                f"| 拖累因素 | {drag_text} |\n"
                f"| 投资风险提示 | {risks_text} |\n"
                f"| 数据覆盖说明 | {coverage_text} |"
            )

        body = "\n\n".join(branch_blocks) if branch_blocks else "_无分支数据。_"
        return f"## § 5 Subagent 决策过程、逻辑和依据\n\n{body}"

    def _section_master_decisions(
        self,
        executive_summary: list[str],
        market_view: str,
        portfolio_plan: dict,
        narrator_md: str,
    ) -> str:
        exec_block = ""
        if executive_summary:
            exec_block = "**IC 综合三句话结论：**\n\n" + "\n".join(
                f"> {i+1}. {line}" for i, line in enumerate(executive_summary)
            )

        mv_block = f"\n\n**市场综合判断：**\n\n> {market_view}" if market_view else ""

        plan_notes = portfolio_plan.get("execution_notes", [])
        notes_block = ""
        if plan_notes:
            notes_block = "\n\n**组合执行备注：**\n" + "\n".join(
                f"- {note}" for note in plan_notes
            )

        target_exp = _safe_float(portfolio_plan.get("target_exposure", 0))
        style = portfolio_plan.get("style_bias", "均衡")
        selected = int(portfolio_plan.get("selected_count", 0))
        planned = _safe_float(portfolio_plan.get("planned_investment", 0))
        cash = _safe_float(portfolio_plan.get("cash_reserve", 0))

        summary_block = (
            f"\n\n**Master Agent 组合决策：**\n\n"
            f"| 决策项 | 结果 |\n|--------|------|\n"
            f"| 建议总仓位 | {target_exp:.1%} |\n"
            f"| 组合风格 | {style} |\n"
            f"| 精选标的数 | {selected} 只 |\n"
            f"| 计划投入 | ¥{planned:,.0f} |\n"
            f"| 保留现金 | ¥{cash:,.0f} |"
        )

        narrator_section = ""
        if narrator_md:
            # 从 narrator 报告中提取关键段落（避免完整复制导致报告过长）
            lines = narrator_md.split("\n")
            relevant = []
            capture = False
            for line in lines:
                if any(kw in line for kw in ["## 市场", "## 执行", "## 组合", "## 宏观", "## 决策"]):
                    capture = True
                if capture:
                    relevant.append(line)
                if len(relevant) > 30:
                    break
            if relevant:
                narrator_section = (
                    "\n\n<details>\n<summary>📋 NarratorAgent 完整报告摘要（展开）</summary>\n\n"
                    + "\n".join(relevant[:30])
                    + "\n\n</details>"
                )

        body = (
            f"{exec_block}{mv_block}{summary_block}{notes_block}{narrator_section}"
        ).strip()

        return f"## § 6 Master Agent 决策过程、逻辑和依据\n\n{body}"

    def _section_investment_recommendations(
        self, recommendations: list[dict], market: str
    ) -> str:
        if not recommendations:
            return (
                "## § 7 最终投资建议\n\n"
                "_本次分析未产生满足买入条件的候选标的。_\n\n"
                "> 可能原因：市场整体偏弱、宏观压制、数据覆盖不足。建议维持观望，等待信号改善。"
            )

        rows = []
        for item in recommendations:
            symbol = item.get("symbol", "")
            name = str(item.get("company_name") or item.get("name") or "").strip()
            action = item.get("action", "观察")
            emoji = _ACTION_EMOJI.get(action, "⚪")
            conf = _safe_float(item.get("confidence", 0))
            cur_price = _safe_float(item.get("current_price", 0))
            entry_price = _safe_float(item.get("recommended_entry_price", cur_price))
            target_price = _safe_float(item.get("target_price", 0))
            stop_loss = _safe_float(item.get("stop_loss_price", 0))
            pos_count = int(item.get("branch_positive_count", 0))
            rank = item.get("rank", "-")

            support = item.get("support_drivers", [])
            drag = item.get("drag_drivers", [])
            support_str = "；".join(str(s) for s in support[:2]) if support else "-"
            drag_str = "；".join(str(d) for d in drag[:2]) if drag else "-"
            one_line = item.get("one_line_conclusion", "")

            rows.append(
                f"\n### {rank}. {emoji} {symbol} {name}  |  {action}\n\n"
                f"- **一句话结论**: {one_line or '详见驱动因素。'}\n"
                f"- **分支支持**: {pos_count}/5 路正向  |  可信度: {_confidence_label(conf)}（{conf:.2f}）\n"
                f"- **价格参数**: 现价 ¥{cur_price:.2f}  →  参考买点 ¥{entry_price:.2f}  |  目标价 ¥{target_price:.2f}  |  止损价 ¥{stop_loss:.2f}\n"
                f"- **支撑因素**: {support_str}\n"
                f"- **拖累/风险**: {drag_str}"
            )

        body = "\n".join(rows)
        return f"## § 7 最终投资建议\n\n共精选 **{len(recommendations)}** 只标的：\n{body}"

    def _section_positions_orders(
        self,
        recommendations: list[dict],
        portfolio_plan: dict,
        config: dict,
    ) -> str:
        total_capital = config["total_capital"]
        target_exp = _safe_float(portfolio_plan.get("target_exposure", 0))
        planned = _safe_float(portfolio_plan.get("planned_investment", 0))
        cash = _safe_float(portfolio_plan.get("cash_reserve", total_capital - planned))
        max_weight = _safe_float(portfolio_plan.get("max_single_weight", 0))

        header = (
            f"**资金分配概览：**\n\n"
            f"| 项目 | 金额 | 比例 |\n|------|------|------|\n"
            f"| 总资金 | ¥{total_capital:,.0f} | 100% |\n"
            f"| 计划投入 | ¥{planned:,.0f} | {target_exp:.1%} |\n"
            f"| 保留现金 | ¥{cash:,.0f} | {1-target_exp:.1%} |\n"
            f"| 单票上限 | - | {max_weight:.1%} |\n"
        )

        if not recommendations:
            return (
                f"## § 8 仓位和买卖指令\n\n{header}\n"
                "_当前无买入指令，建议全仓现金等待机会。_"
            )

        order_rows = []
        buy_orders = [r for r in recommendations if r.get("action") in ("买入", "轻仓试错")]
        watch_orders = [r for r in recommendations if r.get("action") not in ("买入", "轻仓试错")]

        if buy_orders:
            order_rows.append(
                "\n**📥 买入/建仓指令：**\n\n"
                "| 序 | 股票 | 操作 | 参考买点 | 数量(股) | 金额 | 权重 | 止损价 |\n"
                "|---|------|------|---------|---------|------|------|-------|\n"
            )
            for item in buy_orders:
                display_name = self._display_name(item)
                action = item.get("action", "观察")
                entry = _safe_float(item.get("recommended_entry_price") or item.get("current_price", 0))
                shares = int(item.get("portfolio_shares", 0))
                amount = _safe_float(item.get("portfolio_amount", 0))
                weight = _safe_float(item.get("portfolio_weight", 0))
                stop_loss = _safe_float(item.get("stop_loss_price", 0))
                rank = item.get("rank", "-")
                order_rows.append(
                    f"| {rank} | {display_name} | {action} | "
                    f"¥{entry:.2f} | {shares:,} | ¥{amount:,.0f} | {weight:.2%} | ¥{stop_loss:.2f} |\n"
                )

        if watch_orders:
            order_rows.append(
                "\n**👁️ 观察/持续跟踪（暂不执行）：**\n\n"
            )
            for item in watch_orders:
                display_name = self._display_name(item)
                cur_price = _safe_float(item.get("current_price", 0))
                target_price = _safe_float(item.get("target_price", 0))
                one_line = item.get("one_line_conclusion", "信号不足，继续观察。")
                order_rows.append(
                    f"- **{display_name}** — 现价 ¥{cur_price:.2f} | 目标 ¥{target_price:.2f} | {one_line}\n"
                )

        body = header + "".join(order_rows)
        return f"## § 8 仓位和买卖指令\n\n{body}"

    def _section_next_steps(
        self,
        history: list[dict],
        config: dict,
        recommendations: list[dict],
        timing: dict,
    ) -> str:
        from datetime import timedelta

        schedule_time = config.get("schedule_time", "17:30")
        next_run = f"下次分析时间: 明日 {schedule_time}"

        # 上次分析提示
        prev_note = ""
        if history:
            last = history[0]
            last_date = last.get("created_at", "")[:10]
            prev_note = f"- 上次分析: {last_date}（{last.get('status', '')}）"

        # 当前建议摘要
        buy_list = [
            f"`{self._display_name(r)}` ¥{_safe_float(r.get('recommended_entry_price') or r.get('current_price', 0)):.2f}"
            for r in recommendations
            if r.get("action") in ("买入", "轻仓试错")
        ]
        watch_list = [
            f"`{self._display_name(r)}`"
            for r in recommendations
            if r.get("action") not in ("买入", "轻仓试错")
        ]

        buy_text = "、".join(buy_list) if buy_list else "无"
        watch_text = "、".join(watch_list[:5]) if watch_list else "无"

        timing_note = ""
        total_secs = _safe_float(timing.get("total_seconds", 0))
        if total_secs > 0:
            timing_note = f"- 本次分析耗时: {total_secs/60:.1f} 分钟"

        return (
            f"## § 9 下一步计划\n\n"
            f"**执行待办：**\n"
            f"- 待建仓标的: {buy_text}\n"
            f"- 待观察标的: {watch_text}\n"
            f"- 所有买入订单请参考 § 7 的参考买点和止损价执行\n"
            f"- 建议分批建仓，单次不超过目标仓位的 50%\n\n"
            f"**数据与系统：**\n"
            f"- {next_run}\n"
            f"{prev_note}\n"
            f"{timing_note}\n"
            f"- 历史分析记录已保存至 `data/web_runs.db`，可在 Web 工作台查看\n\n"
            f"**风险提示：**\n"
            f"- 本报告为系统自动生成，仅供参考，不构成投资建议\n"
            f"- 请结合实际市场情况和个人风险承受能力做出决策\n"
            f"- 严格执行止损纪律，控制单次亏损"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. 持久化
# ══════════════════════════════════════════════════════════════════════════════

class PersistenceManager:
    """将报告保存到文件和 web_runs.db。"""

    def __init__(self) -> None:
        self._store: Any = None

    def _get_store(self) -> Any:
        if self._store is None:
            from web.services.run_history_store import RunHistoryStore
            self._store = RunHistoryStore()
            self._store.init_db()
        return self._store

    def save(
        self,
        report_md: str,
        pipeline_result: dict[str, Any],
        config: dict[str, Any],
    ) -> str:
        """保存报告到文件和数据库，返回 job_id。"""
        job_id = f"daily-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now().isoformat()

        # 1) 保存 Markdown 文件
        report_dir = ROOT / config["report_dir"]
        report_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_analysis.md"
        report_path = report_dir / filename
        report_path.write_text(report_md, encoding="utf-8")
        log.info("报告已保存: %s", report_path)

        def _as_mapping(value: Any) -> dict[str, Any]:
            if isinstance(value, dict):
                return dict(value)
            if hasattr(value, "to_dict"):
                try:
                    payload = value.to_dict()
                    if isinstance(payload, dict):
                        return dict(payload)
                except Exception:
                    return {}
            return {}

        def _compact_trace_summary(value: Any) -> dict[str, Any]:
            payload = _as_mapping(value)
            steps = []
            for step in payload.get("steps", [])[:8]:
                step_map = _as_mapping(step)
                steps.append(
                    {
                        "stage": step_map.get("stage", ""),
                        "role": step_map.get("role", ""),
                        "success": bool(step_map.get("success", False)),
                        "conclusion": str(step_map.get("conclusion", ""))[:180],
                    }
                )
            return {
                "model_roles": _as_mapping(payload.get("model_roles")),
                "key_parameters": {
                    "selected_count": payload.get("key_parameters", {}).get("selected_count", 0),
                    "target_exposure": payload.get("key_parameters", {}).get("target_exposure", 0.0),
                    "max_single_weight": payload.get("key_parameters", {}).get("max_single_weight", 0.0),
                    "data_quality_issue_count": payload.get("key_parameters", {}).get("data_quality_issue_count", 0),
                    "resolver_strategy": payload.get("key_parameters", {}).get("resolver_strategy", ""),
                },
                "final_deterministic_outcome": _as_mapping(payload.get("final_deterministic_outcome")),
                "steps": steps,
                "resolution_strategy": payload.get("resolution_strategy", ""),
                "local_union_fallback_used": bool(payload.get("local_union_fallback_used", False)),
            }

        def _compact_whatif_summary(value: Any) -> dict[str, Any]:
            payload = _as_mapping(value)
            scenarios = []
            for scenario in payload.get("scenarios", [])[:6]:
                scenario_map = _as_mapping(scenario)
                scenarios.append(
                    {
                        "scenario_name": scenario_map.get("scenario_name", ""),
                        "trigger": str(scenario_map.get("trigger", ""))[:120],
                        "action": str(scenario_map.get("action", ""))[:120],
                        "rerun_full_market_daily_path": bool(scenario_map.get("rerun_full_market_daily_path", False)),
                    }
                )
            return {
                "generated_by": payload.get("generated_by", ""),
                "scenario_count": len(payload.get("scenarios", []) or []),
                "scenarios": scenarios,
                "metadata": _as_mapping(payload.get("metadata")),
            }

        # 2) 构建摘要 JSON
        plan = pipeline_result.get("reports", {})
        report_bundle = plan.get("report_bundle")
        exec_summary = []
        if report_bundle and hasattr(report_bundle, "executive_summary"):
            exec_summary = list(getattr(report_bundle, "executive_summary", []))
        analysis_meta = pipeline_result.get("analysis_meta", {})
        execution_trace = _as_mapping(analysis_meta.get("execution_trace"))
        what_if_plan = _as_mapping(analysis_meta.get("what_if_plan"))

        analysis = pipeline_result.get("analysis", {})
        total_stocks = sum(
            sum(b.get("stock_count", 0) for b in batches)
            for batches in analysis.values()
        )
        timing = pipeline_result.get("timing", {})
        result_summary = {
            "total_stocks": total_stocks,
            "categories": pipeline_result.get("categories", []),
            "universe": pipeline_result.get("universe", config.get("universe", "full_a")),
            "timing_seconds": timing.get("total_seconds", 0),
            "executive_summary": exec_summary,
            "report_file": str(report_path),
            "model_role_metadata": _as_mapping(analysis_meta.get("model_role_metadata", {})),
            "branch_model": analysis_meta.get("branch_model", config.get("agent_model", "")),
            "agent_fallback_model": config.get("agent_fallback_model", ""),
            "master_model": analysis_meta.get("master_model", config.get("master_model", "")),
            "master_fallback_model": config.get("master_fallback_model", ""),
            "master_reasoning_effort": analysis_meta.get(
                "master_reasoning_effort",
                config.get("master_reasoning_effort", ""),
            ),
            "agent_layer_enabled": analysis_meta.get(
                "agent_layer_enabled",
                config.get("enable_agent_layer", False),
            ),
            "selected_count": execution_trace.get("final_deterministic_outcome", {}).get("selected_count", 0) if isinstance(execution_trace, dict) else 0,
            "target_exposure": execution_trace.get("final_deterministic_outcome", {}).get("target_exposure", 0.0) if isinstance(execution_trace, dict) else 0.0,
            "data_quality_issue_count": analysis_meta.get("data_quality_issue_count", 0),
            "researchable_count": analysis_meta.get("researchable_count", 0),
            "quarantined_count": analysis_meta.get("quarantined_count", 0),
            "evidence_pack_token_count": analysis_meta.get("evidence_pack_token_count", 0),
            "evidence_pack_field_limit": analysis_meta.get("evidence_pack_field_limit", 0),
            "evidence_pack_shortlist_limit": analysis_meta.get("evidence_pack_shortlist_limit", 0),
            "model_role_resolution": config.get("model_role_resolution", {}),
            # Bayesian pipeline info
            "pipeline_mode": analysis_meta.get("pipeline_mode", config.get("pipeline_mode", "legacy")),
            "bayesian_shortlist_symbols": analysis_meta.get("bayesian_shortlist_symbols", []),
            "bayesian_record_count": analysis_meta.get("bayesian_record_count", 0),
            "funnel_candidates_count": analysis_meta.get("funnel_candidates_count", 0),
            "funnel_excluded_count": analysis_meta.get("funnel_excluded_count", 0),
        }
        trace_summary = _compact_trace_summary(execution_trace)
        whatif_summary = _compact_whatif_summary(what_if_plan)

        # 3) 存入 web_runs.db
        try:
            store = self._get_store()
            store.save_run(
                job_id=job_id,
                created_at=now_iso,
                status="completed",
                request_json=json.dumps(
                    {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))},
                    ensure_ascii=False,
                ),
                report_markdown=report_md,
                report_path=str(report_path),
                result_summary_json=json.dumps(result_summary, ensure_ascii=False),
                trace_summary_json=json.dumps(trace_summary, ensure_ascii=False),
                whatif_summary_json=json.dumps(whatif_summary, ensure_ascii=False),
                total_time=_safe_float(timing.get("total_seconds", 0)),
                market=config["market"],
                stock_pool=json.dumps([]),
                risk_level=config["risk_level"],
            )
            log.info("分析记录已写入数据库: job_id=%s", job_id)
        except Exception as exc:
            log.warning("写入数据库失败（报告文件已保存）: %s", exc)

        return job_id


# ══════════════════════════════════════════════════════════════════════════════
# 7. 主流程函数
# ══════════════════════════════════════════════════════════════════════════════

def run_once(
    config: dict[str, Any],
    skip_download: bool = False,
    skip_stage1: bool = False,
) -> str:
    """执行一次完整分析，返回报告路径。"""
    if skip_stage1:
        config = {**config, "skip_stage1": True}
    if skip_download:
        config = {**config, "skip_download": True}

    history_loader = HistoryLoader()
    runner = AnalysisRunner()
    builder = ReportBuilder()
    persist = PersistenceManager()

    history = history_loader.load_recent(config["history_lookback"], config["market"])
    log.info("已加载 %d 条历史记录", len(history))

    pipeline_result = runner.run(config)

    log.info("生成决策报告...")
    report_md = builder.build(pipeline_result, config, history)

    job_id = persist.save(report_md, pipeline_result, config)

    # 打印报告到控制台
    print("\n" + "=" * 80)
    print(report_md)
    print("=" * 80)

    log.info("分析完成，job_id=%s", job_id)
    return job_id


def run_daemon(config: dict[str, Any]) -> None:
    """守护模式：启动后端 + 每天定时运行分析。"""
    try:
        schedule_dt = datetime.strptime(config["schedule_time"], "%H:%M")
        schedule_time = schedule_dt.time()
    except ValueError:
        log.error("无效的 schedule_time 格式: %s（应为 HH:MM）", config["schedule_time"])
        sys.exit(1)

    backend_mgr = BackendManager(config["backend_host"], config["backend_port"])
    backend_proc: Optional[subprocess.Popen] = None

    def _shutdown(signum: int, frame: Any) -> None:
        log.info("收到信号 %s，正在退出...", signum)
        backend_mgr.stop(backend_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("守护模式启动 | 每日 %s 自动分析", config["schedule_time"])

    # 启动后端
    try:
        backend_proc = backend_mgr.start()
    except Exception as exc:
        log.error("后端启动失败: %s", exc)
        log.info("继续运行（无 Web 后端，仅执行分析）")

    last_run_date: Optional[date] = None

    while True:
        # 守护后端
        try:
            backend_proc = backend_mgr.ensure_running(backend_proc)
        except Exception as exc:
            log.warning("后端守护失败: %s", exc)

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


def run_backend_only(config: dict[str, Any]) -> None:
    """仅启动并守护后端进程。"""
    backend_mgr = BackendManager(config["backend_host"], config["backend_port"])
    backend_proc: Optional[subprocess.Popen] = None

    def _shutdown(signum: int, frame: Any) -> None:
        log.info("收到信号，正在停止后端...")
        backend_mgr.stop(backend_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        backend_proc = backend_mgr.start()
    except Exception as exc:
        log.error("后端启动失败: %s", exc)
        sys.exit(1)

    log.info(
        "后端运行中: http://%s:%s （Ctrl-C 停止）",
        config["backend_host"],
        config["backend_port"],
    )

    while True:
        try:
            backend_proc = backend_mgr.ensure_running(backend_proc)
        except Exception as exc:
            log.warning("后端守护异常: %s", exc)
        time.sleep(30)


def print_last_report(config: dict[str, Any]) -> None:
    """打印最后一次分析报告。"""
    loader = HistoryLoader()
    report = loader.load_last_report(config["market"])
    if not report:
        print("暂无历史报告。请先运行 python daily_runner.py 执行一次分析。")
        return
    print(report)


def dry_run(config: dict[str, Any]) -> None:
    """验证配置和连接，不实际运行分析。"""
    print("=== DRY RUN 模式 ===\n")

    print("✓ 配置加载成功")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n检查后端连通性...")
    backend_mgr = BackendManager(config["backend_host"], config["backend_port"])
    if backend_mgr.is_healthy():
        print(f"✓ 后端已运行: http://{config['backend_host']}:{config['backend_port']}")
    else:
        print(f"✗ 后端未运行（启动后端: python daily_runner.py --backend-only）")

    print("\n检查数据库...")
    try:
        loader = HistoryLoader()
        runs = loader.load_recent(3, config["market"])
        print(f"✓ 数据库连接成功，历史记录 {len(runs)} 条")
    except Exception as exc:
        print(f"✗ 数据库连接失败: {exc}")

    print("\n检查 Python 环境...")
    try:
        from quant_investor.market.run_pipeline import run_unified_pipeline  # noqa: F401
        print("✓ quant_investor 包可导入")
    except ImportError as exc:
        print(f"✗ quant_investor 导入失败: {exc}")

    pipeline_mode = config.get("pipeline_mode", "legacy")
    print(f"\n决策引擎: {pipeline_mode}")
    if pipeline_mode == "bayesian":
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
        help="守护模式：启动后端 + 每天定时分析",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="仅启动并守护后端",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="打印上次分析报告",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="验证配置和连接，不实际运行",
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
    parser.add_argument(
        "--pipeline-mode",
        choices=["bayesian", "legacy"],
        default="",
        help="决策引擎模式: bayesian（7层）或 legacy（3层）",
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
    if args.pipeline_mode:
        config["pipeline_mode"] = args.pipeline_mode
    if args.skip_stage1 or args.skip_data_check:
        config["skip_stage1"] = True

    # 分支执行
    if args.dry_run:
        dry_run(config)
    elif args.backend_only:
        run_backend_only(config)
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
