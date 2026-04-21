"""
单一主线 CLI 入口。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_investor.config import config
from quant_investor.pipeline import QuantInvestor
from quant_investor.research_run_config import ResearchRunConfig, ResolvedReviewModels


def run_download(**kwargs):
    from quant_investor.market.download import run_download as _run_download

    return _run_download(**kwargs)


def run_market_maintenance(**kwargs):
    from quant_investor.market.download import (
        run_market_maintenance as _run_market_maintenance,
    )

    return _run_market_maintenance(**kwargs)


def run_market_analysis(**kwargs):
    from quant_investor.market.analyze import (
        run_market_analysis as _run_market_analysis,
    )

    return _run_market_analysis(**kwargs)


def run_market_pipeline(**kwargs):
    from quant_investor.market.run_pipeline import (
        run_unified_pipeline as _run_unified_pipeline,
    )

    return _run_unified_pipeline(**kwargs)


def run_market_backtest(**kwargs):
    from quant_investor.market import (
        run_market_backtest as _run_market_backtest,
    )

    return _run_market_backtest(**kwargs)


def run_web_api(
    *,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    import uvicorn

    from web.config import API_HOST, API_PORT

    web_dir = Path(__file__).resolve().parents[2] / "web"
    uvicorn.run(
        "web.main:app",
        host=host or API_HOST,
        port=port or API_PORT,
        reload=reload,
        reload_dirs=[str(web_dir)] if reload else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor",
        description=(
            "Quant-Investor 单一主线 CLI。"
            "启动研究工作台 Web 服务请使用 `quant-investor web`。"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    research_parser = subparsers.add_parser("research", help="执行单一主线研究")
    research_subparsers = research_parser.add_subparsers(
        dest="research_command",
        required=True,
    )
    research_run = research_subparsers.add_parser("run", help="执行当前主线")
    research_run.add_argument("--stocks", nargs="+", required=True)
    research_run.add_argument("--market", default="CN", choices=["CN", "US"])
    research_run.add_argument("--capital", type=float, default=1_000_000.0)
    research_run.add_argument(
        "--risk",
        default="中等",
        choices=["保守", "中等", "积极"],
    )
    research_run.add_argument("--lookback", type=float, default=1.0)
    research_run.add_argument(
        "--kline-backend",
        default="hybrid",
        choices=["heuristic", "kronos", "chronos", "hybrid"],
        help="兼容保留参数；当前主线统一使用 kline backend 配置。",
    )
    research_run.add_argument("--no-macro", action="store_true")
    research_run.add_argument("--no-kline", "--no-kronos", action="store_true")
    research_run.add_argument("--no-quant", action="store_true")
    research_run.add_argument("--no-fundamental", action="store_true")
    research_run.add_argument("--no-intelligence", action="store_true")
    research_run.add_argument(
        "--disable-document-semantics",
        action="store_true",
    )
    research_run.add_argument(
        "--allow-synthetic-for-research",
        action="store_true",
    )
    research_run.add_argument("--output", default="")
    research_run.add_argument(
        "--no-agent-layer",
        action="store_true",
        help="关闭当前主线的 review layer",
    )
    research_run.add_argument(
        "--review-model",
        action="append",
        dest="review_model_priority",
        default=[],
        help="按传入顺序覆盖默认 review 模型优先级，可重复传入",
    )
    research_run.add_argument("--agent-model", default="")
    research_run.add_argument("--agent-fallback-model", default="")
    research_run.add_argument("--master-model", default="")
    research_run.add_argument("--master-fallback-model", default="")
    research_run.add_argument(
        "--agent-timeout",
        type=float,
        default=config.DEFAULT_AGENT_TIMEOUT_SECONDS,
        help="单个 agent 超时（秒）",
    )
    research_run.add_argument(
        "--master-timeout",
        type=float,
        default=config.DEFAULT_MASTER_TIMEOUT_SECONDS,
        help="主协调 agent 超时（秒）",
    )
    research_run.add_argument(
        "--master-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="high",
        help="Master Agent reasoning 强度",
    )
    research_run.add_argument(
        "--funnel-profile",
        default=config.FUNNEL_PROFILE,
        choices=["classic", "momentum_leader"],
        help="候选漏斗配方",
    )
    research_run.add_argument(
        "--max-candidates",
        type=int,
        default=config.FUNNEL_MAX_CANDIDATES,
        help="进入候选研究阶段的最大标的数",
    )
    research_run.add_argument(
        "--trend-windows",
        type=int,
        nargs="+",
        default=list(config.FUNNEL_TREND_WINDOWS),
        help="动量窗口（日），例如 20 60 120",
    )
    research_run.add_argument(
        "--volume-spike-threshold",
        type=float,
        default=config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
        help="放量确认阈值",
    )
    research_run.add_argument(
        "--breakout-distance-pct",
        type=float,
        default=config.FUNNEL_BREAKOUT_DISTANCE_PCT,
        help="距阶段高点的最大距离",
    )

    market_parser = subparsers.add_parser("market", help="全市场工作流")
    market_subparsers = market_parser.add_subparsers(
        dest="market_command",
        required=True,
    )

    market_maintain = market_subparsers.add_parser(
        "maintain",
        help="维护全市场本地数据到最新可得交易日",
    )
    market_maintain.add_argument(
        "--market",
        required=True,
        choices=["CN", "US"],
    )
    market_maintain.add_argument(
        "--category",
        action="append",
        dest="categories",
    )
    market_maintain.add_argument("--years", type=int, default=3)
    market_maintain.add_argument("--workers", type=int, default=4)
    market_maintain.add_argument("--batch-size", type=int, default=50)
    market_maintain.add_argument("--max-rounds", type=int, default=1)
    market_maintain.add_argument("--fail-on-incomplete", action="store_true")
    market_maintain.add_argument("--allowed-stale-symbols", nargs="*")

    market_analyze = market_subparsers.add_parser(
        "analyze",
        help="分析全市场",
    )
    market_analyze.add_argument(
        "--market",
        required=True,
        choices=["CN", "US"],
    )
    market_analyze.add_argument(
        "--mode",
        default="batch",
        choices=["sample", "batch"],
    )
    market_analyze.add_argument(
        "--category",
        action="append",
        dest="categories",
    )
    market_analyze.add_argument("--batch-size", type=int, default=None)
    market_analyze.add_argument("--capital", type=float, default=1_000_000)
    market_analyze.add_argument("--top-k", type=int, default=12)
    market_analyze.add_argument("--no-agent-layer", action="store_true")
    market_analyze.add_argument(
        "--review-model",
        action="append",
        dest="review_model_priority",
        default=[],
    )
    market_analyze.add_argument("--agent-model", default="")
    market_analyze.add_argument("--agent-fallback-model", default="")
    market_analyze.add_argument("--master-model", default="")
    market_analyze.add_argument("--master-fallback-model", default="")
    market_analyze.add_argument(
        "--master-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="high",
    )
    market_analyze.add_argument("--agent-timeout", type=float, default=config.DEFAULT_AGENT_TIMEOUT_SECONDS)
    market_analyze.add_argument("--master-timeout", type=float, default=config.DEFAULT_MASTER_TIMEOUT_SECONDS)
    market_analyze.add_argument("--funnel-profile", default=config.FUNNEL_PROFILE, choices=["classic", "momentum_leader"])
    market_analyze.add_argument("--max-candidates", type=int, default=config.FUNNEL_MAX_CANDIDATES)
    market_analyze.add_argument("--trend-windows", type=int, nargs="+", default=list(config.FUNNEL_TREND_WINDOWS))
    market_analyze.add_argument("--volume-spike-threshold", type=float, default=config.FUNNEL_VOLUME_SPIKE_THRESHOLD)
    market_analyze.add_argument("--breakout-distance-pct", type=float, default=config.FUNNEL_BREAKOUT_DISTANCE_PCT)

    market_run = market_subparsers.add_parser(
        "run",
        help="完整执行全市场 daily pipeline",
    )
    market_run.add_argument(
        "--market",
        required=True,
        choices=["CN", "US"],
    )
    market_run.add_argument(
        "--mode",
        default="batch",
        choices=["sample", "batch"],
    )
    market_run.add_argument(
        "--category",
        action="append",
        dest="categories",
    )
    market_run.add_argument("--batch-size", type=int, default=None)
    market_run.add_argument("--capital", type=float, default=1_000_000)
    market_run.add_argument("--top-k", type=int, default=12)
    market_run.add_argument("--skip-download", action="store_true")
    market_run.add_argument("--years", type=int, default=3)
    market_run.add_argument("--workers", type=int, default=4)
    market_run.add_argument("--max-download-rounds", type=int, default=2)
    market_run.add_argument("--no-agent-layer", action="store_true")
    market_run.add_argument(
        "--review-model",
        action="append",
        dest="review_model_priority",
        default=[],
    )
    market_run.add_argument("--agent-model", default="")
    market_run.add_argument("--agent-fallback-model", default="")
    market_run.add_argument("--master-model", default="")
    market_run.add_argument("--master-fallback-model", default="")
    market_run.add_argument(
        "--master-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="high",
    )
    market_run.add_argument("--agent-timeout", type=float, default=config.DEFAULT_AGENT_TIMEOUT_SECONDS)
    market_run.add_argument("--master-timeout", type=float, default=config.DEFAULT_MASTER_TIMEOUT_SECONDS)
    market_run.add_argument("--funnel-profile", default=config.FUNNEL_PROFILE, choices=["classic", "momentum_leader"])
    market_run.add_argument("--max-candidates", type=int, default=config.FUNNEL_MAX_CANDIDATES)
    market_run.add_argument("--trend-windows", type=int, nargs="+", default=list(config.FUNNEL_TREND_WINDOWS))
    market_run.add_argument("--volume-spike-threshold", type=float, default=config.FUNNEL_VOLUME_SPIKE_THRESHOLD)
    market_run.add_argument("--breakout-distance-pct", type=float, default=config.FUNNEL_BREAKOUT_DISTANCE_PCT)

    market_backtest = market_subparsers.add_parser(
        "backtest",
        help="回测本地全市场数据",
    )
    market_backtest.add_argument(
        "--market",
        required=True,
        choices=["CN", "US"],
    )
    market_backtest.add_argument(
        "--category",
        action="append",
        dest="categories",
    )
    market_backtest.add_argument("--sample-size", type=int, default=None)
    market_backtest.add_argument("--capital", type=float, default=1_000_000)
    market_backtest.add_argument("--n-holdings", type=int, default=10)
    market_backtest.add_argument("--rebalance", default="W")

    web_parser = subparsers.add_parser(
        "web",
        help="启动研究工作台 Web 服务（/api + workspace）",
    )
    web_parser.add_argument("--host", default=None)
    web_parser.add_argument("--port", type=int, default=None)
    web_parser.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    review_models = ResolvedReviewModels.from_mapping(vars(args))

    if args.command == "research" and args.research_command == "run":
        run_config = ResearchRunConfig.from_mapping(vars(args))
        investor = QuantInvestor(**run_config.to_quant_investor_kwargs(verbose=True))
        investor.run()
        if args.output:
            investor.save_report(args.output)
        else:
            investor.print_report()
        return

    if args.command == "market" and args.market_command == "maintain":
        run_market_maintenance(
            market=args.market,
            categories=args.categories,
            years=args.years,
            max_workers=args.workers,
            batch_size=args.batch_size,
            max_rounds=args.max_rounds,
            fail_on_incomplete=args.fail_on_incomplete,
            allowed_stale_symbols=args.allowed_stale_symbols,
        )
        return

    if args.command == "market" and args.market_command == "analyze":
        run_market_analysis(
            market=args.market,
            mode=args.mode,
            categories=args.categories,
            batch_size=args.batch_size,
            total_capital=args.capital,
            top_k=args.top_k,
            enable_agent_layer=not args.no_agent_layer,
            funnel_profile=args.funnel_profile,
            max_candidates=args.max_candidates,
            trend_windows=args.trend_windows,
            volume_spike_threshold=args.volume_spike_threshold,
            breakout_distance_pct=args.breakout_distance_pct,
            **review_models.to_runtime_kwargs(),
        )
        return

    if args.command == "market" and args.market_command == "run":
        run_market_pipeline(
            market=args.market,
            mode=args.mode,
            categories=args.categories,
            batch_size=args.batch_size,
            total_capital=args.capital,
            top_k=args.top_k,
            skip_download=args.skip_download,
            force_download=False,
            years=args.years,
            workers=args.workers,
            max_download_rounds=args.max_download_rounds,
            enable_agent_layer=not args.no_agent_layer,
            funnel_profile=args.funnel_profile,
            max_candidates=args.max_candidates,
            trend_windows=args.trend_windows,
            volume_spike_threshold=args.volume_spike_threshold,
            breakout_distance_pct=args.breakout_distance_pct,
            **review_models.to_runtime_kwargs(),
        )
        return

    if args.command == "market" and args.market_command == "backtest":
        run_market_backtest(
            market=args.market,
            categories=args.categories,
            sample_size=args.sample_size,
            initial_capital=args.capital,
            n_holdings=args.n_holdings,
            rebalance_freq=args.rebalance,
        )
        return

    if args.command == "web":
        run_web_api(host=args.host, port=args.port, reload=args.reload)
        return
