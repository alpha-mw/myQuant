"""
统一 CLI 入口。
"""

from __future__ import annotations

import argparse

from quant_investor.pipeline import QuantInvestorLatest, QuantInvestorV8, QuantInvestorV9, QuantInvestorV10


def run_download(**kwargs):
    from quant_investor.market.download import run_download as _run_download

    return _run_download(**kwargs)


def run_market_analysis(**kwargs):
    from quant_investor.market.analyze import run_market_analysis as _run_market_analysis

    return _run_market_analysis(**kwargs)


def run_market_backtest(**kwargs):
    from quant_investor.market import run_market_backtest as _run_market_backtest

    return _run_market_backtest(**kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor",
        description="Quant-Investor 统一 CLI。V8 为 legacy frozen，V9 为 current architecture，默认 latest=V9。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    research_parser = subparsers.add_parser("research", help="五路并行研究")
    research_subparsers = research_parser.add_subparsers(dest="research_command", required=True)
    research_run = research_subparsers.add_parser("run", help="执行研究")
    research_run.add_argument("--stocks", nargs="+", required=True)
    research_run.add_argument(
        "--architecture",
        default="latest",
        choices=["latest", "v10", "v9", "v8"],
        help="选择研究架构：`v8`=legacy frozen，`v9`=current，`v10`=multi-agent IC，`latest` 默认等于 V10。",
    )
    research_run.add_argument("--market", default="CN", choices=["CN", "US"])
    research_run.add_argument("--capital", type=float, default=1_000_000.0)
    research_run.add_argument("--risk", default="中等", choices=["保守", "中等", "积极"])
    research_run.add_argument("--lookback", type=float, default=1.0)
    research_run.add_argument(
        "--kline-backend",
        default="hybrid",
        choices=["heuristic", "kronos", "chronos", "hybrid"],
        help="兼容保留参数；当前运行时统一使用 hybrid（Kronos+Chronos+evaluator）主链",
    )
    research_run.add_argument("--no-macro", action="store_true")
    research_run.add_argument("--no-kline", "--no-kronos", action="store_true")
    research_run.add_argument("--no-quant", action="store_true")
    research_run.add_argument("--no-fundamental", action="store_true")
    research_run.add_argument("--no-intelligence", action="store_true")
    research_run.add_argument("--no-branch-debate", action="store_true")
    research_run.add_argument("--no-llm-debate", action="store_true")
    research_run.add_argument("--debate-top-k", type=int, default=3)
    research_run.add_argument("--debate-min-abs-score", type=float, default=0.08)
    research_run.add_argument("--debate-timeout-sec", type=float, default=8.0)
    research_run.add_argument("--debate-model", default="gpt-5.4-mini")
    research_run.add_argument("--disable-document-semantics", action="store_true")
    research_run.add_argument("--allow-synthetic-for-research", action="store_true")
    research_run.add_argument("--output", default="")
    # V10 agent layer params
    research_run.add_argument("--no-agent-layer", action="store_true", help="关闭 V10 Agent IC 层")
    research_run.add_argument("--enable-branch-debate", action="store_true", help="V10 下同时启用 branch-local debate")
    research_run.add_argument("--agent-model", default="", help="分支 SubAgent 使用的 LLM 模型")
    research_run.add_argument("--master-model", default="", help="IC Master Agent 使用的 LLM 模型")
    research_run.add_argument("--agent-timeout", type=float, default=15.0, help="单个 agent 超时（秒）")
    research_run.add_argument("--master-timeout", type=float, default=30.0, help="IC agent 超时（秒）")

    market_parser = subparsers.add_parser("market", help="全市场工作流")
    market_subparsers = market_parser.add_subparsers(dest="market_command", required=True)

    market_download = market_subparsers.add_parser("download", help="下载全市场数据")
    market_download.add_argument("--market", required=True, choices=["CN", "US"])
    market_download.add_argument("--category", action="append", dest="categories")
    market_download.add_argument("--years", type=int, default=3)
    market_download.add_argument("--workers", type=int, default=4)
    market_download.add_argument("--batch-size", type=int, default=50)
    market_download.add_argument("--check-complete", action="store_true")
    market_download.add_argument("--max-rounds", type=int, default=1)
    market_download.add_argument("--fail-on-incomplete", action="store_true")
    market_download.add_argument("--allowed-stale-symbols", nargs="*")

    market_analyze = market_subparsers.add_parser("analyze", help="分析全市场")
    market_analyze.add_argument("--market", required=True, choices=["CN", "US"])
    market_analyze.add_argument("--mode", default="batch", choices=["sample", "batch"])
    market_analyze.add_argument("--category", action="append", dest="categories")
    market_analyze.add_argument("--batch-size", type=int, default=None)
    market_analyze.add_argument("--capital", type=float, default=1_000_000)
    market_analyze.add_argument("--top-k", type=int, default=12)

    market_backtest = market_subparsers.add_parser("backtest", help="回测本地全市场数据")
    market_backtest.add_argument("--market", required=True, choices=["CN", "US"])
    market_backtest.add_argument("--category", action="append", dest="categories")
    market_backtest.add_argument("--sample-size", type=int, default=None)
    market_backtest.add_argument("--capital", type=float, default=1_000_000)
    market_backtest.add_argument("--n-holdings", type=int, default=10)
    market_backtest.add_argument("--rebalance", default="W")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "research" and args.research_command == "run":
        architecture_cls = {
            "latest": QuantInvestorLatest,
            "v10": QuantInvestorV10,
            "v9": QuantInvestorV9,
            "v8": QuantInvestorV8,
        }[args.architecture]
        investor_kwargs = dict(
            stock_pool=args.stocks,
            market=args.market,
            lookback_years=args.lookback,
            total_capital=args.capital,
            risk_level=args.risk,
            enable_macro=not args.no_macro,
            enable_kline=not args.no_kline,
            kline_backend=args.kline_backend,
            enable_intelligence=not args.no_intelligence,
            allow_synthetic_for_research=args.allow_synthetic_for_research,
            verbose=True,
        )
        if args.architecture == "v8":
            investor_kwargs["enable_llm_debate"] = not args.no_llm_debate
        else:
            investor_kwargs.update(
                {
                    "enable_quant": not args.no_quant,
                    "enable_fundamental": not args.no_fundamental,
                    "enable_branch_debate": not (args.no_branch_debate or args.no_llm_debate),
                    "debate_top_k": args.debate_top_k,
                    "debate_min_abs_score": args.debate_min_abs_score,
                    "debate_timeout_sec": args.debate_timeout_sec,
                    "debate_model": args.debate_model,
                    "enable_document_semantics": not args.disable_document_semantics,
                }
            )
            # V10 agent layer params
            if args.architecture in ("v10", "latest"):
                investor_kwargs.update(
                    {
                        "enable_agent_layer": not args.no_agent_layer,
                        "enable_branch_debate": args.enable_branch_debate,
                        "agent_model": args.agent_model,
                        "master_model": args.master_model,
                        "agent_timeout": args.agent_timeout,
                        "master_timeout": args.master_timeout,
                    }
                )
        investor = architecture_cls(**investor_kwargs)
        result = investor.run()
        if args.output:
            investor.save_report(args.output)
        else:
            investor.print_report()
        return

    if args.command == "market" and args.market_command == "download":
        run_download(
            market=args.market,
            categories=args.categories,
            years=args.years,
            max_workers=args.workers,
            batch_size=args.batch_size,
            check_complete=args.check_complete,
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
