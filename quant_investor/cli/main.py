"""
单一主线 CLI 入口。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


def run_fundamental_maintenance(**kwargs):
    from quant_investor.market.fundamental_mart import (
        run_cn_fundamental_maintenance as _run_cn_fundamental_maintenance,
    )

    return _run_cn_fundamental_maintenance(**kwargs)


def run_macro_maintenance(**kwargs):
    from quant_investor.market.macro_mart import (
        run_cn_macro_maintenance as _run_cn_macro_maintenance,
    )

    return _run_cn_macro_maintenance(**kwargs)


def run_macro_analysis(**kwargs):
    from quant_investor.macro.observer import (
        build_macro_observer,
        load_macro_observation_generation,
    )

    observations, generation = load_macro_observation_generation(kwargs.pop("observations_path"))
    return build_macro_observer(
        observations,
        enabled=True,
        kill_switch=False,
        persist=True,
        generation_provenance=generation,
        **kwargs,
    )


def run_macro_observation_maintenance(**kwargs):
    from quant_investor.macro.providers import maintain_macro_observations

    return maintain_macro_observations(**kwargs)


def run_macro_replay(**kwargs):
    from quant_investor.macro.replay import run_macro_replay as _run_macro_replay

    return _run_macro_replay(**kwargs)


def run_macro_tushare_normalization(**kwargs):
    from quant_investor.macro.tushare_normalizer import normalize_tushare_bundle_file

    return normalize_tushare_bundle_file(**kwargs)


def run_macro_backfill_publish(**kwargs):
    from quant_investor.macro.tushare_normalizer import publish_tushare_normalization

    return publish_tushare_normalization(**kwargs)


def run_macro_forward_observation(**kwargs):
    from quant_investor.macro.forward import record_macro_forward_observation

    return record_macro_forward_observation(**kwargs)


def run_macro_coverage(**kwargs):
    from quant_investor.macro.coverage import run_macro_coverage_audit

    return run_macro_coverage_audit(**kwargs)


def run_macro_acquisition(**kwargs):
    from quant_investor.macro.acquisition import run_macro_acquisition_plan

    return run_macro_acquisition_plan(**kwargs)


def run_data_governance(**kwargs):
    from quant_investor.market.data_governance import (
        run_data_governance as _run_data_governance,
    )

    return _run_data_governance(**kwargs)


def run_storage_validate(**kwargs):
    from quant_investor.market.market_data_store import (
        run_storage_validate as _run_storage_validate,
    )

    return _run_storage_validate(**kwargs)


def run_storage_validate_clean(**kwargs):
    from quant_investor.market.market_data_store import (
        run_storage_validate_clean as _run_storage_validate_clean,
    )

    return _run_storage_validate_clean(**kwargs)


def run_materialize_serving(**kwargs):
    from quant_investor.market.market_data_store import (
        run_materialize_serving as _run_materialize_serving,
    )

    return _run_materialize_serving(**kwargs)


def run_materialize_features(**kwargs):
    from quant_investor.market.market_data_store import (
        run_materialize_features as _run_materialize_features,
    )

    return _run_materialize_features(**kwargs)


def run_storage_diff(**kwargs):
    from quant_investor.market.market_data_store import (
        run_storage_diff as _run_storage_diff,
    )

    return _run_storage_diff(**kwargs)


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _parse_boolish(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def run_web_api(
    *,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    import uvicorn

    from web.config import API_HOST, API_PORT, warn_if_insecure_binding

    web_dir = Path(__file__).resolve().parents[2] / "web"
    warn_if_insecure_binding(host or API_HOST)
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
        default="v13-retired",
        choices=["v13-retired", "heuristic", "kronos", "chronos", "hybrid"],
        help="兼容保留参数；v13 四分支主线不再执行 kline 分支。",
    )
    research_run.add_argument("--no-macro", action="store_true")
    research_run.add_argument("--no-kline", "--no-kronos", action="store_true", help="兼容保留参数；v13 默认已禁用 kline。")
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
    market_maintain.add_argument("--batch-size", type=int, default=None)
    market_maintain.add_argument("--max-rounds", type=int, default=1)
    market_maintain.add_argument("--fail-on-incomplete", nargs="?", const=True, default=False, type=_parse_boolish)
    market_maintain.add_argument("--allowed-stale-symbols", nargs="*")
    market_maintain.add_argument("--staged", action="store_true")
    market_maintain.add_argument("--resume", action="store_true")
    market_maintain.add_argument("--max-batches-per-run", type=int, default=1)
    market_maintain.add_argument("--min-symbol-success-rate", type=float, default=0.95)
    market_maintain.add_argument("--target-date", default="auto")
    market_maintain.add_argument("--daily-window", action="store_true")
    market_maintain.add_argument(
        "--storage-mode",
        choices=["auto", "legacy", "parquet-direct"],
        default="auto",
        help=(
            "CN 日更存储路径；auto 对 CN 非 staged 解析为 parquet-direct，"
            "staged 仍使用受控批次状态机。legacy 仅保留非 CN 兼容路径。"
        ),
    )

    market_download = market_subparsers.add_parser(
        "download",
        help="兼容别名：维护全市场本地数据到最新可得交易日",
    )
    market_download.add_argument(
        "--market",
        required=True,
        choices=["CN", "US"],
    )
    market_download.add_argument(
        "--category",
        action="append",
        dest="categories",
    )
    market_download.add_argument("--years", type=int, default=3)
    market_download.add_argument("--workers", type=int, default=4)
    market_download.add_argument("--batch-size", type=int, default=50)
    market_download.add_argument("--max-rounds", type=int, default=1)
    market_download.add_argument("--fail-on-incomplete", action="store_true")
    market_download.add_argument("--allowed-stale-symbols", nargs="*")

    market_fundamental = market_subparsers.add_parser(
        "fundamental-maintain",
        help="维护独立 CN PIT fundamental mart，不影响日行情 maintain",
    )
    market_fundamental.add_argument("--market", required=True, choices=["CN"])
    market_fundamental.add_argument(
        "--universes",
        default="hs300,zz500,zz1000",
        help="逗号分隔的 universe 列表，默认 hs300,zz500,zz1000",
    )
    market_fundamental.add_argument("--years", type=int, default=5)
    market_fundamental.add_argument("--as-of", default="")
    market_fundamental.add_argument("--workers", type=int, default=4)
    market_fundamental.add_argument("--raw-input-dir", default="")
    market_fundamental.add_argument("--data-root", default="data/parquet/cn")
    market_fundamental.add_argument(
        "--snapshot-root",
        default="data/cn_market_full/_snapshots/fundamental",
    )
    market_fundamental.add_argument("--reports-root", default="reports/fundamental_readiness")
    market_fundamental.add_argument(
        "--allow-live",
        action="store_true",
        help="显式允许调用 live provider；本地测试默认不使用",
    )

    market_macro_maintain = market_subparsers.add_parser(
        "macro-maintain",
        help="维护 CN macro 兼容 mart；默认离线且无输入不写入",
    )
    market_macro_maintain.add_argument("--market", required=True, choices=["CN"])
    market_macro_maintain.add_argument("--as-of", required=True)
    market_macro_maintain.add_argument(
        "--input-json",
        default="",
        help="包含四字段兼容 macro row 的本地 JSON；不接受隐式 fallback",
    )
    market_macro_maintain.add_argument("--data-root", default="data/parquet/cn/macro_daily")
    market_macro_maintain.add_argument("--snapshot-root", default="data/cn_market_full/_snapshots/macro")
    market_macro_maintain.add_argument("--input-observations", default="")
    market_macro_maintain.add_argument("--observations-root", default="data/parquet/cn/macro_observations")
    market_macro_maintain.add_argument("--run-id", default="")
    market_macro_maintain.add_argument("--indicator-id", action="append", default=[])
    market_macro_maintain.add_argument("--allow-tushare-fallback", action="store_true")
    market_macro_maintain.add_argument("--allow-live", action="store_true")

    market_macro_analyze = market_subparsers.add_parser(
        "macro-analyze",
        help="从本地 PIT observations 生成 measurement-only macro v2 observer 报告",
    )
    market_macro_analyze.add_argument("--market", required=True, choices=["CN"])
    market_macro_analyze.add_argument("--as-of", required=True)
    market_macro_analyze.add_argument("--observations", required=True)
    market_macro_analyze.add_argument("--output-dir", default="results/macro_observer")

    market_macro_replay = market_subparsers.add_parser(
        "macro-replay",
        help="固定一个本地 observations generation，按严格交易日历做 observer-only PIT 回放",
    )
    market_macro_replay.add_argument("--market", required=True, choices=["CN"])
    market_macro_replay.add_argument("--start-date", required=True)
    market_macro_replay.add_argument("--end-date", required=True)
    market_macro_replay.add_argument("--observations-root", default="data/parquet/cn/macro_observations")
    market_macro_replay.add_argument("--calendar", required=True)
    market_macro_replay.add_argument("--output-dir", default="results/macro_replay")
    market_macro_replay.add_argument("--run-id", default="")

    market_macro_normalize = market_subparsers.add_parser(
        "macro-normalize-tushare",
        help="离线编译本地 Tushare 宏观 raw bundle；默认只产出证据包，不晋升",
    )
    market_macro_normalize.add_argument("--market", required=True, choices=["CN"])
    market_macro_normalize.add_argument("--input-json", required=True)
    market_macro_normalize.add_argument("--plan-json", required=True)
    market_macro_normalize.add_argument("--evidence-json", required=True)
    market_macro_normalize.add_argument("--output-dir", default="results/macro_normalization")
    market_macro_normalize.add_argument("--run-id", default="")

    market_macro_publish = market_subparsers.add_parser(
        "macro-backfill-publish",
        help="显式 CAS 晋升一个零 quarantine 的离线 normalization bundle",
    )
    market_macro_publish.add_argument("--market", required=True, choices=["CN"])
    market_macro_publish.add_argument("--manifest", required=True)
    market_macro_publish.add_argument("--observations-root", default="data/parquet/cn/macro_observations")
    market_macro_publish.add_argument("--run-id", required=True)
    market_macro_publish.add_argument(
        "--expected-pointer-sha256",
        required=True,
        help="首次发布传 EMPTY；否则传当前 _latest.json 的 SHA-256",
    )
    market_macro_publish.add_argument("--expected-manifest-sha256", required=True)
    market_macro_publish.add_argument("--expected-plan-sha256", required=True)

    market_macro_forward = market_subparsers.add_parser(
        "macro-observe-forward",
        help="按已完成交易日追加 observer-only Macro 观察期证据；禁止历史回填",
    )
    market_macro_forward.add_argument("--market", required=True, choices=["CN"])
    market_macro_forward.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_forward.add_argument("--calendar", required=True)
    market_macro_forward.add_argument(
        "--state-root",
        default="results/macro_forward_observation",
    )
    market_macro_forward.add_argument(
        "--expected-pointer-sha256",
        required=True,
        help="首次记录传 EMPTY；否则传当前 Macro forward _latest.json 的 SHA-256",
    )

    market_macro_coverage = market_subparsers.add_parser(
        "macro-coverage-audit",
        help="离线审计国家指标与 12 条产业链的真实 PIT 覆盖和采集缺口",
    )
    market_macro_coverage.add_argument("--market", required=True, choices=["CN"])
    market_macro_coverage.add_argument("--as-of", required=True)
    market_macro_coverage.add_argument(
        "--observations",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_coverage.add_argument(
        "--raw-root",
        default="data/parquet/cn/dag_core_raw",
    )
    market_macro_coverage.add_argument(
        "--output-dir",
        default="results/macro_coverage_audit",
    )

    market_macro_acquisition = market_subparsers.add_parser(
        "macro-acquisition-plan",
        help="从 coverage audit 生成离线、observer-only 官方采集任务合同",
    )
    market_macro_acquisition.add_argument(
        "--market", required=True, choices=["CN"]
    )
    market_macro_acquisition.add_argument(
        "--coverage-audit", required=True
    )
    market_macro_acquisition.add_argument(
        "--output-dir", default="results/macro_acquisition_plan"
    )

    market_data_governance = market_subparsers.add_parser(
        "data-governance",
        help="审计四分支数据 readiness，默认只读本地数据",
    )
    market_data_governance.add_argument("--market", required=True, choices=["CN"])
    market_data_governance.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=[],
        help="可重复；默认 full_a",
    )
    market_data_governance.add_argument("--as-of", default="")
    market_data_governance.add_argument("--output-dir", default="reports/branch_readiness")
    market_data_governance.add_argument(
        "--allow-live",
        action="store_true",
        help="显式允许调用 Tushare/live provider 补数",
    )
    market_data_governance.add_argument(
        "--allow-public-fallback",
        action="store_true",
        help="显式允许公开结构化 fallback 补数；fallback 不伪装成 Tushare",
    )

    market_storage_validate = market_subparsers.add_parser(
        "storage-validate",
        help="校验本地 Parquet canonical snapshot 健康状态",
    )
    market_storage_validate.add_argument("--market", required=True, choices=["CN"])

    market_storage_validate_clean = market_subparsers.add_parser(
        "storage-validate-clean",
        help="只读校验本地 clean/readiness lineage 可用性",
    )
    market_storage_validate_clean.add_argument("--market", required=True, choices=["CN"])

    market_materialize_serving = market_subparsers.add_parser(
        "materialize-serving",
        help="从 Parquet canonical 重建 symbol serving layer",
    )
    market_materialize_serving.add_argument("--market", required=True, choices=["CN"])

    market_materialize_features = market_subparsers.add_parser(
        "materialize-features",
        help="按交易日生成 Parquet feature/cache 物化视图",
    )
    market_materialize_features.add_argument("--market", required=True, choices=["CN"])
    market_materialize_features.add_argument("--trade-date", required=True)

    market_storage_diff = market_subparsers.add_parser(
        "storage-diff",
        help="比较 Parquet canonical pointer 与 serving layer 覆盖状态",
    )
    market_storage_diff.add_argument("--market", required=True, choices=["CN"])

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
    market_analyze.add_argument(
        "--shortlist-size",
        type=int,
        default=config.BAYESIAN_SHORTLIST_SIZE,
    )
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
    market_run.add_argument(
        "--shortlist-size",
        type=int,
        default=config.BAYESIAN_SHORTLIST_SIZE,
    )
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
        maintenance_batch_size = args.batch_size if args.batch_size is not None else (200 if args.staged else 50)
        run_market_maintenance(
            market=args.market,
            categories=args.categories,
            years=args.years,
            max_workers=args.workers,
            batch_size=maintenance_batch_size,
            max_rounds=args.max_rounds,
            fail_on_incomplete=args.fail_on_incomplete,
            allowed_stale_symbols=args.allowed_stale_symbols,
            storage_mode=args.storage_mode,
            staged=args.staged,
            resume=args.resume,
            max_batches_per_run=args.max_batches_per_run,
            min_symbol_success_rate=args.min_symbol_success_rate,
            target_date=args.target_date,
            daily_window=args.daily_window,
        )
        return

    if args.command == "market" and args.market_command == "download":
        if not args.categories:
            parser.error("market download compatibility alias requires at least one --category")
        run_download(
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

    if args.command == "market" and args.market_command == "fundamental-maintain":
        run_fundamental_maintenance(
            market=args.market,
            universes=args.universes,
            years=args.years,
            as_of=args.as_of,
            workers=args.workers,
            data_root=args.data_root,
            raw_snapshot_root=args.snapshot_root,
            reports_root=args.reports_root,
            raw_input_dir=args.raw_input_dir or None,
            allow_live=args.allow_live,
        )
        return

    if args.command == "market" and args.market_command == "macro-maintain":
        if args.input_observations or (args.allow_live and not args.input_json):
            from quant_investor.macro.observer import load_macro_observations

            observations = load_macro_observations(args.input_observations) if args.input_observations else []
            generated_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = args.run_id or f"cn_macro_observations_{generated_suffix}"
            _print_json(
                run_macro_observation_maintenance(
                    local_observations=observations,
                    market=args.market,
                    as_of=args.as_of,
                    indicator_ids=args.indicator_id,
                    root=args.observations_root,
                    run_id=run_id,
                    allow_live=args.allow_live,
                    allow_tushare_fallback=args.allow_tushare_fallback,
                )
            )
            return
        indicators = None
        if args.input_json:
            input_path = Path(args.input_json).expanduser()
            if not input_path.exists() or input_path.is_symlink() or not input_path.is_file():
                parser.error("--input-json must reference a safe local file")
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                parser.error("--input-json must contain one JSON object")
            indicators = payload
        _print_json(
            run_macro_maintenance(
                indicators=indicators,
                as_of=args.as_of,
                data_root=args.data_root,
                raw_snapshot_root=args.snapshot_root,
                allow_live=args.allow_live,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-analyze":
        _print_json(
            run_macro_analysis(
                observations_path=args.observations,
                market=args.market,
                as_of=args.as_of,
                output_root=args.output_dir,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-replay":
        _print_json(
            run_macro_replay(
                market=args.market,
                start_date=args.start_date,
                end_date=args.end_date,
                observations_root=args.observations_root,
                calendar_path=args.calendar,
                output_root=args.output_dir,
                run_id=args.run_id,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-normalize-tushare":
        generated_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _print_json(
            run_macro_tushare_normalization(
                path=args.input_json,
                plan_path=args.plan_json,
                evidence_path=args.evidence_json,
                output_root=args.output_dir,
                run_id=args.run_id or f"cn_macro_normalization_{generated_suffix}",
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-backfill-publish":
        expected_pointer = str(args.expected_pointer_sha256).strip()
        if expected_pointer == "EMPTY":
            expected_pointer = ""
        _print_json(
            run_macro_backfill_publish(
                manifest_path=args.manifest,
                observations_root=args.observations_root,
                run_id=args.run_id,
                expected_pointer_sha256=expected_pointer,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_plan_sha256=args.expected_plan_sha256,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-observe-forward":
        expected_pointer = str(args.expected_pointer_sha256).strip()
        if expected_pointer == "EMPTY":
            expected_pointer = ""
        _print_json(
            run_macro_forward_observation(
                market=args.market,
                observations_root=args.observations_root,
                calendar_path=args.calendar,
                root=args.state_root,
                expected_pointer_sha256=expected_pointer,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-coverage-audit":
        _print_json(
            run_macro_coverage(
                market=args.market,
                as_of=args.as_of,
                observations_path=args.observations,
                raw_root=args.raw_root,
                output_root=args.output_dir,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-acquisition-plan":
        _print_json(
            run_macro_acquisition(
                market=args.market,
                coverage_audit=args.coverage_audit,
                output_root=args.output_dir,
            )
        )
        return

    if args.command == "market" and args.market_command == "data-governance":
        run_data_governance(
            market=args.market,
            categories=args.categories or ["full_a"],
            as_of=args.as_of,
            output_dir=args.output_dir,
            allow_live=args.allow_live,
            allow_public_fallback=args.allow_public_fallback,
        )
        return

    if args.command == "market" and args.market_command == "storage-validate":
        _print_json(run_storage_validate(market=args.market))
        return

    if args.command == "market" and args.market_command == "storage-validate-clean":
        _print_json(run_storage_validate_clean(market=args.market))
        return

    if args.command == "market" and args.market_command == "materialize-serving":
        _print_json(run_materialize_serving(market=args.market))
        return

    if args.command == "market" and args.market_command == "materialize-features":
        _print_json(run_materialize_features(market=args.market, trade_date=args.trade_date))
        return

    if args.command == "market" and args.market_command == "storage-diff":
        _print_json(run_storage_diff(market=args.market))
        return

    if args.command == "market" and args.market_command == "analyze":
        run_market_analysis(
            market=args.market,
            mode=args.mode,
            categories=args.categories,
            batch_size=args.batch_size,
            total_capital=args.capital,
            top_k=args.top_k,
            shortlist_size=args.shortlist_size,
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
            shortlist_size=args.shortlist_size,
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
