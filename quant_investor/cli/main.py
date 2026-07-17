"""
单一主线 CLI 入口。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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

    if kwargs.get("decision_protocol", "v15") == "v15":
        kwargs.pop("decision_protocol", None)
    return _run_market_analysis(**kwargs)


def run_market_pipeline(**kwargs):
    from quant_investor.market.run_pipeline import (
        run_unified_pipeline as _run_unified_pipeline,
    )

    if kwargs.get("decision_protocol", "v15") == "v15":
        kwargs.pop("decision_protocol", None)
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


def run_fundamental_promotion(**kwargs):
    from quant_investor.market.fundamental_generation import (
        promote_staged_fundamental_generation,
    )

    return promote_staged_fundamental_generation(**kwargs)


def run_macro_maintenance(**kwargs):
    from quant_investor.market.macro_mart import (
        run_cn_macro_maintenance as _run_cn_macro_maintenance,
    )

    return _run_cn_macro_maintenance(**kwargs)


def run_macro_authoritative_maintenance(**kwargs):
    from quant_investor.market.macro_mart import (
        stage_cn_macro_authoritative_refresh,
    )

    return stage_cn_macro_authoritative_refresh(**kwargs)


def run_macro_promotion(**kwargs):
    from quant_investor.market.macro_mart import promote_staged_macro_generation

    return promote_staged_macro_generation(**kwargs)


def run_macro_analysis(**kwargs):
    from quant_investor.macro.observer import (
        build_macro_observer,
        load_macro_observation_generation,
    )

    observations, generation = load_macro_observation_generation(
        kwargs.pop("observations_path"),
        allow_standalone_offline=True,
    )
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
    from quant_investor.macro.tushare_normalizer import (
        normalize_tushare_bundle_file,
    )

    return normalize_tushare_bundle_file(**kwargs)


def run_macro_official_web_compilation(**kwargs):
    from quant_investor.macro.official_web_compiler import (
        compile_official_web_bundle_file,
    )

    return compile_official_web_bundle_file(**kwargs)


def run_macro_release_calendar_publish(**kwargs):
    from quant_investor.macro.release_calendar import publish_release_calendar

    result = publish_release_calendar(**kwargs)
    return {
        "schema_version": "macro-release-calendar-publication.v1",
        "status": "OK",
        "idempotent": result.idempotent,
        "identity": asdict(result.identity),
        "captured_at": result.evidence.captured_at,
        "open_date_count": len(result.evidence.open_dates),
        "event_count": len(result.evidence.events),
        "resolution_count": len(result.evidence.resolutions),
    }


def run_macro_production_observation_bundle(**kwargs):
    from quant_investor.macro.production_observation_bundle import (
        publish_macro_production_observation_bundle,
    )

    return publish_macro_production_observation_bundle(**kwargs)


def run_macro_official_observation_refresh(**kwargs):
    from quant_investor.macro.production_observation_bundle import (
        publish_macro_official_observation_refresh,
    )

    return publish_macro_official_observation_refresh(**kwargs)


def run_macro_local_breadth_publish(**kwargs):
    from quant_investor.macro.production_observation_bundle import (
        publish_local_market_breadth_update,
    )

    return publish_local_market_breadth_update(**kwargs)


def run_macro_local_breadth_roll(**kwargs):
    from quant_investor.macro.production_observation_bundle import (
        publish_local_market_breadth_roll,
    )

    return publish_local_market_breadth_roll(**kwargs)


def _market_open_dates_for_dispatch(path_value: str) -> list[str]:
    path = Path(path_value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("market open-days file must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("open_dates") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("market open-days file must contain open_dates")
    return [str(value) for value in values]


def run_macro_backfill_publish(**kwargs):
    from quant_investor.macro.tushare_normalizer import (
        publish_tushare_normalization,
    )

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


def run_fundamental_research_prepare(**kwargs):
    from quant_investor.fundamental_research.workflow import prepare_research_requests

    return prepare_research_requests(**kwargs)


def run_fundamental_research_import(**kwargs):
    from quant_investor.fundamental_research.workflow import import_research_response

    return import_research_response(**kwargs)


def run_fundamental_research_status(**kwargs):
    from quant_investor.fundamental_research.workflow import research_status

    return research_status(**kwargs)


def run_fundamental_research_gate_evidence(**kwargs):
    from quant_investor.fundamental_research.workflow import (
        generate_activation_gate_evidence,
    )

    return generate_activation_gate_evidence(**kwargs)


def run_fundamental_research_longitudinal_import(**kwargs):
    from quant_investor.fundamental_research.governance import (
        append_longitudinal_observation,
    )

    return append_longitudinal_observation(**kwargs)


def run_fundamental_research_target_weight_produce(**kwargs):
    from quant_investor.fundamental_research.longitudinal_producer import (
        produce_target_weight_observation,
    )

    return produce_target_weight_observation(**kwargs)


def run_fundamental_research_nav_produce(**kwargs):
    from quant_investor.fundamental_research.longitudinal_producer import (
        produce_nav_attribution_observation,
    )

    return produce_nav_attribution_observation(**kwargs)


def run_codex_review_export(**kwargs):
    from quant_investor.codex_review import export_review_request

    return export_review_request(**kwargs)


def run_codex_review_receive(**kwargs):
    from quant_investor.codex_review import receive_review_response

    return receive_review_response(**kwargs)


def run_codex_review_validate(**kwargs):
    from quant_investor.codex_review import validate_review_response

    return validate_review_response(**kwargs)


def run_codex_review_resume(**kwargs):
    from quant_investor.codex_review import resume_review

    return resume_review(**kwargs)


def run_codex_review_status(**kwargs):
    from quant_investor.codex_review import review_status

    return review_status(**kwargs)


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
        help="兼容保留参数；v15 三分支主线不再执行 kline 分支。",
    )
    research_run.add_argument("--no-macro", action="store_true")
    research_run.add_argument("--no-kline", "--no-kronos", action="store_true", help="兼容保留参数；v15 默认不执行 kline 分支。")
    research_run.add_argument("--no-quant", action="store_true")
    research_run.add_argument("--no-fundamental", action="store_true")
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

    market_codex_export = market_subparsers.add_parser(
        "codex-review-export",
        help="导出已准备的本地 v16 Codex review request",
    )
    market_codex_export.add_argument(
        "--root", default="results/v16/codex_review"
    )
    market_codex_export.add_argument("--run-id", required=True)
    market_codex_export.add_argument("--expected-state-sha256", required=True)

    market_codex_receive = market_subparsers.add_parser(
        "codex-review-receive",
        help="一次性接收严格 JSON Codex review response",
    )
    market_codex_receive.add_argument(
        "--root", default="results/v16/codex_review"
    )
    market_codex_receive.add_argument("--run-id", required=True)
    market_codex_receive.add_argument("--response", required=True)
    market_codex_receive.add_argument("--expected-state-sha256", required=True)

    market_codex_validate = market_subparsers.add_parser(
        "codex-review-validate",
        help="重验 response 的 schema、SHA、PIT、集合和前序绑定",
    )
    market_codex_validate.add_argument(
        "--root", default="results/v16/codex_review"
    )
    market_codex_validate.add_argument("--run-id", required=True)
    market_codex_validate.add_argument("--expected-state-sha256", required=True)

    market_codex_resume = market_subparsers.add_parser(
        "codex-review-resume",
        help="从当前封存边界推进一个确定性本地状态",
    )
    market_codex_resume.add_argument(
        "--root", default="results/v16/codex_review"
    )
    market_codex_resume.add_argument("--run-id", required=True)
    market_codex_resume.add_argument("--expected-state-sha256", required=True)
    market_codex_resume.add_argument("--menu", default="")
    market_codex_resume.add_argument("--total-capital", type=float, default=None)
    market_codex_resume.add_argument("--authorization", default="")

    market_codex_status = market_subparsers.add_parser(
        "codex-review-status",
        help="只读显示 Codex review 状态、expiry 和绑定漂移",
    )
    market_codex_status.add_argument(
        "--root", default="results/v16/codex_review"
    )
    market_codex_status.add_argument("--run-id", required=True)

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
        "--pit-generation-manifest",
        default="",
        help="parquet-direct 必需：显式选择 immutable PIT generation manifest",
    )
    market_maintain.add_argument(
        "--expected-pit-generation-manifest-sha256",
        default="",
        help="parquet-direct 必需：所选 PIT generation manifest 的 SHA-256",
    )
    market_maintain.add_argument(
        "--expected-market-pointer-sha256",
        default="",
        help="parquet-direct 必需：维护开始前 CN _latest.json 的 SHA-256 CAS",
    )
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
    market_download.add_argument("--pit-generation-manifest", default="")
    market_download.add_argument(
        "--expected-pit-generation-manifest-sha256",
        default="",
    )
    market_download.add_argument(
        "--expected-market-pointer-sha256",
        default="",
    )

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
    market_fundamental.add_argument("--run-id", default="")
    market_fundamental.add_argument(
        "--allow-live",
        action="store_true",
        help="显式允许调用 live provider；本地测试默认不使用",
    )
    market_fundamental.add_argument(
        "--authoritative-full-rebuild",
        action="store_true",
        help="在隔离 data root 执行 scope/hash/PIT/audit 绑定的权威全量重建",
    )
    market_fundamental.add_argument("--canonical-scope-path", default="")
    market_fundamental.add_argument("--canonical-market-pointer-path", default="")
    market_fundamental.add_argument("--canonical-membership-path", default="")
    market_fundamental.add_argument("--checkpoint-root", default="")
    market_fundamental.add_argument("--checkpoint-batch-size", type=int, default=500)
    market_fundamental.add_argument("--max-attempts", type=int, default=3)
    market_fundamental.add_argument("--retry-backoff-seconds", type=float, default=0.5)
    market_fundamental.add_argument(
        "--max-retry-backoff-seconds",
        type=float,
        default=8.0,
    )
    market_fundamental.add_argument("--requests-per-second", type=float, default=8.0)

    market_fundamental_promote = market_subparsers.add_parser(
        "fundamental-promote",
        help="以 expected-pointer SHA 原子晋升已验证的 Fundamental staging generation",
    )
    market_fundamental_promote.add_argument("--staging-root", required=True)
    market_fundamental_promote.add_argument(
        "--canonical-root",
        default="data/parquet/cn",
    )
    market_fundamental_promote.add_argument(
        "--expected-pointer-sha256",
        required=True,
    )

    market_macro_maintain = market_subparsers.add_parser(
        "macro-maintain",
        help="暂存 CN Macro 数据；权威刷新也只写隔离 staging",
    )
    market_macro_maintain.add_argument("--market", required=True, choices=["CN"])
    market_macro_maintain.add_argument("--as-of", required=True)
    macro_input_group = market_macro_maintain.add_mutually_exclusive_group()
    macro_input_group.add_argument(
        "--input-json",
        default="",
        help="四字段本地兼容行；只能暂存，不能晋升 canonical",
    )
    macro_input_group.add_argument(
        "--input-observations",
        default="",
        help=(
            "本地 observations 仅写 sanitized observer staging；"
            "永不推进 canonical pointer"
        ),
    )
    market_macro_maintain.add_argument(
        "--data-root",
        default="data/parquet/cn/macro_daily",
    )
    market_macro_maintain.add_argument(
        "--snapshot-root",
        default="data/cn_market_full/_snapshots/macro",
    )
    market_macro_maintain.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_maintain.add_argument(
        "--expected-macro-observations-pointer-sha256",
        default="",
        help=(
            "authoritative refresh 所固定的 canonical Macro observations "
            "_latest.json SHA-256"
        ),
    )
    market_macro_maintain.add_argument(
        "--macro-release-calendar-root",
        default="data/parquet/cn/macro_release_calendar",
        help="authoritative refresh 所固定的 immutable Macro release calendar",
    )
    market_macro_maintain.add_argument(
        "--expected-macro-release-calendar-pointer-sha256",
        default="",
        help="所固定 release calendar _latest.json 的 SHA-256",
    )
    market_macro_maintain.add_argument(
        "--staging-root",
        default="results/v15/macro_observation_staging",
    )
    market_macro_maintain.add_argument("--run-id", default="")
    market_macro_maintain.add_argument(
        "--indicator-id",
        action="append",
        default=[],
    )
    market_macro_maintain.add_argument(
        "--allow-tushare-fallback",
        action="store_true",
    )
    market_macro_maintain.add_argument("--allow-live", action="store_true")
    market_macro_maintain.add_argument(
        "--authoritative-refresh",
        action="store_true",
        help="捕获权威 provider 与市场证据并只写隔离 staging",
    )
    market_macro_maintain.add_argument(
        "--canonical-root",
        default="data/parquet/cn/macro_daily",
    )
    market_macro_maintain.add_argument("--expected-catalog-sha256", default="")
    market_macro_maintain.add_argument(
        "--expected-market-pointer-sha256", default=""
    )
    market_macro_maintain.add_argument("--nbs-cn-pmi-url", default="")

    market_macro_promote = market_subparsers.add_parser(
        "macro-promote",
        help="独立重验 staging generation 并 CAS 更新 strict catalog",
    )
    market_macro_promote.add_argument("--staging-root", required=True)
    market_macro_promote.add_argument(
        "--canonical-root",
        default="data/parquet/cn/macro_daily",
    )
    market_macro_promote.add_argument(
        "--expected-catalog-sha256", required=True
    )

    market_macro_analyze = market_subparsers.add_parser(
        "macro-analyze",
        help="显式读取本地 observations，生成 observer-only v15 报告",
    )
    market_macro_analyze.add_argument("--market", required=True, choices=["CN"])
    market_macro_analyze.add_argument("--as-of", required=True)
    market_macro_analyze.add_argument("--observations", required=True)
    market_macro_analyze.add_argument(
        "--output-dir",
        default="results/v15/macro_observer",
    )

    market_macro_replay = market_subparsers.add_parser(
        "macro-replay",
        help="固定 canonical observations generation 做 observer-only PIT 回放",
    )
    market_macro_replay.add_argument("--market", required=True, choices=["CN"])
    market_macro_replay.add_argument("--start-date", required=True)
    market_macro_replay.add_argument("--end-date", required=True)
    market_macro_replay.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_replay.add_argument("--calendar", required=True)
    market_macro_replay.add_argument(
        "--output-dir",
        default="results/v15/macro_replay",
    )
    market_macro_replay.add_argument("--run-id", default="")

    market_macro_normalize = market_subparsers.add_parser(
        "macro-normalize-tushare",
        help="离线编译 hash-bound Tushare 证据；不自动晋升",
    )
    market_macro_normalize.add_argument("--market", required=True, choices=["CN"])
    market_macro_normalize.add_argument("--input-json", required=True)
    market_macro_normalize.add_argument("--plan-json", required=True)
    market_macro_normalize.add_argument("--evidence-json", required=True)
    market_macro_normalize.add_argument(
        "--output-dir",
        default="results/v15/macro_normalization",
    )
    market_macro_normalize.add_argument("--run-id", required=True)

    market_macro_official_compile = market_subparsers.add_parser(
        "macro-official-compile",
        help=(
            "离线编译 hash-bound NBS/PBC 官方网页捕获；"
            "只写私有 bundle，不推进 canonical pointer"
        ),
    )
    market_macro_official_compile.add_argument("--plan", required=True)
    market_macro_official_compile.add_argument(
        "--capture-manifest", required=True
    )
    market_macro_official_compile.add_argument("--raw-root", required=True)
    market_macro_official_compile.add_argument(
        "--output-dir",
        default="results/v15/macro_official_web",
    )
    market_macro_official_compile.add_argument("--run-id", required=True)

    market_macro_release_calendar = market_subparsers.add_parser(
        "macro-release-calendar-publish",
        help="校验官方发布事件与交易日证据，并以 CAS 发布 immutable 日历",
    )
    market_macro_release_calendar.add_argument("--plan", required=True)
    market_macro_release_calendar.add_argument(
        "--expected-plan-sha256", required=True
    )
    market_macro_release_calendar.add_argument(
        "--capture-manifest", required=True
    )
    market_macro_release_calendar.add_argument(
        "--expected-capture-manifest-sha256", required=True
    )
    market_macro_release_calendar.add_argument("--raw-root", required=True)
    market_macro_release_calendar.add_argument(
        "--market-open-days", required=True
    )
    market_macro_release_calendar.add_argument(
        "--expected-market-open-days-sha256", required=True
    )
    market_macro_release_calendar.add_argument(
        "--canonical-root",
        default="data/parquet/cn/macro_release_calendar",
    )
    market_macro_release_calendar.add_argument("--run-id", required=True)
    market_macro_release_calendar.add_argument(
        "--expected-pointer-sha256", required=True
    )

    market_macro_observation_bootstrap = market_subparsers.add_parser(
        "macro-observation-bootstrap",
        help=(
            "重编译官方网页与 strict-local breadth evidence，"
            "并以一个 CAS 原子发布 39 条 canonical observations"
        ),
    )
    market_macro_observation_bootstrap.add_argument(
        "--official-manifest", required=True
    )
    market_macro_observation_bootstrap.add_argument(
        "--expected-official-manifest-sha256", required=True
    )
    market_macro_observation_bootstrap.add_argument(
        "--expected-official-plan-sha256", required=True
    )
    market_macro_observation_bootstrap.add_argument(
        "--local-binding-plan", required=True
    )
    market_macro_observation_bootstrap.add_argument(
        "--expected-local-binding-plan-sha256", required=True
    )
    market_macro_observation_bootstrap.add_argument("--as-of", required=True)
    market_macro_observation_bootstrap.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_observation_bootstrap.add_argument(
        "--run-id", required=True
    )
    market_macro_observation_bootstrap.add_argument(
        "--expected-pointer-sha256", required=True
    )

    market_macro_official_refresh = market_subparsers.add_parser(
        "macro-observation-official-refresh",
        help=(
            "以完整新官方 bundle 替换 official scope，并保留最新三条"
            " strict-local 历史"
        ),
    )
    market_macro_official_refresh.add_argument(
        "--official-manifest", required=True
    )
    market_macro_official_refresh.add_argument(
        "--expected-official-manifest-sha256", required=True
    )
    market_macro_official_refresh.add_argument(
        "--expected-official-plan-sha256", required=True
    )
    market_macro_official_refresh.add_argument(
        "--target-as-of", required=True
    )
    market_macro_official_refresh.add_argument(
        "--decision-cutoff-at", required=True
    )
    market_macro_official_refresh.add_argument(
        "--market-open-days", required=True
    )
    market_macro_official_refresh.add_argument(
        "--expected-market-open-days-sha256", required=True
    )
    market_macro_official_refresh.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_official_refresh.add_argument("--run-id", required=True)
    market_macro_official_refresh.add_argument(
        "--expected-pointer-sha256", required=True
    )

    market_macro_local_publish = market_subparsers.add_parser(
        "macro-local-observation-publish",
        help=(
            "从显式 cutoff 前 strict-Parquet snapshot 编译并 CAS 追加"
            "一条 target-date market.breadth observation"
        ),
    )
    market_macro_local_publish.add_argument(
        "--snapshot-manifest", required=True
    )
    market_macro_local_publish.add_argument(
        "--expected-snapshot-manifest-sha256", required=True
    )
    market_macro_local_publish.add_argument(
        "--coverage-manifest", required=True
    )
    market_macro_local_publish.add_argument(
        "--expected-coverage-manifest-sha256", required=True
    )
    market_macro_local_publish.add_argument(
        "--target-trade-date", required=True
    )
    market_macro_local_publish.add_argument("--scope-artifact", required=True)
    market_macro_local_publish.add_argument(
        "--expected-scope-artifact-sha256", required=True
    )
    market_macro_local_publish.add_argument("--as-of", required=True)
    market_macro_local_publish.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_local_publish.add_argument("--run-id", required=True)
    market_macro_local_publish.add_argument(
        "--expected-pointer-sha256", required=True
    )

    market_macro_local_roll = market_subparsers.add_parser(
        "macro-local-observation-roll",
        help="在 exact-projection 生产链上 catch up/滚动三条 local breadth 历史",
    )
    market_macro_local_roll.add_argument(
        "--snapshot-manifest", required=True
    )
    market_macro_local_roll.add_argument(
        "--expected-snapshot-manifest-sha256", required=True
    )
    market_macro_local_roll.add_argument(
        "--coverage-manifest", required=True
    )
    market_macro_local_roll.add_argument(
        "--expected-coverage-manifest-sha256", required=True
    )
    market_macro_local_roll.add_argument(
        "--target-trade-date", required=True
    )
    market_macro_local_roll.add_argument(
        "--scope-artifact", required=True
    )
    market_macro_local_roll.add_argument(
        "--expected-scope-artifact-sha256", required=True
    )
    market_macro_local_roll.add_argument("--target-as-of", required=True)
    market_macro_local_roll.add_argument(
        "--decision-cutoff-at", required=True
    )
    market_macro_local_roll.add_argument("--market-open-days", required=True)
    market_macro_local_roll.add_argument(
        "--expected-market-open-days-sha256", required=True
    )
    market_macro_local_roll.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_local_roll.add_argument("--run-id", required=True)
    market_macro_local_roll.add_argument(
        "--expected-pointer-sha256", required=True
    )

    market_macro_publish = market_subparsers.add_parser(
        "macro-backfill-publish",
        help="显式 CAS 晋升零 quarantine 的 observations bundle",
    )
    market_macro_publish.add_argument("--market", required=True, choices=["CN"])
    market_macro_publish.add_argument("--manifest", required=True)
    market_macro_publish.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_publish.add_argument("--run-id", required=True)
    market_macro_publish.add_argument("--expected-pointer-sha256", required=True)
    market_macro_publish.add_argument("--expected-manifest-sha256", required=True)
    market_macro_publish.add_argument("--expected-plan-sha256", required=True)

    market_macro_forward = market_subparsers.add_parser(
        "macro-observe-forward",
        help="追加一个已完成交易日的 observer-only forward 证据",
    )
    market_macro_forward.add_argument("--market", required=True, choices=["CN"])
    market_macro_forward.add_argument(
        "--observations-root",
        default="data/parquet/cn/macro_observations",
    )
    market_macro_forward.add_argument("--calendar", required=True)
    market_macro_forward.add_argument(
        "--state-root",
        default="results/v15/macro_forward_observation",
    )
    market_macro_forward.add_argument("--expected-pointer-sha256", required=True)

    market_macro_coverage = market_subparsers.add_parser(
        "macro-coverage-audit",
        help="离线审计国家指标与产业链 PIT 覆盖",
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
        default="results/v15/macro_coverage_audit",
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
        "--output-dir", default="results/v15/macro_acquisition_plan"
    )

    market_data_governance = market_subparsers.add_parser(
        "data-governance",
        help="审计三分支数据 readiness，默认只读本地数据",
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
    market_data_governance.add_argument(
        "--output-dir", default="reports/v15/branch_readiness"
    )
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

    market_fundamental_prepare = market_subparsers.add_parser(
        "fundamental-research-prepare",
        help="从显式 analysis/manual manifests 生成离线 Codex fundamental 研究任务",
    )
    market_fundamental_prepare.add_argument("--market", required=True, choices=["CN"])
    market_fundamental_prepare.add_argument(
        "--as-of",
        required=True,
        help="带时区的决策截止 ISO-8601 时间，且不得晚于 analysis 数据截止",
    )
    market_fundamental_prepare.add_argument("--analysis-run", required=True)
    market_fundamental_prepare.add_argument("--holdings-manifest", required=True)
    market_fundamental_prepare.add_argument(
        "--root", default="results/fundamental_research"
    )
    market_fundamental_prepare.add_argument(
        "--prompt-version", default="fundamental-dossier-v1"
    )
    market_fundamental_prepare.add_argument(
        "--policy-version", default="v15-fundamental-research"
    )

    market_fundamental_import = market_subparsers.add_parser(
        "fundamental-research-import",
        help="离线校验并导入外部 Codex fundamental dossier",
    )
    market_fundamental_import.add_argument("--request", required=True)
    market_fundamental_import.add_argument("--response", required=True)
    market_fundamental_import.add_argument(
        "--root", default="results/fundamental_research"
    )
    market_fundamental_import.add_argument("--validate-only", action="store_true")

    market_fundamental_status = market_subparsers.add_parser(
        "fundamental-research-status",
        help="只读查看 external fundamental research job 状态",
    )
    market_fundamental_status.add_argument("--market", default="CN", choices=["CN"])
    market_fundamental_status.add_argument(
        "--root", default="results/fundamental_research"
    )

    market_fundamental_gate_evidence = market_subparsers.add_parser(
        "fundamental-research-gate-evidence",
        help="从 private ledgers 重算并写入 activation gate evidence",
    )
    market_fundamental_gate_evidence.add_argument(
        "--holdings-manifest", required=True
    )
    market_fundamental_gate_evidence.add_argument(
        "--root", default="results/fundamental_research"
    )
    market_fundamental_longitudinal = market_subparsers.add_parser(
        "fundamental-research-longitudinal-import",
        help="校验并追加 target-weight/NAV 纵向反事实观察",
    )
    market_fundamental_longitudinal.add_argument("--observation", required=True)
    market_fundamental_longitudinal.add_argument(
        "--root", default="results/fundamental_research"
    )
    market_fundamental_target_weight = market_subparsers.add_parser(
        "fundamental-research-target-weight-produce",
        help="从真实双 control-chain analysis manifests 生成权重反事实观察",
    )
    market_fundamental_target_weight.add_argument("--request", required=True)
    market_fundamental_target_weight.add_argument("--dossier-id", required=True)
    market_fundamental_target_weight.add_argument("--actual-analysis", required=True)
    market_fundamental_target_weight.add_argument(
        "--counterfactual-analysis", required=True
    )
    market_fundamental_target_weight.add_argument(
        "--root", default="results/fundamental_research"
    )
    market_fundamental_nav = market_subparsers.add_parser(
        "fundamental-research-nav-produce",
        help="用 strict Parquet 次日收益生成 canonical NAV 归因观察",
    )
    market_fundamental_nav.add_argument("--target-weight-observation", required=True)
    market_fundamental_nav.add_argument("--attribution-date", required=True)
    market_fundamental_nav.add_argument("--data-root", default="data")
    market_fundamental_nav.add_argument(
        "--root", default="results/fundamental_research"
    )
    market_fundamental_status.add_argument("--run-id", default="")
    market_fundamental_status.add_argument("--symbol", default="")
    market_fundamental_status.add_argument(
        "--state",
        default="",
        choices=[
            "",
            "PREPARED",
            "EXPORTED",
            "RECEIVED",
            "VALIDATED",
            "REJECTED",
            "EXPIRED",
            "SUPERSEDED",
        ],
    )

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
        "--decision-protocol",
        choices=["v15", "v16"],
        default="v15",
        help="候选决策协议；v15 默认保持当前权威路径，v16 仅准备本地 Codex Stage1 pending",
    )
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
        "--decision-protocol",
        choices=["v15", "v16"],
        default="v15",
        help="候选决策协议；v15 默认保持当前权威路径，v16 仅准备本地 Codex Stage1 pending",
    )
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

    codex_review_command = (
        args.market_command
        if args.command == "market"
        and args.market_command.startswith("codex-review-")
        else ""
    )

    if codex_review_command == "codex-review-export":
        _print_json(
            run_codex_review_export(
                root=args.root,
                run_id=args.run_id,
                expected_state_sha256=args.expected_state_sha256,
            )
        )
        return

    if codex_review_command == "codex-review-receive":
        _print_json(
            run_codex_review_receive(
                root=args.root,
                run_id=args.run_id,
                response_path=args.response,
                expected_state_sha256=args.expected_state_sha256,
            )
        )
        return

    if codex_review_command == "codex-review-validate":
        _print_json(
            run_codex_review_validate(
                root=args.root,
                run_id=args.run_id,
                expected_state_sha256=args.expected_state_sha256,
            )
        )
        return

    if codex_review_command == "codex-review-resume":
        _print_json(
            run_codex_review_resume(
                root=args.root,
                run_id=args.run_id,
                expected_state_sha256=args.expected_state_sha256,
                menu_path=args.menu or None,
                total_capital=args.total_capital,
                authorization_path=args.authorization or None,
            )
        )
        return

    if codex_review_command == "codex-review-status":
        _print_json(run_codex_review_status(root=args.root, run_id=args.run_id))
        return

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
            pit_generation_manifest=args.pit_generation_manifest,
            expected_pit_generation_manifest_sha256=(
                args.expected_pit_generation_manifest_sha256
            ),
            expected_market_pointer_sha256=(
                args.expected_market_pointer_sha256
            ),
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
            pit_generation_manifest=args.pit_generation_manifest,
            expected_pit_generation_manifest_sha256=(
                args.expected_pit_generation_manifest_sha256
            ),
            expected_market_pointer_sha256=(
                args.expected_market_pointer_sha256
            ),
        )
        return

    if args.command == "market" and args.market_command == "fundamental-maintain":
        if args.authoritative_full_rebuild and (
            not args.canonical_scope_path
            or not args.canonical_market_pointer_path
            or not args.canonical_membership_path
            or not args.checkpoint_root
            or not args.run_id
        ):
            parser.error(
                "--authoritative-full-rebuild requires --run-id, "
                "--canonical-scope-path, --canonical-market-pointer-path, "
                "--canonical-membership-path, and --checkpoint-root"
            )
        result = run_fundamental_maintenance(
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
            run_id=args.run_id,
            authoritative_full_rebuild=args.authoritative_full_rebuild,
            canonical_scope_path=args.canonical_scope_path or None,
            canonical_market_pointer_path=(
                args.canonical_market_pointer_path or None
            ),
            canonical_membership_path=args.canonical_membership_path or None,
            checkpoint_root=args.checkpoint_root or None,
            checkpoint_batch_size=args.checkpoint_batch_size,
            max_attempts=args.max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            max_retry_backoff_seconds=args.max_retry_backoff_seconds,
            requests_per_second=args.requests_per_second,
        )
        if args.authoritative_full_rebuild:
            _print_json(result)
        return

    if args.command == "market" and args.market_command == "fundamental-promote":
        _print_json(
            run_fundamental_promotion(
                staging_root=args.staging_root,
                canonical_root=args.canonical_root,
                expected_pointer_sha256=args.expected_pointer_sha256,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-maintain":
        if args.allow_tushare_fallback and not args.allow_live:
            parser.error(
                "--allow-tushare-fallback requires explicit --allow-live"
            )
        if args.input_observations and (
            args.allow_live or args.allow_tushare_fallback
        ):
            parser.error(
                "standalone --input-observations cannot be combined with "
                "live provider flags"
            )
        if args.input_json and (
            args.allow_live or args.allow_tushare_fallback
        ):
            parser.error(
                "compatibility --input-json cannot be combined with live provider flags"
            )
        if args.authoritative_refresh:
            if (
                not args.allow_live
                or not args.run_id
                or not args.expected_catalog_sha256
                or not args.expected_market_pointer_sha256
                or not args.expected_macro_observations_pointer_sha256
                or not args.expected_macro_release_calendar_pointer_sha256
                or not args.nbs_cn_pmi_url
            ):
                parser.error(
                    "--authoritative-refresh requires --allow-live, --run-id, "
                    "--expected-catalog-sha256, --expected-market-pointer-sha256, "
                    "--expected-macro-observations-pointer-sha256, "
                    "--expected-macro-release-calendar-pointer-sha256 and "
                    "--nbs-cn-pmi-url"
                )
            if args.input_json or args.input_observations:
                parser.error(
                    "--authoritative-refresh cannot be combined with local input"
                )
            _print_json(
                run_macro_authoritative_maintenance(
                    market=args.market,
                    as_of=args.as_of,
                    canonical_root=args.canonical_root,
                    staging_root=args.staging_root,
                    run_id=args.run_id,
                    expected_catalog_sha256=args.expected_catalog_sha256,
                    expected_market_pointer_sha256=(
                        args.expected_market_pointer_sha256
                    ),
                    macro_observations_root=args.observations_root,
                    expected_macro_observations_pointer_sha256=(
                        args.expected_macro_observations_pointer_sha256
                    ),
                    macro_release_calendar_root=(
                        str(
                            Path(args.macro_release_calendar_root)
                            .expanduser()
                            .resolve()
                        )
                    ),
                    expected_macro_release_calendar_pointer_sha256=(
                        args.expected_macro_release_calendar_pointer_sha256
                    ),
                    allow_live=args.allow_live,
                    nbs_cn_pmi_url=args.nbs_cn_pmi_url,
                    allow_tushare_fallback=args.allow_tushare_fallback,
                )
            )
            return
        if args.input_observations or args.allow_live:
            from quant_investor.macro.observer import load_macro_observations

            observations = (
                load_macro_observations(
                    args.input_observations,
                    allow_standalone_offline=True,
                )
                if args.input_observations
                else []
            )
            generated_suffix = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            run_id = (
                args.run_id
                or f"cn_macro_observations_{generated_suffix}"
            )
            _print_json(
                run_macro_observation_maintenance(
                    local_observations=observations,
                    market=args.market,
                    as_of=args.as_of,
                    indicator_ids=args.indicator_id,
                    root=args.observations_root,
                    staging_root=args.staging_root,
                    run_id=run_id,
                    allow_live=args.allow_live,
                    allow_tushare_fallback=args.allow_tushare_fallback,
                )
            )
            return
        indicators = None
        if args.input_json:
            input_path = Path(args.input_json).expanduser()
            if (
                not input_path.exists()
                or input_path.is_symlink()
                or not input_path.is_file()
            ):
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
                allow_live=False,
                allow_public_fallback=False,
                run_id=args.run_id,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-promote":
        _print_json(
            run_macro_promotion(
                staging_root=args.staging_root,
                canonical_root=args.canonical_root,
                expected_catalog_sha256=args.expected_catalog_sha256,
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
                production_enabled=False,
                production_kill_switch=True,
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

    if (
        args.command == "market"
        and args.market_command == "macro-normalize-tushare"
    ):
        _print_json(
            run_macro_tushare_normalization(
                path=args.input_json,
                plan_path=args.plan_json,
                evidence_path=args.evidence_json,
                output_root=args.output_dir,
                run_id=args.run_id,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-official-compile"
    ):
        _print_json(
            run_macro_official_web_compilation(
                plan_path=args.plan,
                capture_manifest_path=args.capture_manifest,
                raw_root=args.raw_root,
                output_root=args.output_dir,
                run_id=args.run_id,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-release-calendar-publish"
    ):
        _print_json(
            run_macro_release_calendar_publish(
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                capture_manifest_path=args.capture_manifest,
                expected_capture_manifest_sha256=(
                    args.expected_capture_manifest_sha256
                ),
                raw_root=args.raw_root,
                market_open_days_path=args.market_open_days,
                expected_market_open_days_sha256=(
                    args.expected_market_open_days_sha256
                ),
                canonical_root=str(
                    Path(args.canonical_root).expanduser().resolve()
                ),
                run_id=args.run_id,
                expected_pointer_sha256=args.expected_pointer_sha256,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-observation-bootstrap"
    ):
        _print_json(
            run_macro_production_observation_bundle(
                official_bundle_manifest_path=args.official_manifest,
                expected_official_bundle_manifest_sha256=(
                    args.expected_official_manifest_sha256
                ),
                expected_official_plan_sha256=(
                    args.expected_official_plan_sha256
                ),
                local_bootstrap_plan_path=args.local_binding_plan,
                expected_local_bootstrap_plan_sha256=(
                    args.expected_local_binding_plan_sha256
                ),
                as_of=args.as_of,
                canonical_observations_root=args.observations_root,
                run_id=args.run_id,
                expected_pointer_sha256=(
                    ""
                    if args.expected_pointer_sha256 == "EMPTY"
                    else args.expected_pointer_sha256
                ),
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-observation-official-refresh"
    ):
        try:
            pinned_open_dates = _market_open_dates_for_dispatch(
                args.market_open_days
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
        _print_json(
            run_macro_official_observation_refresh(
                official_bundle_manifest_path=args.official_manifest,
                expected_official_bundle_manifest_sha256=(
                    args.expected_official_manifest_sha256
                ),
                expected_official_plan_sha256=(
                    args.expected_official_plan_sha256
                ),
                target_as_of=args.target_as_of,
                decision_cutoff_at=args.decision_cutoff_at,
                pinned_open_dates=pinned_open_dates,
                market_open_days_path=args.market_open_days,
                expected_market_open_days_sha256=(
                    args.expected_market_open_days_sha256
                ),
                canonical_observations_root=args.observations_root,
                run_id=args.run_id,
                expected_pointer_sha256=(
                    ""
                    if args.expected_pointer_sha256 == "EMPTY"
                    else args.expected_pointer_sha256
                ),
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-local-observation-publish"
    ):
        _print_json(
            run_macro_local_breadth_publish(
                snapshot_manifest_path=args.snapshot_manifest,
                expected_snapshot_manifest_sha256=(
                    args.expected_snapshot_manifest_sha256
                ),
                coverage_manifest_path=args.coverage_manifest,
                expected_coverage_manifest_sha256=(
                    args.expected_coverage_manifest_sha256
                ),
                target_trade_date=args.target_trade_date,
                scope_artifact_path=args.scope_artifact,
                expected_scope_artifact_sha256=(
                    args.expected_scope_artifact_sha256
                ),
                as_of=args.as_of,
                canonical_observations_root=args.observations_root,
                run_id=args.run_id,
                expected_pointer_sha256=(
                    ""
                    if args.expected_pointer_sha256 == "EMPTY"
                    else args.expected_pointer_sha256
                ),
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-local-observation-roll"
    ):
        try:
            pinned_open_dates = _market_open_dates_for_dispatch(
                args.market_open_days
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
        _print_json(
            run_macro_local_breadth_roll(
                snapshot_manifest_path=args.snapshot_manifest,
                expected_snapshot_manifest_sha256=(
                    args.expected_snapshot_manifest_sha256
                ),
                coverage_manifest_path=args.coverage_manifest,
                expected_coverage_manifest_sha256=(
                    args.expected_coverage_manifest_sha256
                ),
                target_trade_date=args.target_trade_date,
                scope_artifact_path=args.scope_artifact,
                expected_scope_artifact_sha256=(
                    args.expected_scope_artifact_sha256
                ),
                target_as_of=args.target_as_of,
                decision_cutoff_at=args.decision_cutoff_at,
                pinned_open_dates=pinned_open_dates,
                market_open_days_path=args.market_open_days,
                expected_market_open_days_sha256=(
                    args.expected_market_open_days_sha256
                ),
                canonical_observations_root=args.observations_root,
                run_id=args.run_id,
                expected_pointer_sha256=(
                    ""
                    if args.expected_pointer_sha256 == "EMPTY"
                    else args.expected_pointer_sha256
                ),
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-backfill-publish"
    ):
        _print_json(
            run_macro_backfill_publish(
                manifest_path=args.manifest,
                observations_root=args.observations_root,
                run_id=args.run_id,
                expected_pointer_sha256=(
                    ""
                    if args.expected_pointer_sha256 == "EMPTY"
                    else args.expected_pointer_sha256
                ),
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_plan_sha256=args.expected_plan_sha256,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-observe-forward"
    ):
        _print_json(
            run_macro_forward_observation(
                market=args.market,
                observations_root=args.observations_root,
                calendar_path=args.calendar,
                root=args.state_root,
                expected_pointer_sha256=(
                    ""
                    if args.expected_pointer_sha256 == "EMPTY"
                    else args.expected_pointer_sha256
                ),
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "macro-coverage-audit"
    ):
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

    if (
        args.command == "market"
        and args.market_command == "macro-acquisition-plan"
    ):
        _print_json(
            run_macro_acquisition(
                market=args.market,
                coverage_audit=args.coverage_audit,
                output_root=args.output_dir,
            )
        )
        return

    if args.command == "market" and args.market_command == "data-governance":
        result = run_data_governance(
            market=args.market,
            categories=args.categories or ["full_a"],
            as_of=args.as_of,
            output_dir=args.output_dir,
            allow_live=args.allow_live,
            allow_public_fallback=args.allow_public_fallback,
        )
        _print_json(result)
        if result.get("status") == "blocked":
            raise SystemExit(2)
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

    if args.command == "market" and args.market_command == "fundamental-research-prepare":
        result = run_fundamental_research_prepare(
            market=args.market,
            as_of=args.as_of,
            analysis_run=args.analysis_run,
            holdings_manifest=args.holdings_manifest,
            root=args.root,
            prompt_version=args.prompt_version,
            policy_version=args.policy_version,
        )
        _print_json(result.model_dump(mode="json"))
        return

    if args.command == "market" and args.market_command == "fundamental-research-import":
        _print_json(
            run_fundamental_research_import(
                request_path=args.request,
                response_path=args.response,
                root=args.root,
                validate_only=args.validate_only,
            )
        )
        return

    if args.command == "market" and args.market_command == "fundamental-research-status":
        _print_json(
            run_fundamental_research_status(
                market=args.market,
                root=args.root,
                run_id=args.run_id,
                symbol=args.symbol,
                state=args.state,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "fundamental-research-gate-evidence"
    ):
        _print_json(
            run_fundamental_research_gate_evidence(
                holdings_manifest=args.holdings_manifest,
                root=args.root,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "fundamental-research-longitudinal-import"
    ):
        _print_json(
            run_fundamental_research_longitudinal_import(
                root=args.root,
                observation_path=args.observation,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "fundamental-research-target-weight-produce"
    ):
        _print_json(
            run_fundamental_research_target_weight_produce(
                root=args.root,
                request_path=args.request,
                dossier_id=args.dossier_id,
                actual_analysis_manifest=args.actual_analysis,
                counterfactual_analysis_manifest=args.counterfactual_analysis,
            )
        )
        return

    if (
        args.command == "market"
        and args.market_command == "fundamental-research-nav-produce"
    ):
        _print_json(
            run_fundamental_research_nav_produce(
                root=args.root,
                target_weight_observation=args.target_weight_observation,
                attribution_date=args.attribution_date,
                data_root=args.data_root,
            )
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
            decision_protocol=args.decision_protocol,
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
            decision_protocol=args.decision_protocol,
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
