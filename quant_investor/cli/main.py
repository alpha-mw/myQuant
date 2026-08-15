"""
单一主线 CLI 入口。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re

from quant_investor.pipeline import QuantInvestor


def run_portfolio_cycle_status(**kwargs):
    """Derive the read-only Phase 1 portfolio-cycle readiness diagnostic."""

    from quant_investor.portfolio_cycle.readiness import (
        derive_decision_input_readiness,
    )

    values = dict(kwargs)
    historical_label = values.pop("historical_label")
    return derive_decision_input_readiness(
        expected_historical_label=historical_label,
        synthetic_only=False,
        **values,
    )


def run_download(**kwargs):
    from quant_investor.market.download import run_download as _run_download

    return _run_download(**kwargs)


def run_market_maintenance(**kwargs):
    from quant_investor.market.download import (
        run_market_maintenance as _run_market_maintenance,
    )

    return _run_market_maintenance(**kwargs)


def run_market_analysis(**kwargs):
    return _read_v17_public_run(kwargs, surface="market analyze")


def run_market_pipeline(**kwargs):
    return _read_v17_public_run(kwargs, surface="market run")


def run_market_backtest(**kwargs):
    del kwargs
    from quant_investor.v17_mainline import V17MainlineError

    raise V17MainlineError("V17_BACKTEST_UNAVAILABLE")


def _read_v17_public_run(kwargs: dict, *, surface: str):
    from quant_investor.v17_mainline import V17MainlineError, read_public_run

    values = dict(kwargs)
    if "market" in values and str(values["market"]).upper() != "CN":
        raise V17MainlineError("V17_MARKET_UNSUPPORTED")
    retired = sorted(
        key
        for key in values
        if key not in {"workspace_root", "strategy_id"}
    )
    if retired:
        raise V17MainlineError(
            "V17_PUBLIC_ARGUMENTS_UNSUPPORTED",
            detail=f"{surface}: {', '.join(retired)}",
        )
    return read_public_run(
        Path(values.pop("workspace_root", ".")),
        strategy_id=values.pop("strategy_id", ""),
    )


def run_fundamental_maintenance(**kwargs):
    from quant_investor.market.fundamental_mart import (
        run_cn_fundamental_maintenance as _run_cn_fundamental_maintenance,
    )

    return _run_cn_fundamental_maintenance(**kwargs)


def run_fundamental_promotion(**kwargs):
    values = dict(kwargs)
    safe_successor = bool(values.pop("safe_incremental_successor", False))
    recover = bool(values.pop("recover", False))
    execute = bool(values.pop("execute", False))
    journal_root = values.pop("journal_root", None)
    journal_run_id = values.pop("journal_run_id", None)
    if safe_successor:
        from quant_investor.market.fundamental_successor_promotion import (
            promote_successor_generation,
            recover_successor_promotion,
        )

        if recover:
            return recover_successor_promotion(
                canonical_root=values["canonical_root"],
                journal_root=journal_root,
                journal_run_id=journal_run_id,
                execute=execute,
            )
        return promote_successor_generation(
            staging_root=values["staging_root"],
            canonical_root=values["canonical_root"],
            expected_pointer_sha256=values["expected_pointer_sha256"],
            execute=execute,
            journal_root=journal_root,
            journal_run_id=journal_run_id,
        )
    if recover or execute or journal_root or journal_run_id:
        raise ValueError("safe-successor promotion flags require safe mode")
    from quant_investor.market.fundamental_generation import (
        promote_staged_fundamental_generation,
    )

    return promote_staged_fundamental_generation(**values)


def run_macro_maintenance(**kwargs):
    from quant_investor.macro.maintenance import run_cn_macro_maintenance

    return run_cn_macro_maintenance(**kwargs)


def run_storage_validate(**kwargs):
    from quant_investor.market.market_data_store import (
        run_storage_validate as _run_storage_validate,
    )

    return _run_storage_validate(**kwargs)


def run_storage_reactivate_snapshot(**kwargs):
    from quant_investor.market.market_data_store import (
        run_storage_reactivate_snapshot as _run_storage_reactivate_snapshot,
    )

    return _run_storage_reactivate_snapshot(**kwargs)


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


def _workspace_relative_canonical_path(value: str) -> str:
    """Accept one canonical POSIX path relative to the selected workspace."""

    text = str(value)
    candidate = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or candidate.is_absolute()
        or text != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise argparse.ArgumentTypeError(
            "expected a canonical workspace-relative POSIX path"
        )
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise argparse.ArgumentTypeError(
            "expected an ASCII workspace-relative POSIX path"
        ) from exc
    return text


def _sha256_argument(value: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise argparse.ArgumentTypeError(
            "expected a lowercase 64-character SHA-256"
        )
    return text


def _decision_cutoff_argument(value: str) -> str:
    text = str(value)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "decision cutoff must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise argparse.ArgumentTypeError(
            "decision cutoff must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"
        )
    return text


def _historical_label_argument(value: str) -> str:
    text = str(value)
    if not text or text != text.strip() or len(text) > 160 or "\x00" in text:
        raise argparse.ArgumentTypeError(
            "historical label must be canonical text"
        )
    return text


def _add_v17_public_read_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="V17 mainline workspace root",
    )
    parser.add_argument(
        "--strategy-id",
        required=True,
        help="canonical V17 strategy id",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor",
        description="Quant-Investor 单一主线 CLI。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    research_parser = subparsers.add_parser("research", help="只读 V17 主线研究结果")
    research_subparsers = research_parser.add_subparsers(
        dest="research_command",
        required=True,
    )
    research_run = research_subparsers.add_parser("run", help="读取 V17 活动主线")
    _add_v17_public_read_arguments(research_run)

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
    market_maintain.add_argument(
        "--secondary-daily-source",
        choices=["none", "eastmoney"],
        default="none",
        help=(
            "仅显式开启 CN parquet-direct 的 exact-date secondary bar probe；"
            "默认 none，不能用于 staged 或停牌分类。"
        ),
    )
    market_maintain.add_argument(
        "--official-suspension-evidence",
        default="",
        help=(
            "显式使用已完整 readback 的网页停牌证据包；仅作 exact-date 缺失分类，"
            "不生成 synthetic bar，也不改写 suspend_d v5 cache。"
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
    market_fundamental.add_argument(
        "--safe-incremental-successor",
        action="store_true",
        help="冻结健康 predecessor，仅在隔离 staging 构造 append-only successor",
    )
    market_fundamental.add_argument("--canonical-predecessor-root", default="")
    market_fundamental.add_argument("--expected-pointer-sha256", default="")
    market_fundamental.add_argument("--canonical-scope-path", default="")
    market_fundamental.add_argument("--canonical-market-pointer-path", default="")
    market_fundamental.add_argument("--canonical-pit-pointer-path", default="")
    market_fundamental.add_argument("--canonical-membership-path", default="")
    market_fundamental.add_argument("--history-audit-path", default="")
    market_fundamental.add_argument(
        "--expected-history-audit-sha256",
        default="",
    )
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
    market_fundamental_promote.add_argument("--staging-root", default="")
    market_fundamental_promote.add_argument(
        "--canonical-root",
        default="data/parquet/cn",
    )
    market_fundamental_promote.add_argument(
        "--expected-pointer-sha256",
        default="",
    )
    market_fundamental_promote.add_argument(
        "--safe-incremental-successor",
        action="store_true",
        help="对 safe successor 执行只读 preflight 或显式 journaled CAS",
    )
    market_fundamental_promote.add_argument(
        "--execute",
        action="store_true",
        help="仅在 safe successor 模式执行 canonical CAS；省略时严格只读",
    )
    market_fundamental_promote.add_argument(
        "--recover",
        action="store_true",
        help="检查或恢复指定 safe-successor promotion journal",
    )
    market_fundamental_promote.add_argument("--journal-root", default="")
    market_fundamental_promote.add_argument("--journal-run-id", default="")

    market_macro = market_subparsers.add_parser(
        "macro-maintain",
        help="维护 CN Macro observations 与 official release calendar",
    )
    market_macro.add_argument("--market", required=True, choices=["CN"])
    market_macro.add_argument("--target-date", required=True)
    market_macro.add_argument("--snapshot-manifest-path", required=True)
    market_macro.add_argument("--expected-snapshot-manifest-sha256", required=True)
    market_macro.add_argument("--coverage-manifest-path", required=True)
    market_macro.add_argument("--expected-coverage-manifest-sha256", required=True)
    market_macro.add_argument("--scope-artifact-path", required=True)
    market_macro.add_argument("--expected-scope-artifact-sha256", required=True)
    market_macro.add_argument(
        "--release-root", default="data/parquet/cn/macro_release_calendar"
    )
    market_macro.add_argument("--expected-release-pointer-sha256", required=True)
    market_macro.add_argument(
        "--observations-root", default="data/parquet/cn/macro_observations"
    )
    market_macro.add_argument("--expected-observations-pointer-sha256", required=True)
    market_macro.add_argument("--release-run-id", required=True)
    market_macro.add_argument("--observations-run-id", required=True)
    market_macro.add_argument("--allow-live", action="store_true")
    market_macro.add_argument("--commit", action="store_true")

    market_storage_validate = market_subparsers.add_parser(
        "storage-validate",
        help="校验本地 Parquet canonical snapshot 健康状态",
    )
    market_storage_validate.add_argument("--market", required=True, choices=["CN"])

    market_storage_reactivate_snapshot = market_subparsers.add_parser(
        "storage-reactivate-snapshot",
        help="以 SHA/CAS 绑定恢复已存在的 immutable CN snapshot；默认只做 dry-run",
    )
    market_storage_reactivate_snapshot.add_argument(
        "--market", required=True, choices=["CN"]
    )
    market_storage_reactivate_snapshot.add_argument("--snapshot-id", required=True)
    market_storage_reactivate_snapshot.add_argument(
        "--expected-snapshot-manifest-sha256", required=True
    )
    market_storage_reactivate_snapshot.add_argument(
        "--expected-market-pointer-sha256", required=True
    )
    market_storage_reactivate_snapshot.add_argument(
        "--acknowledge-trade-date", required=True
    )
    market_storage_reactivate_snapshot.add_argument("--reason", required=True)
    market_storage_reactivate_snapshot.add_argument(
        "--commit",
        action="store_true",
        help="显式提交恢复 pointer；未指定时不写 canonical 状态",
    )
    market_storage_reactivate_snapshot.add_argument(
        "--data-root",
        default="",
        help="可选 market data root；默认使用仓库 data",
    )

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

    market_analyze = market_subparsers.add_parser("analyze", help="读取 V17 活动主线")
    _add_v17_public_read_arguments(market_analyze)

    market_run = market_subparsers.add_parser("run", help="读取 V17 活动主线")
    _add_v17_public_read_arguments(market_run)

    market_backtest = market_subparsers.add_parser(
        "backtest",
        help="V17 回测不可用（固定 fail closed）",
    )

    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="只读组合闭环诊断",
    )
    portfolio_subparsers = portfolio_parser.add_subparsers(
        dest="portfolio_command",
        required=True,
    )
    portfolio_cycle_status = portfolio_subparsers.add_parser(
        "cycle-status",
        help="读取 Phase 1 持仓与决策输入 readiness；不生产、不发布、不写入",
    )
    portfolio_cycle_status.add_argument(
        "--workspace-root",
        required=True,
        help="已存在的 myQuant workspace root；命令不会创建目录",
    )
    portfolio_cycle_status.add_argument(
        "--strategy-id",
        default=None,
        help="canonical V17 strategy id；缺失时报告 blocker，不推断历史标签映射",
    )
    portfolio_cycle_status.add_argument(
        "--historical-label",
        type=_historical_label_argument,
        required=True,
        help="仅用于 identity declaration exact equality；不参与策略选择",
    )
    portfolio_cycle_status.add_argument(
        "--identity-path",
        type=_workspace_relative_canonical_path,
        default=None,
        help="workspace-relative canonical identity declaration path",
    )
    portfolio_cycle_status.add_argument(
        "--identity-sha256",
        type=_sha256_argument,
        default=None,
        help="identity declaration exact-byte SHA-256；必须与 --identity-path 成对",
    )
    portfolio_cycle_status.add_argument(
        "--holdings-pointer-path",
        type=_workspace_relative_canonical_path,
        default=None,
        help="workspace-relative canonical holdings pointer path",
    )
    portfolio_cycle_status.add_argument(
        "--holdings-pointer-sha256",
        type=_sha256_argument,
        default=None,
        help="holdings pointer exact-byte SHA-256；必须与 path 成对",
    )
    portfolio_cycle_status.add_argument(
        "--decision-cutoff",
        type=_decision_cutoff_argument,
        required=True,
        help="统一 decision cutoff（canonical UTC YYYY-MM-DDTHH:MM:SSZ）",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "research" and args.research_command == "run":
        investor = QuantInvestor(
            workspace_root=args.workspace_root,
            strategy_id=args.strategy_id,
        )
        _print_json(investor.run())
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
            secondary_daily_source=args.secondary_daily_source,
            official_suspension_evidence=args.official_suspension_evidence,
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
        if args.authoritative_full_rebuild and args.safe_incremental_successor:
            parser.error(
                "--authoritative-full-rebuild and --safe-incremental-successor "
                "are mutually exclusive"
            )
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
        if args.safe_incremental_successor:
            required_successor_args = {
                "--run-id": args.run_id,
                "--as-of": args.as_of,
                "--canonical-predecessor-root": args.canonical_predecessor_root,
                "--expected-pointer-sha256": args.expected_pointer_sha256,
                "--canonical-scope-path": args.canonical_scope_path,
                "--canonical-market-pointer-path": (
                    args.canonical_market_pointer_path
                ),
                "--canonical-pit-pointer-path": args.canonical_pit_pointer_path,
                "--canonical-membership-path": args.canonical_membership_path,
                "--history-audit-path": args.history_audit_path,
                "--expected-history-audit-sha256": (
                    args.expected_history_audit_sha256
                ),
                "--checkpoint-root": args.checkpoint_root,
            }
            missing_successor_args = sorted(
                name for name, value in required_successor_args.items() if not value
            )
            if not args.allow_live:
                missing_successor_args.append("--allow-live")
            if [item.strip().lower() for item in args.universes.split(",") if item.strip()] != [
                "full_a"
            ]:
                parser.error(
                    "--safe-incremental-successor requires --universes full_a"
                )
            if missing_successor_args:
                parser.error(
                    "--safe-incremental-successor requires "
                    + ", ".join(missing_successor_args)
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
            safe_incremental_successor=args.safe_incremental_successor,
            canonical_predecessor_root=(
                args.canonical_predecessor_root or None
            ),
            expected_pointer_sha256=args.expected_pointer_sha256,
            canonical_scope_path=args.canonical_scope_path or None,
            canonical_market_pointer_path=(
                args.canonical_market_pointer_path or None
            ),
            canonical_pit_pointer_path=(
                args.canonical_pit_pointer_path or None
            ),
            canonical_membership_path=args.canonical_membership_path or None,
            history_audit_path=args.history_audit_path or None,
            expected_history_audit_sha256=(
                args.expected_history_audit_sha256
            ),
            checkpoint_root=args.checkpoint_root or None,
            checkpoint_batch_size=args.checkpoint_batch_size,
            max_attempts=args.max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            max_retry_backoff_seconds=args.max_retry_backoff_seconds,
            requests_per_second=args.requests_per_second,
        )
        if args.authoritative_full_rebuild or args.safe_incremental_successor:
            _print_json(result)
        return

    if args.command == "market" and args.market_command == "fundamental-promote":
        if not args.safe_incremental_successor:
            if args.recover or args.execute or args.journal_root or args.journal_run_id:
                parser.error(
                    "--recover, --execute, and journal flags require "
                    "--safe-incremental-successor"
                )
            if not args.staging_root or not args.expected_pointer_sha256:
                parser.error(
                    "fundamental-promote requires --staging-root and "
                    "--expected-pointer-sha256"
                )
        elif args.recover:
            if not args.journal_root or not args.journal_run_id:
                parser.error(
                    "safe successor recovery requires --journal-root and "
                    "--journal-run-id"
                )
        else:
            if not args.staging_root or not args.expected_pointer_sha256:
                parser.error(
                    "safe successor promotion requires --staging-root and "
                    "--expected-pointer-sha256"
                )
            if args.execute and (not args.journal_root or not args.journal_run_id):
                parser.error(
                    "safe successor --execute requires --journal-root and "
                    "--journal-run-id"
                )
        _print_json(
            run_fundamental_promotion(
                staging_root=args.staging_root,
                canonical_root=args.canonical_root,
                expected_pointer_sha256=args.expected_pointer_sha256,
                safe_incremental_successor=args.safe_incremental_successor,
                recover=args.recover,
                execute=args.execute,
                journal_root=args.journal_root or None,
                journal_run_id=args.journal_run_id or None,
            )
        )
        return

    if args.command == "market" and args.market_command == "macro-maintain":
        result = run_macro_maintenance(
            market=args.market,
            target_date=args.target_date,
            snapshot_manifest_path=args.snapshot_manifest_path,
            expected_snapshot_manifest_sha256=args.expected_snapshot_manifest_sha256,
            coverage_manifest_path=args.coverage_manifest_path,
            expected_coverage_manifest_sha256=args.expected_coverage_manifest_sha256,
            scope_artifact_path=args.scope_artifact_path,
            expected_scope_artifact_sha256=args.expected_scope_artifact_sha256,
            release_root=args.release_root,
            expected_release_pointer_sha256=args.expected_release_pointer_sha256,
            observations_root=args.observations_root,
            expected_observations_pointer_sha256=args.expected_observations_pointer_sha256,
            release_run_id=args.release_run_id,
            observations_run_id=args.observations_run_id,
            allow_live=args.allow_live,
            commit=args.commit,
        )
        _print_json(result)
        if str(result.get("status")) not in {"OK", "DRY_RUN_OK"}:
            raise SystemExit(2)
        return

    if args.command == "market" and args.market_command == "storage-validate":
        _print_json(run_storage_validate(market=args.market))
        return

    if args.command == "market" and args.market_command == "storage-reactivate-snapshot":
        _print_json(
            run_storage_reactivate_snapshot(
                market=args.market,
                snapshot_id=args.snapshot_id,
                expected_snapshot_manifest_sha256=(
                    args.expected_snapshot_manifest_sha256
                ),
                expected_market_pointer_sha256=args.expected_market_pointer_sha256,
                acknowledge_trade_date=args.acknowledge_trade_date,
                reason=args.reason,
                commit=args.commit,
                data_root=args.data_root or None,
            )
        )
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
        result = run_market_analysis(
            workspace_root=args.workspace_root,
            strategy_id=args.strategy_id,
        )
        _print_json(result)
        return

    if args.command == "market" and args.market_command == "run":
        result = run_market_pipeline(
            workspace_root=args.workspace_root,
            strategy_id=args.strategy_id,
        )
        _print_json(result)
        return

    if args.command == "market" and args.market_command == "backtest":
        run_market_backtest()

    if args.command == "portfolio" and args.portfolio_command == "cycle-status":
        if (args.identity_path is None) != (args.identity_sha256 is None):
            parser.error(
                "--identity-path and --identity-sha256 must be provided together"
            )
        if (args.holdings_pointer_path is None) != (
            args.holdings_pointer_sha256 is None
        ):
            parser.error(
                "--holdings-pointer-path and --holdings-pointer-sha256 "
                "must be provided together"
            )
        try:
            result = run_portfolio_cycle_status(
                workspace_root=Path(args.workspace_root),
                strategy_id=args.strategy_id,
                historical_label=args.historical_label,
                identity_path=args.identity_path,
                identity_sha256=args.identity_sha256,
                holdings_pointer_path=args.holdings_pointer_path,
                holdings_pointer_sha256=args.holdings_pointer_sha256,
                decision_cutoff=args.decision_cutoff,
            )
        except Exception as exc:  # command boundary: never expose a traceback
            error_code = getattr(
                exc, "code", "PORTFOLIO_CYCLE_INTERNAL_ERROR"
            )
            error_detail = getattr(exc, "detail", None)
            if error_detail is None:
                error_detail = "internal portfolio-cycle diagnostic failure"
            _print_json(
                {
                    "schema_id": "myquant.v17.v4.portfolio-cycle-cli-error.v1",
                    "status": "ERROR",
                    "error": {
                        "code": error_code,
                        "detail": error_detail,
                    },
                    "operational_authority": False,
                    "write_performed": False,
                }
            )
            raise SystemExit(1) from None
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        _print_json(payload)
        if str(payload.get("state")) == "BLOCKED":
            raise SystemExit(2)
        return
