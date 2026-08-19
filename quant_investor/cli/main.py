"""
单一主线 CLI 入口。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path, PurePosixPath
import re

from quant_investor.cli.output import (
    MachineArgumentParser,
    command_boundary,
    emit_json,
)
from quant_investor.cli.unified import (
    factor_evaluate,
    factor_history,
    factor_mine,
    factor_observe,
    factor_production_activate,
    factor_production_signal,
    factor_production_status,
    factor_production_verify,
    factor_status,
    research_compile_evidence,
    research_evaluate,
    research_forward,
    research_inspect,
    research_readiness,
    system_activate,
    system_assemble,
    system_bootstrap_assemble,
    system_bootstrap_admission_preflight,
    system_calendar_capture,
    system_status,
    system_suspend,
    system_verify,
)


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
    return _read_public_run(kwargs, surface="market analyze")


def run_market_pipeline(**kwargs):
    return _read_public_run(kwargs, surface="market run")


def run_market_backtest(**kwargs):
    del kwargs
    from quant_investor.mainline import BACKTEST_UNAVAILABLE, MainlineError

    raise MainlineError(BACKTEST_UNAVAILABLE, blockers=[BACKTEST_UNAVAILABLE])


def _read_public_run(kwargs: dict, *, surface: str):
    from quant_investor.mainline import MAINLINE_ARGUMENTS_INVALID, MainlineError, read_public_run

    values = dict(kwargs)
    if "market" in values and str(values["market"]).upper() != "CN":
        raise MainlineError(
            MAINLINE_ARGUMENTS_INVALID,
            blockers=["MARKET_UNSUPPORTED"],
        )
    retired = sorted(key for key in values if key not in {"workspace_root", "strategy_id"})
    if retired:
        raise MainlineError(
            MAINLINE_ARGUMENTS_INVALID,
            blockers=["PUBLIC_ARGUMENTS_UNSUPPORTED"],
        )
    return read_public_run(
        Path(values.pop("workspace_root", ".")),
        strategy_id=values.pop("strategy_id", ""),
    )


def _read_cli_public_run(*, workspace_root: str, strategy_id: str) -> dict:
    """Emit the exact six-field unavailable state before exiting with code 2."""

    from quant_investor.mainline import MainlineStore

    store = MainlineStore(Path(workspace_root))
    state = store.status(strategy_id=strategy_id)
    if state.get("status") != "ACTIVE":
        _print_json(state)
        raise SystemExit(2)
    return store.read_public_run(strategy_id=strategy_id)


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
    from quant_investor.macro.maintenance import (
        commit_prepared_macro_transaction,
        prepare_cn_macro_maintenance_transaction,
        recover_macro_transaction,
        rollback_macro_transaction,
        run_cn_macro_maintenance,
    )

    values = dict(kwargs)
    prepare = bool(values.pop("prepare_transaction", False))
    commit_prepared = bool(values.pop("commit_prepared", False))
    legacy_commit = bool(values.pop("commit", False))
    recover = bool(values.pop("recover", False))
    execute_forward = bool(values.pop("execute_forward", False))
    execute_rollback = bool(values.pop("execute_rollback", False))
    journal_root = values.pop("journal_root", "")
    journal_run_id = values.pop("journal_run_id", "")
    prepared_path = values.pop("prepared_path", "")
    expected_prepared_sha256 = values.pop("expected_prepared_sha256", "")
    authority_mode = values.pop("authority_mode", "")
    market_pointer_path = values.pop("market_pointer_path", "")
    expected_market_pointer_sha256 = values.pop("expected_market_pointer_sha256", "")
    pit_pointer_path = values.pop("pit_pointer_path", "")
    expected_pit_pointer_sha256 = values.pop("expected_pit_pointer_sha256", "")
    rollback_shas = {
        name: values.pop(name, "")
        for name in (
            "old_release_pointer_sha256",
            "new_release_pointer_sha256",
            "old_observations_pointer_sha256",
            "new_observations_pointer_sha256",
        )
    }
    if prepare:
        journal_base = Path(journal_root).expanduser()
        journal_base.mkdir(parents=True, exist_ok=True, mode=0o700)
        journal_base.chmod(0o700)
        preparation_root = journal_base / "_prepared"
        preparation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        preparation_root.chmod(0o700)
        return prepare_cn_macro_maintenance_transaction(
            **values,
            private_run_root=preparation_root,
            transaction_run_id=journal_run_id,
            market_pointer_path=market_pointer_path,
            expected_market_pointer_sha256=(expected_market_pointer_sha256),
            pit_pointer_path=pit_pointer_path,
            expected_pit_pointer_sha256=expected_pit_pointer_sha256,
            authority_mode=authority_mode,
        )
    if commit_prepared:
        journal_base = Path(journal_root).expanduser()
        journal_base.mkdir(parents=True, exist_ok=True, mode=0o700)
        journal_base.chmod(0o700)
        return commit_prepared_macro_transaction(
            prepared_path=prepared_path,
            expected_prepared_sha256=expected_prepared_sha256,
            journal_root=journal_base,
            journal_run_id=journal_run_id,
            market_pointer_path=market_pointer_path,
            expected_market_pointer_sha256=(expected_market_pointer_sha256),
            pit_pointer_path=pit_pointer_path,
            expected_pit_pointer_sha256=expected_pit_pointer_sha256,
        )
    if recover and execute_rollback:
        return rollback_macro_transaction(
            journal_root=journal_root,
            journal_run_id=journal_run_id,
            market_pointer_path=market_pointer_path,
            expected_market_pointer_sha256=(expected_market_pointer_sha256),
            pit_pointer_path=pit_pointer_path,
            expected_pit_pointer_sha256=expected_pit_pointer_sha256,
            **rollback_shas,
        )
    if recover:
        return recover_macro_transaction(
            journal_root=journal_root,
            journal_run_id=journal_run_id,
            market_pointer_path=market_pointer_path,
            expected_market_pointer_sha256=(expected_market_pointer_sha256),
            pit_pointer_path=pit_pointer_path,
            expected_pit_pointer_sha256=expected_pit_pointer_sha256,
            execute_forward=execute_forward,
        )
    return run_cn_macro_maintenance(**values, commit=legacy_commit)


def run_cn_daily_maintenance(**kwargs):
    from quant_investor.market.daily_maintenance import (
        run_cn_daily_maintenance as _run_cn_daily_maintenance,
    )

    return _run_cn_daily_maintenance(**kwargs)


def clear_cn_daily_write_veto(**kwargs):
    from quant_investor.market.daily_maintenance import (
        clear_cn_daily_write_veto as _clear_cn_daily_write_veto,
    )

    return _clear_cn_daily_write_veto(**kwargs)


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
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if type(payload) is not dict:
        raise TypeError("public CLI responses must be JSON objects")
    emit_json(payload)


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
        raise argparse.ArgumentTypeError("expected a canonical workspace-relative POSIX path")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise argparse.ArgumentTypeError("expected an ASCII workspace-relative POSIX path") from exc
    return text


def _canonical_absolute_path(value: str) -> str:
    """Accept one normalized absolute POSIX path without resolving symlinks."""

    text = str(value)
    candidate = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or not candidate.is_absolute()
        or text != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts[1:])
    ):
        raise argparse.ArgumentTypeError("expected a canonical absolute POSIX path")
    return text


def _sha256_argument(value: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return text


def _pointer_sha_argument(value: str) -> str:
    text = str(value)
    if text == "EMPTY":
        return text
    return _sha256_argument(text)


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
        raise argparse.ArgumentTypeError("historical label must be canonical text")
    return text


def _add_public_read_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="unified runtime workspace root",
    )
    parser.add_argument(
        "--strategy-id",
        required=True,
        help="canonical strategy id",
    )


def _add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="unified runtime workspace root",
    )


def _add_exact_request_arguments(parser: argparse.ArgumentParser) -> None:
    _add_workspace_argument(parser)
    parser.add_argument(
        "--request",
        required=True,
        type=_workspace_relative_canonical_path,
        help="canonical workspace-relative request path",
    )
    parser.add_argument(
        "--expected-request-sha256",
        required=True,
        type=_sha256_argument,
        help="exact canonical request byte SHA-256",
    )


def _add_deployed_release_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deployed-release-ref",
        type=_workspace_relative_canonical_path,
        default=None,
        help="optional exact object-ref JSON for the installed release",
    )
    parser.add_argument(
        "--expected-deployed-release-ref-sha256",
        type=_sha256_argument,
        default=None,
        help="exact byte SHA-256 paired with --deployed-release-ref",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = MachineArgumentParser(
        prog="quant-investor",
        description="Quant-Investor 单一主线 CLI。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    system_parser = subparsers.add_parser("system", help="统一系统状态与激活")
    system_subparsers = system_parser.add_subparsers(dest="system_command", required=True)
    system_status_parser = system_subparsers.add_parser(
        "status", help="读取活动统一 generation 状态"
    )
    _add_workspace_argument(system_status_parser)
    _add_deployed_release_arguments(system_status_parser)
    system_status_parser.add_argument(
        "--external-routing",
        type=_workspace_relative_canonical_path,
        default=None,
        help="optional exact external-routing observation JSON",
    )
    system_status_parser.add_argument(
        "--expected-external-routing-sha256",
        type=_sha256_argument,
        default=None,
    )

    system_verify_parser = system_subparsers.add_parser("verify", help="验证活动或显式 generation")
    _add_workspace_argument(system_verify_parser)
    system_verify_parser.add_argument("--generation", type=_sha256_argument, default=None)
    _add_deployed_release_arguments(system_verify_parser)

    system_assemble_parser = system_subparsers.add_parser(
        "assemble", help="从 sealed request 组装非活动 generation"
    )
    _add_exact_request_arguments(system_assemble_parser)

    system_bootstrap_assemble_parser = system_subparsers.add_parser(
        "bootstrap-assemble",
        help="从显式 strict source closure 组装首个非活动 generation",
    )
    _add_exact_request_arguments(system_bootstrap_assemble_parser)
    system_bootstrap_assemble_parser.add_argument(
        "--input-root",
        required=True,
        type=_workspace_relative_canonical_path,
        help="workspace-relative root containing every sealed input",
    )

    system_bootstrap_preflight_parser = system_subparsers.add_parser(
        "bootstrap-admission-preflight",
        help="验证 strict sources 并生成非授权 Fundamental veto subject",
    )
    _add_exact_request_arguments(system_bootstrap_preflight_parser)
    system_bootstrap_preflight_parser.add_argument(
        "--input-root",
        required=True,
        type=_workspace_relative_canonical_path,
        help="workspace-relative root containing every sealed input",
    )

    system_calendar_capture_parser = system_subparsers.add_parser(
        "calendar-capture",
        help="从 installed release 原子采集 Tushare 降级交易日历证据",
    )
    _add_workspace_argument(system_calendar_capture_parser)
    system_calendar_capture_parser.add_argument("--capture-parent", required=True)
    system_calendar_capture_parser.add_argument(
        "--release-repository-root",
        required=True,
        type=_canonical_absolute_path,
        help="exact clean detached release checkout used by installed capture",
    )
    system_calendar_capture_parser.add_argument("--capture-root-name", required=True)
    system_calendar_capture_parser.add_argument("--cutoff-date", required=True)
    system_calendar_capture_parser.add_argument(
        "--release-install-input",
        required=True,
        type=_workspace_relative_canonical_path,
    )
    system_calendar_capture_parser.add_argument(
        "--expected-release-install-input-sha256",
        required=True,
        type=_sha256_argument,
    )

    system_activate_parser = system_subparsers.add_parser(
        "activate", help="以 CAS 激活已验证 generation"
    )
    _add_workspace_argument(system_activate_parser)
    system_activate_parser.add_argument("--generation", required=True, type=_sha256_argument)
    system_activate_parser.add_argument(
        "--expect-pointer-sha", required=True, type=_pointer_sha_argument
    )
    for option, destination in (
        ("--migration-receipt", "migration_receipt"),
        ("--final-cutover-authorization", "final_cutover_authorization"),
        ("--activation-authorization", "activation_authorization"),
        ("--target-active-pointer", "target_active_pointer"),
        ("--deployed-release-ref", "deployed_release_ref"),
    ):
        system_activate_parser.add_argument(
            option,
            dest=destination,
            required=True,
            type=_workspace_relative_canonical_path,
        )
    for option, destination in (
        ("--expected-migration-receipt-sha256", "expected_migration_receipt_sha256"),
        (
            "--expected-final-cutover-authorization-sha256",
            "expected_final_cutover_authorization_sha256",
        ),
        (
            "--expected-activation-authorization-sha256",
            "expected_activation_authorization_sha256",
        ),
        (
            "--expected-target-active-pointer-sha256",
            "expected_target_active_pointer_sha256",
        ),
        (
            "--expected-deployed-release-ref-sha256",
            "expected_deployed_release_ref_sha256",
        ),
    ):
        system_activate_parser.add_argument(
            option, dest=destination, required=True, type=_sha256_argument
        )

    system_suspend_parser = system_subparsers.add_parser(
        "suspend", help="紧急 CAS 到预构建的最小挂起 generation"
    )
    _add_workspace_argument(system_suspend_parser)
    system_suspend_parser.add_argument("--generation", required=True, type=_sha256_argument)
    system_suspend_parser.add_argument(
        "--expect-pointer-sha", required=True, type=_pointer_sha_argument
    )
    system_suspend_parser.add_argument(
        "--target-active-pointer",
        required=True,
        type=_workspace_relative_canonical_path,
    )
    system_suspend_parser.add_argument(
        "--expected-target-active-pointer-sha256",
        required=True,
        type=_sha256_argument,
    )

    factor_parser = subparsers.add_parser("factor", help="统一 Factor governance")
    factor_subparsers = factor_parser.add_subparsers(dest="factor_command", required=True)
    factor_status_parser = factor_subparsers.add_parser(
        "status", help="从已存储验证闭包构建非授权 Factor 状态"
    )
    _add_exact_request_arguments(factor_status_parser)
    for name, help_text in (
        ("mine", "seal prospective preregistration"),
        ("observe", "seal initial selection or prospective observation"),
        ("evaluate", "evaluate or build an inactive admitted set"),
    ):
        candidate_parser = factor_subparsers.add_parser(name, help=help_text)
        _add_exact_request_arguments(candidate_parser)
    factor_history_parser = factor_subparsers.add_parser(
        "history", help="读取统一 generation 绑定的 Factor lineage"
    )
    _add_workspace_argument(factor_history_parser)
    for name, help_text in (
        ("production-status", "读取隔离 Factor 生产权威状态"),
        ("production-verify", "验证 Factor pointer、marker 与 generation 闭包"),
    ):
        production_read_parser = factor_subparsers.add_parser(name, help=help_text)
        _add_workspace_argument(production_read_parser)
    factor_signal_parser = factor_subparsers.add_parser(
        "production-signal", help="读取活动 generation 中封存的确定性 Factor signal"
    )
    _add_workspace_argument(factor_signal_parser)
    factor_signal_parser.add_argument(
        "--factor-id",
        required=True,
        choices=(
            "pv_low_dollar_volume_5d",
            "pv_blend_volstab19x2_mom90_amihud5_w80",
        ),
    )
    factor_activate_parser = factor_subparsers.add_parser(
        "production-activate",
        help="从严格 source closure 执行唯一 expected-EMPTY Factor 首次激活",
    )
    _add_workspace_argument(factor_activate_parser)
    factor_activate_parser.add_argument("--market-data-root", required=True)
    factor_activate_parser.add_argument("--calendar-capture-root", required=True)
    factor_activate_parser.add_argument(
        "--expected-calendar-success-sha256", required=True, type=_sha256_argument
    )
    factor_activate_parser.add_argument("--release-repository-root", required=True)
    factor_activate_parser.add_argument(
        "--activation-inputs",
        required=True,
        type=_workspace_relative_canonical_path,
    )
    factor_activate_parser.add_argument(
        "--expected-activation-inputs-sha256", required=True, type=_sha256_argument
    )
    factor_activate_parser.add_argument("--expected-empty", action="store_true", required=True)

    research_parser = subparsers.add_parser("research", help="统一主线研究能力")
    research_subparsers = research_parser.add_subparsers(
        dest="research_command",
        required=True,
    )
    research_run = research_subparsers.add_parser("run", help="读取活动统一主线")
    _add_public_read_arguments(research_run)
    for name, help_text in (
        ("forward", "seal inactive forward-research request"),
        ("evaluate", "evaluate precomputed research stages"),
        ("compile-evidence", "compile exact inactive evidence closure"),
        ("readiness", "assess generation-compatible readiness"),
        ("inspect", "inspect one stable artifact without mutation"),
    ):
        research_candidate = research_subparsers.add_parser(name, help=help_text)
        _add_exact_request_arguments(research_candidate)

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
    market_maintain.add_argument(
        "--fail-on-incomplete",
        nargs="?",
        const=True,
        default=False,
        type=_parse_boolish,
    )
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

    market_daily_maintain = market_subparsers.add_parser(
        "daily-maintain",
        help="收盘后执行可审计的 CN 日度维护协调器",
    )
    market_daily_maintain.add_argument("--market", required=True, choices=["CN"])
    _add_workspace_argument(market_daily_maintain)
    market_daily_maintain.add_argument(
        "--run-root",
        required=True,
        type=_canonical_absolute_path,
        help="owner-only private attempt evidence root",
    )
    market_daily_maintain.add_argument("--mode", required=True, choices=["shadow", "execute"])
    market_daily_maintain.add_argument(
        "--attempt-slot",
        default="auto",
        choices=["auto", "1620", "1720", "1820", "2020"],
    )

    market_clear_veto = market_subparsers.add_parser(
        "clear-write-veto",
        help="按 exact SHA 封存并清除 CN 日度维护 write veto",
    )
    market_clear_veto.add_argument("--market", required=True, choices=["CN"])
    market_clear_veto.add_argument("--run-root", required=True, type=_canonical_absolute_path)
    market_clear_veto.add_argument("--expected-veto-sha256", required=True, type=_sha256_argument)
    market_clear_veto.add_argument("--reason", required=True)

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
    market_fundamental.add_argument(
        "--append-first-successor",
        action="store_true",
        help=(
            "以 immutable predecessor 为历史边界，仅采集 (parent,target]；"
            "必须提供可重放的历史冲突证据，当前窗口冲突仍 fail closed"
        ),
    )
    market_fundamental.add_argument(
        "--historical-taint-failure-evidence",
        action="append",
        default=[],
        metavar="ABSOLUTE_FAILURE_ROOT#ORDINAL",
        help=(
            "append-first 所需的历史 failure evidence；可重复，"
            "例如 /private/tmp/run/capture-failures#3564"
        ),
    )
    market_fundamental.add_argument(
        "--successor-income-support",
        action="append",
        default=[],
        metavar="TS_CODE@YYYYMMDD",
        help=(
            "append-first 的精确上年同期 income 计算依赖；可重复，"
            "仅能作为推导支持，不能发布为历史前缀"
        ),
    )
    market_fundamental.add_argument(
        "--successor-financial-support",
        action="append",
        default=[],
        metavar="TABLE:TS_CODE@YYYYMMDD",
        help=(
            "append-first 的精确 financial fallback 依赖；TABLE 仅允许 "
            "income、balancesheet、cashflow，可重复"
        ),
    )
    market_fundamental.add_argument(
        "--taint-analysis-dry-run",
        action="store_true",
        help=(
            "调用 live provider 做 promotion-ineligible taint capture；"
            "不写 staging、canonical 或 promotion"
        ),
    )
    market_fundamental.add_argument("--audit-run-root", default="")
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
    market_macro.add_argument("--target-date", default="")
    market_macro.add_argument("--snapshot-manifest-path", default="")
    market_macro.add_argument("--expected-snapshot-manifest-sha256", default="")
    market_macro.add_argument("--coverage-manifest-path", default="")
    market_macro.add_argument("--expected-coverage-manifest-sha256", default="")
    market_macro.add_argument("--scope-artifact-path", default="")
    market_macro.add_argument("--expected-scope-artifact-sha256", default="")
    market_macro.add_argument("--release-root", default="data/parquet/cn/macro_release_calendar")
    market_macro.add_argument("--expected-release-pointer-sha256", default="")
    market_macro.add_argument("--observations-root", default="data/parquet/cn/macro_observations")
    market_macro.add_argument("--expected-observations-pointer-sha256", default="")
    market_macro.add_argument("--release-run-id", default="")
    market_macro.add_argument("--observations-run-id", default="")
    market_macro.add_argument("--allow-live", action="store_true")
    market_macro.add_argument("--commit", action="store_true")
    market_macro.add_argument("--prepare-transaction", action="store_true")
    market_macro.add_argument("--authority-mode", choices=["candidate", "canonical"], default="")
    market_macro.add_argument("--commit-prepared", action="store_true")
    market_macro.add_argument("--journal-root", default=None, type=_canonical_absolute_path)
    market_macro.add_argument("--journal-run-id", default="")
    market_macro.add_argument("--prepared-path", default=None, type=_canonical_absolute_path)
    market_macro.add_argument("--expected-prepared-sha256", default="")
    market_macro.add_argument("--market-pointer-path", default=None, type=_canonical_absolute_path)
    market_macro.add_argument(
        "--expected-market-pointer-sha256", default=None, type=_sha256_argument
    )
    market_macro.add_argument("--pit-pointer-path", default=None, type=_canonical_absolute_path)
    market_macro.add_argument("--expected-pit-pointer-sha256", default=None, type=_sha256_argument)
    market_macro.add_argument("--recover", action="store_true")
    market_macro.add_argument("--execute-forward", action="store_true")
    market_macro.add_argument("--execute-rollback", action="store_true")
    market_macro.add_argument(
        "--old-release-pointer-sha256", default=None, type=_pointer_sha_argument
    )
    market_macro.add_argument("--new-release-pointer-sha256", default=None, type=_sha256_argument)
    market_macro.add_argument(
        "--old-observations-pointer-sha256", default=None, type=_pointer_sha_argument
    )
    market_macro.add_argument(
        "--new-observations-pointer-sha256", default=None, type=_sha256_argument
    )

    market_storage_validate = market_subparsers.add_parser(
        "storage-validate",
        help="校验本地 Parquet canonical snapshot 健康状态",
    )
    market_storage_validate.add_argument("--market", required=True, choices=["CN"])

    market_storage_reactivate_snapshot = market_subparsers.add_parser(
        "storage-reactivate-snapshot",
        help="以 SHA/CAS 绑定恢复已存在的 immutable CN snapshot；默认只做 dry-run",
    )
    market_storage_reactivate_snapshot.add_argument("--market", required=True, choices=["CN"])
    market_storage_reactivate_snapshot.add_argument("--snapshot-id", required=True)
    market_storage_reactivate_snapshot.add_argument(
        "--expected-snapshot-manifest-sha256", required=True
    )
    market_storage_reactivate_snapshot.add_argument(
        "--expected-market-pointer-sha256", required=True
    )
    market_storage_reactivate_snapshot.add_argument("--acknowledge-trade-date", required=True)
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

    market_analyze = market_subparsers.add_parser("analyze", help="读取活动统一主线")
    _add_public_read_arguments(market_analyze)

    market_run = market_subparsers.add_parser("run", help="读取活动统一主线")
    _add_public_read_arguments(market_run)

    market_subparsers.add_parser(
        "backtest",
        help="正式回测不可用（固定 fail closed）",
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
        help="canonical strategy id；缺失时报告 blocker，不推断历史标签映射",
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


def _dispatch(argv: list[str] | None = None) -> None:  # noqa: C901
    """Route the explicit public command tree without dynamic dispatch."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "system" and args.system_command == "status":
        _print_json(
            system_status(
                workspace_root=args.workspace_root,
                deployed_release_ref_path=args.deployed_release_ref,
                expected_deployed_release_ref_sha256=(args.expected_deployed_release_ref_sha256),
                external_routing_path=args.external_routing,
                expected_external_routing_sha256=(args.expected_external_routing_sha256),
            )
        )
        return

    if args.command == "system" and args.system_command == "verify":
        _print_json(
            system_verify(
                workspace_root=args.workspace_root,
                generation_id=args.generation,
                deployed_release_ref_path=args.deployed_release_ref,
                expected_deployed_release_ref_sha256=(args.expected_deployed_release_ref_sha256),
            )
        )
        return

    if args.command == "system" and args.system_command == "assemble":
        _print_json(
            system_assemble(
                workspace_root=args.workspace_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "system" and args.system_command == "bootstrap-assemble":
        _print_json(
            system_bootstrap_assemble(
                workspace_root=args.workspace_root,
                input_root=args.input_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "system" and args.system_command == "bootstrap-admission-preflight":
        _print_json(
            system_bootstrap_admission_preflight(
                workspace_root=args.workspace_root,
                input_root=args.input_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "system" and args.system_command == "calendar-capture":
        _print_json(
            system_calendar_capture(
                workspace_root=args.workspace_root,
                capture_parent=args.capture_parent,
                capture_root_name=args.capture_root_name,
                cutoff_date=args.cutoff_date,
                release_repository_root=args.release_repository_root,
                release_install_input_path=args.release_install_input,
                expected_release_install_input_sha256=(args.expected_release_install_input_sha256),
            )
        )
        return

    if args.command == "system" and args.system_command == "activate":
        _print_json(
            system_activate(
                workspace_root=args.workspace_root,
                generation_id=args.generation,
                expected_pointer_sha256=args.expect_pointer_sha,
                migration_receipt_path=args.migration_receipt,
                expected_migration_receipt_sha256=(args.expected_migration_receipt_sha256),
                final_cutover_authorization_path=args.final_cutover_authorization,
                expected_final_cutover_authorization_sha256=(
                    args.expected_final_cutover_authorization_sha256
                ),
                activation_authorization_path=args.activation_authorization,
                expected_activation_authorization_sha256=(
                    args.expected_activation_authorization_sha256
                ),
                target_active_pointer_path=args.target_active_pointer,
                expected_target_active_pointer_sha256=(args.expected_target_active_pointer_sha256),
                deployed_release_ref_path=args.deployed_release_ref,
                expected_deployed_release_ref_sha256=(args.expected_deployed_release_ref_sha256),
            )
        )
        return

    if args.command == "system" and args.system_command == "suspend":
        _print_json(
            system_suspend(
                workspace_root=args.workspace_root,
                generation_id=args.generation,
                expected_pointer_sha256=args.expect_pointer_sha,
                target_active_pointer_path=args.target_active_pointer,
                expected_target_active_pointer_sha256=(args.expected_target_active_pointer_sha256),
            )
        )
        return

    if args.command == "factor" and args.factor_command == "status":
        _print_json(
            factor_status(
                workspace_root=args.workspace_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "factor" and args.factor_command == "production-status":
        _print_json(factor_production_status(workspace_root=args.workspace_root))
        return

    if args.command == "factor" and args.factor_command == "production-verify":
        _print_json(factor_production_verify(workspace_root=args.workspace_root))
        return

    if args.command == "factor" and args.factor_command == "production-signal":
        _print_json(
            factor_production_signal(
                workspace_root=args.workspace_root,
                factor_id=args.factor_id,
            )
        )
        return

    if args.command == "factor" and args.factor_command == "production-activate":
        _print_json(
            factor_production_activate(
                workspace_root=args.workspace_root,
                market_data_root=args.market_data_root,
                calendar_capture_root=args.calendar_capture_root,
                expected_calendar_success_sha256=args.expected_calendar_success_sha256,
                release_repository_root=args.release_repository_root,
                activation_inputs_path=args.activation_inputs,
                expected_activation_inputs_sha256=(args.expected_activation_inputs_sha256),
                expected_empty=args.expected_empty,
            )
        )
        return

    if args.command == "factor" and args.factor_command == "mine":
        _print_json(
            factor_mine(
                workspace_root=args.workspace_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "factor" and args.factor_command == "observe":
        _print_json(
            factor_observe(
                workspace_root=args.workspace_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "factor" and args.factor_command == "evaluate":
        _print_json(
            factor_evaluate(
                workspace_root=args.workspace_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "factor" and args.factor_command == "history":
        _print_json(factor_history(workspace_root=args.workspace_root))
        return

    if args.command == "research" and args.research_command == "run":
        _print_json(
            _read_cli_public_run(
                workspace_root=args.workspace_root,
                strategy_id=args.strategy_id,
            )
        )
        return

    research_handlers = {
        "forward": research_forward,
        "evaluate": research_evaluate,
        "compile-evidence": research_compile_evidence,
        "readiness": research_readiness,
        "inspect": research_inspect,
    }
    if args.command == "research" and args.research_command in research_handlers:
        _print_json(
            research_handlers[args.research_command](
                workspace_root=args.workspace_root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
            )
        )
        return

    if args.command == "market" and args.market_command == "maintain":
        maintenance_batch_size = (
            args.batch_size if args.batch_size is not None else (200 if args.staged else 50)
        )
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
            expected_pit_generation_manifest_sha256=(args.expected_pit_generation_manifest_sha256),
            expected_market_pointer_sha256=(args.expected_market_pointer_sha256),
            secondary_daily_source=args.secondary_daily_source,
            official_suspension_evidence=args.official_suspension_evidence,
        )
        return

    if args.command == "market" and args.market_command == "daily-maintain":
        from quant_investor.market.daily_maintenance import cli_exit_required

        result = run_cn_daily_maintenance(
            workspace_root=args.workspace_root,
            run_root=args.run_root,
            mode=args.mode,
            attempt_slot=args.attempt_slot,
        )
        _print_json(result)
        if cli_exit_required(result):
            raise SystemExit(2)
        return

    if args.command == "market" and args.market_command == "clear-write-veto":
        _print_json(
            clear_cn_daily_write_veto(
                run_root=args.run_root,
                expected_veto_sha256=args.expected_veto_sha256,
                reason=args.reason,
            )
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
            expected_pit_generation_manifest_sha256=(args.expected_pit_generation_manifest_sha256),
            expected_market_pointer_sha256=(args.expected_market_pointer_sha256),
        )
        return

    if args.command == "market" and args.market_command == "fundamental-maintain":
        historical_taint_evidence = []
        for raw_evidence in args.historical_taint_failure_evidence:
            root_text, separator, ordinal_text = str(raw_evidence).rpartition("#")
            if (
                not separator
                or not root_text
                or not Path(root_text).expanduser().is_absolute()
                or not ordinal_text.isdigit()
            ):
                parser.error(
                    "--historical-taint-failure-evidence must be " "ABSOLUTE_FAILURE_ROOT#ORDINAL"
                )
            historical_taint_evidence.append(
                {
                    "failure_root": str(Path(root_text).expanduser()),
                    "ordinal": int(ordinal_text),
                }
            )
        income_support_dependencies = []
        for raw_dependency in args.successor_income_support:
            symbol_text, separator, period_text = str(raw_dependency).partition("@")
            symbol = symbol_text.strip().upper()
            if (
                not separator
                or re.fullmatch(r"[0-9]{6}\.(?:BJ|SH|SZ)", symbol) is None
                or re.fullmatch(r"[0-9]{8}", period_text) is None
            ):
                parser.error("--successor-income-support must be TS_CODE@YYYYMMDD")
            income_support_dependencies.append({"ts_code": symbol, "end_date": period_text})
        financial_support_dependencies = []
        for raw_dependency in args.successor_financial_support:
            table_text, table_separator, subject_text = str(raw_dependency).partition(":")
            symbol_text, date_separator, period_text = subject_text.partition("@")
            table = table_text.strip().lower()
            symbol = symbol_text.strip().upper()
            if (
                not table_separator
                or not date_separator
                or table not in {"balancesheet", "cashflow", "income"}
                or re.fullmatch(r"[0-9]{6}\.(?:BJ|SH|SZ)", symbol) is None
                or re.fullmatch(r"[0-9]{8}", period_text) is None
            ):
                parser.error("--successor-financial-support must be " "TABLE:TS_CODE@YYYYMMDD")
            financial_support_dependencies.append(
                {"table": table, "ts_code": symbol, "end_date": period_text}
            )
        selected_special_modes = sum(
            bool(value)
            for value in (
                args.authoritative_full_rebuild,
                args.safe_incremental_successor,
                args.taint_analysis_dry_run,
            )
        )
        if selected_special_modes > 1:
            parser.error(
                "--authoritative-full-rebuild, --safe-incremental-successor, "
                "and --taint-analysis-dry-run are mutually exclusive"
            )
        if args.append_first_successor and not args.safe_incremental_successor:
            parser.error("--append-first-successor requires --safe-incremental-successor")
        if args.append_first_successor and not historical_taint_evidence:
            parser.error(
                "--append-first-successor requires at least one "
                "--historical-taint-failure-evidence"
            )
        if historical_taint_evidence and not args.append_first_successor:
            parser.error("--historical-taint-failure-evidence requires " "--append-first-successor")
        if (
            income_support_dependencies or financial_support_dependencies
        ) and not args.append_first_successor:
            parser.error("successor financial support requires --append-first-successor")
        if args.taint_analysis_dry_run:
            required_taint_args = {
                "--run-id": args.run_id,
                "--as-of": args.as_of,
                "--audit-run-root": args.audit_run_root,
                "--canonical-predecessor-root": args.canonical_predecessor_root,
                "--expected-pointer-sha256": args.expected_pointer_sha256,
                "--canonical-scope-path": args.canonical_scope_path,
                "--canonical-market-pointer-path": (args.canonical_market_pointer_path),
                "--canonical-pit-pointer-path": args.canonical_pit_pointer_path,
                "--canonical-membership-path": args.canonical_membership_path,
                "--history-audit-path": args.history_audit_path,
                "--expected-history-audit-sha256": (args.expected_history_audit_sha256),
            }
            missing_taint_args = sorted(
                name for name, value in required_taint_args.items() if not value
            )
            if not args.allow_live:
                missing_taint_args.append("--allow-live")
            if missing_taint_args:
                parser.error("--taint-analysis-dry-run requires " + ", ".join(missing_taint_args))
            if not Path(args.audit_run_root).expanduser().is_absolute():
                parser.error("--audit-run-root must be absolute")
            if [item.strip().lower() for item in args.universes.split(",") if item.strip()] != [
                "full_a"
            ]:
                parser.error("--taint-analysis-dry-run requires --universes full_a")
            conflicting_taint_args = []
            if args.raw_input_dir:
                conflicting_taint_args.append("--raw-input-dir")
            if args.checkpoint_root:
                conflicting_taint_args.append("--checkpoint-root")
            if args.data_root != "data/parquet/cn":
                conflicting_taint_args.append("--data-root")
            if args.snapshot_root != "data/cn_market_full/_snapshots/fundamental":
                conflicting_taint_args.append("--snapshot-root")
            if args.reports_root != "reports/fundamental_readiness":
                conflicting_taint_args.append("--reports-root")
            if conflicting_taint_args:
                parser.error(
                    "--taint-analysis-dry-run rejects staging/generation "
                    "arguments: " + ", ".join(conflicting_taint_args)
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
                "--canonical-market-pointer-path": (args.canonical_market_pointer_path),
                "--canonical-pit-pointer-path": args.canonical_pit_pointer_path,
                "--canonical-membership-path": args.canonical_membership_path,
                "--history-audit-path": args.history_audit_path,
                "--expected-history-audit-sha256": (args.expected_history_audit_sha256),
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
                parser.error("--safe-incremental-successor requires --universes full_a")
            if missing_successor_args:
                parser.error(
                    "--safe-incremental-successor requires " + ", ".join(missing_successor_args)
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
            append_first_successor=args.append_first_successor,
            historical_taint_evidence=historical_taint_evidence,
            income_support_dependencies=income_support_dependencies,
            financial_support_dependencies=financial_support_dependencies,
            taint_analysis_dry_run=args.taint_analysis_dry_run,
            audit_run_root=args.audit_run_root or None,
            canonical_predecessor_root=(args.canonical_predecessor_root or None),
            expected_pointer_sha256=args.expected_pointer_sha256,
            canonical_scope_path=args.canonical_scope_path or None,
            canonical_market_pointer_path=(args.canonical_market_pointer_path or None),
            canonical_pit_pointer_path=(args.canonical_pit_pointer_path or None),
            canonical_membership_path=args.canonical_membership_path or None,
            history_audit_path=args.history_audit_path or None,
            expected_history_audit_sha256=(args.expected_history_audit_sha256),
            checkpoint_root=args.checkpoint_root or None,
            checkpoint_batch_size=args.checkpoint_batch_size,
            max_attempts=args.max_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            max_retry_backoff_seconds=args.max_retry_backoff_seconds,
            requests_per_second=args.requests_per_second,
        )
        if (
            args.authoritative_full_rebuild
            or args.safe_incremental_successor
            or args.taint_analysis_dry_run
        ):
            _print_json(result)
        if args.taint_analysis_dry_run and result.get("taint_analysis_status") != "PASS":
            raise SystemExit(2)
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
                    "fundamental-promote requires --staging-root and " "--expected-pointer-sha256"
                )
        elif args.recover:
            if not args.journal_root or not args.journal_run_id:
                parser.error(
                    "safe successor recovery requires --journal-root and " "--journal-run-id"
                )
        else:
            if not args.staging_root or not args.expected_pointer_sha256:
                parser.error(
                    "safe successor promotion requires --staging-root and "
                    "--expected-pointer-sha256"
                )
            if args.execute and (not args.journal_root or not args.journal_run_id):
                parser.error(
                    "safe successor --execute requires --journal-root and " "--journal-run-id"
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
        special_modes = sum(
            bool(value)
            for value in (
                args.prepare_transaction,
                args.commit_prepared,
                args.recover,
            )
        )
        if special_modes > 1 or (args.commit and special_modes):
            parser.error(
                "legacy --commit, --prepare-transaction, --commit-prepared, "
                "and --recover are mutually exclusive"
            )
        if args.execute_forward and not args.recover:
            parser.error("--execute-forward requires --recover")
        if args.execute_rollback and not args.recover:
            parser.error("--execute-rollback requires --recover")
        if args.execute_forward and args.execute_rollback:
            parser.error("--execute-forward and --execute-rollback are mutually exclusive")
        core_values = {
            "--target-date": args.target_date,
            "--snapshot-manifest-path": args.snapshot_manifest_path,
            "--expected-snapshot-manifest-sha256": (args.expected_snapshot_manifest_sha256),
            "--coverage-manifest-path": args.coverage_manifest_path,
            "--expected-coverage-manifest-sha256": (args.expected_coverage_manifest_sha256),
            "--scope-artifact-path": args.scope_artifact_path,
            "--expected-scope-artifact-sha256": (args.expected_scope_artifact_sha256),
            "--expected-release-pointer-sha256": (args.expected_release_pointer_sha256),
            "--expected-observations-pointer-sha256": (args.expected_observations_pointer_sha256),
            "--release-run-id": args.release_run_id,
            "--observations-run-id": args.observations_run_id,
        }
        if not args.commit_prepared and not args.recover:
            missing = [name for name, value in core_values.items() if not value]
            if missing:
                parser.error("macro maintenance requires " + ", ".join(missing))
        if args.prepare_transaction and (
            not args.allow_live
            or not args.journal_root
            or not args.journal_run_id
            or not args.authority_mode
            or not args.market_pointer_path
            or not args.expected_market_pointer_sha256
            or not args.pit_pointer_path
            or not args.expected_pit_pointer_sha256
        ):
            parser.error(
                "--prepare-transaction requires --allow-live, --journal-root, "
                "--journal-run-id, --authority-mode, and exact Market/PIT "
                "pointer path+SHA pairs"
            )
        if args.authority_mode and not args.prepare_transaction:
            parser.error("--authority-mode requires --prepare-transaction")
        if args.commit_prepared and (
            not args.prepared_path
            or not args.expected_prepared_sha256
            or not args.journal_root
            or not args.journal_run_id
        ):
            parser.error(
                "--commit-prepared requires --prepared-path, "
                "--expected-prepared-sha256, --journal-root, and --journal-run-id"
            )
        if args.recover and (not args.journal_root or not args.journal_run_id):
            parser.error("--recover requires --journal-root and --journal-run-id")
        authority_values = (
            args.market_pointer_path,
            args.expected_market_pointer_sha256,
            args.pit_pointer_path,
            args.expected_pit_pointer_sha256,
        )
        if special_modes and not all(authority_values):
            parser.error(
                "transaction prepare/commit/recover requires exact Market/PIT "
                "pointer path+SHA pairs"
            )
        if any(authority_values) and not special_modes:
            parser.error(
                "Market/PIT authority arguments require transaction " "prepare/commit/recover"
            )
        rollback_values = (
            args.old_release_pointer_sha256,
            args.new_release_pointer_sha256,
            args.old_observations_pointer_sha256,
            args.new_observations_pointer_sha256,
        )
        if args.execute_rollback and not all(rollback_values):
            parser.error("--execute-rollback requires all four rollback SHA arguments")
        if any(rollback_values) and not args.execute_rollback:
            parser.error("rollback SHA arguments require --execute-rollback")
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
            prepare_transaction=args.prepare_transaction,
            authority_mode=args.authority_mode,
            commit_prepared=args.commit_prepared,
            journal_root=args.journal_root,
            journal_run_id=args.journal_run_id,
            prepared_path=args.prepared_path,
            expected_prepared_sha256=args.expected_prepared_sha256,
            market_pointer_path=args.market_pointer_path,
            expected_market_pointer_sha256=(args.expected_market_pointer_sha256),
            pit_pointer_path=args.pit_pointer_path,
            expected_pit_pointer_sha256=args.expected_pit_pointer_sha256,
            recover=args.recover,
            execute_forward=args.execute_forward,
            execute_rollback=args.execute_rollback,
            old_release_pointer_sha256=args.old_release_pointer_sha256,
            new_release_pointer_sha256=args.new_release_pointer_sha256,
            old_observations_pointer_sha256=(args.old_observations_pointer_sha256),
            new_observations_pointer_sha256=(args.new_observations_pointer_sha256),
        )
        _print_json(result)
        if str(result.get("status")) in {
            "BLOCKED",
            "PARTIAL",
            "PROMOTION_UNCERTAIN",
        }:
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
                expected_snapshot_manifest_sha256=(args.expected_snapshot_manifest_sha256),
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
        result = _read_cli_public_run(
            workspace_root=args.workspace_root, strategy_id=args.strategy_id
        )
        _print_json(result)
        return

    if args.command == "market" and args.market_command == "run":
        result = _read_cli_public_run(
            workspace_root=args.workspace_root, strategy_id=args.strategy_id
        )
        _print_json(result)
        return

    if args.command == "market" and args.market_command == "backtest":
        run_market_backtest()

    if args.command == "portfolio" and args.portfolio_command == "cycle-status":
        if (args.identity_path is None) != (args.identity_sha256 is None):
            parser.error("--identity-path and --identity-sha256 must be provided together")
        if (args.holdings_pointer_path is None) != (args.holdings_pointer_sha256 is None):
            parser.error(
                "--holdings-pointer-path and --holdings-pointer-sha256 " "must be provided together"
            )
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
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        _print_json(payload)
        if str(payload.get("state")) == "BLOCKED":
            raise SystemExit(2)
        return


def main(argv: list[str] | None = None) -> None:
    """Execute the single public CLI under the stable error boundary."""

    command_boundary(lambda: _dispatch(argv))
