"""Inject one installed Top100 post-write/pre-terminal interruption and resume."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def inventory(root):
    return {
        p.relative_to(root).as_posix(): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def dispatch_with_interruption(*, workspace, request_ref, release_install_ref, day):
    from scripts.daily_production import dispatch_daily_request
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.operations.daily_status import read_daily_status
    from quant_investor.operations.daily_contract import ContractError
    from quant_investor.intelligence.storage import (
        DailyResearchPoolStore,
        POOL_ROOT_RELATIVE_PATH,
        POOL_STRATEGY_ID,
    )
    from quant_investor.factors.governance import factor_production_prepare

    root = Path(workspace)
    arguments = dict(
        workspace=workspace,
        request_ref=request_ref,
        release_install_ref=release_install_ref,
        synthetic=True,
    )
    original = DailyJournal.finish
    injected = []

    def interrupted(self, *args, **kwargs):
        if self.trade_date == day and args[0]["node_id"] == "top100":
            injected.append("top100 terminal interrupted")
            raise RuntimeError("SYNTHETIC_TOP100_POST_WRITE_INTERRUPTION")
        return original(self, *args, **kwargs)

    failure = None
    with patch.object(DailyJournal, "finish", interrupted):
        try:
            first = dispatch_daily_request(**arguments)
        except ContractError as exc:
            if str(exc) != "EXECUTION_EARLY_HANDOFF_UNAVAILABLE" or not injected:
                raise
            failure = {"exception": type(exc).__name__, "message": str(exc)}
        else:
            execution = first.get("result", first)
            if not injected or execution["business_state"] == "COMPLETE":
                raise AssertionError("planned Top100 interruption did not leave an incomplete day")
            failure = {"result": first}
    status = read_daily_status(workspace, day)
    if (
        status["nodes"]["factor"]["state"] != "SUCCEEDED"
        or status["nodes"]["top100"]["state"] == "SUCCEEDED"
    ):
        raise AssertionError("interruption did not preserve completed Factor and incomplete Top100")
    factor_terminal = deepcopy(status["nodes"]["factor"]["terminal_ref"])
    pointer_path = root / "results/factors/_active.json"
    pointer_before = (pointer_path.read_bytes(), pointer_path.stat().st_mtime_ns)
    pointer = json.loads(pointer_before[0])
    generation = root / "results/factors/generations" / pointer["factor_generation_id"]
    generation_before = inventory(generation)
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    pool = root / POOL_ROOT_RELATIVE_PATH / POOL_STRATEGY_ID / iso
    pool_before = inventory(pool)
    if set(pool_before) != {
        "manifest.json",
        "factor_research_rank.json",
        "selected_symbols.json",
        "publish_receipt.json",
        "top100.parquet",
    }:
        raise AssertionError("native Top100 outputs were not committed before interruption")
    checkpoint = root.parent / ("configured-top100-interruption-" + day + ".json")
    if checkpoint.exists():
        raise AssertionError("interruption checkpoint already exists; inspect its original state")
    checkpoint.write_text(
        json.dumps(
            {
                "phase": "AFTER_INTERRUPTION_BEFORE_RECOVERY",
                "synthetic": True,
                "trade_date": day,
                "original_request_ref": request_ref,
                "failure": failure,
                "factor_pointer_sha256": hashlib.sha256(pointer_before[0]).hexdigest(),
                "factor_generation_id": pointer["factor_generation_id"],
                "factor_terminal_ref": factor_terminal,
                "native_pool_inventory": pool_before,
            },
            indent=2,
        )
        + "\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("recovery recomputed Factor or republished committed Top100")

    with (
        patch.object(factor_production_prepare, "prepare_factor_production", forbidden),
        patch.object(DailyResearchPoolStore, "publish", forbidden),
    ):
        resumed = dispatch_daily_request(**arguments)
    if (pointer_path.read_bytes(), pointer_path.stat().st_mtime_ns) != pointer_before:
        raise AssertionError("recovery replaced the committed Factor pointer")
    if inventory(generation) != generation_before or inventory(pool) != pool_before:
        raise AssertionError("recovery rewrote the committed Factor generation or Top100 outputs")
    after = read_daily_status(workspace, day)
    if (
        after["nodes"]["factor"]["terminal_ref"] != factor_terminal
        or after["nodes"]["top100"]["state"] != "SUCCEEDED"
    ):
        raise AssertionError("native selected Factor/Top100 recovery did not converge")
    proof = {
        "case": 3,
        "synthetic": True,
        "trade_date": day,
        "original_request_ref": request_ref,
        "injected_failure": failure,
        "same_request_resumed": True,
        "factor_preparation_forbidden_on_resume": True,
        "pool_publish_forbidden_on_resume": True,
        "factor_pointer_generation_and_pool_bytes_mtimes_unchanged": True,
        "factor_terminal_ref": factor_terminal,
        "top100_terminal_ref": after["nodes"]["top100"]["terminal_ref"],
        "full_eod_verified_by_caller": False,
        "production_deployed": False,
    }
    return resumed, proof
