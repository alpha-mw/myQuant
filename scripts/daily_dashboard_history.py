"""Resolve historical Dashboard Store inputs through the native commit verifier."""

from pathlib import Path
import hashlib
from typing import Mapping

from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.strategy_records.store import load_catalog_snapshot


def committed_dashboard_store(
    *,
    project_root: Path,
    record_root: Path,
    plan_ref: Mapping[str, str],
    valuation_date: str | None = None,
) -> tuple[tuple[dict, dict], list[dict[str, str]]]:
    # Local import avoids the native close validator's common Dashboard dependency.
    from cn_official_close_batch import (
        _load_json,
        inspect_close_commit,
        inspect_frozen_close_commit,
    )
    from quant_investor.strategy_records.close_plan_contracts import validate_plan

    ref = validate_ref(plan_ref)
    plan = _load_json(
        project_root / ref["path"], expected_sha=ref["sha256"], label="historical Dashboard plan"
    )
    version = validate_plan(plan, path=ref["path"])
    if valuation_date is not None and valuation_date not in plan["missing_dates"]:
        raise ContractError("DASHBOARD_HISTORY_DATE_NOT_IN_COMMIT")
    expected_path = (
        record_root
        / "_record_store/daily_close_transactions"
        / plan["transaction_id"]
        / f"plan.v{version}.json"
    )
    if project_root / ref["path"] != expected_path:
        raise ContractError("DASHBOARD_HISTORY_PLAN_PATH_MISMATCH")
    inspector = inspect_frozen_close_commit if version == 2 else inspect_close_commit
    proof = inspector(
        record_root=record_root,
        transaction_id=plan["transaction_id"],
        expected_plan_sha=ref["sha256"],
        expected_source_pointer_sha=plan["preimages"]["store_pointer_sha256"],
        expected_target=plan["requested_target"],
        **({"plan_version": version} if version == 2 else {}),
    )
    if proof.get("status") != "VERIFIED":
        raise ContractError("DASHBOARD_HISTORY_COMMIT_UNVERIFIED")
    selected = load_catalog_snapshot(
        record_root,
        pointer_relative_path=proof["pointer_ref"]["path"],
        expected_pointer_sha256=proof["pointer_sha256"],
    )
    pointer, catalog = selected
    if catalog.get("schema_id") != "myquant.strategy_record_catalog.v3":
        raise ContractError("DASHBOARD_HISTORY_NATIVE_V3_REQUIRED")
    prefix = record_root.relative_to(project_root).as_posix() + "/"
    refs = [ref]
    for value in [proof["pointer_ref"], proof["catalog_ref"], *proof["performance_ref"].values()]:
        if isinstance(value, dict) and set(value) >= {"path", "sha256"}:
            refs.append({"path": prefix + value["path"], "sha256": value["sha256"]})
    return selected, refs


def select_historical_records(
    *, record_root: Path, catalog: dict, valuation_date: str
) -> tuple[str, str]:
    from quant_investor.strategy_records.performance import load_performance_history

    history = load_performance_history(record_root, catalog["performance_history_ref"])
    matches = [
        i for i, row in enumerate(history["rows"]) if row["valuation_date"] == valuation_date
    ]
    if len(matches) != 1 or matches[0] == 0:
        raise ContractError("DASHBOARD_HISTORY_DATE_OR_PREDECESSOR_MISSING")
    index = matches[0]
    return history["rows"][index]["record_id"], history["rows"][index - 1]["record_id"]


def registered_intraday_predecessor(
    *, project_root, record_root, pointer, catalog, latest, previous
):
    """Prove a replaced intraday record; it is not another canonical daily point."""
    from quant_investor.strategy_records.close_plan_contracts import (
        RECEIPT_V2,
        validate_registered_receipt,
    )
    from cn_official_close_batch import _read, _load_json

    if (
        previous.get("official_valuation") is not False
        or latest.get("official_valuation") is not True
    ):
        return None
    receipts = [
        r
        for r in catalog["receipts"]
        if r.get("schema_id") == RECEIPT_V2 and r.get("record_id") == latest["record"]
    ]
    if not receipts:
        return None
    if len(receipts) != 1:
        raise ContractError("DASHBOARD_REGISTERED_PREDECESSOR_RECEIPT_INVALID")
    receipt = validate_registered_receipt(receipts[0])
    path = (
        record_root
        / "_record_store/daily_close_transactions"
        / receipt["transaction_id"]
        / "plan.v2.json"
    )
    raw = _read(path, label="registered Dashboard predecessor plan")
    reference = {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    plan = _load_json(
        path, expected_sha=reference["sha256"], label="registered Dashboard predecessor plan"
    )
    selected, refs = committed_dashboard_store(
        project_root=project_root,
        record_root=record_root,
        plan_ref=reference,
        valuation_date=latest["data_date"],
    )
    if (
        selected != (pointer, catalog)
        or plan["source_active_record_id"] != previous["record"]
        or latest["source_record"] != previous["record"]
        or plan["record_ids"][-1] != latest["record"]
        or previous["data_date"] != plan["requested_target"]
    ):
        raise ContractError("DASHBOARD_REGISTERED_PREDECESSOR_MISMATCH")
    return refs
