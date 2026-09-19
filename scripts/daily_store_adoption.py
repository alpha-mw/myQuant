"""Adopt one exact already committed native batch close; never execute it again."""

from datetime import datetime, timezone
from pathlib import Path
import re

from scripts import cn_official_close_batch as native
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"
PREIMAGES = {
    "expected_market_pointer_sha": "market_pointer_sha256",
    "expected_benchmark_pointer_sha": "benchmark_pointer_sha256",
    "expected_event_pointer_sha": "event_pointer_sha256",
    "calendar_receipt_sha": "calendar_receipt_sha256",
    "policy_sha": "policy_sha256",
    "retrospective_sha": "retrospective_sha256",
}


def _proof_args(root, plan, sha):
    return dict(
        record_root=root,
        transaction_id=plan["transaction_id"],
        expected_plan_sha=sha,
        expected_source_pointer_sha=plan["preimages"]["store_pointer_sha256"],
        expected_target=plan["requested_target"],
        **({"plan_version": 2} if native._plan_version(plan) == 2 else {}),
    )


def verify_store_plan_binding(*, arguments, plan_ref, plan, custody_at=None):
    """Derive ordinary/adopted relation from bound SHAs, without current lookup."""
    ref = validate_ref(plan_ref)
    version = native.close_contracts.validate_plan(plan, path=ref["path"])
    if (
        plan.get("schema_id")
        != (native.BATCH_PLAN_SCHEMA if version == 1 else native.close_contracts.PLAN_V2)
        or plan.get("content_sha256") != native.content_sha256(plan)
        or plan.get("broker_order_trade_authority") is not False
        or plan.get("all_or_nothing") is not True
    ):
        raise ContractError("STORE_ADOPTION_PLAN_INVALID")
    if (
        native._load_json(
            Path(arguments["project_root"]) / ref["path"],
            expected_sha=ref["sha256"],
            label="bound native Store plan",
        )
        != plan
    ):
        raise ContractError("STORE_ADOPTION_PLAN_BYTES_MISMATCH")
    if any(arguments[key] != plan["preimages"][field] for key, field in PREIMAGES.items()):
        raise ContractError("STORE_ADOPTION_NON_STORE_PREIMAGE_MISMATCH")
    if arguments.get("registered_event_declaration_ref") != plan.get(
        "registered_event_declaration_ref"
    ):
        raise ContractError("STORE_ADOPTION_REGISTERED_SOURCE_MISMATCH")
    baseline_ref = None
    if version == 2:
        native._registered_plan_proof(arguments["record_root"], plan, retained=True)
        baseline_ref = {
            "path": (
                f"{RECORD_ROOT}/_record_store/daily_close_transactions/"
                f"{plan['transaction_id']}/decision-source-pointer.v1.json"
            ),
            "sha256": plan["decision_baseline_pointer_ref"]["sha256"],
        }
    expected = arguments["expected_store_pointer_sha"]
    source = plan["preimages"]["store_pointer_sha256"]
    if expected == source:
        return {"adopted": False, "source_pointer_ref": baseline_ref}
    proof = native.inspect_frozen_close_commit(
        **_proof_args(arguments["record_root"], plan, ref["sha256"])
    )
    if proof["pointer_sha256"] != expected or proof["status"] != "VERIFIED":
        raise ContractError("STORE_ADOPTION_POINTER_TRANSITION_MISMATCH")
    if custody_at is not None and utc_stamp(proof["completion"]["cas_observed_at"]) > utc_stamp(
        custody_at
    ):
        raise ContractError("STORE_ADOPTION_CUSTODY_BEFORE_COMMIT")
    source_ref = (
        proof["source_pointer_ref"] if version == 1 else proof["decision_source_pointer_ref"]
    )
    return {
        "adopted": True,
        "proof": proof,
        "source_pointer_ref": {
            "path": f"{RECORD_ROOT}/{source_ref['path']}",
            "sha256": source_ref["sha256"],
        },
    }


def adopt_existing_close(arguments, no_action):
    """Caller holds the existing operation lock and has run native NO_ACTION checks."""
    root = Path(arguments["record_root"])
    target = no_action.get("latest_required_close_date")
    if target != no_action.get("last_official_date") or no_action.get("status") != "NO_ACTION":
        raise ContractError("STORE_EXISTING_CLOSE_DATE_MISMATCH")
    loaded = native.load_registered_catalog(root)
    if loaded is None:
        raise ContractError("STORE_EXISTING_CLOSE_UNREGISTERED")
    pointer, catalog = loaded
    observed = native._pointer_sha(root / "_record_store/current.v1.json")
    if (
        observed != arguments["expected_store_pointer_sha"]
        or observed != no_action["pointer_sha256"]
    ):
        raise ContractError("STORE_EXISTING_CLOSE_POINTER_CHANGED")
    receipts = [
        row
        for row in catalog["receipts"]
        if row.get("schema_id") == native.BATCH_RECEIPT_SCHEMA
        and row.get("record_id") == pointer["active_record_id"]
    ]
    if len(receipts) != 1:
        raise ContractError("STORE_EXISTING_CLOSE_RECEIPT_UNSUPPORTED_OR_AMBIGUOUS")
    transaction = receipts[0].get("transaction_id")
    if (
        type(transaction) is not str
        or re.fullmatch(r"daily-close-[0-9]{8}-[0-9a-f]{16}", transaction) is None
    ):
        raise ContractError("STORE_EXISTING_CLOSE_TRANSACTION_INVALID")
    path = native._completion_path(root, transaction).with_name("plan.v1.json")
    raw = native._read(path, label="existing native close plan")
    sha = native._sha(raw)
    plan = native._load_json(path, expected_sha=sha, label="existing native close plan")
    if plan["requested_target"] != target or plan["record_ids"][-1] != pointer["active_record_id"]:
        raise ContractError("STORE_EXISTING_CLOSE_PLAN_TARGET_MISMATCH")
    registered = native.inspect_close_commit(**_proof_args(root, plan, sha))
    if registered["completion"] is None or registered["pointer_sha256"] != observed:
        raise ContractError("STORE_EXISTING_CLOSE_COMMIT_INCOMPLETE")
    ref = {"path": path.relative_to(arguments["project_root"]).as_posix(), "sha256": sha}
    binding = verify_store_plan_binding(
        arguments=arguments,
        plan_ref=ref,
        plan=plan,
        custody_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if not binding["adopted"]:
        raise ContractError("STORE_EXISTING_CLOSE_TRANSITION_UNPROVEN")
    if native._pointer_sha(root / "_record_store/current.v1.json") != observed:
        raise ContractError("STORE_EXISTING_CLOSE_POINTER_CHANGED")
    return {
        "status": "PLAN_ADOPTED",
        "plan_path": ref["path"],
        "plan_sha256": sha,
        "source_pointer_ref": binding["source_pointer_ref"],
        "native_plan": plan,
        "commit_proof": binding["proof"],
        "write_performed": False,
    }
