"""Read-only native Store replay for a recorded EOD; no prepare or recovery calls."""

from pathlib import Path, PurePosixPath
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import request_identity
from scripts import cn_official_close_batch as native
from scripts.daily_production_store_adapter import RECORD_ROOT, store_close_output_refs


def replay_completed_store(*, workspace: str, trade_date: str, completion_ref: dict) -> dict:
    recorded = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    root = Path(workspace).resolve(strict=True)
    storage = SecureSystemStorage(workspace)

    def read(ref):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=8 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("EOD_STORE_SOURCE_SHA_MISMATCH")
        return parse_canonical_json_bytes(raw.data)

    inputs = read(recorded["native_inputs_ref"])
    terminal_ref = recorded["node_terminal_refs"]["store"]
    terminal = read(terminal_ref)
    path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
    request = parse_canonical_json_bytes(
        storage.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024).data
    )
    plan_ref = inputs["store_plan_ref"]
    if (
        request_identity(request)[1] != terminal["request_key"]
        or request["input_refs"]["native_plan"] != plan_ref
    ):
        raise ContractError("EOD_STORE_REQUEST_BINDING_INVALID")
    # Store permits native regular owner-owned 0644 artifacts; retain its reader.
    plan = native._load_json(
        root / plan_ref["path"], expected_sha=plan_ref["sha256"], label="EOD Store plan"
    )
    version = native.close_contracts.validate_plan(plan, path=plan_ref["path"])
    if plan_ref["path"] != (
        f"{RECORD_ROOT}/_record_store/daily_close_transactions/"
        f"{plan['transaction_id']}/plan.v{version}.json"
    ):
        raise ContractError("EOD_STORE_PLAN_PATH_INVALID")
    inspector = (
        native.inspect_frozen_close_commit
        if inputs["schema_version"]
        in {
            "cn-daily-native-inputs.v4",
            "cn-daily-native-inputs.v5",
            "cn-daily-native-inputs.v6",
            "cn-daily-native-inputs.v7",
        }
        else native.inspect_close_commit
    )
    proof = inspector(
        record_root=root / RECORD_ROOT,
        transaction_id=plan["transaction_id"],
        expected_plan_sha=plan_ref["sha256"],
        expected_source_pointer_sha=plan["preimages"]["store_pointer_sha256"],
        expected_target=plan["requested_target"],
        **({"plan_version": version} if version != 1 else {}),
    )
    if proof["completion"] is None or not proof["pointer_ref"]["path"].endswith(
        "/committed-pointer.v1.json"
    ):
        raise ContractError("EOD_STORE_COMMIT_METADATA_MISSING")
    output_refs = store_close_output_refs(
        root=root / RECORD_ROOT, workspace=root, trade_date=trade_date, plan=plan, proof=proof
    )
    if output_refs != terminal["output_refs"]:
        raise ContractError("EOD_STORE_OUTPUT_REPLAY_DIFFERS")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )["recorded_completion"]
        != recorded
    ):
        raise ContractError("EOD_STORE_COMPLETION_CHANGED_DURING_REPLAY")
    return {
        "completion_ref": dict(completion_ref),
        "output_refs": output_refs,
        "validation_scope": "COMPLETED_STORE_NATIVE_REPLAY",
        "consumer_admission": False,
    }
