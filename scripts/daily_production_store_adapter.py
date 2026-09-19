"""Code-owned adapter for the existing official-close transaction.

Loaded by the verified release's script composition layer. No serialized request
can select a callable or replace the accounting implementation.
"""

from pathlib import Path
import hashlib
from typing import Any, Mapping

from scripts import cn_official_close_batch as native
from scripts.manage_cn_strategy_records import _operation_lock
from quant_investor.operations.daily_contract import (
    ContractError,
    GRAPH_SHA256,
    NodeState,
    validate_ref,
)
from quant_investor.operations.daily_runner import NativeOutcome, Probe

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"


def prepare_store_plan(arguments: Mapping[str, Any]) -> dict:
    """Explicit native plan-only writer; never called from a read-only probe."""
    args = _arguments(arguments)
    with _operation_lock(str(args["record_root"])):
        result = native.close_through_latest(**args, execute=False, prepare_only=True)
        if result["status"] == "NO_ACTION":
            from scripts.daily_store_adoption import adopt_existing_close

            return adopt_existing_close(args, result)
        return result


def _arguments(value: Mapping[str, Any]) -> dict:
    allowed = {
        "project_root",
        "record_root",
        "expected_store_pointer_sha",
        "expected_market_pointer_sha",
        "expected_benchmark_pointer_sha",
        "expected_event_pointer_sha",
        "calendar_receipt_path",
        "calendar_receipt_sha",
        "policy_path",
        "policy_sha",
        "retrospective_path",
        "retrospective_sha",
    }
    optional = "registered_event_declaration_ref"
    if set(value) not in (allowed, allowed | {optional}):
        raise ContractError("STORE_ADAPTER_ARGUMENT_SCHEMA_INVALID")
    args = dict(value)
    if args.get(optional) is None:
        args.pop(optional, None)
    else:
        validate_ref(args[optional])
    workspace = Path(args["project_root"]).resolve(strict=True)
    if Path(args["record_root"]).resolve(strict=True) != workspace / RECORD_ROOT:
        raise ContractError("STORE_ADAPTER_ROOT_INVALID")
    args["project_root"], args["record_root"] = workspace, workspace / RECORD_ROOT
    return args


def store_close_output_refs(
    *, root: Path, workspace: Path, trade_date: str, plan: dict, proof: dict
) -> dict:
    """Read exact committed native outputs; no adapter or write path required."""
    day = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    pointer, catalog = native.load_catalog_snapshot(
        root,
        pointer_relative_path=proof["pointer_ref"]["path"],
        expected_pointer_sha256=proof["pointer_sha256"],
    )
    record_id = plan["record_ids"][plan["missing_dates"].index(day)]
    record = next(row for row in catalog["records"] if row["record_id"] == record_id)
    manual = native._load_json(
        root / record["manual_manifest_path"],
        expected_sha=record["manual_manifest_sha256"],
        label="DAG Store day manual",
    )
    if str(manual["valuation_trade_date"]).replace("-", "") != trade_date:
        raise ContractError("STORE_ADAPTER_VALUATION_DATE_MISMATCH")
    refs = {
        "pointer": proof["pointer_ref"],
        "catalog": proof["catalog_ref"],
        "ledger": {"path": record["ledger_path"], "sha256": record["ledger_sha256"]},
        "manual": {
            "path": record["manual_manifest_path"],
            "sha256": record["manual_manifest_sha256"],
        },
    }
    for name in ("manifest", "series", "owner_declaration"):
        value = proof["performance_ref"][name]
        refs["performance_" + name] = {"path": value["path"], "sha256": value["sha256"]}
    completion = native._completion_path(root, plan["transaction_id"], native._plan_version(plan))
    refs["completion"] = {
        "path": str(completion.relative_to(root)),
        "sha256": native._sha(native._read(completion, label="DAG Store completion")),
    }
    return {
        key: {
            "path": str((root / value["path"]).relative_to(workspace)),
            "sha256": value["sha256"],
        }
        for key, value in refs.items()
    }


class StoreCloseAdapter:
    resume_safe = True

    def __init__(
        self,
        *,
        arguments: Mapping[str, Any],
        trade_date: str,
        plan_ref: Mapping[str, str],
        release_ref: Mapping[str, str],
    ):
        self.args = _arguments(arguments)
        self.workspace = self.args["project_root"]
        self.trade_date = trade_date
        self.day = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        self.plan_ref, self.release_ref = validate_ref(plan_ref), validate_ref(release_ref)
        self.plan = native._load_json(
            self.workspace / self.plan_ref["path"],
            expected_sha=self.plan_ref["sha256"],
            label="DAG Store native plan",
        )
        self.version = native.close_contracts.validate_plan(self.plan, path=self.plan_ref["path"])
        expected = (
            self.args["record_root"]
            / "_record_store/daily_close_transactions"
            / self.plan["transaction_id"]
            / f"plan.v{self.version}.json"
        )
        if (
            expected != self.workspace / self.plan_ref["path"]
            or self.day not in self.plan["missing_dates"]
        ):
            raise ContractError("STORE_ADAPTER_PLAN_DATE_OR_PATH_INVALID")
        self._check_preimages()

    def _check_preimages(self) -> None:
        fields = {
            "expected_store_pointer_sha": "store_pointer_sha256",
            "expected_market_pointer_sha": "market_pointer_sha256",
            "expected_benchmark_pointer_sha": "benchmark_pointer_sha256",
            "expected_event_pointer_sha": "event_pointer_sha256",
            "calendar_receipt_sha": "calendar_receipt_sha256",
            "policy_sha": "policy_sha256",
            "retrospective_sha": "retrospective_sha256",
        }
        if any(self.args[k] != self.plan["preimages"][v] for k, v in fields.items()):
            raise ContractError("STORE_ADAPTER_PLAN_PREIMAGE_MISMATCH")
        if self.args.get("registered_event_declaration_ref") != self.plan.get(
            "registered_event_declaration_ref"
        ):
            raise ContractError("STORE_ADAPTER_REGISTERED_SOURCE_MISMATCH")

    def template(self) -> dict:
        return {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": self.trade_date,
            "node_id": "store",
            "graph_sha256": GRAPH_SHA256,
            "release_ref": self.release_ref,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy_refs": {
                "close": {"path": self.args["policy_path"], "sha256": self.args["policy_sha"]}
            },
            "input_refs": {"native_plan": self.plan_ref},
        }

    def _check_request(self, request: dict) -> None:
        observed = {
            **request,
            "input_refs": {
                k: v for k, v in request["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != self.template():
            raise ContractError("STORE_ADAPTER_REQUEST_MISMATCH")
        native._load_json(
            self.workspace / self.plan_ref["path"],
            expected_sha=self.plan_ref["sha256"],
            label="DAG Store plan",
        )
        self._check_preimages()

    def _proof_args(self) -> dict:
        return {
            "record_root": self.args["record_root"],
            "transaction_id": self.plan["transaction_id"],
            "expected_plan_sha": self.plan_ref["sha256"],
            "expected_source_pointer_sha": self.args["expected_store_pointer_sha"],
            "expected_target": self.plan["requested_target"],
            **({"plan_version": self.version} if self.version != 1 else {}),
        }

    def _output_refs(self, proof: dict) -> dict[str, dict[str, str]]:
        return store_close_output_refs(
            root=self.args["record_root"],
            workspace=self.workspace,
            trade_date=self.trade_date,
            plan=self.plan,
            proof=proof,
        )

    def probe(self, request: dict) -> Probe:
        self._check_request(request)
        completion = native._completion_path(
            self.args["record_root"], self.plan["transaction_id"], self.version
        )
        source = completion.with_name("source-pointer.v1.json")
        committed = completion.with_name("committed-pointer.v1.json")
        if source.exists() and completion.exists() and committed.exists():
            proof = native.inspect_frozen_close_commit(**self._proof_args())
            return Probe(NativeOutcome(NodeState.SUCCEEDED, self._output_refs(proof)))
        if source.exists() or source.is_symlink():
            # Pending/CAS-crashed transactions may lack completion metadata, but
            # already retained source custody must validate before any recovery.
            native._read_close_source_pointer(self.args["record_root"], self.plan)
        if completion.exists():
            if not committed.exists():
                raise ContractError("STORE_LEGACY_COMMIT_CUSTODY_INCOMPLETE")
            # Pre-custody legacy completed transactions have a read-only path.
            proof = native.inspect_close_commit(**self._proof_args())
            return Probe(NativeOutcome(NodeState.SUCCEEDED, self._output_refs(proof)))
        if committed.exists() and not source.exists():
            raise ContractError("STORE_LEGACY_COMMIT_CUSTODY_INCOMPLETE")
        current = native._pointer_sha(self.args["record_root"] / "_record_store/current.v1.json")
        if current == self.args["expected_store_pointer_sha"]:
            result = native.close_through_latest(
                **self.args, execute=False, expected_plan_sha=self.plan_ref["sha256"]
            )
            if result["status"] != "PLAN_READY":
                raise ContractError("STORE_ADAPTER_PREPARED_TRANSACTION_NOT_PENDING")
            return Probe(None, safe_to_execute=True)
        if not source.exists():
            raise ContractError("STORE_LEGACY_COMMIT_CUSTODY_INCOMPLETE")
        proof = native.inspect_close_commit(**self._proof_args())
        if proof["completion"] is None or not proof["pointer_ref"]["path"].endswith(
            "/committed-pointer.v1.json"
        ):
            return Probe(None, recovery_only=True)
        return Probe(NativeOutcome(NodeState.SUCCEEDED, self._output_refs(proof)))

    def execute(self, request: dict) -> None:
        self._check_request(request)
        if self.probe(request).outcome is not None:
            return
        with _operation_lock(str(self.args["record_root"])):
            current = native._pointer_sha(
                self.args["record_root"] / "_record_store/current.v1.json"
            )
            if current == self.args["expected_store_pointer_sha"]:
                native.close_through_latest(
                    **self.args, execute=True, expected_plan_sha=self.plan_ref["sha256"]
                )
            else:
                # This replays plan/catalog/receipt/holdings ancestry proof before
                # metadata writes. It does not pass a new preimage to the CAS writer.
                native.recover_close_completion(**self._proof_args())
