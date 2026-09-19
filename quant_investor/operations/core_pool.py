"""One journal owner for verified native core adoption and Top100 publication."""

import hashlib
from pathlib import Path
from typing import Any, Mapping

from quant_investor.contracts import (
    MAX_CANONICAL_JSON_BYTES,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from quant_investor.factors.production_authority import (
    FactorProductionStore,
    FACTOR_ACTIVE_POINTER_PATH,
    FACTOR_POINTER_HISTORY_ROOT,
)
from quant_investor.factors.production_observation import validate_factor_production_observation
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.intelligence import build_factor_research_rank
from quant_investor.intelligence.storage import (
    DailyResearchPoolStore,
    THEME_POLICY_V2_RELATIVE_PATH,
)
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.system.errors import SystemStorageError
from .daily_contract import ContractError, GRAPH_SHA256, NodeState, validate_ref
from .daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority
from .daily_runner import DayRunner, NativeOutcome, Probe
from .dependency_diagnostics import DependencyInputError

CORE_NODES = ("calendar", "pit", "market", "factor", "low_observation", "w80_observation", "top100")


def validate_core_observation(document: dict, *, alias: str, snapshot: dict) -> None:
    try:
        value = validate_factor_production_observation(document)
    except FactorGovernanceError as exc:
        raise DependencyInputError("CORE_OBSERVATION_SCHEMA_MISMATCH") from exc
    row = value["payload"]
    groups = (
        ("CORE_OBSERVATION_DATE_MISMATCH", ("signal_date",)),
        (
            "CORE_OBSERVATION_FACTOR_BINDING_MISMATCH",
            (
                "factor_pointer_sha256",
                "factor_generation_id",
                "factor_generation_sha256",
            ),
        ),
        (
            "CORE_OBSERVATION_MARKET_BINDING_MISMATCH",
            (
                "market_pointer_sha256",
                "market_manifest_sha256",
            ),
        ),
        (
            "CORE_OBSERVATION_PIT_BINDING_MISMATCH",
            (
                "pit_pointer_sha256",
                "pit_manifest_sha256",
                "pit_membership_sha256",
            ),
        ),
        (
            "CORE_OBSERVATION_CALENDAR_BINDING_MISMATCH",
            (
                "calendar_compilation_ref",
                "calendar_capture_custody_attestation_ref",
            ),
        ),
    )
    for reason, fields in groups:
        if any(row[k] != snapshot[k] for k in fields):
            raise DependencyInputError(reason)
    signal = next(r for r in snapshot["factor_rows"] if r["factor_alias"] == alias)
    if (
        row["factor_alias"] != alias
        or row["state"] != "OPEN"
        or row["authority"] != "NON_AUTHORIZING"
        or any(
            row[k] != signal[k]
            for k in ("factor_id", "signal_sha256", "signal_symbol_set_sha256", "symbol_count")
        )
    ):
        raise DependencyInputError("CORE_OBSERVATION_SIGNAL_MISMATCH")


class CoreContext:
    """Source-bound context; no maintenance/Factor/observation writer methods."""

    def __init__(
        self,
        workspace: str,
        trade_date: str,
        pointer_sha: str,
        release_ref: Mapping[str, str],
        *,
        next_session_calendar_proof_ref=None,
        next_session_calendar_failure_ref=None,
    ):
        validate_ref(release_ref)
        self.workspace, self.trade_date = workspace, trade_date
        self.pointer_sha, self.release_ref = pointer_sha, dict(release_ref)
        self.factors = FactorProductionStore(workspace)
        self.reader = SecureSystemStorage(workspace)
        self.pointer_ref: dict[str, str] = {}
        self.pool = DailyResearchPoolStore(workspace)
        self.publish_result: dict = {}
        self.next_proof_ref = next_session_calendar_proof_ref
        self.next_failure_ref = next_session_calendar_failure_ref
        self.future_refs: dict = {}

    def source(self, path: str, expected: str | None = None) -> tuple[dict, dict[str, str]]:
        try:
            stored = self.reader.read_workspace_file_bytes(
                path, maximum_bytes=MAX_CANONICAL_JSON_BYTES
            )
        except FileNotFoundError as exc:
            raise DependencyInputError("CORE_SOURCE_MISSING") from exc
        except SystemStorageError as exc:
            if type(exc) is SystemStorageError and isinstance(exc.__cause__, FileNotFoundError):
                raise DependencyInputError("CORE_SOURCE_MISSING") from exc
            raise
        if expected is not None and stored.byte_sha256 != expected:
            raise DependencyInputError("CORE_SOURCE_SHA_MISMATCH")
        return parse_canonical_json_bytes(stored.data, label="core source"), {
            "path": path,
            "sha256": stored.byte_sha256,
        }

    def snapshot(self) -> dict:
        return self.factors.read_historical_research_inputs(
            expected_pointer_sha256=self.pointer_sha, expected_trade_date=self.trade_date
        )

    def future_outputs(self, snapshot):
        from .future_calendar_binding import future_calendar_outputs

        return future_calendar_outputs(
            workspace=self.workspace,
            trade_date=self.trade_date,
            proof_ref=self.next_proof_ref,
            failure_ref=self.next_failure_ref,
            factor_snapshot=snapshot,
        )

    def prepare(self, journal: DailyJournal) -> None:
        journal._require_lock()
        self.future_refs = self.future_outputs(self.snapshot())
        self.source(self.release_ref["path"], self.release_ref["sha256"])
        pointer = self.factors.read(FACTOR_ACTIVE_POINTER_PATH)
        if pointer.byte_sha256 != self.pointer_sha:
            pointer = self.factors.read(FACTOR_POINTER_HISTORY_ROOT / f"{self.pointer_sha}.json")
        if (
            pointer.byte_sha256 != self.pointer_sha
            or hashlib.sha256(pointer.data).hexdigest() != self.pointer_sha
        ):
            raise ContractError("CORE_POOL_POINTER_SHA_MISMATCH")
        copied = journal.storage.write(
            str(journal.root / "inputs" / f"factor-pointer-{self.pointer_sha}.json"), pointer.data
        )
        self.pointer_ref = {"path": copied.relative_path, "sha256": copied.byte_sha256}

    def template(self, node: str) -> dict:
        if node not in CORE_NODES or not self.pointer_ref:
            raise ContractError("CORE_CONTEXT_NOT_PREPARED")
        policies = {}
        if node == "top100":
            _, policies["research"] = self.source(THEME_POLICY_V2_RELATIVE_PATH)
        return {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": self.trade_date,
            "node_id": node,
            "graph_sha256": GRAPH_SHA256,
            "release_ref": self.release_ref,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy_refs": policies,
            "input_refs": {
                "factor_pointer": self.pointer_ref,
                **(self.future_refs if node == "calendar" else {}),
            },
        }

    def check_request(self, request: dict, node: str) -> None:
        expected = self.template(node)
        observed = {
            **request,
            "input_refs": {
                k: v for k, v in request["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != expected:
            raise ContractError("CORE_ADAPTER_REQUEST_MISMATCH")
        self.source(self.pointer_ref["path"], self.pointer_ref["sha256"])
        self.source(self.release_ref["path"], self.release_ref["sha256"])
        for name, ref in request["input_refs"].items():
            if name.startswith("upstream."):
                self._check_upstream(name.removeprefix("upstream."), ref)

    def _check_upstream(self, node: str, ref: dict) -> None:
        document, _ = self.source(ref["path"], ref["sha256"])
        if (
            document.get("schema_version") != "cn-daily-node-terminal.v1"
            or document.get("state") != "SUCCEEDED"
            or not _false_authority(document.get("authority"))
        ):
            raise ContractError("CORE_UPSTREAM_TERMINAL_INVALID")
        if node == "top100":
            outputs = self.pool.verify(
                **self.pool_arguments(self.template("top100")), required_format="TABULAR"
            )
            if outputs != document["output_refs"]:
                raise ContractError("CORE_UPSTREAM_POOL_OUTPUT_MISMATCH")
            return
        for output in document["output_refs"].values():
            self.source(output["path"], output["sha256"])

    def _object(self, reference: dict) -> dict[str, str]:
        path = f"results/factors/objects/{reference['kind']}/{reference['byte_sha256']}.json"
        _, ref = self.source(path, reference["byte_sha256"])
        return ref

    def observation(self, alias: str, snapshot: dict) -> tuple[dict, dict[str, str]]:
        day = self.trade_date
        path = f"results/factors/observations/{day[:4]}/{day[4:6]}/{day[6:]}/{alias}.json"
        document, ref = self.source(path)
        validate_core_observation(document, alias=alias, snapshot=snapshot)
        return document, ref

    def core_outputs(self, node: str) -> dict[str, dict[str, str]]:
        snapshot = self.snapshot()
        generation = snapshot["factor_generation"]["payload"]
        if node == "calendar":
            return {
                **{
                    name: self._object(generation[name])
                    for name in (
                        "calendar_compilation_ref",
                        "calendar_capture_custody_attestation_ref",
                    )
                },
                **self.future_outputs(snapshot),
            }
        if node in {"pit", "market"}:
            key = "market_pit_selection_ref" if node == "pit" else "market_input_ref"
            return {key: self._object(generation[key])}
        if node == "factor":
            path = f"results/factors/generations/{snapshot['factor_generation_id']}/generation.json"
            _, ref = self.source(path, snapshot["factor_generation_sha256"])
            return {"generation": ref}
        alias = {"low_observation": "LOW", "w80_observation": "W80"}[node]
        _, ref = self.observation(alias, snapshot)
        return {alias: ref}

    def pool_arguments(self, request: dict) -> dict[str, Any]:
        self.check_request(request, "top100")
        selected = self.snapshot()
        observations = [self.observation(alias, selected)[0] for alias in ("LOW", "W80")]
        policy_ref = request["policy_refs"]["research"]
        policy, _ = self.source(policy_ref["path"], policy_ref["sha256"])
        day = self.trade_date
        rank = build_factor_research_rank(
            snapshot=selected,
            observations=observations,
            policy=policy,
            as_of=f"{day[:4]}-{day[4:6]}-{day[6:]}T07:00:00Z",
            created_at=max(
                selected["factor_generation"]["created_at"],
                *(row["created_at"] for row in observations),
            ),
        )
        return {
            "rank": rank,
            "observations": observations,
            "expected_policy_sha256": policy_ref["sha256"],
            "policy_path": policy_ref["path"],
        }

    def recheck(self) -> None:
        self.snapshot()


class CoreEvidenceAdapter:
    resume_safe = False

    def __init__(self, context: CoreContext, node: str):
        self.context, self.node = context, node

    def probe(self, request: dict) -> Probe:
        self.context.check_request(request, self.node)
        return Probe(NativeOutcome(NodeState.SUCCEEDED, self.context.core_outputs(self.node)))

    def execute(self, request: dict) -> None:
        raise ContractError("CORE_ADOPTION_CANNOT_LAUNCH_MAINTENANCE")


class NativePoolAdapter:
    resume_safe = True

    def __init__(self, context: CoreContext):
        self.context = context

    def probe(self, request: dict) -> Probe:
        arguments = self.context.pool_arguments(request)
        try:
            refs = self.context.pool.verify(**arguments, required_format="TABULAR")
        except FileNotFoundError:
            return Probe(None, safe_to_execute=True)
        self.context.recheck()
        return Probe(NativeOutcome(NodeState.SUCCEEDED, refs))

    def execute(self, request: dict) -> None:
        arguments = self.context.pool_arguments(request)
        self.context.publish_result = self.context.pool.publish(
            **arguments, before_publish=self.context.recheck
        )


def core_registry(context: CoreContext) -> dict:
    return {
        node: NativePoolAdapter(context) if node == "top100" else CoreEvidenceAdapter(context, node)
        for node in CORE_NODES
    }


def publish_core_pool(
    *,
    workspace: str,
    trade_date: str,
    factor_pointer_sha256: str,
    release_ref: Mapping[str, str],
    next_session_calendar_proof_ref=None,
    next_session_calendar_failure_ref=None,
) -> dict[str, Any]:
    """Installed core hook: one day lock, one owner, seven validated node receipts."""
    context = CoreContext(
        workspace,
        trade_date,
        factor_pointer_sha256,
        release_ref,
        next_session_calendar_proof_ref=next_session_calendar_proof_ref,
        next_session_calendar_failure_ref=next_session_calendar_failure_ref,
    )
    runner = DayRunner(workspace, trade_date, core_registry(context))
    with runner.journal.locked():
        context.prepare(runner.journal)
        templates = {node: context.template(node) for node in CORE_NODES}
        result = runner.run_locked(templates, resume=True)
        row = result["nodes"]["top100"]
        if any(result["nodes"][node]["state"] != "SUCCEEDED" for node in CORE_NODES):
            raise ContractError("CORE_HANDOFF_INCOMPLETE")
        handoff = {
            "schema_version": "cn-daily-core-handoff.v1",
            "trade_date": trade_date,
            "graph_sha256": GRAPH_SHA256,
            "release_ref": dict(release_ref),
            "node_refs": {node: result["nodes"][node]["terminal_ref"] for node in CORE_NODES},
            "authority": FALSE_AUTHORITY,
        }
        sealed = runner.journal.storage.write(
            str(runner.journal.root / "core-handoff.v1.json"), canonical_json_bytes(handoff)
        )
        status = context.publish_result.get("command_status", row["command_status"])
        return {
            **row,
            "status": row["state"],
            "command_status": status,
            "core_handoff_ref": {"path": sealed.relative_path, "sha256": sealed.byte_sha256},
        }
