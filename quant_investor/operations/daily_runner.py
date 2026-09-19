"""Internal fixed-graph runner over code-owned native adapters.

Adapters are injected by Python composition, never selected by serialized input.
This runner produces a status projection, not an EOD completion/financial seal.
The completion builder must independently replay every required native receipt.
"""

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Callable, Mapping, Protocol

from quant_investor.contracts import canonical_json_bytes
from .daily_contract import ContractError, EOD_NODE_IDS, GRAPH, NodeState, dependencies, failure
from .daily_journal import DailyJournal, FALSE_AUTHORITY, request_identity
from .journal_revisions import append_revision, selected_binding
from .status_projection import project_node_status
from .dependency_diagnostics import DependencyInputError, rejected_probe, upstream_blockers


@dataclass(frozen=True)
class NativeOutcome:
    state: NodeState
    output_refs: dict[str, dict[str, str]]
    failure_code: str | None = None


@dataclass(frozen=True)
class Probe:
    outcome: NativeOutcome | None
    safe_to_execute: bool = False
    recovery_only: bool = False


class NativeAdapter(Protocol):
    # Fixed by the installed adapter, not by request flags.
    resume_safe: bool

    def probe(self, request: dict) -> Probe:
        """Native validation or exact evidence that canonical write has not occurred."""
        ...

    def execute(self, request: dict) -> None:
        """Invoke only its existing governed writer, respecting native permits."""
        ...


class DayRunner:
    def __init__(
        self,
        workspace: str,
        trade_date: str,
        adapters: Mapping[str, NativeAdapter],
        *,
        journal: DailyJournal | None = None,
    ):
        if set(adapters) - EOD_NODE_IDS:
            raise ContractError("DAILY_ADAPTER_NODE_INVALID")
        if journal is not None:
            if (
                type(journal) is not DailyJournal
                or journal.trade_date != trade_date
                or journal.storage._io.workspace_root != Path(workspace).resolve(strict=True)
            ):
                raise ContractError("DAILY_SHARED_JOURNAL_CONTEXT_MISMATCH")
            journal._require_lock()
        self.journal = journal if journal is not None else DailyJournal(workspace, trade_date)
        self.adapters = dict(adapters)
        self._recorded_requests: dict[str, dict] = {}
        self._current_nodes: dict[str, dict] = {}

    @staticmethod
    def _blocked(node: str, code: str, *, state: str = "BLOCKED") -> dict:
        return {"state": state, "failure": failure(code, next_node=node)}

    def _request(self, base: dict, node: str, completed: dict) -> dict:
        if base.get("node_id") != node or any(
            k.startswith("upstream.") for k in base["input_refs"]
        ):
            raise ContractError("DAILY_UPSTREAM_BINDING_INVALID")
        spec = next(value for value in GRAPH if value.node_id == node)
        refs = {f"upstream.{name}": completed[name]["terminal_ref"] for name in spec.requires}
        document = {**base, "input_refs": {**base["input_refs"], **refs}}
        request_identity(document)
        return document

    def _select(self, request: dict) -> dict:
        _, key = request_identity(request)
        node_root = str(self.journal.root / "nodes" / request["node_id"])
        previous, _, _ = selected_binding(self.journal.storage, node_root)
        if previous is not None and previous != key:
            # Only typed no-write input failures permit this explicit linked revision.
            append_revision(self.journal, request, expected_request_key=previous)
        return self.journal.inspect(request)

    def _adopt(self, request: dict, previous: dict, outcome: NativeOutcome) -> dict:
        if "terminal" in previous:
            terminal = previous["terminal"]
            if (
                terminal["state"] != outcome.state.value
                or terminal["output_refs"] != outcome.output_refs
            ):
                raise ContractError("DAILY_NATIVE_REPLAY_CONFLICT")
            return {**previous, "command_status": "NO_ACTION"}
        if previous["state"] == "NOT_STARTED":
            self.journal.begin(request)
        return {
            **self.journal.finish(
                request,
                state=outcome.state,
                output_refs=outcome.output_refs,
                failure_code=outcome.failure_code,
                recovered=True,
            ),
            "command_status": "ADOPTED",
        }

    def _execute(self, request: dict, adapter: NativeAdapter, previous: dict, probe: Probe) -> dict:
        if type(probe.safe_to_execute) is not bool or type(probe.recovery_only) is not bool:
            raise ContractError("NATIVE_PROBE_FLAGS_INVALID")
        if not probe.safe_to_execute and not probe.recovery_only:
            return self._blocked(request["node_id"], "POST_WRITE_IN_DOUBT")
        if previous["state"] == "RUNNING" and not probe.recovery_only:
            self.journal.finish(
                request, state=NodeState.FAILED, output_refs={}, failure_code="IO_TRANSIENT"
            )
        if previous["state"] != "RUNNING" or not probe.recovery_only:
            current = self.journal.begin(request, reconciled_no_write=True)
        else:
            current = previous
        self._recorded_requests[request["node_id"]] = deepcopy(request)
        self._current_nodes[request["node_id"]] = current
        # Publish the existing start before entering a potentially long writer.
        # The projection is informational; the immutable journal owns the attempt.
        self._project(self._current_nodes)
        try:
            adapter.execute(request)
            verified = adapter.probe(request)
            if verified.outcome is None:
                raise ContractError("NATIVE_OUTPUT_UNCONFIRMED_AFTER_WRITE")
        except Exception:
            # Preserve RUNNING start for native reconciliation; no false zero-write
            # terminal and no retry based on the exception's text/type alone.
            return {
                **self.journal.inspect(request),
                "state": "FAILED",
                "failure": failure("POST_WRITE_IN_DOUBT", next_node=request["node_id"]),
            }
        outcome = verified.outcome
        return {
            **self.journal.finish(
                request,
                state=outcome.state,
                output_refs=outcome.output_refs,
                failure_code=outcome.failure_code,
                recovered=probe.recovery_only,
            ),
            "command_status": "EXECUTED",
        }

    def _node(self, request: dict, adapter: NativeAdapter, *, resume: bool) -> dict:
        previous = self._select(request)
        try:
            probe = adapter.probe(request)
        except DependencyInputError as exc:
            return rejected_probe(previous, node=request["node_id"], error=exc)
        if probe.outcome is not None:
            return self._adopt(request, previous, probe.outcome)
        if previous["state"] == "SUCCEEDED":
            return self._blocked(request["node_id"], "VALIDATION_FAILED", state="STALE")
        if resume and not adapter.resume_safe:
            return self._blocked(request["node_id"], "INPUT_MISSING")
        return self._execute(request, adapter, previous, probe)

    def run(self, templates: Mapping[str, dict], *, resume: bool = False) -> dict:
        """Execute one fixed day; missing adapters/inputs are explicit blocked nodes."""
        if set(templates) - EOD_NODE_IDS or type(resume) is not bool:
            raise ContractError("DAILY_REQUEST_NODE_INVALID")
        for document in templates.values():
            request_identity(document)
        with self.journal.locked():
            return self.run_locked(templates, resume=resume)

    def run_locked(
        self,
        templates: Mapping[str, dict],
        *,
        resume: bool = False,
        resolve: Callable[[str, dict], tuple[dict, NativeAdapter] | None] | None = None,
    ) -> dict:
        """Shared owner for the core handoff; never reacquire the same day lock."""
        self.journal._require_lock()
        if set(templates) - EOD_NODE_IDS or type(resume) is not bool:
            raise ContractError("DAILY_REQUEST_NODE_INVALID")
        for document in templates.values():
            request_identity(document)
        nodes: dict[str, dict] = {}
        self._recorded_requests = {}
        self._current_nodes = nodes
        for spec in GRAPH:
            if not spec.eod_required:
                continue
            node = spec.node_id
            states = {key: NodeState(value["state"]) for key, value in nodes.items()}
            missing = dependencies(node, states)
            if missing:
                nodes[node] = {
                    **self._blocked(node, "UPSTREAM_INCOMPLETE", state="SKIPPED"),
                    "blocking_nodes": list(missing),
                    "upstream_blockers": upstream_blockers(missing, nodes),
                }
                continue
            try:
                if resolve is not None:
                    resolved = resolve(node, nodes)
                    if resolved is None:
                        nodes[node] = self._blocked(node, "INPUT_MISSING")
                        continue
                    template, adapter = resolved
                else:
                    if node not in templates or node not in self.adapters:
                        nodes[node] = self._blocked(node, "INPUT_MISSING")
                        continue
                    template, adapter = templates[node], self.adapters[node]
                request = self._request(template, node, nodes)
                nodes[node] = self._node(request, adapter, resume=resume)
                # A validated start proves this request was durably recorded.
                # Constructed but unattempted requests cannot claim consumed refs.
                if nodes[node].get("start") is not None:
                    self._recorded_requests[node] = deepcopy(request)
            except DependencyInputError:
                # Registry construction can fail before a request/probe exists.
                # Preserve the diagnosis without inventing attempt or write custody.
                raise
            except Exception:
                nodes[node] = self._blocked(node, "VALIDATION_FAILED", state="FAILED")
            self._project(nodes)
        return self._project(nodes)

    def _project(self, nodes: dict[str, dict]) -> dict:
        states = {row["state"] for row in nodes.values()}
        if states & {"FAILED", "STALE"}:
            status = "FAILED"
        elif states == {"SUCCEEDED"} and set(nodes) == EOD_NODE_IDS:
            status = "PARTIAL"
        else:
            status = "PARTIAL" if "SUCCEEDED" in states else "BLOCKED"
        value = {
            "schema_version": "cn-daily-dag-status.v1",
            "trade_date": self.journal.trade_date,
            "status": status,
            "nodes": {
                node: project_node_status(
                    row,
                    recorded_request=self._recorded_requests.get(node),
                    invocation_command=row.get("command_status"),
                )
                for node, row in nodes.items()
            },
            "authority": FALSE_AUTHORITY,
            "completion_ref": None,
            "completion_status": "AWAITING_NATIVE_COMPLETION_VALIDATION",
        }
        self.journal.storage.write(
            str(self.journal.root / "dag-status.v1.json"),
            canonical_json_bytes(value),
            projection=True,
        )
        return value
