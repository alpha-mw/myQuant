"""Append-only per-node attempt receipts; recovery never guesses writer success."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from typing import Any, Iterator, Mapping

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .daily_contract import (
    ContractError,
    EOD_NODE_IDS,
    FAILURES,
    GRAPH_SHA256,
    NodeState,
    failure,
    utc_stamp,
    validate_ref,
)
from .journal_storage import JournalStorage, ROOT

FALSE_AUTHORITY = dict.fromkeys(
    ("broker", "order", "trade", "actual_holdings_mutation", "system", "mainline"), False
)
MAX_ATTEMPTS = 16


def _false_authority(value: Any) -> bool:
    return (
        type(value) is dict
        and set(value) == set(FALSE_AUTHORITY)
        and all(item is False for item in value.values())
    )


def _validate_day(day: Any) -> None:
    if type(day) is not str:
        raise ContractError("NODE_REQUEST_DATE_INVALID")
    try:
        parsed = datetime.strptime(day, "%Y%m%d")
    except ValueError as exc:
        raise ContractError("NODE_REQUEST_DATE_INVALID") from exc
    if parsed.strftime("%Y%m%d") != day:
        raise ContractError("NODE_REQUEST_DATE_INVALID")


def request_identity(value: Mapping[str, Any]) -> tuple[dict, str]:
    fields = {
        "schema_version",
        "trade_date",
        "node_id",
        "graph_sha256",
        "release_ref",
        "adapter_sha256",
        "policy_refs",
        "input_refs",
    }
    if set(value) != fields or value["schema_version"] != "cn-daily-node-request.v1":
        raise ContractError("NODE_REQUEST_SCHEMA_INVALID")
    _validate_day(value["trade_date"])
    if value["node_id"] not in EOD_NODE_IDS or value["graph_sha256"] != GRAPH_SHA256:
        raise ContractError("NODE_REQUEST_GRAPH_INVALID")
    validate_ref(value["release_ref"])
    validate_ref({"path": "adapter.py", "sha256": value["adapter_sha256"]})
    for key in ("policy_refs", "input_refs"):
        if type(value[key]) is not dict:
            raise ContractError("NODE_REQUEST_REFS_INVALID")
        for name, ref in value[key].items():
            if type(name) is not str or not name:
                raise ContractError("NODE_REQUEST_REFS_INVALID")
            validate_ref(ref)
    raw = canonical_json_bytes(dict(value))
    return parse_canonical_json_bytes(raw, label="node request"), hashlib.sha256(raw).hexdigest()


class DailyJournal:
    """A day lock spans native adapter validation, execution and reconciliation."""

    def __init__(self, workspace: str, trade_date: str):
        if type(trade_date) is not str:
            raise ContractError("JOURNAL_DATE_INVALID")
        parsed = datetime.strptime(trade_date, "%Y%m%d")
        if parsed.strftime("%Y%m%d") != trade_date:
            raise ContractError("JOURNAL_DATE_INVALID")
        self.trade_date = trade_date
        self.root = ROOT / trade_date
        self.storage = JournalStorage(workspace)
        self._locked = False

    @contextmanager
    def locked(self) -> Iterator[None]:
        if self._locked:
            raise ContractError("JOURNAL_LOCK_REENTRANT")
        with self.storage.lock(str(self.root / ".lock")):
            self._locked = True
            try:
                yield
            finally:
                self._locked = False

    def _require_lock(self) -> None:
        if not self._locked:
            raise ContractError("JOURNAL_LOCK_REQUIRED")

    def _prefix(self, request: Mapping[str, Any]) -> tuple[dict, str, str]:
        document, key = request_identity(request)
        if document["trade_date"] != self.trade_date:
            raise ContractError("JOURNAL_REQUEST_DATE_MISMATCH")
        return document, key, str(self.root / "nodes" / document["node_id"] / key)

    def _existing_request(self, document: dict, key: str, prefix: str) -> bool:
        from .journal_revisions import selected_binding

        selected, _, _ = selected_binding(
            self.storage, str(self.root / "nodes" / document["node_id"])
        )
        if selected is not None and selected != key:
            raise ContractError("JOURNAL_NODE_INPUT_CONFLICT")
        request_file = self.storage.read(prefix + "/request.json")
        if request_file is None:
            return False
        if request_file.data != canonical_json_bytes(document):
            raise ContractError("JOURNAL_REQUEST_CONFLICT")
        return True

    def inspect(self, request: Mapping[str, Any]) -> dict:
        """Read exact attempt ordinals. SUCCEEDED still requires native replay."""
        self._require_lock()
        return self._inspect_request(request)

    def readonly_inspect(self, request: Mapping[str, Any]) -> dict:
        """Optimistic read of immutable attempts; never creates/acquires a writer lock."""
        first = self._inspect_request(request)
        if self._inspect_request(request) != first:
            raise ContractError("JOURNAL_CHANGED_DURING_READ")
        return first

    def _inspect_request(self, request: Mapping[str, Any]) -> dict:
        document, key, prefix = self._prefix(request)
        if not self._existing_request(document, key, prefix):
            return {"state": "NOT_STARTED", "request_key": key, "attempt": 0}
        latest = None
        gap = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            base = f"{prefix}/attempt-{attempt:04d}"
            start = self.storage.read(base + "/start.json")
            terminal = self.storage.read(base + "/terminal.json")
            if start is None:
                if terminal is not None:
                    raise ContractError("JOURNAL_ORPHAN_TERMINAL")
                gap = True
                continue
            if gap or (latest is not None and latest["state"] == "RUNNING"):
                raise ContractError("JOURNAL_ATTEMPT_CHAIN_INVALID")
            started = parse_canonical_json_bytes(start.data, label="attempt start")
            if (
                set(started)
                != {"schema_version", "request_key", "attempt", "started_at", "authority"}
                or started.get("schema_version") != "cn-daily-node-start.v1"
                or started.get("request_key") != key
                or type(started.get("attempt")) is not int
                or started.get("attempt") != attempt
                or not _false_authority(started.get("authority"))
            ):
                raise ContractError("JOURNAL_START_INVALID")
            utc_stamp(started["started_at"])
            latest = {
                "state": "RUNNING",
                "recovery_state": "IN_DOUBT",
                "attempt": attempt,
                "request_key": key,
                "start": started,
                "start_ref": {"path": start.relative_path, "sha256": start.byte_sha256},
            }
            if terminal is not None:
                ended = parse_canonical_json_bytes(terminal.data, label="attempt terminal")
                self._validate_terminal(ended, latest)
                latest.update(
                    state=ended["state"],
                    recovery_state=None,
                    terminal=ended,
                    terminal_ref={"path": terminal.relative_path, "sha256": terminal.byte_sha256},
                )
        if latest is None:
            return {"state": "NOT_STARTED", "request_key": key, "attempt": 0}
        return latest

    @staticmethod
    def _validate_terminal(ended: dict, selected: dict) -> None:
        fields = {
            "schema_version",
            "request_key",
            "attempt",
            "start_ref",
            "finished_at",
            "state",
            "output_refs",
            "failure",
            "authority",
            "recovered",
        }
        if (
            set(ended) != fields
            or ended["schema_version"] != "cn-daily-node-terminal.v1"
            or ended["request_key"] != selected["request_key"]
            or ended["attempt"] != selected["attempt"]
            or ended["start_ref"] != selected["start_ref"]
            or type(ended["attempt"]) is not int
            or not _false_authority(ended["authority"])
            or type(ended["recovered"]) is not bool
        ):
            raise ContractError("JOURNAL_TERMINAL_INVALID")
        if utc_stamp(ended["finished_at"]) < utc_stamp(selected["start"]["started_at"]):
            raise ContractError("JOURNAL_CHRONOLOGY_INVALID")
        if ended["state"] not in {"SUCCEEDED", "PARTIAL", "BLOCKED", "FAILED"}:
            raise ContractError("JOURNAL_TERMINAL_STATE_INVALID")
        if type(ended["output_refs"]) is not dict:
            raise ContractError("JOURNAL_OUTPUT_REFS_INVALID")
        for ref in ended["output_refs"].values():
            validate_ref(ref)
        if ended["state"] == "SUCCEEDED":
            if not ended["output_refs"] or ended["failure"] is not None:
                raise ContractError("JOURNAL_SUCCESS_WITHOUT_OUTPUT")
        elif ended["failure"] is None:
            raise ContractError("JOURNAL_FAILURE_REQUIRED")
        else:
            row = ended["failure"]
            if row != failure(row["code"], next_node=row["recommended_next_node"]):
                raise ContractError("JOURNAL_FAILURE_INVALID")

    def begin(self, request: Mapping[str, Any], *, reconciled_no_write: bool = False) -> dict:
        self._require_lock()
        document, key, prefix = self._prefix(request)
        prior = self.inspect(document)
        if prior["state"] == "SUCCEEDED":
            raise ContractError("JOURNAL_SUCCESS_REPLAY_REQUIRED")
        if prior["state"] == "RUNNING":
            raise ContractError("JOURNAL_RECONCILE_BEFORE_RETRY")
        if prior["attempt"]:
            reason = prior["terminal"]["failure"]
            if not reason or not FAILURES[reason["code"]].retryable or not reconciled_no_write:
                raise ContractError("JOURNAL_RETRY_NOT_AUTHORIZED")
        attempt = prior["attempt"] + 1
        if attempt > MAX_ATTEMPTS:
            raise ContractError("JOURNAL_ATTEMPT_BUDGET_EXHAUSTED")
        binding = str(self.root / "nodes" / document["node_id"] / "binding.json")
        if self.storage.read(binding) is None:
            self.storage.write(binding, canonical_json_bytes({"request_key": key}))
        self.storage.write(prefix + "/request.json", canonical_json_bytes(document))
        start = {
            "schema_version": "cn-daily-node-start.v1",
            "request_key": key,
            "attempt": attempt,
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "authority": FALSE_AUTHORITY,
        }
        self.storage.write(
            f"{prefix}/attempt-{attempt:04d}/start.json", canonical_json_bytes(start)
        )
        return self.inspect(document)

    def finish(
        self,
        request: Mapping[str, Any],
        *,
        state: NodeState,
        output_refs: Mapping[str, dict],
        failure_code: str | None = None,
        recovered: bool = False,
    ) -> dict:
        self._require_lock()
        document, key, prefix = self._prefix(request)
        selected = self.inspect(document)
        if selected["state"] != "RUNNING":
            raise ContractError("JOURNAL_RUNNING_ATTEMPT_REQUIRED")
        terminal = {
            "schema_version": "cn-daily-node-terminal.v1",
            "request_key": key,
            "attempt": selected["attempt"],
            "start_ref": selected["start_ref"],
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "state": state.value,
            "output_refs": dict(output_refs),
            "failure": (
                failure(failure_code, next_node=document["node_id"]) if failure_code else None
            ),
            "authority": FALSE_AUTHORITY,
            "recovered": recovered,
        }
        self._validate_terminal(terminal, selected)
        self.storage.write(
            f"{prefix}/attempt-{selected['attempt']:04d}/terminal.json",
            canonical_json_bytes(terminal),
        )
        return self.inspect(document)
