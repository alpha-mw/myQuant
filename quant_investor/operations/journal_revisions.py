"""Explicit input-arrival successors; completed node inputs are never rebound."""

from typing import Any

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .daily_contract import ContractError, failure, utc_stamp, validate_ref

MAX_REVISIONS = 16
REASONS = frozenset({"INPUT_MISSING", "INPUT_STALE", "UPSTREAM_INCOMPLETE"})
RESEARCH_NODES = frozenset({"industry", "theme", "exposure", "fundamental", "macro", "decision"})


def _research_outputs(storage: Any, node_root: str, key: str, refs: dict) -> bool:
    from quant_investor.contracts import validate_artifact
    from .daily_journal import _false_authority
    from quant_investor.intelligence._common import NO_AUTHORITY

    if node_root.rsplit("/", 1)[1] not in RESEARCH_NODES:
        return False
    prefix = node_root.split("/nodes/", 1)[0] + "/research/"
    for ref in refs.values():
        validate_ref(ref)
        if not ref["path"].startswith(prefix):
            return False
        value, _ = _read(storage, ref["path"], ref["sha256"])
        if value.get("schema_version") == "cn-daily-research-node-capture.v1":
            if value.get("request_key") != key or not _false_authority(value.get("authority")):
                return False
        else:
            artifact = validate_artifact(value)
            if artifact["kind"] not in {
                "industry_source_projection",
                "theme_membership_projection",
                "theme_economic_exposure_projection",
                "fundamental_assessment",
                "market_risk_evidence",
                "investment_decision",
                "company_source_evidence",
                "pcb_ai_hardware_membership",
                "pcb_ai_hardware_evidence",
                "low_frequency_source_freshness",
            }:
                return False
            if artifact["kind"] in {
                "pcb_ai_hardware_membership",
                "pcb_ai_hardware_evidence",
                "low_frequency_source_freshness",
            }:
                body = artifact["payload"]
                if (
                    body["research_only"] is not True
                    or body["production"] is not False
                    or body["run_state"] != "INACTIVE"
                    or type(body["authority"]) is not dict
                    or set(body["authority"]) != set(NO_AUTHORITY)
                    or any(value is not False for value in body["authority"].values())
                ):
                    return False
    return True


def _read(storage: Any, path: str, sha: str | None = None) -> tuple[dict, str]:
    stored = storage.read(path)
    if stored is None or (sha is not None and stored.byte_sha256 != sha):
        raise ContractError("JOURNAL_REVISION_REF_INVALID")
    return parse_canonical_json_bytes(stored.data, label="node revision"), stored.byte_sha256


def input_successor(old: dict, new: dict) -> None:
    if old == new or {k: v for k, v in old.items() if k != "input_refs"} != {
        k: v for k, v in new.items() if k != "input_refs"
    }:
        raise ContractError("JOURNAL_REVISION_SCOPE_INVALID")


def _failure_ref(storage: Any, node_root: str, key: str, ref: dict) -> None:
    from .daily_journal import DailyJournal, _false_authority

    validate_ref(ref)
    terminal, _ = _read(storage, ref["path"], ref["sha256"])
    attempt = terminal.get("attempt")
    if type(attempt) is not int or not 1 <= attempt <= 16:
        raise ContractError("JOURNAL_REVISION_ATTEMPT_INVALID")
    base = f"{node_root}/{key}/attempt-{attempt:04d}"
    if ref["path"] != base + "/terminal.json":
        raise ContractError("JOURNAL_REVISION_TERMINAL_PATH_INVALID")
    start, start_sha = _read(storage, base + "/start.json")
    if (
        start.get("request_key") != key
        or start.get("attempt") != attempt
        or not _false_authority(start.get("authority"))
    ):
        raise ContractError("JOURNAL_REVISION_START_INVALID")
    utc_stamp(start["started_at"])
    selected = {
        "request_key": key,
        "attempt": attempt,
        "start": start,
        "start_ref": {"path": base + "/start.json", "sha256": start_sha},
    }
    DailyJournal._validate_terminal(terminal, selected)
    reason = terminal.get("failure") or {}
    if (
        terminal["state"] not in {"BLOCKED", "FAILED", "PARTIAL"}
        or (terminal["state"] == "PARTIAL" and node_root.rsplit("/", 1)[1] not in RESEARCH_NODES)
        or (
            terminal["output_refs"]
            and not _research_outputs(storage, node_root, key, terminal["output_refs"])
        )
        or reason.get("code") not in REASONS
        or reason != failure(reason["code"], next_node=node_root.rsplit("/", 1)[1])
    ):
        raise ContractError("JOURNAL_REVISION_NOT_RETRYABLE_INPUT_FAILURE")
    if storage.read(f"{node_root}/{key}/attempt-{attempt + 1:04d}/start.json") is not None:
        raise ContractError("JOURNAL_REVISION_NEWER_ATTEMPT_EXISTS")


def selected_binding(storage: Any, node_root: str) -> tuple[str | None, int, str | None]:
    initial = storage.read(node_root + "/binding.json")
    if initial is None:
        return None, 0, None
    document = parse_canonical_json_bytes(initial.data, label="node binding")
    if set(document) != {"request_key"}:
        raise ContractError("JOURNAL_BINDING_INVALID")
    validate_ref({"path": "key.json", "sha256": document["request_key"]})
    key, previous_sha, count = document["request_key"], initial.byte_sha256, 0
    gap = False
    for ordinal in range(1, MAX_REVISIONS + 1):
        stored = storage.read(f"{node_root}/revisions/{ordinal:04d}.json")
        if stored is None:
            gap = True
            continue
        if gap:
            raise ContractError("JOURNAL_REVISION_GAP")
        row = parse_canonical_json_bytes(stored.data, label="node revision")
        _validate_revision(storage, node_root, row, key, previous_sha, ordinal)
        key, previous_sha, count = row["request_key"], stored.byte_sha256, ordinal
    return key, count, previous_sha


def _validate_revision(
    storage: Any, node_root: str, row: dict, key: str, previous_sha: str, ordinal: int
) -> None:
    from .daily_journal import request_identity

    if (
        set(row)
        != {
            "schema_version",
            "previous_key",
            "request_key",
            "previous_sha256",
            "previous_terminal_ref",
            "revision",
        }
        or row["schema_version"] != "cn-daily-node-input-revision.v1"
        or row["previous_key"] != key
        or row["previous_sha256"] != previous_sha
        or type(row["revision"]) is not int
        or row["revision"] != ordinal
    ):
        raise ContractError("JOURNAL_REVISION_CHAIN_INVALID")
    validate_ref({"path": "key.json", "sha256": row["request_key"]})
    old, _ = _read(storage, f"{node_root}/{key}/request.json", key)
    new, _ = _read(storage, f"{node_root}/{row['request_key']}/request.json", row["request_key"])
    request_identity(old)
    request_identity(new)
    input_successor(old, new)
    _failure_ref(storage, node_root, key, row["previous_terminal_ref"])


def append_revision(journal: Any, request: dict, *, expected_request_key: str) -> str:
    from .daily_journal import request_identity

    journal._require_lock()
    document, key = request_identity(request)
    if document["trade_date"] != journal.trade_date:
        raise ContractError("JOURNAL_REQUEST_DATE_MISMATCH")
    node_root = str(journal.root / "nodes" / document["node_id"])
    selected, ordinal, previous_sha = selected_binding(journal.storage, node_root)
    if selected != expected_request_key or selected is None:
        raise ContractError("JOURNAL_REVISION_PREIMAGE_MISMATCH")
    if ordinal >= MAX_REVISIONS:
        raise ContractError("JOURNAL_REVISION_BUDGET_EXHAUSTED")
    old, _ = _read(journal.storage, f"{node_root}/{selected}/request.json", selected)
    input_successor(old, document)
    prior = journal.inspect(old)
    if "terminal_ref" not in prior:
        raise ContractError("JOURNAL_REVISION_RECONCILIATION_REQUIRED")
    _failure_ref(journal.storage, node_root, selected, prior["terminal_ref"])
    journal.storage.write(f"{node_root}/{key}/request.json", canonical_json_bytes(document))
    row = {
        "schema_version": "cn-daily-node-input-revision.v1",
        "previous_key": selected,
        "request_key": key,
        "previous_sha256": previous_sha,
        "revision": ordinal + 1,
        "previous_terminal_ref": prior["terminal_ref"],
    }
    _validate_revision(journal.storage, node_root, row, selected, previous_sha, ordinal + 1)
    journal.storage.write(
        f"{node_root}/revisions/{ordinal + 1:04d}.json", canonical_json_bytes(row)
    )
    return key
