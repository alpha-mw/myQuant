"""Publish only an exact latest registered native EOD; never rerun business producers."""

from dataclasses import dataclass, InitVar, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import json

from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.system.errors import SystemStorageError, SystemNotFound
from quant_investor.operations.daily_contract import ContractError, validate_ref, utc_stamp
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority
from quant_investor.operations.dashboard_serving_contract import (
    POLICY,
    PREFIX,
    HEAD_JSON,
    HEAD_JS,
    EVIDENCE_JSON,
    EVIDENCE_JS,
    SELECTOR_JSON,
    SELECTOR_JS,
    validate_selector,
    FINANCIAL_NAMES,
    digest,
    sealed,
    instant,
    head_bytes,
    head_js,
    raw_js,
    build_head,
    build_selector,
    REGISTERED_VIEW_FIELDS,
    validate_registered_serving_fields,
)
from quant_investor.operations.dashboard_publication_guard import sealed_publication_scope
from quant_investor.operations.dashboard_evidence import KIND

_PROOF_KEY = object()


@dataclass(frozen=True)
class VerifiedDashboard:
    workspace: Path
    day: str
    completion_ref: dict
    completion: dict
    inputs: dict
    terminal_ref: dict
    output_refs: dict
    v1: bytes
    v2: bytes
    evidence: dict
    cutoff: str
    head_preimage: bytes | None
    predecessor_refs: tuple
    anchor_ref: dict
    registered: dict | None = None
    _key: InitVar[object] = None
    _seal: object = field(init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _key):
        if _key is not _PROOF_KEY:
            raise ContractError("DASHBOARD_VERIFICATION_FACTORY_REQUIRED")
        object.__setattr__(self, "_seal", _key)
        object.__setattr__(self, "_fingerprint", self.fingerprint())

    def fingerprint(self):
        payload = {}
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            value = getattr(self, item.name)
            if type(value) is tuple:
                value = list(value)
            payload[item.name] = (
                {"byte_sha256": digest(value)}
                if type(value) is bytes
                else str(value) if isinstance(value, Path) else value
            )
        return digest(canonical_json_bytes(payload))

    def require(self):
        if self._seal is not _PROOF_KEY or self.fingerprint() != self._fingerprint:
            raise ContractError("DASHBOARD_VERIFICATION_PROOF_CHANGED")

    def __reduce__(self):
        raise TypeError("Dashboard verification proofs cannot be serialized")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dashboard_output_roles(inputs):
    roles = {"capture", "v1", "v2", "daily_evidence"}
    if inputs["schema_version"] == "cn-daily-native-inputs.v7":
        roles.add("registered_transition")
    return roles


def _optional(root, path):
    try:
        return (
            SecureSystemStorage(root)
            .read_workspace_file_bytes(path, maximum_bytes=64 * 1024 * 1024)
            .data
        )
    except (FileNotFoundError, SystemNotFound):
        return None
    except SystemStorageError as exc:
        if type(exc) is SystemStorageError and isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise


def _read(root, ref):
    ref = validate_ref(ref)
    raw = _optional(root, ref["path"])
    if raw is None or digest(raw) != ref["sha256"]:
        raise ContractError("DASHBOARD_SEALED_SOURCE_SHA_MISMATCH")
    return raw


def _document(root, ref):
    return parse_canonical_json_bytes(_read(root, ref))


def _inspect(root, ref):
    from quant_investor.operations.completion_readback import inspect_recorded_completion

    day = PurePosixPath(ref["path"]).parent.name
    return inspect_recorded_completion(workspace=str(root), trade_date=day, completion_ref=ref)


def _replay(root, ref):
    from scripts.daily_completion_replay import replay_native_completion

    value = replay_native_completion(
        workspace=str(root), trade_date=PurePosixPath(ref["path"]).parent.name, completion_ref=ref
    )
    if value.get("native_replay_validated") is not True or value.get("completion_ref") != ref:
        raise ContractError("DASHBOARD_NATIVE_EOD_REQUIRED")
    return value


def _predecessors(root, inspected, head):
    snapshot = inspected.get("completed_handoff_snapshot")
    if snapshot is None:
        raise ContractError("DASHBOARD_NEW_PROFILE_REQUIRES_MATERIALIZED_EOD")
    recipe = snapshot.document("recipe")
    current = inspected["recorded_completion"]["trade_date"]
    previous = recipe["previous_completion_ref"]
    if previous is None:
        if head is not None or recipe["bootstrap_ref"] is None:
            raise ContractError("DASHBOARD_HEAD_ANCESTRY_CONFLICT")
        from quant_investor.operations.maintenance_handoff import read_recorded_maintenance_handoff
        from quant_investor.operations.bootstrap import verify_bootstrap_native

        verify_bootstrap_native(
            workspace=str(root), recovered=read_recorded_maintenance_handoff(snapshot)
        )
        return ()
    seen = set()
    refs = []
    for _ in range(32):
        validate_ref(previous)
        identity = (previous["path"], previous["sha256"])
        day = PurePosixPath(previous["path"]).parent.name
        if identity in seen or day >= current:
            raise ContractError("DASHBOARD_HEAD_PREDECESSOR_CYCLE_OR_DATE")
        seen.add(identity)
        refs.append(dict(previous))
        _replay(root, previous)
        if head is None or previous == head["completion_ref"]:
            return tuple(refs)
        prior = _inspect(root, previous)
        source = prior.get("completed_handoff_snapshot")
        if source is None:
            raise ContractError("DASHBOARD_HEAD_PREDECESSOR_LINK_MISSING")
        previous = source.document("recipe")["previous_completion_ref"]
        if previous is None:
            raise ContractError("DASHBOARD_HEAD_ANCESTOR_NOT_REACHED")
        current = day
    raise ContractError("DASHBOARD_HEAD_CHAIN_LIMIT")


def verify_dashboard_for_publication(workspace, completion_ref):
    root = Path(workspace).resolve(strict=True)
    inspected = _inspect(root, completion_ref)
    completion = inspected["recorded_completion"]
    inputs = _document(root, completion["native_inputs_ref"])
    replay = _replay(root, completion_ref)
    if (
        inputs["schema_version"]
        not in {
            "cn-daily-native-inputs.v5",
            "cn-daily-native-inputs.v6",
            "cn-daily-native-inputs.v7",
        }
        or not inputs["publish_current_dashboard"]
    ):
        return None
    if inputs["dashboard_publication_policy"] != POLICY:
        raise ContractError("DASHBOARD_PUBLICATION_POLICY_INVALID")
    terminal_ref = completion["node_terminal_refs"]["dashboard"]
    terminal = _document(root, terminal_ref)
    outputs = terminal["output_refs"]
    if set(outputs) != _dashboard_output_roles(inputs):
        raise ContractError("DASHBOARD_SEALED_OUTPUT_SET_INVALID")
    v1, v2 = _read(root, outputs["v1"]), _read(root, outputs["v2"])
    if json.loads(v1) != replay["dashboard"]["v1"] or json.loads(v2) != replay["dashboard"]["v2"]:
        raise ContractError("DASHBOARD_SEALED_REPLAY_DIFFERS")
    evidence = validate_artifact(_read(root, outputs["daily_evidence"]), expected_kind=KIND)
    decision_ref = evidence["payload"]["source_bindings"]["decision"]["output_refs"][
        "decision.v2.json"
    ]
    cutoff = validate_artifact(
        _read(root, decision_ref), expected_kind="daily_research_decision_report"
    )["payload"]["as_of"]
    registered = None
    if inputs["schema_version"] == "cn-daily-native-inputs.v7":
        dashboard = replay["dashboard"]
        if not REGISTERED_VIEW_FIELDS <= set(dashboard):
            raise ContractError("DASHBOARD_REGISTERED_REPLAY_REQUIRED")
        registered = {k: dashboard[k] for k in REGISTERED_VIEW_FIELDS}
        if registered["registered_transition_ref"] != outputs["registered_transition"]:
            raise ContractError("DASHBOARD_REGISTERED_OUTPUT_MISMATCH")
    raw = _optional(root, PREFIX + "/" + HEAD_JSON)
    head = None if raw is None else head_bytes(raw)
    if head is not None and (
        head["trade_date"] > completion["trade_date"]
        or (
            head["trade_date"] == completion["trade_date"]
            and head["completion_ref"] != completion_ref
        )
    ):
        raise ContractError("DASHBOARD_STALE_COMPLETED_CANDIDATE")
    predecessors = (
        ()
        if head is not None and head["completion_ref"] == completion_ref
        else _predecessors(root, inspected, head)
    )
    recipe = inspected["completed_handoff_snapshot"].document("recipe")
    anchor = validate_ref(recipe["previous_completion_ref"] or recipe["bootstrap_ref"])
    return VerifiedDashboard(
        root,
        completion["trade_date"],
        dict(completion_ref),
        completion,
        inputs,
        terminal_ref,
        outputs,
        v1,
        v2,
        evidence,
        cutoff,
        raw,
        predecessors,
        anchor,
        registered=registered,
        _key=_PROOF_KEY,
    )


def _replace(cap, name, raw):
    from scripts.cn_dashboard_v2_selector import _atomic_replace

    path = cap.root / PREFIX / name
    cap.validate_bytes(path, raw)
    if _optional(cap.root, PREFIX + "/" + name) != raw:
        _atomic_replace(path, raw)
    if _optional(cap.root, PREFIX + "/" + name) != raw:
        raise ContractError("DASHBOARD_SERVING_WRITE_READBACK_DIFFERS")


def _proposal(root):
    raw = _optional(root, PREFIX + "/" + HEAD_JS)
    if raw is None:
        return None
    prefix = b"window.MyQuantCNDailyCompletedHeadRaw = "
    suffix = (
        b";\nwindow.MyQuantCNDailyCompletedHead = "
        b"JSON.parse(window.MyQuantCNDailyCompletedHeadRaw);\n"
    )
    if not raw.startswith(prefix) or not raw.endswith(suffix):
        raise ContractError("DASHBOARD_HEAD_MIRROR_INVALID")
    start, end = len(prefix), len(raw) - len(suffix)
    text = json.loads(raw[start:end])
    value = head_bytes(text.encode("utf-8"))
    if head_js(canonical_json_bytes(value)) != raw:
        raise ContractError("DASHBOARD_HEAD_MIRROR_INVALID")
    return value


def _register_head(proof, cap, journal):
    current = _optional(proof.workspace, PREFIX + "/" + HEAD_JSON)
    if current != proof.head_preimage:
        raise ContractError("DASHBOARD_HEAD_PREIMAGE_CHANGED")
    old = None if current is None else head_bytes(current)
    if (
        current is None
        and journal.storage.read(str(journal.root / "dashboard/serving-intent.v1.json")) is not None
    ):
        raise ContractError("DASHBOARD_HEAD_JSON_MISSING")
    validate_ref(proof.anchor_ref)
    if old is not None:
        if old["trade_date"] > proof.day or (
            old["trade_date"] == proof.day and old["completion_ref"] != proof.completion_ref
        ):
            raise ContractError("DASHBOARD_STALE_COMPLETED_CANDIDATE")
        if old["trade_date"] < proof.day and old["completion_ref"] not in proof.predecessor_refs:
            raise ContractError("DASHBOARD_HEAD_ANCESTRY_CONFLICT")
    if old is not None and old["completion_ref"] == proof.completion_ref:
        head = old
        raw = current
        if _optional(proof.workspace, PREFIX + "/" + HEAD_JS) != head_js(raw):
            raise ContractError("DASHBOARD_COMMITTED_HEAD_MIRROR_DIFFERS")
        if utc_stamp(head["registered_at"]) < utc_stamp(
            proof.completion["native_validation_completed_at"]
        ):
            raise ContractError("DASHBOARD_HEAD_BEFORE_EOD_SEAL")
    else:
        proposal = _proposal(proof.workspace)
        previous_sha = None if current is None else digest(current)
        if proposal is not None and (old is None or proposal != old):
            if (
                proposal["trade_date"] != proof.day
                or proposal["completion_ref"] != proof.completion_ref
                or proposal["previous_head_sha256"] != previous_sha
            ):
                raise ContractError("DASHBOARD_HEAD_PROPOSAL_REQUIRES_RECOVERY")
            head = proposal
        else:
            head = build_head(
                day=proof.day,
                ref=proof.completion_ref,
                previous_sha=previous_sha,
                registered_at=_now(),
            )
        if utc_stamp(head["registered_at"]) < utc_stamp(
            proof.completion["native_validation_completed_at"]
        ):
            raise ContractError("DASHBOARD_HEAD_BEFORE_EOD_SEAL")
        raw = canonical_json_bytes(head)
    cap.select_head(raw)
    replacing = current is not None and current != raw
    if replacing:
        history = str(journal.root.parent / "_dashboard_heads" / (digest(current) + ".json"))
        journal.storage.write(history, current)
    _replace(cap, HEAD_JS, head_js(raw))
    if _optional(proof.workspace, PREFIX + "/" + HEAD_JSON) != current:
        raise ContractError("DASHBOARD_HEAD_CHANGED_BEFORE_CAS")
    _replace(cap, HEAD_JSON, raw)
    if replacing and journal.storage.read(history).data != current:
        raise ContractError("DASHBOARD_HEAD_HISTORY_CHANGED")
    retained = str(journal.root / "dashboard" / "completed-heads" / (digest(raw) + ".json"))
    stored = journal.storage.write(retained, raw)
    return head, raw, {"path": retained, "sha256": stored.byte_sha256}


def _serving_data_files(proof, head_ref):
    extra = {}
    if proof.inputs["schema_version"] == "cn-daily-native-inputs.v7":
        extra = validate_registered_serving_fields(
            proof.registered,
            day=proof.day,
            cutoff=proof.cutoff,
            final_pointer_ref=proof.evidence["payload"]["source_bindings"]["store"]["output_refs"][
                "pointer"
            ],
        )
        if (
            extra["registered_close_summary"]["final_record_id"]
            != json.loads(proof.v1)["latest_valid_record"]
        ):
            raise ContractError("DASHBOARD_REGISTERED_FINANCIAL_RECORD_MISMATCH")
        if extra["registered_transition_ref"] != proof.output_refs["registered_transition"]:
            raise ContractError("DASHBOARD_REGISTERED_SERVING_OUTPUT_MISMATCH")
    elif getattr(proof, "registered", None) is not None:
        raise ContractError("DASHBOARD_UNEXPECTED_REGISTERED_SERVING_PROFILE")
    view = sealed(
        {
            "schema_version": (
                "cn-daily-dashboard-serving.v2" if extra else "cn-daily-dashboard-serving.v1"
            ),
            **extra,
            "view_designation": "LATEST_COMPLETED_EOD",
            "trade_date": proof.day,
            "research_cutoff": proof.cutoff,
            "completion_ref": proof.completion_ref,
            "completed_head_ref": head_ref,
            "financial_refs": {"v1": proof.output_refs["v1"], "v2": proof.output_refs["v2"]},
            "daily_evidence_ref": proof.output_refs["daily_evidence"],
            "evidence": proof.evidence,
            "native_valid_through": json.loads(proof.v2)["freshness"]["valid_through"],
            "authority": FALSE_AUTHORITY,
        }
    )
    evidence = canonical_json_bytes(view)
    return {
        FINANCIAL_NAMES[0]: proof.v1,
        FINANCIAL_NAMES[1]: raw_js(
            "MyQuantCNAggressiveDashboard", "MyQuantCNDailyFinancialV1Raw", proof.v1
        ),
        FINANCIAL_NAMES[2]: proof.v2,
        FINANCIAL_NAMES[3]: raw_js(
            "MyQuantCNAggressiveDashboardV2", "MyQuantCNDailyFinancialV2Raw", proof.v2
        ),
        EVIDENCE_JSON: evidence,
        EVIDENCE_JS: raw_js(
            "MyQuantCNDailyDashboardEvidence", "MyQuantCNDailyDashboardEvidenceRaw", evidence
        ),
    }


def _selector_files(proof, head, head_raw, data, commit_at):
    selector = build_selector(
        head=head,
        head_sha=digest(head_raw),
        v1_raw=proof.v1,
        v2_raw=proof.v2,
        evidence_raw=data[EVIDENCE_JSON],
        commit_at=commit_at,
    )
    raw = canonical_json_bytes(selector)
    return {
        SELECTOR_JSON: raw,
        SELECTOR_JS: raw_js(
            "MyQuantCNAggressiveDashboardSelectorV2", "MyQuantCNDailyDashboardSelectorRaw", raw
        ),
    }


def _byte_refs(journal, files):
    return {
        name: {
            "path": str(journal.root / "dashboard/serving-bytes" / (digest(raw) + ".bin")),
            "sha256": digest(raw),
        }
        for name, raw in files.items()
    }


def _intent_value(proof, head_raw, head_ref, journal, stamp):
    files = _serving_data_files(proof, head_ref)
    value = {
        "schema_version": "cn-daily-dashboard-serving-intent.v1",
        "trade_date": proof.day,
        "completion_ref": proof.completion_ref,
        "completed_head_ref": head_ref,
        "completed_head_sha256": digest(head_raw),
        "dashboard_terminal_ref": proof.terminal_ref,
        "dashboard_output_refs": proof.output_refs,
        "intent_created_at": stamp,
        "valid_through": json.loads(proof.v2)["freshness"]["valid_through"],
        "files": _byte_refs(journal, files),
        "authority": FALSE_AUTHORITY,
    }
    if (
        not utc_stamp(proof.completion["native_validation_completed_at"])
        <= utc_stamp(head_bytes(head_raw)["registered_at"])
        <= utc_stamp(stamp)
        <= datetime.now(timezone.utc)
    ):
        raise ContractError("DASHBOARD_INTENT_TIME_INVALID")
    return value, files


def _intent(proof, head, head_raw, head_ref, journal):
    path = str(journal.root / "dashboard/serving-intent.v1.json")
    stored = journal.storage.read(path)
    existed = stored is not None
    stamp = (
        _now() if stored is None else parse_canonical_json_bytes(stored.data)["intent_created_at"]
    )
    value, files = _intent_value(proof, head_raw, head_ref, journal, stamp)
    if stored is not None:
        _verify_intent(proof, head_raw, head_ref, journal, stored)
    else:
        previous = _optional(proof.workspace, PREFIX + "/" + SELECTOR_JSON)
        if (
            previous is not None
            and json.loads(previous).get("completion_ref") == proof.completion_ref
        ):
            raise ContractError("DASHBOARD_PUBLICATION_INTENT_MISSING")
        for name, ref in value["files"].items():
            journal.storage.write(ref["path"], files[name])
        stored = journal.storage.write(path, canonical_json_bytes(value))
    return value, {"path": path, "sha256": stored.byte_sha256}, files, existed


def _verify_intent(proof, head_raw, head_ref, journal, stored):
    if stored is None:
        raise ContractError("DASHBOARD_PUBLICATION_INTENT_MISSING")
    value = parse_canonical_json_bytes(stored.data)
    expected, files = _intent_value(proof, head_raw, head_ref, journal, value["intent_created_at"])
    if stored.data != canonical_json_bytes(expected):
        raise ContractError("DASHBOARD_SERVING_INTENT_CONFLICT")
    for name, ref in expected["files"].items():
        if _read(proof.workspace, ref) != files[name]:
            raise ContractError("DASHBOARD_SERVING_INTENT_BYTES_MISMATCH")
    return value, files


def _unexpired(intent):
    if datetime.now(timezone.utc) > instant(intent["valid_through"]):
        raise ContractError("EVIDENCE_SEALED_PUBLICATION_EXPIRED")


def _commit_value(proof, head_ref, intent, intent_ref, refs, selector_at, verified_at, recovered):
    return sealed(
        {
            "schema_version": "cn-daily-dashboard-selector-commit.v1",
            "trade_date": proof.day,
            "completion_ref": proof.completion_ref,
            "completed_head_ref": head_ref,
            "completed_head_sha256": head_ref["sha256"],
            "intent_ref": intent_ref,
            "selector_files": refs,
            "selector_updated_at": selector_at,
            "first_verified_at": verified_at,
            "valid_through": intent["valid_through"],
            "recovered_unknown": recovered,
            "authority": FALSE_AUTHORITY,
        }
    )


def _validate_commit(value, proof, head_raw, head_ref, intent, intent_ref, journal, data):
    stamp = value["selector_updated_at"]
    files = _selector_files(proof, head_bytes(head_raw), head_raw, data, stamp)
    expected = _commit_value(
        proof,
        head_ref,
        intent,
        intent_ref,
        _byte_refs(journal, files),
        stamp,
        value["first_verified_at"],
        value["recovered_unknown"],
    )
    if value != expected or type(value["recovered_unknown"]) is not bool:
        raise ContractError("DASHBOARD_SELECTOR_COMMIT_CONFLICT")
    if not utc_stamp(intent["intent_created_at"]) <= utc_stamp(stamp) <= utc_stamp(
        value["first_verified_at"]
    ) <= instant(intent["valid_through"]) or utc_stamp(value["first_verified_at"]) > datetime.now(
        timezone.utc
    ):
        raise ContractError("DASHBOARD_SELECTOR_COMMIT_TIME_INVALID")
    for name, ref in value["selector_files"].items():
        if _read(proof.workspace, ref) != files[name]:
            raise ContractError("DASHBOARD_SELECTOR_COMMIT_BYTES_DIFFER")
    return files


def _readback(root, files, head_raw):
    if _optional(root, PREFIX + "/" + HEAD_JSON) != head_raw or _optional(
        root, PREFIX + "/" + HEAD_JS
    ) != head_js(head_raw):
        raise ContractError("DASHBOARD_HEAD_READBACK_DIFFERS")
    if any(_optional(root, PREFIX + "/" + name) != raw for name, raw in files.items()):
        raise ContractError("DASHBOARD_SERVING_FINAL_READBACK_DIFFERS")


def _receipt_value(proof, head_ref, intent_ref, intent, commit_ref, commit, recorded_at):
    return {
        "schema_version": "cn-daily-dashboard-serving-publication.v2",
        "view_designation": "LATEST_COMPLETED_EOD",
        "trade_date": proof.day,
        "completion_ref": proof.completion_ref,
        "completed_head_ref": head_ref,
        "intent_ref": intent_ref,
        "selector_commit_ref": commit_ref,
        "selector_updated_at": commit["selector_updated_at"],
        "published_at": commit["first_verified_at"],
        "first_verified_at": commit["first_verified_at"],
        "receipt_recorded_at": recorded_at,
        "recovered_unknown": commit["recovered_unknown"],
        "files": {**intent["files"], **commit["selector_files"]},
        "authority": FALSE_AUTHORITY,
    }


def _validate_receipt(value, proof, head_ref, intent_ref, intent, commit_ref, commit):
    expected = _receipt_value(
        proof, head_ref, intent_ref, intent, commit_ref, commit, value["receipt_recorded_at"]
    )
    if (
        value != expected
        or type(value["recovered_unknown"]) is not bool
        or not _false_authority(value["authority"])
    ):
        raise ContractError("DASHBOARD_SERVING_RECEIPT_CONFLICT")
    if (
        not utc_stamp(value["first_verified_at"])
        <= utc_stamp(value["receipt_recorded_at"])
        <= datetime.now(timezone.utc)
    ):
        raise ContractError("DASHBOARD_SERVING_RECEIPT_TIME_INVALID")


def _read_publication_records(proof, journal, head_raw, head_ref, receipt):
    """Shared read-only closure, used by publication replay and status observation."""
    intent_path = str(journal.root / "dashboard/serving-intent.v1.json")
    stored = journal.storage.read(intent_path)
    intent, data = _verify_intent(proof, head_raw, head_ref, journal, stored)
    intent_ref = {"path": intent_path, "sha256": stored.byte_sha256}
    commit_path = str(journal.root / "dashboard/selector-commit.v1.json")
    stored_commit = journal.storage.read(commit_path)
    if stored_commit is None:
        raise ContractError("DASHBOARD_SELECTOR_COMMIT_MISSING")
    commit = parse_canonical_json_bytes(stored_commit.data)
    selectors = _validate_commit(
        commit, proof, head_raw, head_ref, intent, intent_ref, journal, data
    )
    commit_ref = {"path": commit_path, "sha256": stored_commit.byte_sha256}
    _validate_receipt(receipt, proof, head_ref, intent_ref, intent, commit_ref, commit)
    _readback(proof.workspace, {**data, **selectors}, head_raw)
    return intent


def _select_commit(
    proof, cap, journal, head, head_raw, head_ref, intent, intent_ref, data, existed
):
    from scripts.cn_dashboard_v2_selector import publish_selector

    _unexpired(intent)
    # Only a complete pair can retain its actual prior commit-attempt clock.
    existing = _optional(proof.workspace, PREFIX + "/" + SELECTOR_JSON)
    selectors = None
    if existing is not None:
        try:
            candidate = validate_selector(parse_canonical_json_bytes(existing))
            stamp = (
                instant(candidate["updated_at"])
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            expected = _selector_files(proof, head, head_raw, data, stamp)
            if utc_stamp(intent["intent_created_at"]) <= utc_stamp(stamp) <= datetime.now(
                timezone.utc
            ) and all(
                _optional(proof.workspace, PREFIX + "/" + name) == raw
                for name, raw in expected.items()
            ):
                selectors = expected
        except (ValueError, TypeError, KeyError):
            pass
    if selectors is None:
        stamp = _now()
        selectors = _selector_files(proof, head, head_raw, data, stamp)
        cap.select_selector(stamp)
        _unexpired(intent)
        publish_selector(
            json.loads(selectors[SELECTOR_JSON]),
            json_path=proof.workspace / PREFIX / SELECTOR_JSON,
            js_path=proof.workspace / PREFIX / SELECTOR_JS,
            project_root=proof.workspace,
            js_first=False,
            _capability=cap,
        )
    _readback(proof.workspace, {**data, **selectors}, head_raw)
    _unexpired(intent)
    verified_at = _now()
    refs = _byte_refs(journal, selectors)
    commit = _commit_value(proof, head_ref, intent, intent_ref, refs, stamp, verified_at, existed)
    for name, ref in refs.items():
        journal.storage.write(ref["path"], selectors[name])
    _validate_commit(commit, proof, head_raw, head_ref, intent, intent_ref, journal, data)
    _unexpired(intent)
    path = str(journal.root / "dashboard/selector-commit.v1.json")
    stored = journal.storage.write(path, canonical_json_bytes(commit))
    return commit, {"path": path, "sha256": stored.byte_sha256}


def _publish_locked(proof, cap, journal):
    head, head_raw, head_ref = _register_head(proof, cap, journal)
    receipt_path = str(journal.root / "dashboard/serving-publication.v2.json")
    old = journal.storage.read(receipt_path)
    if old is not None:
        value = parse_canonical_json_bytes(old.data)
        _read_publication_records(proof, journal, head_raw, head_ref, value)
        return {"path": receipt_path, "sha256": old.byte_sha256}
    intent, intent_ref, data, existed = _intent(proof, head, head_raw, head_ref, journal)
    commit_path = str(journal.root / "dashboard/selector-commit.v1.json")
    stored_commit = journal.storage.read(commit_path)
    if stored_commit is not None:
        commit = parse_canonical_json_bytes(stored_commit.data)
        selectors = _validate_commit(
            commit, proof, head_raw, head_ref, intent, intent_ref, journal, data
        )
        _readback(proof.workspace, {**data, **selectors}, head_raw)
        commit_ref = {"path": commit_path, "sha256": stored_commit.byte_sha256}
    else:
        _unexpired(intent)
        from scripts.export_cn_aggressive_dashboard_data import publish_bundle_pair

        paths = [proof.workspace / PREFIX / name for name in FINANCIAL_NAMES]
        publish_bundle_pair(
            v1_bundle=json.loads(proof.v1),
            v2_bundle=json.loads(proof.v2),
            v1_json_path=paths[0],
            v1_js_path=paths[1],
            v2_json_path=paths[2],
            v2_js_path=paths[3],
            project_root=proof.workspace,
            _capability=cap,
        )
        for name in (EVIDENCE_JSON, EVIDENCE_JS):
            _replace(cap, name, data[name])
        _readback(proof.workspace, data, head_raw)
        commit, commit_ref = _select_commit(
            proof, cap, journal, head, head_raw, head_ref, intent, intent_ref, data, existed
        )
    # A pre-expiry sealed commit can acquire its missing receipt as historical metadata.
    receipt = _receipt_value(proof, head_ref, intent_ref, intent, commit_ref, commit, _now())
    _read_publication_records(proof, journal, head_raw, head_ref, receipt)
    stored = journal.storage.write(receipt_path, canonical_json_bytes(receipt))
    return {"path": receipt_path, "sha256": stored.byte_sha256}


def publish_completed_dashboard(*, workspace, completion_ref, journal=None):
    proof = verify_dashboard_for_publication(workspace, completion_ref)
    if proof is None:
        return {"status": "NOT_REQUIRED", "publication_ref": None}
    if journal is None:
        journal = DailyJournal(str(proof.workspace), proof.day)
        with journal.locked():
            with sealed_publication_scope(proof.workspace, proof) as cap:
                ref = _publish_locked(proof, cap, journal)
    else:
        journal._require_lock()
        if journal.trade_date != proof.day or journal.storage._io.workspace_root != proof.workspace:
            raise ContractError("DASHBOARD_PUBLICATION_JOURNAL_MISMATCH")
        with sealed_publication_scope(proof.workspace, proof) as cap:
            ref = _publish_locked(proof, cap, journal)
    expired = datetime.now(timezone.utc) > instant(
        json.loads(proof.v2)["freshness"]["valid_through"]
    )
    return {
        "status": "PUBLISHED_HISTORICAL_EXPIRED" if expired else "PUBLISHED",
        "publication_ref": ref,
    }


def complete_serving_result(*, workspace, completion_ref, base=None, journal=None):
    """Execution-only final gate; a recorded EOD is not a successful current publication."""
    value = dict(base or {})
    try:
        root = Path(workspace).resolve(strict=True)
        completed = _document(root, completion_ref)
        inputs = _document(root, completed["native_inputs_ref"])
        from quant_investor.operations.native_input_contract import validate_native_input_shape

        validate_native_input_shape(inputs)
        publication = (
            publish_completed_dashboard(
                workspace=workspace, completion_ref=completion_ref, journal=journal
            )
            if inputs["schema_version"]
            in {
                "cn-daily-native-inputs.v5",
                "cn-daily-native-inputs.v6",
                "cn-daily-native-inputs.v7",
            }
            and inputs["publish_current_dashboard"]
            else {"status": "NOT_REQUIRED", "publication_ref": None}
        )
        if publication["status"] == "PUBLISHED_HISTORICAL_EXPIRED":
            raise ContractError("EVIDENCE_SEALED_PUBLICATION_EXPIRED")
    except Exception as exc:
        recorded = None
        try:
            _read(Path(workspace).resolve(strict=True), completion_ref)
            recorded = completion_ref
        except Exception:
            pass
        reason = str(exc)
        status = (
            "EVIDENCE_SEALED_PUBLICATION_EXPIRED"
            if "PUBLICATION_EXPIRED" in reason
            else "EVIDENCE_SEALED_PUBLICATION_PENDING" if recorded else "EOD_REVALIDATION_FAILED"
        )
        value.update(
            status="PARTIAL",
            completion_ref=None,
            sealed_evidence_ref=recorded,
            completion_status=status,
            publication_error={"code": type(exc).__name__, "detail": reason},
        )
        return value
    value.update(
        status="COMPLETE",
        completion_ref=completion_ref,
        completion_status="NATIVE_COMPLETION_VALIDATED",
    )
    if publication["publication_ref"] is not None:
        value["dashboard_publication_ref"] = publication["publication_ref"]
    return value


def _recorded_serving_sources(root, day, ref, completion, inputs):
    """Load recorded source bytes for observation; this does not mint a capability."""
    from types import SimpleNamespace

    terminal_ref = completion["node_terminal_refs"]["dashboard"]
    outputs = _document(root, terminal_ref)["output_refs"]
    if set(outputs) != _dashboard_output_roles(inputs):
        raise ContractError("DASHBOARD_SEALED_OUTPUT_SET_INVALID")
    evidence = validate_artifact(_read(root, outputs["daily_evidence"]), expected_kind=KIND)
    decision_ref = evidence["payload"]["source_bindings"]["decision"]["output_refs"][
        "decision.v2.json"
    ]
    cutoff = validate_artifact(
        _read(root, decision_ref), expected_kind="daily_research_decision_report"
    )["payload"]["as_of"]
    registered = None
    if inputs["schema_version"] == "cn-daily-native-inputs.v7":
        from quant_investor.operations.registered_dashboard import recorded_registered_view

        registered = recorded_registered_view(workspace=root, completion=completion, inputs=inputs)
    return SimpleNamespace(
        workspace=root,
        day=day,
        completion_ref=ref,
        completion=completion,
        inputs=inputs,
        terminal_ref=terminal_ref,
        output_refs=outputs,
        v1=_read(root, outputs["v1"]),
        v2=_read(root, outputs["v2"]),
        evidence=evidence,
        cutoff=cutoff,
        registered=registered,
    )


def observed_serving_status(workspace, trade_date):
    """Read-only record/byte closure; never native admission or publication."""
    root = Path(workspace).resolve(strict=True)
    journal = DailyJournal(str(root), trade_date)
    path = str(journal.root / "completion.v1.json")
    raw = journal.storage.read(path)
    if raw is None:
        return None
    completion = parse_canonical_json_bytes(raw.data)
    inputs = _document(root, completion["native_inputs_ref"])
    if (
        inputs["schema_version"]
        not in {
            "cn-daily-native-inputs.v5",
            "cn-daily-native-inputs.v6",
            "cn-daily-native-inputs.v7",
        }
        or not inputs["publish_current_dashboard"]
    ):
        return None
    ref = {"path": path, "sha256": raw.byte_sha256}
    proof = _recorded_serving_sources(root, trade_date, ref, completion, inputs)
    result = {
        "sealed_evidence_ref": ref,
        "publication_state": "EVIDENCE_SEALED_PUBLICATION_PENDING",
        "publication_ref": None,
        "validation_scope": "RECORDED_SERVING_BYTES_ONLY",
    }
    receipt = journal.storage.read(str(journal.root / "dashboard/serving-publication.v2.json"))
    expired = datetime.now(timezone.utc) > instant(
        json.loads(proof.v2)["freshness"]["valid_through"]
    )
    if receipt is None:
        if expired:
            result["publication_state"] = "EVIDENCE_SEALED_PUBLICATION_EXPIRED"
        return result
    value = parse_canonical_json_bytes(receipt.data)
    head_ref = validate_ref(value["completed_head_ref"])
    head_raw = _read(root, head_ref)
    expected_path = str(journal.root / "dashboard/completed-heads" / (digest(head_raw) + ".json"))
    if head_ref["path"] != expected_path or head_bytes(head_raw)["completion_ref"] != ref:
        raise ContractError("DASHBOARD_SERVING_RECEIPT_HEAD_INVALID")
    # Identical full validator to successful publication replay, including final bytes.
    _read_publication_records(proof, journal, head_raw, head_ref, value)
    result.update(
        publication_state=(
            "RECORDED_EOD_PUBLICATION_EXPIRED" if expired else "RECORDED_EOD_PUBLICATION"
        ),
        publication_ref={"path": receipt.relative_path, "sha256": receipt.byte_sha256},
    )
    return result
