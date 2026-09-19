"""Assemble a prospective ledger from the already executed native daily context."""

from datetime import datetime, timezone
from pathlib import PurePosixPath
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.operations.daily_contract import (
    ContractError,
    GRAPH_SHA256,
    utc_stamp,
    validate_ref,
)
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, _false_authority
from quant_investor.operations.core_pool import CORE_NODES
from quant_investor.operations.core_timing import recorded_core_timing
from quant_investor.operations.daily_timing import recorded_daily_timing
from quant_investor.operations.maintenance_handoff import read_maintenance_handoff
from quant_investor.operations.prospective_sources import read_research_source_times
from quant_investor.operations.prospective_timing import classify_daily_evidence
from quant_investor.operations.decision_recipe import native_portfolio_is_late
from quant_investor.operations.corporate_adapter import corporate_report_is_late
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_completion import _validate_native_context, _replay
from scripts.daily_materialization import verify_materialized_inputs
from quant_investor.operations.materialization_contract import (
    validate_materialization_shape,
    validate_bound_materialization,
)
from scripts.daily_native_inputs import verify_loaded_native_inputs


def assemble_native_ledger(registry, *, materialized, synthetic: bool) -> dict:
    """No publication. Caller owns the day lock and code-owned provenance.

    Native adapters replay before timing is projected. No input decoder or Store
    planner is called again, preserving the materializer's single loaded context.
    """
    journal = registry.runner.journal
    journal._require_lock()
    _validate_native_context(
        registry, native_inputs_ref=materialized.native_inputs_ref, synthetic=synthetic
    )
    if registry.inputs != materialized.inputs:
        raise ContractError("LEDGER_LOADED_CONTEXT_MISMATCH")
    verify_loaded_native_inputs(
        workspace=registry.workspace,
        input_ref=materialized.native_inputs_ref,
        trade_date=registry.trade_date,
        inputs=registry.inputs,
    )
    storage = SecureSystemStorage(registry.workspace)
    observed = {}

    def read(ref):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("LEDGER_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = raw.data
        return parse_canonical_json_bytes(raw.data)

    record = read(materialized.materialization_ref)
    validate_materialization_shape(record)
    if (
        record["trade_date"] != registry.trade_date
        or record["graph_sha256"] != GRAPH_SHA256
        or record["native_inputs_ref"] != materialized.native_inputs_ref
        or record["research_request_ref"] != registry.inputs.research_request_ref
        or record["store_plan_ref"] != registry.inputs.store_plan_ref
        or not _false_authority(record["authority"])
    ):
        raise ContractError("LEDGER_MATERIALIZATION_BINDING_INVALID")
    recovered = read_maintenance_handoff(
        workspace=registry.workspace, handoff_ref=record["maintenance_handoff_ref"]
    )
    verify_materialized_inputs(journal=journal, recovered=recovered, materialized=materialized)
    handoff, recipe = recovered["handoff"], recovered["recipe"]
    execution = PurePosixPath(record["maintenance_handoff_ref"]["path"]).parent
    validate_bound_materialization(
        workspace=registry.workspace,
        value=record,
        recipe=recipe,
        execution=execution,
        path=materialized.materialization_ref["path"],
    )
    if (
        handoff["trade_date"] != registry.trade_date
        or handoff["release_ref"] != registry.inputs.release_ref
        or handoff["calendar_ref"] != registry.inputs.calendar_ref
        or handoff["factor_pointer_ref"]["sha256"] != registry.inputs.factor_pointer_sha256
        or handoff["market_snapshot_ref"] != registry.inputs.market_snapshot_ref
    ):
        raise ContractError("LEDGER_HANDOFF_CONTEXT_INVALID")
    read(materialized.native_inputs_ref)
    refs = _replay(registry)
    completed = {}
    from quant_investor.operations.daily_contract import GRAPH

    for spec in GRAPH:
        if spec.eod_required:
            request = registry.runner._request(
                registry.templates[spec.node_id], spec.node_id, completed
            )
            row = journal.inspect(request)
            if row.get("terminal_ref") != refs[spec.node_id]:
                raise ContractError("LEDGER_TERMINAL_CHANGED")
            completed[spec.node_id] = row
    pointer_ref = registry.core.pointer_ref
    pointer = read(pointer_ref)
    if read(handoff["factor_pointer_ref"]) != pointer:
        raise ContractError("LEDGER_FACTOR_POINTER_BYTES_MISMATCH")
    snapshot = registry.core.snapshot()
    observation_refs = {
        alias: completed[node]["terminal"]["output_refs"][alias]
        for alias, node in [("LOW", "low_observation"), ("W80", "w80_observation")]
    }
    core = recorded_core_timing(
        pointer=pointer,
        pointer_ref=pointer_ref,
        generation=snapshot["factor_generation"],
        observations=[read(observation_refs[alias]) for alias in ("LOW", "W80")],
        observation_refs=observation_refs,
        terminal_refs={node: refs[node] for node in CORE_NODES},
        terminals={node: completed[node]["terminal"] for node in CORE_NODES},
    )
    published = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if (
        not utc_stamp(handoff["sealed_at"])
        <= utc_stamp(record["sealed_at"])
        <= utc_stamp(published)
    ):
        raise ContractError("LEDGER_SEAL_CHRONOLOGY_INVALID")
    custody = recorded_daily_timing(
        trade_date=registry.trade_date, nodes=completed, terminal_refs=refs, verified_at=published
    )
    sources = read_research_source_times(
        workspace=registry.workspace, request_ref=registry.inputs.research_request_ref
    )
    policy_ref = handoff.get("prospective_policy_ref")
    policy = read(policy_ref) if policy_ref is not None else None
    portfolio_late = native_portfolio_is_late(
        workspace=registry.workspace,
        inputs=read(materialized.native_inputs_ref),
    )
    native_document = read(materialized.native_inputs_ref)
    corporate_late = corporate_report_is_late(
        inputs=native_document,
        report=(
            read(completed["corporate_action_recon"]["terminal"]["output_refs"]["reconciliation"])
            if native_document["schema_version"]
            in {
                "cn-daily-native-inputs.v4",
                "cn-daily-native-inputs.v5",
                "cn-daily-native-inputs.v6",
                "cn-daily-native-inputs.v7",
            }
            else None
        ),
    )
    classification = classify_daily_evidence(
        trade_date=registry.trade_date,
        handoff=handoff,
        recipe=recipe,
        policy=policy,
        core_timing=core,
        node_custody=custody,
        source_times=sources,
        synthetic=synthetic,
        portfolio_late=portfolio_late,
        corporate_late=corporate_late,
    )
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
            raise ContractError("LEDGER_SOURCE_CHANGED_DURING_ASSEMBLY")
    published = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return dict(
        schema_version="cn-daily-evidence-ledger.v1",
        trade_date=registry.trade_date,
        graph_sha256=GRAPH_SHA256,
        release_ref=dict(registry.inputs.release_ref),
        materialization_ref=materialized.materialization_ref,
        native_inputs_ref=materialized.native_inputs_ref,
        calendar_ref=dict(registry.inputs.calendar_ref),
        policy_ref=policy_ref,
        policy_custody_at=handoff["sealed_at"] if policy_ref is not None else None,
        node_terminal_refs=refs,
        source_times=sources,
        core_timing=core,
        node_custody=custody,
        synthetic=synthetic,
        recomputed=synthetic
        or portfolio_late
        or corporate_late
        or recipe["retrospective_ref"] is not None
        or handoff["schema_version"] == "cn-daily-maintenance-handoff.v3",
        classification=classification["classification"],
        prospective=classification["prospective"],
        authority=FALSE_AUTHORITY,
        published_at=published,
    )


def _validate_ledger_bytes(raw: bytes, expected: dict, *, materialization: dict) -> dict:
    """Compare with fresh native replay while preserving the first publication clock."""
    from quant_investor.contracts import canonical_json_bytes

    value = parse_canonical_json_bytes(raw)
    if type(value) is not dict or set(value) != set(expected):
        raise ContractError("LEDGER_IMMUTABLE_CONFLICT")
    original = {key: item for key, item in value.items() if key != "published_at"}
    replayed = {key: item for key, item in expected.items() if key != "published_at"}
    if canonical_json_bytes(original) != canonical_json_bytes(replayed):
        raise ContractError("LEDGER_IMMUTABLE_CONFLICT")
    published = utc_stamp(value["published_at"])
    lower = [utc_stamp(materialization["sealed_at"])]
    lower.extend(utc_stamp(row["completed_at"]) for row in value["node_custody"]["nodes"].values())
    if value["policy_custody_at"] is not None:
        lower.append(utc_stamp(value["policy_custody_at"]))
    if not max(lower) <= published <= datetime.now(timezone.utc):
        raise ContractError("LEDGER_PUBLICATION_TIME_INVALID")
    return value


def publish_native_ledger(registry, *, materialized, synthetic: bool) -> dict:
    """Publish once or adopt exact replayed bytes; caller holds the existing day lock.

    This internal pre-EOD step cannot publish or upgrade a completion. Recovery of
    an already sealed EOD belongs to the full completion reader.
    """
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.operations.prospective_storage import ProspectiveLedgerStorage

    journal = registry.runner.journal
    journal._require_lock()
    occupied = journal.storage.read(str(journal.root / "completion.v1.json"))
    if occupied is not None:
        value = parse_canonical_json_bytes(occupied.data)
        if value.get("schema_version") == "cn-daily-eod-completion.v1":
            raise ContractError("EOD_LEGACY_PATH_OCCUPIED")
        raise ContractError("LEDGER_COMPLETED_EOD_REPLAY_REQUIRED")
    storage = ProspectiveLedgerStorage(registry.workspace)
    existing = storage.read(registry.trade_date)
    expected = assemble_native_ledger(registry, materialized=materialized, synthetic=synthetic)
    ref = validate_ref(materialized.materialization_ref)
    record = journal.storage.read(ref["path"])
    if record is None or record.byte_sha256 != ref["sha256"]:
        raise ContractError("LEDGER_MATERIALIZATION_CHANGED")
    materialization = parse_canonical_json_bytes(record.data)
    if existing is not None:
        document = _validate_ledger_bytes(existing.data, expected, materialization=materialization)
        if storage.read(registry.trade_date) != existing:
            raise ContractError("LEDGER_CHANGED_DURING_REPLAY")
        stored, status = existing, "NO_ACTION"
    else:
        # A completion appearing despite the day lock is an integrity failure.
        if journal.storage.read(str(journal.root / "completion.v1.json")) is not None:
            raise ContractError("LEDGER_COMPLETION_CHANGED")
        expected["published_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = canonical_json_bytes(expected)
        _validate_ledger_bytes(raw, expected, materialization=materialization)
        stored = storage.write(journal=journal, raw=raw)
        document = _validate_ledger_bytes(stored.data, expected, materialization=materialization)
        status = "PUBLISHED"
    return {
        "status": status,
        "ledger_ref": {"path": stored.relative_path, "sha256": stored.byte_sha256},
        "ledger": document,
        "execution_authorized": False,
    }
