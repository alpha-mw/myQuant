"""Commit source custody before installing exact portfolio and research input bytes."""

from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
import hashlib
import time

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.intelligence.portfolio_state import build_portfolio_state
from quant_investor.intelligence.fundamental_time import SHANGHAI
from .dependency_diagnostics import DependencyInputError
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import FALSE_AUTHORITY
from .portfolio_binding import NativePortfolioSource, PortfolioSource, retain_portfolio_source
from .research_source_bundle import SourceBundle, build_source_bundle
from .research_corporate_inputs import derive_corporate_inputs
from .research_timing import CURRENT, assert_fresh_acquisition_open
from .research_cutoff_contract import (
    SCHEMA,
    SCHEMA_V2,
    validate_cutoff_contract,
    ordered_source_refs,
    ordered_source_times,
)
from .exposure_completion import POLICY
from .research_file_readback import parse_research_json

WRAPPER_SCHEMA = "cn-daily-research-request.v2"
WRAPPER_SCHEMA_V3 = "cn-daily-research-request.v3"


def cutoff_version(recipe):
    versions = {"cn-daily-execute-recipe.v5": 1, "cn-daily-execute-recipe.v6": 2}
    version = versions.get(recipe.get("schema_version"))
    if version is None:
        raise ContractError("CUTOFF_RECIPE_VERSION_INVALID")
    return version


def _selected_cutoff(journal, execution, version):
    if journal.storage.read(str(execution / f"research-cutoff.v{3 - version}.json")) is not None:
        raise ContractError("CUTOFF_VERSION_CONFLICT")
    path = str(execution / f"research-cutoff.v{version}.json")
    return path, journal.storage.read(path)


def _clock():
    value = datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError("CUTOFF_CLOCK_UNAWARE")
    return value.astimezone(timezone.utc)


def _stamp(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ref(path, raw):
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _read(journal, ref):
    ref = validate_ref(ref)
    value = journal.storage.read(ref["path"])
    if value is None:
        raise DependencyInputError("CUTOFF_RETAINED_REF_MISSING")
    if value.byte_sha256 != ref["sha256"]:
        raise DependencyInputError("CUTOFF_RETAINED_REF_SHA_MISMATCH")
    return value.data


def _select_creation_time(physical):
    """One bounded alignment and one cutoff sample, never a retry loop."""
    boundary = physical.replace(microsecond=0)
    if physical.microsecond:
        boundary += timedelta(seconds=1)
        time.sleep((boundary - physical).total_seconds())
    sampled = _clock()
    stamp = _stamp(sampled)
    if sampled < physical or utc_stamp(stamp) < physical:
        raise ContractError("CUTOFF_CLOCK_REGRESSED_OR_UNALIGNED")
    return stamp


def _rebuild(*, sources, book, fields, cutoff, created_at, current_macro):
    registered = sources.registered is not None
    if (book.plan_version == 2) != registered:
        raise ContractError("CUTOFF_REGISTERED_PLAN_VERSION_MISMATCH")
    extra = {}
    if registered:
        proof = sources.registered
        declaration = proof["declaration"]
        book.read(
            {
                "path": str(
                    book.root.relative_to(book.workspace)
                    / "_record_store/daily_close_transactions"
                    / book.plan["transaction_id"]
                    / "decision-source-pointer.v1.json"
                ),
                "sha256": book.source_pointer_sha,
            }
        )
        if (
            book.plan["registered_event_declaration_ref"] != proof["declaration_ref"]
            or book.source_pointer_sha != declaration["baseline_store_pointer_ref"]["sha256"]
        ):
            raise ContractError("CUTOFF_REGISTERED_PLAN_BINDING_INVALID")
        extra = {
            "registered_event_declaration_ref": proof["declaration_ref"],
            "registered_writer_pointer_ref": declaration["writer_store_pointer_ref"],
            "registered_source_state": "REGISTERED_INTRADAY",
            "registered_store_plan_ref": None,
        }
    projected = sources.project(cutoff, current_macro=current_macro)
    corporate = derive_corporate_inputs(sources=sources, cutoff=cutoff)
    state = build_portfolio_state(as_of=cutoff, created_at=created_at, **fields)
    if utc_stamp(created_at) < utc_stamp(book.plan["transaction_planned_at"]):
        raise ContractError("CUTOFF_PORTFOLIO_PRECEDES_PLAN")
    state_bytes = canonical_json_bytes(state)
    state_ref = _ref(book.state_path, state_bytes)
    _, payload_bytes, payload_ref = sources.payload(cutoff)
    times = [*projected["source_times"], *corporate["source_times"]]
    for role, subject, ref, stamp, semantics in (
        (
            "STORE_PLAN",
            book.plan["transaction_id"],
            book.plan_ref,
            book.plan["transaction_planned_at"],
            "LOCAL_CLOSURE",
        ),
        (
            "PORTFOLIO_SOURCE_SEAL",
            fields["source_record_id"],
            fields["catalog_ref"],
            fields["source_sealed_at"],
            "LOCAL_CLOSURE",
        ),
        (
            "PORTFOLIO_POINTER_PUBLICATION",
            fields["source_record_id"],
            fields["frozen_pointer_ref"],
            fields["pointer_published_at"],
            "LOCAL_PUBLICATION",
        ),
    ):
        times.append(
            {
                "role": role,
                "subject_id": subject,
                "source_ref": ref,
                "original_time": stamp,
                "time_semantics": semantics,
            }
        )
    refs = ordered_source_refs(
        [
            sources.reference,
            *[ref for ref, _ in sources.files.observed.values()],
            *[ref for ref, _ in book.files.observed.values()],
        ]
    )
    sources.files.recheck()
    book.files.recheck()
    objects = [
        (state_ref, state_bytes),
        (payload_ref, payload_bytes),
        (corporate["corporate_context_ref"], corporate["context_bytes"]),
    ]
    if corporate["corporate_event_list_ref"] is not None:
        objects.append((corporate["corporate_event_list_ref"], corporate["event_list_bytes"]))
    body = {
        "schema_version": SCHEMA_V2 if registered else SCHEMA,
        **extra,
        "trade_date": sources.journal.trade_date,
        "request_ref": sources.document["request_ref"],
        "maintenance_handoff_ref": sources.document["maintenance_handoff_ref"],
        "core_handoff_ref": sources.document["core_handoff_ref"],
        "source_bundle_ref": sources.reference,
        "native_request_ref": payload_ref,
        "store_plan_ref": book.plan_ref,
        "portfolio_state_ref": state_ref,
        "timing_policy_ref": sources.document["timing_policy_ref"],
        "mode": sources.recipe["research_timing"]["mode"],
        "acquisition_deadline": sources.recipe["research_timing"]["acquisition_deadline"],
        "as_of": cutoff,
        "portfolio_created_at": created_at,
        "source_refs": refs,
        "source_times": ordered_source_times(times, registered=registered),
        "projection_sha256s": projected["projection_sha256s"],
        "corporate_event_list_ref": corporate["corporate_event_list_ref"],
        "corporate_context_ref": corporate["corporate_context_ref"],
        "portfolio_timing_status": state["payload"]["timing_status"],
        "prospective": False,
        "authority": dict(FALSE_AUTHORITY),
    }
    return body, objects, state


def _complete(*, journal, sources, book, receipt, cutoff_ref, objects, state, repair):
    version = 3 if receipt["schema_version"] == SCHEMA_V2 else 2
    wrapper = {
        "schema_version": WRAPPER_SCHEMA_V3 if version == 3 else WRAPPER_SCHEMA,
        "trade_date": journal.trade_date,
        "native_request_ref": receipt["native_request_ref"],
        "cutoff_ref": cutoff_ref,
        "source_completion_policy": POLICY,
    }
    raw = canonical_json_bytes(wrapper)
    if (
        journal.storage.read(str(sources.execution / f"inputs/research.v{5 - version}.json"))
        is not None
    ):
        raise ContractError("CUTOFF_WRAPPER_VERSION_CONFLICT")
    wrapper_ref = _ref(sources.execution / f"inputs/research.v{version}.json", raw)
    objects = [*objects, (wrapper_ref, raw)]
    missing = []
    # Discover every conflict before repairing any missing committed object.
    for ref, expected in objects:
        if _ref(ref["path"], expected) != ref:
            raise DependencyInputError("CUTOFF_COMMITTED_BYTES_SHA_MISMATCH")
        stored = journal.storage.read(ref["path"])
        if stored is None:
            missing.append((ref, expected))
        elif stored.data != expected or stored.byte_sha256 != ref["sha256"]:
            raise DependencyInputError("CUTOFF_COMMITTED_OBJECT_CONFLICT")
    if missing and not repair:
        raise DependencyInputError("CUTOFF_COMMITTED_OBJECT_MISSING")
    if repair:
        journal._require_lock()
        for ref, expected in missing:
            journal.storage.write(ref["path"], expected)
    for ref, expected in objects:
        if _read(journal, ref) != expected:
            raise ContractError("CUTOFF_COMMITTED_OBJECT_CHANGED")
    replay = PortfolioSource(
        workspace=sources.workspace,
        trade_date=journal.trade_date,
        as_of=receipt["as_of"],
        store_plan_ref=book.plan_ref,
    )
    if replay.replay(receipt["portfolio_state_ref"]) != state:
        raise ContractError("CUTOFF_PORTFOLIO_REPLAY_MISMATCH")
    sources.files.recheck()
    book.files.recheck()
    if parse_canonical_json_bytes(_read(journal, cutoff_ref)) != receipt:
        raise ContractError("CUTOFF_COMMITMENT_CHANGED")
    return {
        "cutoff_ref": cutoff_ref,
        "receipt": receipt,
        "research_request_ref": wrapper_ref,
        "native_request_ref": receipt["native_request_ref"],
        "portfolio_state_ref": receipt["portfolio_state_ref"],
        "portfolio_state": state,
        "store_plan_ref": book.plan_ref,
        "native_plan": book.plan,
        "corporate_action_context_ref": receipt["corporate_context_ref"],
        "corporate_action_template_ref": sources.document["corporate_action_template_ref"],
        "source_bundle": sources.document,
        "source_handoff": sources.handoff,
        "repaired": bool(missing),
    }


def read_cutoff_inputs(*, journal, cutoff_ref, repair=False):
    if type(repair) is not bool:
        raise ContractError("CUTOFF_REPAIR_MODE_INVALID")
    if repair:
        journal._require_lock()
    receipt = validate_cutoff_contract(parse_research_json(_read(journal, cutoff_ref)))
    if utc_stamp(receipt["sealed_at"]) > _clock():
        raise ContractError("CUTOFF_FUTURE_COMMITMENT")
    document = parse_research_json(_read(journal, receipt["source_bundle_ref"]))
    sources = SourceBundle(journal=journal, document=document)
    selected_path, stored = _selected_cutoff(journal, sources.execution, sources.version)
    if (
        cutoff_ref["path"] != selected_path
        or sources.reference != receipt["source_bundle_ref"]
        or stored is None
        or stored.byte_sha256 != cutoff_ref["sha256"]
        or (receipt["schema_version"] == SCHEMA_V2) != (sources.version == 2)
    ):
        raise DependencyInputError("CUTOFF_COMMITMENT_PATH_INVALID")
    sources.read(sources.reference)
    book = NativePortfolioSource(
        workspace=sources.workspace,
        trade_date=journal.trade_date,
        store_plan_ref=receipt["store_plan_ref"],
    )
    if book.plan_version == 1:
        original = {
            "path": str(PurePosixPath(book.plan_ref["path"]).with_name("source-pointer.v1.json")),
            "sha256": book.source_pointer_sha,
        }
        declared = [ref for ref in receipt["source_refs"] if ref["path"] == original["path"]]
        if declared:
            if declared != [original]:
                raise ContractError("CUTOFF_ORIGINAL_PORTFOLIO_REF_MISMATCH")
            # Adoption captured this exact native copy. Replay only the source
            # explicitly recorded then; do not add a new ref to older cutoffs.
            book.read(original)
    pointer = {"path": book.pointer_path, "sha256": book.source_pointer_sha}
    fields = book.source_fields(pointer)
    body, objects, state = _rebuild(
        sources=sources,
        book=book,
        fields=fields,
        cutoff=receipt["as_of"],
        created_at=receipt["portfolio_created_at"],
        current_macro=False,
    )
    expected = {
        **body,
        "physical_reads_completed_at": receipt["physical_reads_completed_at"],
        "sealed_at": receipt["sealed_at"],
    }
    if expected != receipt:
        raise ContractError("CUTOFF_NATIVE_COMMITMENT_REPLAY_MISMATCH")
    return _complete(
        journal=journal,
        sources=sources,
        book=book,
        receipt=receipt,
        cutoff_ref=cutoff_ref,
        objects=objects,
        state=state,
        repair=repair,
    )


def prepare_cutoff_inputs(*, journal, recovered, auxiliary, prepared):
    journal._require_lock()
    execution = journal.root / "executions" / recovered["handoff"]["request_ref"]["sha256"]
    version = cutoff_version(recovered["recipe"])
    path, prior = _selected_cutoff(journal, execution, version)
    if prior is not None:
        return read_cutoff_inputs(
            journal=journal, cutoff_ref={"path": path, "sha256": prior.byte_sha256}, repair=True
        )
    recipe = recovered["recipe"]
    assert_fresh_acquisition_open(recipe)
    bundle_path = str(execution / f"inputs/acquired-sources.v{version}.json")
    retained_bundle = journal.storage.read(bundle_path)
    bundle = (
        parse_canonical_json_bytes(retained_bundle.data)
        if retained_bundle is not None
        else build_source_bundle(journal=journal, recovered=recovered, auxiliary=auxiliary)
    )
    sources = SourceBundle(journal=journal, document=bundle)
    if prepared is None:
        raise ContractError("CUTOFF_NATIVE_STORE_PLAN_REQUIRED")
    book = NativePortfolioSource(
        workspace=sources.workspace,
        trade_date=journal.trade_date,
        store_plan_ref=prepared["store_plan_ref"],
    )
    if journal.storage.read(book.state_path) is not None:
        raise ContractError("CUTOFF_STATE_WITHOUT_COMMITMENT")
    fields = retain_portfolio_source(
        journal=journal,
        source=book,
        retained_pointer_ref=prepared.get("retained_source_pointer_ref"),
    )
    current = recipe["research_timing"]["mode"] == CURRENT
    preview_cutoff = (
        recipe["research_timing"]["acquisition_deadline"]
        if current
        else recipe["research_sources"]["as_of"]
    )
    sources.project(preview_cutoff, current_macro=current)
    derive_corporate_inputs(sources=sources, cutoff=preview_cutoff)
    sources.files.recheck()
    book.files.recheck()
    journal.storage.write(sources.reference["path"], canonical_json_bytes(bundle))
    sources.read(sources.reference)
    physical = _clock()
    created = _select_creation_time(physical)
    cutoff = created if current else recipe["research_sources"]["as_of"]
    if current and utc_stamp(cutoff).astimezone(SHANGHAI).strftime("%Y%m%d") != journal.trade_date:
        raise DependencyInputError("CUTOFF_CLOCK_TARGET_DAY_CHANGED")
    if utc_stamp(cutoff) > utc_stamp(created):
        raise ContractError("CUTOFF_HISTORICAL_TIME_IN_FUTURE")
    body, objects, state = _rebuild(
        sources=sources,
        book=book,
        fields=fields,
        cutoff=cutoff,
        created_at=created,
        current_macro=current,
    )
    sources.files.recheck()
    book.files.recheck()
    seal_observed = _clock()
    if seal_observed < utc_stamp(created) or (
        current and seal_observed > utc_stamp(recipe["research_timing"]["acquisition_deadline"])
    ):
        raise ContractError("CUTOFF_SEAL_CLOCK_OR_DEADLINE_INVALID")
    receipt = {
        **body,
        "physical_reads_completed_at": physical.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "sealed_at": _stamp(seal_observed),
    }
    validate_cutoff_contract(receipt)
    raw = canonical_json_bytes(receipt)
    cutoff_ref = _ref(path, raw)
    journal.storage.write(path, raw)
    return _complete(
        journal=journal,
        sources=sources,
        book=book,
        receipt=receipt,
        cutoff_ref=cutoff_ref,
        objects=objects,
        state=state,
        repair=True,
    )
