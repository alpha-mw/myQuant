"""Derive native corporate wrappers from exact declared sources, without financial writes."""

import hashlib

from quant_investor.contracts import canonical_json_bytes
from quant_investor.strategy_records import corporate_contracts as contracts
from quant_investor.strategy_records.risk_policy_contract import validate_trailing_policy
from quant_investor.strategy_records.event_store import load_frozen_generation
from quant_investor.strategy_records.event_contracts import SYMBOLIC_RECEIPT
from quant_investor.strategy_records.event_receipts import resolve_catalog_event_receipt
from .corporate_actions import EVENT_ROOT
from .daily_contract import ContractError, validate_ref

TEMPLATE_SCHEMA = "cn-corporate-action-template.v1"


def validate_corporate_template(value):
    contracts.shape(
        value,
        TEMPLATE_SCHEMA,
        {
            "strategy_id",
            "tracking_policy_ref",
            "named_event_refs",
            "anchor_reviews_ref",
        },
    )
    if value["strategy_id"] != contracts.STRATEGY:
        raise ContractError("CUTOFF_CORPORATE_STRATEGY_INVALID")
    validate_ref(value["tracking_policy_ref"])
    if value["anchor_reviews_ref"] is not None:
        validate_ref(value["anchor_reviews_ref"])
    if value["named_event_refs"] is not None:
        contracts.ordered_refs(value["named_event_refs"])
    return value


def derive_corporate_inputs(*, sources, cutoff):
    """Caller retains every source read; null declarations remain null in native context."""
    template_ref = sources.document["corporate_action_template_ref"]
    template = validate_corporate_template(sources.read(template_ref))
    policy_ref = template["tracking_policy_ref"]
    policy = sources.read(policy_ref)
    validate_trailing_policy(policy, as_of=cutoff)
    times = []

    def add(role, subject, ref, stamp, semantics):
        contracts.instant(stamp)
        times.append(
            {
                "role": role,
                "subject_id": subject,
                "source_ref": validate_ref(ref),
                "original_time": stamp,
                "time_semantics": semantics,
            }
        )

    add("CORPORATE_POLICY", "ALL", policy_ref, policy["effective_from"], "OWNER_EFFECTIVE")
    event_ref = validate_ref(sources.document["event_pointer_ref"])
    expected_sha = sources.recipe["store_preimages"]["event_pointer_ref"]["sha256"]
    if event_ref != {
        "path": str(sources.journal.root / "inputs" / f"event-pointer-{expected_sha}.json"),
        "sha256": expected_sha,
    }:
        raise ContractError("CUTOFF_EVENT_POINTER_BINDING_INVALID")
    _, raw, _ = sources.files.source_file(event_ref, code="CUTOFF_EVENT_POINTER_INVALID")
    event_store = load_frozen_generation(
        sources.workspace / EVENT_ROOT, pointer_bytes=raw, expected_pointer_sha256=expected_sha
    )
    generation = event_store["pointer"]["generation"]
    generation_ref = {"path": EVENT_ROOT + "/" + generation["path"], "sha256": generation["sha256"]}
    document = sources.read(generation_ref)
    if document != event_store["generation"]:
        raise ContractError("CUTOFF_EVENT_GENERATION_CHANGED")
    add(
        "EVENT_GENERATION",
        document["generation_id"],
        generation_ref,
        document["generated_at"],
        "LOCAL_CLOSURE",
    )
    closures = [
        row
        for row in event_store["closures"]
        if row["trade_date"].replace("-", "") == sources.journal.trade_date
    ]
    registered = getattr(sources, "registered", None)
    if registered is not None:
        if closures:
            raise ContractError("CUTOFF_REGISTERED_EMPTY_EVENT_CONFLICT")
        declaration = registered["declaration"]
        owner = sources.read(declaration["owner_fact_ref"])
        for role, subject, reference, stamp, semantics in (
            (
                "REGISTERED_OWNER_FACT",
                owner["owner_fact_id"],
                declaration["owner_fact_ref"],
                declaration["owner_declared_at"],
                "SOURCE_DECLARED",
            ),
            (
                "REGISTERED_DECLARATION",
                declaration["declaration_id"],
                registered["declaration_ref"],
                declaration["registered_at"],
                "LOCAL_REGISTRATION",
            ),
            (
                "REGISTERED_STORE_PUBLICATION",
                declaration["writer_record_id"],
                declaration["writer_store_pointer_ref"],
                registered["writer"]["pointer"]["published_at"],
                "LOCAL_PUBLICATION",
            ),
        ):
            add(role, subject, reference, stamp, semantics)
    elif len(closures) != 1:
        raise ContractError("CUTOFF_EVENT_CLOSURE_MISSING")
    else:
        _retain_empty_closure(sources, closures[0])
    events = {}
    for ref in template["named_event_refs"] or []:
        event = contracts.event(sources.read(ref), as_of=cutoff)
        if registered is not None and event["effective_trade_date"] == sources.journal.trade_date:
            raise ContractError("CUTOFF_REGISTERED_CORPORATE_FACT_CONFLICT")
        if any(prior["event_id"] == event["event_id"] for prior in events.values()):
            raise ContractError("CUTOFF_CORPORATE_EVENT_DUPLICATE")
        events[(ref["path"], ref["sha256"])] = event
        sources.files.source_file(event["announcement_ref"], code="CUTOFF_ANNOUNCEMENT_REF_INVALID")
        add("CORPORATE_EVENT", event["event_id"], ref, event["announced_at"], "SOURCE_DECLARED")
    review_ref = template["anchor_reviews_ref"]
    if review_ref is not None:
        review = sources.read(review_ref)
        contracts.owner_reviews(
            review, as_of=cutoff, owner=policy["owner"], policy_ref=policy_ref, events=events
        )
        add(
            "CORPORATE_REVIEW_DECLARATION",
            review["declaration_id"],
            review_ref,
            review["declared_at"],
            "SOURCE_DECLARED",
        )

    def expected_object(label, document):
        raw = canonical_json_bytes(document)
        sha = hashlib.sha256(raw).hexdigest()
        return {
            "path": str(sources.execution / "inputs" / f"corporate-{label}-{sha}.json"),
            "sha256": sha,
        }, raw

    event_list_ref, event_list_bytes = None, None
    if template["named_event_refs"] is not None:
        event_list = {
            "schema_version": "cn-corporate-action-events.v1",
            "strategy_id": contracts.STRATEGY,
            "as_of": cutoff,
            "event_refs": template["named_event_refs"],
        }
        contracts.event_list(event_list, as_of=cutoff)
        event_list_ref, event_list_bytes = expected_object("events", event_list)
    context = {
        "schema_version": "cn-corporate-action-context.v1",
        "strategy_id": contracts.STRATEGY,
        "as_of": cutoff,
        "tracking_policy_ref": policy_ref,
        "named_events_ref": event_list_ref,
        "anchor_reviews_ref": review_ref,
    }
    contracts.context(context, as_of=cutoff)
    context_ref, context_bytes = expected_object("context", context)
    sources.files.recheck()
    return {
        "corporate_event_list_ref": event_list_ref,
        "event_list_bytes": event_list_bytes,
        "corporate_context_ref": context_ref,
        "context_bytes": context_bytes,
        "source_times": times,
    }


def _retain_empty_closure(sources, closure):
    from quant_investor.strategy_records.daily_event_source import validate_daily_closure_source

    daily_source = validate_daily_closure_source(workspace=sources.workspace, closure=closure)
    if daily_source is not None:
        for reference in daily_source["source_refs"]:
            sources.files.source_file(reference, code="CUTOFF_EVENT_SOURCE_INVALID")
    for name in ("policy_ref", "owner_declaration_ref", "source_receipt_ref"):
        ref = closure[name]
        if ref is None:
            continue
        if SYMBOLIC_RECEIPT.fullmatch(ref["path"]):
            ref = resolve_catalog_event_receipt(workspace=sources.workspace, closure=closure)[
                "catalog_ref"
            ]
        sources.files.source_file(ref, code="CUTOFF_EVENT_SOURCE_INVALID")
