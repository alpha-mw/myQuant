"""Pure in-memory historical replay for frozen I5 receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ._contracts import artifact, artifact_ref, closed_artifact, fail, same, when
from .committee import (
    validate_advisory_rank,
    validate_capability_isolation,
    validate_committee_request,
    validate_committee_response,
    validate_decision_evidence_projection,
    validate_private_capability,
)
from .models import REPLAY_RECEIPT_VERSION
from .public_search import (
    validate_declassified_public_packet,
    validate_public_search_capability,
    validate_search_run,
)

_REPLAY_FIELDS: Final = {
    "packet_ref",
    "public_capability_ref",
    "search_run_ref",
    "projection_ref",
    "private_capability_ref",
    "committee_request_ref",
    "committee_response_ref",
    "advisory_rank_ref",
    "knowledge_cutoff",
    "decision_issued_at",
    "target_trade_execution_boundary",
    "mode",
    "external_call_counts",
    "status",
}


def build_historical_replay_receipt(
    *,
    packet: Mapping[str, Any],
    public_capability: Mapping[str, Any],
    search_run: Mapping[str, Any],
    round_bundles: Sequence[Mapping[str, Any]],
    fact_bundles: Sequence[Mapping[str, Any]],
    projection: Mapping[str, Any],
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
    private_capability: Mapping[str, Any],
    committee_request: Mapping[str, Any],
    committee_response: Mapping[str, Any],
    advisory_rank: Mapping[str, Any],
    target_trade_execution_boundary: str,
    replayed_at: str,
) -> dict[str, Any]:
    public = validate_public_search_capability(public_capability)
    packet_doc = validate_declassified_public_packet(
        packet,
        declassification_evidence_ref=public["control_evidence_ref"],
    )
    search = validate_search_run(
        search_run,
        packet=packet_doc,
        capability=public,
        round_bundles=round_bundles,
    )
    projected = validate_decision_evidence_projection(
        projection,
        packet=packet_doc,
        public_capability=public,
        search_run=search,
        round_bundles=round_bundles,
        fact_bundles=fact_bundles,
        decision_receipt=decision_receipt,
        decision_validation_closure=decision_validation_closure,
    )
    private = validate_private_capability(private_capability)
    validate_capability_isolation(public_capability=public, private_capability=private)
    request = validate_committee_request(
        committee_request, capability=private, projection=projected
    )
    response = validate_committee_response(
        committee_response,
        capability=private,
        request=request,
        projection=projected,
    )
    advisory = validate_advisory_rank(
        advisory_rank,
        projection=projected,
        response=response,
        response_validation_closure={"capability": private, "request": request},
        fact_bundles=fact_bundles,
    )
    boundary = when(
        target_trade_execution_boundary,
        label="target_trade_execution_boundary",
    )
    replayed = when(replayed_at, label="replayed_at")
    if not (
        packet_doc["market_data_cutoff"]
        <= search["evidence_collection_started_at"]
        <= search["search_completed_at"]
        <= projected["knowledge_cutoff"]
        <= request["timestamp"]
        <= response["timestamp"]
        <= advisory["timestamp"]
        < boundary
    ):
        fail("I5 end-to-end timeline is invalid")
    if replayed < advisory["timestamp"]:
        fail("historical replay timestamp predates frozen decision")
    return artifact(
        version=REPLAY_RECEIPT_VERSION,
        identity_field="replay_receipt_id",
        timestamp_value=replayed,
        payload={
            "packet_ref": artifact_ref(packet_doc, identity_field="packet_id"),
            "public_capability_ref": artifact_ref(public, identity_field="capability_id"),
            "search_run_ref": artifact_ref(search, identity_field="search_run_id"),
            "projection_ref": artifact_ref(projected, identity_field="projection_id"),
            "private_capability_ref": artifact_ref(private, identity_field="private_capability_id"),
            "committee_request_ref": artifact_ref(request, identity_field="committee_request_id"),
            "committee_response_ref": artifact_ref(
                response, identity_field="committee_response_id"
            ),
            "advisory_rank_ref": artifact_ref(advisory, identity_field="advisory_rank_id"),
            "knowledge_cutoff": projected["knowledge_cutoff"],
            "decision_issued_at": advisory["timestamp"],
            "target_trade_execution_boundary": boundary,
            "mode": "FROZEN_OFFLINE_REPLAY",
            "external_call_counts": {
                "credential_reads": 0,
                "filesystem_discovery": 0,
                "model": 0,
                "network": 0,
            },
            "status": "COMPLETE",
        },
    )


def validate_historical_replay_receipt(
    document: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    public_capability: Mapping[str, Any],
    search_run: Mapping[str, Any],
    round_bundles: Sequence[Mapping[str, Any]],
    fact_bundles: Sequence[Mapping[str, Any]],
    projection: Mapping[str, Any],
    decision_receipt: Mapping[str, Any],
    decision_validation_closure: Mapping[str, Any],
    private_capability: Mapping[str, Any],
    committee_request: Mapping[str, Any],
    committee_response: Mapping[str, Any],
    advisory_rank: Mapping[str, Any],
) -> dict[str, Any]:
    row = closed_artifact(
        document,
        version=REPLAY_RECEIPT_VERSION,
        identity_field="replay_receipt_id",
        payload_fields=_REPLAY_FIELDS,
    )
    expected = build_historical_replay_receipt(
        packet=packet,
        public_capability=public_capability,
        search_run=search_run,
        round_bundles=round_bundles,
        fact_bundles=fact_bundles,
        projection=projection,
        decision_receipt=decision_receipt,
        decision_validation_closure=decision_validation_closure,
        private_capability=private_capability,
        committee_request=committee_request,
        committee_response=committee_response,
        advisory_rank=advisory_rank,
        target_trade_execution_boundary=row["target_trade_execution_boundary"],
        replayed_at=row["timestamp"],
    )
    same(row, expected, label="historical replay")
    return expected


__all__ = [
    "build_historical_replay_receipt",
    "validate_historical_replay_receipt",
]
