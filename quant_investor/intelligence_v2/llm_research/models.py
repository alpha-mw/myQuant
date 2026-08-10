"""Closed constants and injectable boundaries for I5 LLM research."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Protocol

from .._core import IntelligenceV2ContractError

ROLE: Final = "怀疑型 AI 投委会"
ROUND_ORDER: Final = ("DISCOVERY", "CONTRARY_GAPS", "VERIFICATION_CLOSURE")
FACT_CLASSES: Final = frozenset({"FACT", "INFERENCE", "OPINION"})
SOURCE_CLASSES: Final = frozenset({"FIRST_PARTY", "ORIGINAL_SOURCE", "SYNDICATION", "MIRROR"})
CAPTURE_STATES: Final = frozenset({"DISCOVERED", "CAPTURED", "VALIDATED", "MISMATCH", "BLOCKED"})

PUBLIC_PACKET_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-public-packet.v1"
PUBLIC_CAPABILITY_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.i5-public-search-capability.v1"
)
SEARCH_REQUEST_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-search-request.v1"
SEARCH_RESPONSE_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-search-response.v1"
SEARCH_SOURCE_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-search-source.v1"
CITATION_LEAD_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-citation-lead.v1"
SEARCH_RUN_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-search-run.v1"
SEARCH_RUN_STATUS_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-search-run-status.v1"
CAPTURE_POLICY_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-capture-policy.v1"
CAPTURE_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-capture-receipt.v1"
VALIDATED_FACT_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-validated-fact.v1"
PRIVATE_CAPABILITY_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-private-capability.v1"
PROJECTION_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.i5-decision-evidence-projection.v1"
)
COMMITTEE_REQUEST_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-committee-request.v1"
COMMITTEE_RESPONSE_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-committee-response.v1"
ADVISORY_RANK_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-advisory-rank.v1"
ADVISORY_UNAVAILABLE_VERSION: Final = (
    "myquant.v17.research-intelligence-v2.i5-advisory-unavailable.v1"
)
REPLAY_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence-v2.i5-historical-replay.v1"


class I5ContractError(IntelligenceV2ContractError):
    """Fail-closed I5 research contract error."""

    exit_code = 2


class PublicSearchClient(Protocol):
    """Injectable live boundary; I5 v1 provides no implementation or fallback."""

    def create_response(self, request: Mapping[str, Any], /) -> Mapping[str, Any]: ...


class CaptureTransport(Protocol):
    """Injectable SSRF-safe transport boundary; I5 v1 provides no implementation."""

    def capture(self, request: Mapping[str, Any], /) -> Mapping[str, Any]: ...


class PrivateCommitteeClient(Protocol):
    """Injectable private model boundary; I5 v1 provides no implementation."""

    def create_response(self, request: Mapping[str, Any], /) -> Mapping[str, Any]: ...


__all__ = [
    "ADVISORY_RANK_VERSION",
    "ADVISORY_UNAVAILABLE_VERSION",
    "CAPTURE_POLICY_VERSION",
    "CAPTURE_RECEIPT_VERSION",
    "CAPTURE_STATES",
    "CITATION_LEAD_VERSION",
    "COMMITTEE_REQUEST_VERSION",
    "COMMITTEE_RESPONSE_VERSION",
    "CaptureTransport",
    "FACT_CLASSES",
    "I5ContractError",
    "PRIVATE_CAPABILITY_VERSION",
    "PROJECTION_VERSION",
    "PUBLIC_CAPABILITY_VERSION",
    "PUBLIC_PACKET_VERSION",
    "PrivateCommitteeClient",
    "PublicSearchClient",
    "REPLAY_RECEIPT_VERSION",
    "ROLE",
    "ROUND_ORDER",
    "SEARCH_REQUEST_VERSION",
    "SEARCH_RESPONSE_VERSION",
    "SEARCH_RUN_VERSION",
    "SEARCH_RUN_STATUS_VERSION",
    "SEARCH_SOURCE_VERSION",
    "SOURCE_CLASSES",
    "VALIDATED_FACT_VERSION",
]
