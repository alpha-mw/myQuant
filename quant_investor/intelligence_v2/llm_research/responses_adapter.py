"""Explicitly injected OpenAI Responses seams; no credential or client discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._core import canonical_bytes, require_exact_keys
from ._contracts import fail
from .committee import SYSTEM_PROMPT, validate_committee_request, validate_private_capability
from .public_search import (
    PUBLIC_SEARCH_PROMPT,
    validate_declassified_public_packet,
    validate_public_search_capability,
    validate_search_request,
)

ResponsesCreate = Callable[..., Mapping[str, Any]]
_PUBLIC_RESULT_FIELDS = {
    "provider_response_id",
    "response_text",
    "sources",
    "citation_leads",
}
_PRIVATE_RESULT_FIELDS = {
    "provider_response_id",
    "relative_score",
    "findings",
    "thesis_gaps",
    "source_conflicts",
    "time_conflicts",
    "unverifiable_claims",
    "tool_calls",
}


def _identity(capability: Mapping[str, Any], adapter: Any) -> None:
    for field in ("project", "credential_ref", "client_namespace"):
        if capability[field] != getattr(adapter, field):
            fail(f"Responses adapter {field} differs from capability")


@dataclass(frozen=True)
class OpenAIResponsesPublicAdapter:
    responses_create: ResponsesCreate
    project: str
    credential_ref: str
    client_namespace: str

    def execute_three_rounds(
        self,
        *,
        requests: Sequence[Mapping[str, Any]],
        packet: Mapping[str, Any],
        capability: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        capability_doc = validate_public_search_capability(capability)
        packet_doc = validate_declassified_public_packet(
            packet,
            declassification_evidence_ref=capability_doc["control_evidence_ref"],
        )
        _identity(capability_doc, self)
        if len(requests) != 3:
            fail("public Responses adapter requires exactly three requests")
        results = []
        for index, request in enumerate(requests):
            request_doc = validate_search_request(
                request,
                packet=packet_doc,
                capability=capability_doc,
            )
            if request_doc["round_index"] != index + 1:
                fail("public Responses request order is invalid")
            result = self.responses_create(
                model=capability_doc["exact_model"],
                input=[
                    {"content": PUBLIC_SEARCH_PROMPT, "role": "system"},
                    {"content": request_doc["query"], "role": "user"},
                ],
                tools=[{"type": "web_search"}],
                tool_choice="required",
                store=False,
                background=False,
                previous_response_id=None,
            )
            row = require_exact_keys(
                result,
                _PUBLIC_RESULT_FIELDS,
                label=f"public Responses result[{index}]",
            )
            results.append(dict(row))
        return tuple(results)


@dataclass(frozen=True)
class OpenAIResponsesPrivateAdapter:
    responses_create: ResponsesCreate
    project: str
    credential_ref: str
    client_namespace: str

    def execute_once(
        self,
        *,
        request: Mapping[str, Any],
        capability: Mapping[str, Any],
        projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        capability_doc = validate_private_capability(capability)
        _identity(capability_doc, self)
        request_doc = validate_committee_request(
            request,
            capability=capability_doc,
            projection=projection,
        )
        result = self.responses_create(
            model=capability_doc["exact_model"],
            input=[
                {"content": SYSTEM_PROMPT, "role": "system"},
                {
                    "content": canonical_bytes(projection).decode("utf-8"),
                    "role": "user",
                },
            ],
            tools=[],
            store=False,
            background=False,
            previous_response_id=None,
        )
        row = require_exact_keys(
            result,
            _PRIVATE_RESULT_FIELDS,
            label="private Responses result",
        )
        if row["tool_calls"] != []:
            fail("private Responses result contains a tool call")
        if request_doc["request_configuration"]["tools"] != []:
            fail("private Responses request tools are not empty")
        return dict(row)


__all__ = [
    "OpenAIResponsesPrivateAdapter",
    "OpenAIResponsesPublicAdapter",
    "ResponsesCreate",
]
