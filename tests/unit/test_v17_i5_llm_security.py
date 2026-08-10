"""Security, privacy, and no-I/O tests for I5."""

from __future__ import annotations

import builtins
from copy import deepcopy
import io
import os
from pathlib import Path
import socket
import ssl
from typing import Any
import urllib.request

import pytest

from quant_investor.intelligence_v2._core import seal
from quant_investor.intelligence_v2.decision_v2 import engine as decision_engine
from quant_investor.intelligence_v2.llm_research import (
    DECODER_MANIFEST_SHA256,
    I5ContractError,
    PARSER_IDENTITY,
    PARSER_OPTIONS_SHA256,
    PARSER_VERSION,
    SafeLocalCollector,
    StdlibPinnedConnector,
    build_capture_policy,
    build_capture_receipt,
    build_declassified_public_packet,
    build_search_source,
    validate_historical_replay_receipt,
)
from tests.unit.test_v17_i5_open_web_committee import (
    SHA_B,
    SHA_C,
    T0,
    T1,
    T5,
    T8,
    _exact_ref,
    _packet,
    _stack,
)


@pytest.fixture(autouse=True)
def _isolate_decision_v2_fixture_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        decision_engine,
        "validate_evidence_graph_v2",
        lambda document, **_closure: document,
    )
    monkeypatch.setattr(
        decision_engine,
        "validate_fusion_projection_v2",
        lambda document, **_closure: document,
    )


def _capture_args(stack: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": stack["source"],
        "source_request": stack["request"],
        "provider_response_id": stack["provider_response_id"],
        "policy": stack["policy"],
        "transport_observation": deepcopy(stack["observation"]),
        "compressed_entity": stack["body"],
        "decoded_entity": stack["body"],
        "parser_input": stack["body"],
        "publication_evidence_ref": _exact_ref("publication", sha=SHA_B),
        "transport_attestation_ref": _exact_ref("transport-attestation"),
        "captured_at": T5,
    }


def test_declassified_packet_rejects_private_canary_and_extra_fields() -> None:
    with pytest.raises(I5ContractError, match="private canary"):
        build_declassified_public_packet(
            company_code="600000.SH",
            display_name="Public Company",
            public_industry_ids=[],
            public_theme_ids=[],
            thesis="Thesis contains TOP_SECRET_CANARY.",
            search_questions=["Public question"],
            market_data_cutoff=T0,
            target_knowledge_not_before=T1,
            target_knowledge_not_after=T8,
            created_at=T0,
            declassification_evidence_ref=_exact_ref("declassification-evidence"),
            privacy_canaries=["TOP_SECRET_CANARY"],
        )
    packet = _packet()
    packet["portfolio_weights"] = {"600000.SH": "1"}
    with pytest.raises(I5ContractError, match="shape is invalid"):
        from quant_investor.intelligence_v2.llm_research import (
            validate_declassified_public_packet,
        )

        validate_declassified_public_packet(
            packet,
            declassification_evidence_ref=_exact_ref("declassification-evidence"),
        )


def test_declassification_and_prompt_schema_bindings_are_external_and_tamper_evident() -> None:
    stack = _stack()
    from quant_investor.intelligence_v2.llm_research import (
        validate_committee_response,
        validate_declassified_public_packet,
        validate_search_request,
    )

    with pytest.raises(I5ContractError, match="differs from authorized closure"):
        validate_declassified_public_packet(
            stack["packet"],
            declassification_evidence_ref=_exact_ref("different-declassification"),
        )
    request = deepcopy(stack["rounds"][0]["request"])
    request["request_schema_sha256"] = SHA_B
    request.pop("search_request_id")
    request.pop("semantic_sha256")
    request = seal(request, identity_field="search_request_id")
    with pytest.raises(I5ContractError, match="deterministic replay"):
        validate_search_request(
            request,
            packet=stack["packet"],
            capability=stack["public_capability"],
        )
    response = deepcopy(stack["response_receipt"])
    response["output_schema_sha256"] = SHA_C
    response.pop("committee_response_id")
    response.pop("semantic_sha256")
    response = seal(response, identity_field="committee_response_id")
    with pytest.raises(I5ContractError, match="deterministic replay"):
        validate_committee_response(
            response,
            capability=stack["private_capability"],
            request=stack["request_receipt"],
            projection=stack["projection"],
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user@example.com/",
        "http://example.com:8080/",
        "http://EXAMPLE.com/",
        "http://example.com/path#fragment",
        "http://example.com\\evil/",
        "http://éxample.com/",
    ],
)
def test_malformed_or_authority_open_urls_fail_closed(url: str) -> None:
    stack = _stack()
    with pytest.raises(I5ContractError):
        build_search_source(
            request=stack["request"],
            provider_response_id=stack["provider_response_id"],
            url=url,
            title="Invalid",
            publisher=None,
            publication_hint=None,
            media_kind="HTML",
            discovered_at=T5,
        )


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "192.0.2.1",
        "::1",
        "fe80::1",
    ],
)
def test_dns_private_reserved_metadata_and_multicast_addresses_are_rejected(
    address: str,
) -> None:
    stack = _stack()
    args = _capture_args(stack)
    args["transport_observation"]["redirect_chain"][0]["resolved_addresses"] = [address]
    args["transport_observation"]["redirect_chain"][0]["peer_ip"] = address
    with pytest.raises((I5ContractError, ValueError), match="public"):
        build_capture_receipt(**args)


def test_peer_ip_redirect_and_transport_leak_checks_fail_closed() -> None:
    stack = _stack()
    args = _capture_args(stack)
    args["transport_observation"]["redirect_chain"][0]["peer_ip"] = "1.1.1.1"
    with pytest.raises(I5ContractError, match="peer IP"):
        build_capture_receipt(**args)

    for flag in (
        "proxy_environment_used",
        "cookies_sent",
        "authorization_sent",
        "library_reresolution",
    ):
        args = _capture_args(stack)
        args["transport_observation"][flag] = True
        with pytest.raises(I5ContractError, match="leaked authority|DNS"):
            build_capture_receipt(**args)

    args = _capture_args(stack)
    hop = deepcopy(args["transport_observation"]["redirect_chain"][0])
    args["transport_observation"]["redirect_chain"] = [hop] * 7
    with pytest.raises(I5ContractError, match="five redirects"):
        build_capture_receipt(**args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mime_type", "application/json", "MIME"),
        ("charset", "ISO-8859-1", "charset"),
        ("content_encoding", "compress", "encoding"),
        ("html_node_count", 100_001, "node"),
        ("html_max_depth", 129, "depth"),
    ],
)
def test_content_parser_and_decoder_limits_fail_closed(
    field: str, value: Any, message: str
) -> None:
    stack = _stack()
    args = _capture_args(stack)
    args["transport_observation"][field] = value
    with pytest.raises(I5ContractError, match=message):
        build_capture_receipt(**args)


def test_decompression_ratio_and_parser_input_substitution_are_rejected() -> None:
    stack = _stack()
    args = _capture_args(stack)
    args["compressed_entity"] = b"x"
    args["decoded_entity"] = b"x" * 21
    args["parser_input"] = args["decoded_entity"]
    with pytest.raises(I5ContractError, match="decompression ratio"):
        build_capture_receipt(**args)

    args = _capture_args(stack)
    args["parser_input"] = b"substituted"
    with pytest.raises(I5ContractError, match="exact strict UTF-8 projection"):
        build_capture_receipt(**args)


def test_pdf_remains_discovery_only_and_never_enters_capture() -> None:
    stack = _stack()
    pdf = build_search_source(
        request=stack["request"],
        provider_response_id=stack["provider_response_id"],
        url="https://source1.example/report.pdf",
        title="Discovered PDF",
        publisher="Publisher",
        publication_hint=None,
        media_kind="PDF",
        discovered_at=T5,
    )
    assert pdf["status"] == "DISCOVERED"
    args = _capture_args(stack)
    args["source"] = pdf
    with pytest.raises(I5ContractError, match="discovery-only"):
        build_capture_receipt(**args)


def test_private_request_and_response_reject_tool_or_action_forgery() -> None:
    stack = _stack()
    request = deepcopy(stack["request_receipt"])
    request["request_configuration"]["tools"] = [{"type": "web_search"}]
    request.pop("committee_request_id")
    request.pop("semantic_sha256")
    request = seal(request, identity_field="committee_request_id")
    from quant_investor.intelligence_v2.llm_research import validate_committee_request

    with pytest.raises(I5ContractError, match="deterministic replay"):
        validate_committee_request(
            request,
            capability=stack["private_capability"],
            projection=stack["projection"],
        )

    response = deepcopy(stack["response_receipt"])
    response["tool_calls"] = [{"type": "code_interpreter"}]
    response.pop("committee_response_id")
    response.pop("semantic_sha256")
    response = seal(response, identity_field="committee_response_id")
    from quant_investor.intelligence_v2.llm_research import validate_committee_response

    with pytest.raises(I5ContractError, match="deterministic replay"):
        validate_committee_response(
            response,
            capability=stack["private_capability"],
            request=stack["request_receipt"],
            projection=stack["projection"],
        )


def test_historical_replay_performs_zero_network_model_credential_or_filesystem_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("historical replay attempted forbidden I/O")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)

    assert (
        validate_historical_replay_receipt(
            stack["replay"],
            packet=stack["packet"],
            public_capability=stack["public_capability"],
            search_run=stack["search_run"],
            round_bundles=stack["rounds"],
            fact_bundles=[stack["fact_bundle"]],
            projection=stack["projection"],
            decision_receipt=stack["decision_receipt"],
            decision_validation_closure=stack["decision_validation_closure"],
            private_capability=stack["private_capability"],
            committee_request=stack["request_receipt"],
            committee_response=stack["response_receipt"],
            advisory_rank=stack["advisory"],
        )
        == stack["replay"]
    )


def test_i5_source_has_no_concrete_network_or_model_client() -> None:
    root = (
        Path(__file__).resolve().parents[2] / "quant_investor" / "intelligence_v2" / "llm_research"
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden_model_or_proxy_calls = (
        "OpenAI(",
        "requests.get(",
        "urlopen(",
        "subprocess.",
    )
    assert not any(token in source for token in forbidden_model_or_proxy_calls)
    assert StdlibPinnedConnector is not None
    assert "class StdlibPinnedConnector" in source
    assert "socket.getaddrinfo" in source
    assert "ssl.create_default_context" in source
    assert "allowed_domains" not in source


class _FakeClock:
    def __init__(self, *, monotonic_values: list[float] | None = None) -> None:
        self.values = iter(monotonic_values or [0.0] * 20)

    def utc_now(self) -> str:
        return T5

    def monotonic(self) -> float:
        return next(self.values)


class _FakeResolver:
    def __init__(self, values: dict[str, list[str]]) -> None:
        self.values = values
        self.calls: list[tuple[str, int]] = []

    def resolve(self, hostname: str, port: int, /) -> list[str]:
        self.calls.append((hostname, port))
        return self.values[hostname]


class _FakeExchange:
    def __init__(
        self,
        *,
        peer_ip: str,
        status_code: int,
        headers: dict[str, str],
        entity: bytes = b"",
        header_bytes_total: int = 256,
    ) -> None:
        self.peer_ip = peer_ip
        self.status_code = status_code
        self.selected_headers = headers
        self.header_bytes_total = header_bytes_total
        self.entity = entity
        self.reads = 0
        self.closed = 0

    def read_entity(self, max_bytes: int, /) -> bytes:
        self.reads += 1
        if len(self.entity) > max_bytes:
            raise I5ContractError("fixture exceeds entity limit")
        return self.entity

    def close(self) -> None:
        self.closed += 1


class _FakeConnector:
    def __init__(self, exchanges: list[_FakeExchange]) -> None:
        self.exchanges = iter(exchanges)
        self.calls: list[dict[str, Any]] = []

    def open(self, **kwargs: Any) -> _FakeExchange:
        self.calls.append(kwargs)
        return next(self.exchanges)


def _safe_policy() -> dict[str, Any]:
    return build_capture_policy(
        parser_identity=PARSER_IDENTITY,
        parser_version=PARSER_VERSION,
        parser_options_sha256=PARSER_OPTIONS_SHA256,
        decoder_manifest_sha256=DECODER_MANIFEST_SHA256,
        transport_policy_ref=_exact_ref("safe-transport-policy"),
        created_at=T0,
    )


def _collector_args(stack: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": stack["source"],
        "source_request": stack["request"],
        "provider_response_id": stack["provider_response_id"],
        "policy": _safe_policy(),
        "publication_evidence_ref": _exact_ref("publication", sha=SHA_B),
        "transport_attestation_ref": _exact_ref("safe-transport-attestation"),
    }


def test_concrete_safe_collector_pins_dns_peer_revalidates_redirect_and_parses_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack()
    body = b"<html><body><p>Revenue declined.</p></body></html>"
    first = _FakeExchange(
        peer_ip="93.184.216.34",
        status_code=302,
        headers={"location": "https://source2.example/final"},
    )
    second = _FakeExchange(
        peer_ip="93.184.216.35",
        status_code=200,
        headers={
            "content-type": "text/html; charset=UTF-8",
            "content-encoding": "identity",
        },
        entity=body,
    )
    resolver = _FakeResolver(
        {
            "source1.example": ["93.184.216.34"],
            "source2.example": ["93.184.216.35"],
        }
    )
    connector = _FakeConnector([first, second])
    collector = SafeLocalCollector(resolver, connector, _FakeClock())

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("collector bypassed injected network boundary")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    result = collector.capture(**_collector_args(stack))
    assert result["receipt"]["status"] == "VALIDATED"
    assert result["receipt"]["final_url"] == "https://source2.example/final"
    assert len(result["receipt"]["redirect_chain"]) == 2
    assert resolver.calls == [("source1.example", 443), ("source2.example", 443)]
    assert [row["address"] for row in connector.calls] == [
        "93.184.216.34",
        "93.184.216.35",
    ]
    assert first.reads == 0
    assert second.reads == 1
    assert result["parser_input"] == body


def test_stdlib_connector_uses_explicit_ip_sni_and_authority_free_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SocketStub:
        def __init__(self) -> None:
            self.destination: Any = None
            self.request = b""
            self.timeouts: list[float] = []
            self.closed = False
            self.response = io.BytesIO(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=UTF-8\r\n"
                b"Content-Length: 13\r\n\r\n<html></html>"
            )

        def settimeout(self, value: float) -> None:
            self.timeouts.append(value)

        def connect(self, destination: Any) -> None:
            self.destination = destination

        def sendall(self, value: bytes) -> None:
            self.request += value

        def makefile(self, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return self.response

        def getpeername(self) -> tuple[str, int]:
            return ("93.184.216.34", 443)

        def close(self) -> None:
            self.closed = True

    stub = SocketStub()
    sni: list[str] = []

    class ContextStub:
        def wrap_socket(self, value: Any, *, server_hostname: str) -> Any:
            sni.append(server_hostname)
            return value

    monkeypatch.setattr(socket, "socket", lambda *_args: stub)
    monkeypatch.setattr(ssl, "create_default_context", lambda: ContextStub())
    exchange = StdlibPinnedConnector(lambda: 0.0).open(
        address="93.184.216.34",
        port=443,
        hostname="source1.example",
        use_tls=True,
        target="/report",
        connect_timeout=5,
        read_timeout=15,
        total_timeout=30,
        max_header_bytes=64 * 1024,
    )
    assert stub.destination == ("93.184.216.34", 443)
    assert sni == ["source1.example"]
    assert b"Host: source1.example\r\n" in stub.request
    assert b"Cookie:" not in stub.request
    assert b"Authorization:" not in stub.request
    assert b"Proxy-Authorization:" not in stub.request
    assert exchange.peer_ip == "93.184.216.34"
    assert exchange.read_entity(5 * 1024 * 1024) == b"<html></html>"
    exchange.close()


def test_concrete_collector_rejects_nonpublic_dns_and_peer_rebinding() -> None:
    stack = _stack()
    response = _FakeExchange(
        peer_ip="93.184.216.35",
        status_code=200,
        headers={"content-type": "text/html; charset=UTF-8"},
        entity=b"<html></html>",
    )
    resolver = _FakeResolver({"source1.example": ["127.0.0.1"]})
    connector = _FakeConnector([response])
    with pytest.raises(I5ContractError, match="non-public"):
        SafeLocalCollector(resolver, connector, _FakeClock()).capture(**_collector_args(stack))
    assert connector.calls == []

    resolver = _FakeResolver({"source1.example": ["93.184.216.34"]})
    connector = _FakeConnector([response])
    with pytest.raises(I5ContractError, match="peer IP"):
        SafeLocalCollector(resolver, connector, _FakeClock()).capture(**_collector_args(stack))
    assert response.reads == 0


def test_concrete_collector_pdf_is_discovery_only_without_dns_or_body_download() -> None:
    stack = _stack()
    pdf = build_search_source(
        request=stack["request"],
        provider_response_id=stack["provider_response_id"],
        url="https://source1.example/report.pdf",
        title="PDF lead",
        publisher=None,
        publication_hint=None,
        media_kind="PDF",
        discovered_at=T5,
    )
    resolver = _FakeResolver({})
    connector = _FakeConnector([])
    args = _collector_args(stack)
    args["source"] = pdf
    result = SafeLocalCollector(resolver, connector, _FakeClock()).capture(**args)
    assert result["receipt"]["status"] == "BLOCKED"
    assert result["receipt"]["reason_codes"] == ["PDF_DISCOVERY_ONLY"]
    assert resolver.calls == []
    assert connector.calls == []
    assert result["compressed_entity"] is None


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({}, "Content-Type"),
        ({"content-type": "application/pdf; charset=UTF-8"}, "MIME"),
        ({"content-type": "text/html"}, "charset"),
        ({"content-type": "text/html; charset=ISO-8859-1"}, "charset"),
        (
            {"content-type": "text/html; charset=UTF-8", "content-encoding": "compress"},
            "encoding",
        ),
    ],
)
def test_concrete_collector_fails_closed_on_representation_ambiguity(
    headers: dict[str, str], message: str
) -> None:
    stack = _stack()
    response = _FakeExchange(
        peer_ip="93.184.216.34",
        status_code=200,
        headers=headers,
        entity=b"<html></html>",
    )
    collector = SafeLocalCollector(
        _FakeResolver({"source1.example": ["93.184.216.34"]}),
        _FakeConnector([response]),
        _FakeClock(),
    )
    with pytest.raises(I5ContractError, match=message):
        collector.capture(**_collector_args(stack))
    assert response.reads == 0


def test_concrete_collector_enforces_header_timeout_ratio_and_parser_limits() -> None:
    stack = _stack()
    too_many_nodes = ("<i></i>" * 100_001).encode()
    cases = [
        (
            _FakeExchange(
                peer_ip="93.184.216.34",
                status_code=200,
                headers={"content-type": "text/html; charset=UTF-8"},
                entity=b"<html></html>",
                header_bytes_total=65_537,
            ),
            _FakeClock(),
            "headers",
        ),
        (
            _FakeExchange(
                peer_ip="93.184.216.34",
                status_code=200,
                headers={"content-type": "text/html; charset=UTF-8"},
                entity=b"<html></html>",
            ),
            _FakeClock(monotonic_values=[0.0, 31.0]),
            "timeout",
        ),
        (
            _FakeExchange(
                peer_ip="93.184.216.34",
                status_code=200,
                headers={"content-type": "text/html; charset=UTF-8"},
                entity=too_many_nodes,
            ),
            _FakeClock(),
            "node count",
        ),
    ]
    for response, clock, message in cases:
        collector = SafeLocalCollector(
            _FakeResolver({"source1.example": ["93.184.216.34"]}),
            _FakeConnector([response]),
            clock,
        )
        with pytest.raises(I5ContractError, match=message):
            collector.capture(**_collector_args(stack))


def test_concrete_collector_fails_closed_when_brotli_decoder_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack()
    response = _FakeExchange(
        peer_ip="93.184.216.34",
        status_code=200,
        headers={
            "content-type": "text/html; charset=UTF-8",
            "content-encoding": "br",
        },
        entity=b"not-decoded",
    )
    collector = SafeLocalCollector(
        _FakeResolver({"source1.example": ["93.184.216.34"]}),
        _FakeConnector([response]),
        _FakeClock(),
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "brotli":
            raise ImportError("offline fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(I5ContractError, match="brotli decoder is unavailable"):
        collector.capture(**_collector_args(stack))
