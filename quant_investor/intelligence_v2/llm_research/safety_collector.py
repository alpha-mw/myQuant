"""Concrete DNS-pinned HTML collector with injectable offline boundaries.

The default resolver and connector use raw sockets: they do not consult proxy
environment variables, persist cookies, or carry caller-controlled headers.
Tests inject all I/O boundaries and never open a live connection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import ipaddress
import socket
import ssl
import time
from typing import Any, Callable, Final, Protocol
from urllib.parse import urljoin, urlsplit
import zlib

from .._core import canonical_bytes, sha256
from ._contracts import canonical_url, fail
from .capture import build_capture_disposition, build_capture_receipt, validate_capture_policy
from .public_search import validate_search_source

_MIB: Final = 1024 * 1024
_REDIRECT_CODES: Final = frozenset({301, 302, 303, 307, 308})
_ALLOWED_MIME: Final = frozenset({"text/html", "application/xhtml+xml"})
_ALLOWED_CHARSETS: Final = frozenset({"UTF-8", "GB18030", "GBK", "BIG5"})
_ALLOWED_ENCODINGS: Final = frozenset({"identity", "gzip", "deflate", "br"})
PARSER_IDENTITY: Final = "python.stdlib.html.parser"
PARSER_VERSION: Final = "htmlparser-v1"
PARSER_OPTIONS: Final = b'{"convert_charrefs":true,"mode":"html-limit-audit"}'
PARSER_OPTIONS_SHA256: Final = sha256(
    hashlib.sha256(PARSER_OPTIONS).hexdigest(), label="parser_options_sha256"
)
DECODER_MANIFEST: Final = b"identity|gzip:zlib-wrapper|deflate:zlib-wrapper|br:optional-stream"
DECODER_MANIFEST_SHA256: Final = sha256(
    hashlib.sha256(DECODER_MANIFEST).hexdigest(),
    label="decoder_manifest_sha256",
)


class Resolver(Protocol):
    def resolve(self, hostname: str, port: int, /) -> Sequence[str]: ...


class Clock(Protocol):
    def utc_now(self) -> str: ...

    def monotonic(self) -> float: ...


class HTTPExchange(Protocol):
    peer_ip: str
    status_code: int
    selected_headers: dict[str, str]
    header_bytes_total: int

    def read_entity(self, max_bytes: int, /) -> bytes: ...

    def close(self) -> None: ...


class PinnedConnector(Protocol):
    def open(
        self,
        *,
        address: str,
        port: int,
        hostname: str,
        use_tls: bool,
        target: str,
        connect_timeout: int,
        read_timeout: int,
        total_timeout: float,
        max_header_bytes: int,
    ) -> HTTPExchange: ...


class SystemResolver:
    """System DNS resolver; every returned address is validated by the collector."""

    def resolve(self, hostname: str, port: int, /) -> Sequence[str]:
        rows = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        return tuple(dict.fromkeys(str(row[4][0]) for row in rows))


class SystemClock:
    def utc_now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def monotonic(self) -> float:
        return time.monotonic()


def _readline(stream: Any, *, remaining: int, before_read: Callable[[], None]) -> tuple[bytes, int]:
    if remaining <= 0:
        fail("HTTP headers exceed 64 KiB")
    before_read()
    line = stream.readline(remaining + 1)
    if len(line) > remaining:
        fail("HTTP headers exceed 64 KiB")
    if not line.endswith(b"\r\n"):
        fail("HTTP header line is truncated or noncanonical")
    return line, remaining - len(line)


def _status_code(line: bytes) -> int:
    parts = line[:-2].split(b" ", 2)
    if len(parts) < 2 or parts[0] not in {b"HTTP/1.0", b"HTTP/1.1"}:
        fail("HTTP status line is invalid")
    try:
        status = int(parts[1])
    except ValueError as exc:
        raise ValueError("HTTP status code is invalid") from exc
    if not 100 <= status <= 599:
        fail("HTTP status code is invalid")
    return status


def _header_row(line: bytes, *, existing: Mapping[str, str]) -> tuple[str, str]:
    if line[:1] in {b" ", b"\t"} or b":" not in line:
        fail("folded or malformed HTTP header is forbidden")
    raw_name, raw_value = line[:-2].split(b":", 1)
    try:
        name = raw_name.decode("ascii").lower()
        value = raw_value.strip().decode("latin-1")
    except UnicodeError as exc:
        raise ValueError("HTTP header encoding is invalid") from exc
    if not name or not name.replace("-", "").isalnum() or name in existing:
        fail("duplicate or invalid HTTP header is forbidden")
    return name, value


def _read_headers(
    stream: Any, *, maximum: int, before_read: Callable[[], None]
) -> tuple[int, dict[str, str], int]:
    remaining = maximum
    status_line, remaining = _readline(stream, remaining=remaining, before_read=before_read)
    status = _status_code(status_line)
    headers: dict[str, str] = {}
    while True:
        line, remaining = _readline(stream, remaining=remaining, before_read=before_read)
        if line == b"\r\n":
            break
        name, value = _header_row(line, existing=headers)
        headers[name] = value
    return status, headers, maximum - remaining


class _SocketExchange:
    def __init__(
        self,
        connected: socket.socket,
        *,
        max_header_bytes: int,
        read_timeout: int,
        total_timeout: float,
        monotonic: Callable[[], float],
    ) -> None:
        self._socket = connected
        self._stream = connected.makefile("rb")
        self._read_timeout = read_timeout
        self._monotonic = monotonic
        self._deadline = monotonic() + total_timeout
        self.peer_ip = str(connected.getpeername()[0])
        self.status_code, self.selected_headers, self.header_bytes_total = _read_headers(
            self._stream,
            maximum=max_header_bytes,
            before_read=self._before_read,
        )

    def _before_read(self) -> None:
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            fail("capture exceeded total timeout")
        self._socket.settimeout(min(self._read_timeout, remaining))

    def _read_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            self._before_read()
            chunk = self._stream.read(length - len(chunks))
            if not chunk:
                fail("HTTP entity ended before Content-Length")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_chunked(self, maximum: int) -> bytes:
        body = bytearray()
        while True:
            self._before_read()
            line = self._stream.readline(4097)
            if not line.endswith(b"\r\n") or len(line) > 4096:
                fail("chunk header is invalid")
            raw_size = line[:-2].split(b";", 1)[0]
            try:
                size = int(raw_size, 16)
            except ValueError as exc:
                raise ValueError("chunk size is invalid") from exc
            if size == 0:
                if self._stream.readline(4097) != b"\r\n":
                    fail("chunk trailer is forbidden")
                return bytes(body)
            if len(body) + size > maximum:
                fail("compressed entity exceeds 5 MiB")
            body.extend(self._read_exact(size))
            if self._read_exact(2) != b"\r\n":
                fail("chunk terminator is invalid")

    def read_entity(self, max_bytes: int, /) -> bytes:
        transfer = self.selected_headers.get("transfer-encoding")
        length_text = self.selected_headers.get("content-length")
        if transfer is not None:
            if transfer.lower() != "chunked" or length_text is not None:
                fail("ambiguous HTTP entity framing is forbidden")
            return self._read_chunked(max_bytes)
        if length_text is not None:
            try:
                length = int(length_text)
            except ValueError as exc:
                raise ValueError("Content-Length is invalid") from exc
            if not 0 <= length <= max_bytes:
                fail("compressed entity exceeds 5 MiB")
            return self._read_exact(length)
        body = bytearray()
        while True:
            self._before_read()
            chunk = self._stream.read(min(64 * 1024, max_bytes + 1 - len(body)))
            if not chunk:
                return bytes(body)
            body.extend(chunk)
            if len(body) > max_bytes:
                fail("compressed entity exceeds 5 MiB")

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._socket.close()


class StdlibPinnedConnector:
    """Raw-socket connector that never invokes a proxy or a second DNS lookup."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic

    def open(
        self,
        *,
        address: str,
        port: int,
        hostname: str,
        use_tls: bool,
        target: str,
        connect_timeout: int,
        read_timeout: int,
        total_timeout: float,
        max_header_bytes: int,
    ) -> HTTPExchange:
        parsed = ipaddress.ip_address(address)
        family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
        connected = socket.socket(family, socket.SOCK_STREAM)
        connected.settimeout(min(connect_timeout, total_timeout))
        destination: Any = (address, port, 0, 0) if parsed.version == 6 else (address, port)
        try:
            connected.connect(destination)
            if use_tls:
                context = ssl.create_default_context()
                connected = context.wrap_socket(connected, server_hostname=hostname)
            connected.settimeout(min(read_timeout, total_timeout))
            host_header = hostname
            if ":" in hostname:
                host_header = f"[{hostname}]"
            default_port = 443 if use_tls else 80
            if port != default_port:
                host_header = f"{host_header}:{port}"
            request = (
                f"GET {target} HTTP/1.1\r\n"
                f"Host: {host_header}\r\n"
                "Accept: text/html,application/xhtml+xml\r\n"
                "Accept-Encoding: gzip, deflate, br\r\n"
                "Connection: close\r\n"
                "User-Agent: myQuant-I5-SafetyCollector/1\r\n\r\n"
            ).encode("ascii")
            connected.sendall(request)
            return _SocketExchange(
                connected,
                max_header_bytes=max_header_bytes,
                read_timeout=read_timeout,
                total_timeout=total_timeout,
                monotonic=self._monotonic,
            )
        except Exception:
            connected.close()
            raise


class _CountingHTMLParser(HTMLParser):
    _VOID: Final = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.node_count = 0
        self.depth = 0
        self.max_depth = 0

    def _node(self) -> None:
        self.node_count += 1
        if self.node_count > 100_000:
            fail("HTML node count exceeds 100000")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._node()
        if tag not in self._VOID:
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            if self.max_depth > 128:
                fail("HTML depth exceeds 128")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag, attrs
        self._node()

    def handle_endtag(self, tag: str) -> None:
        del tag
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        if data:
            self._node()

    def handle_comment(self, data: str) -> None:
        del data
        self._node()


def _public_addresses(values: Sequence[str]) -> tuple[str, ...]:
    addresses = []
    for value in values:
        try:
            parsed = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("resolver returned an invalid IP address") from exc
        if not parsed.is_global or parsed.is_multicast or parsed.is_reserved:
            fail("resolver returned a non-public IP address")
        addresses.append(parsed.compressed)
    if not addresses or len(addresses) != len(set(addresses)):
        fail("resolver address set is empty or duplicated")
    return tuple(sorted(addresses, key=lambda item: ipaddress.ip_address(item).packed))


def _url_target(url: str) -> tuple[str, int, bool, str]:
    parts = urlsplit(url)
    if parts.hostname is None:
        fail("capture URL hostname is missing")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    return parts.hostname, port, parts.scheme == "https", target


def _representation(headers: Mapping[str, str]) -> tuple[str, str, str]:
    content_type = headers.get("content-type")
    if content_type is None:
        fail("Content-Type is required")
    parts = [part.strip() for part in content_type.split(";")]
    mime = parts[0].lower()
    charsets = []
    for parameter in parts[1:]:
        if "=" not in parameter:
            fail("Content-Type parameter is malformed")
        name, value = parameter.split("=", 1)
        if name.strip().lower() == "charset":
            charsets.append(value.strip().strip('"').upper())
    if mime not in _ALLOWED_MIME:
        fail("MIME type is forbidden")
    if len(charsets) != 1 or charsets[0] not in _ALLOWED_CHARSETS:
        fail("charset is missing, conflicting, guessed, or forbidden")
    encoding = headers.get("content-encoding", "identity").lower()
    if encoding not in _ALLOWED_ENCODINGS:
        fail("content encoding is forbidden")
    return mime, charsets[0], encoding


def _bounded_output(parts: Sequence[bytes], *, compressed_size: int) -> bytes:
    output = b"".join(parts)
    if len(output) > 20 * _MIB or len(output) > max(1, compressed_size) * 20:
        fail("captured entity exceeds decoded size or decompression ratio")
    return output


def _decode_entity(value: bytes, *, encoding: str) -> bytes:
    if encoding == "identity":
        return _bounded_output([value], compressed_size=len(value))
    if encoding in {"gzip", "deflate"}:
        wrapper = 16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
        decoder = zlib.decompressobj(wrapper)
        parts = []
        for start in range(0, len(value), 64 * 1024):
            end = start + 64 * 1024
            parts.append(decoder.decompress(value[start:end], 20 * _MIB + 1))
            _bounded_output(parts, compressed_size=len(value))
        parts.append(decoder.flush(20 * _MIB + 1))
        if not decoder.eof or decoder.unused_data:
            fail("compressed entity is truncated or concatenated")
        return _bounded_output(parts, compressed_size=len(value))
    try:
        import brotli  # type: ignore[import-not-found]
    except ImportError:
        fail("brotli decoder is unavailable")
    decoder = brotli.Decompressor()
    if not hasattr(decoder, "process"):
        fail("brotli streaming decoder is unavailable")
    parts = []
    for start in range(0, len(value), 64 * 1024):
        end = start + 64 * 1024
        parts.append(decoder.process(value[start:end]))
        _bounded_output(parts, compressed_size=len(value))
    if hasattr(decoder, "is_finished") and not decoder.is_finished():
        fail("brotli entity is truncated")
    return _bounded_output(parts, compressed_size=len(value))


def _parse_html(decoded: bytes, *, charset: str) -> tuple[bytes, int, int]:
    try:
        decoded_text = decoded.decode(charset, errors="strict")
    except (LookupError, UnicodeDecodeError) as exc:
        raise ValueError("HTML charset decoding failed closed") from exc
    parser_input = decoded_text.encode("utf-8", errors="strict")
    parser = _CountingHTMLParser()
    parser.feed(decoded_text)
    parser.close()
    return parser_input, parser.node_count, parser.max_depth


@dataclass(frozen=True)
class SafeLocalCollector:
    resolver: Resolver
    connector: PinnedConnector
    clock: Clock

    @classmethod
    def system_default(cls) -> "SafeLocalCollector":
        clock = SystemClock()
        return cls(SystemResolver(), StdlibPinnedConnector(clock.monotonic), clock)

    def _validate_policy_identity(self, policy: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_capture_policy(policy)
        if (
            validated["parser_identity"] != PARSER_IDENTITY
            or validated["parser_version"] != PARSER_VERSION
            or validated["parser_options_sha256"] != PARSER_OPTIONS_SHA256
            or validated["decoder_manifest_sha256"] != DECODER_MANIFEST_SHA256
        ):
            fail("capture policy does not bind the concrete collector implementation")
        return validated

    def _open_hop(
        self, *, current: str, redirect_count: int, elapsed: float
    ) -> tuple[HTTPExchange, dict[str, Any], str | None]:
        hostname, port, use_tls, target = _url_target(current)
        addresses = _public_addresses(self.resolver.resolve(hostname, port))
        exchange = self.connector.open(
            address=addresses[0],
            port=port,
            hostname=hostname,
            use_tls=use_tls,
            target=target,
            connect_timeout=5,
            read_timeout=15,
            total_timeout=30 - elapsed,
            max_header_bytes=64 * 1024,
        )
        try:
            peer = _public_addresses([exchange.peer_ip])[0]
            if peer not in addresses:
                fail("actual peer IP is outside validated DNS set")
            status = exchange.status_code
            location = exchange.selected_headers.get("location")
            next_url = None
            if status in _REDIRECT_CODES:
                if location is None or redirect_count == 5:
                    fail("redirect is missing Location or exceeds five redirects")
                next_url = canonical_url(urljoin(current, location), label="redirect.location")
            elif location is not None:
                fail("non-redirect response carries Location")
            hop = {
                "url": current,
                "resolved_addresses": list(addresses),
                "peer_ip": peer,
                "status_code": status,
                "location": next_url,
            }
            return exchange, hop, next_url
        except Exception:
            exchange.close()
            raise

    def _open_final(self, *, initial_url: str, started: float) -> tuple[HTTPExchange, list]:
        current = initial_url
        hops = []
        for redirect_count in range(6):
            elapsed = self.clock.monotonic() - started
            if elapsed < 0 or elapsed >= 30:
                fail("capture exceeded total timeout")
            exchange, hop, next_url = self._open_hop(
                current=current,
                redirect_count=redirect_count,
                elapsed=elapsed,
            )
            hops.append(hop)
            if next_url is None:
                return exchange, hops
            exchange.close()
            current = next_url
        fail("capture lacks a final response")

    @staticmethod
    def _read_final(exchange: HTTPExchange) -> tuple[bytes, bytes, bytes, str, str, str, int, int]:
        if not 200 <= exchange.status_code <= 299:
            fail("final HTTP status is not successful")
        if (
            type(exchange.header_bytes_total) is not int
            or not 0 <= exchange.header_bytes_total <= 64 * 1024
            or len(canonical_bytes(dict(exchange.selected_headers))) > exchange.header_bytes_total
        ):
            fail("HTTP headers exceed 64 KiB or lack an exact byte count")
        mime, charset, encoding = _representation(exchange.selected_headers)
        compressed = exchange.read_entity(5 * _MIB)
        if type(compressed) is not bytes or len(compressed) > 5 * _MIB:
            fail("compressed entity exceeds 5 MiB")
        decoded = _decode_entity(compressed, encoding=encoding)
        parser_input, node_count, max_depth = _parse_html(decoded, charset=charset)
        return (
            compressed,
            decoded,
            parser_input,
            mime,
            charset,
            encoding,
            node_count,
            max_depth,
        )

    @staticmethod
    def _observation(
        *,
        exchange: HTTPExchange,
        hops: list[dict[str, Any]],
        mime: str,
        charset: str,
        encoding: str,
        node_count: int,
        max_depth: int,
    ) -> dict[str, Any]:
        return {
            "redirect_chain": hops,
            "selected_headers": dict(exchange.selected_headers),
            "content_encoding": encoding,
            "mime_type": mime,
            "charset": charset,
            "html_node_count": node_count,
            "html_max_depth": max_depth,
            "header_bytes_total": exchange.header_bytes_total,
            "connect_timeout_seconds": 5,
            "read_timeout_seconds": 15,
            "total_timeout_seconds": 30,
            "transfer_decoding_complete": True,
            "proxy_environment_used": False,
            "cookies_sent": False,
            "authorization_sent": False,
            "library_reresolution": False,
            "hostname_sni_preserved": True,
        }

    def capture(
        self,
        *,
        source: Mapping[str, Any],
        source_request: Mapping[str, Any],
        provider_response_id: str,
        policy: Mapping[str, Any],
        publication_evidence_ref: Mapping[str, Any] | None,
        transport_attestation_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_doc = validate_search_source(
            source,
            request=source_request,
            provider_response_id=provider_response_id,
        )
        policy_doc = self._validate_policy_identity(policy)
        if source_doc["media_kind"] == "PDF":
            recorded = self.clock.utc_now()
            return {
                "receipt": build_capture_disposition(
                    source=source_doc,
                    policy=policy_doc,
                    status="BLOCKED",
                    reason_codes=["PDF_DISCOVERY_ONLY"],
                    recorded_at=recorded,
                ),
                "transport_observation": None,
                "compressed_entity": None,
                "decoded_entity": None,
                "parser_input": None,
            }
        started = self.clock.monotonic()
        final_exchange, hops = self._open_final(initial_url=source_doc["url"], started=started)
        try:
            (
                compressed,
                decoded,
                parser_input,
                mime,
                charset,
                encoding,
                node_count,
                max_depth,
            ) = self._read_final(final_exchange)
            if self.clock.monotonic() - started > 30:
                fail("capture exceeded total timeout")
            captured_at = self.clock.utc_now()
            observation = self._observation(
                exchange=final_exchange,
                hops=hops,
                mime=mime,
                charset=charset,
                encoding=encoding,
                node_count=node_count,
                max_depth=max_depth,
            )
            receipt = build_capture_receipt(
                source=source_doc,
                source_request=source_request,
                provider_response_id=provider_response_id,
                policy=policy_doc,
                transport_observation=observation,
                compressed_entity=compressed,
                decoded_entity=decoded,
                parser_input=parser_input,
                publication_evidence_ref=publication_evidence_ref,
                transport_attestation_ref=transport_attestation_ref,
                captured_at=captured_at,
            )
            return {
                "receipt": receipt,
                "transport_observation": observation,
                "compressed_entity": compressed,
                "decoded_entity": decoded,
                "parser_input": parser_input,
            }
        finally:
            final_exchange.close()


__all__ = [
    "Clock",
    "DECODER_MANIFEST_SHA256",
    "HTTPExchange",
    "PARSER_IDENTITY",
    "PARSER_OPTIONS_SHA256",
    "PARSER_VERSION",
    "PinnedConnector",
    "Resolver",
    "SafeLocalCollector",
    "StdlibPinnedConnector",
    "SystemClock",
    "SystemResolver",
]
