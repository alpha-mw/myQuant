"""Fail-closed admission boundary for official CN exchange calendars.

No production wire decoder is registered until exact issuer response bytes,
endpoint semantics, response metadata, and a reviewed decoder are admitted as
one immutable source contract. In particular, project-authored JSON must never
be relabelled as a native SSE, SZSE, or BSE response.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Final, Literal
from urllib.parse import urlsplit

from quant_investor.contracts import ContractError, validate_artifact
from quant_investor.system.errors import SystemContractError

EvidenceRole = Literal["DAILY_STATUS", "SESSION_RULE"]

# Adding an entry is a production authority change and requires a retained
# native issuer capture plus an exact endpoint/response/decoder admission
# artifact. Tests may inject synthetic decoders into the assembler module, but
# they must never mutate this registry.
DECODER_IDS: Final[dict[tuple[str, str], str]] = {}
DECODER_ADMISSIONS: Final[dict[tuple[str, str], Mapping[str, Any]]] = {}

_UNVERIFIED = "OFFICIAL_CALENDAR_WIRE_CONTRACT_UNVERIFIED"
_ADMISSION_KIND = "system.exchange_calendar_decoder_admission"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_EXCHANGE_AUTHORITIES: Final = {
    "SSE": ("SSE_OFFICIAL", "www.sse.com.cn"),
    "SZSE": ("SZSE_OFFICIAL", "www.szse.cn"),
    "BSE": ("BSE_OFFICIAL", "www.bse.cn"),
}
_ADMISSION_FIELDS: Final = frozenset(
    {
        "decoder_admission_id",
        "state",
        "exchange_id",
        "evidence_role",
        "issuer",
        "endpoint_scheme",
        "endpoint_host",
        "endpoint_path_query_template",
        "redirect_policy",
        "http_status",
        "raw_media_type",
        "response_headers",
        "fixture_raw_file_ref",
        "fixture_raw_sha256",
        "fixture_captured_at",
        "decoder_id",
        "decoder_sha256",
        "fixture_projection_sha256",
        "review_basis",
    }
)


def decoder_code_sha256() -> str:
    """Return the exact installed admission-boundary module byte identity."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _reject(exchange: str, role: str) -> SystemContractError:
    return SystemContractError(f"{_UNVERIFIED}: {exchange}/{role}")


def _validate_endpoint(payload: Mapping[str, Any]) -> None:
    exchange = payload["exchange_id"]
    role = payload["evidence_role"]
    if exchange not in _EXCHANGE_AUTHORITIES or role not in {"DAILY_STATUS", "SESSION_RULE"}:
        raise SystemContractError("official calendar decoder admission subject differs")
    issuer, hostname = _EXCHANGE_AUTHORITIES[exchange]
    endpoint = payload["endpoint_path_query_template"]
    parsed = urlsplit(endpoint) if type(endpoint) is str else None
    if (
        payload["issuer"] != issuer
        or payload["endpoint_scheme"] != "https"
        or payload["endpoint_host"] != hostname
        or parsed is None
        or not endpoint.startswith("/")
        or parsed.scheme
        or parsed.netloc
        or parsed.fragment
    ):
        raise SystemContractError("official calendar decoder endpoint admission differs")


def _validate_response(payload: Mapping[str, Any]) -> None:
    if (
        payload["redirect_policy"] not in {"NO_REDIRECTS", "SAME_ISSUER_HOST_ONLY"}
        or payload["http_status"] != 200
        or type(payload["raw_media_type"]) is not str
        or not payload["raw_media_type"]
    ):
        raise SystemContractError("official calendar decoder response admission differs")
    headers = payload["response_headers"]
    if (
        type(headers) is not list
        or not headers
        or any(
            type(row) is not dict
            or set(row) != {"name", "value"}
            or type(row["name"]) is not str
            or row["name"] != row["name"].lower()
            or type(row["value"]) is not str
            for row in headers
        )
        or not any(
            row["name"] == "content-type" and row["value"] == payload["raw_media_type"]
            for row in headers
        )
    ):
        raise SystemContractError("official calendar response-header admission differs")


def _validate_fixture(payload: Mapping[str, Any]) -> None:
    fixture_ref = payload["fixture_raw_file_ref"]
    if (
        type(fixture_ref) is not dict
        or set(fixture_ref) != {"relative_path", "byte_sha256", "size"}
        or type(fixture_ref["relative_path"]) is not str
        or not fixture_ref["relative_path"]
        or type(fixture_ref["size"]) is not int
        or fixture_ref["size"] <= 0
        or fixture_ref["byte_sha256"] != payload["fixture_raw_sha256"]
    ):
        raise SystemContractError("official calendar native fixture admission differs")
    for field in ("fixture_raw_sha256", "decoder_sha256", "fixture_projection_sha256"):
        if type(payload[field]) is not str or _SHA256_RE.fullmatch(payload[field]) is None:
            raise SystemContractError("official calendar decoder SHA admission differs")
    captured_at = payload["fixture_captured_at"]
    try:
        parsed_at = datetime.strptime(captured_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError) as exc:
        raise SystemContractError("official calendar fixture capture time differs") from exc
    if parsed_at.strftime("%Y-%m-%dT%H:%M:%SZ") != captured_at:
        raise SystemContractError("official calendar fixture capture time differs")
    for field in ("decoder_admission_id", "decoder_id", "review_basis"):
        if (
            type(payload[field]) is not str
            or not payload[field]
            or payload[field].strip() != payload[field]
        ):
            raise SystemContractError("official calendar decoder text admission differs")


def validate_decoder_admission(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate the mandatory source contract for a future native decoder."""

    try:
        artifact = validate_artifact(document, expected_kind=_ADMISSION_KIND)
    except ContractError as exc:
        raise SystemContractError("official calendar decoder admission contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _ADMISSION_FIELDS or payload["state"] != "ADMITTED":
        raise SystemContractError("official calendar decoder admission fields differ")
    _validate_endpoint(payload)
    _validate_response(payload)
    _validate_fixture(payload)
    return artifact


def decoder_admission(exchange: str, role: EvidenceRole) -> dict[str, Any]:
    try:
        document = DECODER_ADMISSIONS[(exchange, role)]
    except KeyError as exc:
        raise _reject(exchange, role) from exc
    artifact = validate_decoder_admission(document)
    payload = artifact["payload"]
    if payload["exchange_id"] != exchange or payload["evidence_role"] != role:
        raise SystemContractError("official calendar decoder registry subject differs")
    return artifact


def decoder_id(exchange: str, role: EvidenceRole) -> str:
    """Return an admitted production decoder identity or fail closed."""

    artifact = decoder_admission(exchange, role)
    value = artifact["payload"]["decoder_id"]
    if DECODER_IDS.get((exchange, role)) != value:
        raise SystemContractError("official calendar decoder registry identity differs")
    return value


def decode_daily_status(exchange: str, raw: bytes, *, media_type: str) -> list[dict[str, str]]:
    """Reject daily evidence while no native issuer contract is admitted."""

    del raw, media_type
    raise _reject(exchange, "DAILY_STATUS")


def decode_session_intervals(exchange: str, raw: bytes, *, media_type: str) -> list[dict[str, str]]:
    """Reject session evidence while no native issuer contract is admitted."""

    del raw, media_type
    raise _reject(exchange, "SESSION_RULE")


def decode_capture_projection(
    exchange: str,
    role: EvidenceRole,
    raw: bytes,
    *,
    media_type: str,
) -> Mapping[str, object]:
    """Reject every unadmitted issuer body before it can gain authority."""

    del raw, media_type
    decoder_id(exchange, role)
    raise _reject(exchange, role)  # pragma: no cover - registry is empty


__all__ = [
    "DECODER_IDS",
    "DECODER_ADMISSIONS",
    "decode_capture_projection",
    "decode_daily_status",
    "decode_session_intervals",
    "decoder_code_sha256",
    "decoder_admission",
    "decoder_id",
    "validate_decoder_admission",
]
