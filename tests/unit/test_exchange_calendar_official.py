from __future__ import annotations

import copy
import hashlib

import pytest

from quant_investor.contracts import seal_artifact
from quant_investor.market.exchange_calendar_official import (
    DECODER_IDS,
    EVIDENCE_ROLES,
    decode_capture_projection,
    decode_session_intervals,
    decoder_code_sha256,
    decoder_id,
    validate_decoder_admission,
)
from quant_investor.system import SystemContractError

BASE = "2026-08-16T00:00:00Z"


@pytest.mark.parametrize("exchange", ["SSE", "SZSE", "BSE"])
@pytest.mark.parametrize("role", sorted(EVIDENCE_ROLES))
def test_unadmitted_official_calendar_wire_contracts_fail_closed(exchange: str, role: str) -> None:
    assert DECODER_IDS == {}
    with pytest.raises(
        SystemContractError,
        match="OFFICIAL_CALENDAR_WIRE_CONTRACT_UNVERIFIED",
    ):
        decoder_id(exchange, role)  # type: ignore[arg-type]
    with pytest.raises(
        SystemContractError,
        match="OFFICIAL_CALENDAR_WIRE_CONTRACT_UNVERIFIED",
    ):
        decode_capture_projection(
            exchange,
            role,  # type: ignore[arg-type]
            b'{"project_authored":"not issuer bytes"}',
            media_type="application/json",
        )


@pytest.mark.parametrize("exchange", ["SSE", "SZSE", "BSE"])
def test_project_authored_synthetic_bodies_never_gain_production_authority(
    exchange: str,
) -> None:
    synthetic = b'{"result":[{"tradeDate":"2024-01-02","isOpen":true}]}'
    with pytest.raises(
        SystemContractError,
        match="OFFICIAL_CALENDAR_WIRE_CONTRACT_UNVERIFIED",
    ):
        decode_session_intervals(exchange, synthetic, media_type="application/json")
    assert len(decoder_code_sha256()) == 64


def test_decoder_admission_contract_binds_endpoint_response_and_native_fixture() -> None:
    raw = b"retained-real-capture-placeholder-for-contract-test"
    raw_sha = hashlib.sha256(raw).hexdigest()
    admission = seal_artifact(
        "system.exchange_calendar_decoder_admission",
        {
            "decoder_admission_id": "sse-daily-capture-admission-test-only",
            "state": "ADMITTED",
            "exchange_id": "SSE",
            "evidence_role": "ANNUAL_HOLIDAY_NOTICE",
            "issuer": "SSE_OFFICIAL",
            "endpoint_scheme": "https",
            "endpoint_host": "www.sse.com.cn",
            "endpoint_path_query_template": "/issuer/path?date=2024",
            "issuer_category_id": None,
            "category_scope": None,
            "category_completeness_policy": None,
            "query_window_semantics": None,
            "required_query_parameters": [],
            "page_parameter": None,
            "cursor_parameter": None,
            "required_category_set_id": None,
            "discovery_start_date": None,
            "fixture_request_url": "https://www.sse.com.cn/issuer/path?date=2024",
            "fixture_effective_url": "https://www.sse.com.cn/issuer/path?date=2024",
            "fixture_redirect_chain": [],
            "fixture_tls_verified": True,
            "redirect_policy": "NO_REDIRECTS",
            "http_status": 200,
            "raw_media_type": "application/json",
            "response_headers": [{"name": "content-type", "value": "application/json"}],
            "fixture_raw_file_ref": {
                "relative_path": "fixtures/sse-daily-native.bin",
                "byte_sha256": raw_sha,
                "size": len(raw),
            },
            "fixture_raw_sha256": raw_sha,
            "fixture_captured_at": BASE,
            "decoder_id": "test-only.unregistered.sse-daily",
            "decoder_sha256": "1" * 64,
            "fixture_projection_sha256": "2" * 64,
            "review_basis": "contract-shape-test-only-not-production-admission",
        },
        created_at=BASE,
    )
    assert validate_decoder_admission(admission) == admission
    assert DECODER_IDS == {}

    redirected = copy.deepcopy(admission["payload"])
    target = "https://www.sse.com.cn/issuer/path?date=2024&redirected=1"
    redirected["fixture_effective_url"] = target
    redirected["fixture_redirect_chain"] = [target]
    redirected_admission = seal_artifact(
        "system.exchange_calendar_decoder_admission",
        redirected,
        created_at=BASE,
    )
    with pytest.raises(SystemContractError, match="NO_REDIRECTS"):
        validate_decoder_admission(redirected_admission)

    redirected["redirect_policy"] = "SAME_ISSUER_HOST_ONLY"
    same_host_admission = seal_artifact(
        "system.exchange_calendar_decoder_admission",
        redirected,
        created_at=BASE,
    )
    assert validate_decoder_admission(same_host_admission) == same_host_admission

    redirected["fixture_effective_url"] = "https://www.bse.cn/issuer/path"
    redirected["fixture_redirect_chain"] = [redirected["fixture_effective_url"]]
    cross_host = seal_artifact(
        "system.exchange_calendar_decoder_admission",
        redirected,
        created_at=BASE,
    )
    with pytest.raises(SystemContractError, match="redirect authority"):
        validate_decoder_admission(cross_host)
