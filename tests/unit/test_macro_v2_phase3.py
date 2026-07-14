from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pytest

from quant_investor.data._tushare_client import TushareClientPool
from quant_investor.macro.store import (
    MacroObservationStoreError,
    load_observations,
    pointer_sha256,
)
from quant_investor.macro.tushare_normalizer import (
    AVAILABILITY_POLICY,
    MacroNormalizationError,
    normalize_tushare_bundle,
    normalize_tushare_bundle_file,
    publish_tushare_normalization,
)

INDICATORS = (
    "cn.gdp_yoy",
    "cn.cpi_yoy",
    "cn.ppi_yoy",
    "cn.m1_yoy",
    "cn.m2_yoy",
    "cn.social_financing_flow",
    "cn.pmi_manufacturing",
)


def _evidence(indicator_id: str, timestamp: str) -> dict[str, str]:
    pboc = indicator_id in {
        "cn.m1_yoy",
        "cn.m2_yoy",
        "cn.social_financing_flow",
    }
    return {
        "indicator_id": indicator_id,
        "period_end": "2024-03-31",
        "release_at": timestamp,
        "available_at": timestamp,
        "time_precision": "timestamp",
        "evidence_source_system": "pboc_official" if pboc else "nbs_official",
        "evidence_record_id": f"release:{indicator_id}:2024-03-31",
        "evidence_url": (
            "https://www.pbc.gov.cn/fixture"
            if pboc
            else "https://www.stats.gov.cn/fixture"
        ),
    }


def _raw() -> dict[str, object]:
    return {
        "schema_version": "macro-tushare-raw-bundle.v1",
        "provider": "tushare",
        "fetched_at": "2024-05-02T06:00:00+00:00",
        "request_parameters": {"start_m": "202403", "end_m": "202403"},
        "tables": {
            "cn_gdp": [{"quarter": "2024Q1", "gdp_yoy": 5.3}],
            "cn_cpi": [{"month": "202403", "nt_yoy": 0.1}],
            "cn_ppi": [{"month": "202403", "ppi_yoy": -2.8}],
            "cn_m": [{"month": "202403", "m1_yoy": 1.1, "m2_yoy": 8.3}],
            "sf_month": [{"month": "202403", "inc_month": 48675.0}],
            "cn_pmi": [{"month": "202403", "pmi010000": 50.8}],
            "cn_schedule": [],
        },
    }


def _plan(indicators=INDICATORS) -> dict[str, object]:
    return {
        "schema_version": "macro-backfill-plan.v1",
        "market": "CN",
        "requested_scope": [
            {"indicator_id": indicator_id, "period_end": "2024-03-31"}
            for indicator_id in indicators
        ],
    }


def _capture() -> dict[str, object]:
    return {
        "schema_version": "macro-release-evidence-capture.v1",
        "captured_at": "2024-05-03T00:00:00+00:00",
        "records": [
            _evidence("cn.gdp_yoy", "2024-04-16T02:00:00+00:00"),
            _evidence("cn.cpi_yoy", "2024-04-11T01:30:00+00:00"),
            _evidence("cn.ppi_yoy", "2024-04-11T01:30:00+00:00"),
            _evidence("cn.m1_yoy", "2024-04-12T09:00:00+00:00"),
            _evidence("cn.m2_yoy", "2024-04-12T09:00:00+00:00"),
            _evidence("cn.social_financing_flow", "2024-04-12T09:00:00+00:00"),
            _evidence("cn.pmi_manufacturing", "2024-03-31T01:30:00+00:00"),
        ],
    }


def _bytes(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def _normalize(raw=None, plan=None, capture=None):
    raw, plan, capture = raw or _raw(), plan or _plan(), capture or _capture()
    return normalize_tushare_bundle(
        raw,
        plan=plan,
        evidence_capture=capture,
        raw_bundle_file_sha256=hashlib.sha256(_bytes(raw)).hexdigest(),
        plan_file_sha256=hashlib.sha256(_bytes(plan)).hexdigest(),
        evidence_capture_sha256=hashlib.sha256(_bytes(capture)).hexdigest(),
    )


def _prepare(tmp_path: Path, run_id="n1", raw=None, plan=None, capture=None):
    raw, plan, capture = raw or _raw(), plan or _plan(), capture or _capture()
    paths = []
    for name, value in (("raw", raw), ("plan", plan), ("evidence", capture)):
        path = tmp_path / f"{run_id}_{name}.json"
        path.write_bytes(_bytes(value))
        paths.append(path)
    return normalize_tushare_bundle_file(
        paths[0],
        plan_path=paths[1],
        evidence_path=paths[2],
        output_root=tmp_path / "prepared",
        run_id=run_id,
    )


def _publish(prepared, root: Path, run_id="g1", pointer=""):
    return publish_tushare_normalization(
        prepared["artifacts"]["manifest"],
        observations_root=root,
        run_id=run_id,
        expected_pointer_sha256=pointer,
        expected_manifest_sha256=prepared["normalization_manifest_sha256"],
        expected_plan_sha256=prepared["backfill_plan_sha256"],
    )


def test_documented_tushare_tables_normalize_with_verified_evidence():
    result = _normalize()
    assert result.manifest["status"] == "OK"
    assert result.manifest["availability_policy"] == AVAILABILITY_POLICY
    assert result.manifest["national_registry_coverage"] == 7 / 16
    assert len(result.observations) == 7
    assert result.quarantine == ()


def test_pmi_uppercase_local_raw_alias_is_supported():
    raw = _raw()
    raw["tables"]["cn_pmi"] = [{"month": "202403", "PMI010000": 50.8}]
    result = _normalize(raw=raw)
    assert result.manifest["status"] == "OK"
    assert result.quarantine == ()


def test_pmi_alias_conflict_is_quarantined():
    raw = _raw()
    raw["tables"]["cn_pmi"] = [
        {"month": "202403", "pmi010000": 50.8, "PMI010000": 49.9}
    ]
    result = _normalize(raw=raw)
    assert result.manifest["status"] == "degraded"
    assert any(
        item["reason"] == "value_field_alias_conflict"
        for item in result.quarantine
    )


def test_cn_schedule_date_only_is_quarantined():
    raw = _raw()
    raw["tables"] = {
        "cn_cpi": [{"month": "202403", "nt_yoy": 0.1}],
        "cn_schedule": [{"publish_date": "20240411", "data_api": "cn_cpi"}],
    }
    result = _normalize(
        raw, _plan(("cn.cpi_yoy",)), {**_capture(), "records": []}
    )
    assert result.manifest["status"] == "blocked"
    assert (
        result.quarantine[0]["reason"]
        == "cn_schedule_date_only_not_promotable"
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"time_precision": "date"}, "timestamp_required"),
        ({"evidence_source_system": "pboc_official"}, "issuer_mismatch"),
        ({"evidence_url": "https://example.com/fake"}, "domain_mismatch"),
        (
            {"release_at": "2024-03-01T00:00:00+00:00"},
            "release_before_period_end",
        ),
        (
            {"available_at": "2024-06-01T00:00:00+00:00"},
            "availability_after_raw_fetch",
        ),
    ],
)
def test_invalid_availability_evidence_is_quarantined(mutation, reason):
    raw = _raw()
    raw["tables"] = {"cn_cpi": [{"month": "202403", "nt_yoy": 0.1}]}
    evidence = {
        **_evidence("cn.cpi_yoy", "2024-04-11T01:30:00+00:00"),
        **mutation,
    }
    result = _normalize(
        raw, _plan(("cn.cpi_yoy",)), {**_capture(), "records": [evidence]}
    )
    assert result.observations == ()
    assert reason in result.quarantine[0]["reason"]


def test_input_order_does_not_change_observations_but_rebinds_capture():
    capture = _capture()
    random.Random(9).shuffle(capture["records"])
    first, second = _normalize(), _normalize(capture=capture)
    assert [item.content_hash for item in first.observations] == [
        item.content_hash for item in second.observations
    ]
    assert (
        first.manifest["evidence_capture_sha256"]
        != second.manifest["evidence_capture_sha256"]
    )


def test_secret_bearing_input_is_rejected():
    raw = _raw()
    raw["request_parameters"] = {"token": "must-not-enter-artifacts"}
    with pytest.raises(MacroNormalizationError, match="secret_key_rejected"):
        _normalize(raw=raw)


@pytest.mark.parametrize(
    "secret_key",
    (
        "access_token",
        "tushare_token",
        "client_secret",
        "api-token",
        "x-api-key",
        "set-cookie",
        "credentials",
    ),
)
def test_secret_key_variants_are_rejected(secret_key):
    raw = _raw()
    raw["request_parameters"] = {secret_key: "must-not-enter-artifacts"}
    with pytest.raises(MacroNormalizationError, match="secret_key_rejected"):
        _normalize(raw=raw)


@pytest.mark.parametrize(
    ("url", "reason"),
    (
        ("https://user:pass@example.com/data", "url_userinfo_rejected"),
        (
            "https://example.com/data?access_token=secret",
            "url_secret_query_rejected",
        ),
    ),
)
def test_secret_bearing_urls_are_rejected(url, reason):
    raw = _raw()
    raw["request_parameters"] = {"source_url": url}
    with pytest.raises(MacroNormalizationError, match=reason):
        _normalize(raw=raw)


def test_evidence_url_with_leading_space_and_userinfo_is_rejected():
    capture = _capture()
    capture["records"][1]["evidence_url"] = (
        " https://leaked-user:leaked-pass@www.stats.gov.cn/fixture"
    )
    with pytest.raises(MacroNormalizationError, match="url_userinfo_rejected"):
        _normalize(capture=capture)


def test_evidence_cannot_attest_to_future_availability():
    raw = _raw()
    raw["tables"] = {"cn_cpi": [{"month": "202403", "nt_yoy": 0.1}]}
    evidence = _evidence("cn.cpi_yoy", "2024-03-31T01:30:00+00:00")
    evidence["available_at"] = "2024-04-11T01:30:00+00:00"
    capture = {
        "schema_version": "macro-release-evidence-capture.v1",
        "captured_at": "2024-04-01T00:00:00+00:00",
        "records": [evidence],
    }
    result = _normalize(raw, _plan(("cn.cpi_yoy",)), capture)
    assert result.observations == ()
    assert result.quarantine[0]["reason"] == (
        "availability_after_evidence_capture"
    )


def test_independent_plan_blocks_partial_bundle():
    raw = _raw()
    raw["tables"]["cn_ppi"] = []
    result = _normalize(raw=raw)
    assert result.manifest["status"] == "degraded"
    assert result.manifest["missing_scope"] == [
        {"indicator_id": "cn.ppi_yoy", "period_end": "2024-03-31"}
    ]


def test_plan_subset_does_not_hide_unexpected_rows():
    result = _normalize(plan=_plan(("cn.cpi_yoy",)))
    assert result.manifest["status"] == "degraded"
    assert len(result.manifest["unexpected_scope"]) == 6


def test_conflicting_values_without_revision_evidence_are_quarantined():
    raw = _raw()
    raw["tables"]["cn_cpi"].append({"month": "202403", "nt_yoy": 9.9})
    result = _normalize(raw=raw)
    assert result.manifest["status"] == "degraded"
    assert result.manifest["conflicting_scope"] == [
        {"indicator_id": "cn.cpi_yoy", "period_end": "2024-03-31"}
    ]
    assert (
        sum(
            x["reason"] == "conflicting_values_without_revision_evidence"
            for x in result.quarantine
        )
        == 2
    )


def test_normalization_is_offline(monkeypatch):
    monkeypatch.setattr(
        TushareClientPool,
        "query",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("network forbidden")
        ),
    )
    assert len(_normalize().observations) == 7


def test_private_hash_bound_bundle_can_be_cas_published(tmp_path: Path):
    prepared = _prepare(tmp_path)
    for artifact in prepared["artifacts"].values():
        assert os.stat(artifact).st_mode & 0o777 == 0o600
    root = tmp_path / "observations"
    published = _publish(prepared, root)
    rows, pointer = load_observations(root)
    assert published["promoted"] is True
    assert len(rows) == 7
    assert pointer["generation_id"] == "g1"


def test_wrong_pointer_manifest_or_plan_hash_never_advances_pointer(
    tmp_path: Path,
):
    first = _prepare(tmp_path, "n1")
    root = tmp_path / "observations"
    _publish(first, root)
    before = pointer_sha256(root)
    second = _prepare(tmp_path, "n2")
    with pytest.raises(
        MacroObservationStoreError, match="pointer_cas_mismatch"
    ):
        _publish(second, root, "g2", "0" * 64)
    with pytest.raises(
        MacroNormalizationError, match="manifest_hash_mismatch"
    ):
        publish_tushare_normalization(
            second["artifacts"]["manifest"],
            observations_root=root,
            run_id="g2",
            expected_pointer_sha256=before,
            expected_manifest_sha256="0" * 64,
            expected_plan_sha256=second["backfill_plan_sha256"],
        )
    with pytest.raises(MacroNormalizationError, match="plan_hash_mismatch"):
        publish_tushare_normalization(
            second["artifacts"]["manifest"],
            observations_root=root,
            run_id="g2",
            expected_pointer_sha256=before,
            expected_manifest_sha256=second["normalization_manifest_sha256"],
            expected_plan_sha256="0" * 64,
        )
    assert pointer_sha256(root) == before


def test_none_pointer_is_rejected(tmp_path: Path):
    prepared = _prepare(tmp_path)
    with pytest.raises(
        MacroNormalizationError, match="expected_pointer_required"
    ):
        publish_tushare_normalization(
            prepared["artifacts"]["manifest"],
            observations_root=tmp_path / "observations",
            run_id="g1",
            expected_pointer_sha256=None,
            expected_manifest_sha256=prepared["normalization_manifest_sha256"],
            expected_plan_sha256=prepared["backfill_plan_sha256"],
        )


def test_manifest_and_artifact_forgery_are_rejected(tmp_path: Path):
    prepared = _prepare(tmp_path)
    root = tmp_path / "observations"
    quarantine = Path(prepared["artifacts"]["quarantine"])
    quarantine.write_text('{"reason":"forged"}\n')
    manifest_path = Path(prepared["artifacts"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["quarantine.jsonl"] = hashlib.sha256(
        quarantine.read_bytes()
    ).hexdigest()
    manifest["quarantine_count"] = 0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    forged_manifest_hash = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    with pytest.raises(MacroNormalizationError, match="quarantine_not_empty"):
        publish_tushare_normalization(
            manifest_path,
            observations_root=root,
            run_id="g1",
            expected_pointer_sha256="",
            expected_manifest_sha256=forged_manifest_hash,
            expected_plan_sha256=prepared["backfill_plan_sha256"],
        )
    assert not (root / "_latest.json").exists()


def test_evidence_capture_byte_tamper_is_rejected(tmp_path: Path):
    prepared = _prepare(tmp_path)
    evidence = Path(prepared["artifacts"]["evidence_capture"])
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(
        MacroNormalizationError, match="artifact_hash_mismatch"
    ):
        _publish(prepared, tmp_path / "observations")


def test_bundle_with_quarantine_cannot_publish(tmp_path: Path):
    capture = {**_capture(), "records": []}
    prepared = _prepare(tmp_path, "blocked", capture=capture)
    with pytest.raises(
        MacroNormalizationError, match="bundle_not_publishable"
    ):
        _publish(prepared, tmp_path / "observations")
