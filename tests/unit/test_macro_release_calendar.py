from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from quant_investor.macro.contracts import MacroObservation
from quant_investor.macro.official_web_compiler import (
    NBS_QUARTERLY_GDP_PARSER_V2,
    PARSER_CONTRACT_SHA256,
)
from quant_investor.macro.release_calendar import (
    CRITICAL_INDICATOR_IDS,
    CRITICAL_INDICATOR_POLICY,
    CRITICAL_POLICY_SHA256,
    CRITICAL_POLICY_VERSION,
    EMPTY_POINTER_SHA256,
    MACRO_REGISTRY_SHA256,
    MACRO_RELEASE_CALENDAR_CAPTURE_SCHEMA,
    MACRO_RELEASE_CALENDAR_PLAN_SCHEMA,
    MARKET_OPEN_DAYS_SCHEMA,
    ReleaseCalendarCASMismatch,
    ReleaseCalendarEvidence,
    ReleaseCalendarValidationError,
    ReleaseEvent,
    evaluate_critical_event_gap,
    evaluate_release_readiness,
    evaluate_session_lag,
    is_validated_release_calendar_generation,
    load_release_calendar,
    publish_release_calendar,
    release_calendar_pointer_sha256,
    release_calendar_writer_lock,
)
from quant_investor.macro.registry import REGISTRY_VERSION


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _gdp_html() -> bytes:
    title = "2026年二季度和上半年国内生产总值初步核算结果"
    paragraphs = [
        "表2 GDP同比增长速度",
        "单位：%",
        "年份",
        "1季度",
        "2季度",
        "3季度",
        "4季度",
        "2025",
        "5.4",
        "5.2",
        "4.8",
        "4.5",
        "2026",
        "5.0",
        "4.3",
        "注：同比增长速度为与上年同期对比的增长速度。",
    ]
    body = "".join(f"<p>{item}</p>" for item in paragraphs)
    return (
        "<!doctype html><html><head>"
        f'<meta name="ArticleTitle" content="{title}">'
        '<meta name="PubDate" content="2026/07/16 09:30">'
        f"<title>{title} - 国家统计局</title>"
        f"</head><body>{body}</body></html>"
    ).encode("utf-8")


@dataclass
class CalendarFixture:
    plan_path: Path
    capture_path: Path
    raw_root: Path
    open_days_path: Path
    canonical_root: Path
    plan: dict[str, Any]
    capture: dict[str, Any]
    open_days: dict[str, Any]

    def kwargs(self, *, run_id: str, expected_pointer_sha256: str) -> dict[str, Any]:
        return {
            "plan_path": self.plan_path,
            "expected_plan_sha256": _sha(self.plan_path.read_bytes()),
            "capture_manifest_path": self.capture_path,
            "expected_capture_manifest_sha256": _sha(self.capture_path.read_bytes()),
            "raw_root": self.raw_root,
            "market_open_days_path": self.open_days_path,
            "expected_market_open_days_sha256": _sha(
                self.open_days_path.read_bytes()
            ),
            "canonical_root": self.canonical_root,
            "run_id": run_id,
            "expected_pointer_sha256": expected_pointer_sha256,
        }


def _source(
    *,
    source_id: str,
    issuer: str,
    artifact_kind: str,
    url: str,
    captured_at: str,
    raw_path: str,
    raw: bytes,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "issuer": issuer,
        "artifact_kind": artifact_kind,
        "source_url": url,
        "http_status": 200,
        "captured_at": captured_at,
        "raw_path": raw_path,
        "raw_sha256": _sha(raw),
        "size_bytes": len(raw),
        "content_sha256": content_sha256 or _sha(raw),
    }


def _write_fixture(tmp_path: Path) -> CalendarFixture:
    inputs = tmp_path / "inputs"
    raw_root = inputs / "raw"
    raw_root.mkdir(parents=True)
    nbs_url = (
        "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202607/"
        "t20260716_1964142.html"
    )
    pbc_url = (
        "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/"
        "20260717/index.html"
    )
    captured_at = "2026-07-17T10:00:00+08:00"
    nbs_coverage = {
        "schema_version": "macro-release-issuer-coverage.v1",
        "issuer": "nbs_official",
        "through": captured_at,
    }
    pbc_coverage = {
        "schema_version": "macro-release-issuer-coverage.v1",
        "issuer": "pbc_official",
        "through": captured_at,
    }
    notice_raw = _gdp_html()
    bundle_raw = _json_bytes(
        {
            "schema_version": "macro-official-web-normalization.v1",
            "status": "OK",
        }
    )
    parser_hash = PARSER_CONTRACT_SHA256[NBS_QUARTERLY_GDP_PARSER_V2]
    parser_raw = _json_bytes(
        {
            "schema_version": "macro-parser-lineage.v1",
            "parser_id": NBS_QUARTERLY_GDP_PARSER_V2,
            "parser_contract_sha256": parser_hash,
        }
    )
    observation = MacroObservation.from_mapping(
        {
            "indicator_id": "cn.gdp_yoy",
            "dimension_type": "national",
            "period_end": "2026-06-30",
            "release_at": "2026-07-16T09:30:00+08:00",
            "available_at": "2026-07-16T09:30:00+08:00",
            "vintage_id": "official-web.v1:gdp-q2",
            "value": 4.3,
            "unit": "%",
            "frequency": "quarterly",
            "source_system": "nbs_official",
            "source_record_id": "t20260716_1964142",
            "source_url": nbs_url,
            "fetched_at": "2026-07-16T09:31:00+08:00",
            "quality_status": "pass",
        }
    )
    observation_raw = _json_bytes(observation.to_dict())
    raw_payloads = {
        "coverage/nbs.json": _json_bytes(nbs_coverage),
        "coverage/pbc.json": _json_bytes(pbc_coverage),
        "notices/gdp-q2.html": notice_raw,
        "bundles/gdp-q2.json": bundle_raw,
        "parsers/gdp-q2.json": parser_raw,
        "observations/gdp-q2.json": observation_raw,
    }
    for relative, raw in raw_payloads.items():
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    sources = [
        _source(
            source_id="coverage-nbs-1",
            issuer="nbs_official",
            artifact_kind="coverage_receipt",
            url=nbs_url,
            captured_at=captured_at,
            raw_path="coverage/nbs.json",
            raw=raw_payloads["coverage/nbs.json"],
        ),
        _source(
            source_id="coverage-pbc-1",
            issuer="pbc_official",
            artifact_kind="coverage_receipt",
            url=pbc_url,
            captured_at=captured_at,
            raw_path="coverage/pbc.json",
            raw=raw_payloads["coverage/pbc.json"],
        ),
        _source(
            source_id="notice-gdp-q2",
            issuer="nbs_official",
            artifact_kind="release_notice",
            url=nbs_url,
            captured_at="2026-07-16T09:31:00+08:00",
            raw_path="notices/gdp-q2.html",
            raw=notice_raw,
        ),
        _source(
            source_id="bundle-gdp-q2",
            issuer="nbs_official",
            artifact_kind="official_bundle",
            url=nbs_url,
            captured_at="2026-07-16T09:31:00+08:00",
            raw_path="bundles/gdp-q2.json",
            raw=bundle_raw,
        ),
        _source(
            source_id="parser-gdp-q2",
            issuer="nbs_official",
            artifact_kind="parser_contract",
            url=nbs_url,
            captured_at="2026-07-16T09:31:00+08:00",
            raw_path="parsers/gdp-q2.json",
            raw=parser_raw,
            content_sha256=parser_hash,
        ),
        _source(
            source_id="observation-gdp-q2",
            issuer="nbs_official",
            artifact_kind="observation",
            url=nbs_url,
            captured_at="2026-07-16T09:31:00+08:00",
            raw_path="observations/gdp-q2.json",
            raw=observation_raw,
            content_sha256=observation.content_hash,
        ),
    ]
    plan = {
        "schema_version": MACRO_RELEASE_CALENDAR_PLAN_SCHEMA,
        "market": "CN",
        "registry_version": REGISTRY_VERSION,
        "registry_sha256": MACRO_REGISTRY_SHA256,
        "critical_policy_version": CRITICAL_POLICY_VERSION,
        "critical_policy_sha256": CRITICAL_POLICY_SHA256,
        "events": [
            {
                "event_id": "gdp-2026q2-release",
                "event_family": "nbs_quarterly_gdp",
                "issuer": "nbs_official",
                "indicator_ids": ["cn.gdp_yoy"],
                "period": "2026-Q2",
                "scheduled_at": "2026-07-16T09:30:00+08:00",
            }
        ],
    }
    plan_raw = _json_bytes(plan)
    capture = {
        "schema_version": MACRO_RELEASE_CALENDAR_CAPTURE_SCHEMA,
        "market": "CN",
        "plan_sha256": _sha(plan_raw),
        "captured_at": captured_at,
        "issuer_coverage": [
            {
                "issuer": "nbs_official",
                "through": captured_at,
                "source_ids": ["coverage-nbs-1"],
            },
            {
                "issuer": "pbc_official",
                "through": captured_at,
                "source_ids": ["coverage-pbc-1"],
            },
        ],
        "sources": sources,
        "events": [
            {
                "event_id": "gdp-2026q2-release",
                "status": "released",
                "actual_at": "2026-07-16T09:30:00+08:00",
                "rescheduled_at": "",
                "cancelled_at": "",
                "supersedes_event_id": "",
                "source_ids": ["notice-gdp-q2"],
                "resolution_ids": ["resolution-gdp-2026q2"],
            }
        ],
        "resolutions": [
            {
                "resolution_id": "resolution-gdp-2026q2",
                "event_id": "gdp-2026q2-release",
                "indicator_id": "cn.gdp_yoy",
                "period_end": "2026-06-30",
                "frequency": "quarterly",
                "unit": "%",
                "measurement_basis": "current_quarter_real_yoy",
                "value_decimal": "4.3",
                "issuer": "nbs_official",
                "parser_id": NBS_QUARTERLY_GDP_PARSER_V2,
                "parser_contract_sha256": parser_hash,
                "official_bundle_sha256": _sha(bundle_raw),
                "observation_content_hash": observation.content_hash,
                "observation_available_at": "2026-07-16T09:30:00+08:00",
                "source_ids": [
                    "bundle-gdp-q2",
                    "parser-gdp-q2",
                    "observation-gdp-q2",
                ],
            }
        ],
    }
    open_days = {
        "schema_version": MARKET_OPEN_DAYS_SCHEMA,
        "market": "CN",
        "open_dates": [
            "20260709",
            "20260710",
            "20260713",
            "20260715",
            "20260716",
            "20260717",
        ],
    }
    plan_path = inputs / "plan.json"
    capture_path = inputs / "capture.json"
    open_days_path = inputs / "open-days.json"
    plan_path.write_bytes(plan_raw)
    capture_path.write_bytes(_json_bytes(capture))
    open_days_path.write_bytes(_json_bytes(open_days))
    return CalendarFixture(
        plan_path=plan_path,
        capture_path=capture_path,
        raw_root=raw_root,
        open_days_path=open_days_path,
        canonical_root=tmp_path / "canonical",
        plan=plan,
        capture=capture,
        open_days=open_days,
    )


def _publish_initial(tmp_path: Path) -> tuple[CalendarFixture, ReleaseCalendarEvidence]:
    fixture = _write_fixture(tmp_path)
    result = publish_release_calendar(
        **fixture.kwargs(
            run_id="calendar-20260717-a",
            expected_pointer_sha256=EMPTY_POINTER_SHA256,
        )
    )
    assert result.idempotent is False
    return fixture, result.evidence


def _extend_fixture(
    parent: CalendarFixture,
    root: Path,
    *,
    through: str = "2026-07-20T10:00:00+08:00",
) -> CalendarFixture:
    raw_root = root / "raw"
    shutil.copytree(parent.raw_root, raw_root)
    plan = json.loads(json.dumps(parent.plan))
    capture = json.loads(json.dumps(parent.capture))
    open_days = json.loads(json.dumps(parent.open_days))
    capture["captured_at"] = through
    urls = {
        "nbs_official": capture["sources"][0]["source_url"],
        "pbc_official": capture["sources"][1]["source_url"],
    }
    for coverage in capture["issuer_coverage"]:
        issuer = coverage["issuer"]
        suffix = "nbs" if issuer == "nbs_official" else "pbc"
        source_id = f"coverage-{suffix}-2"
        relative = f"coverage/{suffix}-2.json"
        raw = _json_bytes(
            {
                "schema_version": "macro-release-issuer-coverage.v1",
                "issuer": issuer,
                "through": through,
            }
        )
        (raw_root / relative).write_bytes(raw)
        capture["sources"].append(
            _source(
                source_id=source_id,
                issuer=issuer,
                artifact_kind="coverage_receipt",
                url=urls[issuer],
                captured_at=through,
                raw_path=relative,
                raw=raw,
            )
        )
        coverage["through"] = through
        coverage["source_ids"].append(source_id)
    open_days["open_dates"].append("20260720")
    root.mkdir(exist_ok=True)
    plan_path = root / "plan.json"
    capture_path = root / "capture.json"
    open_days_path = root / "open-days.json"
    plan_path.write_bytes(_json_bytes(plan))
    capture_path.write_bytes(_json_bytes(capture))
    open_days_path.write_bytes(_json_bytes(open_days))
    return CalendarFixture(
        plan_path,
        capture_path,
        raw_root,
        open_days_path,
        parent.canonical_root,
        plan,
        capture,
        open_days,
    )


def test_happy_initial_publish_load_private_and_idempotent(tmp_path: Path) -> None:
    fixture, evidence = _publish_initial(tmp_path)
    pointer_sha = release_calendar_pointer_sha256(
        canonical_root=fixture.canonical_root
    )
    loaded = load_release_calendar(
        canonical_root=fixture.canonical_root,
        expected_pointer_sha256=pointer_sha,
    )
    assert loaded == evidence
    assert loaded.identity.pointer_sha256 == pointer_sha
    assert loaded.resolutions[0].value_decimal == "4.3"
    assert len(loaded.validated_ancestry) == 1
    assert loaded.validated_ancestry[-1].generation_id == loaded.identity.generation_id
    for path in Path(loaded.identity.generation_path).rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    repeated = publish_release_calendar(
        **fixture.kwargs(
            run_id=loaded.identity.generation_id,
            expected_pointer_sha256=EMPTY_POINTER_SHA256,
        )
    )
    assert repeated.idempotent is True
    assert repeated.identity == loaded.identity


def test_prefix_extension_and_full_ancestry_matcher(tmp_path: Path) -> None:
    fixture, parent = _publish_initial(tmp_path)
    child_fixture = _extend_fixture(fixture, tmp_path / "child-inputs")
    child_result = publish_release_calendar(
        **child_fixture.kwargs(
            run_id="calendar-20260720-b",
            expected_pointer_sha256=parent.identity.pointer_sha256,
        )
    )
    child = child_result.evidence
    assert [item.generation_id for item in child.validated_ancestry] == [
        parent.identity.generation_id,
        child.identity.generation_id,
    ]
    proof = child.validated_ancestry[0]
    assert is_validated_release_calendar_generation(
        child,
        **proof.__dict__,
    )
    assert not is_validated_release_calendar_generation(
        child,
        **{**proof.__dict__, "semantic_sha256": "0" * 64},
    )


def test_parent_prefix_alteration_and_unrelated_newer_rejected(tmp_path: Path) -> None:
    fixture, parent = _publish_initial(tmp_path)
    child_fixture = _extend_fixture(fixture, tmp_path / "child-inputs")
    child = publish_release_calendar(
        **child_fixture.kwargs(
            run_id="calendar-20260720-b",
            expected_pointer_sha256=parent.identity.pointer_sha256,
        )
    ).evidence
    sibling = _extend_fixture(fixture, tmp_path / "sibling-inputs")
    sibling.open_days["open_dates"][-1] = "20260721"
    sibling.open_days_path.write_bytes(_json_bytes(sibling.open_days))
    with pytest.raises(
        ReleaseCalendarValidationError,
        match="parent_(open_dates|source_artifacts)_prefix_altered",
    ):
        publish_release_calendar(
            **sibling.kwargs(
                run_id="calendar-unrelated",
                expected_pointer_sha256=child.identity.pointer_sha256,
            )
        )


def test_cas_no_clobber_and_writer_lock(tmp_path: Path) -> None:
    fixture, evidence = _publish_initial(tmp_path)
    with pytest.raises(ReleaseCalendarCASMismatch, match="pointer_cas_mismatch"):
        publish_release_calendar(
            **fixture.kwargs(
                run_id="different-run",
                expected_pointer_sha256="0" * 64,
            )
        )
    fixture.capture["captured_at"] = "2026-07-17T10:01:00+08:00"
    fixture.capture_path.write_bytes(_json_bytes(fixture.capture))
    with pytest.raises(ReleaseCalendarValidationError, match="no_clobber"):
        publish_release_calendar(
            **fixture.kwargs(
                run_id=evidence.identity.generation_id,
                expected_pointer_sha256=evidence.identity.pointer_sha256,
            )
        )
    with release_calendar_writer_lock(canonical_root=fixture.canonical_root):
        assert (fixture.canonical_root / ".release-calendar.lock").is_file()


@pytest.mark.parametrize("target", ["plan", "open_days", "raw", "pointer"])
def test_tampered_source_and_pointer_fail_closed(tmp_path: Path, target: str) -> None:
    fixture = _write_fixture(tmp_path)
    if target == "plan":
        fixture.plan_path.write_bytes(fixture.plan_path.read_bytes() + b" ")
        kwargs = fixture.kwargs(
            run_id="calendar-tampered", expected_pointer_sha256=EMPTY_POINTER_SHA256
        )
        kwargs["expected_plan_sha256"] = _sha(_json_bytes(fixture.plan))
        with pytest.raises(ReleaseCalendarValidationError, match="plan_sha256_mismatch"):
            publish_release_calendar(**kwargs)
        return
    if target == "open_days":
        kwargs = fixture.kwargs(
            run_id="calendar-tampered", expected_pointer_sha256=EMPTY_POINTER_SHA256
        )
        fixture.open_days_path.write_bytes(fixture.open_days_path.read_bytes() + b" ")
        with pytest.raises(
            ReleaseCalendarValidationError, match="open_days_sha256_mismatch"
        ):
            publish_release_calendar(**kwargs)
        return
    if target == "raw":
        path = fixture.raw_root / "notices/gdp-q2.html"
        path.write_bytes(path.read_bytes() + b"tamper")
        with pytest.raises(ReleaseCalendarValidationError, match="raw_binding_mismatch"):
            publish_release_calendar(
                **fixture.kwargs(
                    run_id="calendar-tampered",
                    expected_pointer_sha256=EMPTY_POINTER_SHA256,
                )
            )
        return
    result = publish_release_calendar(
        **fixture.kwargs(
            run_id="calendar-published",
            expected_pointer_sha256=EMPTY_POINTER_SHA256,
        )
    )
    pointer = fixture.canonical_root / "_latest.json"
    pointer.write_bytes(pointer.read_bytes() + b" ")
    with pytest.raises(ReleaseCalendarCASMismatch, match="pointer_cas_mismatch"):
        load_release_calendar(
            canonical_root=fixture.canonical_root,
            expected_pointer_sha256=result.identity.pointer_sha256,
        )


def test_policy_drift_unknown_indicator_and_no_self_declared_criticality(
    tmp_path: Path,
) -> None:
    for mutation, match in (
        (("critical_policy_sha256", "0" * 64), "critical_policy_drift"),
        (("registry_sha256", "0" * 64), "registry_policy_drift"),
    ):
        fixture = _write_fixture(tmp_path / mutation[0])
        fixture.plan[mutation[0]] = mutation[1]
        fixture.plan_path.write_bytes(_json_bytes(fixture.plan))
        fixture.capture["plan_sha256"] = _sha(fixture.plan_path.read_bytes())
        fixture.capture_path.write_bytes(_json_bytes(fixture.capture))
        with pytest.raises(ReleaseCalendarValidationError, match=match):
            publish_release_calendar(
                **fixture.kwargs(
                    run_id=f"calendar-{mutation[0]}",
                    expected_pointer_sha256=EMPTY_POINTER_SHA256,
                )
            )
    fixture = _write_fixture(tmp_path / "critical-field")
    fixture.plan["events"][0]["critical"] = False
    fixture.plan_path.write_bytes(_json_bytes(fixture.plan))
    fixture.capture["plan_sha256"] = _sha(fixture.plan_path.read_bytes())
    fixture.capture_path.write_bytes(_json_bytes(fixture.capture))
    with pytest.raises(ReleaseCalendarValidationError, match="event_shape_invalid"):
        publish_release_calendar(
            **fixture.kwargs(
                run_id="calendar-self-critical",
                expected_pointer_sha256=EMPTY_POINTER_SHA256,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("unit", "index", "equivalence_mismatch"),
        ("measurement_basis", "cumulative_yoy", "equivalence_mismatch"),
        ("value_decimal", "4.30", "decimal_invalid"),
        ("issuer", "pbc_official", "observation_issuer_mismatch"),
        ("parser_contract_sha256", "0" * 64, "parser_drift"),
    ],
)
def test_resolution_equivalence_mismatch_fields(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    fixture = _write_fixture(tmp_path)
    fixture.capture["resolutions"][0][field] = value
    fixture.capture_path.write_bytes(_json_bytes(fixture.capture))
    with pytest.raises(ReleaseCalendarValidationError, match=match):
        publish_release_calendar(
            **fixture.kwargs(
                run_id=f"calendar-bad-{field}",
                expected_pointer_sha256=EMPTY_POINTER_SHA256,
            )
        )


def test_symlink_inputs_and_canonical_root_rejected(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "raw-link")
    raw = fixture.raw_root / "notices/gdp-q2.html"
    outside = tmp_path / "outside.html"
    outside.write_bytes(raw.read_bytes())
    raw.unlink()
    raw.symlink_to(outside)
    with pytest.raises(
        ReleaseCalendarValidationError,
        match="symlink|raw_(root|file)_unsafe",
    ):
        publish_release_calendar(
            **fixture.kwargs(
                run_id="calendar-raw-link",
                expected_pointer_sha256=EMPTY_POINTER_SHA256,
            )
        )
    fixture = _write_fixture(tmp_path / "root-link")
    target = tmp_path / "real-canonical"
    target.mkdir()
    fixture.canonical_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReleaseCalendarValidationError, match="root_path_unsafe"):
        publish_release_calendar(
            **fixture.kwargs(
                run_id="calendar-root-link",
                expected_pointer_sha256=EMPTY_POINTER_SHA256,
            )
        )


def test_session_lag_weekend_holiday_and_failures(tmp_path: Path) -> None:
    _fixture, evidence = _publish_initial(tmp_path)
    for macro_date, target, expected in (
        ("2026-07-10", "2026-07-13", 1),
        ("2026-07-10", "2026-07-15", 2),
    ):
        result = evaluate_session_lag(
            evidence,
            macro_logical_date=macro_date,
            target_session_date=target,
            decision_cutoff_at=f"{target}T10:00:00+08:00",
        )
        assert result.ready and result.session_lag == expected
    future = evaluate_session_lag(
        evidence,
        macro_logical_date="2026-07-17",
        target_session_date="2026-07-16",
        decision_cutoff_at="2026-07-16T10:00:00+08:00",
    )
    assert "macro_release_macro_logical_date_in_future" in future.blockers
    above_two = evaluate_session_lag(
        evidence,
        macro_logical_date="2026-07-09",
        target_session_date="2026-07-15",
        decision_cutoff_at="2026-07-15T10:00:00+08:00",
    )
    assert "macro_release_session_lag_above_two" in above_two.blockers
    missing = evaluate_session_lag(
        evidence,
        macro_logical_date="2026-07-14",
        target_session_date="2026-07-20",
        decision_cutoff_at="2026-07-20T10:00:00+08:00",
    )
    assert "macro_release_macro_logical_date_missing_from_calendar" in missing.blockers
    assert "macro_release_target_session_missing_from_calendar" in missing.blockers


def _scheduled_evidence(
    evidence: ReleaseCalendarEvidence,
    *,
    event_id: str,
    scheduled_at: str,
    family: str = "nbs_quarterly_gdp",
    issuer: str = "nbs_official",
    indicator_ids: tuple[str, ...] = ("cn.gdp_yoy",),
    coverage_through: str = "2026-07-20T10:00:00+08:00",
) -> ReleaseCalendarEvidence:
    kind = "date" if "T" not in scheduled_at else "timestamp"
    event = ReleaseEvent(
        event_id=event_id,
        event_family=family,
        issuer=issuer,
        indicator_ids=indicator_ids,
        period="2026-Q2" if family == "nbs_quarterly_gdp" else "2026-06",
        schedule_kind=kind,
        scheduled_at=scheduled_at,
        status="scheduled",
        actual_at="",
        rescheduled_at="",
        reschedule_kind="",
        cancelled_at="",
        supersedes_event_id="",
        source_ids=("notice-gdp-q2",),
        resolution_ids=(),
    )
    coverage = tuple(
        replace(item, through_at=coverage_through) for item in evidence.issuer_coverage
    )
    return replace(evidence, events=(event,), resolutions=(), issuer_coverage=coverage)


@pytest.mark.parametrize(
    ("scheduled_at", "expected_relevant"),
    [
        ("2026-07-16T15:00:00+08:00", False),
        ("2026-07-16T15:00:09+08:00", True),
        ("2026-07-17T10:00:00+08:00", True),
        ("2026-07-17T10:00:01+08:00", False),
        ("2026-07-16", True),
    ],
)
def test_event_window_boundaries_and_date_only(
    tmp_path: Path, scheduled_at: str, expected_relevant: bool
) -> None:
    _fixture, evidence = _publish_initial(tmp_path)
    synthetic = _scheduled_evidence(
        evidence, event_id="boundary", scheduled_at=scheduled_at
    )
    result = evaluate_critical_event_gap(
        synthetic,
        macro_logical_date="2026-07-16",
        decision_cutoff_at="2026-07-17T10:00:00+08:00",
    )
    assert bool(result.relevant_event_ids) is expected_relevant
    assert (not result.ready) is expected_relevant


def test_weekend_event_and_each_pbc_gdp_family_are_critical(tmp_path: Path) -> None:
    _fixture, evidence = _publish_initial(tmp_path)
    weekend = _scheduled_evidence(
        evidence,
        event_id="weekend-gdp",
        scheduled_at="2026-07-18T12:00:00+08:00",
    )
    result = evaluate_critical_event_gap(
        weekend,
        macro_logical_date="2026-07-17",
        decision_cutoff_at="2026-07-20T10:00:00+08:00",
    )
    assert result.blocking_event_ids == ("weekend-gdp",)
    pbc = _scheduled_evidence(
        evidence,
        event_id="pbc-key",
        scheduled_at="2026-07-16T15:00:09+08:00",
        family="pbc_money_stock",
        issuer="pbc_official",
        indicator_ids=("cn.m1_yoy", "cn.m2_yoy"),
    )
    assert evaluate_critical_event_gap(
        pbc,
        macro_logical_date="2026-07-16",
        decision_cutoff_at="2026-07-17T10:00:00+08:00",
    ).blocking_event_ids == ("pbc-key",)


def test_resolved_key_event_blocks_lag_and_post_close_exact_date(
    tmp_path: Path,
) -> None:
    _fixture, evidence = _publish_initial(tmp_path)
    lagged = evaluate_release_readiness(
        evidence,
        macro_logical_date="2026-07-15",
        target_session_date="2026-07-16",
        decision_cutoff_at="2026-07-16T10:00:00+08:00",
    )
    assert not lagged.ready
    assert (
        "macro_release_critical_event_in_gap:gdp-2026q2-release"
        in lagged.blockers
    )
    event = evidence.events[0]
    resolution = evidence.resolutions[0]
    post_close_capture = "2026-07-16T15:35:00+08:00"
    post_close_source_ids = {
        *event.source_ids,
        *resolution.source_ids,
    }
    post_close = replace(
        evidence,
        source_artifacts=tuple(
            replace(item, captured_at=post_close_capture)
            if item.source_id in post_close_source_ids
            else item
            for item in evidence.source_artifacts
        ),
        events=(
            replace(
                event,
                scheduled_at="2026-07-16T15:30:00+08:00",
                actual_at="2026-07-16T15:30:00+08:00",
            ),
        ),
        resolutions=(
            replace(
                resolution,
                observation_available_at=post_close_capture,
            ),
        ),
    )
    exact = evaluate_release_readiness(
        post_close,
        macro_logical_date="2026-07-16",
        target_session_date="2026-07-16",
        decision_cutoff_at="2026-07-16T16:00:00+08:00",
    )
    assert not exact.ready
    assert exact.session_lag.session_lag == 0
    assert (
        "macro_release_critical_event_in_gap:gdp-2026q2-release"
        in exact.blockers
    )
    assert exact.critical_event_gap.blocking_event_ids == (
        "gdp-2026q2-release",
    )
    assert exact.critical_event_gap.resolved_event_ids == ()


def test_missing_and_stale_issuer_coverage_block(tmp_path: Path) -> None:
    _fixture, evidence = _publish_initial(tmp_path)
    missing = replace(
        evidence,
        issuer_coverage=tuple(
            item for item in evidence.issuer_coverage if item.issuer != "pbc_official"
        ),
    )
    result = evaluate_critical_event_gap(
        missing,
        macro_logical_date="2026-07-16",
        decision_cutoff_at="2026-07-17T10:00:00+08:00",
    )
    assert "macro_release_issuer_coverage_missing:pbc_official" in result.blockers
    stale = replace(
        evidence,
        issuer_coverage=tuple(
            replace(item, through_at="2026-07-17T09:59:59+08:00")
            for item in evidence.issuer_coverage
        ),
    )
    result = evaluate_critical_event_gap(
        stale,
        macro_logical_date="2026-07-16",
        decision_cutoff_at="2026-07-17T10:00:00+08:00",
    )
    assert "macro_release_issuer_coverage_stale:nbs_official" in result.blockers
    assert "macro_release_issuer_coverage_stale:pbc_official" in result.blockers


def test_all_twelve_policy_coverage_and_pbc_alias() -> None:
    assert len(CRITICAL_INDICATOR_POLICY) == 12
    assert set(CRITICAL_INDICATOR_IDS) == {
        "cn.cpi_yoy",
        "cn.exports_yoy",
        "cn.fixed_asset_investment_yoy",
        "cn.gdp_yoy",
        "cn.imports_yoy",
        "cn.industrial_value_added_yoy",
        "cn.m1_yoy",
        "cn.m2_yoy",
        "cn.pmi_manufacturing",
        "cn.ppi_yoy",
        "cn.property_investment_yoy",
        "cn.retail_sales_yoy",
    }
    money = [
        item
        for item in CRITICAL_INDICATOR_POLICY
        if item.indicator_id in {"cn.m1_yoy", "cn.m2_yoy"}
    ]
    assert all(item.evidence_issuer == "pbc_official" for item in money)
    assert all(item.observation_issuer == "pboc_official" for item in money)
    assert all(item.allowed_parsers[0][0].endswith(".v2") for item in money)
