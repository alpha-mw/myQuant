from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.v16.evidence_v2.contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    encode_f64,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.schedule import (
    ATTEMPT_GENESIS_V3_SCHEMA,
    CALIBRATION_UNIVERSE_SCHEMA,
    SCHEDULE_DECLARATION_V3_SCHEMA,
    ScheduleAnchorBinding,
    ScheduleAnchorBindingV3,
    TRANSITION_GRAPH_SCHEMA,
    build_attempt_genesis,
    build_schedule_declaration,
    build_schedule_declaration_v3,
    validate_attempt_genesis,
    validate_bound_lineage,
    validate_schedule_anchor_binding,
    validate_schedule_declaration,
    validate_transition_graph,
)
from quant_investor.v16.evidence_v2.target import (
    COST_COMPONENT_ORDER,
    EXPECTED_INDEX_SCHEMA,
    EXPECTED_STOCK_MARK_SCHEMA,
    INDEX_TABLE_SCHEMA,
    STOCK_MARK_TABLE_SCHEMA,
    CostVector,
    MarkCandidate,
    MarkTargetEvidenceBundle,
    MarkTargetEvidenceBundleV3,
    StockMarkSourceBundle,
    build_adjustment_factor_evidence,
    build_cost_evidence,
    build_h00300_manifest,
    build_mark_target_outcome_from_evidence,
    build_pit_membership_evidence,
    build_stock_mark_evidence_from_sources,
    build_suspension_evidence,
    build_terminal_settlement,
    prepare_stock_mark_sources,
    prepare_mark_target_common_evidence_v3,
    resolve_mark,
    validate_h00300_parquet,
    validate_mark_target_outcome_from_evidence,
    validate_stock_mark_parquet,
    validate_terminal_settlement,
)
from quant_investor.v16.evidence_v2.calendar import (
    OPEN_SESSIONS,
    bind_calendar_artifact,
    build_declared_open_session_calendar,
)
from quant_investor.v16.evidence_v2.calendar_recheck import CALENDAR_RECHECK_SCHEMA
from quant_investor.v16.evidence_v2.runtime_identity import (
    MODEL_BUNDLE_SCHEMA,
    PINNED_OPENSSL_PATH,
    RUNTIME_CAPSULE_SCHEMA,
)
from quant_investor.v16.evidence_v2.session_clock import (
    bind_session_clock_artifact,
    build_declared_session_clock,
)
from quant_investor.v16.evidence_v2.timestamp import (
    AnchorWindow,
    BoundArtifact,
    RevocationBinding,
    TIMESTAMP_RECEIPT_SCHEMA,
    TimestampAnchorBinding,
    TimestampVerificationBundle,
    build_timestamp_attempt,
    record_persisted_response,
    validate_timestamp_receipt,
    verify_and_record_timestamp_validation,
)


def _ref(
    name: str,
    *,
    schema: str = "fixture.v1",
    absolute_path: str | None = None,
    payload: bytes | None = None,
) -> EvidenceRef:
    raw = payload if payload is not None else (name + ":bytes").encode()
    byte_sha = hashlib.sha256(raw).hexdigest()
    semantic_sha = hashlib.sha256((name + ":semantic").encode()).hexdigest()
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=absolute_path or f"/private/evidence/{name}.json",
        byte_sha256=byte_sha,
        semantic_sha256=semantic_sha,
        root_policy="v16.private-evidence-root.v2",
    )


def _bound(name: str, payload: dict[str, object]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=str(payload["schema_version"]),
            absolute_path=f"/private/evidence/{name}",
            byte_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_sha256=semantic_sha256(payload),
            root_policy="v16.private-evidence-root.v2",
        ),
        payload=raw,
    )


def _artifact(name: str, payload: bytes) -> BoundArtifact:
    return BoundArtifact(reference=_ref(name, payload=payload), payload=payload)


def _synthetic_timestamp_receipt(
    *,
    bundle: TimestampVerificationBundle,
    anchor_window: AnchorWindow,
    expected_policy_oid: str,
    openssl_binary_sha256: str,
    verification_time: str,
    openssl_path: str = PINNED_OPENSSL_PATH,
    **_unused: object,
) -> dict[str, Any]:
    upper = datetime.fromisoformat(anchor_window.not_after.replace("Z", "+00:00"))
    lower = (
        None
        if anchor_window.not_before is None
        else datetime.fromisoformat(anchor_window.not_before.replace("Z", "+00:00"))
    )
    gen_time = upper - timedelta(seconds=1) if lower is None else lower + (upper - lower) / 2
    anchor_window.validate_gen_time(gen_time)
    return validate_timestamp_receipt(
        seal_semantic(
            {
                "schema_version": TIMESTAMP_RECEIPT_SCHEMA,
                "anchored_artifact_ref": bundle.anchored_artifact.reference.to_dict(),
                "request_ref": bundle.query.reference.to_dict(),
                "response_ref": bundle.response.reference.to_dict(),
                "trust_anchor_ref": bundle.trust_anchor.reference.to_dict(),
                "untrusted_chain_ref": bundle.untrusted_chain.reference.to_dict(),
                "revocation_refs": [
                    {
                        "certificate_ref": binding.certificate.reference.to_dict(),
                        "issuer_certificate_ref": binding.issuer_certificate.reference.to_dict(),
                        "crl_ref": binding.crl.reference.to_dict(),
                    }
                    for binding in bundle.revocations
                ],
                "policy_oid": expected_policy_oid,
                "gen_time": gen_time.isoformat().replace("+00:00", "Z"),
                "verified_at": verification_time,
                "anchor_kind": anchor_window.anchor_kind,
                "anchor_not_before": anchor_window.not_before,
                "anchor_not_after": anchor_window.not_after,
                "openssl_path": openssl_path,
                "openssl_binary_sha256": openssl_binary_sha256,
                "response_projection_sha256": hashlib.sha256(
                    b"synthetic-projection:" + bundle.response.payload
                ).hexdigest(),
                "verification_stdout_sha256": hashlib.sha256(
                    b"synthetic-query-verify:" + bundle.query.payload
                ).hexdigest(),
                "data_verification_stdout_sha256": hashlib.sha256(
                    b"synthetic-data-verify:" + bundle.anchored_artifact.payload
                ).hexdigest(),
                "warnings": [],
                "cryptographically_valid_at_gen_time": True,
                "activation_candidate": False,
                "new_risk_authorized": False,
                "production_apply_enabled": False,
            }
        )
    )


@pytest.fixture(scope="module", autouse=True)
def _route_synthetic_timestamp_crypto() -> object:
    import quant_investor.v16.evidence_v2.timestamp as timestamp_module

    original = timestamp_module.verify_rfc3161_bundle

    def routed(**kwargs: Any) -> dict[str, Any]:
        if (
            kwargs["expected_policy_oid"] == "1.2.3.4"
            and kwargs["openssl_binary_sha256"] == "a" * 64
        ):
            return _synthetic_timestamp_receipt(**kwargs)
        return original(**kwargs)

    patcher = pytest.MonkeyPatch()
    patcher.setattr(timestamp_module, "verify_rfc3161_bundle", routed)
    yield
    patcher.undo()


def _timestamp_binding(
    artifact: BoundCanonicalArtifact,
    *,
    anchor_id: str,
    window: AnchorWindow,
) -> TimestampAnchorBinding:
    certificate_payload = b"-----BEGIN CERTIFICATE-----\nTSA\n-----END CERTIFICATE-----\n"
    issuer_payload = b"-----BEGIN CERTIFICATE-----\nROOT\n-----END CERTIFICATE-----\n"
    crl_payload = b"-----BEGIN X509 CRL-----\nCRL\n-----END X509 CRL-----\n"
    certificate = _artifact(
        f"{anchor_id}-tsa.pem",
        certificate_payload,
    )
    issuer = _artifact(
        f"{anchor_id}-root.pem",
        issuer_payload,
    )
    crl = _artifact(
        f"{anchor_id}-crl.pem",
        crl_payload,
    )
    response_directory = "/private/evidence/timestamp-responses"
    query_payload = f"query:{anchor_id}".encode()
    response_payload = f"response:{anchor_id}".encode()
    query = BoundArtifact(
        reference=_ref(f"{anchor_id}-request.tsq", payload=query_payload),
        payload=query_payload,
    )
    response = BoundArtifact(
        reference=_ref(
            f"{anchor_id}-response.tsr",
            absolute_path=f"{response_directory}/{anchor_id}.tsr",
            payload=response_payload,
        ),
        payload=response_payload,
    )
    revocations = (
        RevocationBinding(
            certificate=certificate,
            issuer_certificate=issuer,
            crl=crl,
        ),
    )
    bundle = TimestampVerificationBundle(
        anchored_artifact=BoundArtifact(reference=artifact.reference, payload=artifact.payload),
        query=query,
        response=response,
        trust_anchor=issuer,
        untrusted_chain=_artifact(f"{anchor_id}-chain.pem", certificate_payload),
        revocations=revocations,
    )
    attempt = build_timestamp_attempt(
        protocol_attempt_id="attempt-v16-001",
        anchor_id=anchor_id,
        anchored_artifact_ref=artifact.reference,
        request_ref=query.reference,
        response_directory=response_directory,
        anchor_window=window,
        expected_policy_oid="1.2.3.4",
        openssl_binary_sha256="a" * 64,
        trust_anchor_ref=bundle.trust_anchor.reference,
        untrusted_chain_ref=bundle.untrusted_chain.reference,
        revocations=revocations,
    )
    attempt = record_persisted_response(attempt, response_ref=response.reference)
    upper = datetime.fromisoformat(window.not_after.replace("Z", "+00:00"))
    lower = (
        None
        if window.not_before is None
        else datetime.fromisoformat(window.not_before.replace("Z", "+00:00"))
    )
    gen_time = upper - timedelta(seconds=1) if lower is None else lower + (upper - lower) / 2
    gen_time_text = gen_time.isoformat().replace("+00:00", "Z")
    validation = verify_and_record_timestamp_validation(
        attempt,
        bundle=bundle,
        verification_time=gen_time_text,
        validation_receipt_path=f"/private/evidence/{anchor_id}-receipt.json",
    )
    return TimestampAnchorBinding(
        attempt=_bound(f"{anchor_id}-attempt.json", validation.attempt),
        validation_receipt=validation.validation_receipt,
        verification_bundle=bundle,
    )


def _schedule_anchor(
    name: str,
    payload: dict[str, object],
) -> ScheduleAnchorBinding:
    schedule = _bound(name, payload)
    first_s0_open = min(str(slot["s0_open_at"]) for slot in payload["slots"])
    timestamp = _timestamp_binding(
        schedule,
        anchor_id=name.removesuffix(".json"),
        window=AnchorWindow(
            anchor_kind="schedule_declaration",
            not_before=None,
            not_after=first_s0_open,
        ),
    )
    return ScheduleAnchorBinding(schedule=schedule, timestamp=timestamp)


def _slot(epoch: str, *, start: date) -> dict[str, object]:
    window = 30 if epoch == "B" else 20
    s1 = start + timedelta(days=1)
    return {
        "slot_id": f"{epoch.lower()}-slot-1",
        "s0_date": start.isoformat(),
        "s0_open_at": f"{start.isoformat()}T01:30:00Z",
        "s0_close_at": f"{start.isoformat()}T07:00:00Z",
        "decision_cutoff_at": f"{start.isoformat()}T07:30:00Z",
        "s1_open_at": f"{s1.isoformat()}T01:30:00Z",
        "target_sessions": [(s1 + timedelta(days=index)).isoformat() for index in range(window)],
    }


def test_attempt_and_ordered_abc_schedule_lineage_stay_nonauthorizing() -> None:
    runtime = _ref("runtime", schema="v16.hermetic-runtime-capsule.v2")
    calendar = _ref("open-session-calendar")
    models = {
        branch: _ref(f"model-{branch}", schema="v16.frozen-model-bundle.v2")
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    genesis = build_attempt_genesis(
        protocol_attempt_id="attempt-v16-001",
        runtime_capsule=runtime,
        proposed_factor_graph=_ref("factor-graph"),
        open_session_calendar=calendar,
    )
    schedule_payloads = [
        build_schedule_declaration(
            protocol_attempt_id="attempt-v16-001",
            epoch=epoch,
            schedule_id=f"schedule-{epoch.lower()}",
            seed_hex=hashlib.sha256(f"seed-{epoch}".encode()).hexdigest(),
            runtime_capsule=runtime,
            open_session_calendar=calendar,
            model_bundle_refs=None if epoch == "A" else models,
            calibration_universe_ref=(
                None
                if epoch == "A"
                else _ref(
                    f"universe-{epoch.lower()}",
                    schema="v16.calibration-universe-plan.v2",
                )
            ),
            slots=[
                _slot(
                    epoch,
                    start=date(2026, 1, 2) + timedelta(days=index * 60),
                )
            ],
        )
        for index, epoch in enumerate(("A", "B", "C"))
    ]
    schedule_anchors = [
        _schedule_anchor(f"schedule-{epoch.lower()}.json", payload)
        for epoch, payload in zip(("A", "B", "C"), schedule_payloads)
    ]

    assert validate_attempt_genesis(genesis)["max_attempts_v16"] == 1
    assert all(
        validate_schedule_declaration(item)["activation_candidate"] is False
        for item in schedule_payloads
    )
    projection = validate_bound_lineage(
        genesis=genesis,
        schedule_anchors=schedule_anchors,
    )
    assert projection["readiness_status"] == "no_new_risk"
    assert "global_attempt_registry_authority_not_integrated" in projection["blockers"]


def test_schedule_rejects_epoch_gap_and_prediction_model_drift() -> None:
    runtime = _ref("runtime", schema="v16.hermetic-runtime-capsule.v2")
    calendar = _ref("open-session-calendar")
    models = {
        branch: _ref(f"model-{branch}", schema="v16.frozen-model-bundle.v2")
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    genesis = build_attempt_genesis(
        protocol_attempt_id="attempt-v16-001",
        runtime_capsule=runtime,
        proposed_factor_graph=_ref("factor-graph"),
        open_session_calendar=calendar,
    )
    schedule_b = build_schedule_declaration(
        protocol_attempt_id="attempt-v16-001",
        epoch="B",
        schedule_id="schedule-b",
        seed_hex="ab" * 32,
        runtime_capsule=runtime,
        open_session_calendar=calendar,
        model_bundle_refs=models,
        calibration_universe_ref=_ref(
            "universe-b",
            schema="v16.calibration-universe-plan.v2",
        ),
        slots=[_slot("B", start=date(2026, 4, 1))],
    )
    with pytest.raises(EvidenceV2Error, match="ordered A/B/C"):
        validate_bound_lineage(
            genesis=genesis,
            schedule_anchors=[_schedule_anchor("schedule-b.json", schedule_b)],
        )

    schedule_b["model_bundle_refs"] = None
    schedule_b = seal_semantic(
        {key: value for key, value in schedule_b.items() if key != "semantic_sha256"}
    )
    with pytest.raises(EvidenceV2Error, match="exactly four model bundles"):
        validate_schedule_declaration(schedule_b)


def test_schedule_rejects_an_anchor_not_bound_to_its_pre_s0_deadline() -> None:
    models = {
        branch: _ref(f"late-model-{branch}", schema="v16.frozen-model-bundle.v2")
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    payload = build_schedule_declaration(
        protocol_attempt_id="attempt-v16-001",
        epoch="C",
        schedule_id="late-schedule-c",
        seed_hex="ef" * 32,
        runtime_capsule=_ref("late-runtime"),
        open_session_calendar=_ref("late-calendar"),
        model_bundle_refs=models,
        calibration_universe_ref=_ref(
            "late-universe-c",
            schema="v16.calibration-universe-plan.v2",
        ),
        slots=[_slot("C", start=date(2026, 8, 1))],
    )
    schedule = _bound("late-schedule-c.json", payload)
    late_timestamp = _timestamp_binding(
        schedule,
        anchor_id="late-schedule-c",
        window=AnchorWindow(
            anchor_kind="schedule_declaration",
            not_before=None,
            not_after=str(payload["slots"][0]["s1_open_at"]),
        ),
    )

    with pytest.raises(EvidenceV2Error, match="pre-s0 anchor lineage"):
        validate_schedule_anchor_binding(
            ScheduleAnchorBinding(schedule=schedule, timestamp=late_timestamp)
        )


def test_transition_graph_preserves_exact_abcd_factor_set_references() -> None:
    graph = seal_semantic(
        {
            "schema_version": TRANSITION_GRAPH_SCHEMA,
            "protocol_attempt_id": "attempt-v16-001",
            "transitions": [
                {
                    "transition_id": "factor-add-001",
                    "mode": "add",
                    "incumbent": None,
                    "challenger": "quality-growth-v4",
                    "arm_factor_sets": {
                        arm: _ref(f"arm-{arm}").to_dict() for arm in ("A", "B", "C", "D")
                    },
                }
            ],
        }
    )
    assert validate_transition_graph(graph)["transitions"][0]["mode"] == "add"


def _costs() -> CostVector:
    return CostVector.from_rows(
        [{"name": name, "value": "f64:0x1.0c6f7a0b5ed8dp-14"} for name in COST_COMPONENT_ORDER]
    )


def test_terminal_settlement_precedes_exact_and_suspension_marks() -> None:
    terminal = build_terminal_settlement(
        symbol="000001.SZ",
        raw_cash_per_terminal_share=2.0,
        settlement_effective_date="2026-03-20",
        applicable_adj_factor=1.5,
        adj_factor_effective_date="2026-03-19",
        official_event_ref=_ref("terminal-event"),
        adj_factor_ref=_ref("adj-factor"),
    )
    validated = validate_terminal_settlement(terminal)
    mark, source = resolve_mark(
        symbol="000001.SZ",
        boundary_date="2026-03-20",
        candidate=MarkCandidate(
            exact_mark=9.0,
            pit_listed=False,
            authoritative_suspension=True,
            stale_mark=8.0,
            terminal_settlement=validated,
        ),
        phase="exit",
    )

    assert mark == 3.0
    assert source == "terminal_cash_settlement"


def test_mark_target_is_recomputed_from_bound_stock_cost_schedule_and_h00300() -> None:
    runtime = _ref("runtime", schema="v16.hermetic-runtime-capsule.v2")
    calendar = _ref("open-session-calendar")
    models = {
        branch: _ref(f"model-{branch}", schema="v16.frozen-model-bundle.v2")
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    schedule_payload = build_schedule_declaration(
        protocol_attempt_id="attempt-v16-001",
        epoch="C",
        schedule_id="schedule-c",
        seed_hex="cd" * 32,
        runtime_capsule=runtime,
        open_session_calendar=calendar,
        model_bundle_refs=models,
        calibration_universe_ref=_ref(
            "universe-c",
            schema="v16.calibration-universe-plan.v2",
        ),
        slots=[_slot("C", start=date(2026, 3, 1))],
    )
    schedule_anchor = _schedule_anchor("schedule-c.json", schedule_payload)
    schedule = schedule_anchor.schedule
    costs_payload = build_cost_evidence(
        protocol_attempt_id="attempt-v16-001",
        sample_id="sample-001",
        costs=_costs(),
        cost_model_ref=_ref("cost-model"),
    )
    costs = _bound("costs-sample-001.json", costs_payload)
    sessions = schedule_payload["slots"][0]["target_sessions"]
    stock_parquet = _stock_parquet(
        rows=[
            ("000001.SZ", date.fromisoformat(sessions[0]), 10.0),
            ("000001.SZ", date.fromisoformat(sessions[-1]), 11.0),
        ]
    )
    stock_projection = validate_stock_mark_parquet(stock_parquet)
    stock_table_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=STOCK_MARK_TABLE_SCHEMA,
        absolute_path="/governed/cn-stock-marks.parquet",
        byte_sha256=hashlib.sha256(stock_parquet).hexdigest(),
        semantic_sha256=stock_projection["parquet_metadata_semantic_sha256"],
        root_policy="v16.governed-data-root.v2",
    )
    stock_rows = [
        {"symbol": "000001.SZ", "trade_date": item, "adj_factor": 1.0}
        for item in (sessions[0], sessions[-1])
    ]
    pit_rows = [
        {"symbol": "000001.SZ", "trade_date": item, "pit_listed": True}
        for item in (sessions[0], sessions[-1])
    ]
    suspension_rows = [
        {
            "symbol": "000001.SZ",
            "trade_date": item,
            "authoritative_suspension": False,
            "stale_trade_date": None,
        }
        for item in (sessions[0], sessions[-1])
    ]
    stock_sources = StockMarkSourceBundle(
        market_parquet=BoundRawArtifact(stock_table_ref, stock_parquet),
        adjustment_factors=_bound(
            "stock-adjustment.json",
            build_adjustment_factor_evidence(
                generation_id="stock-adjustment-001",
                market_table_ref=stock_table_ref,
                rows=stock_rows,
            ),
        ),
        pit_membership=_bound(
            "stock-membership.json",
            build_pit_membership_evidence(
                generation_id="stock-membership-001",
                calendar_ref=calendar,
                rows=pit_rows,
            ),
        ),
        suspensions=_bound(
            "stock-suspension.json",
            build_suspension_evidence(
                generation_id="stock-suspension-001",
                calendar_ref=calendar,
                rows=suspension_rows,
            ),
        ),
    )
    stock_payload = build_stock_mark_evidence_from_sources(
        sources=prepare_stock_mark_sources(stock_sources),
        protocol_attempt_id="attempt-v16-001",
        sample_id="sample-001",
        symbol="000001.SZ",
        slot_id="c-slot-1",
        schedule_ref=schedule.reference,
        entry_date=sessions[0],
        exit_date=sessions[-1],
    )
    stock = _bound("stock-sample-001.json", stock_payload)
    benchmark_payload = _h00300_parquet(dates=[date.fromisoformat(item) for item in sessions])
    table_projection = validate_h00300_parquet(benchmark_payload)
    table_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=INDEX_TABLE_SCHEMA,
        absolute_path="/governed/h00300.parquet",
        byte_sha256=hashlib.sha256(benchmark_payload).hexdigest(),
        semantic_sha256=table_projection["parquet_metadata_semantic_sha256"],
        root_policy="v16.governed-data-root.v2",
    )
    manifest_payload = build_h00300_manifest(
        generation_id="h00300-202603",
        created_at="2026-03-22T00:00:00Z",
        table_ref=table_ref,
        parquet_payload=benchmark_payload,
        official_source_receipt=_ref("h00300-official-source"),
        calendar_ref=calendar,
    )
    manifest = _bound("h00300-manifest.json", manifest_payload)
    bundle = MarkTargetEvidenceBundle(
        schedule_anchor=schedule_anchor,
        stock_marks=stock,
        stock_sources=stock_sources,
        costs=costs,
        benchmark_manifest=manifest,
        benchmark_parquet=benchmark_payload,
    )
    outcome = build_mark_target_outcome_from_evidence(bundle)

    assert outcome["target_id"] == "CN_20D_MARK_NET_TOTAL_RETURN_EXCESS_VS_CSI300_TRI_V1"
    assert outcome["non_executable_research_target"] is True
    assert outcome["new_risk_authorized"] is False
    assert validate_mark_target_outcome_from_evidence(outcome, bundle=bundle) == outcome

    tampered = {key: value for key, value in outcome.items() if key != "semantic_sha256"}
    tampered["h00300_s20_close"] = encode_f64(1.0)
    with pytest.raises(EvidenceV2Error, match="recomputation mismatch"):
        validate_mark_target_outcome_from_evidence(seal_semantic(tampered), bundle=bundle)


def _h00300_parquet(
    *,
    duplicate_date: bool = False,
    dates: list[date] | None = None,
) -> bytes:
    dates = list(dates or [date(2026, 3, 2), date(2026, 3, 3)])
    if duplicate_date:
        dates[1] = dates[0]
    observed = [
        datetime.combine(item, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=10)
        for item in dates
    ]
    source_sha = hashlib.sha256(b"official-source").hexdigest()
    table = pa.Table.from_arrays(
        [
            pa.array(["H00300.CSI"] * len(dates), type=pa.string()),
            pa.array(dates, type=pa.date32()),
            pa.array(
                [1000.0 + 5.0 * index for index in range(len(dates))],
                type=pa.float64(),
            ),
            pa.array(["CNY"] * len(dates), type=pa.string()),
            pa.array(
                ["gross_pre_tax_total_return"] * len(dates),
                type=pa.string(),
            ),
            pa.array(["csindex_official"] * len(dates), type=pa.string()),
            pa.array(observed, type=pa.timestamp("us", tz="UTC")),
            pa.array([source_sha] * len(dates), type=pa.string()),
        ],
        schema=EXPECTED_INDEX_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=1, write_statistics=True)
    return sink.getvalue().to_pybytes()


def _stock_parquet(*, rows: list[tuple[str, date, float]]) -> bytes:
    ordered = sorted(rows, key=lambda row: (row[0], row[1]))
    source_sha = hashlib.sha256(b"official-stock-source").hexdigest()
    observed = [
        datetime.combine(trade_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=10)
        for _, trade_date, _ in ordered
    ]
    table = pa.Table.from_arrays(
        [
            pa.array([symbol for symbol, _, _ in ordered], type=pa.string()),
            pa.array([trade_date for _, trade_date, _ in ordered], type=pa.date32()),
            pa.array([close for _, _, close in ordered], type=pa.float64()),
            pa.array(observed, type=pa.timestamp("us", tz="UTC")),
            pa.array([source_sha] * len(ordered), type=pa.string()),
        ],
        schema=EXPECTED_STOCK_MARK_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=1, write_statistics=True)
    return sink.getvalue().to_pybytes()


def test_h00300_parquet_binds_schema_rows_and_metadata_projection() -> None:
    validated = validate_h00300_parquet(_h00300_parquet())

    assert validated["row_count"] == 2
    assert validated["min_trade_date"] == "2026-03-02"
    assert validated["max_trade_date"] == "2026-03-03"
    assert len(validated["parquet_metadata_semantic_sha256"]) == 64
    assert validated["rows"][0]["instrument_id"] == "H00300.CSI"


def test_h00300_parquet_rejects_duplicate_trade_dates() -> None:
    with pytest.raises(EvidenceV2Error, match="strictly increasing"):
        validate_h00300_parquet(_h00300_parquet(duplicate_date=True))


def _valid_v3_schedule_for_target() -> tuple[dict[str, object], EvidenceRef]:
    source_root = "/private/v16/calendar-sources"
    calendar = build_declared_open_session_calendar(source_root)
    clock = build_declared_session_clock(source_root)
    calendar_ref = bind_calendar_artifact(
        calendar,
        absolute_path="/private/evidence/calendar-v3.json",
    ).reference
    clock_ref = bind_session_clock_artifact(
        clock,
        absolute_path="/private/evidence/clock-v3.json",
    ).reference
    s0_date = "2026-07-06"
    s0_index = OPEN_SESSIONS.index(s0_date)
    target_sessions = list(OPEN_SESSIONS[s0_index + 1 : s0_index + 21])
    models = {
        branch: _ref(f"model-v3-{branch}", schema=MODEL_BUNDLE_SCHEMA)
        for branch in ("quant", "fundamental", "macro", "llm")
    }
    schedule = build_schedule_declaration_v3(
        protocol_attempt_id="attempt-v16-001",
        epoch="C",
        schedule_id="schedule-v3-c",
        seed_hex="3" * 64,
        genesis_ref=_ref("genesis-v3", schema=ATTEMPT_GENESIS_V3_SCHEMA),
        runtime_capsule=_ref("runtime-v3", schema=RUNTIME_CAPSULE_SCHEMA),
        open_session_calendar=calendar_ref,
        session_clock=clock_ref,
        calendar_recheck_ref=_ref("recheck-v3", schema=CALENDAR_RECHECK_SCHEMA),
        model_bundle_refs=models,
        calibration_universe_ref=_ref(
            "universe-v3",
            schema=CALIBRATION_UNIVERSE_SCHEMA,
        ),
        slots=[
            {
                "slot_id": "c-slot-v3-1",
                "s0_date": s0_date,
                "s0_open_at": f"{s0_date}T01:15:00Z",
                "s0_close_at": f"{s0_date}T07:00:00Z",
                "decision_cutoff_at": f"{s0_date}T07:30:00Z",
                "s1_open_at": f"{target_sessions[0]}T01:15:00Z",
                "target_sessions": target_sessions,
            }
        ],
        calendar=calendar,
        session_clock_value=clock,
    )
    assert schedule["schema_version"] == SCHEDULE_DECLARATION_V3_SCHEMA
    return schedule, calendar_ref


def test_mark_target_v3_common_requires_v3_binding_before_any_artifact_read() -> None:
    v2_binding = ScheduleAnchorBinding(schedule=None, timestamp=None)  # type: ignore[arg-type]

    with pytest.raises(EvidenceV2Error, match="requires ScheduleAnchorBindingV3"):
        prepare_mark_target_common_evidence_v3(
            schedule_anchor=v2_binding,  # type: ignore[arg-type]
            benchmark_manifest=None,  # type: ignore[arg-type]
            benchmark_parquet=b"",
        )


def test_mark_target_v3_common_accepts_validated_v3_schedule_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.v16.evidence_v2.target as target_module

    schedule, calendar_ref = _valid_v3_schedule_for_target()
    target_sessions = list(schedule["slots"][0]["target_sessions"])
    benchmark_payload = _h00300_parquet(
        dates=[date.fromisoformat(item) for item in target_sessions]
    )
    table_projection = validate_h00300_parquet(benchmark_payload)
    table_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=INDEX_TABLE_SCHEMA,
        absolute_path="/governed/h00300-v3.parquet",
        byte_sha256=hashlib.sha256(benchmark_payload).hexdigest(),
        semantic_sha256=table_projection["parquet_metadata_semantic_sha256"],
        root_policy="v16.governed-data-root.v2",
    )
    manifest = _bound(
        "h00300-v3-manifest.json",
        build_h00300_manifest(
            generation_id="h00300-v3",
            created_at="2026-07-07T00:00:00Z",
            table_ref=table_ref,
            parquet_payload=benchmark_payload,
            official_source_receipt=_ref("h00300-v3-official-source"),
            calendar_ref=calendar_ref,
        ),
    )
    binding = ScheduleAnchorBindingV3(
        evidence=None,  # type: ignore[arg-type]
        timestamp=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        target_module,
        "validate_schedule_anchor_binding_v3",
        lambda value: schedule if value is binding else pytest.fail("unexpected binding"),
    )

    common = prepare_mark_target_common_evidence_v3(
        schedule_anchor=binding,
        benchmark_manifest=manifest,
        benchmark_parquet=benchmark_payload,
    )

    assert common.epoch == "C"
    assert common.schedule_id == "schedule-v3-c"
    assert common.calendar_ref == calendar_ref
    assert common.slots[0][1] == tuple(target_sessions)


def test_mark_target_v3_bundle_rejects_v2_bundle_type() -> None:
    assert MarkTargetEvidenceBundleV3 is not MarkTargetEvidenceBundle
