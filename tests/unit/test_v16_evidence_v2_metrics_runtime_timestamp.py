from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import base64
import hashlib
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

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
from quant_investor.v16.evidence_v2.metrics import (
    CALIBRATION_BRANCHES,
    CalibrationArtifactPair,
    benjamini_hochberg_qvalues,
    bootstrap_draw_index,
    build_branch_prediction,
    build_calibration_evidence,
    build_calibration_universe_plan,
    build_lambda_fold_evidence,
    factor_b_multiple_testing_gates,
    one_sided_student_t_pvalue,
    validate_calibration_evidence,
)
from quant_investor.v16.evidence_v2.schedule import (
    ScheduleAnchorBinding,
    build_schedule_declaration,
)
from quant_investor.v16.evidence_v2.target import (
    EXPECTED_INDEX_SCHEMA,
    EXPECTED_STOCK_MARK_SCHEMA,
    INDEX_TABLE_SCHEMA,
    STOCK_MARK_TABLE_SCHEMA,
    CostVector,
    MarkTargetEvidenceBundle,
    StockMarkSourceBundle,
    ValidatedMarkTargetCommonEvidence,
    ValidatedStockMarkSources,
    build_adjustment_factor_evidence,
    build_cost_evidence,
    build_h00300_manifest,
    build_mark_target_outcome_from_common_evidence,
    build_pit_membership_evidence,
    build_stock_mark_evidence_from_sources,
    build_suspension_evidence,
    prepare_mark_target_common_evidence,
    prepare_stock_mark_sources,
    validate_h00300_parquet,
    validate_stock_mark_parquet,
)
from quant_investor.v16.evidence_v2.runtime_identity import (
    LLMProviderBuildIdentity,
    PINNED_OPENSSL_PATH,
    REQUIRED_ENVIRONMENT_CONTROLS,
    RUNTIME_COMPONENT_ORDER,
    RuntimeComponent,
    build_frozen_model_bundle,
    build_runtime_capsule,
    validate_frozen_model_bundle,
    validate_runtime_capsule,
)
from quant_investor.v16.evidence_v2.timestamp import (
    AnchorWindow,
    BoundArtifact,
    CommandResult,
    RevocationBinding,
    TIMESTAMP_RECEIPT_SCHEMA,
    TimestampAnchorBinding,
    TimestampPersistenceTerminalError,
    TimestampVerificationBundle,
    build_timestamp_attempt,
    persist_first_timestamp_response,
    record_persisted_response,
    record_partial_response_failure,
    record_timestamp_validation,
    record_transport_failure,
    validate_timestamp_attempt,
    validate_timestamp_receipt,
    verify_and_record_timestamp_validation,
    verify_rfc3161_bundle,
)


def _ref(
    name: str,
    *,
    payload: bytes | None = None,
    schema: str = "fixture.v1",
    absolute_path: str | None = None,
) -> EvidenceRef:
    raw = payload if payload is not None else (name + ":bytes").encode()
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=schema,
        absolute_path=absolute_path or f"/private/evidence/{name}",
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=hashlib.sha256((name + ":semantic").encode()).hexdigest(),
        root_policy="v16.private-evidence-root.v2",
    )


def _bound(name: str, payload: dict[str, Any]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    reference = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=payload["schema_version"],
        absolute_path=f"/private/evidence/{name}",
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=semantic_sha256(payload),
        root_policy="v16.private-evidence-root.v2",
    )
    return BoundCanonicalArtifact(reference=reference, payload=raw)


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
    verified_at = datetime.fromisoformat(verification_time.replace("Z", "+00:00"))
    if verified_at < gen_time:
        raise EvidenceV2Error("synthetic verification time predates genTime")
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


def _fake_timestamp_binding(
    artifact: BoundCanonicalArtifact,
    *,
    anchor_id: str,
    anchor_window: AnchorWindow,
    attempt_name: str,
    receipt_name: str,
) -> TimestampAnchorBinding:
    certificate_payload = b"-----BEGIN CERTIFICATE-----\nTSA\n-----END CERTIFICATE-----\n"
    issuer_payload = b"-----BEGIN CERTIFICATE-----\nROOT\n-----END CERTIFICATE-----\n"
    crl_payload = b"-----BEGIN X509 CRL-----\nCRL\n-----END X509 CRL-----\n"
    certificate = _artifact(
        f"{anchor_id}-certificate.pem",
        certificate_payload,
    )
    issuer = _artifact(
        f"{anchor_id}-issuer.pem",
        issuer_payload,
    )
    crl = _artifact(
        f"{anchor_id}-crl.pem",
        crl_payload,
    )
    response_directory = "/private/evidence/timestamp-responses"
    query = _artifact(f"{anchor_id}-request.tsq", f"query:{anchor_id}".encode())
    response = _artifact(
        f"{anchor_id}-response.tsr",
        f"response:{anchor_id}".encode(),
    )
    response = BoundArtifact(
        reference=replace(
            response.reference,
            absolute_path=f"{response_directory}/{anchor_id}.tsr",
        ),
        payload=response.payload,
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
        anchor_window=anchor_window,
        expected_policy_oid="1.2.3.4",
        openssl_binary_sha256="a" * 64,
        trust_anchor_ref=bundle.trust_anchor.reference,
        untrusted_chain_ref=bundle.untrusted_chain.reference,
        revocations=revocations,
    )
    attempt = record_persisted_response(attempt, response_ref=response.reference)
    upper = datetime.fromisoformat(anchor_window.not_after.replace("Z", "+00:00"))
    lower = (
        None
        if anchor_window.not_before is None
        else datetime.fromisoformat(anchor_window.not_before.replace("Z", "+00:00"))
    )
    gen_time = upper - timedelta(seconds=1) if lower is None else lower + (upper - lower) / 2
    timestamp = gen_time.isoformat().replace("+00:00", "Z")
    validation = verify_and_record_timestamp_validation(
        attempt,
        bundle=bundle,
        verification_time=timestamp,
        validation_receipt_path=f"/private/evidence/{receipt_name}",
    )
    return TimestampAnchorBinding(
        attempt=_bound(attempt_name, validation.attempt),
        validation_receipt=validation.validation_receipt,
        verification_bundle=bundle,
    )


def _calibration_slot(*, cohort_index: int, cohort_start: date) -> dict[str, Any]:
    s0 = cohort_start - timedelta(days=1)
    return {
        "slot_id": f"c-slot-{cohort_index:02d}",
        "s0_date": s0.isoformat(),
        "s0_open_at": f"{s0.isoformat()}T01:30:00Z",
        "s0_close_at": f"{s0.isoformat()}T07:00:00Z",
        "decision_cutoff_at": f"{s0.isoformat()}T07:30:00Z",
        "s1_open_at": f"{cohort_start.isoformat()}T01:30:00Z",
        "target_sessions": [
            (cohort_start + timedelta(days=index)).isoformat() for index in range(20)
        ],
    }


def _calibration_benchmark_parquet(dates: Sequence[date]) -> bytes:
    source_sha = hashlib.sha256(b"csindex-official-fixture").hexdigest()
    table = pa.Table.from_arrays(
        [
            pa.array(["H00300.CSI"] * len(dates), type=pa.string()),
            pa.array(dates, type=pa.date32()),
            pa.array([1000.0] * len(dates), type=pa.float64()),
            pa.array(["CNY"] * len(dates), type=pa.string()),
            pa.array(["gross_pre_tax_total_return"] * len(dates), type=pa.string()),
            pa.array(["csindex_official"] * len(dates), type=pa.string()),
            pa.array(
                [
                    datetime.combine(item, datetime.min.time(), tzinfo=timezone.utc)
                    + timedelta(hours=10)
                    for item in dates
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([source_sha] * len(dates), type=pa.string()),
        ],
        schema=EXPECTED_INDEX_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=20, write_statistics=True)
    return sink.getvalue().to_pybytes()


def _calibration_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    base = date(2026, 1, 2)
    for branch in CALIBRATION_BRANCHES:
        for cohort_index in range(8):
            cohort_start = base + timedelta(days=cohort_index * 30)
            cohort_end = cohort_start + timedelta(days=19)
            for sample_index in range(38):
                positive = sample_index % 2 == 0
                realized = 0.04 if positive else -0.01
                covered = sample_index < 34
                sample_id = f"{branch}-c{cohort_index:02d}-s{sample_index:02d}"
                specs.append(
                    {
                        "sample_id": sample_id,
                        "branch": branch,
                        "cohort_id": f"{branch}-cohort-{cohort_index:02d}",
                        "cohort_start": cohort_start,
                        "cohort_end": cohort_end,
                        "slot_id": f"c-slot-{cohort_index:02d}",
                        "symbol": f"S{cohort_index:02d}{sample_index:02d}.CN",
                        "probability": 0.98 if positive else 0.02,
                        "predicted": 0.035 if positive else -0.008,
                        "realized": realized,
                        "lower": realized - 0.02 if covered else realized + 0.10,
                        "upper": realized + 0.02 if covered else realized + 0.20,
                    }
                )
    return specs


def _calibration_stock_parquet(specs: Sequence[dict[str, Any]]) -> bytes:
    marks: dict[tuple[str, date], float] = {}
    for spec in specs:
        marks[(spec["symbol"], spec["cohort_start"])] = 10.0
        marks[(spec["symbol"], spec["cohort_end"])] = 10.0 * (1.0 + spec["realized"])
    ordered = sorted((symbol, trade_date, close) for (symbol, trade_date), close in marks.items())
    source_sha = hashlib.sha256(b"cn-stock-official-fixture").hexdigest()
    table = pa.Table.from_arrays(
        [
            pa.array([row[0] for row in ordered], type=pa.string()),
            pa.array([row[1] for row in ordered], type=pa.date32()),
            pa.array([row[2] for row in ordered], type=pa.float64()),
            pa.array(
                [
                    datetime.combine(row[1], datetime.min.time(), tzinfo=timezone.utc)
                    + timedelta(hours=10)
                    for row in ordered
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([source_sha] * len(ordered), type=pa.string()),
        ],
        schema=EXPECTED_STOCK_MARK_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=128, write_statistics=True)
    return sink.getvalue().to_pybytes()


def _calibration_stock_sources(
    *,
    specs: Sequence[dict[str, Any]],
    calendar_ref: EvidenceRef,
) -> tuple[StockMarkSourceBundle, ValidatedStockMarkSources]:
    parquet_payload = _calibration_stock_parquet(specs)
    projection = validate_stock_mark_parquet(parquet_payload)
    table_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=STOCK_MARK_TABLE_SCHEMA,
        absolute_path="/governed/cn-calibration-stock-marks.parquet",
        byte_sha256=hashlib.sha256(parquet_payload).hexdigest(),
        semantic_sha256=projection["parquet_metadata_semantic_sha256"],
        root_policy="v16.governed-data-root.v2",
    )
    keys = sorted(
        {(spec["symbol"], spec["cohort_start"].isoformat()) for spec in specs}
        | {(spec["symbol"], spec["cohort_end"].isoformat()) for spec in specs}
    )
    bundle = StockMarkSourceBundle(
        market_parquet=BoundRawArtifact(table_ref, parquet_payload),
        adjustment_factors=_bound(
            "calibration-adjustment-factors.json",
            build_adjustment_factor_evidence(
                generation_id="calibration-adjustment-001",
                market_table_ref=table_ref,
                rows=[
                    {"symbol": symbol, "trade_date": trade_date, "adj_factor": 1.0}
                    for symbol, trade_date in keys
                ],
            ),
        ),
        pit_membership=_bound(
            "calibration-pit-membership.json",
            build_pit_membership_evidence(
                generation_id="calibration-membership-001",
                calendar_ref=calendar_ref,
                rows=[
                    {"symbol": symbol, "trade_date": trade_date, "pit_listed": True}
                    for symbol, trade_date in keys
                ],
            ),
        ),
        suspensions=_bound(
            "calibration-suspensions.json",
            build_suspension_evidence(
                generation_id="calibration-suspension-001",
                calendar_ref=calendar_ref,
                rows=[
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "authoritative_suspension": False,
                        "stale_trade_date": None,
                    }
                    for symbol, trade_date in keys
                ],
            ),
        ),
    )
    return bundle, prepare_stock_mark_sources(bundle)


def _calibration_common_evidence(
    *,
    model_refs: dict[str, EvidenceRef],
    lambda_artifacts: dict[str, list[BoundCanonicalArtifact]],
    specs: Sequence[dict[str, Any]],
) -> tuple[
    BoundCanonicalArtifact,
    ScheduleAnchorBinding,
    BoundCanonicalArtifact,
    bytes,
    ValidatedMarkTargetCommonEvidence,
    EvidenceRef,
]:
    calendar_ref = _ref("calendar-c.json")
    sample_plans = [
        {
            "sample_id": spec["sample_id"],
            "branch": spec["branch"],
            "symbol": spec["symbol"],
            "cohort_id": spec["cohort_id"],
            "slot_id": spec["slot_id"],
            "prediction_path": f"/private/evidence/prediction-{spec['sample_id']}.json",
            "outcome_path": f"/private/evidence/outcome-{spec['sample_id']}.json",
            "stock_marks_path": f"/private/evidence/stock-{spec['sample_id']}.json",
            "costs_path": f"/private/evidence/cost-{spec['sample_id']}.json",
            "prediction_timestamp_attempt_path": (
                f"/private/evidence/timestamp-attempt-{spec['sample_id']}.json"
            ),
            "prediction_timestamp_receipt_path": (
                f"/private/evidence/timestamp-receipt-{spec['sample_id']}.json"
            ),
        }
        for spec in specs
    ]
    universe_payload = build_calibration_universe_plan(
        protocol_attempt_id="attempt-v16-001",
        epoch="C",
        schedule_id="schedule-c",
        model_bundle_refs=model_refs,
        sample_plans=sample_plans,
        lambda_fold_refs_by_branch={
            branch: [artifact.reference for artifact in lambda_artifacts[branch]]
            for branch in CALIBRATION_BRANCHES
        },
    )
    universe = _bound("calibration-universe.json", universe_payload)
    base = date(2026, 1, 2)
    cohort_starts = [base + timedelta(days=index * 30) for index in range(8)]
    schedule_payload = build_schedule_declaration(
        protocol_attempt_id="attempt-v16-001",
        epoch="C",
        schedule_id="schedule-c",
        seed_hex=hashlib.sha256(b"schedule-c-seed").hexdigest(),
        runtime_capsule=_ref("runtime-capsule.json"),
        open_session_calendar=calendar_ref,
        model_bundle_refs=model_refs,
        calibration_universe_ref=universe.reference,
        slots=[
            _calibration_slot(cohort_index=index, cohort_start=start)
            for index, start in enumerate(cohort_starts)
        ],
    )
    schedule = _bound("schedule-c.json", schedule_payload)
    schedule_timestamp = _fake_timestamp_binding(
        schedule,
        anchor_id="schedule-c",
        anchor_window=AnchorWindow(
            anchor_kind="schedule_declaration",
            not_before=None,
            not_after=str(schedule_payload["slots"][0]["s0_open_at"]),
        ),
        attempt_name="schedule-c-timestamp-attempt.json",
        receipt_name="schedule-c-timestamp-receipt.json",
    )
    schedule_anchor = ScheduleAnchorBinding(schedule=schedule, timestamp=schedule_timestamp)
    benchmark_dates = sorted(
        {start + timedelta(days=offset) for start in cohort_starts for offset in range(20)}
    )
    benchmark_parquet = _calibration_benchmark_parquet(benchmark_dates)
    table_projection = validate_h00300_parquet(benchmark_parquet)
    table_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=INDEX_TABLE_SCHEMA,
        absolute_path="/governed/h00300-calibration.parquet",
        byte_sha256=hashlib.sha256(benchmark_parquet).hexdigest(),
        semantic_sha256=table_projection["parquet_metadata_semantic_sha256"],
        root_policy="v16.governed-data-root.v2",
    )
    manifest_payload = build_h00300_manifest(
        generation_id="h00300-calibration-001",
        created_at="2026-09-01T00:00:00Z",
        table_ref=table_ref,
        parquet_payload=benchmark_parquet,
        official_source_receipt=_ref("h00300-source-receipt.json"),
        calendar_ref=calendar_ref,
    )
    manifest = _bound("h00300-calibration-manifest.json", manifest_payload)
    common = prepare_mark_target_common_evidence(
        schedule_anchor=schedule_anchor,
        benchmark_manifest=manifest,
        benchmark_parquet=benchmark_parquet,
    )
    return universe, schedule_anchor, manifest, benchmark_parquet, common, calendar_ref


def _calibration_artifacts(
    *,
    specs: Sequence[dict[str, Any]],
    schedule_anchor: ScheduleAnchorBinding,
    benchmark_manifest: BoundCanonicalArtifact,
    benchmark_parquet: bytes,
    common: ValidatedMarkTargetCommonEvidence,
    model_refs: dict[str, EvidenceRef],
    stock_source_bundle: StockMarkSourceBundle,
    stock_sources: ValidatedStockMarkSources,
) -> list[CalibrationArtifactPair]:
    pairs: list[CalibrationArtifactPair] = []
    zero_costs = CostVector((0.0,) * 8)
    slot_windows = {
        slot_id: (s0_close, s1_open)
        for slot_id, s0_close, s1_open in common.prediction_anchor_windows
    }
    for spec in specs:
        sample_id = spec["sample_id"]
        prediction_payload = build_branch_prediction(
            protocol_attempt_id="attempt-v16-001",
            epoch="C",
            sample_id=sample_id,
            branch=spec["branch"],
            cohort_id=spec["cohort_id"],
            cohort_start_date=spec["cohort_start"].isoformat(),
            cohort_end_date=spec["cohort_end"].isoformat(),
            probability=spec["probability"],
            prior_probability=0.5,
            predicted_alpha=spec["predicted"],
            interval_lower=spec["lower"],
            interval_upper=spec["upper"],
            model_bundle_ref=model_refs[spec["branch"]],
            schedule_ref=schedule_anchor.schedule.reference,
        )
        prediction = _bound(f"prediction-{sample_id}.json", prediction_payload)
        s0_close, s1_open = slot_windows[spec["slot_id"]]
        prediction_timestamp = _fake_timestamp_binding(
            prediction,
            anchor_id=f"prediction-{sample_id}",
            anchor_window=AnchorWindow(
                anchor_kind="prediction",
                not_before=s0_close,
                not_after=s1_open,
            ),
            attempt_name=f"timestamp-attempt-{sample_id}.json",
            receipt_name=f"timestamp-receipt-{sample_id}.json",
        )
        costs = _bound(
            f"cost-{sample_id}.json",
            build_cost_evidence(
                protocol_attempt_id="attempt-v16-001",
                sample_id=sample_id,
                costs=zero_costs,
                cost_model_ref=_ref("zero-cost-model.json"),
            ),
        )
        stock = _bound(
            f"stock-{sample_id}.json",
            build_stock_mark_evidence_from_sources(
                sources=stock_sources,
                protocol_attempt_id="attempt-v16-001",
                sample_id=sample_id,
                symbol=spec["symbol"],
                slot_id=spec["slot_id"],
                schedule_ref=schedule_anchor.schedule.reference,
                entry_date=spec["cohort_start"].isoformat(),
                exit_date=spec["cohort_end"].isoformat(),
            ),
        )
        outcome = build_mark_target_outcome_from_common_evidence(
            common=common,
            stock_marks=stock,
            stock_sources=stock_sources,
            costs=costs,
        )
        target_sources = MarkTargetEvidenceBundle(
            schedule_anchor=schedule_anchor,
            stock_marks=stock,
            stock_sources=stock_source_bundle,
            costs=costs,
            benchmark_manifest=benchmark_manifest,
            benchmark_parquet=benchmark_parquet,
        )
        pairs.append(
            CalibrationArtifactPair(
                prediction=prediction,
                prediction_timestamp=prediction_timestamp,
                outcome=_bound(f"outcome-{sample_id}.json", outcome),
                target_sources=target_sources,
            )
        )
    return pairs


def _lambda_artifacts(
    *,
    model_refs: dict[str, EvidenceRef],
) -> dict[str, list[BoundCanonicalArtifact]]:
    result: dict[str, list[BoundCanonicalArtifact]] = {}
    for branch in CALIBRATION_BRANCHES:
        result[branch] = []
        for index, value in enumerate((0.30, 0.35, 0.45)):
            payload = build_lambda_fold_evidence(
                protocol_attempt_id="attempt-v16-001",
                epoch="C",
                branch=branch,
                fold_id=f"fold-{index}",
                lambda_value=value,
                model_bundle_ref=model_refs[branch],
                fit_sample_ref=_ref(f"lambda-{branch}-{index}-fit.json"),
                holdout_sample_ref=_ref(f"lambda-{branch}-{index}-holdout.json"),
            )
            result[branch].append(_bound(f"lambda-{branch}-{index}.json", payload))
    return result


@pytest.fixture(scope="module")
def calibration_evidence() -> dict[str, Any]:
    model_refs = {
        branch: _ref(
            f"model-{branch}.json",
            schema="v16.frozen-model-bundle.v2",
        )
        for branch in CALIBRATION_BRANCHES
    }
    lambda_artifacts = _lambda_artifacts(model_refs=model_refs)
    specs = _calibration_specs()
    (
        universe,
        schedule_anchor,
        benchmark_manifest,
        benchmark_parquet,
        common,
        calendar_ref,
    ) = _calibration_common_evidence(
        model_refs=model_refs,
        lambda_artifacts=lambda_artifacts,
        specs=specs,
    )
    stock_source_bundle, stock_sources = _calibration_stock_sources(
        specs=specs,
        calendar_ref=calendar_ref,
    )
    pairs = _calibration_artifacts(
        specs=specs,
        schedule_anchor=schedule_anchor,
        benchmark_manifest=benchmark_manifest,
        benchmark_parquet=benchmark_parquet,
        common=common,
        model_refs=model_refs,
        stock_source_bundle=stock_source_bundle,
        stock_sources=stock_sources,
    )
    evidence = build_calibration_evidence(
        protocol_attempt_id="attempt-v16-001",
        epoch="C",
        universe=universe,
        schedule_ref=schedule_anchor.schedule.reference,
        model_bundle_refs=model_refs,
        sample_artifacts=pairs,
        lambda_fold_artifacts_by_branch=lambda_artifacts,
    )
    return {
        "evidence": evidence,
        "universe": universe,
        "pairs": pairs,
        "lambda_artifacts": lambda_artifacts,
    }


def test_four_branch_calibration_is_recomputed_and_all_gates_are_separate(
    calibration_evidence: dict[str, Any],
) -> None:
    validated = validate_calibration_evidence(
        calibration_evidence["evidence"],
        universe=calibration_evidence["universe"],
        sample_artifacts=calibration_evidence["pairs"],
        lambda_fold_artifacts_by_branch=calibration_evidence["lambda_artifacts"],
    )

    assert validated["all_metric_gates_passed"] is True
    assert validated["blockers"] == []
    assert validated["bootstrap"]["seed_hex"] == hashlib.sha256(b"schedule-c-seed").hexdigest()
    assert validated["new_risk_authorized"] is False
    for branch in CALIBRATION_BRANCHES:
        branch_result = validated["branches"][branch]
        assert branch_result["samples"] == 304
        assert branch_result["nonoverlap_cohorts"] == 8
        assert len(branch_result["gates"]) == 9
        assert all(branch_result["gates"].values())


def test_calibration_rejects_bootstrap_seed_not_bound_by_schedule(
    calibration_evidence: dict[str, Any],
) -> None:
    evidence = calibration_evidence["evidence"]
    tampered = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    tampered["bootstrap"] = dict(tampered["bootstrap"])
    tampered["bootstrap"]["seed_hex"] = hashlib.sha256(b"post-selected-seed").hexdigest()

    with pytest.raises(EvidenceV2Error, match="drifts from pre-s0 schedule"):
        validate_calibration_evidence(
            seal_semantic(tampered),
            universe=calibration_evidence["universe"],
            sample_artifacts=calibration_evidence["pairs"],
            lambda_fold_artifacts_by_branch=calibration_evidence["lambda_artifacts"],
        )


def test_calibration_rejects_resealed_caller_metric_injection(
    calibration_evidence: dict[str, Any],
) -> None:
    evidence = calibration_evidence["evidence"]
    tampered = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    branches = {key: dict(value) for key, value in tampered["branches"].items()}
    quant = dict(branches["quant"])
    metrics = dict(quant["metrics"])
    metrics["ece"] = "f64:0x0.0p+0"
    quant["metrics"] = metrics
    branches["quant"] = quant
    tampered["branches"] = branches
    resealed = seal_semantic(tampered)

    with pytest.raises(EvidenceV2Error, match="deterministic recomputation"):
        validate_calibration_evidence(
            resealed,
            universe=calibration_evidence["universe"],
            sample_artifacts=calibration_evidence["pairs"],
            lambda_fold_artifacts_by_branch=calibration_evidence["lambda_artifacts"],
        )


def test_calibration_rejects_sample_values_not_derived_from_bound_artifacts(
    calibration_evidence: dict[str, Any],
) -> None:
    evidence = calibration_evidence["evidence"]
    tampered = {key: value for key, value in evidence.items() if key != "semantic_sha256"}
    samples = [dict(item) for item in tampered["samples"]]
    samples[0]["realized_alpha"] = "f64:0x1.0p+0"
    tampered["samples"] = samples
    resealed = seal_semantic(tampered)

    with pytest.raises(EvidenceV2Error, match="deterministic recomputation"):
        validate_calibration_evidence(
            resealed,
            universe=calibration_evidence["universe"],
            sample_artifacts=calibration_evidence["pairs"],
            lambda_fold_artifacts_by_branch=calibration_evidence["lambda_artifacts"],
        )


def test_calibration_rejects_outcome_drift_from_bound_stock_source(
    calibration_evidence: dict[str, Any],
) -> None:
    pairs = list(calibration_evidence["pairs"])
    original = pairs[0]
    stock_payload = original.target_sources.stock_marks.read()
    tampered = {key: value for key, value in stock_payload.items() if key != "semantic_sha256"}
    tampered["exit"] = dict(tampered["exit"])
    tampered["exit"]["exact_mark"] = encode_f64(20.0)
    tampered_stock = _bound("tampered-stock-source.json", seal_semantic(tampered))
    pairs[0] = replace(
        original,
        target_sources=replace(
            original.target_sources,
            stock_marks=tampered_stock,
        ),
    )

    with pytest.raises(EvidenceV2Error, match="drifts from bound market sources"):
        validate_calibration_evidence(
            calibration_evidence["evidence"],
            universe=calibration_evidence["universe"],
            sample_artifacts=pairs,
            lambda_fold_artifacts_by_branch=calibration_evidence["lambda_artifacts"],
        )


def test_calibration_rejects_omitted_predeclared_sample(
    calibration_evidence: dict[str, Any],
) -> None:
    with pytest.raises(EvidenceV2Error, match="omit predeclared samples"):
        validate_calibration_evidence(
            calibration_evidence["evidence"],
            universe=calibration_evidence["universe"],
            sample_artifacts=calibration_evidence["pairs"][:-1],
            lambda_fold_artifacts_by_branch=calibration_evidence["lambda_artifacts"],
        )


def test_calibration_rejects_omitted_predeclared_lambda_fold(
    calibration_evidence: dict[str, Any],
) -> None:
    lambda_artifacts = dict(calibration_evidence["lambda_artifacts"])
    lambda_artifacts["quant"] = lambda_artifacts["quant"][:-1]
    with pytest.raises(EvidenceV2Error, match="drift from predeclared universe"):
        validate_calibration_evidence(
            calibration_evidence["evidence"],
            universe=calibration_evidence["universe"],
            sample_artifacts=calibration_evidence["pairs"],
            lambda_fold_artifacts_by_branch=lambda_artifacts,
        )


def test_calibration_universe_rejects_duplicate_branch_slot_symbol() -> None:
    model_refs = {
        branch: _ref(
            f"duplicate-model-{branch}.json",
            schema="v16.frozen-model-bundle.v2",
        )
        for branch in CALIBRATION_BRANCHES
    }
    sample_plans: list[dict[str, Any]] = []
    for branch in CALIBRATION_BRANCHES:
        count = 2 if branch == "quant" else 1
        for index in range(count):
            prefix = f"duplicate-{branch}-{index}"
            sample_plans.append(
                {
                    "sample_id": prefix,
                    "branch": branch,
                    "symbol": "S0001.CN",
                    "cohort_id": f"{branch}-cohort",
                    "slot_id": "slot-001",
                    "prediction_path": f"/private/evidence/{prefix}-prediction.json",
                    "outcome_path": f"/private/evidence/{prefix}-outcome.json",
                    "stock_marks_path": f"/private/evidence/{prefix}-stock.json",
                    "costs_path": f"/private/evidence/{prefix}-costs.json",
                    "prediction_timestamp_attempt_path": (
                        f"/private/evidence/{prefix}-timestamp-attempt.json"
                    ),
                    "prediction_timestamp_receipt_path": (
                        f"/private/evidence/{prefix}-timestamp-receipt.json"
                    ),
                }
            )
    lambda_refs = {
        branch: [
            _ref(
                f"duplicate-lambda-{branch}-{index}.json",
                schema="v16.lambda-fold-evidence.v2",
            )
            for index in range(2)
        ]
        for branch in CALIBRATION_BRANCHES
    }

    with pytest.raises(EvidenceV2Error, match="one sample per branch/slot/symbol"):
        build_calibration_universe_plan(
            protocol_attempt_id="attempt-v16-001",
            epoch="C",
            schedule_id="duplicate-schedule-c",
            model_bundle_refs=model_refs,
            sample_plans=sample_plans,
            lambda_fold_refs_by_branch=lambda_refs,
        )


def test_bootstrap_mapping_and_multiple_testing_are_deterministic() -> None:
    seed = hashlib.sha256(b"bootstrap").hexdigest()
    expected = (
        int.from_bytes(
            __import__("hmac")
            .new(
                bytes.fromhex(seed),
                b"7:3",
                hashlib.sha256,
            )
            .digest(),
            "big",
        )
        % 11
    )

    assert bootstrap_draw_index(seed_hex=seed, replicate=7, draw=3, count=11) == expected
    assert benjamini_hochberg_qvalues([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    gates = factor_b_multiple_testing_gates([0.001, 0.02])
    assert gates[0]["pass"] is True
    assert gates[1]["bonferroni_pass"] is True
    with pytest.raises(EvidenceV2Error, match="constant series"):
        one_sided_student_t_pvalue([1.0, 1.0, 1.0])


def _runtime_components() -> list[RuntimeComponent]:
    kinds = {
        "python_interpreter": "cpython-interpreter",
        "source_tree": "git-source-tree",
        "dependency_lock": "uv-lock",
        "installed_distributions": "python-distribution-manifest",
        "platform_manifest": "platform-manifest",
        "pyarrow_backend": "python-parquet-backend",
        "scipy_backend": "python-statistics-backend",
        "openssl_backend": "openssl-rfc3161-cli",
    }
    versions = {
        "python_interpreter": "CPython 3.12.10",
        "source_tree": "c03d36f115c0",
        "dependency_lock": "uv.lock-v1",
        "installed_distributions": "2026-07-18",
        "platform_manifest": "macOS-arm64",
        "pyarrow_backend": "24.0.0",
        "scipy_backend": "1.17.1",
        "openssl_backend": "OpenSSL 3.6.1",
    }
    components: list[RuntimeComponent] = []
    for index, component_id in enumerate(RUNTIME_COMPONENT_ORDER):
        runtime_path = (
            PINNED_OPENSSL_PATH if component_id == "openssl_backend" else f"/runtime/{component_id}"
        )
        artifact_ref = _ref(
            f"runtime-{component_id}.bin",
            absolute_path=runtime_path,
        )
        components.append(
            RuntimeComponent(
                component_id=component_id,
                component_kind=kinds[component_id],
                version=versions[component_id],
                build_id=f"build-{index}",
                absolute_runtime_path=runtime_path,
                artifact_ref=artifact_ref,
            )
        )
    return components


def test_runtime_capsule_binds_backends_and_exact_environment() -> None:
    capsule = build_runtime_capsule(
        protocol_attempt_id="attempt-v16-001",
        capsule_id="runtime-capsule-001",
        components=_runtime_components(),
        environment_controls=REQUIRED_ENVIRONMENT_CONTROLS,
    )

    assert validate_runtime_capsule(capsule)["network_access"] == "forbidden_during_recompute"
    broken_controls = dict(REQUIRED_ENVIRONMENT_CONTROLS)
    broken_controls["OMP_NUM_THREADS"] = "2"
    with pytest.raises(EvidenceV2Error, match="not hermetic"):
        build_runtime_capsule(
            protocol_attempt_id="attempt-v16-001",
            capsule_id="runtime-capsule-002",
            components=_runtime_components(),
            environment_controls=broken_controls,
        )


def test_llm_frozen_bundle_requires_immutable_provider_build_attestation() -> None:
    provider = LLMProviderBuildIdentity(
        provider_id="provider-a",
        model_id="model-family-a",
        immutable_model_build_id="model-family-a-20260718-build-7",
        endpoint_contract_id="responses-v1",
        tokenizer_ref=_ref("tokenizer.json"),
        inference_config_ref=_ref("inference-config.json"),
        provider_attestation_ref=_ref("provider-attestation.json"),
    )
    bundle = build_frozen_model_bundle(
        protocol_attempt_id="attempt-v16-001",
        branch="llm",
        bundle_id="llm-bundle-001",
        training_schedule_ref=_ref("schedule-a.json"),
        training_capture_ref=_ref("capture-a.json"),
        feature_contract_ref=_ref("features-a.json"),
        hyperparameter_ref=_ref("hyperparameters-a.json"),
        serialized_model_ref=_ref("llm-adapter-a.bin"),
        deterministic_inference_entrypoint="quant_investor.v16.llm:predict",
        llm_provider_build=provider,
    )

    assert validate_frozen_model_bundle(bundle)["frozen_after_epoch_a"] is True
    with pytest.raises(EvidenceV2Error, match="mutable or non-specific"):
        replace(provider, immutable_model_build_id="latest")


def _attempt_material() -> dict[str, Any]:
    certificate = _artifact(
        "attempt-certificate.pem",
        b"-----BEGIN CERTIFICATE-----\nCERT\n-----END CERTIFICATE-----\n",
    )
    issuer = _artifact(
        "attempt-issuer.pem",
        b"-----BEGIN CERTIFICATE-----\nROOT\n-----END CERTIFICATE-----\n",
    )
    crl = _artifact(
        "attempt-crl.pem",
        b"-----BEGIN X509 CRL-----\nCRL\n-----END X509 CRL-----\n",
    )
    return {
        "anchored_artifact_ref": _ref("attempt-anchored.json"),
        "anchor_window": AnchorWindow(
            anchor_kind="prediction",
            not_before="2026-07-18T11:59:59Z",
            not_after="2026-07-18T12:00:01Z",
        ),
        "expected_policy_oid": "1.2.3.4",
        "openssl_binary_sha256": "a" * 64,
        "trust_anchor_ref": issuer.reference,
        "untrusted_chain_ref": _ref("attempt-chain.pem"),
        "revocations": (
            RevocationBinding(
                certificate=certificate,
                issuer_certificate=issuer,
                crl=crl,
            ),
        ),
    }


def _receipt_for_attempt(
    state: dict[str, Any],
    *,
    name: str,
) -> BoundCanonicalArtifact:
    payload = seal_semantic(
        {
            "schema_version": TIMESTAMP_RECEIPT_SCHEMA,
            "anchored_artifact_ref": state["anchored_artifact_ref"],
            "request_ref": state["request_ref"],
            "response_ref": state["response_ref"],
            "trust_anchor_ref": state["trust_anchor_ref"],
            "untrusted_chain_ref": state["untrusted_chain_ref"],
            "revocation_refs": state["revocation_refs"],
            "policy_oid": state["policy_oid"],
            "gen_time": "2026-07-18T12:00:00Z",
            "verified_at": "2026-07-18T12:00:00Z",
            "anchor_kind": state["anchor_kind"],
            "anchor_not_before": state["anchor_not_before"],
            "anchor_not_after": state["anchor_not_after"],
            "openssl_path": state["openssl_path"],
            "openssl_binary_sha256": state["openssl_binary_sha256"],
            "response_projection_sha256": "b" * 64,
            "verification_stdout_sha256": "c" * 64,
            "data_verification_stdout_sha256": "d" * 64,
            "warnings": [],
            "cryptographically_valid_at_gen_time": True,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )
    validate_timestamp_receipt(payload)
    return _bound(name, payload)


def test_timestamp_state_first_response_wins_and_invalid_is_terminal(tmp_path: Path) -> None:
    private = tmp_path / "timestamps"
    private.mkdir(mode=0o700)
    request_ref = _ref("request.tsq")
    state = build_timestamp_attempt(
        protocol_attempt_id="attempt-v16-001",
        anchor_id="prediction-001",
        request_ref=request_ref,
        response_directory=private,
        **_attempt_material(),
    )
    state = record_transport_failure(state)
    assert state["transport_failure_count"] == 1
    persisted = persist_first_timestamp_response(
        private_directory=private,
        anchor_id="prediction-001",
        response=b"first-response",
    )
    response_ref = _ref(
        "response.tsr",
        payload=b"first-response",
        absolute_path=persisted.absolute_path,
    )
    state = record_persisted_response(state, response_ref=response_ref)
    failed = record_timestamp_validation(
        state,
        valid=False,
        blockers=["rfc3161_signature_invalid"],
    )

    assert validate_timestamp_attempt(failed)["state"] == "failed_terminal"
    with pytest.raises(EvidenceV2Error, match="retry is forbidden"):
        record_transport_failure(failed)

    assert persisted.byte_sha256 == hashlib.sha256(b"first-response").hexdigest()
    assert (private / "prediction-001.tsr").stat().st_mode & 0o777 == 0o600
    with pytest.raises(EvidenceV2Error, match="already exists"):
        persist_first_timestamp_response(
            private_directory=private,
            anchor_id="prediction-001",
            response=b"replacement-response",
        )


def test_timestamp_validation_rejects_caller_declared_success(tmp_path: Path) -> None:
    private = tmp_path / "timestamps"
    private.mkdir(mode=0o700)
    state = build_timestamp_attempt(
        protocol_attempt_id="attempt-v16-001",
        anchor_id="prediction-bound",
        request_ref=_ref("bound-request.tsq"),
        response_directory=private,
        **_attempt_material(),
    )
    response_ref = _ref(
        "bound-response.tsr",
        payload=b"response",
        absolute_path=str(private / "prediction-bound.tsr"),
    )
    state = record_persisted_response(state, response_ref=response_ref)
    receipt = _receipt_for_attempt(state, name="bound-receipt.json")
    with pytest.raises(EvidenceV2Error, match="caller-declared RFC3161 success is forbidden"):
        record_timestamp_validation(
            state,
            valid=True,
            validation_receipt=receipt,
        )


def test_timestamp_binding_revalidates_exact_bound_raw_bundle() -> None:
    artifact = _bound(
        "revalidated-anchor.json",
        seal_semantic({"schema_version": "fixture.v1", "value": "bound"}),
    )
    binding = _fake_timestamp_binding(
        artifact,
        anchor_id="revalidated-anchor",
        anchor_window=AnchorWindow(
            anchor_kind="prediction",
            not_before="2026-07-18T11:59:59Z",
            not_after="2026-07-18T12:00:01Z",
        ),
        attempt_name="revalidated-attempt.json",
        receipt_name="revalidated-receipt.json",
    )

    attempt, receipt = binding.read()
    assert attempt["state"] == "validated"
    assert receipt["anchored_artifact_ref"] == artifact.reference.to_dict()

    wrong_artifact = BoundArtifact(
        reference=_ref("wrong-anchor.bin", payload=b"wrong"),
        payload=b"wrong",
    )
    drifted_bundle = replace(binding.verification_bundle, anchored_artifact=wrong_artifact)
    with pytest.raises(EvidenceV2Error, match="bundle drifts from attempt"):
        replace(binding, verification_bundle=drifted_bundle).read()


def test_timestamp_receipt_rejects_verification_before_gen_time(tmp_path: Path) -> None:
    private = tmp_path / "timestamps"
    private.mkdir(mode=0o700)
    state = build_timestamp_attempt(
        protocol_attempt_id="attempt-v16-001",
        anchor_id="prediction-time-order",
        request_ref=_ref("time-order-request.tsq"),
        response_directory=private,
        **_attempt_material(),
    )
    state = record_persisted_response(
        state,
        response_ref=_ref(
            "time-order-response.tsr",
            absolute_path=str(private / "prediction-time-order.tsr"),
        ),
    )
    receipt = _receipt_for_attempt(state, name="time-order-receipt.json").read()
    invalid = {key: value for key, value in receipt.items() if key != "semantic_sha256"}
    invalid["verified_at"] = "2026-07-18T11:59:59Z"

    with pytest.raises(EvidenceV2Error, match="verification predates genTime"):
        validate_timestamp_receipt(seal_semantic(invalid))


def test_partial_timestamp_write_becomes_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.v16.evidence_v2.timestamp as timestamp_module

    private = tmp_path / "timestamps"
    private.mkdir(mode=0o700)
    state = build_timestamp_attempt(
        protocol_attempt_id="attempt-v16-001",
        anchor_id="prediction-partial",
        request_ref=_ref("partial-request.tsq"),
        response_directory=private,
        **_attempt_material(),
    )
    real_write = timestamp_module.os.write
    calls = 0

    def partial_write(descriptor: int, payload: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(payload)[:4])
        raise OSError("injected write failure")

    monkeypatch.setattr(timestamp_module.os, "write", partial_write)
    with pytest.raises(TimestampPersistenceTerminalError) as captured:
        persist_first_timestamp_response(
            private_directory=private,
            anchor_id="prediction-partial",
            response=b"partial-response",
        )
    monkeypatch.setattr(timestamp_module.os, "write", real_write)
    failed = record_partial_response_failure(
        state,
        persisted=captured.value.persisted,
    )

    assert failed["state"] == "failed_terminal"
    assert failed["partial_response"]["size"] == 4
    with pytest.raises(EvidenceV2Error, match="retry is forbidden"):
        record_transport_failure(failed)


def test_rfc3161_bundle_requires_crl_for_every_non_root_certificate() -> None:
    certificate_a = b"-----BEGIN CERTIFICATE-----\nCERT-A\n-----END CERTIFICATE-----\n"
    certificate_b = b"-----BEGIN CERTIFICATE-----\nCERT-B\n-----END CERTIFICATE-----\n"
    root = b"-----BEGIN CERTIFICATE-----\nROOT\n-----END CERTIFICATE-----\n"
    crl = b"-----BEGIN X509 CRL-----\nCRL-A\n-----END X509 CRL-----\n"

    with pytest.raises(EvidenceV2Error, match="cover every non-root"):
        TimestampVerificationBundle(
            anchored_artifact=_artifact("coverage-anchored.json", b"anchored"),
            query=_artifact("coverage-query.tsq", b"query"),
            response=_artifact("coverage-response.tsr", b"response"),
            trust_anchor=_artifact("coverage-root.pem", root),
            untrusted_chain=_artifact(
                "coverage-chain.pem",
                certificate_a + certificate_b,
            ),
            revocations=(
                RevocationBinding(
                    certificate=_artifact("coverage-cert-a.pem", certificate_a),
                    issuer_certificate=_artifact("coverage-issuer.pem", root),
                    crl=_artifact("coverage-crl.pem", crl),
                ),
            ),
        )


def test_rfc3161_verification_uses_gen_time_and_warns_on_later_revocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.v16.evidence_v2.timestamp as timestamp_module

    certificate = b"-----BEGIN CERTIFICATE-----\nCERT-A\n-----END CERTIFICATE-----\n"
    root = b"-----BEGIN CERTIFICATE-----\nROOT\n-----END CERTIFICATE-----\n"
    crl = b"-----BEGIN X509 CRL-----\nCRL-A\n-----END X509 CRL-----\n"
    bundle = TimestampVerificationBundle(
        anchored_artifact=_artifact("anchored.json", b"anchored"),
        query=_artifact("query.tsq", b"query"),
        response=_artifact("response.tsr", b"response"),
        trust_anchor=_artifact("root.pem", root),
        untrusted_chain=_artifact("chain.pem", certificate),
        revocations=(
            RevocationBinding(
                certificate=_artifact("certificate.pem", certificate),
                issuer_certificate=_artifact("root-issuer.pem", root),
                crl=_artifact("crl.pem", crl),
            ),
        ),
    )
    backend_sha = "a" * 64
    backend_signature = (1, 2, 3)
    monkeypatch.setattr(
        timestamp_module,
        "_hash_backend",
        lambda _path: (backend_sha, backend_signature),
    )
    monkeypatch.setattr(
        timestamp_module,
        "_stage_verified_backend",
        lambda **kwargs: str(kwargs["destination"]),
    )

    def runner(command: Sequence[str], _cwd: Path) -> CommandResult:
        if command[1:3] == ["ts", "-reply"]:
            return CommandResult(
                0,
                b"Status: Granted.\nPolicy OID: 1.2.3.4\n"
                b"Time stamp: Jul 18 12:00:00 2026 GMT\n",
                b"",
            )
        if command[1:3] == ["ts", "-verify"]:
            assert "-crl_check_all" not in command
            assert command[command.index("-attime") + 1] == "1784376000"
            return CommandResult(0, b"Verification: OK\n", b"")
        if command[1] == "verify":
            return CommandResult(0, b"certificate.pem: OK\n", b"")
        if command[1] == "x509":
            return CommandResult(0, b"serial=01\n", b"")
        if command[1] == "crl":
            if "-verify" in command:
                return CommandResult(0, b"", b"verify OK\n")
            return CommandResult(
                0,
                b"Last Update: Jul 19 00:00:00 2026 GMT\n"
                b"Next Update: Jul 26 00:00:00 2026 GMT\n"
                b"Revoked Certificates:\n"
                b"    Serial Number: 01\n"
                b"        Revocation Date: Jul 19 12:00:00 2026 GMT\n",
                b"",
            )
        raise AssertionError(command)

    receipt = verify_rfc3161_bundle(
        bundle=bundle,
        anchor_window=AnchorWindow(
            anchor_kind="prediction",
            not_before="2026-07-18T11:59:59Z",
            not_after="2026-07-18T12:00:01Z",
        ),
        expected_policy_oid="1.2.3.4",
        openssl_binary_sha256=backend_sha,
        verification_time="2026-07-20T12:00:00Z",
        runner=runner,
    )

    assert receipt["cryptographically_valid_at_gen_time"] is True
    assert receipt["warnings"] == [
        "certificate_revoked_after_gen_time:"
        + bundle.revocations[0].certificate.reference.byte_sha256
    ]
    assert receipt["new_risk_authorized"] is False


@pytest.mark.skipif(
    not Path(PINNED_OPENSSL_PATH).exists(),
    reason="pinned Homebrew OpenSSL backend is macOS-specific",
)
@pytest.mark.parametrize(
    "fixture_name",
    ["rfc3161_valid_bundle.json", "rfc3161_revoked_bundle.json"],
)
def test_real_rfc3161_fixture_verifies_with_pinned_openssl(fixture_name: str) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "v16_evidence_v2" / fixture_name
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    artifacts = {
        name: base64.b64decode(payload, validate=True)
        for name, payload in fixture["artifacts_base64"].items()
    }
    bundle = TimestampVerificationBundle(
        anchored_artifact=_artifact(
            f"{fixture_name}-anchored.json",
            artifacts["anchored_artifact"],
        ),
        query=_artifact(f"{fixture_name}-query.tsq", artifacts["query"]),
        response=_artifact(f"{fixture_name}-response.tsr", artifacts["response"]),
        trust_anchor=_artifact(
            f"{fixture_name}-root.pem",
            artifacts["trust_anchor"],
        ),
        untrusted_chain=_artifact(
            f"{fixture_name}-chain.pem",
            artifacts["untrusted_chain"],
        ),
        revocations=(
            RevocationBinding(
                certificate=_artifact(
                    f"{fixture_name}-certificate.pem",
                    artifacts["certificate"],
                ),
                issuer_certificate=_artifact(
                    f"{fixture_name}-issuer.pem",
                    artifacts["issuer_certificate"],
                ),
                crl=_artifact(f"{fixture_name}-crl.pem", artifacts["crl"]),
            ),
        ),
    )
    anchor = fixture["anchor"]
    with open(PINNED_OPENSSL_PATH, "rb") as handle:
        openssl_sha = hashlib.sha256(handle.read()).hexdigest()

    receipt = verify_rfc3161_bundle(
        bundle=bundle,
        anchor_window=AnchorWindow(
            anchor_kind=anchor["anchor_kind"],
            not_before=anchor["not_before"],
            not_after=anchor["not_after"],
        ),
        expected_policy_oid=fixture["policy_oid"],
        openssl_binary_sha256=openssl_sha,
        verification_time=fixture["verification_time"],
    )

    assert receipt["gen_time"] == fixture["expected_gen_time"]
    assert bool(receipt["warnings"]) is fixture["expected_post_gen_time_revocation_warning"]
