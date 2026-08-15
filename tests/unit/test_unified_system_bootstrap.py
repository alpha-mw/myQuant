from __future__ import annotations

import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.contracts import canonical_json_bytes, get_contract, seal_artifact
from quant_investor.factors.governance import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    FactorGovernanceError,
    FactorValidationStore,
    validate_bootstrap_exception_evidence,
    validate_bootstrap_factor_set,
)
from quant_investor.factors.governance.bootstrap import (
    _factor_set_sha256,
    _set_rows,
    bootstrap_factor_definitions,
    compute_bootstrap_signals,
)
from quant_investor.factors.governance.common import business_identity
from quant_investor.factors.governance.contextual import _signal_hashes, _signal_statistics
from quant_investor.factors.governance.implementations import installed_semantic_row
from quant_investor.factors.governance.source import decode_source_role, role_schema
from quant_investor.intelligence import assess_readiness
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    SystemStore,
    build_emergency_controller,
    build_suspended_generation,
    installed_code_manifest_sha256,
)

BASE = "2026-08-14T00:00:00Z"
DECISION_SOURCE_RELATIVE_PATH = "operations/unified_cutover/bootstrap-decision.json"
DECISION_SOURCE_SHA256 = "f4add792c25eafa61730dfc839e1e5d6cd9c81de25b3c47b455359d26fb2ce95"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_system_source_is_factor_business_agnostic() -> None:
    system_root = Path(__file__).resolve().parents[2] / "quant_investor/system"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(system_root.rglob("*.py"))
    )
    forbidden = {
        "_expected_bootstrap_definitions",
        "_validate_bootstrap_ready",
        "pv_low_dollar_volume_5d",
        "pv_blend_volstab19x2_mom90_amihud5_w80",
        "pv_blend_volstab19x2_mom90_amihud5_w75",
        "rank(volume_stability_19x2)",
        "-log(mean(amount[t-4:t]))",
        "0.500000000000",
        "0.000000000000",
        "CONTROL_ONLY",
        "EQUAL_WEIGHT",
        "BOOTSTRAP_EXCEPTION",
        "MarketDataReader",
        "user-approved-unified-runtime-cutover",
        "bootstrap-decision.json",
    }
    assert not {token for token in forbidden if token in source}


def _write(root: Path, relative: str, raw: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _write_parquet(
    root: Path,
    relative: str,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    path.chmod(0o600)


def _source(
    store: SystemStore,
    relative: str,
    *,
    source_format: str,
    media_type: str,
) -> dict[str, str]:
    return store.put_source_file(
        relative,
        source_object_id="source-" + relative.replace("/", "-"),
        media_type=media_type,
        source_format=source_format,
        created_at=BASE,
    )


def _bundle(
    store: SystemStore,
    bundle_id: str,
    rows: list[tuple[str, dict[str, str]]],
) -> dict[str, str]:
    artifact = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": bundle_id,
            "state": "IMMUTABLE",
            "sources": [
                {"role": role, "source_ref": ref}
                for role, ref in sorted(rows, key=lambda row: row[0])
            ],
        },
        created_at=BASE,
    )
    return store.put_object(artifact)


def _generic(store: SystemStore, kind: str, identity: str) -> dict[str, str]:
    definition = get_contract(kind)
    payload: dict[str, Any] = {field: None for field in definition.required_payload_fields}
    payload[definition.identity_field] = identity
    return store.put_object(seal_artifact(kind, payload, created_at=BASE))


def _decision_bytes() -> bytes:
    repository = Path(__file__).resolve().parents[2]
    raw = (repository / DECISION_SOURCE_RELATIVE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DECISION_SOURCE_SHA256
    return raw


def _calendar_rows() -> list[dict[str, Any]]:
    first = date(2026, 8, 17)
    rows: list[dict[str, Any]] = []
    for ordinal in range(391):
        session = first + timedelta(days=ordinal)
        opens = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
            hours=1
        )
        rows.append(
            {
                "ordinal": ordinal,
                "open_session": session,
                "opens_at_utc": opens,
                "closes_at_utc": opens + timedelta(hours=6),
            }
        )
    return rows


def _market_rows() -> list[dict[str, Any]]:
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]
    first = date(2026, 4, 1)
    rows: list[dict[str, Any]] = []
    for ordinal in range(100):
        trade_date = first + timedelta(days=ordinal)
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "adj_close": 10.0 + symbol_index + ordinal * (0.01 + symbol_index * 0.001),
                    "amount": 1000.0 + symbol_index * 100.0 + ordinal,
                    "vol": 100.0 + symbol_index * 10.0 + np.sin(ordinal / 7.0),
                }
            )
    return rows


def _pit_rows() -> list[dict[str, Any]]:
    session = date(2026, 7, 9)
    return [
        {
            "signal_session": session,
            "symbol": symbol,
            "industry": "industry-a" if index < 2 else "industry-b",
            "total_mv": float(1_000_000 + index * 100_000),
            "tradable": True,
        }
        for index, symbol in enumerate(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"])
    ]


def _release_and_manifest(
    store: SystemStore,
    factor_store: FactorValidationStore,
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    release = seal_artifact(
        "system.release",
        {
            "release_id": "system-bootstrap-release",
            "state": "OPERATIONAL",
            "code_sha256": _sha("release-code"),
            "wheel_sha256": _sha("release-wheel"),
            "code_manifest_sha256": installed_code_manifest_sha256(),
        },
        created_at=BASE,
    )
    release_ref = store.put_object(release)
    contextual_ref = store.build_contextual_validator_component(
        BOOTSTRAP_VALIDATION_PROFILE,
        release_manifest_ref=release_ref,
        created_at=BASE,
    )
    decoder_ref = store.build_source_decoder_component(
        release_manifest_ref=release_ref,
        created_at=BASE,
    )
    implementation_refs: dict[str, dict[str, str]] = {}
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        row = installed_semantic_row(factor_id)
        implementation_refs[factor_id] = store.build_installed_component(
            component_id=row["implementation_id"],
            component_role="SOURCE_IMPLEMENTATION",
            package_name="quant_investor.factors.governance",
            module_names=[row["module_name"]],
            entrypoint_specs=[(row["module_name"], row["qualified_name"])],
            release_manifest_ref=release_ref,
            created_at=BASE,
        )
    manifest = factor_store.build_validator_manifest(
        release_manifest_ref=release_ref,
        contextual_validator_component_ref=contextual_ref,
        source_decoder_component_ref=decoder_ref,
        implementation_component_refs=implementation_refs,
    )
    return release_ref, manifest, store.put_object(manifest)


def _bootstrap_sources(
    store: SystemStore,
    factor_store: FactorValidationStore,
    source_root: Path,
    *,
    release_ref: dict[str, str],
    manifest: dict[str, Any],
) -> tuple[Any, dict[str, str], dict[str, str], dict[str, str]]:
    _write_parquet(
        source_root,
        "bootstrap/calendar.parquet",
        _calendar_rows(),
        role_schema("exchange_calendar"),
    )
    _write_parquet(
        source_root,
        "bootstrap/market.parquet",
        _market_rows(),
        role_schema("market_history"),
    )
    _write_parquet(
        source_root,
        "bootstrap/pit.parquet",
        _pit_rows(),
        role_schema("pit_universe"),
    )
    calendar_ref = _source(
        store,
        "bootstrap/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_ref = _source(
        store,
        "bootstrap/market.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    pit_ref = _source(
        store,
        "bootstrap/pit.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )

    normalized: dict[str, str] = {}
    for role, ref in (
        ("exchange_calendar", calendar_ref),
        ("market_history", market_ref),
        ("pit_universe", pit_ref),
    ):
        decoded = decode_source_role(
            system_store=store,
            source_object_ref=ref,
            role=role,
            projector=lambda table, binding: None,
        )
        normalized[role] = decoded.binding["normalized_sha256"]
    market_frame = pa.Table.from_pylist(
        _market_rows(), schema=role_schema("market_history")
    ).to_pandas()
    frames = {
        symbol: frame.drop(columns=["symbol"]).reset_index(drop=True)
        for symbol, frame in market_frame.groupby("symbol", sort=True)
    }
    signals = compute_bootstrap_signals(frames, source_format="PARQUET")
    canonical_signals = {
        factor_id: {
            symbol: None if np.isnan(value) else float(value).hex()
            for symbol, value in signals[factor_id].sort_index().items()
        }
        for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
    }
    signal_hashes = _signal_hashes(canonical_signals)
    definitions = bootstrap_factor_definitions()
    factor_rows, control_rows = _set_rows(definitions)
    factor_set_sha = _factor_set_sha256(
        definitions=definitions,
        factor_rows=factor_rows,
        control_rows=control_rows,
    )
    implementation_raw = canonical_json_bytes(
        {
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": manifest["payload"]["implementation_rows"],
        }
    )
    market_bundle_ref = _bundle(store, "market", [("market", market_ref)])
    implementation_sha = hashlib.sha256(implementation_raw).hexdigest()
    signal_statistics = _signal_statistics(
        canonical_signals,
        eligible_symbols=sorted(
            row["symbol"]
            for row in _pit_rows()
            if row["tradable"] and row["total_mv"] > 0
        ),
        implementation_sha256s={
            LOW_DOLLAR_VOLUME: implementation_sha,
            BLEND_W80: implementation_sha,
        },
        source_bundle_sha256=market_bundle_ref["byte_sha256"],
    )
    recomputation_raw = canonical_json_bytes(
        {
            "authority": "NON_AUTHORIZING",
            "domain": "myquant-bootstrap-recomputation",
            "factor_set_sha256": factor_set_sha,
            "factor_weights": [
                {"factor_id": row["factor_id"], "weight": row["weight"]} for row in factor_rows
            ],
            "implementation_rows": manifest["payload"]["implementation_rows"],
            "normalized_source_sha256s": normalized,
            "result": "EXACT_MATCH",
            "signal_sha256s": signal_hashes,
            "signal_statistics": signal_statistics,
        }
    )
    source_rows = [
        {
            "role": role,
            "source_ref": ref,
            "source_byte_sha256": store.get_object(ref)["payload"]["byte_sha256"],
        }
        for role, ref in (
            ("exchange_calendar", calendar_ref),
            ("market", market_ref),
            ("pit_universe", pit_ref),
        )
    ]
    source_rows.sort(key=lambda row: row["role"])
    generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": {
            "reader": "MarketDataReader",
            "market": "CN",
            "mode_policy": "strict",
            "source_format": "PARQUET",
            "fallback_allowed": False,
        },
        "source_rows": source_rows,
    }
    source_generation_raw = canonical_json_bytes(
        {
            **generation_body,
            "generation_sha256": hashlib.sha256(canonical_json_bytes(generation_body)).hexdigest(),
        }
    )
    for relative, raw in (
        (DECISION_SOURCE_RELATIVE_PATH, _decision_bytes()),
        ("bootstrap/implementation-tree.json", implementation_raw),
        ("bootstrap/recomputation.json", recomputation_raw),
        ("bootstrap/source-generation.json", source_generation_raw),
    ):
        _write(source_root, relative, raw)
    decision_ref = _source(
        store,
        DECISION_SOURCE_RELATIVE_PATH,
        source_format="JSON",
        media_type="application/json",
    )
    implementation_ref = _source(
        store,
        "bootstrap/implementation-tree.json",
        source_format="JSON",
        media_type="application/json",
    )
    recomputation_ref = _source(
        store,
        "bootstrap/recomputation.json",
        source_format="JSON",
        media_type="application/json",
    )
    source_generation_ref = _source(
        store,
        "bootstrap/source-generation.json",
        source_format="JSON",
        media_type="application/json",
    )
    bundle_refs = {
        "decision_source_bundle_ref": _bundle(
            store, "decision", [("bootstrap_decision", decision_ref)]
        ),
        "exchange_calendar_bundle_ref": _bundle(store, "calendar", [("calendar", calendar_ref)]),
        "implementation_bundle_ref": _bundle(
            store,
            "implementation",
            [("implementation_tree_manifest", implementation_ref)],
        ),
        "market_bundle_ref": market_bundle_ref,
        "pit_universe_bundle_ref": _bundle(store, "pit", [("pit", pit_ref)]),
        "recomputation_bundle_ref": _bundle(
            store, "recomputation", [("recomputation", recomputation_ref)]
        ),
        "source_generation_bundle_ref": _bundle(
            store,
            "source-generation",
            [("source_generation", source_generation_ref)],
        ),
    }
    closure = factor_store.initialize_bootstrap(release_ref=release_ref, **bundle_refs)
    return closure, calendar_ref, market_ref, pit_ref


def _operational_sources(
    store: SystemStore,
    source_root: Path,
    *,
    calendar_ref: dict[str, str],
    market_ref: dict[str, str],
    pit_ref: dict[str, str],
) -> dict[str, str]:
    _write(source_root, "data/market-manifest.json", b'{"status":"immutable"}')
    _write(source_root, "data/fundamental-manifest.json", b'{"status":"stale"}')
    _write(source_root, "data/fundamental.parquet", b"PAR1-fundamental")
    market_manifest_ref = _source(
        store,
        "data/market-manifest.json",
        source_format="JSON",
        media_type="application/json",
    )
    fundamental_manifest_ref = _source(
        store,
        "data/fundamental-manifest.json",
        source_format="JSON",
        media_type="application/json",
    )
    fundamental_ref = _source(
        store,
        "data/fundamental.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_bundle_ref = _bundle(
        store,
        "market-top",
        [("manifest", market_manifest_ref), ("table", market_ref)],
    )
    fundamental_bundle_ref = _bundle(
        store,
        "fundamental-top",
        [("manifest", fundamental_manifest_ref), ("table", fundamental_ref)],
    )
    return _bundle(
        store,
        "operational-source-closure",
        [
            ("exchange_calendar", calendar_ref),
            ("fundamental_generation", fundamental_bundle_ref),
            ("market_snapshot", market_bundle_ref),
            ("pit_membership", pit_ref),
        ],
    )


def _closure(tmp_path: Path) -> dict[str, Any]:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "canonical-source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="canonical-bootstrap-source",
    )
    factor_store = FactorValidationStore._for_testing(
        system_store=store,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    release_ref, manifest, manifest_ref = _release_and_manifest(store, factor_store)
    bootstrap, calendar_ref, market_ref, pit_ref = _bootstrap_sources(
        store,
        factor_store,
        source_root,
        release_ref=release_ref,
        manifest=manifest,
    )
    request = store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=manifest_ref,
        intrinsic_receipt_ref=bootstrap.intrinsic_receipt_ref,
    )
    validation = store.run_validation(request["validation_request_ref"])
    status = factor_store.build_status(
        active_factor_set_ref=bootstrap.active_set_ref,
        active_validation_receipt_ref=bootstrap.intrinsic_receipt_ref,
        active_contextual_result_ref=validation["contextual_result_ref"],
        active_validation_attestation_ref=validation["validation_attestation_ref"],
    )
    status_ref = store.put_object(status)
    readiness = assess_readiness(
        producer_identity=status["payload"]["active"]["producer_identity"],
        assessed_at=BASE,
        factor_status=status,
        source_blockers=["FUNDAMENTAL_SOURCE_STALE"],
        readiness_id="final-readiness-valid",
    )
    readiness_ref = store.put_object(readiness)
    top_sources_ref = _operational_sources(
        store,
        source_root,
        calendar_ref=calendar_ref,
        market_ref=market_ref,
        pit_ref=pit_ref,
    )
    suspended = build_suspended_generation(
        store,
        blockers=["EMERGENCY_TARGET"],
        created_at=BASE,
    )
    controller = build_emergency_controller(
        store,
        suspended_generation_id=suspended["generation_id"],
    )
    receipt = store.get_object(bootstrap.intrinsic_receipt_ref)
    context_payload = validation["contextual_result"]["payload"]
    kwargs = {
        "generation_state": "OPERATIONAL",
        "release_manifest_ref": release_ref,
        "source_refs": [top_sources_ref],
        "factor_source_object_refs": context_payload["source_object_refs"],
        "factor_policy_ref": bootstrap.policy_ref,
        "factor_evidence_refs": receipt["payload"]["evidence_refs"],
        "factor_active_set_ref": bootstrap.active_set_ref,
        "factor_validation_attestation_ref": validation["validation_attestation_ref"],
        "mainline_ref": None,
        "research_refs": [],
        "migration_receipt_ref": None,
        "migration_marker_ref": None,
        "skill_tree_sha256": _sha("skills"),
        "automation_semantic_sha256": _sha("automation"),
        "readiness_matrix_ref": readiness_ref,
        "emergency_controller_sha256": controller["byte_sha256"],
        "created_at": BASE,
    }
    return {
        "store": store,
        "workspace": workspace,
        "release_ref": release_ref,
        "policy": store.get_object(bootstrap.policy_ref),
        "active_set": store.get_object(bootstrap.active_set_ref),
        "receipt": receipt,
        "status": status,
        "status_ref": status_ref,
        "readiness": readiness,
        "validation": validation,
        "kwargs": kwargs,
    }


def test_operational_bootstrap_closure_assembles_and_deep_verifies_without_activation(
    tmp_path: Path,
) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]

    generation = store.assemble_generation(**closure["kwargs"])
    verified = store.verify_generation(
        generation["generation_id"],
        deployed_release_ref=closure["release_ref"],
    )

    assert verified["generation_state"] == "OPERATIONAL"
    assert verified["factor_policy"] == closure["policy"]
    assert verified["factor_active_set"] == closure["active_set"]
    assert verified["factor_validation_receipt"] == closure["receipt"]
    assert verified["factor_validation_resolution"]["outcome"] == "VALIDATED"
    assert verified["readiness"]["payload"]["factor_state"] == "READY"
    assert verified["readiness"]["payload"]["blockers"] == [
        "FUNDAMENTAL_SOURCE_STALE",
        "MAINLINE_CANDIDATE_ABSENT",
    ]
    assert (
        verified["factor_source_object_refs"]
        == closure["validation"]["contextual_result"]["payload"]["source_object_refs"]
    )
    assert not (closure["workspace"] / "results/system/_active.json").exists()


def _evidence_identity(payload: dict[str, Any]) -> str:
    return business_identity(
        "bootstrap-evidence",
        {
            "decision_source_sha256": payload["decision_source_sha256"],
            "factor_set_sha256": payload["factor_set_sha256"],
            "source_refs": payload["source_refs"],
        },
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    ["source_role", "code_sha", "implementation_sha", "w75_selectable"],
)
def test_operational_bootstrap_mutations_fail_before_generation_publication(
    tmp_path: Path,
    mutation: str,
) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    generations = closure["workspace"] / "results/system/generations"
    before = sorted(path.name for path in generations.iterdir())

    if mutation == "w75_selectable":
        payload = copy.deepcopy(closure["active_set"]["payload"])
        definition = next(
            row for row in payload["factor_definitions"] if row["factor_id"] == BLEND_W75_CONTROL
        )
        definition["selectable"] = True
        payload["control_rows"][0]["selectable"] = True
        forged = seal_artifact("factor.bootstrap_set", payload, created_at=BASE)
        store.put_object(forged)
        with pytest.raises(FactorGovernanceError):
            validate_bootstrap_factor_set(forged)
    else:
        payload = copy.deepcopy(closure["policy"]["payload"])
        if mutation == "source_role":
            row = next(row for row in payload["source_refs"] if row["role"] == "market")
            row["role"] = "market_wrong"
            payload["source_refs"].sort(key=lambda value: value["role"])
        elif mutation == "code_sha":
            payload["factor_rows"][0]["code_sha256"] = "0" * 64
        else:
            payload["factor_rows"][0]["implementation_sha256"] = "0" * 64
        payload["bootstrap_evidence_id"] = _evidence_identity(payload)
        forged = seal_artifact(
            "factor.bootstrap_exception_evidence",
            payload,
            created_at=BASE,
        )
        store.put_object(forged)
        with pytest.raises(FactorGovernanceError):
            validate_bootstrap_exception_evidence(forged)

    after = sorted(path.name for path in generations.iterdir())
    assert after == before
