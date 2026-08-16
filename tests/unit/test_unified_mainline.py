from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import quant_investor.factors.governance.production as production_module

from quant_investor.cli.main import main
from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.factors.governance import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    FactorValidationStore,
    bootstrap_factor_definitions,
    compute_bootstrap_signals,
    validate_factor_status,
)
from quant_investor.factors.governance.bootstrap import _factor_set_sha256, _set_rows
from quant_investor.factors.governance.contextual import _signal_hashes, _signal_statistics
from quant_investor.factors.governance.implementations import installed_semantic_row
from quant_investor.factors.governance.source import decode_source_role, role_schema
from quant_investor.intelligence import (
    IntelligenceError,
    assess_fundamental,
    assess_industry,
    assess_readiness,
    assess_theme,
    build_decision_context,
    make_investment_decision,
    validate_readiness,
)
from quant_investor.intelligence._common import NO_AUTHORITY, artifact_ref
from quant_investor.mainline import (
    MAINLINE_BLOCKED,
    MAINLINE_UNINITIALIZED,
    MainlineError,
    MainlineStore,
    build_mainline_candidate,
    compose_mainline_readiness,
    mainline_status,
    read_public_run,
    validate_mainline_readiness,
)
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    EMPTY,
    SystemContractError,
    SystemStore,
    build_emergency_controller,
    build_suspended_generation,
    installed_code_manifest_sha256,
    suspend_system,
)

NOW = "2026-08-14T00:00:00Z"
BOOTSTRAP_NOW = "2026-08-16T00:00:00Z"
BOOTSTRAP_DECISION_PATH = "operations/unified_cutover/bootstrap-decision.json"
STRATEGY = "research-strategy"


def _research_payload(**fields: object) -> dict:
    return {
        "authority": dict(NO_AUTHORITY),
        "production": False,
        "research_only": True,
        "run_state": "INACTIVE",
        **fields,
    }


def _candidate_dependencies() -> tuple[dict, dict, dict]:
    source = seal_artifact(
        "system.source_bundle",
        {"source_bundle_id": "source-a", "sources": [], "state": "READY"},
        created_at=NOW,
    )
    source_ref = artifact_ref(source)
    evidence = seal_artifact(
        "evidence_bundle",
        _research_payload(
            bundle_id="evidence-a",
            blocker_codes=[],
            compiled_at=NOW,
            evaluation_ref=source_ref,
            evidence_refs=[source_ref],
            status="READY",
            strategy_id=STRATEGY,
        ),
        created_at=NOW,
    )
    industry = assess_industry(
        company="000001.SZ",
        memberships=[
            {
                "available_at": NOW,
                "effective_from": NOW,
                "exposure": "1",
                "industry_id": "BANK",
                "provider": "official",
                "retired": False,
            }
        ],
        provider_precedence=["official"],
        as_of=NOW,
        metric_rows=[
            {
                "metric_id": "profitability",
                "status": "AVAILABLE",
                "value": "0.8",
                "weight": "1",
            }
        ],
    )
    theme = assess_theme(
        company="000001.SZ",
        memberships=[
            {
                "available_at": NOW,
                "exposure": "1",
                "exposure_basis": "REVENUE",
                "provider": "official",
                "score": "0.7",
                "status": "ACTIVE",
                "theme_id": "FINTECH",
            }
        ],
        provider_precedence=["official"],
        catalog_complete=True,
        as_of=NOW,
    )
    fundamental = assess_fundamental(
        company="000001.SZ",
        component_scores={
            "business_quality": "0.8",
            "earnings_quality": "0.7",
            "growth_durability": "0.6",
            "industry_cycle": "0.5",
            "valuation": "0.9",
        },
        component_weights={
            "business_quality": "0.2",
            "earnings_quality": "0.2",
            "growth_durability": "0.2",
            "industry_cycle": "0.2",
            "valuation": "0.2",
        },
        minimum_coverage="0.6",
        as_of=NOW,
        source_refs=[source_ref],
        industry_assessment=industry,
        theme_assessment=theme,
    )
    context = build_decision_context(
        company="000001.SZ",
        as_of=NOW,
        hypothesis_status="VALID",
        risk_status="AVAILABLE",
        evidence_refs=[source_ref],
        industry_assessment=industry,
        theme_assessment=theme,
        fundamental_assessment=fundamental,
        quant_ref=source_ref,
    )
    decision = make_investment_decision(
        context=context,
        deterministic_percentile="0.95",
        thresholds={"paper_candidate": "0.90", "research_approved": "0.70"},
        as_of=NOW,
    )
    portfolio = seal_artifact(
        "research_portfolio",
        _research_payload(
            portfolio_id="portfolio-a",
            as_of=NOW,
            blocker_codes=[],
            cash_weight="0.800000000000",
            decision_refs=[artifact_ref(decision)],
            gross_weight="0.200000000000",
            hard_veto_codes=[],
            status="AVAILABLE",
            strategy_id=STRATEGY,
            targets=[
                {
                    "company_code": "000001.SZ",
                    "final_weight": "0.200000000000",
                    "rank": 1,
                }
            ],
        ),
        created_at=NOW,
    )
    return evidence, decision, portfolio


@dataclass(frozen=True)
class _BootstrapRuntime:
    release_ref: dict[str, str]
    factor_policy_ref: dict[str, str]
    factor_evidence_refs: list[dict[str, str]]
    factor_active_set_ref: dict[str, str]
    factor_validation_attestation_ref: dict[str, str]
    factor_source_object_refs: list[dict[str, str]]
    factor_status: dict[str, Any]
    factor_status_ref: dict[str, str]
    validation_result: dict[str, Any]


def _write_source(root: Path, relative: str, raw: bytes) -> None:
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


def _put_source(
    store: SystemStore,
    relative: str,
    *,
    source_format: str,
    media_type: str,
) -> dict[str, str]:
    return store.put_source_file(
        relative,
        source_object_id="mainline-" + relative.replace("/", "-"),
        media_type=media_type,
        source_format=source_format,
        created_at=BOOTSTRAP_NOW,
    )


def _put_bootstrap_bundle(
    store: SystemStore,
    bundle_id: str,
    role: str,
    source_ref: dict[str, str],
) -> dict[str, str]:
    return store.put_object(
        seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": f"mainline-{bundle_id}",
                "state": "IMMUTABLE",
                "sources": [{"role": role, "source_ref": source_ref}],
            },
            created_at=BOOTSTRAP_NOW,
        )
    )


def _bootstrap_calendar_rows() -> list[dict[str, Any]]:
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


def _bootstrap_market_rows() -> list[dict[str, Any]]:
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


def _bootstrap_pit_rows() -> list[dict[str, Any]]:
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


def _bootstrap_release_and_manifest(
    store: SystemStore,
    factor_store: FactorValidationStore,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    release = seal_artifact(
        "system.release",
        {
            "release_id": "mainline-bootstrap-release",
            "state": "OPERATIONAL",
            "code_sha256": hashlib.sha256(b"mainline-release-code").hexdigest(),
            "wheel_sha256": hashlib.sha256(b"mainline-release-wheel").hexdigest(),
            "code_manifest_sha256": installed_code_manifest_sha256(),
        },
        created_at=BOOTSTRAP_NOW,
    )
    release_ref = store.put_object(release)
    contextual_ref = store.build_contextual_validator_component(
        BOOTSTRAP_VALIDATION_PROFILE,
        release_manifest_ref=release_ref,
        created_at=BOOTSTRAP_NOW,
    )
    decoder_ref = store.build_source_decoder_component(
        release_manifest_ref=release_ref,
        created_at=BOOTSTRAP_NOW,
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
            created_at=BOOTSTRAP_NOW,
        )
    manifest = factor_store.build_validator_manifest(
        release_manifest_ref=release_ref,
        contextual_validator_component_ref=contextual_ref,
        source_decoder_component_ref=decoder_ref,
        implementation_component_refs=implementation_refs,
    )
    return release_ref, store.put_object(manifest), manifest


def _build_bootstrap_runtime(
    store: SystemStore,
    factor_store: FactorValidationStore,
    source_root: Path,
) -> _BootstrapRuntime:
    release_ref, manifest_ref, manifest = _bootstrap_release_and_manifest(store, factor_store)

    _write_parquet(
        source_root,
        "bootstrap/calendar.parquet",
        _bootstrap_calendar_rows(),
        role_schema("exchange_calendar"),
    )
    _write_parquet(
        source_root,
        "bootstrap/market.parquet",
        _bootstrap_market_rows(),
        role_schema("market_history"),
    )
    _write_parquet(
        source_root,
        "bootstrap/pit.parquet",
        _bootstrap_pit_rows(),
        role_schema("pit_universe"),
    )
    calendar_ref = _put_source(
        store,
        "bootstrap/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_ref = _put_source(
        store,
        "bootstrap/market.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    pit_ref = _put_source(
        store,
        "bootstrap/pit.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )

    normalized: dict[str, str] = {}
    for role, reference in (
        ("exchange_calendar", calendar_ref),
        ("market_history", market_ref),
        ("pit_universe", pit_ref),
    ):
        decoded = decode_source_role(
            system_store=store,
            source_object_ref=reference,
            role=role,
            projector=lambda table, binding: None,
        )
        normalized[role] = decoded.binding["normalized_sha256"]
    market_frame = pa.Table.from_pylist(
        _bootstrap_market_rows(), schema=role_schema("market_history")
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

    repository = Path(__file__).resolve().parents[2]
    decision_raw = (repository / BOOTSTRAP_DECISION_PATH).read_bytes()
    implementation_raw = canonical_json_bytes(
        {
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": manifest["payload"]["implementation_rows"],
        }
    )
    market_bundle_ref = _put_bootstrap_bundle(store, "market", "market", market_ref)
    implementation_sha = hashlib.sha256(implementation_raw).hexdigest()
    signal_statistics = _signal_statistics(
        canonical_signals,
        eligible_symbols=sorted(
            row["symbol"]
            for row in _bootstrap_pit_rows()
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
            "source_ref": reference,
            "source_byte_sha256": store.get_object(reference)["payload"]["byte_sha256"],
        }
        for role, reference in (
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
        (BOOTSTRAP_DECISION_PATH, decision_raw),
        ("bootstrap/implementation-tree.json", implementation_raw),
        ("bootstrap/recomputation.json", recomputation_raw),
        ("bootstrap/source-generation.json", source_generation_raw),
    ):
        _write_source(source_root, relative, raw)
    decision_ref = _put_source(
        store,
        BOOTSTRAP_DECISION_PATH,
        source_format="JSON",
        media_type="application/json",
    )
    implementation_ref = _put_source(
        store,
        "bootstrap/implementation-tree.json",
        source_format="JSON",
        media_type="application/json",
    )
    recomputation_ref = _put_source(
        store,
        "bootstrap/recomputation.json",
        source_format="JSON",
        media_type="application/json",
    )
    source_generation_ref = _put_source(
        store,
        "bootstrap/source-generation.json",
        source_format="JSON",
        media_type="application/json",
    )
    bundle_refs = {
        "decision_source_bundle_ref": _put_bootstrap_bundle(
            store, "decision", "bootstrap_decision", decision_ref
        ),
        "exchange_calendar_bundle_ref": _put_bootstrap_bundle(
            store, "calendar", "calendar", calendar_ref
        ),
        "implementation_bundle_ref": _put_bootstrap_bundle(
            store,
            "implementation",
            "implementation_tree_manifest",
            implementation_ref,
        ),
        "market_bundle_ref": market_bundle_ref,
        "pit_universe_bundle_ref": _put_bootstrap_bundle(store, "pit", "pit", pit_ref),
        "recomputation_bundle_ref": _put_bootstrap_bundle(
            store, "recomputation", "recomputation", recomputation_ref
        ),
        "source_generation_bundle_ref": _put_bootstrap_bundle(
            store,
            "source-generation",
            "source_generation",
            source_generation_ref,
        ),
    }
    closure = factor_store.initialize_bootstrap(
        release_ref=release_ref,
        **bundle_refs,
    )
    request = store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=manifest_ref,
        intrinsic_receipt_ref=closure.intrinsic_receipt_ref,
    )
    validation_result = store.run_validation(request["validation_request_ref"])
    factor_status = factor_store.build_status(
        active_factor_set_ref=closure.active_set_ref,
        active_validation_receipt_ref=closure.intrinsic_receipt_ref,
        active_contextual_result_ref=validation_result["contextual_result_ref"],
        active_validation_attestation_ref=validation_result["validation_attestation_ref"],
    )
    factor_status = validate_factor_status(factor_status)
    receipt = store.get_object(closure.intrinsic_receipt_ref)
    context_payload = validation_result["contextual_result"]["payload"]
    return _BootstrapRuntime(
        release_ref=release_ref,
        factor_policy_ref=closure.policy_ref,
        factor_evidence_refs=list(receipt["payload"]["evidence_refs"]),
        factor_active_set_ref=closure.active_set_ref,
        factor_validation_attestation_ref=validation_result["validation_attestation_ref"],
        factor_source_object_refs=list(context_payload["source_object_refs"]),
        factor_status=factor_status,
        factor_status_ref=store.put_object(factor_status),
        validation_result=validation_result,
    )


def _operational_source_bundle(store: SystemStore, workspace_root: Path) -> dict[str, str]:
    fixture_root = workspace_root / "source-fixtures"
    fixture_root.mkdir()
    files = {
        "calendar.parquet": b"PAR1-calendar",
        "fundamental.json": b"{}",
        "fundamental.parquet": b"PAR1-fundamental",
        "market.json": b"{}",
        "market.parquet": b"PAR1-market",
        "membership.parquet": b"PAR1-membership",
    }
    for name, raw in files.items():
        fixture = fixture_root / name
        fixture.write_bytes(raw)
        fixture.chmod(0o400)

    def source(name: str, source_format: str) -> dict[str, str]:
        media_type = (
            "application/json" if source_format == "JSON" else "application/vnd.apache.parquet"
        )
        return store.put_source_file(
            f"source-fixtures/{name}",
            source_object_id=f"source-{name.replace('.', '-')}",
            media_type=media_type,
            source_format=source_format,
            created_at=NOW,
        )

    calendar_ref = source("calendar.parquet", "PARQUET")
    fundamental_refs = [
        ("json", source("fundamental.json", "JSON")),
        ("parquet", source("fundamental.parquet", "PARQUET")),
    ]
    market_refs = [
        ("json", source("market.json", "JSON")),
        ("parquet", source("market.parquet", "PARQUET")),
    ]
    membership_ref = source("membership.parquet", "PARQUET")

    def bundle(bundle_id: str, rows: list[tuple[str, dict[str, str]]]) -> dict[str, str]:
        artifact = seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": bundle_id,
                "state": "READY",
                "sources": [{"role": role, "source_ref": reference} for role, reference in rows],
            },
            created_at=NOW,
        )
        return store.put_object(artifact)

    fundamental_ref = bundle("fundamental-generation-a", fundamental_refs)
    market_ref = bundle("market-snapshot-a", market_refs)
    return bundle(
        "operational-source-closure-a",
        [
            ("exchange_calendar", calendar_ref),
            ("fundamental_generation", fundamental_ref),
            ("market_snapshot", market_ref),
            ("pit_membership", membership_ref),
        ],
    )


def test_mainline_candidate_is_inactive_acyclic_and_business_identified() -> None:
    evidence, decision, portfolio = _candidate_dependencies()
    first = build_mainline_candidate(
        strategy_id=STRATEGY,
        as_of=NOW,
        evidence_bundle=evidence,
        decision=decision,
        portfolio=portfolio,
        result={"companies": ["000001.SZ"], "summary": "paper candidate"},
        candidate_id="owner-candidate-1",
    )
    changed_nonidentity_result = build_mainline_candidate(
        strategy_id=STRATEGY,
        as_of=NOW,
        evidence_bundle=evidence,
        decision=decision,
        portfolio=portfolio,
        result={"companies": ["000001.SZ"], "summary": "changed research"},
        candidate_id="owner-candidate-1",
    )

    assert first["artifact_id"] == changed_nonidentity_result["artifact_id"]
    assert first["semantic_sha256"] != changed_nonidentity_result["semantic_sha256"]
    assert first["payload"]["run_state"] == "INACTIVE"
    assert first["payload"]["research_only"] is True
    assert first["payload"]["production"] is False
    assert all(value is False for value in first["payload"]["authority"].values())
    assert "readiness_ref" not in first["payload"]
    assert "active_generation_id" not in first["payload"]

    with pytest.raises(IntelligenceError, match="activation binding"):
        build_mainline_candidate(
            strategy_id=STRATEGY,
            as_of=NOW,
            evidence_bundle=evidence,
            decision=decision,
            portfolio=portfolio,
            result={"active_generation_id": "0" * 64},
        )


def test_mainline_readiness_preserves_factor_and_source_blockers() -> None:
    evidence, decision, portfolio = _candidate_dependencies()
    candidate = build_mainline_candidate(
        strategy_id=STRATEGY,
        as_of=NOW,
        evidence_bundle=evidence,
        decision=decision,
        portfolio=portfolio,
        result={"companies": ["000001.SZ"], "summary": "blocked research"},
    )
    base = assess_readiness(
        producer_identity="SYSTEM",
        assessed_at=NOW,
        factor_status=None,
        source_blockers=["FUNDAMENTAL_CUTOFF_STALE"],
    )

    composed = compose_mainline_readiness(
        base,
        mainline_candidate=candidate,
    )

    assert composed["artifact_id"] == base["artifact_id"]
    assert composed["semantic_sha256"] != base["semantic_sha256"]
    assert composed["payload"]["blockers"] == [
        "FACTOR_STATUS_UNAVAILABLE",
        "FUNDAMENTAL_CUTOFF_STALE",
    ]
    assert composed["payload"]["factor_state"] == "BLOCKED"
    assert composed["payload"]["mainline_candidate_ref"] == artifact_ref(candidate)
    assert composed["payload"]["mainline_state"] == "BLOCKED"
    assert composed["payload"]["investment_state"] == "BLOCKED"
    assert validate_mainline_readiness(composed) == composed
    with pytest.raises(IntelligenceError, match="Mainline"):
        validate_readiness(composed)


def test_real_operational_generation_returns_only_generation_bound_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test exercises Mainline state transitions below the independently
    # tested production-source admission boundary.
    monkeypatch.setattr(
        production_module,
        "validate_production_bootstrap_generation_closure",
        lambda **_kwargs: {},
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(mode=0o700)
    source_root = workspace_root
    store = SystemStore(workspace_root)
    from unified_activation_helpers import prepare_migration_context

    migration_context = prepare_migration_context(
        store,
        created_at="2026-08-13T23:59:59Z",
    )
    factor_store = FactorValidationStore._for_testing(
        system_store=store,
        clock=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    evidence, decision, portfolio = _candidate_dependencies()
    candidate = build_mainline_candidate(
        strategy_id=STRATEGY,
        as_of=NOW,
        evidence_bundle=evidence,
        decision=decision,
        portfolio=portfolio,
        result={"companies": ["000001.SZ"], "summary": "official paper candidate"},
    )
    bootstrap = _build_bootstrap_runtime(store, factor_store, source_root)
    factor_status = bootstrap.factor_status
    assert bootstrap.validation_result["outcome"] == "VALIDATED"
    assert bootstrap.validation_result["validation_completion"] is not None
    assert bootstrap.validation_result["source_verification_snapshot"] is not None
    status_request = {
        "active_factor_set_ref": bootstrap.factor_active_set_ref,
        "active_validation_receipt_ref": factor_status["payload"]["active"][
            "validation_receipt_ref"
        ],
        "active_contextual_result_ref": bootstrap.validation_result["contextual_result_ref"],
        "active_validation_attestation_ref": bootstrap.factor_validation_attestation_ref,
        "observed_composite_state_ref": None,
    }
    status_request_raw = canonical_json_bytes(status_request)
    status_request_path = workspace_root / "factor-status-request.json"
    status_request_path.write_bytes(status_request_raw)
    status_request_path.chmod(0o600)
    capsys.readouterr()
    main(
        [
            "factor",
            "status",
            "--workspace-root",
            str(workspace_root),
            "--request",
            status_request_path.name,
            "--expected-request-sha256",
            hashlib.sha256(status_request_raw).hexdigest(),
        ]
    )
    status_output = capsys.readouterr()
    assert status_output.err == ""
    assert status_output.out.count("\n") == 1
    status_projection = json.loads(status_output.out)
    assert set(status_projection) == {"blockers", "readiness", "status_ref"}
    assert status_projection["blockers"] == []
    assert status_projection["readiness"] == "READY"
    assert status_projection["status_ref"]["kind"] == "factor.status"
    assert (
        status_projection["status_ref"]["artifact_id"] == bootstrap.factor_status_ref["artifact_id"]
    )
    cli_status = validate_factor_status(store.get_object(status_projection["status_ref"]))
    assert cli_status["payload"] == factor_status["payload"]
    assert status_output.out == canonical_json_bytes(status_projection).decode("utf-8") + "\n"
    assert not (workspace_root / "results/system/_active.json").exists()
    initial_readiness = assess_readiness(
        producer_identity="NOT_CLAIMED",
        assessed_at=NOW,
        factor_status=factor_status,
        source_blockers=["FUNDAMENTAL_CUTOFF_STALE"],
    )
    assert initial_readiness["payload"]["factor_state"] == "READY"
    assert initial_readiness["payload"]["mainline_state"] == "UNINITIALIZED"
    assert initial_readiness["payload"]["investment_state"] == "BLOCKED"
    assert initial_readiness["payload"]["blockers"] == [
        "FUNDAMENTAL_CUTOFF_STALE",
        "MAINLINE_CANDIDATE_ABSENT",
    ]
    source_blocked_readiness = compose_mainline_readiness(
        initial_readiness,
        mainline_candidate=candidate,
    )
    assert source_blocked_readiness["payload"]["blockers"] == ["FUNDAMENTAL_CUTOFF_STALE"]
    assert source_blocked_readiness["payload"]["mainline_candidate_ref"] == artifact_ref(candidate)
    assert source_blocked_readiness["payload"]["mainline_state"] == "BLOCKED"
    assert source_blocked_readiness["payload"]["investment_state"] == "BLOCKED"
    candidate_free_readiness = assess_readiness(
        producer_identity="NOT_CLAIMED",
        assessed_at=NOW,
        factor_status=factor_status,
    )
    readiness = compose_mainline_readiness(
        candidate_free_readiness,
        mainline_candidate=candidate,
    )
    assert validate_mainline_readiness(readiness) == readiness
    assert readiness["payload"]["admission_route"] == "BOOTSTRAP_EXCEPTION"
    assert readiness["payload"]["factor_state"] == "READY"
    assert readiness["payload"]["mainline_state"] == "READY"

    release_ref = bootstrap.release_ref
    source_ref = _operational_source_bundle(store, source_root)
    factor_policy_ref = bootstrap.factor_policy_ref
    factor_evidence_refs = bootstrap.factor_evidence_refs
    active_factor_set_ref = bootstrap.factor_active_set_ref
    attestation = store.get_object(bootstrap.factor_validation_attestation_ref)
    assert (
        factor_status["payload"]["active"]["validation_receipt_ref"]
        == attestation["payload"]["intrinsic_receipt_ref"]
    )
    initial_readiness_ref = store.put_object(initial_readiness)
    suspended = build_suspended_generation(
        store,
        blockers=["MAINLINE_OPERATIONAL_FIXTURE_CONTROLLER"],
        created_at=BOOTSTRAP_NOW,
    )
    controller = build_emergency_controller(
        store,
        suspended_generation_id=suspended["generation_id"],
    )

    initial_generation = store.assemble_generation(
        generation_state="OPERATIONAL",
        release_manifest_ref=release_ref,
        source_refs=[source_ref],
        factor_source_object_refs=bootstrap.factor_source_object_refs,
        factor_policy_ref=factor_policy_ref,
        factor_evidence_refs=factor_evidence_refs,
        factor_active_set_ref=active_factor_set_ref,
        factor_validation_attestation_ref=bootstrap.factor_validation_attestation_ref,
        mainline_ref=None,
        research_refs=[],
        migration_receipt_ref=None,
        migration_marker_ref=None,
        skill_tree_sha256="b" * 64,
        automation_semantic_sha256="c" * 64,
        readiness_matrix_ref=initial_readiness_ref,
        emergency_controller_sha256=controller["byte_sha256"],
        created_at=BOOTSTRAP_NOW,
    )
    from unified_activation_helpers import activate_initial

    initial_active = activate_initial(
        store,
        initial_generation,
        release_ref,
        prepared_at="2026-08-13T23:59:59Z",
        activated_at="2026-08-14T00:00:00Z",
        migration_context=migration_context,
    )
    initial_state = MainlineStore(workspace_root).status(strategy_id=STRATEGY)
    assert initial_state == {
        "active_generation_id": initial_generation["generation_id"],
        "blockers": ["FUNDAMENTAL_CUTOFF_STALE", "MAINLINE_CANDIDATE_ABSENT"],
        "investment_state": "BLOCKED",
        "mainline_state": "UNINITIALIZED",
        "result": None,
        "status": "BLOCKED",
    }
    with pytest.raises(MainlineError) as uninitialized:
        read_public_run(workspace_root, strategy_id=STRATEGY)
    assert uninitialized.value.code == MAINLINE_UNINITIALIZED
    assert uninitialized.value.public_fields == initial_state

    candidate_ref = store.put_object(candidate)
    research_refs = [
        store.put_object(evidence),
        store.put_object(decision),
        store.put_object(portfolio),
    ]
    readiness_ref = store.put_object(readiness)

    generation = store.assemble_generation(
        generation_state="OPERATIONAL",
        release_manifest_ref=release_ref,
        source_refs=[source_ref],
        factor_source_object_refs=bootstrap.factor_source_object_refs,
        factor_policy_ref=factor_policy_ref,
        factor_evidence_refs=factor_evidence_refs,
        factor_active_set_ref=active_factor_set_ref,
        factor_validation_attestation_ref=bootstrap.factor_validation_attestation_ref,
        mainline_ref=candidate_ref,
        research_refs=research_refs,
        migration_receipt_ref=None,
        migration_marker_ref=None,
        skill_tree_sha256="b" * 64,
        automation_semantic_sha256="c" * 64,
        readiness_matrix_ref=readiness_ref,
        emergency_controller_sha256=controller["byte_sha256"],
        created_at=BOOTSTRAP_NOW,
    )
    active = store.activate_generation(
        generation["generation_id"],
        expected_pointer_sha256=initial_active["pointer_byte_sha256"],
        activated_at=BOOTSTRAP_NOW,
        os_actor="test-suite",
        deployed_release_ref=release_ref,
    )
    assert active["generation_id"] == generation["generation_id"]

    resolved_active = store.read_active()
    assert resolved_active is not None
    assert resolved_active["deployed_release_verified"] is True
    truncated_resolution = dict(resolved_active["factor_validation_resolution"])
    truncated_resolution.pop("validation_completion")
    validation_blocked = {
        "active_generation_id": generation["generation_id"],
        "blockers": ["FACTOR_VALIDATION_CLOSURE_INVALID"],
        "investment_state": "BLOCKED",
        "mainline_state": "BLOCKED",
        "result": None,
        "status": "BLOCKED",
    }
    for field, replacement in (
        ("factor_validation_resolution", truncated_resolution),
        ("factor_validation_attestation", None),
        ("factor_validation_completion", None),
        ("factor_source_verification_snapshot", None),
        ("factor_validation_receipt", None),
        ("factor_active_set", None),
        ("factor_policy", None),
        ("factor_evidence", []),
        ("factor_source_objects", []),
    ):
        assert (
            mainline_status(
                {**resolved_active, field: replacement},
                strategy_id=STRATEGY,
            )
            == validation_blocked
        )

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    state = MainlineStore(workspace_root).status(strategy_id=STRATEGY)
    assert state == {
        "active_generation_id": generation["generation_id"],
        "blockers": [],
        "investment_state": "PAPER_CANDIDATE",
        "mainline_state": "ACTIVE",
        "result": candidate["payload"]["result"],
        "status": "ACTIVE",
    }
    public_run = read_public_run(workspace_root, strategy_id=STRATEGY)
    assert public_run["kind"] == "public_run"
    assert public_run["created_at"] == generation["manifest"]["created_at"]
    assert public_run["payload"] == {
        "run_id": public_run["artifact_id"],
        "candidate_ref": candidate_ref,
        "active_generation_id": generation["generation_id"],
        "investment_state": "PAPER_CANDIDATE",
        "readiness_ref": readiness_ref,
        "result": candidate["payload"]["result"],
        "status": "ACTIVE",
        "strategy_id": STRATEGY,
    }

    with pytest.raises(MainlineError) as mismatched:
        read_public_run(workspace_root, strategy_id="different-strategy")
    assert mismatched.value.code == MAINLINE_BLOCKED
    assert mismatched.value.public_fields == {
        "active_generation_id": generation["generation_id"],
        "blockers": ["MAINLINE_CANDIDATE_MISMATCH"],
        "investment_state": "BLOCKED",
        "mainline_state": "BLOCKED",
        "result": None,
        "status": "BLOCKED",
    }
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert after == before


def test_initial_store_is_uninitialized_and_reads_do_not_fabricate_result(
    tmp_path: Path,
) -> None:
    store = MainlineStore(tmp_path)
    expected = {
        "active_generation_id": None,
        "blockers": ["ACTIVE_GENERATION_ABSENT"],
        "investment_state": "BLOCKED",
        "mainline_state": "UNINITIALIZED",
        "result": None,
        "status": "BLOCKED",
    }

    assert store.status(strategy_id=STRATEGY) == expected
    with pytest.raises(MainlineError) as captured:
        read_public_run(tmp_path, strategy_id=STRATEGY)
    assert captured.value.code == MAINLINE_UNINITIALIZED
    assert captured.value.exit_code == 2
    assert captured.value.public_fields == expected
    assert not (tmp_path / "results").exists()


def test_real_suspended_generation_activation_is_blocked_and_read_only(
    tmp_path: Path,
) -> None:
    store = SystemStore(tmp_path)
    generation = build_suspended_generation(
        store,
        blockers=["SOURCE_CLOSURE_BLOCKED"],
        created_at=NOW,
    )
    with pytest.raises(SystemContractError, match="initial activation"):
        suspend_system(
            store,
            target_active_pointer_raw=b"{}",
            expected_pointer_sha256=EMPTY,
        )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())

    state = MainlineStore(tmp_path).status(strategy_id=STRATEGY)
    assert state == {
        "active_generation_id": None,
        "blockers": ["ACTIVE_GENERATION_ABSENT"],
        "investment_state": "BLOCKED",
        "mainline_state": "UNINITIALIZED",
        "result": None,
        "status": "BLOCKED",
    }
    assert generation["generation_state"] == "SYSTEM_SUSPENDED"
    assert not (tmp_path / "results/system/_active.json").exists()

    with pytest.raises(MainlineError) as captured:
        MainlineStore(tmp_path).read_public_run(strategy_id=STRATEGY)
    assert captured.value.code == MAINLINE_UNINITIALIZED
    assert captured.value.public_fields == state
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert after == before


def _verified_suspended_projection(tmp_path: Path) -> dict[str, Any]:
    store = SystemStore(tmp_path)
    generation = build_suspended_generation(
        store,
        blockers=["SOURCE_CLOSURE_BLOCKED"],
        created_at=NOW,
    )
    pointer = {
        "activated_at": NOW,
        "generation_id": generation["generation_id"],
        "manifest_sha256": generation["manifest_sha256"],
        "os_actor": "test-suite",
        "previous_pointer_sha256": "1" * 64,
    }
    pointer_raw = canonical_json_bytes(pointer)
    return {
        **store.verify_generation(generation["generation_id"]),
        "pointer": pointer,
        "pointer_byte_sha256": hashlib.sha256(pointer_raw).hexdigest(),
    }


def test_caller_supplied_active_payload_must_retain_verified_pointer_binding(
    tmp_path: Path,
) -> None:
    active = _verified_suspended_projection(tmp_path)
    tampered = {**active, "pointer": {**active["pointer"], "os_actor": "other"}}
    tampered["pointer"]["generation_id"] = "0" * 64

    status = mainline_status(tampered, strategy_id=STRATEGY)
    assert status["status"] == "BLOCKED"
    assert status["blockers"] == ["ACTIVE_POINTER_BINDING_INVALID"]
    assert status["active_generation_id"] is None
    assert status["result"] is None

    unverified = {**active, "verified": False}
    blocked = mainline_status(unverified, strategy_id=STRATEGY)
    assert blocked == {
        "active_generation_id": active["generation_id"],
        "blockers": ["GENERATION_NOT_VERIFIED"],
        "investment_state": "BLOCKED",
        "mainline_state": "BLOCKED",
        "result": None,
        "status": "BLOCKED",
    }


def test_caller_supplied_operational_mapping_requires_deployed_release_verification() -> None:
    forged = {
        "deployed_release_verified": False,
        "generation_state": "OPERATIONAL",
        "verified": True,
    }

    status = mainline_status(forged, strategy_id=STRATEGY)

    assert status == {
        "active_generation_id": None,
        "blockers": ["DEPLOYED_RELEASE_NOT_VERIFIED"],
        "investment_state": "BLOCKED",
        "mainline_state": "BLOCKED",
        "result": None,
        "status": "BLOCKED",
    }
