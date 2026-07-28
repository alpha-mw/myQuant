from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
import hashlib
from typing import Any

import pytest

from quant_investor.v17_v4_contract import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_runtime.calibration import (
    build_calibration_receipt,
    build_fusion_promotion_receipt,
)
from quant_investor.v17_v4_runtime.deep_control import (
    DEEP_BUNDLE_VERSION,
    FUSION_TOP24_VERSION,
    DeepClosureError,
    DeepEvidenceInput,
    FusionTop24Input,
    build_deep_evidence_bundle,
    build_fusion_top24,
)
from quant_investor.v17_v4_runtime.portfolio_control import (
    HoldingInput,
    PermissionInput,
    PortfolioControlError,
    artifact_ref as portfolio_artifact_ref,
    build_holdings_snapshot,
    build_macro_overlay,
    build_markov_overlay,
    build_pretrade_permissions,
    build_production_portfolio,
    build_regime_evidence,
    build_risk_policy,
)
from tests.unit.test_v17_v4_calibration import (
    _fixture,
    _run,
    _store_v4_artifact,
)

NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
PIT_ROLES = (
    "benchmark_total_return",
    "cn_open_day_calendar",
    "corporate_actions",
    "market_bars",
    "official_delisting_cash",
    "pit_fundamentals",
    "universe_membership",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sessions_ending(end: str, count: int) -> list[str]:
    current = date.fromisoformat(end)
    sessions: list[str] = []
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current.isoformat())
        current -= timedelta(days=1)
    return list(reversed(sessions))


def _calendar_rows(
    sessions: list[str],
    *,
    available_at: str,
) -> list[dict[str, Any]]:
    return [
        {
            "available_at": available_at,
            "is_open": True,
            "market_id": "CN",
            "session": session,
        }
        for session in sessions
    ]


def _ordered_key_hash(values: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(canonical_bytes(list(value)))
        digest.update(b"\n")
    return digest.hexdigest()


def _calendar_row_set_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows,
        key=lambda item: (item["market_id"], item["session"]),
    ):
        digest.update(canonical_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _pit_catalog(
    *,
    strategy_id: str,
    cutoff: str,
    sessions: list[str],
    calendar_rows: list[dict[str, Any]],
    calendar_ref: dict[str, str],
) -> dict[str, Any]:
    dataset_refs: dict[str, dict[str, str]] = {}
    expected_refs: dict[str, dict[str, str]] = {}
    for role in PIT_ROLES:
        dataset_refs[role] = (
            dict(calendar_ref)
            if role == "cn_open_day_calendar"
            else {
                "artifact_id": f"dataset-{role}",
                "artifact_version": f"myquant.v17.v4.dataset.{role}.v1",
                "byte_sha256": _sha(f"dataset-bytes:{role}"),
                "cutoff": cutoff,
                "relative_path": (
                    f"data/private/v17_v4_sources/{role}/dataset.json"
                ),
                "semantic_sha256": _sha(f"dataset-semantic:{role}"),
                "strategy_id": strategy_id,
            }
        )
        expected_refs[role] = {
            "artifact_id": f"expected-keys-{role}",
            "artifact_version": (
                f"myquant.v17.v4.expected-keys.{role}.v1"
            ),
            "byte_sha256": _sha(f"expected-bytes:{role}"),
            "cutoff": cutoff,
            "relative_path": (
                f"data/private/v17_v4_sources/{role}/expected-keys.json"
            ),
            "semantic_sha256": _sha(f"expected-semantic:{role}"),
            "strategy_id": strategy_id,
        }
    calendar_key_hash = _ordered_key_hash(
        [("CN", session) for session in sessions]
    )
    summaries = [
        {
            "expected_keys_sha256": (
                calendar_key_hash
                if role == "cn_open_day_calendar"
                else _sha(f"expected:{role}")
            ),
            "latest_available_at": cutoff,
            "natural_key_fields": (
                ["market_id", "session"]
                if role == "cn_open_day_calendar"
                else ["key_a", "key_b"]
            ),
            "observed_keys_sha256": (
                calendar_key_hash
                if role == "cn_open_day_calendar"
                else _sha(f"observed:{role}")
            ),
            "role": role,
            "row_count": (
                len(sessions)
                if role == "cn_open_day_calendar"
                else 1
            ),
            "row_set_sha256": (
                _calendar_row_set_hash(calendar_rows)
                if role == "cn_open_day_calendar"
                else _sha(f"rows:{role}")
            ),
        }
        for role in PIT_ROLES
    ]
    admission_closure = hashlib.sha256(
        canonical_bytes(
            {
                "history_start": sessions[0],
                "decision_session": sessions[-1],
                "decision_cutoff": cutoff,
                "datasets": summaries,
            }
        )
    ).hexdigest()
    source_closure = hashlib.sha256(
        canonical_bytes(
            {
                "admission_closure_sha256": admission_closure,
                "dataset_refs": dataset_refs,
                "expected_key_inventory_refs": expected_refs,
            }
        )
    ).hexdigest()
    return seal_semantic(
        {
            "admission_closure_sha256": admission_closure,
            "authority": dict(NO_AUTHORITY),
            "catalog_id": "pit-catalog-portfolio",
            "cutoff": cutoff,
            "dataset_refs": dataset_refs,
            "dataset_summaries": summaries,
            "decision_session": sessions[-1],
            "expected_key_inventory_refs": expected_refs,
            "generation_id": "pit-generation-portfolio",
            "history_start": sessions[0],
            "protocol_version": "myquant.v17.v4",
            "source_closure_sha256": source_closure,
            "strategy_id": strategy_id,
            "version": "myquant.v17.v4.pit-generation-catalog.v1",
        }
    )


def _store(
    artifact: dict[str, Any],
    *,
    path: str,
    artifacts: dict[str, bytes],
) -> dict[str, str]:
    reference = portfolio_artifact_ref(artifact, relative_path=path)
    artifacts[reference["byte_sha256"]] = canonical_resource_bytes(
        artifact
    )
    return reference


@pytest.fixture(scope="module")
def deep_fixture() -> dict[str, Any]:
    rows, sessions, active_cutoff, cutoff, artifacts = _fixture()
    closure = _run(
        rows,
        sessions,
        active_cutoff,
        cutoff,
        artifacts,
    )
    loader = lambda reference: artifacts[reference["byte_sha256"]]
    inventory_ref = _store_v4_artifact(
        dict(closure.origin_inventory),
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/origins.json"
        ),
        artifacts=artifacts,
    )
    quant = build_calibration_receipt(
        closure,
        calibration_kind="QUANT_TIMING",
        receipt_id="quant-calibration-deep",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    fundamental = build_calibration_receipt(
        closure,
        calibration_kind="FUNDAMENTAL_FORWARD",
        receipt_id="fundamental-calibration-deep",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    quant_ref = _store(
        quant,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/quant-deep.json"
        ),
        artifacts=artifacts,
    )
    fundamental_ref = _store(
        fundamental,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/fundamental-deep.json"
        ),
        artifacts=artifacts,
    )
    promotion = build_fusion_promotion_receipt(
        closure,
        receipt_id="fusion-promotion-deep",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        quant_calibration_receipt_ref=quant_ref,
        fundamental_calibration_receipt_ref=fundamental_ref,
        artifact_loader=loader,
    )
    promotion_ref = _store(
        promotion,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/fusion-deep.json"
        ),
        artifacts=artifacts,
    )
    top24 = build_fusion_top24(
        [
            FusionTop24Input(
                symbol=f"{index:06d}.SZ",
                fused_score=str(25 - index),
                base_target="0.03",
            )
            for index in range(1, 25)
        ],
        output_id="fusion-top24-deep",
        run_id=closure.run_id,
        strategy_id=closure.strategy_id,
        cutoff=cutoff,
        created_at=cutoff,
        promotion_receipt_ref=promotion_ref,
        artifact_loader=loader,
    )
    top24_ref = _store(
        top24,
        path=(
            "data/private/v17_v4_runs/"
            "calibration-run-1/fusion-top24.json"
        ),
        artifacts=artifacts,
    )
    inputs: list[DeepEvidenceInput] = []
    for index in range(1, 25):
        symbol = f"{index:06d}.SZ"
        official = seal_semantic(
            {
                "authority": dict(NO_AUTHORITY),
                "available_at": cutoff,
                "content_sha256": _sha(f"content:{symbol}"),
                "cutoff": cutoff,
                "evidence_id": f"official-{index}",
                "evidence_kind": "FILING",
                "official_source_id": "cninfo",
                "protocol_version": "myquant.v17.v4",
                "published_at": cutoff,
                "strategy_id": closure.strategy_id,
                "symbol": symbol,
                "version": "myquant.v17.v4.official-evidence.v1",
            }
        )
        official_ref = _store(
            official,
            path=(
                "data/private/v17_v4_deep/"
                f"{symbol}/official.json"
            ),
            artifacts=artifacts,
        )
        dossier = seal_semantic(
            {
                "as_of": cutoff[:10],
                "authority": dict(NO_AUTHORITY),
                "created_at": cutoff,
                "cutoff": cutoff,
                "dossier_id": f"dossier-{index}",
                "official_evidence_refs": [official_ref],
                "protocol_version": "myquant.v17.v4",
                "strategy_id": closure.strategy_id,
                "summary_sha256": _sha(f"dossier:{symbol}"),
                "symbol": symbol,
                "version": "myquant.v17.v4.issuer-dossier.v1",
            }
        )
        dossier_ref = _store(
            dossier,
            path=(
                "data/private/v17_v4_deep/"
                f"{symbol}/dossier.json"
            ),
            artifacts=artifacts,
        )
        event = seal_semantic(
            {
                "as_of": cutoff[:10],
                "authority": dict(NO_AUTHORITY),
                "created_at": cutoff,
                "cutoff": cutoff,
                "flags": [],
                "official_evidence_refs": [official_ref],
                "protocol_version": "myquant.v17.v4",
                "scan_id": f"event-scan-{index}",
                "strategy_id": closure.strategy_id,
                "symbol": symbol,
                "version": "myquant.v17.v4.event-scan.v1",
            }
        )
        event_ref = _store(
            event,
            path=(
                "data/private/v17_v4_deep/"
                f"{symbol}/event-scan.json"
            ),
            artifacts=artifacts,
        )
        inputs.append(
            DeepEvidenceInput(
                symbol=symbol,
                status="COMPLETE",
                official_evidence_refs=(official_ref,),
                issuer_dossier_ref=dossier_ref,
                event_scan_ref=event_ref,
                signal="-0.5",
            )
        )
    return {
        "artifacts": artifacts,
        "cutoff": cutoff,
        "inputs": tuple(inputs),
        "top24": top24,
        "top24_ref": top24_ref,
    }


def test_top24_and_deep_bundle_close_exact_native_evidence(
    deep_fixture: dict[str, Any],
) -> None:
    artifacts = dict(deep_fixture["artifacts"])
    inputs = list(deep_fixture["inputs"])
    inputs[0] = DeepEvidenceInput(
        symbol=inputs[0].symbol,
        status="UNAVAILABLE",
        buy_veto=True,
        reason="official_evidence_unavailable",
    )
    loader = lambda reference: artifacts[reference["byte_sha256"]]
    bundle = build_deep_evidence_bundle(
        inputs,
        bundle_id="deep-bundle-1",
        fusion_top24_ref=deep_fixture["top24_ref"],
        created_at=deep_fixture["cutoff"],
        artifact_loader=loader,
    )

    assert deep_fixture["top24"]["version"] == FUSION_TOP24_VERSION
    assert bundle["version"] == DEEP_BUNDLE_VERSION
    assert len(bundle["rows"]) == 24
    assert bundle["rows"][0]["status"] == "UNAVAILABLE"
    assert bundle["rows"][0]["buy_veto"] is True
    assert bundle["rows"][0]["target_after_deep"] == "0"
    assert bundle["rows"][1]["target_after_deep"] == "0.0285"
    validate_artifact(deep_fixture["top24"])
    validate_artifact(bundle)


def test_deep_omission_staleness_and_readback_drift_fail_closed(
    deep_fixture: dict[str, Any],
) -> None:
    artifacts = dict(deep_fixture["artifacts"])
    loader = lambda reference: artifacts[reference["byte_sha256"]]
    with pytest.raises(DeepClosureError, match="deep_top24_row_count"):
        build_deep_evidence_bundle(
            deep_fixture["inputs"][:-1],
            bundle_id="deep-bundle-missing",
            fusion_top24_ref=deep_fixture["top24_ref"],
            created_at=deep_fixture["cutoff"],
            artifact_loader=loader,
        )

    stale_inputs = list(deepcopy(deep_fixture["inputs"]))
    first = stale_inputs[0]
    dossier_ref = dict(first.issuer_dossier_ref or {})
    dossier_raw = artifacts[dossier_ref["byte_sha256"]]
    dossier = __import__("json").loads(dossier_raw)
    dossier.pop("semantic_sha256")
    dossier["as_of"] = "1999-01-01"
    dossier = seal_semantic(dossier)
    stale_dossier_ref = _store(
        dossier,
        path=(
            "data/private/v17_v4_deep/"
            f"{first.symbol}/stale-dossier.json"
        ),
        artifacts=artifacts,
    )
    stale_inputs[0] = DeepEvidenceInput(
        **{
            **first.__dict__,
            "issuer_dossier_ref": stale_dossier_ref,
        }
    )
    with pytest.raises(
        DeepClosureError,
        match="deep_freshness_or_lineage",
    ):
        build_deep_evidence_bundle(
            stale_inputs,
            bundle_id="deep-bundle-stale",
            fusion_top24_ref=deep_fixture["top24_ref"],
            created_at=deep_fixture["cutoff"],
            artifact_loader=loader,
        )

    drifted = dict(artifacts)
    official_ref = first.official_evidence_refs[0]
    drifted[official_ref["byte_sha256"]] += b" "
    with pytest.raises(DeepClosureError, match="official_evidence\\[0\\]_byte_sha"):
        build_deep_evidence_bundle(
            deep_fixture["inputs"],
            bundle_id="deep-bundle-drift",
            fusion_top24_ref=deep_fixture["top24_ref"],
            created_at=deep_fixture["cutoff"],
            artifact_loader=lambda reference: drifted[
                reference["byte_sha256"]
            ],
        )


def test_deep_symbol_or_caller_target_forgery_is_not_accepted(
    deep_fixture: dict[str, Any],
) -> None:
    artifacts = dict(deep_fixture["artifacts"])
    inputs = list(deepcopy(deep_fixture["inputs"]))
    first = inputs[0]
    inputs[0] = DeepEvidenceInput(
        **{
            **first.__dict__,
            "symbol": "999999.SZ",
        }
    )
    with pytest.raises(DeepClosureError, match="deep_top24_exact_domain"):
        build_deep_evidence_bundle(
            inputs,
            bundle_id="deep-bundle-forged",
            fusion_top24_ref=deep_fixture["top24_ref"],
            created_at=deep_fixture["cutoff"],
            artifact_loader=lambda reference: artifacts[
                reference["byte_sha256"]
            ],
        )


def test_holdings_aware_portfolio_applies_two_monotonic_overlays(
    deep_fixture: dict[str, Any],
) -> None:
    artifacts = dict(deep_fixture["artifacts"])
    loader = lambda reference: artifacts[reference["byte_sha256"]]
    cutoff = deep_fixture["cutoff"]
    strategy_id = deep_fixture["top24"]["strategy_id"]
    run_id = deep_fixture["top24"]["run_id"]
    sessions = _sessions_ending("2021-01-19", 2520)

    deep = build_deep_evidence_bundle(
        deep_fixture["inputs"],
        bundle_id="deep-bundle-portfolio",
        fusion_top24_ref=deep_fixture["top24_ref"],
        created_at=cutoff,
        artifact_loader=loader,
    )
    deep_ref = _store(
        deep,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/deep-portfolio.json"
        ),
        artifacts=artifacts,
    )
    holdings = build_holdings_snapshot(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        as_of_session=sessions[-2],
        available_at=cutoff,
        nav="100",
        cash="90",
        positions=(
            HoldingInput(symbol="000001.SZ", market_value="10"),
        ),
    )
    holdings_ref = _store(
        holdings,
        path=(
            "data/private/v17_v4_sources/"
            f"holdings/{run_id}.json"
        ),
        artifacts=artifacts,
    )
    risk = build_risk_policy(
        strategy_id=strategy_id,
        cutoff=cutoff,
        effective_from="2021-01-01T00:00:00Z",
        expires_at="2022-01-01T00:00:00Z",
        gross_cap="0.80",
        cash_floor="0.20",
        single_name_cap="0.20",
        industry_cap="1",
        cluster_cap="1",
        turnover_cap="1",
    )
    risk_ref = _store(
        risk,
        path=(
            "data/private/v17_v4_sources/"
            f"risk/{run_id}.json"
        ),
        artifacts=artifacts,
    )
    calendar_rows = _calendar_rows(sessions, available_at=cutoff)
    calendar_raw = canonical_resource_bytes({"rows": calendar_rows})
    calendar_sha256 = hashlib.sha256(calendar_raw).hexdigest()
    artifacts[calendar_sha256] = calendar_raw
    calendar_ref = {
        "artifact_id": "calendar-cn-2019",
        "artifact_version": (
            "myquant.v17.v4.dataset.cn_open_day_calendar.v1"
        ),
        "byte_sha256": calendar_sha256,
        "cutoff": cutoff,
        "relative_path": (
            "data/private/v17_v4_sources/calendar/cn-2019.json"
        ),
        "semantic_sha256": _sha("calendar-semantic"),
        "strategy_id": strategy_id,
    }
    catalog = _pit_catalog(
        strategy_id=strategy_id,
        cutoff=cutoff,
        sessions=sessions,
        calendar_rows=calendar_rows,
        calendar_ref=calendar_ref,
    )
    catalog_ref = _store(
        catalog,
        path=(
            "data/private/v17_v4_sources/"
            "pit_catalog/generations/pit-generation-portfolio.json"
        ),
        artifacts=artifacts,
    )
    fusion_symbols = tuple(
        row["symbol"] for row in deep_fixture["top24"]["rows"]
    )
    decisions = tuple(
        PermissionInput(
            symbol=f"{index:06d}.SZ",
            can_buy=True,
            can_sell=True,
            industry=f"industry-{index}",
            cluster=f"cluster-{index}",
        )
        for index in range(1, 25)
    )
    permissions = build_pretrade_permissions(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        decision_session=sessions[-1],
        canonical_calendar_ref=calendar_ref,
        pit_catalog_ref=catalog_ref,
        holdings_snapshot_ref=holdings_ref,
        risk_policy_ref=risk_ref,
        decisions=decisions,
        fusion_symbols=fusion_symbols,
        artifact_loader=loader,
    )
    permissions_ref = _store(
        permissions,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/permissions.json"
        ),
        artifacts=artifacts,
    )
    stale_holdings = build_holdings_snapshot(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        as_of_session="2000-01-03",
        available_at=cutoff,
        nav="100",
        cash="90",
        positions=(
            HoldingInput(symbol="000001.SZ", market_value="10"),
        ),
    )
    stale_holdings_ref = _store(
        stale_holdings,
        path=(
            "data/private/v17_v4_sources/"
            f"holdings/{run_id}-stale.json"
        ),
        artifacts=artifacts,
    )
    sparse_sessions = [
        *_sessions_ending("2000-01-03", 2519),
        sessions[-1],
    ]
    sparse_calendar_rows = _calendar_rows(
        sparse_sessions,
        available_at=cutoff,
    )
    sparse_calendar_raw = canonical_resource_bytes(
        {"rows": sparse_calendar_rows}
    )
    sparse_calendar_sha = hashlib.sha256(
        sparse_calendar_raw
    ).hexdigest()
    artifacts[sparse_calendar_sha] = sparse_calendar_raw
    sparse_calendar_ref = {
        **calendar_ref,
        "artifact_id": "calendar-cn-sparse",
        "byte_sha256": sparse_calendar_sha,
        "relative_path": (
            "data/private/v17_v4_sources/calendar/cn-sparse.json"
        ),
        "semantic_sha256": _sha("calendar-sparse-semantic"),
    }
    sparse_catalog = _pit_catalog(
        strategy_id=strategy_id,
        cutoff=cutoff,
        sessions=sparse_sessions,
        calendar_rows=sparse_calendar_rows,
        calendar_ref=sparse_calendar_ref,
    )
    sparse_catalog_ref = _store(
        sparse_catalog,
        path=(
            "data/private/v17_v4_sources/"
            "pit_catalog/generations/pit-generation-sparse.json"
        ),
        artifacts=artifacts,
    )
    with pytest.raises(
        PortfolioControlError,
        match="canonical_calendar_session_inventory",
    ):
        build_pretrade_permissions(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            decision_session=sessions[-1],
            canonical_calendar_ref=sparse_calendar_ref,
            pit_catalog_ref=sparse_catalog_ref,
            holdings_snapshot_ref=stale_holdings_ref,
            risk_policy_ref=risk_ref,
            decisions=decisions,
            fusion_symbols=fusion_symbols,
            artifact_loader=loader,
        )
    macro_evidence = build_regime_evidence(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        role="macro_evidence",
        available_at=cutoff,
        gross_multiplier="0.90",
    )
    macro_evidence_ref = _store(
        macro_evidence,
        path=(
            "data/private/v17_v4_sources/"
            f"regime/{run_id}-macro.json"
        ),
        artifacts=artifacts,
    )
    no_sell_decisions = tuple(
        PermissionInput(
            symbol=item.symbol,
            can_buy=item.can_buy,
            can_sell=False if item.symbol == "000001.SZ" else item.can_sell,
            industry=item.industry,
            cluster=item.cluster,
        )
        for item in decisions
    )
    no_sell_permissions = build_pretrade_permissions(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        decision_session=sessions[-1],
        canonical_calendar_ref=calendar_ref,
        pit_catalog_ref=catalog_ref,
        holdings_snapshot_ref=holdings_ref,
        risk_policy_ref=risk_ref,
        decisions=no_sell_decisions,
        fusion_symbols=fusion_symbols,
        artifact_loader=loader,
    )
    no_sell_permissions_ref = _store(
        no_sell_permissions,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/no-sell-permissions.json"
        ),
        artifacts=artifacts,
    )
    with pytest.raises(
        PortfolioControlError,
        match="sell_not_permitted:000001.SZ",
    ):
        build_macro_overlay(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            deep_bundle_ref=deep_ref,
            permissions_ref=no_sell_permissions_ref,
            risk_policy_ref=risk_ref,
            evidence_refs=(macro_evidence_ref,),
            artifact_loader=loader,
        )
    outside_holdings = build_holdings_snapshot(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        as_of_session=sessions[-2],
        available_at=cutoff,
        nav="100",
        cash="85",
        positions=(
            HoldingInput(symbol="000001.SZ", market_value="10"),
            HoldingInput(symbol="600000.SH", market_value="5"),
        ),
    )
    outside_holdings_ref = _store(
        outside_holdings,
        path=(
            "data/private/v17_v4_sources/"
            f"holdings/{run_id}-outside.json"
        ),
        artifacts=artifacts,
    )
    permissive_risk = build_risk_policy(
        strategy_id=strategy_id,
        cutoff=cutoff,
        effective_from="2021-01-01T00:00:00Z",
        expires_at="2022-01-01T00:00:00Z",
        gross_cap="1",
        cash_floor="0",
        single_name_cap="0.20",
        industry_cap="1",
        cluster_cap="1",
        turnover_cap="1",
    )
    permissive_risk_ref = _store(
        permissive_risk,
        path=(
            "data/private/v17_v4_sources/"
            f"risk/{run_id}-permissive.json"
        ),
        artifacts=artifacts,
    )
    outside_decisions = (
        *decisions,
        PermissionInput(
            symbol="600000.SH",
            can_buy=False,
            can_sell=False,
            industry="industry-outside",
            cluster="cluster-outside",
        ),
    )
    outside_permissions = build_pretrade_permissions(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        decision_session=sessions[-1],
        canonical_calendar_ref=calendar_ref,
        pit_catalog_ref=catalog_ref,
        holdings_snapshot_ref=outside_holdings_ref,
        risk_policy_ref=permissive_risk_ref,
        decisions=outside_decisions,
        fusion_symbols=fusion_symbols,
        artifact_loader=loader,
    )
    outside_permissions_ref = _store(
        outside_permissions,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/outside-no-sell-permissions.json"
        ),
        artifacts=artifacts,
    )
    with pytest.raises(
        PortfolioControlError,
        match="sell_not_permitted:600000.SH",
    ):
        build_macro_overlay(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            deep_bundle_ref=deep_ref,
            permissions_ref=outside_permissions_ref,
            risk_policy_ref=permissive_risk_ref,
            evidence_refs=(macro_evidence_ref,),
            artifact_loader=loader,
        )
    forged_deep = deepcopy(deep)
    forged_deep.pop("semantic_sha256")
    forged_deep["rows"][0]["target_after_deep"] = "0.03"
    forged_deep = seal_semantic(forged_deep)
    forged_deep_ref = _store(
        forged_deep,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/forged-deep.json"
        ),
        artifacts=artifacts,
    )
    with pytest.raises(
        DeepClosureError,
        match="deep_bundle_row_replay",
    ):
        build_macro_overlay(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            deep_bundle_ref=forged_deep_ref,
            permissions_ref=permissions_ref,
            risk_policy_ref=risk_ref,
            evidence_refs=(macro_evidence_ref,),
            artifact_loader=loader,
        )
    forged_permissions = deepcopy(permissions)
    forged_permissions.pop("semantic_sha256")
    forged_permissions["payload"][0]["current_target"] = "0.2"
    forged_permissions = seal_semantic(forged_permissions)
    forged_permissions_ref = _store(
        forged_permissions,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/forged-permissions.json"
        ),
        artifacts=artifacts,
    )
    with pytest.raises(
        PortfolioControlError,
        match="permissions_holdings_reconciliation",
    ):
        build_macro_overlay(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            deep_bundle_ref=deep_ref,
            permissions_ref=forged_permissions_ref,
            risk_policy_ref=risk_ref,
            evidence_refs=(macro_evidence_ref,),
            artifact_loader=loader,
        )
    calendar_drift = dict(artifacts)
    calendar_drift[calendar_sha256] += b" "
    with pytest.raises(
        PortfolioControlError,
        match="canonical_calendar_byte_sha",
    ):
        build_macro_overlay(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            deep_bundle_ref=deep_ref,
            permissions_ref=permissions_ref,
            risk_policy_ref=risk_ref,
            evidence_refs=(macro_evidence_ref,),
            artifact_loader=lambda reference: calendar_drift[
                reference["byte_sha256"]
            ],
        )
    with pytest.raises(
        PortfolioControlError,
        match="macro_evidence_unavailable",
    ):
        build_macro_overlay(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            deep_bundle_ref=deep_ref,
            permissions_ref=permissions_ref,
            risk_policy_ref=risk_ref,
            evidence_refs=(),
            artifact_loader=loader,
        )
    macro = build_macro_overlay(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        deep_bundle_ref=deep_ref,
        permissions_ref=permissions_ref,
        risk_policy_ref=risk_ref,
        evidence_refs=(macro_evidence_ref,),
        artifact_loader=loader,
    )
    macro_ref = _store(
        macro,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/macro-overlay.json"
        ),
        artifacts=artifacts,
    )
    markov_evidence = build_regime_evidence(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        role="markov_evidence",
        available_at=cutoff,
        gross_multiplier="0.80",
    )
    markov_evidence_ref = _store(
        markov_evidence,
        path=(
            "data/private/v17_v4_sources/"
            f"regime/{run_id}-markov.json"
        ),
        artifacts=artifacts,
    )
    markov = build_markov_overlay(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        macro_overlay_ref=macro_ref,
        permissions_ref=permissions_ref,
        risk_policy_ref=risk_ref,
        evidence_refs=(markov_evidence_ref,),
        artifact_loader=loader,
    )
    markov_ref = _store(
        markov,
        path=(
            "data/private/v17_v4_runs/"
            f"{run_id}/markov-overlay.json"
        ),
        artifacts=artifacts,
    )
    output = build_production_portfolio(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        fusion_top24_ref=deep_fixture["top24_ref"],
        deep_bundle_ref=deep_ref,
        holdings_snapshot_ref=holdings_ref,
        permissions_ref=permissions_ref,
        risk_policy_ref=risk_ref,
        macro_overlay_ref=macro_ref,
        markov_overlay_ref=markov_ref,
        artifact_loader=loader,
    )

    assert permissions["portfolio_basis"] == "HOLDINGS_AWARE"
    assert permissions["holdings_snapshot_age_sessions"] == 1
    assert macro["status"] == markov["status"] == "APPLIED"
    assert Decimal(macro["output_gross"]) < Decimal(macro["input_gross"])
    assert Decimal(markov["output_gross"]) < Decimal(markov["input_gross"])
    assert output["status"] == "COMPLETE"
    assert output["portfolio_basis"] == "HOLDINGS_AWARE"
    assert Decimal(output["gross_weight"]) + Decimal(output["cash_weight"]) == 1
    validate_artifact(output)
    drifted = dict(artifacts)
    drifted[markov_ref["byte_sha256"]] += b" "
    with pytest.raises(
        PortfolioControlError,
        match="markov_byte_sha",
    ):
        build_production_portfolio(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            fusion_top24_ref=deep_fixture["top24_ref"],
            deep_bundle_ref=deep_ref,
            holdings_snapshot_ref=holdings_ref,
            permissions_ref=permissions_ref,
            risk_policy_ref=risk_ref,
            macro_overlay_ref=macro_ref,
            markov_overlay_ref=markov_ref,
            artifact_loader=lambda reference: drifted[
                reference["byte_sha256"]
            ],
        )
