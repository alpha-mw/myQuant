from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quant_investor.v17.holdings import (
    build_available_holdings_snapshot,
    build_unavailable_holdings_snapshot,
)
from quant_investor.v17.pretrade import build_execution_cost_policy
from quant_investor.v17.risk_policy import (
    build_available_risk_policy_snapshot,
    build_unavailable_risk_policy_snapshot,
)
from quant_investor.v17.runtime_pipeline import (
    REGIME_INPUT_SOURCE_VERSION,
    REGIME_MAPPING_SOURCE_VERSION,
)
from quant_investor.v17.semantic import canonical_json_bytes, seal_semantic
from quant_investor.v17.source_bindings import (
    REQUIRED_SOURCE_ROLES,
    SOURCE_PLAN_VERSION,
    V17SourceBindingError,
    load_source_manifest_binding,
)
from quant_investor.v17.source_maintain import maintain_sources
from quant_investor.v17.state_machine import EMPTY_SHA

CUTOFF = "2026-07-22T00:00:00Z"


def _write(path: Path, payload: dict[str, object]) -> str:
    raw = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _holdings(*, available: bool) -> dict[str, object]:
    if not available:
        return build_unavailable_holdings_snapshot(
            snapshot_id="holdings-missing",
            strategy_id="cn-shadow",
            market="CN",
            reason="not_supplied",
        )
    return build_available_holdings_snapshot(
        snapshot_id="holdings-cash",
        strategy_id="cn-shadow",
        market="CN",
        pit_cutoff="2026-07-21T00:00:00Z",
        as_of="2026-07-21T00:00:00Z",
        nav=1_000_000.0,
        cash=1_000_000.0,
        declared_all_cash=True,
        positions=[],
    )


def _risk(*, available: bool) -> dict[str, object]:
    if not available:
        return build_unavailable_risk_policy_snapshot(
            policy_id="risk-missing",
            strategy_id="cn-shadow",
            market="CN",
            reason="not_supplied",
        )
    return build_available_risk_policy_snapshot(
        policy_id="risk-v1",
        strategy_id="cn-shadow",
        market="CN",
        pit_cutoff="2026-07-21T00:00:00Z",
        as_of="2026-07-21T00:00:00Z",
        expires_at="2026-07-24T00:00:00Z",
        gross_cap=0.8,
        cash_floor=0.2,
        single_name_cap=0.1,
        industry_cap=0.3,
        cluster_cap=0.2,
        beta_cap=1.2,
        stress_loss_cap=0.15,
        adv20_participation_cap=0.05,
        turnover_cap=0.3,
        stress_scenario="stress-v1",
        source_refs=[
            {
                "source_id": "owner",
                "path": "private/owner.json",
                "byte_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            }
        ],
    )


def _regime_input(name: str, state: str) -> dict[str, object]:
    if state == "DISABLED":
        values: dict[str, object] = {
            "availability": None,
            "state": None,
            "reason": None,
            "enabled": False,
        }
    elif state == "UNAVAILABLE":
        values = {
            "availability": "UNAVAILABLE",
            "state": None,
            "reason": "canonical_state_missing",
            "enabled": True,
        }
    else:
        values = {
            "availability": "AVAILABLE",
            "state": state,
            "reason": None,
            "enabled": True,
        }
    return seal_semantic(
        {
            "version": REGIME_INPUT_SOURCE_VERSION,
            "market": "CN",
            "cutoff": CUTOFF,
            "name": name,
            **values,
            "authority": False,
        }
    )


def _build_plan(
    tmp_path: Path,
    *,
    manifest_id: str,
    holdings_available: bool = True,
    risk_available: bool = True,
    macro_state: str = "RISK_ON",
    macro_mapping_available: bool = True,
) -> dict[str, object]:
    payloads: dict[str, dict[str, object]] = {
        role: seal_semantic({"role": role, "authority": False}) for role in REQUIRED_SOURCE_ROLES
    }
    payloads["holdings"] = _holdings(available=holdings_available)
    payloads["risk_policy"] = _risk(available=risk_available)
    payloads["execution_cost_policy"] = build_execution_cost_policy(
        buy_commission=0.0003,
        sell_commission=0.0003,
        sell_stamp_tax=0.0005,
        buy_transfer_fee=0.00001,
        sell_transfer_fee=0.00001,
        buy_slippage=0.001,
        sell_slippage=0.0012,
        market_impact=0.0004,
    )
    payloads["macro_overlay_input"] = _regime_input("macro", macro_state)
    payloads["markov_overlay_input"] = _regime_input("markov", "DISABLED")
    for name in ("macro", "markov"):
        payloads[f"{name}_overlay_mapping"] = seal_semantic(
            {
                "version": REGIME_MAPPING_SOURCE_VERSION,
                "market": "CN",
                "cutoff": CUTOFF,
                "name": name,
                "states": {
                    "RISK_ON": {"gross_cap": 0.8, "cash_floor": 0.2},
                },
                "authority": False,
            }
        )

    sources: list[dict[str, object]] = []
    source_root = tmp_path / "inputs" / manifest_id
    for role in sorted(REQUIRED_SOURCE_ROLES):
        if role == "macro_overlay_mapping" and not macro_mapping_available:
            sources.append(
                {
                    "source_id": f"src-{role}",
                    "role": role,
                    "availability": "UNAVAILABLE",
                    "reason": "mapping_disabled_by_owner",
                }
            )
            continue
        path = source_root / f"{role}.json"
        sha = _write(path, payloads[role])
        sources.append(
            {
                "source_id": f"src-{role}",
                "role": role,
                "availability": "AVAILABLE",
                "path": str(path),
                "expected_byte_sha256": sha,
            }
        )
    return seal_semantic(
        {
            "version": SOURCE_PLAN_VERSION,
            "manifest_id": manifest_id,
            "market": "CN",
            "cutoff": CUTOFF,
            "created_at": "2026-07-22T00:01:00Z",
            "sources": sources,
            "authority": False,
        }
    )


def test_source_manifest_is_content_addressed_and_nonempty_cas_checks_plan_identity(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path, manifest_id="source-set-1")
    manifest, path, sha = maintain_sources(
        tmp_path,
        plan,
        expected_manifest_sha256=EMPTY_SHA,
    )
    assert path.name == "source-set-1.json"
    assert len(manifest["sources"]) == len(REQUIRED_SOURCE_ROLES)
    bundle = load_source_manifest_binding(
        tmp_path,
        path,
        expected_manifest_sha256=sha,
    )
    assert bundle.rank_unavailable_roles == ()
    assert bundle.portfolio_unavailable_roles == ()

    repeated = maintain_sources(tmp_path, plan, expected_manifest_sha256=sha)
    assert repeated[2] == sha

    changed = _build_plan(tmp_path, manifest_id="source-set-1")
    changed["created_at"] = "2026-07-22T00:02:00Z"
    changed = seal_semantic(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17SourceBindingError, match="does not match plan"):
        maintain_sources(tmp_path, changed, expected_manifest_sha256=sha)


@pytest.mark.parametrize("unavailable_role", ["holdings", "risk_policy"])
def test_embedded_unavailable_portfolio_payload_is_not_interpreted_as_empty(
    tmp_path: Path,
    unavailable_role: str,
) -> None:
    plan = _build_plan(
        tmp_path,
        manifest_id=f"source-set-{unavailable_role}",
        holdings_available=unavailable_role != "holdings",
        risk_available=unavailable_role != "risk_policy",
    )
    _, path, sha = maintain_sources(tmp_path, plan, expected_manifest_sha256=EMPTY_SHA)
    bundle = load_source_manifest_binding(
        tmp_path,
        path,
        expected_manifest_sha256=sha,
    )
    assert unavailable_role in bundle.portfolio_unavailable_roles


def test_disabled_overlay_does_not_require_mapping_but_enabled_unavailable_blocks(
    tmp_path: Path,
) -> None:
    disabled = _build_plan(
        tmp_path,
        manifest_id="source-set-disabled",
        macro_state="DISABLED",
        macro_mapping_available=False,
    )
    _, disabled_path, disabled_sha = maintain_sources(
        tmp_path, disabled, expected_manifest_sha256=EMPTY_SHA
    )
    bundle = load_source_manifest_binding(
        tmp_path,
        disabled_path,
        expected_manifest_sha256=disabled_sha,
    )
    assert "macro_overlay_mapping" not in bundle.portfolio_unavailable_roles

    unavailable = _build_plan(
        tmp_path,
        manifest_id="source-set-unavailable",
        macro_state="UNAVAILABLE",
    )
    _, unavailable_path, unavailable_sha = maintain_sources(
        tmp_path, unavailable, expected_manifest_sha256=EMPTY_SHA
    )
    bundle = load_source_manifest_binding(
        tmp_path,
        unavailable_path,
        expected_manifest_sha256=unavailable_sha,
    )
    assert "macro_overlay_input" in bundle.portfolio_unavailable_roles
