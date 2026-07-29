from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors.governance_literature_incubator_v4 import (
    candidate_catalog_v4,
    evaluate_candidate_v4,
)
from quant_investor.v17_v4_contract import canonical_bytes, seal_semantic
from quant_investor.v17_v4_runtime.shadow_prepare_forward import (
    ForwardShadowPreparationError,
    SOURCE_ROLES,
    artifact_ref,
    build_quant_first_forward_shadow,
    classify_current_canonical_preflight,
    preflight_current_canonical_sources,
)

CUTOFF = "2026-07-28T07:00:00Z"
SESSION = "2026-07-28"
STRATEGY = "cn-aggressive-tech-manufacturing"
RUN_ID = "forward-shadow-1"
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _write(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


def _write_json(path: Path, value: dict[str, Any]) -> str:
    return _write(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def _source_fixture(root: Path) -> dict[str, Any]:
    run = root / "results/strategy_records/CN/aggressive/20260728_1600"
    v15_path = run / "manifest.json"
    v15_sha = _write_json(v15_path, {"analysis_trade_date": "20260728"})

    ledger_path = run / "ledger_after_manual_switch.parquet"
    ledger_raw = _parquet_bytes(
        pd.DataFrame(
            [
                {
                    "current_price": 2.0,
                    "current_value": 20.0,
                    "shares": 10,
                    "symbol": "000001.SZ",
                    "trade_date": SESSION,
                }
            ]
        )
    )
    ledger_sha = _write(ledger_path, ledger_raw)
    manual = {
        "capital_cny": 100.0,
        "cash_after": 80.0,
        "effective_manual_holding_count": 1,
        "effective_manual_ledger_path": ledger_path.name,
        "ledger_after_manual_switch_parquet": ledger_path.name,
        "ledger_after_manual_switch_parquet_sha256": ledger_sha,
        "ledger_provenance": {
            "declared_next_ledger_path": ledger_path.name,
            "declared_sha256": ledger_sha,
            "parquet_sha256": ledger_sha,
        },
        "market_value_after": 20.0,
        "next_ledger_path": ledger_path.name,
        "next_ledger_sha256": ledger_sha,
        "portfolio_pnl_after": 0.0,
        "total_value_after": 100.0,
    }
    manual_path = run / "manual_execution_manifest.json"
    manual_sha = _write_json(manual_path, manual)

    pit_membership_path = (
        root / "data/parquet/cn/reference/_generations/pit-current/membership.parquet"
    )
    pit_membership_sha = _write(
        pit_membership_path,
        _parquet_bytes(pd.DataFrame({"symbol": ["000001.SZ"]})),
    )
    pit_manifest_path = pit_membership_path.parent / "manifest.json"
    pit_manifest_sha = _write_json(
        pit_manifest_path,
        {
            "canonical_sha256": pit_membership_sha,
            "generation_id": "pit-current",
        },
    )
    market_manifest_path = root / "data/parquet/cn/_snapshots/current.json"
    market_manifest = {
        "coverage": {
            "pit_generation_id": "pit-current",
            "pit_generation_manifest_path": str(pit_manifest_path),
            "pit_generation_manifest_sha256": pit_manifest_sha,
            "pit_membership_path": str(pit_membership_path),
            "pit_membership_sha256": pit_membership_sha,
        },
        "latest_complete_trade_date": "20260728",
        "readback_validated": True,
        "snapshot_id": "current",
        "status": "OK",
    }
    market_manifest_sha = _write_json(market_manifest_path, market_manifest)
    market_pointer_path = root / "data/parquet/cn/_latest.json"
    market_pointer = {
        "blockers": [],
        "coverage": {"complete": True},
        "latest_complete_trade_date": "20260728",
        "manifest_path": str(market_manifest_path),
        "snapshot_id": "current",
        "status": "OK",
    }
    market_pointer_sha = _write_json(market_pointer_path, market_pointer)

    fundamental_manifest_path = (
        root / "data/parquet/cn/_fundamental_generations/fundamental-current/manifest.json"
    )
    fundamental_manifest_sha = _write_json(
        fundamental_manifest_path,
        {"generation_id": "fundamental-current", "status": "OK"},
    )
    fundamental_pointer_path = root / "data/parquet/cn/_fundamental_latest.json"
    fundamental_pointer_sha = _write_json(
        fundamental_pointer_path,
        {
            "generation_id": "fundamental-current",
            "manifest_path": ("_fundamental_generations/fundamental-current/manifest.json"),
            "metadata": {"gate2_passed": True},
            "primary_provenance": {"status": "verified_live_tushare"},
            "status": "OK",
        },
    )

    macro_calendar_path = root / "data/parquet/cn/macro_release_calendar/_latest.json"
    macro_calendar_sha = _write_json(
        macro_calendar_path,
        {
            "generation_id": "calendar-current",
            "manifest_sha256": "9" * 64,
            "schema_version": "macro-release-calendar-pointer.v1",
        },
    )
    macro_manifest_path = (
        root / "data/parquet/cn/macro_daily/_generations/macro-current/manifest.json"
    )
    macro_manifest_sha = _write_json(
        macro_manifest_path,
        {
            "as_of": SESSION,
            "generation_id": "macro-current",
            "macro_release_calendar_generation": {
                "macro_release_calendar_generation_id": "calendar-current",
                "manifest_sha256": "9" * 64,
                "pointer_sha256": macro_calendar_sha,
            },
            "production_eligible": True,
        },
    )
    macro_pointer_path = root / "data/parquet/cn/_catalog.json"
    macro_pointer_sha = _write_json(
        macro_pointer_path,
        {
            "tables": {
                "macro_daily": {
                    "generation_id": "macro-current",
                    "generation_manifest_sha256": macro_manifest_sha,
                    "latest_date": "20260728",
                }
            }
        },
    )

    universe_path = run / "strategy_universe.parquet"
    universe_sha = _write(
        universe_path,
        _parquet_bytes(
            pd.DataFrame(
                {
                    "available_at": [CUTOFF] * 24,
                    "symbol": [f"{index:06d}.SZ" for index in range(1, 25)],
                    "trade_date": [SESSION] * 24,
                }
            )
        ),
    )
    return {
        "cutoff": CUTOFF,
        "decision_session": SESSION,
        "fundamental_manifest_path": str(fundamental_manifest_path),
        "fundamental_manifest_sha256": fundamental_manifest_sha,
        "fundamental_pointer_path": str(fundamental_pointer_path),
        "fundamental_pointer_sha256": fundamental_pointer_sha,
        "locator_id": "locator-forward-1",
        "macro_manifest_path": str(macro_manifest_path),
        "macro_manifest_sha256": macro_manifest_sha,
        "macro_pointer_path": str(macro_pointer_path),
        "macro_pointer_sha256": macro_pointer_sha,
        "macro_release_calendar_path": str(macro_calendar_path),
        "macro_release_calendar_sha256": macro_calendar_sha,
        "manual_execution_manifest_path": str(manual_path),
        "manual_execution_manifest_sha256": manual_sha,
        "market_manifest_path": str(market_manifest_path),
        "market_manifest_sha256": market_manifest_sha,
        "market_pointer_path": str(market_pointer_path),
        "market_pointer_sha256": market_pointer_sha,
        "pit_manifest_path": str(pit_manifest_path),
        "pit_manifest_sha256": pit_manifest_sha,
        "pit_membership_path": str(pit_membership_path),
        "pit_membership_sha256": pit_membership_sha,
        "strategy_id": STRATEGY,
        "strategy_universe_path": str(universe_path),
        "strategy_universe_sha256": universe_sha,
        "v15_run_manifest_path": str(v15_path),
        "v15_run_manifest_sha256": v15_sha,
    }


def test_preflight_binds_all_exact_sources_and_contained_ledger(
    tmp_path: Path,
) -> None:
    arguments = _source_fixture(tmp_path)
    result = preflight_current_canonical_sources(str(tmp_path.resolve()), **arguments)
    assert [row["role"] for row in result.source_locator["source_refs"]] == list(SOURCE_ROLES)
    assert result.source_locator["provider_calls_performed"] is False
    assert result.source_locator["maintenance_calls_performed"] is False
    assert result.source_locator["formal_activation_eligible"] is False
    assert result.source_locator["canary_evidence_eligible"] is False
    assert result.source_locator["performance_evidence_eligible"] is False


def test_preflight_rejects_source_sha_tamper(tmp_path: Path) -> None:
    arguments = _source_fixture(tmp_path)
    arguments["market_pointer_sha256"] = "0" * 64
    with pytest.raises(ForwardShadowPreparationError, match="sha256_mismatch"):
        preflight_current_canonical_sources(str(tmp_path.resolve()), **arguments)


def test_preflight_classifies_stale_current_input_gap(tmp_path: Path) -> None:
    arguments = _source_fixture(tmp_path)
    pointer_path = Path(arguments["market_pointer_path"])
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["latest_complete_trade_date"] = "20260727"
    arguments["market_pointer_sha256"] = _write_json(pointer_path, pointer)
    result = classify_current_canonical_preflight(
        str(tmp_path.resolve()),
        **arguments,
    )
    assert result == {
        "maintenance_calls_performed": False,
        "provider_calls_performed": False,
        "reason": "market_pointer_stale_or_blocked",
        "status": "TRUE_CURRENT_CANONICAL_INPUT_GAP",
    }


def test_preflight_rejects_ambiguous_declared_ledger(tmp_path: Path) -> None:
    arguments = _source_fixture(tmp_path)
    manual_path = Path(arguments["manual_execution_manifest_path"])
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    manual["next_ledger_path"] = "different.parquet"
    arguments["manual_execution_manifest_sha256"] = _write_json(manual_path, manual)
    with pytest.raises(ForwardShadowPreparationError, match="ledger_path_ambiguous"):
        preflight_current_canonical_sources(str(tmp_path.resolve()), **arguments)


def _dynamic_controls() -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, str],
]:
    catalog = candidate_catalog_v4()
    catalog_row = next(row for row in catalog if row["name"] == "cn_low_total_skewness_20d")
    selected = {
        "definition": catalog_row["definition"],
        "definition_sha256": catalog_row["definition_sha256"],
        "direction": format(catalog_row["direction"], ".1f"),
        "family": catalog_row["family"],
        "implementation": catalog_row["implementation"],
        "implementation_resource_sha256": "4" * 64,
        "implementation_sha256": "5" * 64,
        "lookback": catalog_row["lookback"],
        "name": catalog_row["name"],
        "params": catalog_row["params"],
        "required_fields": catalog_row["required_fields"],
        "selection_gates": [{"gate": f"gate-{index}", "passed": True} for index in range(1, 5)],
        "selection_score": 4,
        "slot": catalog_row["slot"],
    }
    locator = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "cutoff": CUTOFF,
            "formal_activation_eligible": False,
            "locator_id": "locator-dynamic",
            "maintenance_calls_performed": False,
            "origin": SESSION,
            "performance_evidence_eligible": False,
            "preflight_status": "CURRENT_CANONICAL_READY",
            "protocol_version": "myquant.v17.v4",
            "provider_calls_performed": False,
            "shadow_only": True,
            "source_refs": [
                {
                    "byte_sha256": f"{index + 1:064x}",
                    "media_type": (
                        "application/json"
                        if role
                        not in {
                            "contained_ledger",
                            "pit_membership",
                            "v15_strategy_universe",
                        }
                        else "application/vnd.apache.parquet"
                    ),
                    "relative_path": f"synthetic/{role}.json",
                    "role": role,
                }
                for index, role in enumerate(SOURCE_ROLES)
            ],
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.research-source-locator.v2",
        }
    )
    locator_ref = artifact_ref(
        locator,
        relative_path=(f"data/private/v17_v4_runs/{RUN_ID}/research_source_locator.json"),
    )
    factor_set = seal_semantic(
        {
            "audit_session": SESSION,
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "candidate_catalog_sha256": _sha(canonical_bytes(catalog)),
            "catalog_resource_sha256": "1" * 64,
            "cutoff": CUTOFF,
            "effective_from_session": SESSION,
            "eligible_distinct_slot_count": 1,
            "eligible_factor_count": 1,
            "factor_set_id": "factor-set-dynamic",
            "formal_activation_eligible": False,
            "implementation_resource_sha256": "2" * 64,
            "monthly_audit_ref": {
                "artifact_id": "audit",
                "artifact_version": "audit.v1",
                "byte_sha256": "3" * 64,
                "cutoff": CUTOFF,
                "relative_path": "synthetic/audit.json",
                "semantic_sha256": "3" * 64,
                "strategy_id": STRATEGY,
            },
            "performance_evidence_eligible": False,
            "previous_factor_set_ref": None,
            "protocol_version": "myquant.v17.v4",
            "selected_at": CUTOFF,
            "selected_factors": [selected],
            "selection_policy_sha256": "6" * 64,
            "shadow_only": True,
            "strategy_id": STRATEGY,
            "target_cardinality": 1,
            "version": "myquant.v17.v4.research-shadow-factor-set.v1",
        }
    )
    factor_set_ref = artifact_ref(
        factor_set,
        relative_path="data/private/v17_v4_sources/factor_sets/factor-set.json",
    )
    input_bundle = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "bundle_id": "bundle-dynamic",
            "canary_evidence_eligible": False,
            "cutoff": CUTOFF,
            "decision_session": SESSION,
            "factor_set_ref": factor_set_ref,
            "field_slices": [
                {
                    "available_at": CUTOFF,
                    "field_name": "adj_close",
                    "first_session": "2025-07-01",
                    "last_session": SESSION,
                    "row_count": 1,
                    "slice_ref": {
                        "artifact_id": "adj-close",
                        "artifact_version": "field-slice.v1",
                        "byte_sha256": "7" * 64,
                        "cutoff": CUTOFF,
                        "relative_path": (
                            f"data/private/v17_v4_runs/{RUN_ID}/"
                            "research_factor_inputs/adj_close.parquet"
                        ),
                        "semantic_sha256": "7" * 64,
                        "strategy_id": STRATEGY,
                    },
                }
            ],
            "formal_activation_eligible": False,
            "performance_evidence_eligible": False,
            "protocol_version": "myquant.v17.v4",
            "required_fields": ["adj_close"],
            "research_source_locator_ref": locator_ref,
            "run_id": RUN_ID,
            "shadow_only": True,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.research-factor-input-bundle.v1",
        }
    )
    input_bundle_ref = artifact_ref(
        input_bundle,
        relative_path=(f"data/private/v17_v4_runs/{RUN_ID}/research_factor_input_bundle.json"),
    )
    return (
        locator,
        locator_ref,
        factor_set,
        factor_set_ref,
        input_bundle,
        input_bundle_ref,
    )


def _market_frame() -> pd.DataFrame:
    sessions = pd.bdate_range(end=SESSION, periods=40)
    symbols = [f"{index:06d}.SZ" for index in range(1, 31)]
    values: dict[str, np.ndarray] = {}
    for index, symbol in enumerate(symbols, start=1):
        steps = np.arange(len(sessions), dtype=float)
        returns = (
            0.0002
            + np.sin(steps / (2.1 + index / 20.0)) * (0.002 + index * 0.00002)
            + np.cos(steps / (4.2 + index / 30.0)) * 0.001
        )
        values[symbol] = 10.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(values, index=sessions)


def _universe(*, missing_fundamental: str | None = None) -> pd.DataFrame:
    symbols = [f"{index:06d}.SZ" for index in range(1, 31)]
    frame = pd.DataFrame(
        {
            "available_at": [CUTOFF] * 30,
            "fin_debt_to_assets": np.linspace(0.1, 0.6, 30),
            "fin_ocf_to_profit": np.linspace(0.2, 1.4, 30),
            "fin_roe": np.linspace(0.01, 0.3, 30),
            "symbol": symbols,
            "trade_date": [SESSION] * 30,
        }
    )
    if missing_fundamental is not None:
        frame.loc[frame["symbol"].eq(missing_fundamental), "fin_roe"] = np.nan
    return frame


def _build(
    *,
    missing_fundamental: str | None = None,
) -> tuple[dict[str, dict[str, Any]], tuple[Any, ...]]:
    controls = _dynamic_controls()
    locator, locator_ref, factor_set, factor_set_ref, bundle, bundle_ref = controls
    artifacts = build_quant_first_forward_shadow(
        run_id=RUN_ID,
        factor_set=factor_set,
        input_bundle=bundle,
        source_locator=locator,
        source_locator_ref=locator_ref,
        factor_set_ref=factor_set_ref,
        input_bundle_ref=bundle_ref,
        strategy_universe=_universe(missing_fundamental=missing_fundamental),
        field_frames={"adj_close": _market_frame()},
        initial_pool_size=24,
        initial_pool_id="initial-dynamic",
        quant_output_id="quant-dynamic",
        fundamental_output_id="fundamental-dynamic",
        initial_pool_relative_path=(f"data/private/v17_v4_runs/{RUN_ID}/initial_pool.v2.json"),
    )
    return artifacts, controls


def test_dynamic_quant_preselection_and_branches_use_exact_same_pool() -> None:
    artifacts, _ = _build()
    initial = artifacts["initial_pool"]
    quant = artifacts["quant_branch"]
    fundamental = artifacts["fundamental_branch"]
    market = _market_frame()
    signal = evaluate_candidate_v4(
        name="cn_low_total_skewness_20d",
        inputs={"adj_close": market},
        pit_mask=market.notna(),
    )
    expected_scores = signal.loc[pd.Timestamp(SESSION)].rank(
        method="average",
        pct=True,
    )
    expected = sorted(
        expected_scores.dropna().index,
        key=lambda symbol: (-expected_scores.loc[symbol], symbol),
    )[:24]
    assert initial["ordered_pool"] == expected
    assert [row["symbol"] for row in quant["score_rows"]] == expected
    assert [row["symbol"] for row in fundamental["score_rows"]] == expected
    assert all(len(row["score"].partition(".")[2]) == 16 for row in quant["score_rows"])
    for document in artifacts.values():
        assert document["shadow_only"] is True
        assert document["formal_activation_eligible"] is False
        assert document["canary_evidence_eligible"] is False
        assert document["performance_evidence_eligible"] is False


def test_missing_fundamental_is_zero_and_excluded_from_percentiles() -> None:
    baseline, _ = _build()
    missing_symbol = baseline["initial_pool"]["ordered_pool"][0]
    artifacts, _ = _build(missing_fundamental=missing_symbol)
    row = next(
        item
        for item in artifacts["fundamental_branch"]["score_rows"]
        if item["symbol"] == missing_symbol
    )
    assert row == {
        "component_ranks": [],
        "coverage": "0.0000000000000000",
        "evidence_status": "UNAVAILABLE_MISSING_FUNDAMENTAL",
        "score": "0.0000000000000000",
        "symbol": missing_symbol,
    }


def test_unsupported_selected_factor_blocks_whole_set() -> None:
    controls = list(_dynamic_controls())
    factor_set = deepcopy(controls[2])
    factor_set["selected_factors"][0]["name"] = "unsupported_factor"
    factor_set = seal_semantic(
        {key: value for key, value in factor_set.items() if key != "semantic_sha256"}
    )
    controls[2] = factor_set
    controls[3] = artifact_ref(
        factor_set,
        relative_path="data/private/v17_v4_sources/factor_sets/factor-set.json",
    )
    bundle = deepcopy(controls[4])
    bundle["factor_set_ref"] = controls[3]
    bundle = seal_semantic(
        {key: value for key, value in bundle.items() if key != "semantic_sha256"}
    )
    controls[4] = bundle
    controls[5] = artifact_ref(
        bundle,
        relative_path=(f"data/private/v17_v4_runs/{RUN_ID}/research_factor_input_bundle.json"),
    )
    with pytest.raises(ForwardShadowPreparationError, match="unsupported_selected_factor"):
        build_quant_first_forward_shadow(
            run_id=RUN_ID,
            factor_set=controls[2],
            input_bundle=controls[4],
            source_locator=controls[0],
            source_locator_ref=controls[1],
            factor_set_ref=controls[3],
            input_bundle_ref=controls[5],
            strategy_universe=_universe(),
            field_frames={"adj_close": _market_frame()},
            initial_pool_size=24,
            initial_pool_id="initial-dynamic",
            quant_output_id="quant-dynamic",
            fundamental_output_id="fundamental-dynamic",
            initial_pool_relative_path=(f"data/private/v17_v4_runs/{RUN_ID}/initial_pool.v2.json"),
        )


def test_missing_selected_factor_input_never_silently_drops_factor() -> None:
    locator, locator_ref, factor_set, factor_set_ref, bundle, bundle_ref = _dynamic_controls()
    with pytest.raises(
        ForwardShadowPreparationError,
        match="missing_selected_factor_input",
    ):
        build_quant_first_forward_shadow(
            run_id=RUN_ID,
            factor_set=factor_set,
            input_bundle=bundle,
            source_locator=locator,
            source_locator_ref=locator_ref,
            factor_set_ref=factor_set_ref,
            input_bundle_ref=bundle_ref,
            strategy_universe=_universe(),
            field_frames={},
            initial_pool_size=24,
            initial_pool_id="initial-dynamic",
            quant_output_id="quant-dynamic",
            fundamental_output_id="fundamental-dynamic",
            initial_pool_relative_path=(f"data/private/v17_v4_runs/{RUN_ID}/initial_pool.v2.json"),
        )
