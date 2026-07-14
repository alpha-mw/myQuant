#!/usr/bin/env python3
"""Run a read-only full-A Quant factor selection shadow on strict Parquet data.

The runner loads an explicit self-hashed historical baseline, binds every
factor to its raw registry-record content, and promotes one governed zero-weight
candidate only inside an in-memory registry. It recomputes runtime factor
components and four comparison arms, then writes measurement-only
JSON/Markdown/Parquet evidence. It never writes the formal factor registry or
invokes market maintenance, analysis, portfolios, orders, brokers, providers,
or execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_investor.factors.governance import (  # noqa: E402
    FactorLifecycleState,
    FactorRecord,
)
from quant_investor.factors.governance_protocol_v2 import (  # noqa: E402
    FORWARD_PRODUCTION_APPLY_BLOCKER,
)
from quant_investor.factors.historical_shadow import (  # noqa: E402
    load_historical_shadow_baseline,
)
from quant_investor.factors.runtime import (  # noqa: E402
    MinedFactorRegistry,
    MinedFactorScorer,
    REPORT_ONLY_SHADOW_RUNTIME_MODE,
    _price_volume_required_lookback_rows,
    score_with_mined_factors,
)
from quant_investor.factors.shadow_scoring import (  # noqa: E402
    SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
)
from quant_investor.market.market_data_reader import MarketDataReader  # noqa: E402


SCHEMA_VERSION = "2026-07-12.quant-factor-selection-shadow.v2"
DEFAULT_REGISTRY_PATH = Path("quant_investor/factor_registry/mined_factors.json")
DEFAULT_CANDIDATE = "fund_fin_net_profit_yoy"
DEFAULT_SHADOW_EFFECTIVE_SHARE = 0.03
DEFAULT_EXPECTED_PRODUCTION_FACTOR_COUNT = 14
DEFAULT_LOOKBACK_CALENDAR_DAYS = 220
DEFAULT_MIN_SYMBOL_LOAD_COVERAGE = 0.95
DEFAULT_MIN_AS_OF_BAR_COVERAGE = 0.90
DEFAULT_MIN_CANDIDATE_COVERAGE = 0.60
DEFAULT_STATE_DIR = Path("reports/factor_governance/selection_shadow/_state")
PREREGISTRATION_FILENAME = "preregistration_v3.json"
BASELINE_CONTRACT_FILENAME = "baseline_contract_v3.json"
OBSERVATION_LEDGER_FILENAME = "observation_ledger_v3.jsonl"
SCORE_FILENAME = "quant_factor_selection_shadow_scores.parquet"
JSON_FILENAME = "quant_factor_selection_shadow.json"
MARKDOWN_FILENAME = "quant_factor_selection_shadow.md"


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    arm_type: str
    factor_weights: dict[str, float]
    removed_factor: str = ""
    candidate_name: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("quant_factor_selection_shadow_%Y%m%dT%H%M%SZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes()) if path.exists() else ""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _git_text(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def governed_experiment_config(
    *,
    expected_production_factor_count: int = DEFAULT_EXPECTED_PRODUCTION_FACTOR_COUNT,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    min_symbol_load_coverage: float = DEFAULT_MIN_SYMBOL_LOAD_COVERAGE,
    min_as_of_bar_coverage: float = DEFAULT_MIN_AS_OF_BAR_COVERAGE,
    min_candidate_coverage: float = DEFAULT_MIN_CANDIDATE_COVERAGE,
    replacement_weight_override: float | None = None,
    candidate_shadow_weight_override: float | None = None,
) -> dict[str, Any]:
    """Return every governed experiment parameter that can affect acceptance."""

    return {
        "expected_historical_baseline_factor_count": int(
            expected_production_factor_count
        ),
        "lookback_calendar_days": int(lookback_calendar_days),
        "min_symbol_load_coverage": float(min_symbol_load_coverage),
        "min_as_of_bar_coverage": float(min_as_of_bar_coverage),
        "min_candidate_coverage": float(min_candidate_coverage),
        "shadow_effective_share": DEFAULT_SHADOW_EFFECTIVE_SHARE,
        "replacement_weight_override": replacement_weight_override,
        "candidate_shadow_weight_override": candidate_shadow_weight_override,
        "sensitivity_overrides_allowed_in_governed_series": False,
    }


def preregistration_policy() -> dict[str, Any]:
    return {
        "schema_version": "2026-07-12.quant-factor-selection-preregistration.v3",
        "scope": "Quant-score selection shadow",
        "primary_candidate": "fund_fin_net_profit_yoy",
        "fallback_candidate": "fund_fin_net_profit_yoy_60d",
        "research_only_candidate": "formula_mom120_np_yoy_resid_w30",
        "research_only_runtime_eligible": False,
        "arms": {
            "A": (
                "all factors from an explicit hash-bound historical baseline "
                "manifest at manifest shadow weights"
            ),
            "B_i": "remove one old production factor",
            "C_i": "B_i plus candidate at removed factor absolute weight",
            "D": "A plus candidate at 3 percent effective absolute-weight share",
        },
        "missing_value_runtime_treatment": "fillna(0) with fixed total absolute-weight denominator",
        "prospective_checkpoint_trading_days": 90,
        "checkpoint_is_sufficient_sample": False,
        "min_month_end_rankic_count": 12,
        "min_nonoverlap_30d_cohort_count": 8,
        "maturity_threshold_logic": (
            "month_end_rankic_count>=12 OR nonoverlap_30d_cohort_count>=8"
        ),
        "formal_registry_write_allowed": False,
        "production_factor_promotion_allowed": False,
        "historical_baseline_manifest_required": True,
        "historical_records_become_production_selectable": False,
        "complete_production_screening_effect_claimed": False,
        "experiment_config": governed_experiment_config(),
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def ensure_preregistration(
    path: Path,
    *,
    policy: Mapping[str, Any],
    created_at: str,
) -> tuple[dict[str, Any], list[str]]:
    """Create-once immutable preregistration, then require an exact policy hash."""

    expected_policy = dict(_json_safe(policy))
    expected_hash = _sha256_bytes(_canonical_json_bytes(expected_policy))
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "matched"
    blockers: list[str] = []
    if not path.exists():
        payload = {
            "schema_version": "2026-07-12.quant-factor-selection-preregistration-file.v3",
            "created_at": created_at,
            "policy_sha256": expected_hash,
            "policy": expected_policy,
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
                handle.write("\n")
            status = "created"
        except FileExistsError:
            status = "matched"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        existing = {}
        blockers.append(f"preregistration_unreadable:{exc}")
    existing_policy = existing.get("policy") if isinstance(existing, Mapping) else None
    if not isinstance(existing_policy, Mapping):
        blockers.append("preregistration_policy_missing")
        actual_hash = ""
    else:
        actual_hash = _sha256_bytes(_canonical_json_bytes(existing_policy))
    recorded_hash = str(existing.get("policy_sha256") or "") if isinstance(existing, Mapping) else ""
    if actual_hash and recorded_hash != actual_hash:
        blockers.append("preregistration_recorded_hash_mismatch")
    if actual_hash and actual_hash != expected_hash:
        blockers.append("preregistration_policy_mismatch")
    return {
        "path": str(path),
        "status": "blocked" if blockers else status,
        "expected_policy_sha256": expected_hash,
        "recorded_policy_sha256": recorded_hash,
        "actual_policy_sha256": actual_hash,
        "file_sha256": _sha256_file(path),
        "immutable": True,
    }, blockers


def ensure_baseline_contract(
    path: Path,
    *,
    identity: Mapping[str, Any],
    start_audit: Mapping[str, Any],
    created_at: str,
) -> tuple[dict[str, Any], list[str]]:
    """Create once and lock strategy-relevant baseline identity fields.

    ``start_audit`` records the initial snapshot and full registry hash but is
    intentionally not compared with later runs.  This lets canonical snapshots
    advance and unrelated zero-weight registry records change while still
    locking production factor/candidate/code semantics.
    """

    expected_identity = dict(_json_safe(identity))
    expected_identity_hash = _sha256_bytes(_canonical_json_bytes(expected_identity))
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "matched"
    blockers: list[str] = []
    if not path.exists():
        immutable_payload = {
            "identity": expected_identity,
            "start_audit": dict(_json_safe(start_audit)),
        }
        contract_hash = _sha256_bytes(_canonical_json_bytes(immutable_payload))
        payload = {
            "schema_version": "2026-07-12.quant-factor-selection-baseline-contract.v3",
            "created_at": created_at,
            "contract_sha256": contract_hash,
            **immutable_payload,
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
                handle.write("\n")
            status = "created"
        except FileExistsError:
            status = "matched"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        existing = {}
        blockers.append(f"baseline_contract_unreadable:{exc}")
    existing_identity = existing.get("identity") if isinstance(existing, Mapping) else None
    existing_start_audit = (
        existing.get("start_audit") if isinstance(existing, Mapping) else None
    )
    if not isinstance(existing_identity, Mapping):
        blockers.append("baseline_contract_identity_missing")
        actual_identity_hash = ""
    else:
        actual_identity_hash = _sha256_bytes(_canonical_json_bytes(existing_identity))
    if not isinstance(existing_start_audit, Mapping):
        blockers.append("baseline_contract_start_audit_missing")
        actual_contract_hash = ""
    else:
        actual_contract_hash = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "identity": existing_identity,
                    "start_audit": existing_start_audit,
                }
            )
        )
    recorded_contract_hash = (
        str(existing.get("contract_sha256") or "") if isinstance(existing, Mapping) else ""
    )
    if actual_contract_hash and recorded_contract_hash != actual_contract_hash:
        blockers.append("baseline_contract_recorded_hash_mismatch")
    if actual_identity_hash and actual_identity_hash != expected_identity_hash:
        blockers.append("baseline_contract_identity_mismatch")
    return {
        "path": str(path),
        "status": "blocked" if blockers else status,
        "expected_identity_sha256": expected_identity_hash,
        "actual_identity_sha256": actual_identity_hash,
        "recorded_contract_sha256": recorded_contract_hash,
        "actual_contract_sha256": actual_contract_hash,
        "file_sha256": _sha256_file(path),
        "start_audit": dict(existing_start_audit or {}),
        "current_snapshot_may_advance": True,
        "full_registry_hash_is_audit_only": True,
        "immutable": True,
    }, blockers


def _record_sha256(record: FactorRecord) -> str:
    return _sha256_bytes(_canonical_json_bytes(record.to_dict()))


def _runtime_code_hashes() -> dict[str, str]:
    return {
        "scripts/run_quant_factor_selection_shadow.py": _sha256_file(Path(__file__).resolve()),
        "quant_investor/factors/runtime.py": _sha256_file(
            REPO_ROOT / "quant_investor/factors/runtime.py"
        ),
        "quant_investor/factors/price_volume.py": _sha256_file(
            REPO_ROOT / "quant_investor/factors/price_volume.py"
        ),
        "quant_investor/factors/aquant_expression.py": _sha256_file(
            REPO_ROOT / "quant_investor/factors/aquant_expression.py"
        ),
        "quant_investor/factors/pit_fundamentals.py": _sha256_file(
            REPO_ROOT / "quant_investor/factors/pit_fundamentals.py"
        ),
    }


def build_baseline_identity(
    production_records: Sequence[FactorRecord],
    candidate: FactorRecord,
    *,
    experiment_config: Mapping[str, Any] | None = None,
    historical_baseline_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "historical_baseline_manifest": {
            "baseline_id": str(
                dict(historical_baseline_audit or {}).get("baseline_id") or ""
            ),
            "manifest_sha256": str(
                dict(historical_baseline_audit or {}).get("manifest_sha256")
                or ""
            ),
            "source_record_sha256_by_name": dict(
                dict(historical_baseline_audit or {}).get(
                    "source_record_sha256_by_name", {}
                )
                or {}
            ),
            "runtime_mode": "report_only_shadow",
            "production_eligible": False,
        },
        "historical_baseline_factors": [
            {
                "name": record.name,
                "weight": float(record.weight),
                "record_sha256": _record_sha256(record),
            }
            for record in sorted(production_records, key=lambda item: item.name)
        ],
        "candidate": {
            "name": candidate.name,
            "version": candidate.version,
            "implementation": candidate.implementation,
            "expression": str(candidate.metadata.get("expression") or ""),
            "record_sha256": _record_sha256(candidate),
        },
        "runtime_code_hashes": _runtime_code_hashes(),
        "ranking_contract": {
            "top_n": [20, 50],
            "profile": "quant_score_ranking_exact_as_of_bar",
            "gate8_linear_overlay_used": False,
        },
        "data_contract": {
            "market": "CN",
            "backend": "parquet",
            "mode_policy": "strict",
            "pit": "strict",
            "allow_legacy_fundamental_fallback": False,
            "csv_fallback_allowed": False,
        },
        "experiment_config": dict(
            _json_safe(experiment_config or governed_experiment_config())
        ),
    }


def append_observation_once(
    path: Path,
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Append one unique snapshot/as-of/candidate observation under a file lock."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    key = str(observation.get("observation_key") or "")
    if not key:
        return {"path": str(path), "status": "blocked"}, ["observation_key_missing"]
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                blockers.append(f"observation_ledger_malformed_line:{line_number}:{exc}")
                continue
            if not isinstance(row, Mapping):
                blockers.append(f"observation_ledger_non_object_line:{line_number}")
                continue
            existing_rows.append(dict(row))
        existing_keys = {str(row.get("observation_key") or "") for row in existing_rows}
        status = "duplicate_not_appended" if key in existing_keys else "appended"
        if not blockers and status == "appended":
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(_json_safe(observation), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            existing_rows.append(dict(observation))
            existing_keys.add(key)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "path": str(path),
        "status": "blocked" if blockers else status,
        "observation_key": key,
        "unique_observation_count": len({item for item in existing_keys if item}),
        "duplicate_does_not_increment_count": True,
        "file_sha256": _sha256_file(path),
    }, blockers


def _clone_record(record: FactorRecord) -> FactorRecord:
    return FactorRecord.from_dict(record.to_dict())


def _promote_candidate_for_shadow(
    record: FactorRecord,
    *,
    nominal_weight: float,
    fundamental_mart_root: Path,
) -> FactorRecord:
    promoted = _clone_record(record)
    promoted.state = FactorLifecycleState.PRODUCTION_FACTOR
    promoted.weight = float(nominal_weight)
    promoted.metadata = {
        **dict(promoted.metadata),
        "fundamental_mart_root": str(fundamental_mart_root),
        "allow_legacy_fundamental_fallback": False,
        "shadow_only_in_memory_promotion": True,
        "runtime_effect": "none",
    }
    return promoted


def validate_registry_contract(
    registry: MinedFactorRegistry,
    *,
    candidate_name: str,
    expected_production_factor_count: int,
    historical_baseline_records: Sequence[FactorRecord] | None = None,
) -> tuple[list[FactorRecord], FactorRecord | None, list[str]]:
    blockers: list[str] = []
    if registry.metadata.get("missing"):
        blockers.append("registry_missing")
    if registry.metadata.get("load_error"):
        blockers.append(f"registry_load_error:{registry.metadata['load_error']}")
    production = list(historical_baseline_records or [])
    if historical_baseline_records is None:
        blockers.append("historical_baseline_manifest_required")
    if len(production) != int(expected_production_factor_count):
        blockers.append(
            "historical_baseline_factor_count_mismatch:"
            f"expected={expected_production_factor_count}:actual={len(production)}"
        )
    baseline_names = [record.name for record in production]
    if len(baseline_names) != len(set(baseline_names)):
        blockers.append("historical_baseline_factor_names_not_unique")
    if candidate_name in baseline_names:
        blockers.append("candidate_present_in_historical_baseline")
    matches = [record for record in registry.factors if record.name == candidate_name]
    candidate = matches[0] if len(matches) == 1 else None
    if len(matches) != 1:
        blockers.append(f"candidate_registry_record_count:{candidate_name}:{len(matches)}")
    elif candidate is not None:
        if candidate.state != FactorLifecycleState.PRODUCTION_CANDIDATE:
            blockers.append(f"candidate_not_production_candidate:{candidate.state.value}")
        if abs(float(candidate.weight)) > 1e-12:
            blockers.append(f"candidate_registry_weight_nonzero:{candidate.weight}")
        if not candidate.all_gates_passed():
            blockers.append("candidate_not_8_gate_passed")
        if not str(candidate.implementation).startswith("aquant_expression:"):
            blockers.append(f"candidate_runtime_implementation_unsupported:{candidate.implementation}")
        if not str(candidate.metadata.get("expression") or "").strip():
            blockers.append("candidate_expression_missing")
    return production, candidate, blockers


def build_arm_specs(
    production_records: Sequence[FactorRecord],
    *,
    candidate_name: str,
    replacement_weight_override: float | None = None,
    shadow_weight_override: float | None = None,
    shadow_effective_share: float = DEFAULT_SHADOW_EFFECTIVE_SHARE,
) -> list[ArmSpec]:
    old_weights = {record.name: abs(float(record.weight)) for record in production_records}
    arms = [ArmSpec(arm_id="A_old14", arm_type="baseline", factor_weights=old_weights)]
    for record in production_records:
        loo_weights = {
            name: weight for name, weight in old_weights.items() if name != record.name
        }
        slug = record.name.replace("/", "_").replace(":", "_")
        arms.append(
            ArmSpec(
                arm_id=f"B_loo__{slug}",
                arm_type="leave_one_out",
                factor_weights=loo_weights,
                removed_factor=record.name,
            )
        )
        arms.append(
            ArmSpec(
                arm_id=f"C_replace__{slug}",
                arm_type="one_for_one_replacement",
                factor_weights={
                    **loo_weights,
                    candidate_name: float(replacement_weight_override)
                    if replacement_weight_override is not None
                    else abs(float(record.weight)),
                },
                removed_factor=record.name,
                candidate_name=candidate_name,
            )
        )
    old_total_weight = sum(old_weights.values())
    if not 0.0 < float(shadow_effective_share) < 1.0:
        raise ValueError("shadow_effective_share must be in (0, 1)")
    dynamic_shadow_weight = (
        old_total_weight * float(shadow_effective_share) / (1.0 - float(shadow_effective_share))
    )
    arms.append(
        ArmSpec(
            arm_id="D_add_candidate_3pct",
            arm_type="additive_shadow",
            factor_weights={
                **old_weights,
                candidate_name: float(shadow_weight_override)
                if shadow_weight_override is not None
                else dynamic_shadow_weight,
            },
            candidate_name=candidate_name,
        )
    )
    return arms


def compute_runtime_components(
    frames: Mapping[str, pd.DataFrame],
    records: Sequence[FactorRecord],
) -> tuple[
    dict[str, pd.Series],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, set[str]],
]:
    """Compute signed normalized components with the exact runtime transforms."""

    symbols = [str(symbol) for symbol in frames if str(symbol).strip()]
    scorer = MinedFactorScorer(
        MinedFactorRegistry.from_records(records),
        runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE,
    )
    components: dict[str, pd.Series] = {}
    details: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    covered_symbols: dict[str, set[str]] = {}
    price_volume_names = [
        str(record.implementation).split(":", 1)[1]
        for record in records
        if str(record.implementation).startswith("price_volume:")
    ]
    prepared: Mapping[str, Any] | None = None
    cache: dict[str, Any] = {"active_price_volume_names": tuple(price_volume_names)}
    if price_volume_names:
        from quant_investor.factors.price_volume import prepare_price_volume_frames

        include_amihud = any(
            name.startswith("pv_amihud_illiquidity_")
            or name.startswith("pv_blend_volstab19x2_mom90_amihud5_w")
            for name in price_volume_names
        )
        prepared = prepare_price_volume_frames(
            frames,
            include_amihud_base=include_amihud,
            lookback_rows=_price_volume_required_lookback_rows(price_volume_names),
        )
    for record in records:
        try:
            implementation = str(record.implementation or "").strip()
            if implementation.startswith("price_volume:"):
                raw = scorer._price_volume_factor(
                    implementation.split(":", 1)[1],
                    frames,
                    prepared_frames=prepared,
                    factor_cache=cache,
                )
            else:
                raw = scorer._compute_factor(record, frames)
            raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = raw.dropna()
            if valid.empty:
                skipped[record.name] = "empty_factor_values"
                continue
            normalized = scorer._rank_normalize(raw.reindex(symbols))
            direction = 1.0 if float(record.direction) >= 0.0 else -1.0
            components[record.name] = normalized.fillna(0.0).mul(direction)
            covered_symbols[record.name] = {
                str(symbol) for symbol in valid.index.intersection(symbols)
            }
            details[record.name] = {
                "implementation": implementation,
                "valid_symbol_count": int(valid.index.intersection(symbols).size),
                "symbol_count": len(symbols),
                "coverage_rate": float(valid.index.intersection(symbols).size / max(len(symbols), 1)),
                "direction": direction,
                "component_score_min": float(components[record.name].min()),
                "component_score_max": float(components[record.name].max()),
            }
        except Exception as exc:
            skipped[record.name] = f"compute_error={exc}"
    return components, details, skipped, covered_symbols


def combine_runtime_components(
    components: Mapping[str, pd.Series],
    factor_weights: Mapping[str, float],
) -> pd.Series:
    names = [name for name, weight in factor_weights.items() if abs(float(weight)) > 1e-12]
    missing = [name for name in names if name not in components]
    if missing:
        raise ValueError(f"missing runtime factor components: {','.join(sorted(missing))}")
    if not names:
        raise ValueError("arm has no nonzero factor weights")
    index = components[names[0]].index
    weighted = pd.Series(0.0, index=index, dtype=float)
    total_weight = 0.0
    for name in names:
        weight = abs(float(factor_weights[name]))
        weighted = weighted.add(components[name].reindex(index).fillna(0.0).mul(weight))
        total_weight += weight
    if total_weight <= 1e-12:
        raise ValueError("arm total absolute weight is zero")
    return weighted.div(total_weight).clip(-1.0, 1.0).fillna(0.0)


def _deterministic_ranks(scores: pd.Series, symbols: Sequence[str]) -> pd.Series:
    available = [(str(symbol), float(scores.get(symbol, 0.0))) for symbol in symbols]
    ordered = sorted(available, key=lambda item: (-item[1], item[0]))
    return pd.Series(
        {symbol: index + 1 for index, (symbol, _score) in enumerate(ordered)},
        dtype=int,
    )


def compare_rankings(
    baseline_scores: pd.Series,
    arm_scores: pd.Series,
    *,
    symbols: Sequence[str],
    top_ns: Sequence[int] = (20, 50),
) -> dict[str, Any]:
    baseline_ranks = _deterministic_ranks(baseline_scores, symbols)
    arm_ranks = _deterministic_ranks(arm_scores, symbols)
    common = baseline_ranks.index.intersection(arm_ranks.index)
    rank_spearman = baseline_ranks.loc[common].astype(float).corr(
        arm_ranks.loc[common].astype(float), method="pearson"
    )
    deltas = baseline_ranks.loc[common].sub(arm_ranks.loc[common])
    result: dict[str, Any] = {
        "symbol_count": int(len(common)),
        "rank_spearman": float(rank_spearman) if pd.notna(rank_spearman) else None,
        "changed_rank_count": int(deltas.ne(0).sum()),
        "mean_abs_rank_delta": float(deltas.abs().mean()) if not deltas.empty else None,
        "max_abs_rank_delta": int(deltas.abs().max()) if not deltas.empty else None,
        "largest_rank_flips": [
            {
                "symbol": str(symbol),
                "baseline_rank": int(baseline_ranks[symbol]),
                "arm_rank": int(arm_ranks[symbol]),
                "rank_delta": int(deltas[symbol]),
            }
            for symbol in deltas.abs().sort_values(ascending=False).head(20).index
        ],
        "top_n": {},
    }
    for raw_n in top_ns:
        n = min(int(raw_n), len(common))
        baseline_top = list(baseline_ranks.sort_values().head(n).index)
        arm_top = list(arm_ranks.sort_values().head(n).index)
        baseline_set = set(baseline_top)
        arm_set = set(arm_top)
        union = baseline_set | arm_set
        result["top_n"][str(raw_n)] = {
            "jaccard": float(len(baseline_set & arm_set) / max(len(union), 1)),
            "overlap_count": int(len(baseline_set & arm_set)),
            "entered": [symbol for symbol in arm_top if symbol not in baseline_set],
            "exited": [symbol for symbol in baseline_top if symbol not in arm_set],
            "membership_flip_count": int(len(baseline_set ^ arm_set)),
        }
    return result


def _score_hash(scores: pd.Series) -> str:
    payload = "\n".join(
        f"{symbol}\t{float(scores[symbol]):.17g}" for symbol in sorted(scores.index)
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _top_scores(scores: pd.Series, ranks: pd.Series, n: int = 50) -> list[dict[str, Any]]:
    return [
        {"symbol": str(symbol), "rank": int(ranks[symbol]), "score": float(scores[symbol])}
        for symbol in ranks.sort_values().head(n).index
    ]


def _load_frames(
    reader: MarketDataReader,
    *,
    as_of: str,
    lookback_calendar_days: int,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str], dict[str, Any]]:
    symbols = reader.list_symbols("full_a", as_of=as_of)
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=int(lookback_calendar_days))).strftime(
        "%Y%m%d"
    )
    results = reader.read_symbol_frames(
        symbols,
        universe_key="full_a",
        category="full_a",
        start_date=start,
        end_date=as_of,
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "adj_close",
        ],
    )
    frames: dict[str, pd.DataFrame] = {}
    read_errors: dict[str, list[str]] = {}
    latest_present: list[str] = []
    future_symbols: list[str] = []
    for symbol in symbols:
        result = results.get(symbol)
        if result is None:
            read_errors[symbol] = ["missing_batch_read_result"]
            continue
        issues = [str(getattr(issue, "message", issue)) for issue in getattr(result, "issues", [])]
        if issues:
            read_errors[symbol] = issues
        frame = getattr(result, "frame", pd.DataFrame())
        if frame is None or frame.empty or "trade_date" not in frame.columns:
            continue
        working = frame.copy()
        working["trade_date"] = working["trade_date"].astype(str).str.replace("-", "", regex=False)
        working = working[working["trade_date"].str.len().eq(8)]
        working = working.sort_values("trade_date", kind="stable").reset_index(drop=True)
        if working.empty:
            continue
        latest = str(working["trade_date"].iloc[-1])
        if latest > as_of:
            future_symbols.append(symbol)
            continue
        frames[symbol] = working
        if latest == as_of:
            latest_present.append(symbol)
    diagnostics = {
        "requested_symbol_count": len(symbols),
        "loaded_symbol_count": len(frames),
        "as_of_bar_symbol_count": len(latest_present),
        "load_coverage": float(len(frames) / max(len(symbols), 1)),
        "as_of_bar_coverage": float(len(latest_present) / max(len(symbols), 1)),
        "read_error_symbol_count": len(read_errors),
        "read_error_samples": dict(list(read_errors.items())[:20]),
        "future_symbol_count": len(future_symbols),
        "future_symbol_samples": future_symbols[:20],
        "start_date": start,
        "end_date": as_of,
        "data_source": "MarketDataReader strict Parquet canonical batch",
        "csv_fallback_used": False,
    }
    return frames, symbols, latest_present, diagnostics


def _arm_payload(
    spec: ArmSpec,
    scores: pd.Series,
    *,
    selection_symbols: Sequence[str],
    baseline_scores: pd.Series,
    loo_scores: pd.Series | None = None,
) -> dict[str, Any]:
    ranks = _deterministic_ranks(scores, selection_symbols)
    total_weight = sum(abs(float(weight)) for weight in spec.factor_weights.values())
    candidate_weight = abs(float(spec.factor_weights.get(spec.candidate_name, 0.0)))
    payload = {
        "arm_id": spec.arm_id,
        "arm_type": spec.arm_type,
        "removed_factor": spec.removed_factor or None,
        "candidate_name": spec.candidate_name or None,
        "factor_count": len(spec.factor_weights),
        "factor_weights": dict(spec.factor_weights),
        "total_absolute_weight": float(total_weight),
        "candidate_nominal_weight": float(candidate_weight),
        "candidate_effective_share": float(candidate_weight / total_weight)
        if total_weight
        else 0.0,
        "score_hash_sha256": _score_hash(scores),
        "score_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
        },
        "top50": _top_scores(scores, ranks, 50),
        "comparison_to_A": compare_rankings(
            baseline_scores,
            scores,
            symbols=selection_symbols,
        ),
    }
    if loo_scores is not None:
        payload["candidate_increment_vs_B_loo"] = compare_rankings(
            loo_scores,
            scores,
            symbols=selection_symbols,
        )
    return payload


def _scores_frame(
    arms: Sequence[ArmSpec],
    scores_by_arm: Mapping[str, pd.Series],
    *,
    selection_symbols: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_ranks = _deterministic_ranks(scores_by_arm["A_old14"], selection_symbols)
    eligible = set(selection_symbols)
    for spec in arms:
        scores = scores_by_arm[spec.arm_id]
        ranks = _deterministic_ranks(scores, selection_symbols)
        for symbol in scores.index:
            rank = int(ranks[symbol]) if symbol in ranks.index else None
            baseline_rank = (
                int(baseline_ranks[symbol]) if symbol in baseline_ranks.index else None
            )
            rows.append(
                {
                    "arm_id": spec.arm_id,
                    "arm_type": spec.arm_type,
                    "removed_factor": spec.removed_factor or None,
                    "candidate_name": spec.candidate_name or None,
                    "symbol": str(symbol),
                    "score": float(scores[symbol]),
                    "eligible_exact_as_of_bar": symbol in eligible,
                    "rank": rank,
                    "baseline_rank": baseline_rank,
                    "rank_delta": (
                        baseline_rank - rank
                        if baseline_rank is not None and rank is not None
                        else None
                    ),
                    "top20": bool(rank is not None and rank <= 20),
                    "top50": bool(rank is not None and rank <= 50),
                }
            )
    return pd.DataFrame(rows)


def _covered_uncovered_selection_bias(
    *,
    candidate_name: str,
    covered_symbols: set[str],
    selection_symbols: Sequence[str],
    baseline_scores: pd.Series,
    additive_scores: pd.Series,
) -> dict[str, Any]:
    selection_set = set(selection_symbols)
    covered = selection_set & set(covered_symbols)
    uncovered = selection_set - covered

    def arm_stats(scores: pd.Series) -> dict[str, Any]:
        ranks = _deterministic_ranks(scores, selection_symbols)
        result: dict[str, Any] = {
            "covered_mean_score": float(scores.reindex(sorted(covered)).mean())
            if covered
            else None,
            "uncovered_mean_score": float(scores.reindex(sorted(uncovered)).mean())
            if uncovered
            else None,
            "covered_mean_rank": float(ranks.reindex(sorted(covered)).mean())
            if covered
            else None,
            "uncovered_mean_rank": float(ranks.reindex(sorted(uncovered)).mean())
            if uncovered
            else None,
            "top_n": {},
        }
        for n in (20, 50):
            top = set(ranks.sort_values().head(min(n, len(ranks))).index)
            covered_count = len(top & covered)
            uncovered_count = len(top & uncovered)
            covered_selection_rate = float(covered_count / max(len(covered), 1))
            uncovered_selection_rate = float(uncovered_count / max(len(uncovered), 1))
            result["top_n"][str(n)] = {
                "covered_count": covered_count,
                "uncovered_count": uncovered_count,
                "covered_share": float(covered_count / max(len(top), 1)),
                "covered_selection_rate": covered_selection_rate,
                "uncovered_selection_rate": uncovered_selection_rate,
                "covered_to_uncovered_selection_rate_ratio": (
                    float(covered_selection_rate / uncovered_selection_rate)
                    if uncovered_selection_rate > 0.0
                    else None
                ),
            }
        return result

    return {
        "candidate": candidate_name,
        "selection_symbol_count": len(selection_set),
        "covered_symbol_count": len(covered),
        "uncovered_symbol_count": len(uncovered),
        "covered_share": float(len(covered) / max(len(selection_set), 1)),
        "missing_value_runtime_treatment": "candidate component score is fillna(0); total arm weight remains in denominator",
        "A_old14": arm_stats(baseline_scores),
        "D_add_candidate_3pct": arm_stats(additive_scores),
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    snapshot = dict(payload.get("snapshot", {}) or {})
    candidate = dict(payload.get("candidate", {}) or {})
    data = dict(payload.get("data", {}) or {})
    lines = [
        "# Quant Factor Selection Shadow",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- Snapshot: `{snapshot.get('snapshot_id', '')}` / `{snapshot.get('latest_complete_trade_date', '')}`",
        f"- Registry write: `{payload.get('registry_update_status')}`",
        f"- Historical baseline manifest: `{payload.get('registry', {}).get('historical_baseline_manifest', {}).get('manifest_sha256')}`",
        f"- Baseline contract: `{payload.get('baseline_contract', {}).get('status')}` / `{payload.get('baseline_contract', {}).get('actual_contract_sha256')}`",
        f"- Candidate: `{candidate.get('name', '')}`",
        f"- Candidate monthly RankIC count: `{candidate.get('maturity', {}).get('monthly_rankic_count')}`",
        f"- C_i weight policy: `{candidate.get('replacement_weight_policy', '')}`",
        f"- D weight policy: `{candidate.get('additive_weight_policy', '')}`",
        f"- Full-A symbols loaded: `{data.get('loaded_symbol_count', 0)}`",
        f"- Exact as-of selection symbols: `{data.get('as_of_bar_symbol_count', 0)}`",
        "",
        "## Fail-closed blockers",
        "",
    ]
    blockers = list(payload.get("fail_closed_blockers", []) or [])
    lines.extend([f"- `{item}`" for item in blockers] or ["- None"])
    lines.extend(
        [
            "",
            "## Runtime parity",
            "",
            f"- Old14 direct/composed max absolute delta: `{payload.get('runtime_parity', {}).get('old14_max_abs_delta')}`",
            f"- Candidate direct/component max absolute delta: `{payload.get('runtime_parity', {}).get('candidate_max_abs_delta')}`",
            "",
            "## Candidate covered/uncovered selection bias",
            "",
            f"- Candidate covered share: `{payload.get('covered_uncovered_selection_bias', {}).get('covered_share')}`",
            f"- D Top20 covered/uncovered selection-rate ratio: `{payload.get('covered_uncovered_selection_bias', {}).get('D_add_candidate_3pct', {}).get('top_n', {}).get('20', {}).get('covered_to_uncovered_selection_rate_ratio')}`",
            f"- D Top50 covered/uncovered selection-rate ratio: `{payload.get('covered_uncovered_selection_bias', {}).get('D_add_candidate_3pct', {}).get('top_n', {}).get('50', {}).get('covered_to_uncovered_selection_rate_ratio')}`",
            "",
            "## Arm comparison versus A",
            "",
            "| Arm | Type | Removed | Candidate share | Rank Spearman | Top20 Jaccard | Top50 Jaccard |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in list(payload.get("arms", []) or []):
        comparison = dict(arm.get("comparison_to_A", {}) or {})
        top_n = dict(comparison.get("top_n", {}) or {})
        lines.append(
            "| "
            f"`{arm.get('arm_id', '')}` | "
            f"{arm.get('arm_type', '')} | "
            f"`{arm.get('removed_factor') or ''}` | "
            f"{float(arm.get('candidate_effective_share', 0.0)):.4%} | "
            f"{comparison.get('rank_spearman')} | "
            f"{dict(top_n.get('20', {}) or {}).get('jaccard')} | "
            f"{dict(top_n.get('50', {}) or {}).get('jaccard')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a point-in-time Quant ranking/selection measurement. It does not prove future return improvement, and it does not include downstream Theme, liquidity, sector-bucket, Bayesian, RiskGuard, PortfolioConstructor, portfolio, order, or execution decisions.",
            "",
            SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--historical-baseline-manifest",
        type=Path,
        default=None,
        help=(
            "Required self-hashed old-factor manifest. Omitting it produces a "
            "fail-closed report; selectable registry count is never used as A_old14."
        ),
    )
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-shadow-weight",
        type=float,
        default=None,
        help="Explicit sensitivity override; default dynamically targets 3% effective share.",
    )
    parser.add_argument(
        "--replacement-weight",
        type=float,
        default=None,
        help="Explicit sensitivity override; default equals each removed old factor weight.",
    )
    parser.add_argument(
        "--expected-production-factor-count",
        type=int,
        default=DEFAULT_EXPECTED_PRODUCTION_FACTOR_COUNT,
        help=(
            "Legacy flag name: expected factor count in the explicit historical "
            "shadow baseline, not the current selectable production count."
        ),
    )
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
    )
    parser.add_argument(
        "--min-symbol-load-coverage",
        type=float,
        default=DEFAULT_MIN_SYMBOL_LOAD_COVERAGE,
    )
    parser.add_argument(
        "--min-as-of-bar-coverage",
        type=float,
        default=DEFAULT_MIN_AS_OF_BAR_COVERAGE,
    )
    parser.add_argument(
        "--min-candidate-coverage",
        type=float,
        default=DEFAULT_MIN_CANDIDATE_COVERAGE,
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser


def run_shadow(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    generated_at = _now_iso()
    output_dir = args.output_dir or (
        Path("reports/factor_governance/selection_shadow") / _run_id()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = preregistration_policy()
    experiment_config = governed_experiment_config(
        expected_production_factor_count=int(args.expected_production_factor_count),
        lookback_calendar_days=int(args.lookback_calendar_days),
        min_symbol_load_coverage=float(args.min_symbol_load_coverage),
        min_as_of_bar_coverage=float(args.min_as_of_bar_coverage),
        min_candidate_coverage=float(args.min_candidate_coverage),
        replacement_weight_override=args.replacement_weight,
        candidate_shadow_weight_override=args.candidate_shadow_weight,
    )
    preregistration, preregistration_blockers = ensure_preregistration(
        Path(args.state_dir) / PREREGISTRATION_FILENAME,
        policy=policy,
        created_at=generated_at,
    )
    registry_path = Path(args.registry_path).expanduser()
    registry_hash_before = _sha256_file(registry_path)
    blockers: list[str] = list(preregistration_blockers)
    registered_config = dict(policy.get("experiment_config", {}) or {})
    if experiment_config != registered_config:
        blockers.append("experiment_config_not_preregistered")
    allowed_candidates = {
        str(policy["primary_candidate"]),
        str(policy["fallback_candidate"]),
    }
    if str(args.candidate) not in allowed_candidates:
        blockers.append(f"candidate_not_preregistered_for_runtime_shadow:{args.candidate}")
    backend = str(os.environ.get("MYQUANT_MARKET_DATA_BACKEND", "")).strip().lower()
    mode_policy = str(os.environ.get("MYQUANT_MARKET_DATA_MODE_POLICY", "")).strip().lower()
    if backend != "parquet":
        blockers.append(f"market_data_backend_not_parquet:{backend or 'unset'}")
    if mode_policy != "strict":
        blockers.append(f"market_data_mode_policy_not_strict:{mode_policy or 'unset'}")

    reader = MarketDataReader(market="CN", data_root=args.data_root, mode_policy="strict")
    snapshot = reader.snapshot()
    if not snapshot.get("healthy"):
        blockers.extend(f"parquet_snapshot:{item}" for item in snapshot.get("blockers", []))
    as_of = str(snapshot.get("latest_complete_trade_date") or "")
    if not as_of:
        blockers.append("latest_complete_trade_date_missing")
    latest_pointer = Path(str(snapshot.get("latest_pointer_path") or ""))
    manifest_path = Path(str(snapshot.get("manifest_path") or ""))
    pointer_hash_before = _sha256_file(latest_pointer)

    registry = MinedFactorRegistry.load(registry_path)
    historical_baseline_audit: dict[str, Any] = {
        "status": "blocked",
        "production_eligible": False,
        "formal_registry_mutated": False,
    }
    historical_baseline_records: list[FactorRecord] | None = None
    if args.historical_baseline_manifest is None:
        blockers.append("historical_baseline_manifest_required")
    else:
        try:
            historical_registry, historical_baseline_audit = (
                load_historical_shadow_baseline(
                    manifest_path=args.historical_baseline_manifest,
                    registry_path=registry_path,
                    expected_factor_count=int(
                        args.expected_production_factor_count
                    ),
                )
            )
            historical_baseline_records = list(historical_registry.factors)
            historical_baseline_audit = {
                **historical_baseline_audit,
                "status": "verified",
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"historical_baseline_manifest_blocked:{exc}")
            historical_baseline_audit = {
                **historical_baseline_audit,
                "manifest_path": str(args.historical_baseline_manifest),
                "blocker": str(exc),
            }
    production, candidate, registry_blockers = validate_registry_contract(
        registry,
        candidate_name=str(args.candidate),
        expected_production_factor_count=int(args.expected_production_factor_count),
        historical_baseline_records=historical_baseline_records,
    )
    blockers.extend(registry_blockers)
    baseline_contract: dict[str, Any] = {
        "path": str(Path(args.state_dir) / BASELINE_CONTRACT_FILENAME),
        "status": "not_checked_due_prior_blockers",
    }
    if not blockers and candidate is not None:
        baseline_contract, baseline_blockers = ensure_baseline_contract(
            Path(args.state_dir) / BASELINE_CONTRACT_FILENAME,
            identity=build_baseline_identity(
                production,
                candidate,
                experiment_config=experiment_config,
                historical_baseline_audit=historical_baseline_audit,
            ),
            start_audit={
                "snapshot_id": str(snapshot.get("snapshot_id") or ""),
                "latest_complete_trade_date": as_of,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "latest_pointer_sha256": pointer_hash_before,
                "full_registry_sha256": registry_hash_before,
            },
            created_at=generated_at,
        )
        blockers.extend(baseline_blockers)

    frames: dict[str, pd.DataFrame] = {}
    requested_symbols: list[str] = []
    selection_symbols: list[str] = []
    data_diagnostics: dict[str, Any] = {}
    components: dict[str, pd.Series] = {}
    component_details: dict[str, dict[str, Any]] = {}
    covered_symbols_by_factor: dict[str, set[str]] = {}
    runtime_parity: dict[str, Any] = {}
    arms_payload: list[dict[str, Any]] = []
    scores_path = output_dir / SCORE_FILENAME

    if not blockers:
        try:
            frames, requested_symbols, selection_symbols, data_diagnostics = _load_frames(
                reader,
                as_of=as_of,
                lookback_calendar_days=int(args.lookback_calendar_days),
            )
        except Exception as exc:
            blockers.append(f"strict_parquet_read_failed:{exc}")
    if data_diagnostics:
        if data_diagnostics["load_coverage"] < float(args.min_symbol_load_coverage):
            blockers.append(
                "symbol_load_coverage_below_threshold:"
                f"{data_diagnostics['load_coverage']:.6f}<{args.min_symbol_load_coverage:.6f}"
            )
        if data_diagnostics["as_of_bar_coverage"] < float(args.min_as_of_bar_coverage):
            blockers.append(
                "as_of_bar_coverage_below_threshold:"
                f"{data_diagnostics['as_of_bar_coverage']:.6f}<{args.min_as_of_bar_coverage:.6f}"
            )
        if data_diagnostics["future_symbol_count"]:
            blockers.append("future_dated_market_rows_detected")
    if not selection_symbols and not blockers:
        blockers.append("selection_universe_empty")

    candidate_shadow: FactorRecord | None = None
    if not blockers and candidate is not None:
        candidate_shadow = _promote_candidate_for_shadow(
            candidate,
            nominal_weight=1.0,
            fundamental_mart_root=Path(args.data_root) / "parquet" / "cn",
        )
        old_components, old_details, old_skipped, old_covered = compute_runtime_components(
            frames, production
        )
        components.update(old_components)
        component_details.update(old_details)
        covered_symbols_by_factor.update(old_covered)
        if old_skipped:
            blockers.extend(
                f"old_factor_runtime_skip:{name}:{reason}"
                for name, reason in sorted(old_skipped.items())
            )

        (
            candidate_components,
            candidate_details,
            candidate_skipped,
            candidate_covered,
        ) = compute_runtime_components(frames, [candidate_shadow])
        components.update(candidate_components)
        component_details.update(candidate_details)
        covered_symbols_by_factor.update(candidate_covered)
        if candidate_skipped:
            blockers.extend(
                f"candidate_runtime_skip:{name}:{reason}"
                for name, reason in sorted(candidate_skipped.items())
            )
        candidate_coverage = float(
            candidate_details.get(candidate.name, {}).get("coverage_rate", 0.0)
        )
        if candidate_coverage < float(args.min_candidate_coverage):
            blockers.append(
                "candidate_coverage_below_threshold:"
                f"{candidate_coverage:.6f}<{args.min_candidate_coverage:.6f}"
            )

        if not old_skipped:
            direct_old = score_with_mined_factors(
                frames,
                registry=MinedFactorRegistry(
                    factors=list(production),
                    metadata={"historical_shadow_only": True},
                ),
                runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE,
            )
            composed_old = combine_runtime_components(
                components,
                {record.name: abs(float(record.weight)) for record in production},
            )
            direct_old_series = pd.Series(direct_old.symbol_scores, dtype=float).reindex(
                composed_old.index
            )
            old_delta = float((direct_old_series - composed_old).abs().max())
            runtime_parity["old14_max_abs_delta"] = old_delta
            runtime_parity["old14_direct_factor_count"] = direct_old.factor_count
            runtime_parity["old14_direct_skipped_factors"] = dict(direct_old.skipped_factors)
            if direct_old.factor_count != len(production) or direct_old.skipped_factors:
                blockers.append("old14_direct_runtime_not_complete")
            if old_delta > 1e-12:
                blockers.append(f"old14_runtime_parity_failed:{old_delta}")

        if not candidate_skipped and candidate_shadow is not None:
            direct_candidate = score_with_mined_factors(
                frames,
                registry=MinedFactorRegistry.from_records([candidate_shadow]),
                runtime_mode=REPORT_ONLY_SHADOW_RUNTIME_MODE,
            )
            candidate_series = components[candidate.name]
            direct_candidate_series = pd.Series(
                direct_candidate.symbol_scores, dtype=float
            ).reindex(candidate_series.index)
            candidate_delta = float((direct_candidate_series - candidate_series).abs().max())
            runtime_parity["candidate_max_abs_delta"] = candidate_delta
            runtime_parity["candidate_direct_factor_count"] = direct_candidate.factor_count
            runtime_parity["candidate_direct_skipped_factors"] = dict(
                direct_candidate.skipped_factors
            )
            if direct_candidate.factor_count != 1 or direct_candidate.skipped_factors:
                blockers.append("candidate_direct_runtime_not_complete")
            if candidate_delta > 1e-12:
                blockers.append(f"candidate_runtime_parity_failed:{candidate_delta}")

    if not blockers and candidate is not None:
        arms = build_arm_specs(
            production,
            candidate_name=candidate.name,
            replacement_weight_override=args.replacement_weight,
            shadow_weight_override=args.candidate_shadow_weight,
        )
        scores_by_arm = {
            spec.arm_id: combine_runtime_components(components, spec.factor_weights)
            for spec in arms
        }
        baseline = scores_by_arm["A_old14"]
        loo_by_removed = {
            spec.removed_factor: scores_by_arm[spec.arm_id]
            for spec in arms
            if spec.arm_type == "leave_one_out"
        }
        arms_payload = [
            _arm_payload(
                spec,
                scores_by_arm[spec.arm_id],
                selection_symbols=selection_symbols,
                baseline_scores=baseline,
                loo_scores=(
                    loo_by_removed.get(spec.removed_factor)
                    if spec.arm_type == "one_for_one_replacement"
                    else None
                ),
            )
            for spec in arms
        ]
        score_frame = _scores_frame(
            arms,
            scores_by_arm,
            selection_symbols=selection_symbols,
        )
        score_frame.to_parquet(scores_path, index=False)

    selection_bias: dict[str, Any] = {}
    if arms_payload and candidate is not None:
        score_lookup = {
            spec.arm_id: combine_runtime_components(components, spec.factor_weights)
            for spec in build_arm_specs(
                production,
                candidate_name=candidate.name,
                replacement_weight_override=args.replacement_weight,
                shadow_weight_override=args.candidate_shadow_weight,
            )
        }
        selection_bias = _covered_uncovered_selection_bias(
            candidate_name=candidate.name,
            covered_symbols=covered_symbols_by_factor.get(candidate.name, set()),
            selection_symbols=selection_symbols,
            baseline_scores=score_lookup["A_old14"],
            additive_scores=score_lookup["D_add_candidate_3pct"],
        )

    registry_hash_after = _sha256_file(registry_path)
    pointer_hash_after = _sha256_file(latest_pointer)
    if registry_hash_before != registry_hash_after:
        blockers.append("formal_registry_hash_changed_during_shadow")
    if pointer_hash_before != pointer_hash_after:
        blockers.append("latest_pointer_hash_changed_during_shadow")

    observation_ledger: dict[str, Any] = {
        "path": str(Path(args.state_dir) / OBSERVATION_LEDGER_FILENAME),
        "status": "not_appended_blocked_run",
        "unique_observation_count": 0,
        "duplicate_does_not_increment_count": True,
    }
    if not blockers and candidate is not None and arms_payload:
        observation_key = "|".join(
            [
                str(snapshot.get("snapshot_id") or ""),
                as_of,
                candidate.name,
                str(baseline_contract.get("actual_contract_sha256") or ""),
            ]
        )
        observation_ledger, ledger_blockers = append_observation_once(
            Path(args.state_dir) / OBSERVATION_LEDGER_FILENAME,
            {
                "schema_version": "2026-07-12.quant-factor-selection-observation.v3",
                "observation_key": observation_key,
                "generated_at": generated_at,
                "snapshot_id": str(snapshot.get("snapshot_id") or ""),
                "as_of": as_of,
                "candidate": candidate.name,
                "registry_sha256": registry_hash_after,
                "historical_baseline_manifest_sha256": (
                    historical_baseline_audit.get("manifest_sha256")
                ),
                "preregistration_policy_sha256": preregistration.get(
                    "actual_policy_sha256"
                ),
                "baseline_contract_sha256": baseline_contract.get(
                    "actual_contract_sha256"
                ),
                "output_dir": str(output_dir),
                "scores_parquet_sha256": _sha256_file(scores_path),
                "monthly_rankic_count_from_registry": candidate.metrics.get("rank_ic_count"),
                "nonoverlap_30d_cohort_count": None,
                "measurement_only": True,
                "registry_write": False,
            },
        )
        blockers.extend(ledger_blockers)

    candidate_metrics = dict(candidate.metrics) if candidate is not None else {}
    maturity_count = candidate_metrics.get("rank_ic_count")
    maturity_ready = bool(maturity_count is not None and int(maturity_count) >= 12)
    blockers = list(dict.fromkeys(blockers))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "fail_closed": bool(blockers),
        "fail_closed_blockers": blockers,
        "generated_at": generated_at,
        "measurement_only": True,
        "registry_update_status": "not_written_read_only_shadow",
        "production_runtime_effect": "none",
        "non_runtime_impact_note": SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
        "snapshot": {
            **snapshot,
            "latest_pointer_sha256_before": pointer_hash_before,
            "latest_pointer_sha256_after": pointer_hash_after,
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "registry": {
            "path": str(registry_path),
            "sha256_before": registry_hash_before,
            "sha256_after": registry_hash_after,
            "unchanged": registry_hash_before == registry_hash_after,
            "current_selectable_factor_count": len(
                registry.selectable_factors()
            ),
            "current_selectable_factor_names": [
                record.name for record in registry.selectable_factors()
            ],
            "historical_baseline_factor_count": len(production),
            "historical_baseline_factor_names": [
                record.name for record in production
            ],
            "historical_baseline_manifest": historical_baseline_audit,
        },
        "candidate": {
            "name": candidate.name if candidate is not None else str(args.candidate),
            "registry_state": candidate.state.value if candidate is not None else None,
            "registry_weight": float(candidate.weight) if candidate is not None else None,
            "implementation": candidate.implementation if candidate is not None else None,
            "expression": candidate.metadata.get("expression") if candidate is not None else None,
            "all_eight_gates_passed": candidate.all_gates_passed()
            if candidate is not None
            else False,
            "in_memory_only_state": "production_factor_for_shadow"
            if candidate_shadow is not None
            else None,
            "in_memory_legacy_fundamental_fallback": False,
            "replacement_weight_policy": (
                f"explicit_override:{args.replacement_weight}"
                if args.replacement_weight is not None
                else "equal_to_removed_factor_absolute_weight"
            ),
            "additive_weight_policy": (
                f"explicit_override:{args.candidate_shadow_weight}"
                if args.candidate_shadow_weight is not None
                else "dynamic_old_total_abs_weight_times_0.03_div_0.97"
            ),
            "maturity": {
                "monthly_rankic_count": int(maturity_count)
                if maturity_count is not None
                else None,
                "source": "registry_candidate_evidence"
                if maturity_count is not None
                else "unavailable",
                "fresh_forward_return_recomputed": False,
                "prospective_status": "pre_registered_awaiting_future_returns",
                "current_nonoverlap_30d_cohort_count": None,
                "current_nonoverlap_30d_cohort_count_source": "unavailable",
                "min_month_end_rankic_count": 12,
                "min_nonoverlap_30d_cohort_count": 8,
                "threshold_logic": "month_end_rankic_count>=12 OR nonoverlap_30d_cohort_count>=8",
                "threshold_ready_from_available_evidence": maturity_ready,
                "scheduled_checkpoint_trading_days": 90,
                "unique_snapshot_observation_count": observation_ledger.get(
                    "unique_observation_count", 0
                ),
                "effective_analysis_start_date": candidate.metadata.get(
                    "effective_analysis_start_date"
                )
                if candidate is not None
                else None,
            },
        },
        "data": {
            **data_diagnostics,
            "requested_symbol_count": len(requested_symbols),
            "selection_universe_semantics": (
                "full-A symbols with an exact latest_complete_trade_date bar; factor "
                "normalization uses all loaded full-A frames; downstream funnel gates are not run"
            ),
        },
        "runtime_components": component_details,
        "runtime_parity": runtime_parity,
        "experiment_config": experiment_config,
        "covered_uncovered_selection_bias": selection_bias,
        "preregistration": {**preregistration, "policy": policy},
        "baseline_contract": baseline_contract,
        "observation_ledger": observation_ledger,
        "arm_protocol": {
            "A": "manifest-bound historical old14 baseline (report-only)",
            "B_i": "leave one old factor out (LOO13)",
            "C_i": (
                "LOO13 plus candidate at the removed factor's actual absolute weight "
                "unless an explicit sensitivity override is supplied"
            ),
            "D": (
                "old factors plus candidate with dynamic nominal weight targeting exactly "
                "3% effective absolute-weight share"
            ),
            "replacement_weight_override": args.replacement_weight,
            "additive_weight_override": args.candidate_shadow_weight,
            "gate8_linear_overlay_used": False,
            "runtime_composite_recomputed": True,
            "top_n": [20, 50],
        },
        "scope": {
            "name": "Quant-score selection shadow",
            "what_is_recomputed": (
                "governed mined-factor runtime composite scores plus deterministic Top20/Top50 "
                "ranking across exact-as-of-bar full-A symbols"
            ),
            "what_is_not_recomputed": [
                "DataQualityGate/TradabilityGate beyond exact-as-of-bar proxy",
                "LiquidityGate",
                "Theme Candidate Pool",
                "momentum-leader profile context",
                "sector bucket limits",
                "Fundamental branch scores and conditional incremental correlation",
                "Bayesian posterior",
                "RiskGuard/ICCoordinator/PortfolioConstructor",
            ],
            "complete_production_screening_effect_claimed": False,
            "full_v13_dag_evidence_claimed": False,
            "full_v13_dag_blocker": FORWARD_PRODUCTION_APPLY_BLOCKER,
        },
        "scope_limitations": [
            {
                "code": "downstream_deterministic_funnel_not_reproduced",
                "impact": "Top20/Top50 are Quant-score rankings, not the final production candidate pool",
            },
            {
                "code": "theme_pool_conditional_effect_not_evaluated",
                "impact": "candidate impact after Theme Candidate Pool filtering is unavailable",
            },
            {
                "code": "quant_fundamental_conditional_increment_not_evaluated",
                "impact": (
                    "fin_net_profit_yoy may overlap the Fundamental branch; no cross-branch "
                    "promotion claim is permitted"
                ),
            },
            {
                "code": "forward_return_maturity_not_freshly_observable",
                "impact": "this point-in-time run measures rank selection only",
            },
        ],
        "arms": arms_payload,
        "artifacts": {
            "scores_parquet": str(scores_path) if scores_path.exists() else None,
            "scores_parquet_sha256": _sha256_file(scores_path),
            "json": str(output_dir / JSON_FILENAME),
            "markdown": str(output_dir / MARKDOWN_FILENAME),
        },
        "code": {
            "git_commit": _git_text("rev-parse", "HEAD"),
            "working_tree_status": _git_text("status", "--short"),
            "hashes": _runtime_code_hashes(),
        },
        "explicit_non_actions": [
            "no_formal_registry_write",
            "no_market_maintain",
            "no_market_run",
            "no_market_analyze",
            "no_live_provider",
            "no_portfolio_or_strategy_record",
            "no_order_or_broker_action",
        ],
    }
    payload = dict(_json_safe(payload))
    json_path = output_dir / JSON_FILENAME
    markdown_path = output_dir / MARKDOWN_FILENAME
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload, output_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload, output_dir = run_shadow(args)
    print(f"status: {payload['status']}")
    print(f"snapshot_id: {payload.get('snapshot', {}).get('snapshot_id', '')}")
    print(f"candidate: {payload.get('candidate', {}).get('name', '')}")
    print(f"arm_count: {len(payload.get('arms', []))}")
    print(f"blockers: {', '.join(payload.get('fail_closed_blockers', []))}")
    print(f"output_dir: {output_dir}")
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
