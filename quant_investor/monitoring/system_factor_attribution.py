"""Report-only system attribution and Factor v4 analysis for CN V15 reviews.

The module reads supported local/manual execution manifests and the neutral
Factor v4 readiness artifact.  It never reads a ledger fallback, mutates
governance state, authorizes risk, or performs execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quant_investor.factors.governance_protocol_v4 import semantic_sha256
from quant_investor.factors.readiness_store import (
    FactorReadinessStoreError,
    read_factor_readiness,
)

SCHEMA_VERSION = "cn_aggressive_system_factor_attribution.v1"
FACTOR_READINESS_RELATIVE_PATH = Path("results/factor_governance/readiness.json")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
ATTRIBUTION_BUCKETS = (
    "portfolio_decision",
    "factor_signal",
    "risk_control",
    "insufficient_evidence",
)
SUPPORTED_MANUAL_MANIFEST_SCHEMAS = {
    "cn_aggressive_manual_execution.v2",
    "cn_aggressive_manual_execution.v3",
}
_RISK_REASON_MARKERS = (
    "risk",
    "stop",
    "giveback",
    "take_profit",
    "take-profit",
    "reduce",
    "drawdown",
    "风险",
    "止损",
    "止盈",
    "回吐",
    "减仓",
)
_FACTOR_REASON_MARKERS = ("factor", "因子")
_PORTFOLIO_REASON_MARKERS = (
    "portfolio",
    "rebalance",
    "switch",
    "target",
    "score",
    "rank",
    "allocation",
    "weight",
    "评分",
    "换仓",
    "目标",
    "候选",
    "低分",
    "持仓",
    "权重",
    "分配",
)


class SystemFactorAttributionError(ValueError):
    """Raised when a generated report-only artifact is internally invalid."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _stable_regular_file_bytes(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not_a_regular_file")
        if before.st_nlink != 1:
            raise ValueError("hard_linked_file")
        if before.st_size > max_bytes:
            raise ValueError("file_too_large")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError("file_changed_during_read")
    finally:
        os.close(descriptor)
    path_after = os.lstat(path)
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if after_identity != path_identity:
        raise ValueError("file_path_identity_changed")
    return b"".join(chunks)


def _decode_object(payload: bytes) -> dict[str, Any]:
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non_finite_json_value:{token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("json_root_not_object")
    return value


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _v3_financial_seal_valid(manifest: Mapping[str, Any]) -> bool:
    ledger_sha256 = str(
        manifest.get("ledger_after_manual_switch_csv_sha256")
        or manifest.get("next_ledger_sha256")
        or ""
    ).strip()
    declared_sha256 = str(manifest.get("financial_state_sha256") or "").strip()
    if not _is_sha256(ledger_sha256) or not _is_sha256(declared_sha256):
        return False
    financial_state = {
        key: manifest.get(key)
        for key in (
            "capital_cny",
            "cash_after",
            "market_value_after",
            "total_value_after",
            "portfolio_pnl_after",
            "portfolio_return_after",
        )
    }
    financial_state["ledger_sha256"] = ledger_sha256
    expected_sha256 = hashlib.sha256(
        json.dumps(
            financial_state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return declared_sha256 == expected_sha256


def _manifest_trade_date(run_id: str, manifest: Mapping[str, Any]) -> str:
    for value in (
        manifest.get("recorded_at"),
        manifest.get("record_timestamp"),
        run_id,
    ):
        text = str(value or "").strip()
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
        digits = "".join(character for character in text if character.isdigit())
        if len(digits) >= 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _classify_reason(reason: str) -> str:
    normalized = reason.strip().lower()
    if any(marker in normalized for marker in _RISK_REASON_MARKERS):
        return "risk_control"
    if any(marker in normalized for marker in _FACTOR_REASON_MARKERS):
        return "factor_signal"
    if any(marker in normalized for marker in _PORTFOLIO_REASON_MARKERS):
        return "portfolio_decision"
    return "insufficient_evidence"


def _empty_bucket() -> dict[str, Any]:
    return {
        "realized_sell_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "flat_count": 0,
        "realized_pnl": 0.0,
        "hit_rate": None,
        "examples": [],
    }


def _historical_trade_attribution(
    *,
    base_dir: Path,
    current_run_id: str,
) -> dict[str, Any]:
    buckets = {name: _empty_bucket() for name in ATTRIBUTION_BUCKETS}
    valid_manifest_count = 0
    skipped = Counter()
    integrity_levels = Counter()
    schema_versions: set[str] = set()
    evidence_rows: list[dict[str, Any]] = []
    if not base_dir.is_dir() or base_dir.is_symlink():
        return {
            "status": "unavailable",
            "classification_method": "explicit_manual_manifest_reason_code_markers",
            "records_scanned": 0,
            "valid_manifest_count": 0,
            "skipped_manifest_count": 0,
            "skipped_reasons": {"record_root_unavailable": 1},
            "manifest_schema_versions": [],
            "manifest_integrity_levels": {},
            "evidence_quality": "unavailable",
            "realized_sell_count": 0,
            "buckets": buckets,
            "causal_limit": (
                "Bucket labels describe documented execution reasons only; "
                "they do not prove factor causality or factor validity."
            ),
        }

    record_dirs = [
        path
        for path in sorted(base_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir() and not path.is_symlink() and path.name != current_run_id
    ]
    for record_dir in record_dirs:
        manifest_path = record_dir / "manual_execution_manifest.json"
        if not manifest_path.exists():
            skipped["manual_execution_manifest_missing"] += 1
            continue
        try:
            payload_bytes = _stable_regular_file_bytes(
                manifest_path,
                max_bytes=MAX_MANIFEST_BYTES,
            )
            manifest = _decode_object(payload_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            skipped[f"manual_execution_manifest_invalid:{type(exc).__name__}"] += 1
            continue
        schema_version = str(manifest.get("schema_version") or "")
        if schema_version not in SUPPORTED_MANUAL_MANIFEST_SCHEMAS:
            skipped["manual_execution_manifest_schema_unsupported"] += 1
            continue
        if schema_version == "cn_aggressive_manual_execution.v3" and not _v3_financial_seal_valid(
            manifest
        ):
            skipped["v3_manifest_financial_seal_invalid"] += 1
            continue
        applied_trades = manifest.get("applied_local_trades")
        if not isinstance(applied_trades, list):
            skipped["applied_local_trades_not_list"] += 1
            continue
        valid_manifest_count += 1
        schema_versions.add(schema_version)
        if schema_version == "cn_aggressive_manual_execution.v3":
            integrity_levels["sealed_v3_manifest"] += 1
        else:
            integrity_levels["legacy_unsealed_manifest"] += 1
        manifest_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        for raw_trade in applied_trades:
            if not isinstance(raw_trade, Mapping):
                skipped["applied_trade_not_object"] += 1
                continue
            action = str(raw_trade.get("action") or "").strip().lower()
            if "sell" not in action:
                continue
            realized_pnl = _safe_float(raw_trade.get("realized_pnl"))
            if realized_pnl is None:
                skipped["realized_sell_pnl_missing"] += 1
                continue
            reason = str(raw_trade.get("reason") or "").strip()
            category = _classify_reason(reason)
            evidence_rows.append(
                {
                    "run_id": record_dir.name,
                    "trade_date": _manifest_trade_date(record_dir.name, manifest),
                    "symbol": str(raw_trade.get("symbol") or "").strip().upper(),
                    "name": str(raw_trade.get("name") or "").strip() or "UNKNOWN_NAME",
                    "category": category,
                    "realized_pnl": round(realized_pnl, 2),
                    "reason": reason,
                    "manual_manifest_path": str(manifest_path),
                    "manual_manifest_sha256": manifest_sha256,
                }
            )

    for row in evidence_rows:
        bucket = buckets[row["category"]]
        realized_pnl = float(row["realized_pnl"])
        bucket["realized_sell_count"] += 1
        bucket["realized_pnl"] = round(
            float(bucket["realized_pnl"]) + realized_pnl,
            2,
        )
        if realized_pnl > 0:
            bucket["positive_count"] += 1
        elif realized_pnl < 0:
            bucket["negative_count"] += 1
        else:
            bucket["flat_count"] += 1
    for category, bucket in buckets.items():
        count = int(bucket["realized_sell_count"])
        bucket["hit_rate"] = round(int(bucket["positive_count"]) / count, 6) if count else None
        bucket_rows = [row for row in reversed(evidence_rows) if row["category"] == category]
        bucket["examples"] = bucket_rows[:5]

    return {
        "status": "available" if valid_manifest_count else "unavailable",
        "classification_method": "explicit_manual_manifest_reason_code_markers",
        "records_scanned": len(record_dirs),
        "valid_manifest_count": valid_manifest_count,
        "skipped_manifest_count": sum(skipped.values()),
        "skipped_reasons": dict(sorted(skipped.items())),
        "manifest_schema_versions": sorted(schema_versions),
        "manifest_integrity_levels": dict(sorted(integrity_levels.items())),
        "evidence_quality": (
            "legacy_manifest_limited"
            if integrity_levels.get("legacy_unsealed_manifest")
            else ("sealed_v3" if valid_manifest_count else "unavailable")
        ),
        "realized_sell_count": len(evidence_rows),
        "buckets": buckets,
        "causal_limit": (
            "Bucket labels describe documented execution reasons only; "
            "a single trade or bucket P&L does not prove factor causality, "
            "factor validity, or factor invalidity."
        ),
    }


def _factor_rows(
    readiness: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    raw_factors = readiness.get("factors")
    factors = raw_factors if isinstance(raw_factors, list) else []
    raw_weights = readiness.get("normalized_abs_weights")
    weights = raw_weights if isinstance(raw_weights, Mapping) else {}
    raw_families = readiness.get("family_normalized_abs_weights")
    families = raw_families if isinstance(raw_families, Mapping) else {}
    rows: list[dict[str, Any]] = []
    names_seen: set[str] = set()
    for raw_factor in factors:
        if isinstance(raw_factor, Mapping):
            name = str(raw_factor.get("name") or "").strip()
            family = str(raw_factor.get("family") or "").strip()
            health = str(raw_factor.get("health_status") or raw_factor.get("health") or "").strip()
        else:
            name = str(raw_factor or "").strip()
            family = ""
            health = ""
        if not name:
            continue
        names_seen.add(name)
        rows.append(
            {
                "name": name,
                "family": family,
                "health_status": health,
                "normalized_abs_weight": _safe_float(weights.get(name)),
            }
        )
    for name, raw_weight in weights.items():
        factor_name = str(name).strip()
        if not factor_name or factor_name in names_seen:
            continue
        rows.append(
            {
                "name": factor_name,
                "family": "",
                "health_status": "",
                "normalized_abs_weight": _safe_float(raw_weight),
            }
        )
    rows.sort(
        key=lambda row: (
            -(
                float(row["normalized_abs_weight"])
                if row["normalized_abs_weight"] is not None
                else -1.0
            ),
            row["name"],
        )
    )
    return rows, {
        str(name): _safe_float(value)
        for name, value in sorted(families.items(), key=lambda item: str(item[0]))
    }


def _factor_v4_analysis(path: Path) -> dict[str, Any]:
    try:
        readiness = read_factor_readiness(path)
        payload_bytes = _stable_regular_file_bytes(
            path,
            max_bytes=MAX_MANIFEST_BYTES,
        )
        reread_payload = _decode_object(payload_bytes)
        if reread_payload != readiness:
            raise FactorReadinessStoreError("Factor readiness changed between validated reads")
    except (
        FactorReadinessStoreError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return {
            "status": "unavailable_fail_closed",
            "source": {
                "path": str(path),
                "sha256": "",
                "semantic_sha256": "",
            },
            "protocol_version": "v4",
            "readiness_status": "unavailable",
            "factor_governance_ready": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
            "production_factor_target": 10,
            "production_factor_count": 0,
            "healthy_factor_count": 0,
            "production_family_count": 0,
            "factors": [],
            "family_normalized_abs_weights": {},
            "blockers": [f"factor_v4_readiness_unavailable:{type(exc).__name__}"],
            "contribution_evidence": {
                "status": "unavailable",
                "reason": "authoritative_factor_v4_contribution_evidence_missing",
            },
            "factor_verdict": "insufficient_evidence",
            "legacy_or_shadow_fallback_used": False,
        }
    factor_rows, family_weights = _factor_rows(readiness)
    production_factor_count = int(readiness.get("production_factor_count") or 0)
    factor_governance_ready = readiness.get("factor_governance_ready") is True
    factor_verdict = (
        "production_factor_set_ready_observation_only"
        if factor_governance_ready and production_factor_count > 0
        else "no_production_factor_verdict"
    )
    return {
        "status": "available",
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "semantic_sha256": semantic_sha256(readiness),
            "schema_version": str(readiness.get("schema_version") or ""),
            "as_of": str(readiness.get("as_of") or ""),
        },
        "protocol_version": str(readiness.get("protocol_version") or "v4"),
        "protocol_hash": str(readiness.get("protocol_hash") or ""),
        "readiness_status": str(readiness.get("status") or "unknown"),
        "factor_governance_ready": factor_governance_ready,
        "new_risk_authorized": readiness.get("new_risk_authorized") is True,
        "production_apply_enabled": readiness.get("production_apply_enabled") is True,
        "production_apply_blocker": str(readiness.get("production_apply_blocker") or ""),
        "production_factor_target": int(readiness.get("production_factor_target") or 10),
        "production_factor_count": production_factor_count,
        "healthy_factor_count": int(readiness.get("healthy_factor_count") or 0),
        "production_family_count": int(readiness.get("production_family_count") or 0),
        "factors": factor_rows,
        "family_normalized_abs_weights": family_weights,
        "blockers": [
            str(item) for item in list(readiness.get("blockers") or []) if str(item).strip()
        ],
        "contribution_evidence": {
            "status": "unavailable",
            "reason": "authoritative_factor_v4_contribution_evidence_missing",
        },
        "factor_verdict": factor_verdict,
        "legacy_or_shadow_fallback_used": False,
    }


def _today_diagnosis(
    *,
    completeness_passed: bool,
    decision_data_sufficient: bool,
    action_taken_today: bool,
    execution_rejections: Sequence[Mapping[str, Any]],
    factor_v4: Mapping[str, Any],
) -> dict[str, Any]:
    rejection_count = len(execution_rejections)
    if not completeness_passed or not decision_data_sufficient:
        primary_driver = "market_data_staleness"
    elif rejection_count:
        primary_driver = "execution"
    elif action_taken_today:
        primary_driver = "portfolio_decision"
    else:
        primary_driver = "insufficient_evidence"
    return {
        "primary_driver": primary_driver,
        "completeness_passed": bool(completeness_passed),
        "decision_data_sufficient": bool(decision_data_sufficient),
        "action_taken_today": bool(action_taken_today),
        "execution_rejection_count": rejection_count,
        "factor_verdict": str(factor_v4.get("factor_verdict") or "insufficient_evidence"),
        "interpretation": (
            "Factor verdict is governed only by Factor v4 readiness and "
            "authoritative contribution evidence; execution outcomes cannot "
            "promote or invalidate a factor."
        ),
    }


def build_system_factor_attribution(
    *,
    project_root: Path,
    base_dir: Path,
    run_id: str,
    generated_at: str,
    trade_date: str,
    analysis_trade_date: str,
    completeness_passed: bool,
    decision_data_sufficient: bool,
    action_taken_today: bool,
    execution_rejections: Sequence[Mapping[str, Any]],
    factor_readiness_path: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic, nonauthorizing daily analysis payload."""

    readiness_path = factor_readiness_path or (project_root / FACTOR_READINESS_RELATIVE_PATH)
    factor_v4 = _factor_v4_analysis(readiness_path)
    historical = _historical_trade_attribution(
        base_dir=base_dir,
        current_run_id=run_id,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "run_id": run_id,
        "generated_at": generated_at,
        "trade_date": trade_date,
        "analysis_trade_date": analysis_trade_date,
        "report_only": True,
        "production_authority": False,
        "source_policy": {
            "historical_execution_source": (
                "contained manual_execution_manifest.json applied_local_trades only"
            ),
            "ledger_csv_fallback_used": False,
            "factor_source": "FactorGovernanceProtocol v4 readiness only",
            "legacy_or_shadow_factor_fallback_used": False,
            "external_provider_used": False,
        },
        "historical_trade_attribution": historical,
        "factor_v4_analysis": factor_v4,
        "today_diagnosis": _today_diagnosis(
            completeness_passed=completeness_passed,
            decision_data_sufficient=decision_data_sufficient,
            action_taken_today=action_taken_today,
            execution_rejections=execution_rejections,
            factor_v4=factor_v4,
        ),
        "execution_boundary": {
            "broker_connected_by_analysis": False,
            "order_created_by_analysis": False,
            "trade_executed_by_analysis": False,
        },
    }
    validate_system_factor_attribution(payload)
    return payload


def validate_system_factor_attribution(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemFactorAttributionError("schema_version_mismatch")
    if payload.get("report_only") is not True:
        raise SystemFactorAttributionError("report_only_must_be_true")
    if payload.get("production_authority") is not False:
        raise SystemFactorAttributionError("production_authority_must_be_false")
    source_policy = payload.get("source_policy")
    if not isinstance(source_policy, Mapping):
        raise SystemFactorAttributionError("source_policy_missing")
    if source_policy.get("ledger_csv_fallback_used") is not False:
        raise SystemFactorAttributionError("ledger_csv_fallback_forbidden")
    if source_policy.get("legacy_or_shadow_factor_fallback_used") is not False:
        raise SystemFactorAttributionError("legacy_factor_fallback_forbidden")
    factor_v4 = payload.get("factor_v4_analysis")
    if not isinstance(factor_v4, Mapping):
        raise SystemFactorAttributionError("factor_v4_analysis_missing")
    if factor_v4.get("legacy_or_shadow_fallback_used") is not False:
        raise SystemFactorAttributionError("factor_v4_fallback_forbidden")
    execution_boundary = payload.get("execution_boundary")
    if not isinstance(execution_boundary, Mapping) or any(
        execution_boundary.get(field) is not False
        for field in (
            "broker_connected_by_analysis",
            "order_created_by_analysis",
            "trade_executed_by_analysis",
        )
    ):
        raise SystemFactorAttributionError("execution_boundary_invalid")


def system_factor_attribution_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_system_factor_attribution(payload)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def system_factor_attribution_reference(
    payload: Mapping[str, Any],
    *,
    path: str = "system_factor_attribution.json",
) -> dict[str, str]:
    return {
        "path": path,
        "sha256": hashlib.sha256(system_factor_attribution_bytes(payload)).hexdigest(),
        "schema_version": SCHEMA_VERSION,
    }


def _format_money(value: Any) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:+,.2f}"


def render_system_factor_attribution_markdown(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return Factor v4 and historical-attribution Markdown lines."""

    validate_system_factor_attribution(payload)
    factor = payload["factor_v4_analysis"]
    source = factor.get("source") if isinstance(factor.get("source"), Mapping) else {}
    factor_lines = [
        (
            f"- v4 readiness：status=`{factor.get('readiness_status', 'unavailable')}`；"
            f"factor_governance_ready=`{str(bool(factor.get('factor_governance_ready'))).lower()}`；"
            f"new_risk_authorized=`{str(bool(factor.get('new_risk_authorized'))).lower()}`；"
            f"production_apply_enabled=`{str(bool(factor.get('production_apply_enabled'))).lower()}`"
        ),
        (
            f"- 因子槽位：production=`{int(factor.get('production_factor_count') or 0)}`/"
            f"`{int(factor.get('production_factor_target') or 10)}`；"
            f"healthy=`{int(factor.get('healthy_factor_count') or 0)}`；"
            f"families=`{int(factor.get('production_family_count') or 0)}`"
        ),
        (
            f"- 权威输入：`{source.get('path') or 'UNAVAILABLE'}`；"
            f"SHA256=`{source.get('sha256') or 'UNAVAILABLE'}`；"
            f"as_of=`{source.get('as_of') or 'UNAVAILABLE'}`"
        ),
        (
            f"- 因子贡献证据：`{factor.get('contribution_evidence', {}).get('status', 'unavailable')}`；"
            f"原因 `{factor.get('contribution_evidence', {}).get('reason', 'missing')}`。"
        ),
        (
            f"- Factor verdict：`{factor.get('factor_verdict', 'insufficient_evidence')}`；"
            "单笔/少量交易盈亏不用于判定因子有效或失效。"
        ),
    ]
    factor_rows = list(factor.get("factors") or [])
    if factor_rows:
        factor_lines.extend(
            [
                "",
                "| Factor v4 因子 | Family | 权重 | Health |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for row in factor_rows:
            weight = _safe_float(row.get("normalized_abs_weight"))
            factor_lines.append(
                f"| `{row.get('name') or 'UNKNOWN_FACTOR'}` | "
                f"`{row.get('family') or 'UNCONFIRMED'}` | "
                f"{'N/A' if weight is None else f'{weight:.2%}'} | "
                f"`{row.get('health_status') or 'UNCONFIRMED'}` |"
            )
    else:
        factor_lines.append(
            "- 当前没有可展示的生产 Factor v4 因子；不借用 legacy/shadow/challenger 补齐。"
        )
    blockers = list(factor.get("blockers") or [])
    factor_lines.append(
        "- blockers：`" + ("；".join(str(item) for item in blockers) or "none") + "`"
    )

    history = payload["historical_trade_attribution"]
    diagnosis = payload["today_diagnosis"]
    history_lines = [
        (
            f"- 今日主因：`{diagnosis.get('primary_driver', 'insufficient_evidence')}`；"
            f"Factor verdict=`{diagnosis.get('factor_verdict', 'insufficient_evidence')}`；"
            f"execution rejections=`{int(diagnosis.get('execution_rejection_count') or 0)}`。"
        ),
        (
            f"- 历史证据：valid manifests=`{int(history.get('valid_manifest_count') or 0)}`/"
            f"`{int(history.get('records_scanned') or 0)}`；"
            f"realized sells=`{int(history.get('realized_sell_count') or 0)}`；"
            f"quality=`{history.get('evidence_quality') or 'unavailable'}`；"
            "仅使用 manifest 内 `applied_local_trades`，无 `ledger.csv` fallback。"
        ),
    ]
    for category in ATTRIBUTION_BUCKETS:
        bucket = history.get("buckets", {}).get(category, {})
        examples = list(bucket.get("examples") or [])
        example_text = (
            "；".join(
                (
                    f"{row.get('symbol') or 'UNKNOWN_SYMBOL'} "
                    f"{row.get('name') or 'UNKNOWN_NAME'} "
                    f"{row.get('trade_date') or 'UNCONFIRMED'} "
                    f"{_format_money(row.get('realized_pnl'))}"
                )
                for row in examples[:3]
            )
            or "无"
        )
        hit_rate = _safe_float(bucket.get("hit_rate"))
        history_lines.append(
            f"- `{category}`：realized sells=`{int(bucket.get('realized_sell_count') or 0)}`；"
            f"PNL=`{_format_money(bucket.get('realized_pnl'))}`；"
            f"hit rate=`{'N/A' if hit_rate is None else f'{hit_rate:.2%}'}`；"
            f"例证：{example_text}。"
        )
    history_lines.extend(
        [
            (
                "- 因果边界：归类只描述已记录的执行 reason；"
                "同一笔或少量交易不能证明系统/因子有效或失效。"
            ),
            ("- 分析边界：report-only，无外部 provider、无 broker、" "不创建订单、不执行交易。"),
        ]
    )
    return factor_lines, history_lines


__all__ = [
    "ATTRIBUTION_BUCKETS",
    "FACTOR_READINESS_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "SystemFactorAttributionError",
    "build_system_factor_attribution",
    "render_system_factor_attribution_markdown",
    "system_factor_attribution_bytes",
    "system_factor_attribution_reference",
    "validate_system_factor_attribution",
]
