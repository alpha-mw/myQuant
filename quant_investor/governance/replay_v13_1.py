from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "myquant.joint_replay_gate.v1"
FREEZE_EXCEPTION_CYCLE_ID = "myquant-v13.1-freeze-exception"
REQUIRED_REPLAY_SCENARIOS = (
    "industry_baseline",
    "theme_v2_observer",
    "theme_v2_formal_gate",
    "factor_protocol_v2",
    "theme_v2_plus_factor_v2",
)
MIN_THEME_LIVE_SHADOW_DAYS = 20
# A verifier must never substitute for the historical five-path DAG producer.
# This remains false until that producer exists and is covered by integration
# evidence; therefore Theme/Factor production activation is intentionally
# impossible in the current freeze-exception branch.
CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE = False
SCENARIO_EVIDENCE_SCHEMA_VERSION = "myquant.replay_scenario_evidence.v1"
SHADOW_EVIDENCE_SCHEMA_VERSION = "myquant.theme_live_shadow_evidence.v1"
ACCEPTANCE_EVIDENCE_SCHEMA_VERSION = "myquant.v13_1_acceptance_evidence.v1"
REQUIRED_SCENARIO_CHECKS = (
    "pit_readback_pass",
    "train_pass",
    "validation_pass",
    "holdout_pass",
    "offline_only",
    "no_network",
    "no_llm",
    "no_broker",
    "no_registry_mutation",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _payload_sha256(
    payload: Mapping[str, Any],
    *,
    excluded: tuple[str, ...] = ("evidence_sha256",),
) -> str:
    return _sha256(
        {
            str(key): value
            for key, value in payload.items()
            if str(key) not in excluded
            and not str(key).startswith("_artifact_")
        }
    )


def _date_key(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if not text:
        return ""
    digits = "".join(
        character for character in text[:10] if character.isdigit()
    )
    if len(digits) != 8:
        return ""
    try:
        parsed = datetime.strptime(digits, "%Y%m%d").date()
    except ValueError:
        return ""
    return parsed.isoformat()


@dataclass(frozen=True)
class ReplaySplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    holdout_ratio: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "train": list(self.train),
            "validation": list(self.validation),
            "holdout": list(self.holdout),
        }


@dataclass(frozen=True)
class ThresholdSeal:
    threshold_hash: str
    dataset_sha256: str
    validation_end_date: str
    thresholds: Mapping[str, Any]
    freeze_exception_cycle_id: str = FREEZE_EXCEPTION_CYCLE_ID
    schema_version: str = "myquant.holdout_threshold_seal.v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "threshold_hash": self.threshold_hash,
            "freeze_exception_cycle_id": self.freeze_exception_cycle_id,
            "dataset_sha256": self.dataset_sha256,
            "validation_end_date": self.validation_end_date,
            "thresholds": dict(self.thresholds),
        }


def build_replay_split(trade_dates: Iterable[Any]) -> ReplaySplit:
    """Return a deterministic chronological 60/20/20 split.

    Duplicate same-day snapshots never expand a split. At least three distinct
    trading dates are required so validation and final holdout both remain
    non-empty.
    """

    normalized = sorted(
        {_date_key(value) for value in trade_dates if _date_key(value)}
    )
    if len(normalized) < 3:
        raise ValueError(
            "at least three distinct valid trading dates are required"
        )

    count = len(normalized)
    train_count = max(1, int(count * 0.60))
    validation_count = max(1, int(count * 0.20))
    if train_count + validation_count >= count:
        train_count = max(1, count - 2)
        validation_count = 1

    train_end = train_count
    validation_end = train_end + validation_count
    return ReplaySplit(
        train=tuple(normalized[:train_end]),
        validation=tuple(normalized[train_end:validation_end]),
        holdout=tuple(normalized[validation_end:]),
    )


def build_threshold_seal(
    *,
    thresholds: Mapping[str, Any],
    dataset_sha256: str,
    validation_end_date: Any,
) -> ThresholdSeal:
    dataset_hash = str(dataset_sha256 or "").strip().lower()
    if len(dataset_hash) != 64 or any(
        ch not in "0123456789abcdef" for ch in dataset_hash
    ):
        raise ValueError(
            "dataset_sha256 must be a lowercase 64-character SHA-256"
        )
    end_date = _date_key(validation_end_date)
    if not end_date:
        raise ValueError("validation_end_date must be a valid date")
    threshold_payload = dict(thresholds or {})
    return ThresholdSeal(
        threshold_hash=_sha256(threshold_payload),
        dataset_sha256=dataset_hash,
        validation_end_date=end_date,
        thresholds=threshold_payload,
    )


def validate_threshold_seal(
    seal: ThresholdSeal | Mapping[str, Any],
    *,
    current_thresholds: Mapping[str, Any],
    dataset_sha256: str,
    holdout_opened: bool,
    expected_threshold_hash: str = "",
) -> list[str]:
    payload = (
        seal.to_dict()
        if isinstance(seal, ThresholdSeal)
        else dict(seal or {})
    )
    blockers: list[str] = []
    if payload.get("schema_version") != "myquant.holdout_threshold_seal.v2":
        blockers.append("holdout_threshold_seal_schema_mismatch")
    if payload.get("freeze_exception_cycle_id") != FREEZE_EXCEPTION_CYCLE_ID:
        blockers.append("holdout_freeze_exception_cycle_mismatch")
    sealed_hash = str(payload.get("threshold_hash") or "")
    current_hash = _sha256(dict(current_thresholds or {}))
    if sealed_hash != current_hash:
        blockers.append("holdout_thresholds_changed_after_seal")
    sealed_dataset_hash = str(payload.get("dataset_sha256") or "").lower()
    current_dataset_hash = str(dataset_sha256 or "").lower()
    if sealed_dataset_hash != current_dataset_hash:
        blockers.append("holdout_dataset_hash_changed_after_seal")
    expected = str(expected_threshold_hash or "").strip().lower()
    if holdout_opened and not expected:
        blockers.append("expected_threshold_hash_required_when_holdout_opened")
    elif expected and expected != sealed_hash:
        blockers.append("expected_threshold_hash_mismatch")
    return blockers


def _strict_true(payload: Mapping[str, Any], key: str) -> bool:
    return payload.get(key) is True


def build_activation_decision(
    acceptance: Mapping[str, Any],
    *,
    distinct_theme_shadow_days: int,
    dashboard_external_blockers: Iterable[str] = (),
    theme_external_blockers: Iterable[str] = (),
    factor_external_blockers: Iterable[str] = (),
    joint_external_blockers: Iterable[str] = (),
) -> dict[str, Any]:
    """Build independent, fail-closed production decisions for each surface."""

    dashboard = dict(acceptance.get("dashboard") or {})
    theme = dict(acceptance.get("theme") or {})
    factor = dict(acceptance.get("factor") or {})
    dag = dict(acceptance.get("dag") or {})

    dashboard_blockers: list[str] = [
        str(item) for item in dashboard_external_blockers if str(item)
    ]
    for key in (
        "p0_clear",
        "attribution_reconciled",
        "private_data_boundary_pass",
    ):
        if not _strict_true(dashboard, key):
            dashboard_blockers.append(f"dashboard:{key}")

    theme_blockers: list[str] = [
        str(item) for item in theme_external_blockers if str(item)
    ]
    for key in (
        "coverage_pass",
        "pit_pass",
        "forced_admission_removed",
        "rollback_pass",
    ):
        if not _strict_true(theme, key):
            theme_blockers.append(f"theme:{key}")
    if int(theme.get("forced_theme_count", -1)) != 0:
        theme_blockers.append("theme:forced_theme_count_not_zero")
    if distinct_theme_shadow_days < MIN_THEME_LIVE_SHADOW_DAYS:
        theme_blockers.append(
            "theme:insufficient_distinct_live_shadow_days:"
            f"{distinct_theme_shadow_days}<{MIN_THEME_LIVE_SHADOW_DAYS}"
        )

    factor_blockers: list[str] = [
        str(item) for item in factor_external_blockers if str(item)
    ]
    for key in (
        "targeted_transition_pass",
        "idempotent_readback_pass",
        "rollback_pass",
    ):
        if not _strict_true(factor, key):
            factor_blockers.append(f"factor:{key}")

    dag_blockers: list[str] = [
        str(item) for item in joint_external_blockers if str(item)
    ]
    if not _strict_true(dag, "offline_replay_pass"):
        dag_blockers.append("dag:offline_replay_pass")

    dashboard_blockers = list(dict.fromkeys(dashboard_blockers))
    theme_blockers = list(dict.fromkeys(theme_blockers))
    factor_blockers = list(dict.fromkeys(factor_blockers))
    dag_blockers = list(dict.fromkeys(dag_blockers))
    return {
        "dashboard": {
            "enabled": not dashboard_blockers,
            "blockers": dashboard_blockers,
        },
        "theme_formal": {
            "enabled": not theme_blockers and not dag_blockers,
            "fallback": "observer_only",
            "blockers": theme_blockers + dag_blockers,
        },
        "factor_transitions": {
            "enabled": not factor_blockers and not dag_blockers,
            "fallback": "governance_blocked",
            "blockers": factor_blockers + dag_blockers,
        },
        "joint_path": {
            "enabled": not (
                dashboard_blockers
                + theme_blockers
                + factor_blockers
                + dag_blockers
            ),
            "blockers": (
                dashboard_blockers
                + theme_blockers
                + factor_blockers
                + dag_blockers
            ),
        },
    }


def _validate_acceptance_evidence(
    acceptance: Mapping[str, Any],
    *,
    dataset_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    payload = dict(acceptance or {})
    blockers: list[str] = []
    if payload.get("schema_version") != ACCEPTANCE_EVIDENCE_SCHEMA_VERSION:
        blockers.append("acceptance_evidence_schema_invalid")
    if payload.get("_artifact_readback_verified") is not True:
        blockers.append("acceptance_evidence_readback_unverified")
    if not _is_sha256(payload.get("_artifact_sha256")):
        blockers.append("acceptance_artifact_sha256_missing")
    if str(payload.get("dataset_sha256") or "").lower() != str(
        dataset_sha256 or ""
    ).lower():
        blockers.append("acceptance_dataset_hash_mismatch")
    evidence_hash = str(payload.get("evidence_sha256") or "").lower()
    if not _is_sha256(evidence_hash) or evidence_hash != _payload_sha256(payload):
        blockers.append("acceptance_evidence_hash_mismatch")
    return payload, blockers


def _validate_scenario_evidence(
    *,
    name: str,
    result: Mapping[str, Any],
    dataset_sha256: str,
    split_sha256: str,
    trade_dates_sha256: str,
    protocol_hashes: Mapping[str, Any],
    metric_thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    payload = dict(result or {})
    blockers: list[str] = []
    if payload.get("schema_version") != SCENARIO_EVIDENCE_SCHEMA_VERSION:
        blockers.append("scenario_evidence_schema_invalid")
    if str(payload.get("scenario") or "") != name:
        blockers.append("scenario_name_mismatch")
    if payload.get("_artifact_readback_verified") is not True:
        blockers.append("scenario_artifact_readback_unverified")
    if not _is_sha256(payload.get("_artifact_sha256")):
        blockers.append("scenario_artifact_sha256_missing")
    if str(payload.get("dataset_sha256") or "").lower() != str(
        dataset_sha256 or ""
    ).lower():
        blockers.append("scenario_dataset_hash_mismatch")
    if str(payload.get("split_sha256") or "").lower() != split_sha256:
        blockers.append("scenario_split_hash_mismatch")
    if (
        str(payload.get("trade_dates_sha256") or "").lower()
        != trade_dates_sha256
    ):
        blockers.append("scenario_trade_dates_hash_mismatch")
    if not _is_sha256(payload.get("source_snapshot_sha256")):
        blockers.append("scenario_source_snapshot_hash_missing")
    evidence_hash = str(payload.get("evidence_sha256") or "").lower()
    if not _is_sha256(evidence_hash) or evidence_hash != _payload_sha256(payload):
        blockers.append("scenario_evidence_hash_mismatch")

    result_protocols = dict(payload.get("protocol_hashes") or {})
    for protocol_name, expected_hash in dict(protocol_hashes or {}).items():
        if str(result_protocols.get(protocol_name) or "") != str(
            expected_hash or ""
        ):
            blockers.append(f"scenario_protocol_hash_mismatch:{protocol_name}")

    checks = dict(payload.get("checks") or {})
    for check in REQUIRED_SCENARIO_CHECKS:
        if checks.get(check) is not True:
            blockers.append(f"scenario_check_failed:{check}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        blockers.append("scenario_metrics_missing")
        metrics = {}
    required_metrics = dict(metric_thresholds or {})
    if not required_metrics:
        blockers.append("scenario_metric_thresholds_missing")
    computed_metric_checks: dict[str, bool] = {}
    for metric_name, raw_rule in required_metrics.items():
        try:
            actual = float(metrics.get(metric_name))
        except (TypeError, ValueError):
            computed_metric_checks[str(metric_name)] = False
            blockers.append(f"scenario_metric_missing:{metric_name}")
            continue
        rule = (
            dict(raw_rule)
            if isinstance(raw_rule, Mapping)
            else {"min": raw_rule}
        )
        passed = True
        if rule.get("min") is not None:
            passed = passed and actual >= float(rule["min"])
        if rule.get("max") is not None:
            passed = passed and actual <= float(rule["max"])
        computed_metric_checks[str(metric_name)] = passed
        if not passed:
            blockers.append(f"scenario_metric_threshold_failed:{metric_name}")

    blockers = list(dict.fromkeys(blockers))
    audit = {
        **{
            key: value
            for key, value in payload.items()
            if key != "_artifact_path"
        },
        "passed": not blockers,
        "blockers": blockers,
        "computed_metric_checks": computed_metric_checks,
    }
    return audit, blockers


def _validate_shadow_evidence(
    observations: Iterable[Any],
    *,
    dataset_sha256: str,
    theme_protocol_hash: str,
    trade_dates: set[str],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    valid_dates: list[str] = []
    audit: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, raw in enumerate(observations):
        payload = dict(raw) if isinstance(raw, Mapping) else {}
        item_blockers: list[str] = []
        trade_date = _date_key(payload.get("trade_date"))
        if payload.get("schema_version") != SHADOW_EVIDENCE_SCHEMA_VERSION:
            item_blockers.append("shadow_evidence_schema_invalid")
        if not trade_date:
            item_blockers.append("shadow_trade_date_invalid")
        elif trade_date not in trade_dates:
            item_blockers.append("shadow_trade_date_not_in_pit_dataset")
        if payload.get("_artifact_readback_verified") is not True:
            item_blockers.append("shadow_artifact_readback_unverified")
        if not _is_sha256(payload.get("_artifact_sha256")):
            item_blockers.append("shadow_artifact_sha256_missing")
        if not _is_sha256(payload.get("snapshot_sha256")):
            item_blockers.append("shadow_snapshot_sha256_missing")
        if str(payload.get("dataset_sha256") or "").lower() != str(
            dataset_sha256 or ""
        ).lower():
            item_blockers.append("shadow_dataset_hash_mismatch")
        if str(payload.get("protocol_hash") or "") != str(
            theme_protocol_hash or ""
        ):
            item_blockers.append("shadow_protocol_hash_mismatch")
        if payload.get("pit_verified") is not True:
            item_blockers.append("shadow_pit_unverified")
        evidence_hash = str(payload.get("evidence_sha256") or "").lower()
        if (
            not _is_sha256(evidence_hash)
            or evidence_hash != _payload_sha256(payload)
        ):
            item_blockers.append("shadow_evidence_hash_mismatch")
        item_blockers = list(dict.fromkeys(item_blockers))
        if not item_blockers and trade_date:
            valid_dates.append(trade_date)
        else:
            blockers.extend(
                f"theme_shadow[{index}]:{item}" for item in item_blockers
            )
        audit.append(
            {
                **{
                    key: value
                    for key, value in payload.items()
                    if key != "_artifact_path"
                },
                "trade_date": trade_date,
                "verified": not item_blockers,
                "blockers": item_blockers,
            }
        )
    return sorted(set(valid_dates)), audit, list(dict.fromkeys(blockers))


def build_joint_replay_manifest(
    *,
    run_id: str,
    trade_dates: Iterable[Any],
    dataset_sha256: str,
    protocol_hashes: Mapping[str, Any],
    scenario_results: Mapping[str, Mapping[str, Any]],
    theme_shadow_dates: Iterable[Any],
    threshold_seal: ThresholdSeal | Mapping[str, Any],
    current_thresholds: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    holdout_opened: bool = False,
    expected_threshold_hash: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized_trade_dates = sorted(
        {_date_key(value) for value in trade_dates if _date_key(value)}
    )
    split = build_replay_split(normalized_trade_dates)
    split_sha256 = _sha256(split.to_dict())
    trade_dates_sha256 = _sha256(normalized_trade_dates)
    seal_payload = (
        threshold_seal.to_dict()
        if isinstance(threshold_seal, ThresholdSeal)
        else dict(threshold_seal or {})
    )
    threshold_blockers = validate_threshold_seal(
        seal_payload,
        current_thresholds=current_thresholds,
        dataset_sha256=dataset_sha256,
        holdout_opened=holdout_opened,
        expected_threshold_hash=expected_threshold_hash,
    )
    if seal_payload.get("_artifact_readback_verified") is not True:
        threshold_blockers.append("threshold_seal_readback_unverified")
    if not _is_sha256(seal_payload.get("_artifact_sha256")):
        threshold_blockers.append("threshold_seal_artifact_sha256_missing")
    if seal_payload.get("_canonical_seal_ledger_verified") is not True:
        threshold_blockers.append("threshold_seal_canonical_ledger_unverified")
    if not _is_sha256(seal_payload.get("_seal_ledger_sha256")):
        threshold_blockers.append("threshold_seal_ledger_sha256_missing")
    if _date_key(seal_payload.get("validation_end_date")) != split.validation[-1]:
        threshold_blockers.append("threshold_seal_validation_end_mismatch")
    if not holdout_opened:
        threshold_blockers.append("holdout_not_opened")
    threshold_blockers = list(dict.fromkeys(threshold_blockers))

    protocol_blockers = [
        f"protocol_hash_invalid:{name}"
        for name in ("theme_v2", "factor_v2", "dashboard_contract_v2")
        if not _is_sha256(protocol_hashes.get(name))
    ]
    producer_blockers = (
        []
        if CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE
        else ["canonical_joint_replay_producer_not_implemented"]
    )
    scenario_audit: dict[str, Any] = {}
    scenario_blockers: list[str] = []
    scenario_metric_thresholds = dict(
        dict(current_thresholds or {}).get("scenario_metrics") or {}
    )
    for name in REQUIRED_REPLAY_SCENARIOS:
        metric_thresholds = {
            **dict(scenario_metric_thresholds.get("*") or {}),
            **dict(scenario_metric_thresholds.get(name) or {}),
        }
        audit, scenario_item_blockers = _validate_scenario_evidence(
            name=name,
            result=dict(scenario_results.get(name) or {}),
            dataset_sha256=dataset_sha256,
            split_sha256=split_sha256,
            trade_dates_sha256=trade_dates_sha256,
            protocol_hashes=protocol_hashes,
            metric_thresholds=metric_thresholds,
        )
        scenario_blockers.extend(
            f"{name}:{item}" for item in scenario_item_blockers
        )
        scenario_audit[name] = audit

    normalized_shadow_dates, shadow_audit, shadow_blockers = (
        _validate_shadow_evidence(
            theme_shadow_dates,
            dataset_sha256=dataset_sha256,
            theme_protocol_hash=str(protocol_hashes.get("theme_v2") or ""),
            trade_dates=set(normalized_trade_dates),
        )
    )
    acceptance_payload, acceptance_blockers = _validate_acceptance_evidence(
        acceptance,
        dataset_sha256=dataset_sha256,
    )
    acceptance_for_activation = dict(acceptance_payload)
    acceptance_for_activation["dag"] = {
        "offline_replay_pass": not (
            threshold_blockers + protocol_blockers + scenario_blockers
        )
    }

    theme_scenario_blockers = [
        blocker
        for blocker in scenario_blockers
        if blocker.split(":", 1)[0]
        in {
            "theme_v2_observer",
            "theme_v2_formal_gate",
            "theme_v2_plus_factor_v2",
        }
    ]
    factor_scenario_blockers = [
        blocker
        for blocker in scenario_blockers
        if blocker.split(":", 1)[0]
        in {"factor_protocol_v2", "theme_v2_plus_factor_v2"}
    ]
    dashboard_protocol_blockers = [
        blocker
        for blocker in protocol_blockers
        if blocker.endswith(":dashboard_contract_v2")
    ]
    theme_protocol_blockers = [
        blocker
        for blocker in protocol_blockers
        if blocker.endswith(":theme_v2")
    ]
    factor_protocol_blockers = [
        blocker
        for blocker in protocol_blockers
        if blocker.endswith(":factor_v2")
    ]

    activation = build_activation_decision(
        acceptance_for_activation,
        distinct_theme_shadow_days=len(normalized_shadow_dates),
        dashboard_external_blockers=(
            acceptance_blockers + dashboard_protocol_blockers
        ),
        theme_external_blockers=(
            acceptance_blockers
            + producer_blockers
            + threshold_blockers
            + theme_protocol_blockers
            + theme_scenario_blockers
            + shadow_blockers
        ),
        factor_external_blockers=(
            acceptance_blockers
            + producer_blockers
            + threshold_blockers
            + factor_protocol_blockers
            + factor_scenario_blockers
        ),
        joint_external_blockers=(
            acceptance_blockers
            + producer_blockers
            + threshold_blockers
            + protocol_blockers
            + scenario_blockers
            + shadow_blockers
        ),
    )
    blockers = list(
        dict.fromkeys(
            threshold_blockers
            + producer_blockers
            + protocol_blockers
            + scenario_blockers
            + shadow_blockers
            + acceptance_blockers
            + list(activation["joint_path"]["blockers"])
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "freeze_exception_cycle_id": FREEZE_EXCEPTION_CYCLE_ID,
        "run_id": str(run_id or "").strip(),
        "generated_at": generated_at
        or datetime.now(timezone.utc)
        .astimezone()
        .replace(microsecond=0)
        .isoformat(),
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "dataset_sha256": str(dataset_sha256 or "").lower(),
        "protocol_hashes": dict(protocol_hashes or {}),
        "split": split.to_dict(),
        "split_sha256": split_sha256,
        "trade_dates_sha256": trade_dates_sha256,
        "threshold_seal": {
            key: value
            for key, value in seal_payload.items()
            if key != "_artifact_path"
        },
        "holdout_opened": bool(holdout_opened),
        "scenarios": scenario_audit,
        "theme_live_shadow": {
            "distinct_trade_day_count": len(normalized_shadow_dates),
            "required_distinct_trade_day_count": MIN_THEME_LIVE_SHADOW_DAYS,
            "dates": normalized_shadow_dates,
            "observations": shadow_audit,
            "passed": (
                len(normalized_shadow_dates) >= MIN_THEME_LIVE_SHADOW_DAYS
            ),
        },
        "acceptance_evidence": {
            key: value
            for key, value in acceptance_payload.items()
            if key != "_artifact_path"
        },
        "activation": activation,
        "controls": {
            "offline_only": True,
            "no_network": True,
            "no_llm": True,
            "no_broker": True,
            "registry_mutation": False,
            "merge_requires_maxwell_confirmation": True,
        },
    }


def write_manifest_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    sealed_payload = dict(payload)
    sealed_payload.pop("manifest_sha256", None)
    sealed_payload["manifest_sha256"] = _sha256(sealed_payload)
    data = json.dumps(
        sealed_payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        readback = json.loads(target.read_text(encoding="utf-8"))
        readback_hash = str(readback.pop("manifest_sha256", ""))
        if readback_hash != _sha256(readback):
            raise RuntimeError("manifest readback hash mismatch")
        if target.stat().st_mode & 0o777 != 0o600:
            raise RuntimeError("manifest permissions are not 0600")
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def verify_joint_replay_manifest(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_theme_protocol_hash: str,
) -> dict[str, Any]:
    """Verify a persisted canonical joint-replay manifest end to end.

    Runtime activation must consume this verifier rather than trusting a subset
    of self-declared manifest fields.  The original evidence paths stay private,
    but their readback flags, artifact hashes, evidence hashes, sealed split,
    five scenario audits, shadow observations, acceptance evidence, activation
    decisions, and manifest hash are all revalidated here.
    """

    blockers: list[str] = []
    source = Path(path)
    expected_artifact = str(expected_artifact_sha256 or "").strip().lower()
    expected_theme_hash = str(expected_theme_protocol_hash or "").strip().lower()
    if CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE is not True:
        blockers.append("canonical_joint_replay_producer_not_implemented")
        return _manifest_verification_result(blockers=blockers)
    if not source.is_file():
        return _manifest_verification_result(
            blockers=["joint_manifest_missing"]
        )
    if source.stat().st_mode & 0o777 != 0o600:
        blockers.append("joint_manifest_permissions_not_0600")
    try:
        raw = source.read_bytes()
        readback = source.read_bytes()
    except OSError:
        return _manifest_verification_result(
            blockers=blockers + ["joint_manifest_unreadable"]
        )
    if raw != readback:
        blockers.append("joint_manifest_readback_mismatch")
    artifact_sha = hashlib.sha256(raw).hexdigest()
    if not _is_sha256(expected_artifact) or artifact_sha != expected_artifact:
        blockers.append("joint_manifest_artifact_sha256_mismatch")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _manifest_verification_result(
            blockers=blockers + ["joint_manifest_unreadable"],
            artifact_sha256=artifact_sha,
        )
    if not isinstance(manifest, Mapping):
        return _manifest_verification_result(
            blockers=blockers + ["joint_manifest_not_object"],
            artifact_sha256=artifact_sha,
        )
    payload = dict(manifest)
    supplied_manifest_hash = str(payload.pop("manifest_sha256", ""))
    if supplied_manifest_hash != _sha256(payload):
        blockers.append("joint_manifest_self_hash_mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("joint_manifest_schema_mismatch")
    if payload.get("freeze_exception_cycle_id") != FREEZE_EXCEPTION_CYCLE_ID:
        blockers.append("joint_manifest_freeze_exception_cycle_mismatch")
    if payload.get("status") != "ready":
        blockers.append("joint_manifest_not_ready")
    if list(payload.get("blockers") or []):
        blockers.append("joint_manifest_blockers_present")

    dataset_sha = str(payload.get("dataset_sha256") or "").strip().lower()
    if not _is_sha256(dataset_sha):
        blockers.append("joint_manifest_dataset_sha256_invalid")
    protocol_hashes = dict(payload.get("protocol_hashes") or {})
    for protocol_name in (
        "theme_v2",
        "factor_v2",
        "dashboard_contract_v2",
    ):
        if not _is_sha256(protocol_hashes.get(protocol_name)):
            blockers.append(f"joint_manifest_protocol_hash_invalid:{protocol_name}")
    if (
        not _is_sha256(expected_theme_hash)
        or str(protocol_hashes.get("theme_v2") or "").lower()
        != expected_theme_hash
    ):
        blockers.append("joint_manifest_protocol_hash_mismatch")

    split_payload = dict(payload.get("split") or {})
    split_dates: list[str] = []
    split_valid = True
    for split_name in ("train", "validation", "holdout"):
        raw_dates = split_payload.get(split_name)
        if not isinstance(raw_dates, list) or not raw_dates:
            blockers.append(f"joint_manifest_split_{split_name}_invalid")
            split_valid = False
            continue
        normalized = [_date_key(value) for value in raw_dates]
        if any(not value for value in normalized) or normalized != raw_dates:
            blockers.append(f"joint_manifest_split_{split_name}_invalid")
            split_valid = False
        split_dates.extend(normalized)
    if split_valid:
        if len(split_dates) != len(set(split_dates)) or split_dates != sorted(
            split_dates
        ):
            blockers.append("joint_manifest_split_order_invalid")
            split_valid = False
        expected_split = build_replay_split(split_dates).to_dict()
        if split_payload != expected_split:
            blockers.append("joint_manifest_split_contract_mismatch")
        if str(payload.get("split_sha256") or "") != _sha256(split_payload):
            blockers.append("joint_manifest_split_sha256_mismatch")
        if str(payload.get("trade_dates_sha256") or "") != _sha256(
            split_dates
        ):
            blockers.append("joint_manifest_trade_dates_sha256_mismatch")
    if payload.get("holdout_opened") is not True:
        blockers.append("joint_manifest_holdout_not_opened")

    threshold_seal = dict(payload.get("threshold_seal") or {})
    thresholds = dict(threshold_seal.get("thresholds") or {})
    threshold_hash = str(threshold_seal.get("threshold_hash") or "")
    threshold_blockers = validate_threshold_seal(
        threshold_seal,
        current_thresholds=thresholds,
        dataset_sha256=dataset_sha,
        holdout_opened=payload.get("holdout_opened") is True,
        expected_threshold_hash=threshold_hash,
    )
    blockers.extend(f"joint_manifest:{item}" for item in threshold_blockers)
    if threshold_seal.get("_artifact_readback_verified") is not True:
        blockers.append("joint_manifest_threshold_seal_readback_unverified")
    if not _is_sha256(threshold_seal.get("_artifact_sha256")):
        blockers.append("joint_manifest_threshold_seal_artifact_sha256_missing")
    if threshold_seal.get("_canonical_seal_ledger_verified") is not True:
        blockers.append("joint_manifest_threshold_seal_ledger_unverified")
    if not _is_sha256(threshold_seal.get("_seal_ledger_sha256")):
        blockers.append("joint_manifest_threshold_seal_ledger_sha256_missing")
    if split_valid and _date_key(
        threshold_seal.get("validation_end_date")
    ) != split_payload["validation"][-1]:
        blockers.append("joint_manifest_threshold_validation_end_mismatch")

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(
        REQUIRED_REPLAY_SCENARIOS
    ):
        blockers.append("joint_manifest_scenario_set_invalid")
        scenarios = {}
    metric_thresholds_by_scenario = dict(
        thresholds.get("scenario_metrics") or {}
    )
    for name in REQUIRED_REPLAY_SCENARIOS:
        audit = dict(scenarios.get(name) or {})
        declared_blockers = list(audit.get("blockers") or [])
        if audit.get("passed") is not True or declared_blockers:
            blockers.append(f"joint_manifest_scenario_not_passed:{name}")
        supplied_checks = dict(audit.get("computed_metric_checks") or {})
        source_evidence = {
            key: value
            for key, value in audit.items()
            if key not in {"passed", "blockers", "computed_metric_checks"}
        }
        metric_thresholds = {
            **dict(metric_thresholds_by_scenario.get("*") or {}),
            **dict(metric_thresholds_by_scenario.get(name) or {}),
        }
        recomputed, item_blockers = _validate_scenario_evidence(
            name=name,
            result=source_evidence,
            dataset_sha256=dataset_sha,
            split_sha256=str(payload.get("split_sha256") or ""),
            trade_dates_sha256=str(payload.get("trade_dates_sha256") or ""),
            protocol_hashes=protocol_hashes,
            metric_thresholds=metric_thresholds,
        )
        if item_blockers:
            blockers.extend(
                f"joint_manifest_scenario_invalid:{name}:{item}"
                for item in item_blockers
            )
        if supplied_checks != dict(recomputed.get("computed_metric_checks") or {}):
            blockers.append(
                f"joint_manifest_scenario_metric_checks_mismatch:{name}"
            )

    shadow = dict(payload.get("theme_live_shadow") or {})
    observations = shadow.get("observations")
    if not isinstance(observations, list):
        blockers.append("joint_manifest_shadow_observations_invalid")
        observations = []
    source_observations = [
        {
            key: value
            for key, value in dict(observation or {}).items()
            if key not in {"verified", "blockers"}
        }
        for observation in observations
        if isinstance(observation, Mapping)
    ]
    shadow_dates, _shadow_audit, shadow_blockers = _validate_shadow_evidence(
        source_observations,
        dataset_sha256=dataset_sha,
        theme_protocol_hash=str(protocol_hashes.get("theme_v2") or ""),
        trade_dates=set(split_dates),
    )
    blockers.extend(f"joint_manifest:{item}" for item in shadow_blockers)
    declared_shadow_dates = list(shadow.get("dates") or [])
    if declared_shadow_dates != shadow_dates:
        blockers.append("joint_manifest_shadow_dates_mismatch")
    if (
        int(shadow.get("distinct_trade_day_count") or -1)
        != len(shadow_dates)
        or len(shadow_dates) < MIN_THEME_LIVE_SHADOW_DAYS
        or int(shadow.get("required_distinct_trade_day_count") or -1)
        != MIN_THEME_LIVE_SHADOW_DAYS
        or shadow.get("passed") is not True
    ):
        blockers.append("joint_manifest_shadow_coverage_invalid")

    acceptance = dict(payload.get("acceptance_evidence") or {})
    acceptance_payload, acceptance_blockers = _validate_acceptance_evidence(
        acceptance,
        dataset_sha256=dataset_sha,
    )
    blockers.extend(
        f"joint_manifest:{item}" for item in acceptance_blockers
    )
    acceptance_for_activation = dict(acceptance_payload)
    acceptance_for_activation["dag"] = {"offline_replay_pass": True}
    expected_activation = build_activation_decision(
        acceptance_for_activation,
        distinct_theme_shadow_days=len(shadow_dates),
    )
    if dict(payload.get("activation") or {}) != expected_activation:
        blockers.append("joint_manifest_activation_recompute_mismatch")
    if any(
        decision.get("enabled") is not True
        or list(decision.get("blockers") or [])
        for decision in expected_activation.values()
    ):
        blockers.append("joint_manifest_activation_not_fully_ready")

    controls = dict(payload.get("controls") or {})
    for key, expected in (
        ("offline_only", True),
        ("no_network", True),
        ("no_llm", True),
        ("no_broker", True),
        ("registry_mutation", False),
        ("merge_requires_maxwell_confirmation", True),
    ):
        if controls.get(key) is not expected:
            blockers.append(f"joint_manifest_control_invalid:{key}")
    blockers = list(dict.fromkeys(blockers))
    return _manifest_verification_result(
        blockers=blockers,
        manifest=dict(manifest),
        artifact_sha256=artifact_sha,
        readback_verified=raw == readback,
    )


def _manifest_verification_result(
    *,
    blockers: Iterable[str],
    manifest: Mapping[str, Any] | None = None,
    artifact_sha256: str = "",
    readback_verified: bool = False,
) -> dict[str, Any]:
    normalized = list(
        dict.fromkeys(str(blocker) for blocker in blockers if str(blocker))
    )
    return {
        "status": "ready" if not normalized else "blocked",
        "ready": not normalized,
        "blockers": normalized,
        "manifest": dict(manifest or {}),
        "artifact_sha256": str(artifact_sha256 or ""),
        "readback_verified": bool(readback_verified),
    }
