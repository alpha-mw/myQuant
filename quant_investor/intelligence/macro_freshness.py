"""Advisory Macro freshness from the native vintage model and frozen closure."""

from datetime import date
from decimal import Decimal
from pathlib import Path, PurePosixPath

from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.macro.registry import (
    NATIONAL_INDICATORS,
    INDUSTRY_CHAINS,
    INDUSTRY_COMPONENT_WEIGHTS,
    FRESHNESS_MAX_AGE_DAYS,
    PERIOD_MAX_LAG_DAYS,
    REGISTRY_VERSION,
    definition_for,
)
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import load_frozen_observations
from quant_investor.macro.readiness_closure import validate_macro_readiness_closure
from ._common import IntelligenceError, decimal_text
from .fundamental_time import availability_instant, SHANGHAI
from .low_frequency import empty_entry, freshness_artifact


def macro_subjects():
    return sorted(
        {item.indicator_id for item in NATIONAL_INDICATORS}
        | {
            f"industry.{chain}.{component}"
            for chain in INDUSTRY_CHAINS
            for component in INDUSTRY_COMPONENT_WEIGHTS
        }
    )


def _macro_entry(subject, *, selected, frequency, cutoff, native_freshness):
    if selected is None:
        return empty_entry(subject, warnings=["MACRO_INDICATOR_NOT_AVAILABLE"])
    definition = definition_for(subject, frequency)
    if definition is None:
        raise IntelligenceError("MACRO_FRESHNESS_UNREGISTERED_SUBJECT")
    known = availability_instant(selected["available_at"])
    elapsed = cutoff - known
    age = Decimal(elapsed.days) + Decimal(
        elapsed.seconds * 1000000 + elapsed.microseconds
    ) / Decimal(86400000000)
    lag = (cutoff.date() - date.fromisoformat(selected["period_end"])).days
    availability_limit = FRESHNESS_MAX_AGE_DAYS[definition.frequency]
    period_limit = PERIOD_MAX_LAG_DAYS[definition.frequency]
    warnings = []
    stale = age > availability_limit or lag > period_limit or lag < 0
    if age > availability_limit:
        warnings.append("MACRO_STALE_AVAILABILITY")
    if lag > period_limit or lag < 0:
        warnings.append("MACRO_STALE_PERIOD")
    for name, subjects in native_freshness.items():
        if subject in subjects:
            warnings.append("MACRO_NATIVE_" + name.upper())
    row = empty_entry(subject, warnings=warnings)
    row.update(
        snapshot_date=selected["period_end"],
        known_at=selected["available_at"],
        age_days=lag,
        frequency=definition.frequency,
        availability_age_days=decimal_text(age),
        period_lag_days=lag,
        availability_limit_days=availability_limit,
        period_lag_limit_days=period_limit,
        freshness_state=("STALE_WARNING" if stale else "FRESH" if lag == 0 else "ACCEPTABLE_LAG"),
    )
    return row


def macro_freshness(*, observations, as_of, refs, missing_code=None):
    cutoff = availability_instant(as_of)
    snapshot = build_macro_snapshot(observations, as_of=as_of, decision_cutoff_at=as_of)
    body = snapshot.to_dict()
    lineage = body["source_lineage"]
    frequencies = {row["content_hash"]: row["frequency"] for row in observations}
    subjects = macro_subjects()
    eligible = set(lineage) & set(subjects)
    critical = [] if eligible else [missing_code or "MACRO_PIT_OBSERVATIONS_MISSING"]
    return freshness_artifact(
        domain="MACRO",
        as_of=as_of,
        refs=refs,
        policy={
            "registry_version": REGISTRY_VERSION,
            "threshold_source": "macro.registry.FRESHNESS_MAX_AGE_DAYS+PERIOD_MAX_LAG_DAYS",
            "availability_limits_days": dict(FRESHNESS_MAX_AGE_DAYS),
            "period_lag_limits_days": dict(PERIOD_MAX_LAG_DAYS),
            "effect": "WARNING_ONLY",
            "native_readiness_unchanged": True,
        },
        entries=[
            _macro_entry(
                subject,
                selected=lineage.get(subject),
                cutoff=cutoff,
                frequency=(
                    frequencies[lineage[subject]["content_hash"]] if subject in lineage else None
                ),
                native_freshness=body["freshness"],
            )
            for subject in subjects
        ],
        warnings=["MACRO_NATIVE:" + code for code in body["blockers"]],
        critical=critical,
    )


def macro_freshness_from_closure(*, workspace, as_of, closure_ref, source_file, closure=None):
    """Caller uses native admission; this reader also supports sealed historical replay."""
    if closure is None:
        _, raw, _ = source_file(closure_ref, code="MACRO_FRESHNESS_CLOSURE_INVALID")
        closure = validate_macro_readiness_closure(
            workspace_root=workspace,
            closure=parse_json_bytes(raw, label="Macro freshness closure", require_canonical=False),
        )
    cutoff = availability_instant(as_of)
    if (
        closure["target_date"] != cutoff.astimezone(SHANGHAI).strftime("%Y%m%d")
        or availability_instant(closure["available_at"]) > cutoff
    ):
        raise IntelligenceError("MACRO_FRESHNESS_CLOSURE_NOT_AVAILABLE")
    frozen = closure["frozen_pointers"]["observations"]
    _, raw, ref = source_file(frozen["frozen_ref"], code="MACRO_FRESHNESS_POINTER_INVALID")
    prefix = PurePosixPath(frozen["current_path"]).parent
    observations, pointer = load_frozen_observations(
        Path(workspace) / prefix,
        pointer_raw=raw,
        expected_pointer_sha256=ref["sha256"],
        expected_generation_id=frozen["generation_id"],
    )
    generation = prefix / PurePosixPath(pointer["manifest_path"]).parent
    refs = [
        closure_ref,
        ref,
        {"path": str(prefix / pointer["manifest_path"]), "sha256": pointer["manifest_sha256"]},
        {"path": str(prefix / pointer["table_path"]), "sha256": pointer["parquet_sha256"]},
        *[
            {"path": str(generation / row["path"]), "sha256": row["sha256"]}
            for row in pointer["generation_manifest"].get("evidence_files", [])
        ],
    ]
    return macro_freshness(observations=observations, as_of=as_of, refs=refs)
