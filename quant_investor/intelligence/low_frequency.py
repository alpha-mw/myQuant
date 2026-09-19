"""Descriptive low-frequency freshness; native coverage retains control authority."""

from decimal import Decimal

from quant_investor.macro.registry import PERIOD_MAX_LAG_DAYS
from ._common import build_artifact, business_identity, company_code, IntelligenceError, timestamp
from .fundamental_time import availability_instant, session_date, SHANGHAI
from .pcb_ai_hardware import physical_refs

FRESHNESS_KIND = "low_frequency_source_freshness"
FRESHNESS_CONTRACT = "low-frequency-source-freshness.v1"
ENTRY_FIELDS = frozenset(
    {
        "subject_id",
        "snapshot_date",
        "known_at",
        "age_days",
        "freshness_state",
        "warning_codes",
        "critical_missing_codes",
        "frequency",
        "availability_age_days",
        "period_lag_days",
        "availability_limit_days",
        "period_lag_limit_days",
    }
)


def freshness_profile(recipe):
    if "freshness_contract" not in recipe:
        return None
    if recipe["freshness_contract"] != FRESHNESS_CONTRACT:
        raise IntelligenceError("LOW_FREQUENCY_CONTRACT_INVALID")
    return FRESHNESS_CONTRACT


def empty_entry(subject, *, critical=(), warnings=()):
    return {
        "subject_id": subject,
        "snapshot_date": None,
        "known_at": None,
        "age_days": None,
        "freshness_state": "MISSING",
        "warning_codes": sorted(set(warnings)),
        "critical_missing_codes": sorted(set(critical)),
        "frequency": None,
        "availability_age_days": None,
        "period_lag_days": None,
        "availability_limit_days": None,
        "period_lag_limit_days": None,
    }


def freshness_artifact(*, domain, as_of, refs, policy, entries, warnings=(), critical=()):
    stamp = timestamp(as_of, label="freshness cutoff")
    if domain not in {"FUNDAMENTAL", "MACRO"}:
        raise IntelligenceError("LOW_FREQUENCY_DOMAIN_INVALID")
    entries = sorted(entries, key=lambda row: row["subject_id"])
    if len({row["subject_id"] for row in entries}) != len(entries) or any(
        set(row) != ENTRY_FIELDS for row in entries
    ):
        raise IntelligenceError("LOW_FREQUENCY_ENTRIES_INVALID")
    warning_codes, missing_codes = set(warnings), set(critical)
    for row in entries:
        warning_codes.update(code + ":" + row["subject_id"] for code in row["warning_codes"])
        missing_codes.update(
            code + ":" + row["subject_id"] for code in row["critical_missing_codes"]
        )
    states = {row["freshness_state"] for row in entries}
    state = (
        "MISSING"
        if missing_codes
        else next(
            (value for value in ("STALE_WARNING", "ACCEPTABLE_LAG", "FRESH") if value in states),
            "MISSING",
        )
    )
    fields = {
        "domain": domain,
        "as_of": stamp,
        "trade_date": availability_instant(stamp).astimezone(SHANGHAI).strftime("%Y%m%d"),
        "source_refs": physical_refs(refs),
        "policy": policy,
        "entries": entries,
        "freshness_state": state,
        "warning_codes": sorted(warning_codes),
        "critical_missing_codes": sorted(missing_codes),
    }
    return build_artifact(
        kind=FRESHNESS_KIND,
        identity_field="freshness_id",
        identity=business_identity(kind=FRESHNESS_KIND, identity_inputs=fields),
        created_at=stamp,
        fields=fields,
    )


def _fundamental_entry(company, *, source, assessment, day, known_at):
    if source is None:
        return empty_entry(company, critical=["FUNDAMENTAL_ELIGIBLE_SNAPSHOT_MISSING"])
    metrics = source["payload"]["metrics"]
    snapshot = session_date(int(Decimal(metrics["snapshot_trade_date"])))
    age = (day - snapshot).days
    if age < 0:
        raise IntelligenceError("FUNDAMENTAL_FRESHNESS_FUTURE_SNAPSHOT")
    critical = [] if assessment is not None else ["FUNDAMENTAL_NATIVE_ASSESSMENT_MISSING"]
    if assessment is not None:
        critical.extend(assessment["payload"]["blocker_codes"])
    entry = empty_entry(company, critical=critical)
    entry.update(snapshot_date=snapshot.isoformat(), known_at=known_at, age_days=age)
    entry["warning_codes"] = sorted(
        "FUNDAMENTAL_METRIC_MISSING:" + key for key, value in metrics.items() if value is None
    )
    if age > PERIOD_MAX_LAG_DAYS["quarterly"]:
        entry["warning_codes"].append("FUNDAMENTAL_SNAPSHOT_LAG_WARNING")
    if not critical:
        entry["freshness_state"] = (
            "STALE_WARNING"
            if age > PERIOD_MAX_LAG_DAYS["quarterly"]
            else "FRESH" if age == 0 else "ACCEPTABLE_LAG"
        )
    return entry


def fundamental_freshness(*, companies, as_of, descriptor, sources, assessments, known_at=None):
    cutoff = availability_instant(as_of)
    if known_at is not None and availability_instant(known_at) > cutoff:
        raise IntelligenceError("FUNDAMENTAL_FRESHNESS_FUTURE_AVAILABILITY")
    by_company = {source["payload"]["company_code"]: source for source in sources}
    refs = [] if descriptor is None else [descriptor["pointer"], descriptor["daily_parquet"]]
    return freshness_artifact(
        domain="FUNDAMENTAL",
        as_of=as_of,
        refs=refs,
        policy={
            "threshold_source": "macro.registry.PERIOD_MAX_LAG_DAYS.quarterly",
            "warning_after_days": PERIOD_MAX_LAG_DAYS["quarterly"],
            "blocking_after_days": None,
            "effect": "WARNING_ONLY",
            "fundamental_policy": "ADVISORY_NO_FIXED_MAXIMUM",
        },
        entries=[
            _fundamental_entry(
                company_code(company),
                source=by_company.get(company_code(company)),
                assessment=assessments.get(company_code(company)),
                day=cutoff.astimezone(SHANGHAI).date(),
                known_at=known_at,
            )
            for company in companies
        ],
    )
