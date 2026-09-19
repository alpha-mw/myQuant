"""Code-owned Theme acquisition policy; no provider calls are authorized by parsing it."""

from datetime import datetime, timezone
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref, utc_stamp
from .daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority, _validate_day

POLICY_SCHEMA = "cn-daily-theme-acquisition.v1"
POLICY_SCHEMA_V2 = "cn-daily-theme-acquisition.v2"
POLICY_SCHEMA_V3 = "cn-daily-theme-acquisition.v3"
CLAIM_SCHEMA = "cn-daily-theme-acquisition-claim.v1"
IDENTITY_REFS = frozenset(
    {
        "request_ref",
        "core_handoff_ref",
        "pool_manifest_ref",
        "rank_ref",
        "acquisition_policy_ref",
        "release_ref",
    }
)
IDENTITY_FIELDS = IDENTITY_REFS | {"company_set_sha256"}


def _identity(value: dict) -> dict:
    if type(value) is not dict or set(value) != IDENTITY_FIELDS:
        raise ContractError("THEME_ACQUISITION_IDENTITY_INVALID")
    result = {key: validate_ref(value[key]) for key in IDENTITY_REFS}
    validate_ref({"path": "company-set.json", "sha256": value["company_set_sha256"]})
    return {**result, "company_set_sha256": value["company_set_sha256"]}


def validate_theme_acquisition_claim(value: dict, *, trade_date: str, identity: dict) -> dict:
    _validate_day(trade_date)
    expected = _identity(identity)
    if (
        type(value) is not dict
        or set(value)
        != IDENTITY_FIELDS
        | {"schema_version", "trade_date", "graph_sha256", "claimed_at", "authority"}
        or value["schema_version"] != CLAIM_SCHEMA
        or value["trade_date"] != trade_date
        or value["graph_sha256"] != GRAPH_SHA256
        or not _false_authority(value["authority"])
    ):
        raise ContractError("THEME_ACQUISITION_CLAIM_INVALID")
    if _identity({key: value[key] for key in IDENTITY_FIELDS}) != expected:
        raise ContractError("THEME_ACQUISITION_ALREADY_CLAIMED")
    if utc_stamp(value["claimed_at"]) > datetime.now(timezone.utc):
        raise ContractError("THEME_ACQUISITION_CLAIM_FUTURE")
    return value


def reserve_theme_acquisition(*, journal: DailyJournal, identity: dict) -> dict:
    """Single day-wide identity, after caller's native pool validation and before providers."""
    journal._require_lock()
    identity = _identity(identity)
    path = str(journal.root / "theme-acquisition.v1.json")
    existing = journal.storage.read(path)
    if existing is not None:
        validate_theme_acquisition_claim(
            parse_canonical_json_bytes(existing.data),
            trade_date=journal.trade_date,
            identity=identity,
        )
        if journal.storage.read(path) != existing:
            raise ContractError("THEME_ACQUISITION_CLAIM_CHANGED")
        return {"path": path, "sha256": existing.byte_sha256}
    value = {
        "schema_version": CLAIM_SCHEMA,
        "trade_date": journal.trade_date,
        "graph_sha256": GRAPH_SHA256,
        **identity,
        "claimed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": FALSE_AUTHORITY,
    }
    validate_theme_acquisition_claim(value, trade_date=journal.trade_date, identity=identity)
    stored = journal.storage.write(path, canonical_json_bytes(value))
    if journal.storage.read(path) != stored:
        raise ContractError("THEME_ACQUISITION_CLAIM_CHANGED")
    return {"path": path, "sha256": stored.byte_sha256}


def validate_theme_acquisition_policy(value: dict) -> dict:
    from quant_investor.factors.production_pit import FOCUS_COMPANIES

    v2 = type(value) is dict and value.get("schema_version") in {POLICY_SCHEMA_V2, POLICY_SCHEMA_V3}
    fields = {"schema_version", "provider_priority", "fallback_mode", "maximum_companies"}
    if v2:
        fields.add("special_company_keyset")
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] not in {POLICY_SCHEMA, POLICY_SCHEMA_V2, POLICY_SCHEMA_V3}
        or value["provider_priority"] != ["TUSHARE_DC", "TUSHARE_TDX"]
        or value["fallback_mode"] != "NATIVE_DC_PARTITION_FALLBACK"
        or type(value["maximum_companies"]) is not int
        or value["maximum_companies"] != 100
        or (v2 and value["special_company_keyset"] != list(FOCUS_COMPANIES))
    ):
        raise ContractError("THEME_ACQUISITION_POLICY_INVALID")
    return {
        **value,
        "provider_priority": list(value["provider_priority"]),
        **({"special_company_keyset": list(FOCUS_COMPANIES)} if v2 else {}),
    }


def validate_theme_policy_profile(recipe, policy):
    from .execution_recipe import SCHEMA_V5, SCHEMA_V6

    if (recipe["schema_version"] in {SCHEMA_V5, SCHEMA_V6}) != (
        policy["schema_version"] == POLICY_SCHEMA_V3
    ):
        raise ContractError("THEME_ACQUISITION_RECIPE_PROFILE_MISMATCH")
