"""First daily DAG proof over existing native baselines; never initializes them."""

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.factors.production_authority import (
    FactorProductionStore,
    FACTOR_POINTER_HISTORY_ROOT,
    FACTOR_EMPTY_POINTER_SHA256,
    FACTOR_PRODUCTION_MARKER_PATH,
    FACTOR_AUTHORITY_ACTIVE,
    FactorStoredBytes,
    FACTOR_ACTIVE_POINTER_PATH,
)
from quant_investor.market.close_session_authority import replay_close_session_authority
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref
from .daily_journal import _false_authority, _validate_day

FIELDS = frozenset(
    {
        "schema_version",
        "market",
        "strategy_id",
        "first_trade_date",
        "previous_trade_date",
        "graph_sha256",
        "factor_parent_pointer_sha256",
        "store_preimages",
        "authority",
    }
)


def validate_bootstrap_declaration(value: dict, *, recipe: dict) -> dict:
    if (
        type(value) is not dict
        or set(value) != FIELDS
        or value["schema_version"] != "cn-daily-bootstrap.v1"
        or value["market"] != "CN"
        or value["strategy_id"] != "aggressive_tech_manufacturing"
        or value["graph_sha256"] != GRAPH_SHA256
        or value["first_trade_date"] != recipe["target_trade_date"]
        or not _false_authority(value["authority"])
    ):
        raise ContractError("BOOTSTRAP_DECLARATION_INVALID")
    for field in ("first_trade_date", "previous_trade_date"):
        _validate_day(value[field])
    if value["previous_trade_date"] >= value["first_trade_date"]:
        raise ContractError("BOOTSTRAP_DATE_ORDER_INVALID")
    sha = value["factor_parent_pointer_sha256"]
    if sha == FACTOR_EMPTY_POINTER_SHA256:
        raise ContractError("BOOTSTRAP_NATIVE_BASELINE_REQUIRED")
    validate_ref({"path": "parent.json", "sha256": sha})
    refs = value["store_preimages"]
    if type(refs) is not dict or set(refs) != {
        "store_pointer_ref",
        "event_pointer_ref",
        "benchmark_pointer_ref",
    }:
        raise ContractError("BOOTSTRAP_STORE_PREIMAGES_INVALID")
    for ref in refs.values():
        validate_ref(ref)
    if canonical_json_bytes(refs) != canonical_json_bytes(recipe["store_preimages"]):
        raise ContractError("BOOTSTRAP_STORE_PREIMAGES_MISMATCH")
    return value


def read_bootstrap_declaration(*, workspace: str, recipe: dict) -> dict:
    if recipe["previous_completion_ref"] is not None or recipe["bootstrap_ref"] is None:
        raise ContractError("BOOTSTRAP_ANCHOR_INVALID")
    ref = validate_ref(recipe["bootstrap_ref"])
    raw = SecureSystemStorage(workspace).read_workspace_file_bytes(
        ref["path"], maximum_bytes=1024 * 1024
    )
    if raw.byte_sha256 != ref["sha256"]:
        raise ContractError("BOOTSTRAP_SHA_MISMATCH")
    return validate_bootstrap_declaration(parse_canonical_json_bytes(raw.data), recipe=recipe)


def verify_initial_bootstrap_baseline(*, workspace: str, recipe: dict) -> dict:
    """Validate an existing active genesis/lineage before first maintenance starts.

    Immediate OPEN-session succession remains a post-maintenance Calendar check;
    this preflight neither manufactures Calendar evidence nor activates a baseline.
    """
    declaration = read_bootstrap_declaration(workspace=workspace, recipe=recipe)
    store = FactorProductionStore(workspace)
    pointer = store.read(FACTOR_ACTIVE_POINTER_PATH)
    marker = store.read(FACTOR_PRODUCTION_MARKER_PATH)
    if (
        pointer is None
        or marker is None
        or pointer.byte_sha256 != declaration["factor_parent_pointer_sha256"]
    ):
        raise ContractError("BOOTSTRAP_INITIAL_ACTIVE_BASELINE_MISMATCH")
    verified = store.verify_active()
    if (
        verified.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE
        or verified.get("as_of") != declaration["previous_trade_date"]
        or verified.get("factor_pointer_byte_sha256") != pointer.byte_sha256
    ):
        raise ContractError("BOOTSTRAP_INITIAL_BASELINE_INVALID")
    if (
        store.read(FACTOR_ACTIVE_POINTER_PATH) != pointer
        or store.read(FACTOR_PRODUCTION_MARKER_PATH) != marker
        or read_bootstrap_declaration(workspace=workspace, recipe=recipe) != declaration
    ):
        raise ContractError("BOOTSTRAP_INITIAL_BASELINE_CHANGED")
    return {
        "previous_trade_date": declaration["previous_trade_date"],
        "factor_pointer_sha256": pointer.byte_sha256,
        "execution_authorized": False,
    }


def verify_bootstrap_native(*, workspace: str, recovered: dict) -> dict:
    """Native Calendar/Factor evidence replay, with no locks or mutable selectors."""
    recipe, handoff = recovered["recipe"], recovered["handoff"]
    value = read_bootstrap_declaration(workspace=workspace, recipe=recipe)
    reader = SecureSystemStorage(workspace)
    observed = {}

    def raw(ref):
        validate_ref(ref)
        stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if stored.byte_sha256 != ref["sha256"]:
            raise ContractError("BOOTSTRAP_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = stored.data
        return stored.data

    receipt = parse_canonical_json_bytes(raw(handoff["calendar_ref"]))
    calendar = replay_close_session_authority(receipt, raw(handoff["raw_calendar_ref"])).receipt
    dates = calendar["ordered_open_dates"]
    previous, first = value["previous_trade_date"], value["first_trade_date"]
    if (
        calendar["target_trade_date"] != first
        or first not in dates
        or dates.index(first) == 0
        or dates[dates.index(first) - 1] != previous
    ):
        raise ContractError("BOOTSTRAP_CALENDAR_PREDECESSOR_INVALID")
    target_raw = raw(handoff["factor_pointer_ref"])
    target = parse_canonical_json_bytes(target_raw)
    parent_sha = value["factor_parent_pointer_sha256"]
    if type(target) is not dict or target.get("previous_pointer_sha256") != parent_sha:
        raise ContractError("BOOTSTRAP_FACTOR_PARENT_MISMATCH")
    store = FactorProductionStore(workspace)
    store.inspect_recorded_research_inputs(
        pointer_raw=target_raw,
        expected_pointer_sha256=handoff["factor_pointer_ref"]["sha256"],
        expected_trade_date=first,
    )
    parent_ref = {
        "path": str(FACTOR_POINTER_HISTORY_ROOT / f"{parent_sha}.json"),
        "sha256": parent_sha,
    }
    parent_raw = raw(parent_ref)
    # A genesis baseline can be lineage-valid without being an observation input.
    # Keep target research/PIT validation above; the parent is used only as the
    # registered predecessor, through the original full native lineage verifier.
    marker = store.read(FACTOR_PRODUCTION_MARKER_PATH)
    baseline = store._verify_pointer_lineage(
        FactorStoredBytes(parent_ref["path"], parent_raw, parent_sha), marker
    )
    if (
        baseline.get("factor_authority") != FACTOR_AUTHORITY_ACTIVE
        or baseline.get("as_of") != previous
    ):
        raise ContractError("BOOTSTRAP_FACTOR_BASELINE_INVALID")
    if store.read(FACTOR_PRODUCTION_MARKER_PATH) != marker:
        raise ContractError("BOOTSTRAP_FACTOR_MARKER_CHANGED")
    for path, data in observed.items():
        if reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != data:
            raise ContractError("BOOTSTRAP_SOURCE_CHANGED")
    if read_bootstrap_declaration(workspace=workspace, recipe=recipe) != value:
        raise ContractError("BOOTSTRAP_DECLARATION_CHANGED")
    return {
        "declaration": value,
        "previous_trade_date": previous,
        "ordered_open_dates": dates,
        "factor_parent_pointer_ref": parent_ref,
        "execution_authorized": False,
    }


def verify_bootstrap_store_plan(*, proof: dict, plan: dict, calendar_ref: dict) -> None:
    """After native plan production/readback, check bootstrap-specific bindings."""
    declaration = proof["declaration"]
    first = declaration["first_trade_date"]
    iso = lambda day: f"{day[:4]}-{day[4:6]}-{day[6:]}"
    if plan["requested_target"] != iso(first):
        raise ContractError("BOOTSTRAP_STORE_TARGET_MISMATCH")
    for name, key in [
        ("store_pointer_ref", "store_pointer_sha256"),
        ("event_pointer_ref", "event_pointer_sha256"),
        ("benchmark_pointer_ref", "benchmark_pointer_sha256"),
    ]:
        if plan["preimages"][key] != declaration["store_preimages"][name]["sha256"]:
            raise ContractError("BOOTSTRAP_STORE_PLAN_PREIMAGE_MISMATCH")
    if plan["preimages"]["calendar_receipt_sha256"] != calendar_ref["sha256"]:
        raise ContractError("BOOTSTRAP_STORE_CALENDAR_MISMATCH")
    dates = [iso(day) for day in proof["ordered_open_dates"]]
    frontier = plan["last_official_date"]
    if (
        frontier not in dates
        or frontier >= iso(first)
        or plan["missing_dates"] != [day for day in dates if frontier < day <= iso(first)]
    ):
        raise ContractError("BOOTSTRAP_STORE_PREFIX_INVALID")


def verify_bootstrap_initial_absence(*, journal, proof: dict) -> None:
    journal._require_lock()
    previous = proof["previous_trade_date"]
    for day in (previous, journal.trade_date):
        path = str(journal.root.parent / day / "completion.v1.json")
        if journal.storage.read(path) is not None:
            raise ContractError("BOOTSTRAP_EXISTING_EOD")
