"""Exact v2 Decision-report recipe over unchanged research inputs and frozen book."""

import hashlib

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.intelligence.portfolio_state import REPORT_POLICY, STRATEGY_ID
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal
from .portfolio_binding import freeze_portfolio_state, PortfolioSource
from .research_request import load_research_request

SCHEMA = "cn-daily-decision-recipe.v2"
FIELDS = frozenset(
    {
        "schema_version",
        "trade_date",
        "as_of",
        "strategy_id",
        "research_request_ref",
        "store_plan_ref",
        "portfolio_source_ref",
        "report_policy",
    }
)


def _read(workspace, ref):
    ref = validate_ref(ref)
    raw = SecureSystemStorage(workspace).read_workspace_file_bytes(
        ref["path"], maximum_bytes=16 * 1024 * 1024
    )
    if raw.byte_sha256 != ref["sha256"]:
        raise ContractError("DECISION_RECIPE_SOURCE_SHA_MISMATCH")
    return parse_canonical_json_bytes(raw.data)


def read_decision_recipe(
    *, workspace, trade_date, recipe_ref, research_request_ref, store_plan_ref
):
    recipe = _read(workspace, recipe_ref)
    retained_request = _read(workspace, research_request_ref)
    loaded = load_research_request(workspace=workspace, reference=research_request_ref)
    original = loaded["document"]
    journal = DailyJournal(str(workspace), trade_date)
    expected_path = str(journal.root / "inputs" / f"decision-recipe-{recipe_ref['sha256']}.json")
    if (
        type(recipe) is not dict
        or set(recipe) != FIELDS
        or recipe["schema_version"] != SCHEMA
        or recipe_ref["path"] != expected_path
        or recipe["trade_date"] != trade_date
        or recipe["as_of"] != original["as_of"]
        or recipe["strategy_id"] != STRATEGY_ID
        or original["strategy_id"] != STRATEGY_ID
        or recipe["research_request_ref"] != validate_ref(research_request_ref)
        or recipe["store_plan_ref"] != validate_ref(store_plan_ref)
        or recipe["report_policy"] != REPORT_POLICY
        or utc_stamp(recipe["as_of"]).strftime("%Y%m%d") != trade_date
    ):
        raise ContractError("DECISION_RECIPE_BINDING_INVALID")
    source = PortfolioSource(
        workspace=workspace,
        trade_date=trade_date,
        as_of=recipe["as_of"],
        store_plan_ref=store_plan_ref,
    )
    portfolio = source.replay(validate_ref(recipe["portfolio_source_ref"]))
    if loaded["cutoff"] is not None and (
        loaded["cutoff"]["store_plan_ref"] != store_plan_ref
        or loaded["cutoff"]["portfolio_state_ref"] != recipe["portfolio_source_ref"]
        or loaded["cutoff"]["receipt"]["as_of"] != recipe["as_of"]
    ):
        raise ContractError("DECISION_RECIPE_CUTOFF_MISMATCH")
    if (
        _read(workspace, recipe_ref) != recipe
        or _read(workspace, research_request_ref) != retained_request
    ):
        raise ContractError("DECISION_RECIPE_SOURCE_CHANGED")
    return {
        "recipe": recipe,
        "portfolio": portfolio,
        **({"cutoff": loaded["cutoff"]} if loaded["cutoff"] is not None else {}),
    }


def publish_decision_recipe(
    *, journal, research_request_ref, store_plan_ref, retained_pointer_ref=None
):
    journal._require_lock()
    workspace = journal.storage._io.workspace_root
    original = load_research_request(workspace=workspace, reference=research_request_ref)[
        "document"
    ]
    if original["strategy_id"] != STRATEGY_ID:
        raise ContractError("DECISION_RECIPE_STRATEGY_INVALID")
    frozen = freeze_portfolio_state(
        journal=journal,
        store_plan_ref=store_plan_ref,
        as_of=original["as_of"],
        retained_pointer_ref=retained_pointer_ref,
    )
    recipe = {
        "schema_version": SCHEMA,
        "trade_date": journal.trade_date,
        "as_of": original["as_of"],
        "strategy_id": STRATEGY_ID,
        "research_request_ref": validate_ref(research_request_ref),
        "store_plan_ref": validate_ref(store_plan_ref),
        "portfolio_source_ref": frozen["state_ref"],
        "report_policy": REPORT_POLICY,
    }
    raw = canonical_json_bytes(recipe)
    sha = hashlib.sha256(raw).hexdigest()
    path = str(journal.root / "inputs" / f"decision-recipe-{sha}.json")
    stored = journal.storage.write(path, raw)
    ref = {"path": path, "sha256": stored.byte_sha256}
    read_decision_recipe(
        workspace=workspace,
        trade_date=journal.trade_date,
        recipe_ref=ref,
        research_request_ref=research_request_ref,
        store_plan_ref=store_plan_ref,
    )
    return ref


def native_portfolio_is_late(*, workspace, inputs):
    from .native_input_contract import validate_native_input_shape

    validate_native_input_shape(inputs)
    if inputs["schema_version"] not in {
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        return False
    bound = read_decision_recipe(
        workspace=workspace,
        trade_date=inputs["trade_date"],
        recipe_ref=inputs["decision_recipe_ref"],
        research_request_ref=inputs["research_request_ref"],
        store_plan_ref=inputs["store_plan_ref"],
    )
    return bound["portfolio"]["payload"]["timing_status"] == "LATE_RECORDED"
