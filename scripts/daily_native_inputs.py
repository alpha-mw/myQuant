"""Strict offline decoder for one day's native coordinator inputs.

Paths are workspace-relative evidence references, never commands or writer roots.
The native prepared Store plan supplies original transaction preimages; the caller
cannot replace them with current heads when resuming.
"""

from pathlib import Path
from typing import Mapping

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import _validate_day
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_native_registry import NativeDailyInputs
from scripts.daily_production_store_adapter import RECORD_ROOT, StoreCloseAdapter
from cn_official_close_batch import _calendar_dates, _load_json
from quant_investor.strategy_records.close_plan_contracts import validate_plan
from quant_investor.strategy_records.store import content_sha256

from quant_investor.operations.native_input_contract import validate_native_input_shape

REF_FIELDS = (
    "release_ref",
    "research_request_ref",
    "store_plan_ref",
    "store_policy_ref",
    "retrospective_ref",
    "calendar_ref",
    "market_snapshot_ref",
    "benchmark_ref",
    "risk_free_ref",
)


def load_native_inputs(
    *, workspace: str, input_ref: Mapping[str, str]
) -> tuple[str, NativeDailyInputs]:
    root = Path(workspace).resolve(strict=True)
    ref = validate_ref(input_ref)
    stored = SecureSystemStorage(str(root)).read_workspace_file_bytes(
        ref["path"], maximum_bytes=1024 * 1024
    )
    if stored.byte_sha256 != ref["sha256"]:
        raise ContractError("NATIVE_INPUT_DOCUMENT_SHA_MISMATCH")
    value = parse_canonical_json_bytes(stored.data, label="native daily inputs")
    validate_native_input_shape(value)
    _verify_decision_recipe(workspace=workspace, value=value)
    from quant_investor.operations.future_calendar_binding import future_calendar_outputs

    future_calendar_outputs(
        workspace=workspace,
        trade_date=value["trade_date"],
        proof_ref=value.get("next_session_calendar_proof_ref"),
        failure_ref=value.get("next_session_calendar_failure_ref"),
    )
    _validate_day(value["trade_date"])
    if value["previous_trade_date"] is not None:
        _validate_day(value["previous_trade_date"])
    if type(value["publish_current_dashboard"]) is not bool:
        raise ContractError("NATIVE_INPUT_PUBLICATION_MODE_INVALID")
    for key in REF_FIELDS:
        if key == "retrospective_ref" and value[key] is None:
            continue
        validate_ref(value[key])
    validate_ref({"path": "factor-pointer.json", "sha256": value["factor_pointer_sha256"]})
    market_refs = value["adjustment_market_refs"]
    if type(market_refs) is not dict:
        raise ContractError("NATIVE_INPUT_ADJUSTMENT_REFS_INVALID")
    for symbol, source in market_refs.items():
        if type(symbol) is not str or not symbol:
            raise ContractError("NATIVE_INPUT_ADJUSTMENT_SYMBOL_INVALID")
        validate_ref(source)
    calendar = value["calendar_ref"]
    dates, _ = _calendar_dates(root / calendar["path"], calendar["sha256"])
    day = value["trade_date"]
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    if iso not in dates:
        raise ContractError("NATIVE_INPUT_DATE_NOT_AUTHORIZED")
    previous = value["previous_trade_date"]
    prior = [date for date in dates if date < iso]
    if previous is not None and (not prior or prior[-1].replace("-", "") != previous):
        raise ContractError("NATIVE_INPUT_PREVIOUS_CALENDAR_MISMATCH")
    plan_ref = value["store_plan_ref"]
    plan = _load_json(
        root / plan_ref["path"], expected_sha=plan_ref["sha256"], label="native input Store plan"
    )
    if (
        not _store_plan_profile(value, plan)
        or plan.get("content_sha256") != content_sha256(plan)
        or plan.get("all_or_nothing") is not True
        or plan.get("broker_order_trade_authority") is not False
        or iso not in plan.get("missing_dates", [])
    ):
        raise ContractError("NATIVE_INPUT_STORE_PLAN_INVALID")
    preimages = plan["preimages"]
    for name, expected in {
        "calendar_receipt_sha256": calendar["sha256"],
        "policy_sha256": value["store_policy_ref"]["sha256"],
        "retrospective_sha256": (
            value["retrospective_ref"]["sha256"] if value["retrospective_ref"] is not None else None
        ),
    }.items():
        if preimages.get(name) != expected:
            raise ContractError("NATIVE_INPUT_STORE_SOURCE_BINDING_MISMATCH")
    args = _store_arguments(root, value, preimages)
    # Constructor validates the fixed native transaction path/date and all preimages.
    StoreCloseAdapter(
        arguments=args, trade_date=day, plan_ref=plan_ref, release_ref=value["release_ref"]
    )
    inputs = NativeDailyInputs(
        factor_pointer_sha256=value["factor_pointer_sha256"],
        release_ref=value["release_ref"],
        research_request_ref=value["research_request_ref"],
        store_arguments=args,
        store_plan_ref=plan_ref,
        event_pointer_sha256=preimages["event_pointer_sha256"],
        market_snapshot_ref=value["market_snapshot_ref"],
        benchmark_ref=value["benchmark_ref"],
        risk_free_ref=value["risk_free_ref"],
        calendar_ref=calendar,
        previous_trade_date=previous,
        adjustment_market_refs=market_refs,
        publish_current_dashboard=value["publish_current_dashboard"],
        next_session_calendar_proof_ref=value.get("next_session_calendar_proof_ref"),
        next_session_calendar_failure_ref=value.get("next_session_calendar_failure_ref"),
        decision_recipe_ref=value.get("decision_recipe_ref"),
        corporate_action_context_ref=value.get("corporate_action_context_ref"),
        dashboard_publication_policy=value.get("dashboard_publication_policy"),
        cutoff_ref=value.get("cutoff_ref"),
        registered_event_declaration_ref=value.get("registered_event_declaration_ref"),
    )
    return day, inputs


def run_native_input(*, workspace: str, input_ref: Mapping[str, str], resume: bool = False) -> dict:
    """Internal installed composition entry; no maintenance/provider invocation."""
    from scripts.daily_native_registry import NativeDailyRegistry

    if type(resume) is not bool:
        raise ContractError("NATIVE_INPUT_RESUME_FLAG_INVALID")
    day, inputs = load_native_inputs(workspace=workspace, input_ref=input_ref)
    registry = NativeDailyRegistry(workspace, day, inputs)
    with registry.runner.journal.locked():
        return run_loaded_native_input(registry, input_ref=input_ref, resume=resume)


def run_loaded_native_input(registry, *, input_ref: Mapping[str, str], resume: bool) -> dict:
    """Shared internal runner; caller owns the single day lock and loaded context."""
    registry.runner.journal._require_lock()
    if type(resume) is not bool:
        raise ContractError("NATIVE_INPUT_RESUME_FLAG_INVALID")
    verify_loaded_native_inputs(
        workspace=registry.workspace,
        input_ref=dict(input_ref),
        trade_date=registry.trade_date,
        inputs=registry.inputs,
    )
    validate_ref(input_ref)
    reader = SecureSystemStorage(registry.workspace)
    raw = reader.read_workspace_file_bytes(input_ref["path"], maximum_bytes=1024 * 1024)
    if raw.byte_sha256 != input_ref["sha256"]:
        raise ContractError("NATIVE_INPUT_DOCUMENT_CHANGED_BEFORE_RUN")
    registry.runner.journal.storage.write(
        str(
            registry.runner.journal.root / "inputs" / ("native-inputs-" + raw.byte_sha256 + ".json")
        ),
        raw.data,
    )
    return registry.runner.run_locked({}, resume=resume, resolve=registry.resolve)


def _store_arguments(root: Path, value: dict, preimages: dict) -> dict:
    calendar = value["calendar_ref"]
    return {
        **(
            {"registered_event_declaration_ref": value["registered_event_declaration_ref"]}
            if value["schema_version"] == "cn-daily-native-inputs.v7"
            else {}
        ),
        "project_root": root,
        "record_root": root / RECORD_ROOT,
        "expected_store_pointer_sha": preimages["store_pointer_sha256"],
        "expected_market_pointer_sha": preimages["market_pointer_sha256"],
        "expected_benchmark_pointer_sha": preimages["benchmark_pointer_sha256"],
        "expected_event_pointer_sha": preimages["event_pointer_sha256"],
        "calendar_receipt_path": root / calendar["path"],
        "calendar_receipt_sha": calendar["sha256"],
        "policy_path": value["store_policy_ref"]["path"],
        "policy_sha": value["store_policy_ref"]["sha256"],
        "retrospective_path": (
            value["retrospective_ref"]["path"] if value["retrospective_ref"] is not None else None
        ),
        "retrospective_sha": (
            value["retrospective_ref"]["sha256"] if value["retrospective_ref"] is not None else None
        ),
    }


def verify_loaded_native_inputs(
    *, workspace: str, input_ref: dict, trade_date: str, inputs: NativeDailyInputs
) -> None:
    """Recheck exact input/context bindings without another decoder or adapter creation."""
    root = Path(workspace).resolve(strict=True)
    ref = validate_ref(input_ref)
    raw = SecureSystemStorage(str(root)).read_workspace_file_bytes(
        ref["path"], maximum_bytes=1024 * 1024
    )
    if raw.byte_sha256 != ref["sha256"]:
        raise ContractError("NATIVE_LOADED_INPUT_SHA_MISMATCH")
    value = parse_canonical_json_bytes(raw.data)
    validate_native_input_shape(value)
    expected = {
        "trade_date": trade_date,
        "factor_pointer_sha256": inputs.factor_pointer_sha256,
        "release_ref": inputs.release_ref,
        "research_request_ref": inputs.research_request_ref,
        "store_plan_ref": inputs.store_plan_ref,
        "calendar_ref": inputs.calendar_ref,
        "market_snapshot_ref": inputs.market_snapshot_ref,
        "benchmark_ref": inputs.benchmark_ref,
        "risk_free_ref": inputs.risk_free_ref,
        "previous_trade_date": inputs.previous_trade_date,
        "adjustment_market_refs": inputs.adjustment_market_refs,
        "publish_current_dashboard": inputs.publish_current_dashboard,
    }
    if value["schema_version"] in {
        "cn-daily-native-inputs.v2",
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        expected.update(
            next_session_calendar_proof_ref=inputs.next_session_calendar_proof_ref,
            next_session_calendar_failure_ref=inputs.next_session_calendar_failure_ref,
        )
    elif (
        inputs.next_session_calendar_proof_ref is not None
        or inputs.next_session_calendar_failure_ref is not None
    ):
        raise ContractError("NATIVE_LOADED_INPUT_CONTEXT_MISMATCH")
    if value["schema_version"] in {
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        expected["decision_recipe_ref"] = inputs.decision_recipe_ref
        _verify_decision_recipe(workspace=workspace, value=value)
    elif inputs.decision_recipe_ref is not None:
        raise ContractError("NATIVE_LOADED_DECISION_PROFILE_MISMATCH")
    if value["schema_version"] in {
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        expected["corporate_action_context_ref"] = inputs.corporate_action_context_ref
    elif inputs.corporate_action_context_ref is not None:
        raise ContractError("NATIVE_LOADED_CORPORATE_PROFILE_MISMATCH")
    if value["schema_version"] in {
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        expected["dashboard_publication_policy"] = inputs.dashboard_publication_policy
    elif inputs.dashboard_publication_policy is not None:
        raise ContractError("NATIVE_LOADED_DASHBOARD_PROFILE_MISMATCH")
    if value["schema_version"] in {"cn-daily-native-inputs.v6", "cn-daily-native-inputs.v7"}:
        expected["cutoff_ref"] = inputs.cutoff_ref
    elif inputs.cutoff_ref is not None:
        raise ContractError("NATIVE_LOADED_CUTOFF_PROFILE_MISMATCH")
    if value["schema_version"] == "cn-daily-native-inputs.v7":
        expected["registered_event_declaration_ref"] = inputs.registered_event_declaration_ref
    elif inputs.registered_event_declaration_ref is not None:
        raise ContractError("NATIVE_LOADED_REGISTERED_PROFILE_MISMATCH")
    # Canonical-byte equality also rejects bool/int substitutions in loaded context.
    from quant_investor.contracts import canonical_json_bytes

    if canonical_json_bytes({key: value[key] for key in expected}) != canonical_json_bytes(
        expected
    ):
        raise ContractError("NATIVE_LOADED_INPUT_CONTEXT_MISMATCH")
    plan_ref = value["store_plan_ref"]
    plan = _load_json(
        root / plan_ref["path"], expected_sha=plan_ref["sha256"], label="loaded Store plan"
    )
    if not _store_plan_profile(value, plan) or plan.get("content_sha256") != content_sha256(plan):
        raise ContractError("NATIVE_LOADED_STORE_PLAN_INVALID")
    preimages = plan["preimages"]
    if inputs.event_pointer_sha256 != preimages[
        "event_pointer_sha256"
    ] or inputs.store_arguments != _store_arguments(root, value, preimages):
        raise ContractError("NATIVE_LOADED_STORE_CONTEXT_MISMATCH")
    if (
        SecureSystemStorage(str(root)).read_workspace_file_bytes(
            ref["path"], maximum_bytes=1024 * 1024
        )
        != raw
    ):
        raise ContractError("NATIVE_LOADED_INPUT_CHANGED")


def _verify_decision_recipe(*, workspace, value):
    if value["schema_version"] in {
        "cn-daily-native-inputs.v3",
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        from quant_investor.operations.decision_recipe import read_decision_recipe

        bound = read_decision_recipe(
            workspace=workspace,
            trade_date=value["trade_date"],
            recipe_ref=value["decision_recipe_ref"],
            research_request_ref=value["research_request_ref"],
            store_plan_ref=value["store_plan_ref"],
        )
        if value["schema_version"] in {"cn-daily-native-inputs.v6", "cn-daily-native-inputs.v7"}:
            cutoff = bound.get("cutoff")
            if (
                cutoff is None
                or cutoff["cutoff_ref"] != value["cutoff_ref"]
                or cutoff["corporate_action_context_ref"] != value["corporate_action_context_ref"]
            ):
                raise ContractError("NATIVE_INPUT_CUTOFF_BINDING_INVALID")
            registered = value["schema_version"] == "cn-daily-native-inputs.v7"
            if (
                cutoff["receipt"]["schema_version"]
                != ("cn-daily-research-cutoff.v2" if registered else "cn-daily-research-cutoff.v1")
                or registered
                and cutoff["receipt"]["registered_event_declaration_ref"]
                != value["registered_event_declaration_ref"]
            ):
                raise ContractError("NATIVE_INPUT_REGISTERED_CUTOFF_MISMATCH")
            handoff = cutoff["source_handoff"]
            for name in ("release_ref", "calendar_ref", "market_snapshot_ref"):
                if value[name] != handoff[name]:
                    raise ContractError("NATIVE_INPUT_CUTOFF_CORE_MISMATCH")
            if value["factor_pointer_sha256"] != handoff["factor_pointer_ref"]["sha256"]:
                raise ContractError("NATIVE_INPUT_CUTOFF_FACTOR_MISMATCH")
        elif bound.get("cutoff") is not None:
            raise ContractError("NATIVE_INPUT_UNEXPECTED_CUTOFF_PROFILE")
        if value["schema_version"] in {
            "cn-daily-native-inputs.v4",
            "cn-daily-native-inputs.v5",
            "cn-daily-native-inputs.v6",
            "cn-daily-native-inputs.v7",
        }:
            from quant_investor.strategy_records.corporate_contracts import context
            from quant_investor.strategy_records.event_receipts import read_event_source

            context(
                parse_canonical_json_bytes(
                    read_event_source(workspace, value["corporate_action_context_ref"])
                ),
                as_of=bound["recipe"]["as_of"],
            )


def _store_plan_profile(value, plan):
    version = validate_plan(plan, path=value["store_plan_ref"]["path"])
    registered = value["schema_version"] == "cn-daily-native-inputs.v7"
    return version == (2 if registered else 1) and (
        not registered
        or plan["registered_event_declaration_ref"] == value["registered_event_declaration_ref"]
    )
