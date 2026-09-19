"""Read-only reconstruction of a completed ledger; no maintenance or writer locks."""

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.completion_core import replay_completed_core
from quant_investor.operations.maintenance_handoff import read_recorded_maintenance_handoff
from quant_investor.operations.prospective_sources import read_research_source_times
from quant_investor.operations.prospective_timing import classify_daily_evidence
from quant_investor.operations.decision_recipe import native_portfolio_is_late
from quant_investor.operations.corporate_adapter import corporate_report_is_late
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_native_inputs import load_native_inputs, verify_loaded_native_inputs
from scripts.daily_materialization import MaterializedInputs, verify_materialized_inputs


def replay_completed_ledger(
    *, workspace: str, trade_date: str, completion_ref: dict, loaded_inputs=None
) -> dict:
    """Caller also replays all 16 business nodes before reporting EOD validity."""
    inspected = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    completion = inspected["recorded_completion"]
    if completion["schema_version"] != "cn-daily-eod-completion.v2":
        raise ContractError("LEDGER_COMPLETION_V2_REQUIRED")
    storage = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        validate_ref(ref)
        raw = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if raw.byte_sha256 != ref["sha256"]:
            raise ContractError("LEDGER_NATIVE_REPLAY_SHA_MISMATCH")
        observed[ref["path"]] = raw.data
        return parse_canonical_json_bytes(raw.data)

    snapshot = inspected["completed_handoff_snapshot"]
    ledger = snapshot.document("ledger")
    recovered = read_recorded_maintenance_handoff(snapshot)
    handoff, recipe = recovered["handoff"], recovered["recipe"]
    if loaded_inputs is None:
        day, inputs = load_native_inputs(
            workspace=workspace, input_ref=completion["native_inputs_ref"]
        )
    else:
        verify_loaded_native_inputs(
            workspace=workspace,
            input_ref=completion["native_inputs_ref"],
            trade_date=trade_date,
            inputs=loaded_inputs,
        )
        day, inputs = trade_date, loaded_inputs
    if day != trade_date:
        raise ContractError("LEDGER_NATIVE_REPLAY_DATE_MISMATCH")
    materialized = MaterializedInputs(
        completion["materialization_ref"], completion["native_inputs_ref"], inputs, "NO_ACTION"
    )
    verify_materialized_inputs(
        journal=DailyJournal(workspace, trade_date),
        recovered=recovered,
        materialized=materialized,
        readonly=True,
    )
    core = replay_completed_core(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )["recorded_timing"]
    # Both exact pointer paths are separately read; only equal original bytes may
    # represent the same native generation in the handoff and journal snapshots.
    pointer = read(core["factor_pointer_ref"])
    if read(handoff["factor_pointer_ref"]) != pointer:
        raise ContractError("LEDGER_NATIVE_POINTER_MISMATCH")
    sources = read_research_source_times(
        workspace=workspace, request_ref=dict(inputs.research_request_ref)
    )
    policy_ref = handoff.get("prospective_policy_ref")
    policy = read(policy_ref) if policy_ref is not None else None
    custody = inspected["recorded_custody_timing"]
    portfolio_late = native_portfolio_is_late(
        workspace=workspace,
        inputs=read(completion["native_inputs_ref"]),
    )
    native_document = read(completion["native_inputs_ref"])
    corporate_late = corporate_report_is_late(
        inputs=native_document,
        report=(
            read(
                read(completion["node_terminal_refs"]["corporate_action_recon"])["output_refs"][
                    "reconciliation"
                ]
            )
            if native_document["schema_version"]
            in {
                "cn-daily-native-inputs.v4",
                "cn-daily-native-inputs.v5",
                "cn-daily-native-inputs.v6",
                "cn-daily-native-inputs.v7",
            }
            else None
        ),
    )
    result = classify_daily_evidence(
        trade_date=trade_date,
        handoff=handoff,
        recipe=recipe,
        policy=policy,
        core_timing=core,
        node_custody=custody,
        source_times=sources,
        synthetic=completion["synthetic"],
        portfolio_late=portfolio_late,
        corporate_late=corporate_late,
    )
    expected = {
        "core_timing": core,
        "source_times": sources,
        "node_custody": custody,
        "classification": result["classification"],
        "prospective": result["prospective"],
        "recomputed": completion["synthetic"]
        or portfolio_late
        or corporate_late
        or recipe["retrospective_ref"] is not None
        or handoff["schema_version"] == "cn-daily-maintenance-handoff.v3",
    }
    if canonical_json_bytes({key: ledger[key] for key in expected}) != canonical_json_bytes(
        expected
    ):
        raise ContractError("LEDGER_NATIVE_DERIVATION_MISMATCH")
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
            raise ContractError("LEDGER_NATIVE_SOURCE_CHANGED")
    if (
        inspect_recorded_completion(
            workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
        )
        != inspected
    ):
        raise ContractError("LEDGER_NATIVE_COMPLETION_CHANGED")
    return {
        "ledger_ref": completion["prospective_ledger_ref"],
        "classification": result["classification"],
        "prospective": result["prospective"],
        "synthetic": completion["synthetic"],
        "recomputed": expected["recomputed"],
        "validation_scope": "NATIVE_LEDGER_DERIVATION",
        "native_business_replay_required": True,
    }


def inspect_daily_evidence_availability(
    *, workspace: str, trade_date: str, completion_ref: dict
) -> dict:
    """Select one explicit full-native EOD, never a current-head/latest-file scan."""
    from scripts.daily_completion_replay import replay_native_completion
    from quant_investor.operations.daily_contract import EOD_NODE_IDS
    from quant_investor.operations.daily_journal import _validate_day

    _validate_day(trade_date)
    selected = validate_ref(completion_ref)
    result = replay_native_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=selected
    )
    if (
        result.get("native_replay_validated") is not True
        or result.get("completion_ref") != selected
        or result.get("trade_date") != trade_date
        or result.get("validated_nodes") != sorted(EOD_NODE_IDS)
        or type(result.get("synthetic")) is not bool
    ):
        raise ContractError("DAILY_ELIGIBLE_NATIVE_REPLAY_INVALID")
    ledger = result.get("ledger")
    if ledger is None:
        return {
            "completion_ref": selected,
            "trade_date": trade_date,
            "ledger_ref": None,
            "classification": "UNKNOWN_LEGACY",
            "prospective": False,
            "synthetic": result["synthetic"],
            "recomputed": None,
            "validation_scope": "FULL_NATIVE_EOD_AVAILABILITY",
        }
    _validate_availability_ledger(ledger, synthetic=result["synthetic"])
    return {
        "completion_ref": selected,
        "trade_date": trade_date,
        "ledger_ref": validate_ref(ledger["ledger_ref"]),
        **{
            key: ledger[key] for key in ("classification", "prospective", "synthetic", "recomputed")
        },
        "validation_scope": "FULL_NATIVE_EOD_AVAILABILITY",
    }


def _validate_availability_ledger(ledger, *, synthetic):
    if (
        type(ledger) is not dict
        or ledger.get("validation_scope") != "NATIVE_LEDGER_DERIVATION"
        or ledger.get("classification")
        not in {"CONTEMPORANEOUS", "LATE_REGISTERED", "UNKNOWN_LEGACY", "RETROSPECTIVE_RECOMPUTE"}
        or any(
            type(ledger.get(key)) is not bool for key in ("prospective", "synthetic", "recomputed")
        )
        or ledger["synthetic"] != synthetic
        or ledger["prospective"] is not (ledger["classification"] == "CONTEMPORANEOUS")
        or (ledger["prospective"] and (synthetic or ledger["recomputed"]))
        or (
            (synthetic or ledger["recomputed"])
            and ledger["classification"] != "RETROSPECTIVE_RECOMPUTE"
        )
    ):
        raise ContractError("DAILY_EVIDENCE_AVAILABILITY_INVALID")


def select_eligible_daily_evidence(
    *, workspace: str, trade_date: str, completion_ref: dict
) -> dict:
    """Keep the existing Morning selection contract over the shared native reader."""
    from quant_investor.operations.daily_journal import FALSE_AUTHORITY

    available = inspect_daily_evidence_availability(
        workspace=workspace, trade_date=trade_date, completion_ref=completion_ref
    )
    if available["prospective"] is not True:
        raise ContractError("DAILY_EVIDENCE_NOT_PROSPECTIVE")
    return {
        "schema_version": "cn-daily-eligible-evidence-selection.v1",
        "market": "CN",
        "trade_date": trade_date,
        "completion_ref": available["completion_ref"],
        "ledger_ref": available["ledger_ref"],
        "eligibility_scope": "LOCAL_COORDINATOR_AVAILABILITY",
        "factor_admission": False,
        "authority": dict(FALSE_AUTHORITY),
    }
