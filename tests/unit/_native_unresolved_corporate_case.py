"""Synthetic historical unresolved action for the separate sixth acceptance day."""

from copy import deepcopy
import hashlib
import json


def read(workspace, ref):
    raw = (workspace / ref["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("case6 retained source changed")
    return json.loads(raw)


def prepare(workspace, config_ref, *, day, symbol):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.operations.daily_preparation_contract import validate_config
    from quant_investor.operations.research_corporate_inputs import validate_corporate_template
    from quant_investor.strategy_records.corporate_contracts import event

    if day != "20260903":
        raise ValueError("case6 must remain outside the normal five-session sequence")
    config = deepcopy(read(workspace, config_ref))
    template = deepcopy(read(workspace, config["corporate_action_template_ref"]))
    if template["named_event_refs"] is not None or template["anchor_reviews_ref"] is not None:
        raise ValueError("case6 requires the unchanged normal fixture as its baseline")
    prefix = workspace / "configured-case6-inputs" / day
    prefix.mkdir(parents=True, mode=0o700)
    prefix.parent.chmod(0o700)

    def put(name, value):
        raw = canonical_json_bytes(value)
        path = prefix / name
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o600)
        return {
            "path": str(path.relative_to(workspace)),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    announcement = put(
        "announcement.json",
        {"synthetic": True, "kind": "DIVIDEND", "symbol": symbol},
    )
    declared = {
        "schema_version": "cn-corporate-action-event.v1",
        "event_id": "synthetic-unresolved-historical-dividend",
        "symbol": symbol,
        "kind": "DIVIDEND",
        "effective_trade_date": "20260820",
        "announced_at": "2026-08-19T08:00:00Z",
        "announcement_ref": announcement,
        "accounting_records": None,
    }
    event(declared, as_of="2026-09-03T13:20:00Z")
    event_ref = put("unresolved-event.json", declared)
    template["named_event_refs"] = [event_ref]
    validate_corporate_template(template)
    config["corporate_action_template_ref"] = put("corporate-template.json", template)
    validate_config(config)
    changed_ref = put("configuration.json", config)
    if {k for k in config if config[k] != read(workspace, config_ref)[k]} != {
        "corporate_action_template_ref"
    }:
        raise ValueError("case6 changed unrelated configured inputs")
    return changed_ref, config, event_ref


def verify(workspace, *, completion_ref, replay, event_ref):
    from quant_investor.contracts import canonical_json_bytes, validate_artifact

    completion = read(workspace, completion_ref)
    day = completion["trade_date"]
    if (
        day != "20260903"
        or completion["synthetic"] is not True
        or replay["native_replay_validated"] is not True
        or replay["completion_ref"] != completion_ref
        or len(replay["validated_nodes"]) != 16
    ):
        raise ValueError("case6 requires actual complete sixth-day native replay")
    corporate = read(workspace, completion["node_terminal_refs"]["corporate_action_recon"])
    store = read(workspace, completion["node_terminal_refs"]["store"])
    report_ref = corporate["output_refs"]["reconciliation"]
    report = validate_artifact(canonical_json_bytes(read(workspace, report_ref)))["payload"]
    projection = read(workspace, corporate["output_refs"]["financial_events"])
    declared = read(workspace, event_ref)
    row = next(x for x in report["company_rows"] if x["symbol"] == declared["symbol"])
    events = [x for x in row["events"] if x["event_ref"] == event_ref]
    if (
        len(events) != 1
        or events[0]["financial"]["state"] != "UNCONFIRMED"
        or "ACCOUNTING_EVIDENCE_MISSING" not in events[0]["financial"]["blocker_codes"]
        or events[0]["reconciliation_state"] != "UNCONFIRMED"
        or row["threshold_state"] != "NON_EXECUTABLE"
        or report["summary_state"] != "UNCONFIRMED"
        or corporate["state"] != "SUCCEEDED"
        or store["state"] != "SUCCEEDED"
        or replay["corporate_action"] != projection
        or projection["reconciliation_ref"] != report_ref
        or projection["threshold_anchor_mutation"] is not False
    ):
        raise ValueError("case6 native unresolved action or threshold evidence differs")
    committed = read(workspace, store["output_refs"]["completion"])
    manual = read(workspace, store["output_refs"]["manual"])
    if (
        committed["status"] != "COMMITTED"
        or committed["committed_through"].replace("-", "") != day
        or manual["valuation_trade_date"].replace("-", "") != day
        or any(manual[k] != 0 for k in ("trade_count", "order_count", "fill_count"))
    ):
        raise ValueError("case6 native daily valuation did not complete without trades")
    return {
        "case": 6,
        "synthetic": True,
        "trade_date": day,
        "completion_ref": completion_ref,
        "event_ref": event_ref,
        "reconciliation_ref": report_ref,
        "store_completion_ref": store["output_refs"]["completion"],
        "unresolved_state": events[0]["reconciliation_state"],
        "financial_evidence": events[0]["financial"]["state"],
        "moving_threshold": row["threshold_state"],
        "threshold_anchor_mutation": False,
        "valuation_committed_through": committed["committed_through"],
        "full_eod_native_replay_verified": True,
        "historical_action_scope_only": True,
        "production_deployed": False,
    }
