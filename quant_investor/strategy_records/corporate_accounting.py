"""Attribute an observed historical native record transition, without applying events."""

from quant_investor.strategy_records.corporate_contracts import number
from quant_investor.strategy_records.holdings import load_holdings_identity
from quant_investor.intelligence.portfolio_state import normalize_positions


def adjustment(before, after, unit):
    a, b = number(before), number(after)
    return {
        "before": format(a, "f"),
        "after": format(b, "f"),
        "delta": format(b - a, "f"),
        "unit": unit,
    }


def empty_financial(code="ACCOUNTING_EVIDENCE_MISSING", records=None):
    return {
        "state": "UNCONFIRMED",
        "before_record_id": (records or {}).get("before_record_id"),
        "after_record_id": (records or {}).get("after_record_id"),
        "cost_basis_adjustment": None,
        "shares_adjustment": None,
        "cash_adjustment": None,
        "unattributed_transition": None,
        "source_refs": [],
        "blocker_codes": [code],
    }


def _isolated_application(manual, manifest, event_ref, symbol):
    if manual.get("schema_version") != "cn_aggressive_manual_execution.v3":
        return False
    expected = [
        {
            "schema_version": "cn-corporate-action-application.v1",
            "event_ref": event_ref,
            "symbol": symbol,
        }
    ]
    if manual.get("corporate_actions") != expected:
        return False
    fields = (
        "applied_local_trades",
        "applied_owner_declared_trades",
        "reconciled_source_trades",
        "rejected_or_pending_trades",
        "funding_events",
        "fills",
        "orders",
        "manual_changes",
    )
    if any(manual.get(field) != [] for field in fields):
        return False
    if any(
        type(manifest.get(k)) is not int or manifest[k] != 0
        for k in ("trade_count", "order_count", "fill_count")
    ):
        return False
    if any(
        number(manual.get(k)) != 0
        for k in ("net_external_flow", "excluded_external_flow", "gross_trade_value")
    ):
        return False
    return all(
        manual.get(k) in (None, [], {}) for k in ("funding_correction", "funding", "cash_movements")
    )


def _record_books(root, records, before, after, read):
    identities, cash, manuals, manifests, refs = {}, {}, {}, {}, []
    prefix = "results/strategy_records/CN/aggressive_tech_manufacturing/"
    for record_id in (before, after):
        record = records[record_id]
        for stem in ("ledger", "manual_manifest", "manifest"):
            ref = {"path": prefix + record[stem + "_path"], "sha256": record[stem + "_sha256"]}
            value = read(ref, json_document=stem != "ledger")
            refs.append(ref)
            if stem == "manual_manifest":
                manuals[record_id] = value
            if stem == "manifest":
                manifests[record_id] = value
        frame, cash[record_id] = load_holdings_identity(root, record)
        identities[record_id] = {
            r["symbol"]: r for r in normalize_positions(frame.to_dict("records"))
        }
    return identities, cash, manuals, manifests, refs


def reconcile_posting(*, root, catalog, ancestry, event, event_ref, read):
    pair = event["accounting_records"]
    result = empty_financial(records=pair)
    if pair is None:
        return result
    records = {r["record_id"]: r for r in catalog["records"]}
    lineage = {r["record_id"]: r for r in catalog["lineage_index"]}
    before, after = pair["before_record_id"], pair["after_record_id"]
    if (
        before not in ancestry
        or after not in ancestry
        or lineage[after]["source_record_id"] != before
    ):
        return empty_financial("ACCOUNTING_ANCESTRY_UNCONFIRMED", pair)
    left, right = lineage[before]["valuation_date"].replace("-", ""), lineage[after][
        "valuation_date"
    ].replace("-", "")
    if not left < event["effective_trade_date"] <= right:
        return empty_financial("ACCOUNTING_ANCESTRY_UNCONFIRMED", pair)
    identities, cash, manuals, manifests, refs = _record_books(root, records, before, after, read)
    symbol = event["symbol"]
    if symbol not in identities[before] or symbol not in identities[after]:
        return empty_financial("ACCOUNTING_ANCESTRY_UNCONFIRMED", pair)
    changes = {
        "cost_basis_adjustment": adjustment(
            identities[before][symbol]["cost_basis"], identities[after][symbol]["cost_basis"], "CNY"
        ),
        "shares_adjustment": adjustment(
            identities[before][symbol]["shares"], identities[after][symbol]["shares"], "SHARES"
        ),
        "cash_adjustment": adjustment(cash[before], cash[after], "CNY"),
    }
    result.update(
        unattributed_transition=changes,
        source_refs=sorted(refs, key=lambda r: (r["path"], r["sha256"])),
    )
    applications = manuals[after].get("corporate_actions")
    match = {
        "schema_version": "cn-corporate-action-application.v1",
        "event_ref": event_ref,
        "symbol": symbol,
    }
    if type(applications) is not list or applications.count(match) != 1:
        result["blocker_codes"] = ["ACCOUNTING_EVENT_LINK_MISSING"]
        return result
    others = (set(identities[before]) | set(identities[after])) - {symbol}
    changed = any(identities[before].get(s) != identities[after].get(s) for s in others)
    if changed or not _isolated_application(manuals[after], manifests[after], event_ref, symbol):
        result["blocker_codes"] = ["ACCOUNTING_MIXED_TRANSITION"]
        return result
    if not any(number(value["delta"]) != 0 for value in changes.values()):
        result["blocker_codes"] = ["ACCOUNTING_ZERO_DELTA"]
        return result
    result.update(
        state="OBSERVED_NATIVE_POSTING", **changes, unattributed_transition=None, blocker_codes=[]
    )
    return result
