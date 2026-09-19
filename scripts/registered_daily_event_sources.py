"""Native registered-event evidence reader; publication is owned by the manager."""

from pathlib import Path
import hashlib
import json
import re

from scripts.cn_dashboard_common import validate_record
from quant_investor.strategy_records import registered_event_contracts as contracts
from quant_investor.strategy_records import store, performance, event_store
from quant_investor.strategy_records.daily_event_source import DailyEventSources

Error = contracts.RegisteredEventError
ROOT = contracts.RECORD_ROOT
MAX_JSON = 4 * 1024 * 1024


class RegisteredSources(DailyEventSources):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.record_root = self.root / ROOT

    def document(self, ref):
        raw = self.read(ref)
        if len(raw) > MAX_JSON:
            raise Error("REGISTERED_EVENT_JSON_BUDGET_EXCEEDED")
        value = json.loads(raw)
        if type(value) is not dict:
            raise Error("REGISTERED_EVENT_DOCUMENT_INVALID")
        return value

    def catalog(self, pointer_ref):
        pointer_ref = contracts.ref(pointer_ref)
        pointer, catalog = store.load_catalog_snapshot_bytes(
            self.record_root,
            pointer_bytes=self.read(pointer_ref),
            expected_pointer_sha256=pointer_ref["sha256"],
        )
        if catalog["schema_id"] != store.CATALOG_SCHEMA_V3:
            raise Error("REGISTERED_EVENT_STORE_V3_REQUIRED")
        catalog_ref = {
            "path": ROOT + "/" + pointer["catalog_path"],
            "sha256": pointer["catalog_sha256"],
        }
        if self.document(catalog_ref) != catalog:
            raise Error("REGISTERED_EVENT_CATALOG_CHANGED")
        rows = [r for r in catalog["records"] if r["record_id"] == pointer["active_record_id"]]
        if len(rows) != 1 or rows[0].get("state") != "ONLINE":
            raise Error("REGISTERED_EVENT_ACTIVE_RECORD_UNAVAILABLE")
        row = rows[0]
        directory = self.record_root / row["record_id"]
        from scripts.manage_cn_strategy_records import build_inventory

        inventory = build_inventory(directory, enforce_new_record_budget=False)
        if any(row.get(k) != v for k, v in inventory.items()):
            raise Error("REGISTERED_EVENT_INVENTORY_CHANGED")
        for item in inventory["inventory"]:
            if item["type"] == "file":
                self.read(
                    {"path": f"{ROOT}/{row['record_id']}/{item['path']}", "sha256": item["sha256"]}
                )
        strict = validate_record(directory, self.record_root, self.root)
        refs = {}
        for name, key in (
            ("manifest", "manifest"),
            ("manual_manifest", "manual_manifest"),
            ("ledger", "ledger"),
            ("pnl", "pnl"),
        ):
            reference = {"path": ROOT + "/" + row[key + "_path"], "sha256": row[key + "_sha256"]}
            if (
                strict[key + "_path"] != reference["path"]
                or strict[key + "_sha256"] != reference["sha256"]
            ):
                raise Error("REGISTERED_EVENT_RECORD_CLOSURE_CHANGED")
            self.read(reference)
            refs[name] = reference
        if strict["financial_state_sha256"] != row["financial_state_sha256"]:
            raise Error("REGISTERED_EVENT_FINANCIAL_STATE_CHANGED")
        for source in strict["source_refs"]:
            self.read(source)
        history_ref = catalog["performance_history_ref"]
        for name in ("manifest", "series", "owner_declaration"):
            value = history_ref[name]
            reference = {"path": ROOT + "/" + value["path"], "sha256": value["sha256"]}
            self.read(reference)
            refs["performance_" + name] = reference
        history = performance.load_performance_history(self.record_root, history_ref)
        tail = history["rows"][-1]
        if tail["record_id"] != strict["record"] or tail["valuation_date"] != strict["data_date"]:
            raise Error("REGISTERED_EVENT_PERFORMANCE_RECORD_MISMATCH")
        for source_key, row_key in (
            ("manual_manifest_sha256", "manual_manifest_sha256"),
            ("ledger_sha256", "ledger_parquet_sha256"),
            ("financial_state_sha256", "financial_state_sha256"),
        ):
            if tail[row_key] != strict[source_key]:
                raise Error("REGISTERED_EVENT_PERFORMANCE_SHA_MISMATCH")
        for key, field in (
            ("cash_after", "cash_cny"),
            ("market_value_after", "equity_market_value_cny"),
            ("total_value_after", "raw_nav_cny"),
            ("portfolio_pnl_after", "portfolio_pnl_cny"),
        ):
            if performance.money(strict["accounting"][key], label=key) != tail[field]:
                raise Error("REGISTERED_EVENT_PERFORMANCE_ACCOUNTING_MISMATCH")
        return {
            "pointer": pointer,
            "catalog": catalog,
            "catalog_ref": catalog_ref,
            "record": strict,
            "refs": refs,
            "history": history,
            "manual": self.document(refs["manual_manifest"]),
        }

    def pair(self, fact, writer_pointer_ref):
        contracts.validate_fact(fact)
        if writer_pointer_ref["sha256"] != fact["writer_pointer_sha256"]:
            raise Error("REGISTERED_EVENT_WRITER_SHA_MISMATCH")
        baseline = self.catalog(fact["baseline_store_pointer_ref"])
        writer = self.catalog(writer_pointer_ref)
        bp, wp = baseline["pointer"], writer["pointer"]
        if (
            wp["previous_pointer_sha256"] != fact["baseline_store_pointer_ref"]["sha256"]
            or wp["previous_record_id"] != bp["active_record_id"]
            or wp["active_record_id"] != fact["writer_record_id"]
            or writer["catalog"]["lineage_index"][:-1] != baseline["catalog"]["lineage_index"]
            or writer["catalog"]["lineage_index"][-1]["source_record_id"] != bp["active_record_id"]
            or writer["catalog"]["lineage_index"][-1]["record_id"] != wp["active_record_id"]
            or writer["history"]["rows"][:-1] != baseline["history"]["rows"]
            or contracts.stamp(wp["published_at"]) < contracts.stamp(bp["published_at"])
        ):
            raise Error("REGISTERED_EVENT_DIRECT_TRANSITION_UNPROVEN")
        by_id = {r["record_id"]: r for r in writer["catalog"]["records"]}
        if any(by_id.get(r["record_id"]) != r for r in baseline["catalog"]["records"]):
            raise Error("REGISTERED_EVENT_PRIOR_RECORD_CHANGED")
        before, after = baseline["history"]["rows"][-1], writer["history"]["rows"][-1]
        if after["evidence_kind"] != "REGISTERED_APPLIED_TRADES" or any(
            before[k] != after[k] for k in ("unit_count", "excluded_external_flow_cny")
        ):
            raise Error("REGISTERED_EVENT_TRADE_PERFORMANCE_CHANGED")
        profile = contracts.validate_buy_transition(
            baseline=baseline["record"],
            writer=writer["record"],
            manual=writer["manual"],
            fact=fact,
            record_refs=writer["refs"],
        )
        self.recheck()
        return {"baseline": baseline, "writer": writer, "profile": profile}

    def no_empty_event(self, trade_date):
        path = ROOT + "/_event_store/current.v1.json"
        raw = self.files.optional(path)
        if raw is None:
            return
        loaded = event_store.load_frozen_generation(
            self.record_root / "_event_store",
            pointer_bytes=raw,
            expected_pointer_sha256=hashlib.sha256(raw).hexdigest(),
        )
        generation = loaded["pointer"]["generation"]
        self.read(
            {"path": ROOT + "/_event_store/" + generation["path"], "sha256": generation["sha256"]}
        )
        if any(row["trade_date"] == trade_date for row in loaded["closures"]):
            raise Error("REGISTERED_EVENT_EMPTY_CLOSURE_CONFLICT")


def declaration_document(*, fact, fact_ref, pair, registered_at):
    writer, baseline = pair["writer"], pair["baseline"]
    if any(
        contracts.stamp(registered_at) < contracts.stamp(stamp)
        for stamp in (
            fact["owner_declared_at"],
            writer["pointer"]["published_at"],
            baseline["pointer"]["published_at"],
        )
    ):
        raise Error("REGISTERED_EVENT_REGISTRATION_TIME_INVALID")
    value = {k: fact[k] for k in contracts.COMMON - {"schema_version", "content_sha256"}}
    value.update(
        schema_version=contracts.DECLARATION_SCHEMA,
        declaration_id="registered-" + fact["writer_pointer_sha256"],
        owner_fact_ref=fact_ref,
        registered_at=registered_at,
        baseline_catalog_ref=baseline["catalog_ref"],
        baseline_record_id=baseline["record"]["record"],
        writer_store_pointer_ref={
            "path": contracts.paths(fact["writer_pointer_sha256"])["pointer"],
            "sha256": fact["writer_pointer_sha256"],
        },
        writer_catalog_ref=writer["catalog_ref"],
        writer_record_refs=writer["refs"],
        late_event_behavior="OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
    )
    return contracts.validate_declaration(contracts.seal(value))


def read_declaration(*, workspace, declaration_ref):
    """Replay explicit immutable custody only; no current pointers or directory selection."""
    sources = RegisteredSources(workspace)
    declaration_ref = contracts.ref(declaration_ref)
    value = contracts.validate_declaration(
        sources.document(declaration_ref), declaration_ref=declaration_ref
    )
    fact = contracts.validate_fact(sources.document(value["owner_fact_ref"]))
    pair = sources.pair(fact, value["writer_store_pointer_ref"])
    expected = declaration_document(
        fact=fact, fact_ref=value["owner_fact_ref"], pair=pair, registered_at=value["registered_at"]
    )
    if expected != value:
        raise Error("REGISTERED_EVENT_DECLARATION_REPLAY_MISMATCH")
    sources.recheck()
    return {
        "declaration": value,
        "declaration_ref": declaration_ref,
        **pair,
        "source_refs": [
            {"path": p, "sha256": hashlib.sha256(raw).hexdigest()}
            for p, raw in sorted(sources.files.observed.items())
        ],
    }


def select_daily_source(*, workspace, store_pointer_ref, trade_date):
    """Classify one exact observed book before producers; never enroll a closed day.

    Original request recovery belongs to the caller and must precede this fresh
    selection. A finalized v2 book alone cannot supply original Calendar custody.
    """
    from quant_investor.strategy_records.event_contracts import event_date
    from quant_investor.strategy_records.close_plan_contracts import RECEIPT_V2, validate_plan

    event_date(trade_date)
    sources = RegisteredSources(workspace)
    observed_ref = contracts.ref(store_pointer_ref)
    selected = sources.catalog(observed_ref)
    record = selected["record"]
    if record["data_date"] > trade_date:
        raise Error("REGISTERED_SOURCE_TARGET_BEFORE_STORE")
    if record["official_valuation"] is True:
        if record["data_date"] != trade_date:
            sources.recheck()
            return {"state": "ORDINARY"}
        receipts = [
            row
            for row in selected["catalog"]["receipts"]
            if row.get("record_id") == record["record"] and row.get("schema_id") == RECEIPT_V2
        ]
        if not receipts:
            sources.recheck()
            return {"state": "ORDINARY"}
        if len(receipts) != 1:
            raise Error("REGISTERED_SOURCE_FINAL_RECEIPT_AMBIGUOUS")
        transaction = receipts[0]["transaction_id"]
        if (
            type(transaction) is not str
            or re.fullmatch(r"daily-close-[0-9]{8}-[0-9a-f]{16}", transaction) is None
        ):
            raise Error("REGISTERED_SOURCE_FINAL_TRANSACTION_INVALID")
        plan_ref = sources.files.pin(
            ROOT + f"/_record_store/daily_close_transactions/{transaction}/plan.v2.json"
        )
        plan = sources.document(plan_ref)
        if validate_plan(plan, path=plan_ref["path"]) != 2:
            raise Error("REGISTERED_SOURCE_FINAL_PLAN_INVALID")
        from scripts.cn_official_close_batch import inspect_frozen_close_commit

        final = inspect_frozen_close_commit(
            record_root=sources.record_root,
            transaction_id=transaction,
            expected_plan_sha=plan_ref["sha256"],
            expected_source_pointer_sha=plan["preimages"]["store_pointer_sha256"],
            expected_target=trade_date,
            plan_version=2,
        )
        if final["pointer_sha256"] != observed_ref["sha256"]:
            raise Error("REGISTERED_SOURCE_FINAL_POINTER_MISMATCH")
        sources.recheck()
        return {
            "state": "BLOCKED_FINALIZED_V2_INPUT_CUSTODY_MISSING",
            "registered_store_plan_ref": plan_ref,
            "registered_event_declaration_ref": plan["registered_event_declaration_ref"],
        }
    if record["data_date"] != trade_date:
        raise Error("REGISTERED_SOURCE_HISTORICAL_FINALIZATION_UNSUPPORTED")
    declaration_path = contracts.paths(observed_ref["sha256"])["declaration"]
    if sources.files.optional(declaration_path) is None:
        raise Error("REGISTERED_SOURCE_DECLARATION_REQUIRED")
    declaration_ref = sources.files.pin(declaration_path)
    proof = read_declaration(workspace=workspace, declaration_ref=declaration_ref)
    declaration = proof["declaration"]
    if (
        declaration["writer_store_pointer_ref"]["sha256"] != observed_ref["sha256"]
        or declaration["trade_date"] != trade_date
        or proof["writer"]["record"] != record
    ):
        raise Error("REGISTERED_SOURCE_WRITER_BINDING_MISMATCH")
    sources.no_empty_event(trade_date)
    for reference in proof["source_refs"]:
        sources.read(reference)
    sources.recheck()
    return {
        "state": "REGISTERED_INTRADAY",
        "registered_event_declaration_ref": declaration_ref,
        "registered_store_plan_ref": None,
        "decision_baseline_pointer_ref": declaration["baseline_store_pointer_ref"],
        "writer_pointer_ref": declaration["writer_store_pointer_ref"],
        "source_refs": [
            {"path": p, "sha256": hashlib.sha256(raw).hexdigest()}
            for p, raw in sorted(sources.files.observed.items())
        ],
    }


def read_recipe_source(*, workspace, recipe):
    """Replay a v6 recipe's writer and its exact previous-EOD baseline."""
    from scripts.daily_completion_store import replay_completed_store
    from quant_investor.operations.automatic_catchup_contract import completion_day

    if recipe.get("schema_version") != "cn-daily-execute-recipe.v6":
        raise Error("REGISTERED_RECIPE_VERSION_REQUIRED")
    previous = recipe.get("previous_completion_ref")
    if previous is None or recipe.get("bootstrap_ref") is not None:
        raise Error("REGISTERED_PREVIOUS_EOD_REQUIRED")
    proof = read_declaration(
        workspace=workspace, declaration_ref=recipe["registered_event_declaration_ref"]
    )
    declaration = proof["declaration"]
    previous_day = completion_day(previous)
    if (
        declaration["trade_date"].replace("-", "") != recipe["target_trade_date"]
        or proof["baseline"]["record"]["data_date"].replace("-", "") != previous_day
        or recipe["store_preimages"]["store_pointer_ref"]["sha256"]
        != declaration["writer_store_pointer_ref"]["sha256"]
        or recipe.get("retrospective_ref") is not None
    ):
        raise Error("REGISTERED_RECIPE_SOURCE_BINDING_INVALID")
    completed = replay_completed_store(
        workspace=str(workspace), trade_date=previous_day, completion_ref=previous
    )
    if completed["output_refs"]["pointer"] != declaration["baseline_store_pointer_ref"]:
        raise Error("REGISTERED_PREVIOUS_EOD_BASELINE_MISMATCH")
    return proof


def prepare_publication(*, workspace, owner_fact_ref, expected_pointer_sha):
    sources = RegisteredSources(workspace)
    fact_ref = sources.ref(owner_fact_ref)
    fact = contracts.validate_fact(sources.document(fact_ref))
    current = sources.files.pin(ROOT + "/_record_store/current.v1.json")
    if (
        current["sha256"] != expected_pointer_sha
        or expected_pointer_sha != fact["writer_pointer_sha256"]
    ):
        raise Error("REGISTERED_EVENT_CURRENT_POINTER_CHANGED")
    pair = sources.pair(fact, current)
    sources.no_empty_event(fact["trade_date"])
    selected = contracts.paths(expected_pointer_sha)
    # This exact path is an owned publication output. Do not register its
    # expected absence as an immutable input that our own write would violate.
    from quant_investor.operations.automatic_catchup_resolution import optional_bytes

    existing = optional_bytes(sources.files, selected["declaration"])
    previous = None
    if existing is not None:
        reference = {
            "path": selected["declaration"],
            "sha256": hashlib.sha256(existing).hexdigest(),
        }
        previous = read_declaration(workspace=workspace, declaration_ref=reference)["declaration"]
        if previous["owner_fact_ref"] != fact_ref:
            raise Error("REGISTERED_EVENT_FACT_IDENTITY_CONFLICT")
    sources.recheck()
    return {
        "sources": sources,
        "fact": fact,
        "fact_ref": fact_ref,
        "pair": pair,
        "current_ref": current,
        "previous": previous,
    }


def publish_prepared(prepared, *, registered_at):
    """Retired writer: publication must enter the operation-locked Record manager."""
    raise Error("REGISTERED_EVENT_MANAGER_PUBLICATION_REQUIRED")


def assert_no_registered_event(*, workspace, trade_date):
    """Manager-lock guard: a known registered change can never be declared empty."""
    sources = RegisteredSources(workspace)
    pointer_ref = sources.files.pin(ROOT + "/_record_store/current.v1.json")
    pointer, catalog = store.load_catalog_snapshot_bytes(
        sources.record_root,
        pointer_bytes=sources.read(pointer_ref),
        expected_pointer_sha256=pointer_ref["sha256"],
    )
    if catalog["schema_id"] != store.CATALOG_SCHEMA_V3:
        raise Error("REGISTERED_EVENT_EMPTY_GUARD_STORE_V3_REQUIRED")
    if (
        sources.document(
            {"path": ROOT + "/" + pointer["catalog_path"], "sha256": pointer["catalog_sha256"]}
        )
        != catalog
    ):
        raise Error("REGISTERED_EVENT_CATALOG_CHANGED")
    chain = performance.validate_lineage_index(
        catalog["lineage_index"], active_record_id=pointer["active_record_id"]
    )
    by_id = {row["record_id"]: row for row in catalog["lineage_index"]}
    for record_id in chain:
        row = by_id[record_id]
        if row["valuation_date"] != trade_date and record_id != pointer["active_record_id"]:
            continue
        if row["valuation_date"] == trade_date and row["execution_class"] != "NO_TRADE":
            raise Error("REGISTERED_EVENT_NONEMPTY_DAY_CANNOT_BE_EMPTY")
        record = validate_record(sources.record_root / record_id, sources.record_root, sources.root)
        for source in record["source_refs"]:
            sources.read(source)
        manual = sources.document(
            {"path": record["manual_manifest_path"], "sha256": record["manual_manifest_sha256"]}
        )
        same_day = record["data_date"] == trade_date or any(
            isinstance(item, dict)
            and str(item.get("trade_date", "")).replace("-", "") == trade_date.replace("-", "")
            for key in ("applied_owner_declared_trades", "applied_local_trades")
            for item in (manual.get(key) or [])
        )
        if same_day and (
            record["execution_kind"] == "applied_effective_ledger"
            or any(record.get(k) is not None for k in ("funding", "funding_correction"))
            or any(
                manual.get(k) not in (None, [])
                for k in ("funding_events", "corporate_actions", "manual_changes")
            )
            or manual.get("corporate_action_application_ref") is not None
            or manual.get("corporate_action_application") is not None
            or any(
                k in manual and contracts.number(manual[k]) != 0
                for k in ("trade_count", "order_count", "fill_count")
            )
        ):
            raise Error("REGISTERED_EVENT_NONEMPTY_DAY_CANNOT_BE_EMPTY")
    sources.recheck()
