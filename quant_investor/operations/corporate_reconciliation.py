"""Replay exact corporate sources and native historical books; never select a head."""

from pathlib import Path

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.strategy_records import corporate_contracts as contracts
from quant_investor.strategy_records.corporate_accounting import reconcile_posting
from quant_investor.strategy_records.event_receipts import read_event_source
from quant_investor.strategy_records.store import load_catalog_snapshot_bytes
from quant_investor.strategy_records.holdings import load_holdings_identity
from quant_investor.intelligence.portfolio_state import normalize_positions
from quant_investor.intelligence.corporate_reconciliation import (
    tracking_window,
    build_reconciliation,
)
from .daily_contract import ContractError, validate_ref
from .decision_recipe import read_decision_recipe

ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"


class ReconciliationSources:
    def __init__(
        self, evidence, *, context_ref, decision_recipe_ref, research_request_ref, store_plan_ref
    ):
        self.evidence = evidence
        self.workspace = Path(evidence.workspace)
        self.trade_date = evidence.trade_date
        self.seen = {}
        self.context_ref = validate_ref(context_ref)
        self.decision_recipe_ref = validate_ref(decision_recipe_ref)
        self.plan_ref = validate_ref(store_plan_ref)
        bound = read_decision_recipe(
            workspace=self.workspace,
            trade_date=self.trade_date,
            recipe_ref=decision_recipe_ref,
            research_request_ref=research_request_ref,
            store_plan_ref=store_plan_ref,
        )
        self.portfolio = bound["portfolio"]["payload"]
        self.portfolio_ref = bound["recipe"]["portfolio_source_ref"]
        self.as_of = bound["recipe"]["as_of"]
        for ref in (
            decision_recipe_ref,
            research_request_ref,
            self.portfolio_ref,
            *self.portfolio["source_refs"],
        ):
            self.read(ref, json_document=False)
        self.context = contracts.context(self.read(context_ref, canonical=True), as_of=self.as_of)
        self._retain_empty_event_sources()
        self.policy_ref = self.context["tracking_policy_ref"]
        self.policy = self.read(self.policy_ref)
        self._policy()
        self.pointer, self.catalog = load_catalog_snapshot_bytes(
            self.workspace / ROOT,
            pointer_bytes=self.read(self.portfolio["frozen_pointer_ref"], json_document=False),
            expected_pointer_sha256=self.portfolio["frozen_pointer_ref"]["sha256"],
        )
        self.records = {r["record_id"]: r for r in self.catalog["records"]}
        self.lineage = {r["record_id"]: r for r in self.catalog["lineage_index"]}
        self.ancestry = []
        cursor = self.pointer["active_record_id"]
        while cursor is not None:
            if cursor in self.ancestry or cursor not in self.lineage:
                raise ContractError("CORPORATE_FROZEN_ANCESTRY_INVALID")
            self.ancestry.append(cursor)
            cursor = self.lineage[cursor]["source_record_id"]
        self.identities = {}
        self.events = self._events()
        self.reviews = self._reviews()
        self.company_rows = self._companies()
        self.recheck()

    def read(self, ref, *, json_document=True, canonical=False):
        ref = validate_ref(ref)
        raw = read_event_source(self.workspace, ref)
        key = (ref["path"], ref["sha256"])
        if key in self.seen and self.seen[key] != raw:
            raise ContractError("CORPORATE_SOURCE_CHANGED")
        self.seen[key] = raw
        if not json_document:
            return raw
        if canonical:
            return parse_canonical_json_bytes(raw)
        return parse_json_bytes(raw, label="corporate native source", require_canonical=False)

    def recheck(self):
        for (path, sha), raw in list(self.seen.items()):
            if read_event_source(self.workspace, {"path": path, "sha256": sha}) != raw:
                raise ContractError("CORPORATE_SOURCE_CHANGED")

    def _retain_empty_event_sources(self):
        from .corporate_actions import EVENT_ROOT
        from quant_investor.strategy_records.event_contracts import SYMBOLIC_RECEIPT
        from quant_investor.strategy_records.event_receipts import resolve_catalog_event_receipt

        self.read(self.evidence.event_pointer_ref, json_document=False)
        events = self.evidence._events()
        generation = events["pointer"]["generation"]
        self.read(
            {"path": EVENT_ROOT + "/" + generation["path"], "sha256": generation["sha256"]},
            json_document=False,
        )
        for closure in events["closures"]:
            if closure["trade_date"].replace("-", "") != self.trade_date:
                continue
            for key in ("policy_ref", "owner_declaration_ref", "source_receipt_ref"):
                ref = closure[key]
                if ref is None:
                    continue
                if SYMBOLIC_RECEIPT.fullmatch(ref["path"]):
                    resolved = resolve_catalog_event_receipt(
                        workspace=self.workspace, closure=closure
                    )
                    ref = resolved["catalog_ref"]
                self.read(ref, json_document=False)

    def _policy(self):
        from quant_investor.strategy_records.risk_policy_contract import validate_trailing_policy

        validate_trailing_policy(self.policy, as_of=self.as_of)

    def _events(self):
        ref = self.context["named_events_ref"]
        result, ids = [], set()
        if ref is None:
            return result
        for source in contracts.event_list(self.read(ref, canonical=True), as_of=self.as_of):
            event = contracts.event(self.read(source, canonical=True), as_of=self.as_of)
            if event["event_id"] in ids:
                raise ContractError("CORPORATE_EVENT_DUPLICATE_ID")
            ids.add(event["event_id"])
            self.read(event["announcement_ref"], json_document=False)
            result.append((source, event))
        return result

    def _reviews(self):
        ref = self.context["anchor_reviews_ref"]
        if ref is None:
            return {}
        events = {(r["path"], r["sha256"]): e for r, e in self.events}
        return contracts.owner_reviews(
            self.read(ref, canonical=True),
            as_of=self.as_of,
            owner=self.policy["owner"],
            policy_ref=self.policy_ref,
            events=events,
        )

    def identity(self, record_id):
        if record_id not in self.identities:
            record = self.records[record_id]
            for stem in ("ledger", "manual_manifest"):
                self.read(
                    {
                        "path": ROOT + "/" + record[stem + "_path"],
                        "sha256": record[stem + "_sha256"],
                    },
                    json_document=False,
                )
            frame, _ = load_holdings_identity(self.workspace / ROOT, record)
            self.identities[record_id] = {
                r["symbol"]: r for r in normalize_positions(frame.to_dict("records"))
            }
        return self.identities[record_id]

    def _baseline(self):
        binding = self.policy["store_binding"]
        record_id = binding["active_record_id"]
        if record_id not in self.ancestry:
            return False
        record = self.records[record_id]
        if (
            binding["ledger_path"] != ROOT + "/" + record["ledger_path"]
            or binding["ledger_sha256"] != record["ledger_sha256"]
        ):
            return False
        self.identity(record_id)
        return True

    def _anchor(self, ref, event, financial, start):
        result = {
            "state": "OWNER_REVIEW_REQUIRED",
            "review_ref": None,
            "policy_ref": self.policy_ref,
            "source_record_id": None,
            "old_tracking_start_date": start,
            "tracking_start_date": None,
            "blocker_codes": ["OWNER_REVIEW_REQUIRED"],
        }
        review = self.reviews.get((ref["path"], ref["sha256"]))
        if review is None:
            return result
        result["review_ref"] = self.context["anchor_reviews_ref"]
        source = review["source_record_id"]
        if (
            not self.baseline_valid
            or source not in self.ancestry
            or source != financial["after_record_id"]
            or financial["state"] != "OBSERVED_NATIVE_POSTING"
        ):
            result["blocker_codes"] = ["OWNER_REVIEW_RECORD_UNCONFIRMED"]
            return result
        result.update(
            state="OWNER_DECLARED_RESEARCH_RESET",
            source_record_id=source,
            tracking_start_date=review["tracking_start_date"],
            blocker_codes=[],
        )
        return result

    def _companies(self):
        calendar_ref = self.evidence.calendar_ref
        calendar = self.read(calendar_ref) if calendar_ref else {}
        if calendar and (
            calendar.get("schema_version") != "cn-close-session-receipt.v1"
            or calendar.get("status") != "TARGET_AUTHORIZED"
        ):
            raise ContractError("CORPORATE_CALENDAR_CONTRACT_INVALID")
        dates = calendar.get("ordered_open_dates", [])
        if type(dates) is not list:
            raise ContractError("CORPORATE_CALENDAR_DATES_INVALID")
        self.read(self.evidence.market_snapshot_ref, json_document=False)
        reader = self.evidence._market_reader()
        baseline = self._baseline()
        self.baseline_valid = baseline
        anchors = {r["symbol"]: r for r in self.policy["anchors"]}
        result = []
        for position in self.portfolio["positions"]:
            if contracts.number(position["shares"]) <= 0:
                continue
            symbol = position["symbol"]
            start = anchors.get(symbol, {}).get("tracking_start_date")
            ref = self.evidence.market_refs.get(symbol)
            frame_rows = []
            if ref is not None and reader is not None:
                path = reader.resolve_symbol_path(symbol)
                if path != self.workspace / ref["path"]:
                    raise ContractError("CORPORATE_MARKET_FRAME_PATH_MISMATCH")
                self.read(ref, json_document=False)
                frame_rows = reader._read_strict_catalog_parquet(
                    path,
                    table_meta={"sha256": ref["sha256"]},
                    logical_table="corporate tracking window",
                ).to_dict("records")
            relevant = [
                (r, e)
                for r, e in self.events
                if e["symbol"] == symbol
                and (start is None or start <= e["effective_trade_date"])
                and e["effective_trade_date"] <= self.trade_date
            ]
            row = tracking_window(
                symbol=symbol,
                start=start,
                trade_date=self.trade_date,
                calendar_dates=dates,
                rows=frame_rows,
                market_ref=ref,
                events=[e for _, e in relevant],
                baseline_valid=baseline,
            )
            proved_pairs = set()
            for event_ref, event in relevant:
                financial = reconcile_posting(
                    root=self.workspace / ROOT,
                    catalog=self.catalog,
                    ancestry=self.ancestry,
                    event=event,
                    event_ref=event_ref,
                    read=self.read,
                )
                anchor = self._anchor(event_ref, event, financial, start)
                row["events"].append(
                    {
                        "event_ref": event_ref,
                        "event_id": event["event_id"],
                        "kind": event["kind"],
                        "effective_trade_date": event["effective_trade_date"],
                        "source_state": "SOURCE_DECLARED",
                        "financial": financial,
                        "anchor": anchor,
                        "reconciliation_state": "UNCONFIRMED",
                        "blocker_codes": [],
                    }
                )
                if financial["state"] == "OBSERVED_NATIVE_POSTING":
                    proved_pairs.add((financial["before_record_id"], financial["after_record_id"]))
                if event["effective_trade_date"] == self.trade_date:
                    row["blocker_codes"].append("CURRENT_EVENT_EMPTY_CONFLICT")
            self._lifecycle(row, symbol, proved_pairs, baseline)
            result.append(row)
        return result

    def _lifecycle(self, row, symbol, proved_pairs, baseline):
        if baseline:
            upto = self.ancestry.index(self.policy["store_binding"]["active_record_id"])
            for after in self.ancestry[:upto]:
                before = self.lineage[after]["source_record_id"]
                if (
                    self.identity(before).get(symbol) != self.identity(after).get(symbol)
                    and (before, after) not in proved_pairs
                ):
                    row["blocker_codes"].append("POSITION_LIFECYCLE_UNCONFIRMED")

    def report(self, custody_at):
        self.recheck()
        return build_reconciliation(
            as_of=self.as_of,
            trade_date=self.trade_date,
            context_ref=self.context_ref,
            decision_recipe_ref=self.decision_recipe_ref,
            store_plan_ref=self.plan_ref,
            portfolio_source_ref=self.portfolio_ref,
            tracking_policy_ref=self.policy_ref,
            source_refs=[{"path": p, "sha256": h} for p, h in self.seen],
            company_rows=self.company_rows,
            custody_at=custody_at,
        )
