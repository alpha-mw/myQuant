"""Exact frozen EOD sources for Morning risk; no producer or mutable-head reader."""

from pathlib import Path, PurePosixPath
from datetime import datetime, time
from zoneinfo import ZoneInfo
import re


from quant_investor.contracts import parse_canonical_json_bytes, validate_artifact
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.strategy_records import corporate_contracts as contracts
from quant_investor.strategy_records.event_receipts import (
    read_event_source,
    resolve_catalog_event_receipt,
)
from quant_investor.strategy_records.event_contracts import SYMBOLIC_RECEIPT
from quant_investor.strategy_records.holdings import load_holdings_identity
from quant_investor.strategy_records.store import load_catalog_snapshot_bytes
from quant_investor.strategy_records.risk_policy_contract import (
    validate_trailing_policy,
    validate_initial_stop_policy,
)
from quant_investor.intelligence.corporate_reconciliation import _required_sessions
from .corporate_actions import CorporateActionEvidence, EVENT_ROOT
from .corporate_adapter import SCHEMA, SCHEMA_V3, LEGACY_FIELDS, EXTRA_FIELDS
from .daily_contract import ContractError, validate_ref
from .daily_journal import _validate_day, request_identity
from .held_market_extract import load_held_market_rows

ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"
PROFILES = {
    "cn-daily-native-inputs.v4",
    "cn-daily-native-inputs.v5",
    "cn-daily-native-inputs.v6",
    "cn-daily-native-inputs.v7",
}


class MorningRiskSources:
    """Use only after the caller's existing full native EOD admission."""

    def __init__(self, *, workspace, recorded, completion_ref, policy_refs, quote_requested_at):
        self.workspace = Path(workspace).resolve(strict=True)
        self.recorded = recorded
        self.completion_ref = validate_ref(completion_ref)
        self.day = PurePosixPath(completion_ref["path"]).parent.name
        _validate_day(self.day)
        self.seen = {}
        self.identities = {}
        self.policy_refs = {k: validate_ref(v) for k, v in policy_refs.items()}
        if set(self.policy_refs) != {"trailing", "initial_stop"}:
            raise ContractError("MORNING_THRESHOLD_POLICY_REFS_INVALID")
        self.inputs = self.read(recorded["native_inputs_ref"])
        self.registered = self.inputs["schema_version"] == "cn-daily-native-inputs.v7"
        self.transition = None
        self.transition_rows = {}
        if self.inputs["schema_version"] not in PROFILES or self.inputs["trade_date"] != self.day:
            raise ContractError("MORNING_THRESHOLD_EOD_PROFILE_INVALID")
        self._corporate()
        self._store()
        self.policies = {
            name: self.read(ref, native=True) for name, ref in self.policy_refs.items()
        }
        validate_trailing_policy(self.policies["trailing"], as_of=quote_requested_at)
        validate_initial_stop_policy(self.policies["initial_stop"], as_of=quote_requested_at)
        if self.policies["trailing"]["owner"] != self.policies["initial_stop"]["owner"]:
            raise ContractError("MORNING_THRESHOLD_POLICY_OWNER_MISMATCH")
        allowed_anchors = {
            "EXACT_OWNER_CONFIRMED_BUY",
            "EXACT_RECONCILED_ADD",
            "EXACT_RECONCILED_BUY",
            "OWNER_APPROVED_RESET_NO_STRUCTURED_BUY_FOUND",
        }
        if any(
            row["anchor_state"] not in allowed_anchors
            or type(row["symbol"]) is not str
            or not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", row["symbol"])
            for row in self.policies["trailing"]["anchors"]
        ):
            raise ContractError("MORNING_THRESHOLD_ANCHOR_POLICY_UNSUPPORTED")
        self.baselines = {k: self._baseline(v) for k, v in self.policies.items()}
        self.calendar = self.read(self.inputs["calendar_ref"], native=True)
        self.dates = self.calendar["ordered_open_dates"]
        if (
            self.calendar["schema_version"] != "cn-close-session-receipt.v1"
            or self.calendar["status"] != "TARGET_AUTHORIZED"
            or type(self.dates) is not list
            or self.dates != sorted(set(self.dates))
            or self.day not in self.dates
        ):
            raise ContractError("MORNING_THRESHOLD_CALENDAR_INVALID")
        for day in self.dates:
            _validate_day(day)
        self.dates = [day for day in self.dates if day <= self.day]
        self.market = self.evidence._market_reader()
        self.bytes(self.inputs["market_snapshot_ref"])
        self.event = self.evidence._events()
        event_ref = self.event["pointer"]["generation"]
        self.bytes({"path": EVENT_ROOT + "/" + event_ref["path"], "sha256": event_ref["sha256"]})
        self._event_custody()
        self.named_events = self._named_events()
        self.rows = self._positions()
        self.recheck()

    def bytes(self, ref):
        ref = validate_ref(ref)
        if Path(ref["path"]).name in {"current.v1.json", "_latest.json", "_active.json"}:
            raise ContractError("MORNING_THRESHOLD_MUTABLE_HEAD_FORBIDDEN")
        raw = read_event_source(self.workspace, ref)
        key = (ref["path"], ref["sha256"])
        if key in self.seen and self.seen[key] != raw:
            raise ContractError("MORNING_THRESHOLD_SOURCE_CHANGED")
        self.seen[key] = raw
        return raw

    def read(self, ref, *, native=False):
        raw = self.bytes(ref)
        return (
            parse_json_bytes(raw, label="Morning native source", require_canonical=False)
            if native
            else parse_canonical_json_bytes(raw)
        )

    def recheck(self):
        for (path, sha), raw in self.seen.items():
            if read_event_source(self.workspace, {"path": path, "sha256": sha}) != raw:
                raise ContractError("MORNING_THRESHOLD_SOURCE_CHANGED")

    def source_refs(self):
        return [{"path": path, "sha256": sha} for path, sha in sorted(self.seen)]

    def _corporate(self):
        terminal_ref = self.recorded["node_terminal_refs"]["corporate_action_recon"]
        terminal = self.read(terminal_ref)
        request_path = str(PurePosixPath(terminal_ref["path"]).parent.parent / "request.json")
        from quant_investor.system.storage import SecureSystemStorage

        stored = SecureSystemStorage(self.workspace).read_workspace_file_bytes(
            request_path, maximum_bytes=8 * 1024 * 1024
        )
        request = self.read({"path": request_path, "sha256": stored.byte_sha256})
        if request_identity(request)[1] != terminal["request_key"]:
            raise ContractError("MORNING_THRESHOLD_CORPORATE_REQUEST_INVALID")
        self.recipe = self.read(request["input_refs"]["recipe"])
        if set(self.recipe) != LEGACY_FIELDS | EXTRA_FIELDS | {"schema_version", "custody_at"} | (
            {"registered_event_declaration_ref"} if self.registered else set()
        ) or self.recipe["schema_version"] != (SCHEMA_V3 if self.registered else SCHEMA):
            raise ContractError("MORNING_THRESHOLD_CORPORATE_PROFILE_INVALID")
        for source, target in (
            ("previous_trade_date", "previous_trade_date"),
            ("calendar_ref", "calendar_ref"),
            ("market_snapshot_ref", "market_snapshot_ref"),
            ("market_refs", "adjustment_market_refs"),
            ("corporate_action_context_ref", "corporate_action_context_ref"),
            ("decision_recipe_ref", "decision_recipe_ref"),
            ("research_request_ref", "research_request_ref"),
            ("store_plan_ref", "store_plan_ref"),
        ):
            if self.recipe[source] != self.inputs[target]:
                raise ContractError("MORNING_THRESHOLD_MIXED_EOD_PROFILE")
        if (
            self.registered
            and self.recipe["registered_event_declaration_ref"]
            != self.inputs["registered_event_declaration_ref"]
        ):
            raise ContractError("MORNING_REGISTERED_DECLARATION_MISMATCH")
        self.context = self.read(self.inputs["corporate_action_context_ref"])
        self.report = validate_artifact(
            self.bytes(terminal["output_refs"]["reconciliation"]),
            expected_kind="corporate_action_reconciliation",
        )
        if self.report["payload"]["tracking_policy_ref"] != self.context["tracking_policy_ref"]:
            raise ContractError("MORNING_THRESHOLD_CORPORATE_POLICY_MISMATCH")
        plan = self.read(self.inputs["store_plan_ref"], native=True)
        self.plan = plan
        if self.recipe["event_pointer_ref"]["sha256"] != plan["preimages"]["event_pointer_sha256"]:
            raise ContractError("MORNING_THRESHOLD_EVENT_BINDING_INVALID")
        self.bytes(self.recipe["event_pointer_ref"])
        self.evidence = CorporateActionEvidence(
            workspace=str(self.workspace), trade_date=self.day, recipe=self.recipe
        )
        if self.registered:
            from .corporate_adapter import derive_corporate_projection

            projection, report, report_ref, transition, transition_ref = (
                derive_corporate_projection(evidence=self.evidence, recipe=self.recipe)
            )
            outputs = terminal["output_refs"]
            if (
                set(outputs)
                != {
                    "financial_events",
                    "event_generation",
                    "reconciliation",
                    "registered_transition",
                }
                or outputs["reconciliation"] != report_ref
                or report != self.report
                or outputs["registered_transition"] != transition_ref
                or self.read(transition_ref) != transition
                or self.read(outputs["financial_events"]) != projection
                or outputs["event_generation"] != projection["event_generation_ref"]
            ):
                raise ContractError("MORNING_REGISTERED_RECONCILIATION_MISMATCH")
            self.transition = transition["payload"]
            self.transition_rows = {r["symbol"]: r for r in self.transition["position_rows"]}
            for reference in self.transition["source_refs"]:
                self.bytes(reference)

    def _store(self):
        terminal = self.read(self.recorded["node_terminal_refs"]["store"])
        self.store_outputs = terminal["output_refs"]
        pointer_ref = self.store_outputs["pointer"]
        self.pointer, self.catalog = load_catalog_snapshot_bytes(
            self.workspace / ROOT,
            pointer_bytes=self.bytes(pointer_ref),
            expected_pointer_sha256=pointer_ref["sha256"],
        )
        self.bytes(
            {
                "path": ROOT + "/" + self.pointer["catalog_path"],
                "sha256": self.pointer["catalog_sha256"],
            }
        )
        self.records = {row["record_id"]: row for row in self.catalog["records"]}
        self.lineage = {row["record_id"]: row for row in self.catalog["lineage_index"]}
        self.ancestry = []
        cursor = self.pointer["active_record_id"]
        while cursor is not None:
            if cursor in self.ancestry or cursor not in self.lineage:
                raise ContractError("MORNING_THRESHOLD_ANCESTRY_INVALID")
            self.ancestry.append(cursor)
            cursor = self.lineage[cursor]["source_record_id"]
        active = self.records[self.pointer["active_record_id"]]
        if self.store_outputs["ledger"] != {
            "path": ROOT + "/" + active["ledger_path"],
            "sha256": active["ledger_sha256"],
        }:
            raise ContractError("MORNING_THRESHOLD_LEDGER_BINDING_INVALID")
        if self.lineage[active["record_id"]]["valuation_date"].replace("-", "") != self.day:
            raise ContractError("MORNING_THRESHOLD_HOLDINGS_DATE_INVALID")
        self.current = self._identity(active["record_id"])
        if self.registered:
            transition = self.transition
            if (
                self.pointer["active_record_id"] != self.plan["record_ids"][-1]
                or self.pointer["previous_pointer_sha256"]
                != transition["writer_pointer_ref"]["sha256"]
                or self.lineage[active["record_id"]]["source_record_id"]
                != transition["writer_record_id"]
                or set(self.current) != set(self.transition_rows)
            ):
                raise ContractError("MORNING_REGISTERED_FINAL_STORE_MISMATCH")
            for symbol, position in self.current.items():
                row = self.transition_rows[symbol]
                if row["writer_position_state"] != "PRESENT" or any(
                    position[field] != contracts.number(row[field + "_after"])
                    for field in ("shares", "avg_cost", "cost_basis")
                ):
                    raise ContractError("MORNING_REGISTERED_FINAL_POSITION_MISMATCH")

    def _identity(self, record_id):
        if record_id not in self.identities:
            row = self.records[record_id]
            for path, sha in (
                ("ledger_path", "ledger_sha256"),
                ("manual_manifest_path", "manual_manifest_sha256"),
            ):
                self.bytes({"path": ROOT + "/" + row[path], "sha256": row[sha]})
            frame, _ = load_holdings_identity(self.workspace / ROOT, row)
            if frame["symbol"].duplicated().any():
                raise ContractError("MORNING_THRESHOLD_HOLDING_SYMBOL_DUPLICATED")
            self.identities[record_id] = {r["symbol"]: r for r in frame.to_dict("records")}
        return self.identities[record_id]

    def _manual(self, record_id):
        row = self.records[record_id]
        return self.read(
            {
                "path": ROOT + "/" + row["manual_manifest_path"],
                "sha256": row["manual_manifest_sha256"],
            },
            native=True,
        )

    def _baseline(self, policy):
        binding = policy["store_binding"]
        record_id = binding["active_record_id"]
        if (
            record_id not in self.ancestry
            or binding["pointer_path"] != ROOT + "/_record_store/current.v1.json"
        ):
            raise ContractError("MORNING_THRESHOLD_POLICY_BASELINE_NOT_ANCESTOR")
        row = self.records[record_id]
        if (
            binding["ledger_path"] != ROOT + "/" + row["ledger_path"]
            or binding["ledger_sha256"] != row["ledger_sha256"]
        ):
            raise ContractError("MORNING_THRESHOLD_POLICY_BASELINE_LEDGER_MISMATCH")
        self._identity(record_id)
        return record_id

    def _event_custody(self):
        start = min(
            self.lineage[v]["valuation_date"].replace("-", "") for v in self.baselines.values()
        )
        self.event_dates = set()
        for row in self.event["closures"]:
            day = row["trade_date"].replace("-", "")
            if not start < day <= self.day:
                continue
            self.event_dates.add(day)
            for key in ("policy_ref", "owner_declaration_ref", "source_receipt_ref"):
                ref = row[key]
                if ref is not None:
                    if SYMBOLIC_RECEIPT.fullmatch(ref["path"]):
                        ref = resolve_catalog_event_receipt(workspace=self.workspace, closure=row)[
                            "catalog_ref"
                        ]
                    self.bytes(ref)

    def _named_events(self):
        reference = self.context["named_events_ref"]
        if reference is None:
            return []
        values = contracts.event_list(self.read(reference), as_of=self.context["as_of"])
        return [contracts.event(self.read(ref), as_of=self.context["as_of"]) for ref in values]

    def _lifecycle(self, symbol, lane):
        base = self.baselines[lane]
        baseline = self._identity(base)
        base_date = self.lineage[base]["valuation_date"].replace("-", "")
        blockers = [
            "LIFECYCLE_UNCONFIRMED:" + day
            for day in self.dates
            if base_date < day <= self.day
            and day not in self.event_dates
            and not (self.registered and day == self.day)
        ]
        for record_id in self.ancestry[: self.ancestry.index(base)]:
            rows = self._identity(record_id)
            if symbol not in baseline or symbol not in rows:
                blockers.append("POSITION_LIFECYCLE_CHANGED")
            elif any(
                rows[symbol][field] != baseline[symbol][field]
                for field in ("shares", "avg_cost", "cost_basis")
            ):
                blockers.append("POSITION_COST_OR_QUANTITY_CHANGED")
            manual = self._manual(record_id)
            if any(
                row.get("symbol") == symbol
                for key in (
                    "applied_owner_declared_trades",
                    "applied_local_trades",
                    "corporate_actions",
                )
                for row in manual.get(key, [])
            ):
                blockers.append("NEW_EVENT_REQUIRES_ANCHOR_REVIEW")
        return blockers

    def _frame(self, symbol):
        ref = self.inputs["adjustment_market_refs"].get(symbol)
        if ref is None or self.market is None:
            return [], ["STRICT_CLOSE_UNAVAILABLE"]
        try:
            rows = load_held_market_rows(self.market, symbol, self.bytes(ref)).to_dict("records")
        except ContractError as exc:
            if str(exc).endswith("HELD_MARKET_EXTRACT_SYMBOL_MISMATCH"):
                raise ContractError("MORNING_THRESHOLD_MARKET_SYMBOL_MISMATCH") from exc
            raise ContractError("MORNING_THRESHOLD_MARKET_FRAME_PATH_MISMATCH") from exc
        for row in rows:
            if row.get("ts_code") != symbol:
                raise ContractError("MORNING_THRESHOLD_MARKET_SYMBOL_MISMATCH")
            row["trade_date"] = str(row["trade_date"]).replace("-", "")
        return rows, []

    def _entry(self, symbol, anchor):
        if not anchor:
            return []
        if "anchor_ref" not in anchor:
            return (
                ["ENTRY_REFERENCE_UNCONFIRMED"]
                if anchor["anchor_state"].startswith("EXACT_")
                else []
            )
        ref = anchor["anchor_ref"]
        fields = {
            "path",
            "sha256",
            "symbol",
            "shares",
            "execution_price_cny",
            "final_total_fee_cny",
        }
        if type(ref) is not dict or set(ref) != fields or ref["symbol"] != symbol:
            raise ContractError("MORNING_THRESHOLD_ANCHOR_REFERENCE_INVALID")
        validate_ref({"path": ref["path"], "sha256": ref["sha256"]})
        if (
            contracts.number(ref["shares"]) <= 0
            or contracts.number(ref["execution_price_cny"]) <= 0
            or contracts.number(ref["final_total_fee_cny"]) < 0
        ):
            raise ContractError("MORNING_THRESHOLD_ANCHOR_REFERENCE_INVALID")
        matches = [
            r
            for r in self.ancestry
            if ROOT + "/" + self.records[r]["manual_manifest_path"] == ref["path"]
            and self.records[r]["manual_manifest_sha256"] == ref["sha256"]
        ]
        if len(matches) != 1:
            return ["ENTRY_REFERENCE_UNCONFIRMED"]
        manual = self._manual(matches[0])
        for key in (
            "reconciled_source_trades",
            "applied_owner_declared_trades",
            "applied_local_trades",
        ):
            for row in manual.get(key, []):
                if row.get("symbol") != symbol:
                    continue
                if _entry_matches(row, ref, manual, anchor["tracking_start_date"]):
                    return []
        return ["EXACT_ENTRY_REF_MISMATCH"]

    def _positions(self):
        anchors = {r["symbol"]: r for r in self.policies["trailing"]["anchors"]}
        stops = {r["symbol"]: r for r in self.policies["initial_stop"]["stops"]}
        corporate = {r["symbol"]: r for r in self.report["payload"]["company_rows"]}
        stop_start = (
            contracts.instant(self.policies["initial_stop"]["effective_from"])
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .strftime("%Y%m%d")
        )
        result = []
        for symbol, position in sorted(self.current.items()):
            if position["shares"] <= 0:
                continue
            rows, missing = self._frame(symbol)
            anchor, stop = anchors.get(symbol), stops.get(symbol)
            trailing = self._lifecycle(symbol, "trailing") + missing + self._entry(symbol, anchor)
            stop_blockers = self._lifecycle(symbol, "initial_stop") + missing
            same_policy = self.policy_refs["trailing"] == self.context["tracking_policy_ref"]
            changed = self.transition_rows.get(symbol, {}).get(
                "policy_revalidation_required", False
            )
            if changed:
                trailing.extend(self._registered_policy_blockers("trailing", anchor))
                stop_blockers.extend(self._registered_policy_blockers("initial_stop"))
            if not same_policy:
                trailing.append("TRAILING_POLICY_NOT_RECONCILED_AT_PRIOR_EOD")
            elif changed:
                # This symbol's post-trade lifecycle is proved by the separate
                # registered report and exact post-trade policy, not a T-1 row.
                pass
            elif symbol not in corporate:
                trailing.append("POSITION_LIFECYCLE_UNCONFIRMED")
            else:
                trailing.extend(corporate[symbol]["blocker_codes"])
            if (
                anchor
                and _required_sessions(self.dates, anchor["tracking_start_date"], self.day) is None
            ):
                trailing.append("CALENDAR_WINDOW_INCOMPLETE")
            stop_price, stop_blockers = self._stop_window(
                symbol, position, stop, rows, stop_start, stop_blockers
            )
            result.append(
                {
                    "position": {
                        **position,
                        "name": (anchor or stop or {}).get("company_name", ""),
                    },
                    "anchor": anchor,
                    "stop": stop,
                    "closes": rows,
                    "expected_dates": self.dates,
                    "trailing_blockers": sorted(set(trailing)),
                    "owner_stop_blockers": sorted(set(stop_blockers)),
                    "owner_stop": stop_price,
                    "same_trailing_policy": same_policy,
                }
            )
        return result

    def _registered_policy_blockers(self, lane, anchor=None):
        binding = self.policies[lane]["store_binding"]
        expected = {
            self.transition["writer_record_id"]: self.transition["writer_pointer_ref"]["sha256"],
            self.pointer["active_record_id"]: self.store_outputs["pointer"]["sha256"],
        }
        if expected.get(binding["active_record_id"]) != binding["pointer_sha256"]:
            return ["OWNER_POLICY_REVALIDATION_REQUIRED"]
        declaration = self.read(self.transition["registered_event_declaration_ref"], native=True)
        writer = self.read(declaration["writer_store_pointer_ref"], native=True)
        field = "effective_from" if lane == "trailing" else "owner_confirmation_recorded_at"
        if contracts.instant(self.policies[lane][field]) < contracts.instant(
            writer["published_at"]
        ):
            return ["OWNER_POLICY_REVALIDATION_REQUIRED"]
        if lane == "trailing" and (anchor is None or anchor["tracking_start_date"] < self.day):
            return ["OWNER_POLICY_REVALIDATION_REQUIRED"]
        return []

    def _stop_availability(self):
        policy = self.policies["initial_stop"]
        available = max(
            contracts.instant(policy[k])
            for k in ("effective_from", "owner_confirmation_recorded_at")
        )
        local = available.astimezone(ZoneInfo("Asia/Shanghai"))
        day = local.strftime("%Y%m%d")
        if day > self.day:
            return "OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD"
        if day < self.day:
            return None
        # Same-day activation needs the original native close-clock evidence.
        from quant_investor.market.close_session_authority import SESSION_CLOSE_LOCAL

        close = self.calendar.get("session_close_local")
        if (
            self.calendar.get("timezone") != "Asia/Shanghai"
            or close != SESSION_CLOSE_LOCAL.isoformat()
        ):
            return "OWNER_STOP_EVIDENCE_UNCONFIRMED"
        moment = datetime.combine(
            datetime.strptime(self.day, "%Y%m%d").date(),
            time.fromisoformat(close),
            tzinfo=ZoneInfo("Asia/Shanghai"),
        )
        return "OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD" if local > moment else None

    def _stop_window(self, symbol, position, stop, rows, stop_start, stop_blockers):
        stop_price = None
        if stop is not None:
            stop_price = stop["initial_stop_price_cny"]
            unavailable = self._stop_availability()
            if unavailable:
                stop_blockers.append(unavailable)
                stop_price = None
            else:
                required = _required_sessions(self.dates, stop_start, self.day)
                selected = [r for r in rows if stop_start <= r["trade_date"] <= self.day]
                if required is None or sorted(r["trade_date"] for r in selected) != required:
                    stop_blockers.append("OWNER_STOP_HISTORY_GAP")
                try:
                    factors = {contracts.number(r.get("adj_factor")) for r in selected}
                    if len(factors) != 1 or any(f <= 0 for f in factors):
                        stop_blockers.append("OWNER_STOP_CORPORATE_ACTION_REVIEW")
                except ContractError:
                    stop_blockers.append("OWNER_STOP_CORPORATE_ACTION_REVIEW")
            if any(
                position[field] != contracts.number(stop[name])
                for field, name in (
                    ("shares", "current_shares"),
                    ("avg_cost", "fee_inclusive_avg_cost_cny"),
                )
            ):
                stop_blockers.append("OWNER_STOP_POSITION_CHANGED")
            if any(
                e["symbol"] == symbol and stop_start <= e["effective_trade_date"] <= self.day
                for e in self.named_events
            ):
                stop_blockers.append("OWNER_STOP_CORPORATE_ACTION_REVIEW")
        return stop_price, stop_blockers


def _entry_matches(row, ref, manual, start):
    try:
        return (
            contracts.number(row.get("shares")) == contracts.number(ref["shares"])
            and contracts.number(row.get("execution_price"))
            == contracts.number(ref["execution_price_cny"])
            and contracts.number(row.get("final_total_fee_cny", row.get("fees_cny")))
            == contracts.number(ref["final_total_fee_cny"])
            and str(row.get("trade_date", manual.get("trade_date", ""))).replace("-", "") == start
        )
    except ContractError:
        return False
