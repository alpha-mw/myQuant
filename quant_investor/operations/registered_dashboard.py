"""Frozen Corporate/Store cross-proof for the registered Dashboard profile."""

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.strategy_records.event_receipts import read_event_source
from quant_investor.strategy_records import registered_event_contracts as registered
from quant_investor.strategy_records.close_plan_contracts import validate_plan
from quant_investor.strategy_records.holdings import load_holdings_identity
from quant_investor.strategy_records.store import load_catalog_snapshot_bytes
from .corporate_actions import CorporateActionEvidence
from .corporate_adapter import SCHEMA_V3, derive_corporate_projection
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal, request_identity, _false_authority


class RegisteredDashboardSources:
    def __init__(
        self,
        *,
        workspace,
        trade_date,
        release_ref,
        corporate_terminal_ref,
        store_terminal_ref,
        store_plan_ref,
        registered_event_declaration_ref,
    ):
        self.workspace = Path(workspace).resolve(strict=True)
        self.day = trade_date
        self.release = validate_ref(release_ref)
        self.plan_ref = validate_ref(store_plan_ref)
        self.declaration_ref = validate_ref(registered_event_declaration_ref)
        self.journal = DailyJournal(str(self.workspace), trade_date)
        self.seen, self.finished = {}, []
        self.raw(self.release)
        corporate, request = self.terminal("corporate_action_recon", corporate_terminal_ref)
        store, store_request = self.terminal("store", store_terminal_ref)
        recipe = self.read(request["input_refs"]["recipe"])
        if (
            recipe.get("schema_version") != SCHEMA_V3
            or recipe.get("store_plan_ref") != self.plan_ref
            or recipe.get("registered_event_declaration_ref") != self.declaration_ref
            or store_request["input_refs"].get("native_plan") != self.plan_ref
        ):
            raise ContractError("DASHBOARD_REGISTERED_RECIPE_BINDING_INVALID")
        evidence = CorporateActionEvidence(
            workspace=str(self.workspace), trade_date=trade_date, recipe=recipe
        )
        projection, report, report_ref, transition, transition_ref = derive_corporate_projection(
            evidence=evidence, recipe=recipe
        )
        outputs = corporate["output_refs"]
        if (
            set(outputs)
            != {"financial_events", "event_generation", "reconciliation", "registered_transition"}
            or outputs["reconciliation"] != report_ref
            or self.read(report_ref) != report
            or outputs["registered_transition"] != transition_ref
            or self.read(transition_ref) != transition
            or outputs["event_generation"] != projection["event_generation_ref"]
            or self.read(outputs["financial_events"]) != projection
        ):
            raise ContractError("DASHBOARD_REGISTERED_CORPORATE_REPLAY_MISMATCH")
        for reference in transition["payload"]["source_refs"]:
            self.raw(reference)
        self.transition_ref, self.transition = transition_ref, transition
        self.summary = self.close_summary(store)
        self.recheck()

    def raw(self, ref):
        reference = validate_ref(ref)
        if PurePosixPath(reference["path"]).name in {
            "current.v1.json",
            "_latest.json",
            "_active.json",
        }:
            raise ContractError("DASHBOARD_REGISTERED_MUTABLE_SOURCE_FORBIDDEN")
        raw = read_event_source(self.workspace, reference)
        key = reference["path"], reference["sha256"]
        if self.seen.setdefault(key, raw) != raw:
            raise ContractError("DASHBOARD_REGISTERED_SOURCE_CHANGED")
        return raw

    def read(self, ref, *, native=False):
        raw = self.raw(ref)
        return (
            parse_json_bytes(
                raw, label="registered Dashboard native source", require_canonical=False
            )
            if native
            else parse_canonical_json_bytes(raw)
        )

    def terminal(self, node, ref, *, source_time=True):
        ref = validate_ref(ref)
        terminal = self.read(ref)
        path = str(PurePosixPath(ref["path"]).parent.parent / "request.json")
        from quant_investor.system.storage import SecureSystemStorage

        raw = SecureSystemStorage(self.workspace).read_workspace_file_bytes(
            path, maximum_bytes=8 * 1024 * 1024
        )
        request = self.read({"path": path, "sha256": raw.byte_sha256})
        if (
            request_identity(request)[1] != terminal["request_key"]
            or request["node_id"] != node
            or request["trade_date"] != self.day
            or request["release_ref"] != self.release
            or terminal["state"] != "SUCCEEDED"
            or not _false_authority(terminal["authority"])
        ):
            raise ContractError("DASHBOARD_REGISTERED_TERMINAL_INVALID")
        selected = self.journal.readonly_inspect(request)
        if selected.get("terminal_ref") != ref or selected.get("terminal") != terminal:
            raise ContractError("DASHBOARD_REGISTERED_TERMINAL_SELECTION_MISMATCH")
        for reference in terminal["output_refs"].values():
            self.raw(reference)
        if source_time:
            self.finished.append(terminal["finished_at"])
        return terminal, request

    def close_summary(self, terminal):
        from scripts import cn_official_close_batch as native
        from scripts.daily_production_store_adapter import store_close_output_refs

        root = self.workspace / registered.RECORD_ROOT
        plan = self.read(self.plan_ref, native=True)
        if validate_plan(plan, path=self.plan_ref["path"]) != 2:
            raise ContractError("DASHBOARD_REGISTERED_PLAN_V2_REQUIRED")
        body = self.transition["payload"]
        if (
            plan["registered_event_declaration_ref"] != self.declaration_ref
            or body["registered_event_declaration_ref"] != self.declaration_ref
            or any(
                plan[k] != body[k]
                for k in (
                    "decision_baseline_pointer_ref",
                    "decision_baseline_catalog_ref",
                    "decision_baseline_record_id",
                    "source_profile",
                )
            )
            or plan["source_active_record_id"] != body["writer_record_id"]
            or plan["preimages"]["store_pointer_sha256"] != body["writer_pointer_ref"]["sha256"]
            or plan["requested_target"].replace("-", "") != self.day
        ):
            raise ContractError("DASHBOARD_REGISTERED_PLAN_SOURCE_MISMATCH")
        proof = native.inspect_frozen_close_commit(
            record_root=root,
            transaction_id=plan["transaction_id"],
            expected_plan_sha=self.plan_ref["sha256"],
            expected_source_pointer_sha=plan["preimages"]["store_pointer_sha256"],
            expected_target=plan["requested_target"],
            plan_version=2,
        )
        outputs = store_close_output_refs(
            root=root, workspace=self.workspace, trade_date=self.day, plan=plan, proof=proof
        )
        if outputs != terminal["output_refs"]:
            raise ContractError("DASHBOARD_REGISTERED_STORE_OUTPUT_MISMATCH")
        manual = self.read(outputs["manual"], native=True)
        pointer, catalog = load_catalog_snapshot_bytes(
            root,
            pointer_bytes=self.raw(outputs["pointer"]),
            expected_pointer_sha256=outputs["pointer"]["sha256"],
        )
        final_id = pointer["active_record_id"]
        if (
            final_id != plan["record_ids"][-1]
            or manual["source_record"] != body["writer_record_id"]
            or pointer["previous_pointer_sha256"] != body["writer_pointer_ref"]["sha256"]
            or manual.get("official_valuation") is not True
            or any(
                type(manual.get(k)) is not int or manual[k] != 0
                for k in ("trade_count", "order_count", "fill_count")
            )
            or registered.number(manual["cash_after"]) != registered.number(body["cash_after_cny"])
        ):
            raise ContractError("DASHBOARD_REGISTERED_FINAL_CLOSE_MISMATCH")
        row = next(r for r in catalog["records"] if r["record_id"] == final_id)
        frame, _ = load_holdings_identity(root, row)
        final_positions = registered.positions({"positions": frame.to_dict("records")})
        expected = {
            r["symbol"]: tuple(
                registered.number(r[k + "_after"]) for k in ("shares", "avg_cost", "cost_basis")
            )
            for r in body["position_rows"]
        }
        if final_positions != expected:
            raise ContractError("DASHBOARD_REGISTERED_FINAL_HOLDINGS_MISMATCH")
        return {
            "source_profile": plan["source_profile"],
            "store_plan_ref": self.plan_ref,
            "decision_baseline_pointer_ref": body["decision_baseline_pointer_ref"],
            "writer_pointer_ref": body["writer_pointer_ref"],
            "final_pointer_ref": outputs["pointer"],
            "writer_record_id": body["writer_record_id"],
            "final_record_id": final_id,
            "official_valuation": True,
            "close_writer_trade_count": 0,
            "close_writer_order_count": 0,
            "close_writer_fill_count": 0,
        }

    def recheck(self):
        for (path, sha), raw in self.seen.items():
            if read_event_source(self.workspace, {"path": path, "sha256": sha}) != raw:
                raise ContractError("DASHBOARD_REGISTERED_SOURCE_CHANGED")

    def result(self, created_at):
        stamp = utc_stamp(created_at)
        if stamp > datetime.now(timezone.utc) or any(utc_stamp(t) > stamp for t in self.finished):
            raise ContractError("DASHBOARD_REGISTERED_CUSTODY_INVALID")
        self.recheck()
        return {
            "registered_transition_ref": self.transition_ref,
            "registered_transition": self.transition,
            "registered_close_summary": self.summary,
        }


def recorded_registered_view(*, workspace, completion, inputs):
    """Rebuild registered serving fields without minting whole-EOD authority."""
    from .dashboard_evidence import DOMAINS
    from .dashboard_serving_contract import POLICY

    if inputs["schema_version"] != "cn-daily-native-inputs.v7":
        raise ContractError("DASHBOARD_REGISTERED_NATIVE7_REQUIRED")
    terminals = completion["node_terminal_refs"]
    source = RegisteredDashboardSources(
        workspace=workspace,
        trade_date=inputs["trade_date"],
        release_ref=completion["release_ref"],
        corporate_terminal_ref=terminals["corporate_action_recon"],
        store_terminal_ref=terminals["store"],
        store_plan_ref=inputs["store_plan_ref"],
        registered_event_declaration_ref=inputs["registered_event_declaration_ref"],
    )
    terminal, request = source.terminal("dashboard", terminals["dashboard"], source_time=False)
    recipe_ref = request["input_refs"]["daily_evidence_recipe"]
    recipe = source.read(recipe_ref)
    identity = {
        "schema_version": "cn-daily-dashboard-evidence-recipe.v2",
        "trade_date": inputs["trade_date"],
        "release_ref": completion["release_ref"],
        "terminal_refs": {k: terminals[k] for k in DOMAINS},
        "publication_policy": POLICY,
        "corporate_terminal_ref": terminals["corporate_action_recon"],
        "store_plan_ref": inputs["store_plan_ref"],
        "registered_event_declaration_ref": inputs["registered_event_declaration_ref"],
    }
    key = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    if (
        set(recipe) != {*identity, "created_at"}
        or any(recipe[k] != v for k, v in identity.items())
        or recipe_ref["path"]
        != str(source.journal.root / "inputs" / f"dashboard-evidence-{key}.json")
        or request["input_refs"].get("registered.corporate_terminal")
        != terminals["corporate_action_recon"]
        or utc_stamp(recipe["created_at"]) > utc_stamp(terminal["finished_at"])
        or terminal["output_refs"].get("registered_transition") != source.transition_ref
    ):
        raise ContractError("DASHBOARD_REGISTERED_RECORDED_BINDING_INVALID")
    return source.result(recipe["created_at"])
