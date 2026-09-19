"""Bind a native registered transition to its separate T-1 Decision report."""

from quant_investor.strategy_records.close_plan_contracts import validate_plan
from quant_investor.strategy_records.registered_event_contracts import RECORD_ROOT, positions
from quant_investor.intelligence.registered_transition import build_registered_transition
from .daily_contract import ContractError, validate_ref


class RegisteredTransitionSources:
    def __init__(self, baseline, *, declaration_ref):
        # Fixed native composition, already admitted by the installed bridge.
        from scripts.cn_official_close_batch import _registered_plan_proof

        self.baseline = baseline
        self.declaration_ref = validate_ref(declaration_ref)
        plan = baseline.read(baseline.plan_ref)
        if (
            validate_plan(plan, path=baseline.plan_ref["path"]) != 2
            or plan["registered_event_declaration_ref"] != self.declaration_ref
        ):
            raise ContractError("REGISTERED_TRANSITION_PLAN_BINDING_INVALID")
        self.proof = _registered_plan_proof(baseline.workspace / RECORD_ROOT, plan, retained=True)
        declaration = self.proof["declaration"]
        portfolio = baseline.portfolio
        if (
            declaration["trade_date"].replace("-", "") != baseline.trade_date
            or portfolio["frozen_pointer_ref"]["sha256"]
            != declaration["baseline_store_pointer_ref"]["sha256"]
            or portfolio["source_record_id"] != declaration["baseline_record_id"]
            or positions(portfolio) != positions(self.proof["baseline"]["record"])
        ):
            raise ContractError("REGISTERED_TRANSITION_DECISION_BASELINE_MISMATCH")
        events = baseline.evidence._events()
        if any(r["trade_date"] == declaration["trade_date"] for r in events["closures"]):
            raise ContractError("REGISTERED_TRANSITION_EMPTY_EVENT_CONFLICT")
        # The BUY profile declares no corporate posting. Include new positions
        # here even though the old baseline report correctly has no row for them.
        if any(e["effective_trade_date"] == baseline.trade_date for _, e in baseline.events):
            raise ContractError("REGISTERED_TRANSITION_NAMED_CORPORATE_CONFLICT")
        for ref in self.proof["source_refs"]:
            baseline.read(ref, json_document=False)
        baseline.recheck()

    def report(self, custody_at):
        self.baseline.recheck()
        return build_registered_transition(
            proof=self.proof,
            as_of=self.baseline.as_of,
            custody_at=custody_at,
            source_refs=[{"path": p, "sha256": s} for p, s in self.baseline.seen],
        )
