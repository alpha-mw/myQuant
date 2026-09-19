"""Fixed native DAG composition, constructed only after upstream nodes complete.

No serialized callable/adapter selector is accepted. The native Factor loop must
produce its governed source closures before this downstream coordinator is entered.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from quant_investor.operations.core_pool import CoreContext, CORE_NODES, core_registry
from quant_investor.operations.corporate_actions import CorporateActionAdapter
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS, validate_ref
from quant_investor.operations.daily_runner import DayRunner
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.research_sources import (
    ResearchSources,
    ResearchSourceAdapter,
    SOURCE_NODES,
)
from quant_investor.operations.research_decision import ResearchDecisionAdapter
from scripts.daily_production_store_adapter import StoreCloseAdapter
from scripts.daily_dashboard_adapter import HistoricalDashboardAdapter, CurrentDashboardAdapter


@dataclass(frozen=True)
class NativeDailyInputs:
    factor_pointer_sha256: str
    release_ref: Mapping[str, str]
    research_request_ref: Mapping[str, str]
    store_arguments: Mapping
    store_plan_ref: Mapping[str, str]
    event_pointer_sha256: str
    market_snapshot_ref: Mapping[str, str]
    benchmark_ref: Mapping[str, str]
    risk_free_ref: Mapping[str, str]
    calendar_ref: Mapping[str, str]
    previous_trade_date: str | None = None
    adjustment_market_refs: Mapping[str, dict] | None = None
    publish_current_dashboard: bool = False
    next_session_calendar_proof_ref: Mapping[str, str] | None = None
    next_session_calendar_failure_ref: Mapping[str, str] | None = None
    decision_recipe_ref: Mapping[str, str] | None = None
    corporate_action_context_ref: Mapping[str, str] | None = None
    dashboard_publication_policy: str | None = None
    cutoff_ref: Mapping[str, str] | None = None
    registered_event_declaration_ref: Mapping[str, str] | None = None


class NativeDailyRegistry:
    node_ids = frozenset(
        (*CORE_NODES, *SOURCE_NODES, "decision", "corporate_action_recon", "store", "dashboard")
    )

    def __init__(
        self,
        workspace: str,
        trade_date: str,
        inputs: NativeDailyInputs,
        *,
        journal: DailyJournal | None = None,
    ):
        if self.node_ids != EOD_NODE_IDS:
            raise ContractError("NATIVE_REGISTRY_GRAPH_MISMATCH")
        if type(inputs.publish_current_dashboard) is not bool:
            raise ContractError("NATIVE_REGISTRY_PUBLICATION_MODE_INVALID")
        if inputs.dashboard_publication_policy is not None:
            from quant_investor.operations.dashboard_serving_contract import POLICY

            if (
                inputs.dashboard_publication_policy != POLICY
                or inputs.corporate_action_context_ref is None
                or inputs.decision_recipe_ref is None
            ):
                raise ContractError("NATIVE_REGISTRY_DASHBOARD_PROFILE_INVALID")
        if inputs.cutoff_ref is not None:
            validate_ref(inputs.cutoff_ref)
            if inputs.dashboard_publication_policy is None:
                raise ContractError("NATIVE_REGISTRY_CUTOFF_PROFILE_INVALID")
        if inputs.registered_event_declaration_ref is not None:
            validate_ref(inputs.registered_event_declaration_ref)
            if (
                inputs.cutoff_ref is None
                or inputs.store_arguments.get("registered_event_declaration_ref")
                != inputs.registered_event_declaration_ref
            ):
                raise ContractError("NATIVE_REGISTRY_REGISTERED_PROFILE_INVALID")
        for ref in (
            inputs.release_ref,
            inputs.research_request_ref,
            inputs.store_plan_ref,
            inputs.market_snapshot_ref,
            inputs.benchmark_ref,
            inputs.risk_free_ref,
            inputs.calendar_ref,
        ):
            validate_ref(ref)
        self.workspace, self.trade_date, self.inputs = workspace, trade_date, inputs
        self.runner = DayRunner(workspace, trade_date, {}, journal=journal)
        self.core: CoreContext | None = None
        self.research: ResearchSources | None = None
        self.adapters = {}
        self.templates = {}

    def resolve(self, node: str, completed: dict):
        journal, inputs = self.runner.journal, self.inputs
        journal._require_lock()
        if node not in self.node_ids:
            raise ContractError("NATIVE_REGISTRY_NODE_UNKNOWN")
        if node in self.adapters:
            return self.templates[node], self.adapters[node]
        adapter: Any
        if node in CORE_NODES:
            if self.core is None:
                core = CoreContext(
                    self.workspace,
                    self.trade_date,
                    inputs.factor_pointer_sha256,
                    inputs.release_ref,
                    next_session_calendar_proof_ref=inputs.next_session_calendar_proof_ref,
                    next_session_calendar_failure_ref=inputs.next_session_calendar_failure_ref,
                )
                core.prepare(journal)
                self.core = core
            adapter = core_registry(self.core)[node]
            template = self.core.template(node)
        elif node in SOURCE_NODES or node == "decision":
            if self.research is None:
                top = completed.get("top100")
                if top is None or top.get("state") != "SUCCEEDED":
                    return None
                manifest = top["terminal"]["output_refs"]["manifest.json"]
                research = ResearchSources(
                    workspace=self.workspace,
                    journal=journal,
                    native_request_ref=inputs.research_request_ref,
                    pool_manifest_ref=manifest,
                    release_ref=inputs.release_ref,
                )
                research.prepare()
                self.research = research
            adapter = (
                ResearchDecisionAdapter(
                    self.research,
                    decision_recipe_ref=inputs.decision_recipe_ref,
                    store_plan_ref=inputs.store_plan_ref,
                )
                if node == "decision"
                else ResearchSourceAdapter(self.research, node)
            )
            template = adapter.template() if node == "decision" else self.research.template(node)
        elif node == "corporate_action_recon":
            from quant_investor.operations.corporate_adapter import CorporateReconciliationAdapter

            corporate_type: Any = CorporateActionAdapter
            corporate_args = {}
            if inputs.corporate_action_context_ref is not None:
                corporate_type = CorporateReconciliationAdapter
                corporate_args = {
                    "corporate_action_context_ref": inputs.corporate_action_context_ref,
                    "decision_recipe_ref": inputs.decision_recipe_ref,
                    "research_request_ref": inputs.research_request_ref,
                    "store_plan_ref": inputs.store_plan_ref,
                }
                if inputs.registered_event_declaration_ref is not None:
                    corporate_args["registered_event_declaration_ref"] = (
                        inputs.registered_event_declaration_ref
                    )
            adapter = corporate_type(
                **corporate_args,
                workspace=self.workspace,
                journal=journal,
                event_pointer_sha256=inputs.event_pointer_sha256,
                release_ref=inputs.release_ref,
                previous_trade_date=inputs.previous_trade_date,
                market_refs=inputs.adjustment_market_refs,
                calendar_ref=inputs.calendar_ref,
                market_snapshot_ref=inputs.market_snapshot_ref,
            )
            adapter.prepare()
            template = adapter.template()
        elif node == "store":
            adapter = StoreCloseAdapter(
                arguments=inputs.store_arguments,
                trade_date=self.trade_date,
                plan_ref=inputs.store_plan_ref,
                release_ref=inputs.release_ref,
            )
            template = adapter.template()
        else:
            dashboard_args = {}
            if inputs.dashboard_publication_policy is not None:
                from scripts.daily_dashboard_sealed import SealedDashboardAdapter
                from quant_investor.operations.dashboard_evidence import DOMAINS

                if any(completed.get(name, {}).get("state") != "SUCCEEDED" for name in DOMAINS):
                    return None
                adapter_type: Any = SealedDashboardAdapter
                dashboard_args = {
                    "authority_terminal_refs": {
                        name: completed[name]["terminal_ref"] for name in DOMAINS
                    },
                    "publish_current_dashboard": inputs.publish_current_dashboard,
                }
                if inputs.registered_event_declaration_ref is not None:
                    corporate = completed.get("corporate_action_recon")
                    if corporate is None or corporate.get("state") != "SUCCEEDED":
                        return None
                    dashboard_args.update(
                        corporate_terminal_ref=corporate["terminal_ref"],
                        registered_event_declaration_ref=inputs.registered_event_declaration_ref,
                    )
            else:
                adapter_type = (
                    CurrentDashboardAdapter
                    if inputs.publish_current_dashboard
                    else HistoricalDashboardAdapter
                )
            adapter = adapter_type(
                **dashboard_args,
                workspace=self.workspace,
                journal=journal,
                release_ref=inputs.release_ref,
                plan_ref=inputs.store_plan_ref,
                market_ref=inputs.market_snapshot_ref,
                benchmark_ref=inputs.benchmark_ref,
                risk_free_ref=inputs.risk_free_ref,
            )
            adapter.prepare()
            template = adapter.template()
        self.adapters[node], self.templates[node] = adapter, template
        return template, adapter

    def run(self, *, resume: bool = False) -> dict:
        with self.runner.journal.locked():
            return self.runner.run_locked({}, resume=resume, resolve=self.resolve)

    def seal_completion(self, *, native_inputs_ref: dict, synthetic: bool) -> dict:
        """Seal only an already executed complete native context under its day lock."""
        from scripts.daily_completion import seal_native_completion

        return seal_native_completion(
            self, native_inputs_ref=native_inputs_ref, synthetic=synthetic
        )
