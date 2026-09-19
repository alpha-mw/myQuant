"""Native inactive Decision compilation and immutable custody adapter."""

from pathlib import Path
import hashlib

from quant_investor.cli.unified import research_compile_daily
from quant_investor.contracts import canonical_json_bytes
from .daily_contract import ContractError, GRAPH_SHA256, NodeState
from .daily_runner import NativeOutcome, Probe
from .research_capture import ResearchCapture
from .research_sources import ResearchSources


class ResearchDecisionAdapter:
    resume_safe = True

    def __init__(self, context: ResearchSources, *, decision_recipe_ref=None, store_plan_ref=None):
        self.context = context
        self.capture = ResearchCapture(context.journal)
        self.report = None
        if decision_recipe_ref is not None:
            from .decision_publication import DecisionReportPublication

            self.report = DecisionReportPublication(
                journal=context.journal,
                recipe_ref=decision_recipe_ref,
                research_request_ref=context.request_ref,
                store_plan_ref=store_plan_ref,
                context=context,
            )

    def template(self) -> dict:
        c = self.context
        value = {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": c.journal.trade_date,
            "node_id": "decision",
            "graph_sha256": GRAPH_SHA256,
            "release_ref": c.release_ref,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy_refs": {
                "research": {
                    "path": c.pool["payload"]["policy_path"],
                    "sha256": c.pool["payload"]["policy_byte_sha256"],
                }
            },
            "input_refs": {"native_request": c.request_ref, "pool": c.pool_ref},
        }
        if self.report is not None:
            value["input_refs"]["decision_recipe"] = self.report.recipe_ref
        return value

    def _compile(self, request: dict) -> dict:
        observed = {
            **request,
            "input_refs": {
                k: v for k, v in request["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != self.template():
            raise ContractError("DECISION_ADAPTER_REQUEST_MISMATCH")
        self.context._verify_pool()
        result = research_compile_daily(
            workspace_root=self.context.workspace,
            request_path=self.context.request_ref["path"],
            expected_request_sha256=self.context.request_ref["sha256"],
        )
        ranks = [a for a in result["artifacts"] if a["kind"] == "factor_research_rank"]
        if (
            len(ranks) != 1
            or ranks[0]["payload"]["pool_rows"] != self.context.rank["payload"]["pool_rows"]
            or result["strategy_id"] != "aggressive_tech_manufacturing"
        ):
            raise ContractError("DECISION_TOP100_CLOSURE_MISMATCH")
        return result

    def probe(self, request: dict) -> Probe:
        result = self._compile(request)
        captured = self.capture.read(self.context.request_ref)
        if captured is None:
            return Probe(None, safe_to_execute=True)
        manifest, previous = captured
        if canonical_json_bytes(previous) != canonical_json_bytes(result):
            raise ContractError("DECISION_COMPILATION_DRIFT")
        # Execution completeness is distinct from native admission/research state.
        # The fixed graph must have satisfied all critical source prerequisites.
        path = self.capture._root(self.context.request_ref) + "/capture.v1.json"
        stored = self.context.journal.storage.read(path)
        if stored is None:
            raise ContractError("DECISION_CAPTURE_DISAPPEARED")
        reports = {} if self.report is None else self.report.probe(request, captured)
        if reports is None:
            return Probe(None, safe_to_execute=True)
        return Probe(
            NativeOutcome(
                NodeState.SUCCEEDED,
                {
                    "capture": {"path": path, "sha256": stored.byte_sha256},
                    "result": manifest["result_ref"],
                    **reports,
                },
            )
        )

    def execute(self, request: dict) -> None:
        self.context.journal._require_lock()
        result = self._compile(request)
        self.capture.publish(self.context.request_ref, result)
        if self.report is not None:
            captured = self.capture.read(self.context.request_ref)
            if captured is None:
                raise ContractError("DECISION_CAPTURE_DISAPPEARED")
            self.report.execute(request, captured)
