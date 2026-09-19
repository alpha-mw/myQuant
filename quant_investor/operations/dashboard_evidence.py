"""Dated Dashboard evidence derived from five exact successful native DAG sources."""

from pathlib import Path, PurePosixPath
from datetime import datetime, timezone

from quant_investor.contracts import (
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.strategy_records.event_receipts import read_event_source
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.intelligence._common import build_artifact, business_identity, artifact_ref
from quant_investor.intelligence.investment_decision import DECISION_STATES
from quant_investor.intelligence.pool_tabular import TABULAR_MANIFEST_KIND, verify_top100
from quant_investor.intelligence.pcb_ai_hardware import physical_refs
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal, request_identity, _false_authority

KIND = "daily_dashboard_evidence"
DOMAINS = ("store", "factor", "top100", "theme", "decision")
STRATEGY = "aggressive_tech_manufacturing"


class DashboardEvidenceSources:
    def __init__(self, *, workspace, trade_date, release_ref, terminal_refs):
        if type(terminal_refs) is not dict or set(terminal_refs) != set(DOMAINS):
            raise ContractError("DASHBOARD_DAG_SOURCE_SET_INVALID")
        self.workspace = Path(workspace)
        self.day = trade_date
        self.release = validate_ref(release_ref)
        self.refs = {name: validate_ref(ref) for name, ref in terminal_refs.items()}
        self.journal = DailyJournal(str(workspace), trade_date)
        self.reader = SecureSystemStorage(workspace)
        self.seen = {}
        self.requests = {}
        self.finished = {}
        self.bindings = {name: self._terminal(name, self.refs[name]) for name in DOMAINS}
        self._derive()
        self.recheck()

    def read_bytes(self, ref):
        ref = validate_ref(ref)
        raw = read_event_source(self.workspace, ref)
        key = (ref["path"], ref["sha256"])
        if key in self.seen and self.seen[key] != raw:
            raise ContractError("DASHBOARD_DAG_SOURCE_CHANGED")
        self.seen[key] = raw
        return raw

    def read(self, ref, *, native=False):
        raw = self.read_bytes(ref)
        return (
            parse_json_bytes(raw, label="Dashboard native source")
            if native
            else parse_canonical_json_bytes(raw)
        )

    def _terminal(self, name, ref):
        terminal = self.read(ref)
        path = str(PurePosixPath(ref["path"]).parent.parent / "request.json")
        stored = self.reader.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
        request_ref = {"path": path, "sha256": stored.byte_sha256}
        request = self.read(request_ref)
        if (
            request_identity(request)[1] != terminal["request_key"]
            or request["node_id"] != name
            or request["trade_date"] != self.day
            or request["release_ref"] != self.release
            or terminal.get("state") != "SUCCEEDED"
            or not _false_authority(terminal.get("authority"))
        ):
            raise ContractError("DASHBOARD_DAG_TERMINAL_BINDING_INVALID")
        selected = self.journal.readonly_inspect(request)
        if selected.get("terminal_ref") != ref or selected.get("terminal") != terminal:
            raise ContractError("DASHBOARD_DAG_SELECTED_TERMINAL_MISMATCH")
        for output in terminal["output_refs"].values():
            self.read_bytes(output)
        self.requests[name] = request
        self.finished[name] = terminal["finished_at"]
        return {
            "node_id": name,
            "request_ref": request_ref,
            "terminal_ref": ref,
            "output_refs": terminal["output_refs"],
            "trade_date": self.day,
            "state": "SUCCEEDED",
        }

    def artifact(self, domain, output, kind):
        return validate_artifact(
            self.read(self.bindings[domain]["output_refs"][output]), expected_kind=kind
        )

    def _derive(self):
        factor = self.artifact("factor", "generation", "factor.production_generation")
        pool = self.artifact("top100", "manifest.json", TABULAR_MANIFEST_KIND)
        rank = self.artifact("top100", "factor_research_rank.json", "factor_research_rank")
        selected = self.artifact(
            "top100", "selected_symbols.json", "daily_research_selected_symbols"
        )
        theme = self.artifact("theme", "artifact", "theme_membership_projection")
        decision = self.artifact("decision", "decision.v2.json", "daily_research_decision_report")
        if decision["payload"]["native_result_ref"] != self.bindings["decision"]["output_refs"].get(
            "result"
        ):
            raise ContractError("DASHBOARD_DAG_DECISION_RESULT_MISMATCH")
        pool_ref = self.bindings["top100"]["output_refs"]["manifest.json"]
        if any(
            self.requests[name]["input_refs"].get("pool") != pool_ref
            for name in ("theme", "decision")
        ):
            raise ContractError("DASHBOARD_DAG_POOL_REQUEST_MISMATCH")
        portfolio = validate_artifact(
            self.read(decision["payload"]["portfolio_state_ref"]),
            expected_kind="research_portfolio_state",
        )
        if portfolio["payload"]["store_plan_ref"] != self.requests["store"]["input_refs"].get(
            "native_plan"
        ):
            raise ContractError("DASHBOARD_DAG_PORTFOLIO_PLAN_MISMATCH")
        manual = self.read(self.bindings["store"]["output_refs"]["manual"], native=True)
        if any(
            str(stamp).replace("-", "")[:8] != self.day
            for stamp in (
                factor["payload"]["as_of"][:10],
                pool["payload"]["signal_date"],
                rank["payload"]["signal_date"],
                selected["payload"]["signal_date"],
                theme["payload"]["trade_date"],
                decision["payload"]["trade_date"],
                manual["valuation_trade_date"],
            )
        ):
            raise ContractError("DASHBOARD_DAG_SOURCE_DATE_MISMATCH")
        if any(
            value["payload"]["strategy_id"] != STRATEGY
            for value in (pool, rank, selected, decision)
        ):
            raise ContractError("DASHBOARD_DAG_STRATEGY_MISMATCH")
        if pool["payload"]["factor_generation_ref"] != artifact_ref(factor) or rank["payload"][
            "factor_generation_ref"
        ] != artifact_ref(factor):
            raise ContractError("DASHBOARD_DAG_FACTOR_BINDING_MISMATCH")
        parquet = self.bindings["top100"]["output_refs"]["top100.parquet"]
        verify_top100(
            self.read_bytes(parquet), expected_sha=pool["payload"]["top100_sha"], rank=rank
        )
        symbols = selected["payload"]["ordered_symbols"]
        rank_symbols = [row["symbol"] for row in rank["payload"]["pool_rows"]]
        decisions = decision["payload"]["company_rows"]
        decision_symbols = [row["symbol"] for row in decisions]
        theme_symbols = [row["company_code"] for row in theme["payload"]["company_rows"]]
        if (
            symbols != rank_symbols
            or len(symbols) != len(set(symbols))
            or set(symbols) != set(decision_symbols)
            or len(decision_symbols) != len(symbols)
            or set(symbols) != set(theme_symbols)
            or len(theme_symbols) != len(symbols)
            or pool["payload"]["pool_size"] != len(symbols)
            or pool["payload"]["row_count"] != len(symbols)
        ):
            raise ContractError("DASHBOARD_DAG_COMPANY_SET_MISMATCH")
        self.counts = dict.fromkeys(DECISION_STATES, 0)
        for row in decisions:
            if row["decision"] not in self.counts:
                raise ContractError("DASHBOARD_DAG_DECISION_STATE_INVALID")
            self.counts[row["decision"]] += 1
        self.pool_size = len(symbols)
        self.cutoff = decision["payload"]["as_of"]

    def recheck(self):
        for (path, sha), raw in self.seen.items():
            if read_event_source(self.workspace, {"path": path, "sha256": sha}) != raw:
                raise ContractError("DASHBOARD_DAG_SOURCE_CHANGED")

    def build(self, created_at):
        self.recheck()
        stamp = utc_stamp(created_at)
        if stamp > datetime.now(timezone.utc) or any(
            utc_stamp(t) > stamp for t in self.finished.values()
        ):
            raise ContractError("DASHBOARD_EVIDENCE_CUSTODY_INVALID")
        fields = {
            "trade_date": self.day,
            "strategy_id": STRATEGY,
            "source_bindings": self.bindings,
            "source_refs": physical_refs([{"path": p, "sha256": h} for p, h in self.seen]),
            "top100_count": self.pool_size,
            "decision_state_counts": self.counts,
            "research_state": "EVIDENCE_BOUND",
        }
        return build_artifact(
            kind=KIND,
            identity_field="dashboard_evidence_id",
            identity=business_identity(
                kind=KIND, identity_inputs={"created_at": created_at, **fields}
            ),
            created_at=created_at,
            fields=fields,
        )
