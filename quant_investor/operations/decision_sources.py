"""Bind the exact recorded upstream closures used by a new Decision report."""

from pathlib import PurePosixPath, Path
import hashlib

from quant_investor.contracts import (
    parse_canonical_json_bytes,
    canonical_json_bytes,
    validate_artifact,
)
from quant_investor.intelligence._common import artifact_ref
from quant_investor.intelligence.low_frequency import FRESHNESS_KIND
from quant_investor.intelligence.decision_report import DOMAINS
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref
from .daily_journal import DailyJournal, request_identity, _false_authority
from .research_file_readback import ResearchFileReadback


class DecisionSources:
    def __init__(
        self,
        *,
        workspace,
        trade_date,
        request,
        result,
        portfolio_state_ref,
        portfolio,
        context=None,
    ):
        self.workspace, self.trade_date, self.request = workspace, trade_date, request
        self.result, self.context = result, context
        self.reader = SecureSystemStorage(workspace)
        self.journal = DailyJournal(str(workspace), trade_date)
        self.observed = {}
        self.files = ResearchFileReadback(str(workspace))
        self.physical = {domain: [] for domain in DOMAINS}
        self.freshness = {"fundamental": None, "macro": None}
        self.physical["portfolio"] = [portfolio_state_ref, *portfolio["payload"]["source_refs"]]

    def read_bytes(self, ref):
        ref = validate_ref(ref)
        path, raw, _ = self.files.source_file(ref, code="DECISION_SOURCE_SHA_MISMATCH")
        if path != Path(self.workspace).resolve(strict=True) / ref["path"]:
            raise ContractError("DECISION_SOURCE_PATH_ALIAS_REJECTED")
        prior = self.observed.get(ref["path"])
        if prior is not None and prior != raw:
            raise ContractError("DECISION_SOURCE_CHANGED")
        self.observed[ref["path"]] = raw
        return raw

    def read(self, ref):
        return parse_canonical_json_bytes(self.read_bytes(ref))

    def terminal(self, node, ref):
        terminal = self.read(ref)
        path = str(PurePosixPath(ref["path"]).parent.parent / "request.json")
        stored = self.reader.read_workspace_file_bytes(path, maximum_bytes=8 * 1024 * 1024)
        request_ref = {"path": path, "sha256": stored.byte_sha256}
        request = self.read(request_ref)
        if (
            request_identity(request)[1] != terminal["request_key"]
            or request["node_id"] != node
            or request["trade_date"] != self.trade_date
            or request["release_ref"] != self.request["release_ref"]
            or terminal.get("state") != "SUCCEEDED"
            or not _false_authority(terminal.get("authority"))
        ):
            raise ContractError("DECISION_UPSTREAM_TERMINAL_BINDING_INVALID")
        selected = self.journal.readonly_inspect(request)
        if selected.get("terminal_ref") != ref or selected.get("terminal") != terminal:
            raise ContractError("DECISION_UPSTREAM_JOURNAL_MISMATCH")
        refs = [ref, request_ref, *terminal["output_refs"].values()]
        for value in refs:
            self.read_bytes(value)
        return terminal, request, refs

    def collect(self):
        rank = next(a for a in self.result["artifacts"] if a["kind"] == "factor_research_rank")
        top, top_request, refs = self.terminal(
            "top100", self.request["input_refs"]["upstream.top100"]
        )
        if top["output_refs"].get("manifest.json") != self.request["input_refs"]["pool"]:
            raise ContractError("DECISION_TOP100_MANIFEST_MISMATCH")
        self.physical["top100"] = refs
        for node, output in (
            ("factor", "generation"),
            ("low_observation", "LOW"),
            ("w80_observation", "W80"),
        ):
            terminal, _, refs = self.terminal(node, top_request["input_refs"]["upstream." + node])
            artifact = validate_artifact(self.read(terminal["output_refs"][output]))
            expected = (
                [rank["payload"]["factor_generation_ref"]]
                if node == "factor"
                else rank["payload"]["observation_refs"]
            )
            if artifact_ref(artifact) not in expected:
                raise ContractError("DECISION_FACTOR_OBSERVATION_BINDING_INVALID")
            self.physical["factor"].extend(refs)
        native = {canonical_json_bytes(artifact_ref(a)): a for a in self.result["artifacts"]}
        for node in ("theme", "industry", "exposure", "fundamental", "macro"):
            terminal, request, refs = self.terminal(
                node, self.request["input_refs"]["upstream." + node]
            )
            self.physical[node] = refs
            if self.context is not None:
                from .research_sources import ResearchSourceAdapter

                outcome = ResearchSourceAdapter(self.context, node).probe(request).outcome
                if (
                    outcome is None
                    or outcome.state.value != "SUCCEEDED"
                    or outcome.output_refs != terminal["output_refs"]
                ):
                    raise ContractError("DECISION_SOURCE_NATIVE_PROBE_MISMATCH")
            for name, ref in terminal["output_refs"].items():
                if name == "capture":
                    continue
                value = validate_artifact(self.read(ref))
                if value["kind"] == FRESHNESS_KIND:
                    if node not in self.freshness or name != FRESHNESS_KIND:
                        raise ContractError("DECISION_FRESHNESS_OUTPUT_MISMATCH")
                    self.freshness[node] = value
                    continue
                key = canonical_json_bytes(artifact_ref(value))
                if key in native and value != native[key]:
                    raise ContractError("DECISION_NATIVE_SOURCE_ARTIFACT_MISMATCH")
                if (
                    value["kind"]
                    in {
                        "industry_source_projection",
                        "theme_membership_projection",
                        "theme_economic_exposure_projection",
                        "market_risk_evidence",
                    }
                    and key not in native
                ):
                    raise ContractError("DECISION_NATIVE_PROJECTION_BINDING_MISMATCH")
        if any(value is None for value in self.freshness.values()):
            raise ContractError("DECISION_FRESHNESS_REPORT_MISSING")
        for ref in self.physical["portfolio"]:
            self.read_bytes(ref)
        self.recheck()
        return {"domain_physical_refs": self.physical, "freshness_reports": self.freshness}

    def recheck(self):
        for path, raw in self.observed.items():
            self.read_bytes({"path": path, "sha256": hashlib.sha256(raw).hexdigest()})
