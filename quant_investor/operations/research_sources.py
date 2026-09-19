"""Native Industry/Theme source projections as independently resumable DAG nodes."""

from datetime import datetime, timezone
from functools import partial
import hashlib
from pathlib import PurePosixPath, Path
from typing import Mapping

from quant_investor.cli import unified
from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.intelligence.storage import DailyResearchPoolStore, approved_theme_policy_v2
from quant_investor.intelligence.pool_tabular import POOL_MANIFEST_KINDS
from quant_investor.system.storage import SecureSystemStorage
from .dependency_diagnostics import DependencyInputError
from .daily_contract import ContractError, GRAPH_SHA256, NodeState, validate_ref
from .daily_journal import DailyJournal, FALSE_AUTHORITY, request_identity, _false_authority
from .daily_contract import utc_stamp
from .daily_runner import NativeOutcome, Probe

from .research_projection import Projected, project_research_source
from .research_file_readback import read_research_bytes, parse_research_json
from .research_recipes import derive_focus_context, source_recipe, artifact_output_names
from quant_investor.intelligence.pcb_ai_hardware import MEMBERSHIP_KIND
from quant_investor.intelligence.low_frequency import freshness_profile
from quant_investor.intelligence.macro_freshness import (
    macro_freshness,
    macro_freshness_from_closure,
)

SOURCE_NODES = {
    "industry": "industry_source",
    "theme": "theme_source",
    "exposure": "exposure_rows",
    "fundamental": "fundamental_source",
    "macro": "macro_risk",
}


class ResearchSources:
    def __init__(
        self,
        *,
        workspace: str,
        journal: DailyJournal,
        native_request_ref: Mapping[str, str],
        pool_manifest_ref: Mapping[str, str],
        release_ref: Mapping[str, str],
        _request_bytes: bytes | None = None,
    ):
        self.workspace = workspace
        self.journal = journal
        self.reader = SecureSystemStorage(workspace)
        self.request_ref = validate_ref(native_request_ref)
        self.pool_ref = validate_ref(pool_manifest_ref)
        self.release_ref = validate_ref(release_ref)
        self._preview_only = _request_bytes is not None
        self.source_completion_policy = None
        if self._preview_only:
            if (
                type(_request_bytes) is not bytes
                or hashlib.sha256(_request_bytes).hexdigest() != self.request_ref["sha256"]
            ):
                raise DependencyInputError("RESEARCH_PREVIEW_REQUEST_SHA_MISMATCH")
            self.request = parse_research_json(_request_bytes)
        else:
            from .research_request import load_research_request

            loaded = load_research_request(workspace=workspace, reference=self.request_ref)
            self.request = loaded["document"]
            self.source_completion_policy = loaded["source_completion_policy"]
        required = {
            "as_of",
            "strategy_id",
            "policy",
            "expected_factor_pointer_sha256",
            "industry_source",
            "theme_source",
        }
        optional = {
            "low_observation_path",
            "low_observation_sha256",
            "w80_observation_path",
            "w80_observation_sha256",
            "company_evidence",
            "expected_trade_date",
        }
        if not required <= set(self.request) or set(self.request) - required - optional:
            raise DependencyInputError("RESEARCH_SOURCE_SCHEMA_INVALID")
        if (
            self.request.get("company_evidence") is not None
            and type(self.request["company_evidence"]) is not dict
        ):
            raise DependencyInputError("RESEARCH_COMPANY_SOURCE_SCHEMA_INVALID")
        self._read(self.release_ref)
        self.pool = self._read(self.pool_ref)
        validate_artifact(self.pool)
        day = journal.trade_date
        expected = (
            "results/intelligence/research_pool/aggressive_tech_manufacturing/"
            f"{day[:4]}-{day[4:6]}-{day[6:]}/manifest.json"
        )
        if self.pool["kind"] not in POOL_MANIFEST_KINDS:
            raise DependencyInputError("RESEARCH_SOURCE_POOL_BINDING_INVALID")
        if (
            self.pool["payload"]["signal_date"] != day
            or self.request["as_of"][:10].replace("-", "") != day
        ):
            raise DependencyInputError("RESEARCH_SOURCE_DATE_MISMATCH")
        if (
            self.pool_ref["path"] != expected
            or self.request["strategy_id"] != "aggressive_tech_manufacturing"
            or self.request["policy"] != approved_theme_policy_v2()
            or self.request["expected_factor_pointer_sha256"]
            != self.pool["payload"]["factor_pointer_sha256"]
        ):
            raise DependencyInputError("RESEARCH_SOURCE_POOL_BINDING_INVALID")
        rank_path = str(PurePosixPath(expected).parent / "factor_research_rank.json")
        self.rank = self._read(
            {"path": rank_path, "sha256": self.pool["payload"]["rank_byte_sha256"]}
        )
        self.pool_store = DailyResearchPoolStore(workspace)
        self.companies = [row["symbol"] for row in self.rank["payload"]["pool_rows"]]
        self.recipe_refs = {}
        self._verify_pool()
        self.focus_context = derive_focus_context(
            workspace=self.workspace,
            request=self.request,
            rank=self.rank,
            pool_store=self.pool_store,
        )

    def _read(self, ref: Mapping[str, str]) -> dict:
        validate_ref(ref)
        stored = read_research_bytes(self.reader, ref, maximum_bytes=8 * 1024 * 1024)
        if stored.byte_sha256 != ref["sha256"]:
            raise DependencyInputError("RESEARCH_SOURCE_SHA_MISMATCH")
        return parse_research_json(stored.data, label="research source")

    def _verify_pool(self) -> None:
        self._read(self.pool_ref)
        self.pool_store.verify(
            rank=self.rank,
            expected_policy_sha256=self.pool["payload"]["policy_byte_sha256"],
            policy_path=self.pool["payload"]["policy_path"],
        )

    def prepare(self) -> None:
        self.journal._require_lock()
        if self._preview_only:
            raise ContractError("RESEARCH_PREVIEW_CANNOT_PUBLISH")
        for node, field in SOURCE_NODES.items():
            recipe = source_recipe(
                node=node,
                field=field,
                request=self.request,
                pool_ref=self.pool_ref,
                focus_context=self.focus_context,
                completion_policy=self.source_completion_policy,
            )
            raw = canonical_json_bytes(recipe)
            sha = hashlib.sha256(raw).hexdigest()
            path = str(self.journal.root / "research" / "recipes" / (sha + ".json"))
            self.journal.storage.write(path, raw)
            self.recipe_refs[node] = {"path": path, "sha256": sha}

    def template(self, node: str) -> dict:
        if self._preview_only:
            raise ContractError("RESEARCH_PREVIEW_CANNOT_PUBLISH")
        return {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": self.journal.trade_date,
            "node_id": node,
            "graph_sha256": GRAPH_SHA256,
            "release_ref": self.release_ref,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy_refs": {
                "research": {
                    "path": self.pool["payload"]["policy_path"],
                    "sha256": self.pool["payload"]["policy_byte_sha256"],
                }
            },
            "input_refs": {"pool": self.pool_ref, "recipe": self.recipe_refs[node]},
        }

    def project(self, node: str, request: dict) -> Projected | None:
        observed = {
            **request,
            "input_refs": {
                k: v for k, v in request["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != self.template(node):
            raise DependencyInputError("RESEARCH_SOURCE_REQUEST_MISMATCH")
        recipe = self._read(self.recipe_refs[node])
        self._read(self.release_ref)
        self._verify_pool()
        if node in {"theme", "exposure"} and self.focus_context is not None:
            if recipe.get("focus_context") != self.focus_context:
                raise DependencyInputError("FOCUS_RECIPE_CONTEXT_MISMATCH")
            for ref in self.focus_context["source_refs"]:
                stored = self.reader.read_workspace_file_bytes(
                    ref["path"], maximum_bytes=256 * 1024 * 1024
                )
                if stored.byte_sha256 != ref["sha256"]:
                    raise ContractError("FOCUS_PIT_SOURCE_CHANGED")
        source_document = partial(unified._daily_source_document, self.workspace)
        source_file = partial(unified._daily_source_file, Path(self.workspace).resolve(strict=True))
        if node != "macro":
            theme = None
            focus_membership = None
            if node == "exposure":
                terminal = self._read(request["input_refs"]["upstream.theme"])
                if terminal["state"] != "SUCCEEDED":
                    raise DependencyInputError("EXPOSURE_THEME_UPSTREAM_INCOMPLETE")
                theme = self._read(terminal["output_refs"]["artifact"])
                if self.focus_context is not None:
                    if MEMBERSHIP_KIND not in terminal["output_refs"]:
                        raise DependencyInputError("FOCUS_MEMBERSHIP_UPSTREAM_MISSING")
                    focus_membership = self._read(terminal["output_refs"][MEMBERSHIP_KIND])
            return project_research_source(
                node=node,
                recipe=recipe,
                companies=self.companies,
                source_document=source_document,
                source_file=source_file,
                workspace=self.workspace,
                theme=theme,
                focus_membership=focus_membership,
            )
        return _project_macro(self, recipe, source_file)


def _project_macro(context, recipe, source_file):
    profile = freshness_profile(recipe)
    artifact, _, _ = unified._daily_macro_evidence(
        recipe["macro_risk"], source_file, Path(context.workspace), context.rank, recipe
    )
    artifacts = [] if artifact is None else [artifact]
    missing = artifact is None or artifact["payload"]["classification"] == "PIPELINE_DATA_VETO"
    if profile is not None:
        if missing:
            report = macro_freshness(
                observations=[],
                as_of=recipe["as_of"],
                refs=[] if recipe["macro_risk"] is None else [recipe["macro_risk"]["source"]],
                missing_code=(
                    "MACRO_READINESS_SOURCE_MISSING"
                    if artifact is None
                    else "MACRO_RELEASE_CONTRACT_BLOCKED"
                ),
            )
        else:
            report = macro_freshness_from_closure(
                workspace=context.workspace,
                as_of=recipe["as_of"],
                closure_ref=recipe["macro_risk"]["source"],
                source_file=source_file,
            )
        missing |= bool(report["payload"]["critical_missing_codes"])
        artifacts.append(report)
    if not artifacts:
        return None
    return Projected(artifacts, NodeState.PARTIAL if missing else NodeState.SUCCEEDED)


class ResearchSourceAdapter:
    resume_safe = True

    def __init__(self, context: ResearchSources, node: str):
        if node not in SOURCE_NODES:
            raise ContractError("RESEARCH_SOURCE_NODE_INVALID")
        self.context, self.node = context, node

    def _expected(self, request: dict):
        projected = self.context.project(self.node, request)
        if projected is None:
            return None
        rows = []
        for artifact in projected.artifacts:
            validate_artifact(artifact)
            raw = canonical_json_bytes(artifact)
            sha = hashlib.sha256(raw).hexdigest()
            path = str(self.context.journal.root / "research" / "artifacts" / (sha + ".json"))
            rows.append((raw, {"path": path, "sha256": sha}))
        return projected, rows

    def probe(self, request: dict) -> Probe:
        expected = self._expected(request)
        if expected is None:
            return Probe(NativeOutcome(NodeState.BLOCKED, {}, "INPUT_MISSING"))
        projected, rows = expected
        _, key = request_identity(request)
        index = str(self.context.journal.root / "research" / "nodes" / self.node / (key + ".json"))
        stored = self.context.journal.storage.read(index)
        if stored is None:
            return Probe(None, safe_to_execute=True)
        value = parse_canonical_json_bytes(stored.data, label="research node capture")
        refs = [ref for _, ref in rows]
        if (
            value.get("artifact_refs") != refs
            or value.get("request_key") != key
            or value.get("state") != projected.state.value
            or not _false_authority(value.get("authority"))
            or value.get("schema_version") != "cn-daily-research-node-capture.v1"
        ):
            raise ContractError("RESEARCH_SOURCE_CAPTURE_MISMATCH")
        cutoff = projected.artifacts[0]["created_at"]
        if value["research_cutoff"] != cutoff or utc_stamp(value["captured_at"]) < utc_stamp(
            cutoff
        ):
            raise ContractError("RESEARCH_SOURCE_CAPTURE_CLOCK_INVALID")
        for raw, ref in rows:
            leaf = self.context.journal.storage.read(ref["path"])
            if leaf is None or leaf.data != raw:
                raise ContractError("RESEARCH_SOURCE_ARTIFACT_DRIFT")
        output = dict(zip(artifact_output_names(projected.artifacts), refs))
        output["capture"] = {"path": index, "sha256": stored.byte_sha256}
        return Probe(
            NativeOutcome(
                projected.state,
                output,
                "INPUT_MISSING" if projected.state == NodeState.PARTIAL else None,
            )
        )

    def execute(self, request: dict) -> None:
        self.context.journal._require_lock()
        expected = self._expected(request)
        if expected is None:
            raise ContractError("RESEARCH_SOURCE_NOT_AVAILABLE")
        projected, rows = expected
        cutoff = projected.artifacts[0]["created_at"]
        if datetime.now(timezone.utc) < utc_stamp(cutoff):
            raise ContractError("RESEARCH_SOURCE_CUTOFF_IN_FUTURE")
        _, key = request_identity(request)
        storage = self.context.journal.storage
        for raw, ref in rows:
            storage.write(ref["path"], raw)
        index = str(self.context.journal.root / "research" / "nodes" / self.node / (key + ".json"))
        value = {
            "schema_version": "cn-daily-research-node-capture.v1",
            "request_key": key,
            "artifact_refs": [ref for _, ref in rows],
            "state": projected.state.value,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "research_cutoff": cutoff,
            "authority": FALSE_AUTHORITY,
        }
        storage.write(index, canonical_json_bytes(value))
