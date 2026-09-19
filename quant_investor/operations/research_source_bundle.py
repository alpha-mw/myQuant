"""Exact research-source selection and unpublished native preview for cutoff sealing."""

import hashlib
from pathlib import Path

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.intelligence.theme_sources import split_theme_source
from quant_investor.intelligence.pcb_ai_hardware import MEMBERSHIP_KIND
from quant_investor.intelligence.macro_freshness import macro_freshness_from_closure
from quant_investor.intelligence.daily_evidence import build_market_risk_evidence
from quant_investor.macro.readiness_closure import (
    validate_macro_readiness_closure,
    _readiness_projection,
)
from quant_investor.cli import unified
from .dependency_diagnostics import DependencyInputError
from .daily_contract import ContractError, NodeState, validate_ref, utc_stamp
from .daily_journal import _false_authority
from .production_request import validate_production_request
from .execution_recipe import SCHEMA_V5, SCHEMA_V6, validate_execution_recipe
from .maintenance_handoff_contract import validate_handoff_shape
from .research_file_readback import ResearchFileReadback, parse_research_json
from .research_materialization import collect_research_input
from .research_sources import ResearchSources
from .research_recipes import source_recipe, artifact_output_names
from .research_projection import project_research_source
from .research_timing import validate_research_timing_policy, CURRENT
from .exposure_completion import POLICY
from .exposure_catalog import select_exposure_rows

SCHEMA = "cn-daily-acquired-sources.v1"
SCHEMA_V2 = "cn-daily-acquired-sources.v2"
FIELDS = frozenset(
    {
        "schema_version",
        "trade_date",
        "request_ref",
        "maintenance_handoff_ref",
        "core_handoff_ref",
        "pool_manifest_ref",
        "theme_source_handoff_ref",
        "corporate_action_template_ref",
        "timing_policy_ref",
        "auxiliary_stage_refs",
        "event_pointer_ref",
        "native_request_fields",
    }
)
NATIVE_FIELDS = frozenset(
    {
        "strategy_id",
        "policy",
        "expected_trade_date",
        "expected_factor_pointer_sha256",
        "industry_source",
        "theme_source",
        "company_evidence",
        "low_observation_path",
        "low_observation_sha256",
        "w80_observation_path",
        "w80_observation_sha256",
    }
)


def build_source_bundle(*, journal, recovered, auxiliary):
    """Return candidate bytes only. Caller freezes them only after mandatory preview."""
    journal._require_lock()
    recipe, handoff = recovered["recipe"], recovered["handoff"]
    if recipe["schema_version"] not in {SCHEMA_V5, SCHEMA_V6}:
        raise ContractError("CUTOFF_RECIPE_V5_REQUIRED")
    collected = collect_research_input(journal=journal, recovered=recovered, auxiliary=auxiliary)
    fields = {k: v for k, v in collected["document"].items() if k != "as_of"}
    from .corporate_actions import retain_event_pointer

    event_ref = retain_event_pointer(
        journal=journal,
        event_pointer_sha256=recipe["store_preimages"]["event_pointer_ref"]["sha256"],
    )
    return {
        "schema_version": SCHEMA_V2 if recipe["schema_version"] == SCHEMA_V6 else SCHEMA,
        **(
            {"registered_event_declaration_ref": recipe["registered_event_declaration_ref"]}
            if recipe["schema_version"] == SCHEMA_V6
            else {}
        ),
        "trade_date": journal.trade_date,
        "request_ref": handoff["request_ref"],
        "maintenance_handoff_ref": recovered["handoff_ref"],
        "core_handoff_ref": handoff["core_handoff_ref"],
        "pool_manifest_ref": collected["pool_manifest_ref"],
        "theme_source_handoff_ref": collected["theme_source_handoff_ref"],
        "corporate_action_template_ref": recipe["corporate_action_template_ref"],
        "timing_policy_ref": recipe["research_timing"]["policy_ref"],
        "auxiliary_stage_refs": collected["auxiliary_stage_refs"],
        "event_pointer_ref": event_ref,
        "native_request_fields": fields,
    }


class SourceBundle:
    """Intrinsic source closure, not installed-release or whole-EOD admission."""

    def __init__(self, *, journal, document):
        self.journal = journal
        self.workspace = Path(journal.storage._io.workspace_root)
        self.files = ResearchFileReadback(str(self.workspace))
        self.document = parse_canonical_json_bytes(canonical_json_bytes(document))
        value = self.document
        registered = type(value) is dict and value.get("schema_version") == SCHEMA_V2
        fields = FIELDS | ({"registered_event_declaration_ref"} if registered else set())
        if (
            type(value) is not dict
            or set(value) != fields
            or value["schema_version"] not in {SCHEMA, SCHEMA_V2}
        ):
            raise DependencyInputError("CUTOFF_SOURCE_BUNDLE_SHAPE_INVALID")
        if value["trade_date"] != journal.trade_date:
            raise DependencyInputError("CUTOFF_SOURCE_BUNDLE_DAY_INVALID")
        for name in fields - {
            "schema_version",
            "trade_date",
            "native_request_fields",
            "auxiliary_stage_refs",
        }:
            if name != "theme_source_handoff_ref" or value[name] is not None:
                validate_ref(value[name])
        self.execution = journal.root / "executions" / value["request_ref"]["sha256"]
        raw = canonical_json_bytes(value)
        self.version = 2 if registered else 1
        self.reference = {
            "path": str(self.execution / f"inputs/acquired-sources.v{self.version}.json"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        existing = journal.storage.read(self.reference["path"])
        if (
            journal.storage.read(
                str(self.execution / f"inputs/acquired-sources.v{3 - self.version}.json")
            )
            is not None
        ):
            raise DependencyInputError("CUTOFF_SOURCE_BUNDLE_VERSION_CONFLICT")
        if existing is not None and existing.data != raw:
            raise DependencyInputError("CUTOFF_SOURCE_BUNDLE_CONFLICT")
        self.handoff = self.read(value["maintenance_handoff_ref"])
        validate_handoff_shape(self.handoff)
        if (
            value["maintenance_handoff_ref"]["path"]
            != str(self.execution / "maintenance-handoff.v1.json")
            or self.handoff["trade_date"] != journal.trade_date
            or self.handoff["request_ref"] != value["request_ref"]
            or self.handoff["core_handoff_ref"] != value["core_handoff_ref"]
        ):
            raise ContractError("CUTOFF_SOURCE_HANDOFF_BINDING_INVALID")
        for name, prefix in (
            ("request_ref", "request"),
            ("recipe_ref", "recipe"),
            ("market_pointer_ref", "market-pointer"),
        ):
            ref = validate_ref(self.handoff[name])
            if ref["path"] != str(self.execution / "inputs" / f"{prefix}-{ref['sha256']}.json"):
                raise DependencyInputError("CUTOFF_RETAINED_HANDOFF_PATH_INVALID")
        request = self.read(value["request_ref"])
        validate_production_request(request, release_install_ref=request["release_install_ref"])
        self.recipe = validate_execution_recipe(
            self.read(self.handoff["recipe_ref"]), request=request
        )
        if (
            self.recipe["schema_version"] != (SCHEMA_V6 if registered else SCHEMA_V5)
            or request["recipe_ref"]["sha256"] != self.handoff["recipe_ref"]["sha256"]
            or self.recipe["target_trade_date"] != journal.trade_date
            or self.recipe["corporate_action_template_ref"]
            != value["corporate_action_template_ref"]
            or self.recipe["research_timing"]["policy_ref"] != value["timing_policy_ref"]
            or self.handoff["release_ref"] != self.recipe["release_ref"]
            or self.handoff["release_install_ref"] != self.recipe["release_install_ref"]
        ):
            raise ContractError("CUTOFF_SOURCE_RECIPE_BINDING_INVALID")
        self.registered = None
        if registered:
            from scripts.registered_daily_event_sources import read_recipe_source

            if (
                value["registered_event_declaration_ref"]
                != self.recipe["registered_event_declaration_ref"]
            ):
                raise ContractError("CUTOFF_REGISTERED_DECLARATION_MISMATCH")
            self.registered = read_recipe_source(workspace=self.workspace, recipe=self.recipe)
            for reference in self.registered["source_refs"]:
                self.files.source_file(reference, code="CUTOFF_REGISTERED_SOURCE_INVALID")
            self.read(self.recipe["previous_completion_ref"])
        self.timing_policy = validate_research_timing_policy(self.read(value["timing_policy_ref"]))
        if self.timing_policy["mode"] != self.recipe["research_timing"]["mode"]:
            raise DependencyInputError("CUTOFF_TIMING_POLICY_MODE_MISMATCH")
        expected_handoffs = (
            {"cn-daily-maintenance-handoff.v2", "cn-daily-maintenance-handoff.v4"}
            if self.timing_policy["mode"] == CURRENT
            else {"cn-daily-maintenance-handoff.v3"}
        )
        if self.handoff["schema_version"] not in expected_handoffs:
            raise DependencyInputError("CUTOFF_HANDOFF_MODE_MISMATCH")
        if self.handoff["schema_version"] == "cn-daily-maintenance-handoff.v4":
            from .automatic_origin import read_automatic_origin, require_origin_execution

            if not registered:
                raise DependencyInputError("CUTOFF_AUTOMATIC_ORIGIN_PROFILE_INVALID")
            origin = read_automatic_origin(
                workspace=self.workspace, reference=self.handoff["automatic_origin_ref"]
            )
            require_origin_execution(
                origin,
                request_ref=value["request_ref"],
                request=request,
                recipe=self.recipe,
                retained=True,
                sealed_at=self.handoff["sealed_at"],
            )
            # Enrol every exact forward-chain byte in the existing cutoff
            # custody recheck, including the origin and original auto request.
            for source in (
                origin["sources"],
                origin["resolution"]["sources"],
                origin["bound"]["sources"],
            ):
                for path, raw in source.observed.items():
                    self.files.source_file(
                        {"path": path, "sha256": hashlib.sha256(raw).hexdigest()},
                        code="CUTOFF_AUTOMATIC_ORIGIN_SOURCE_INVALID",
                    )
        self.fields = value["native_request_fields"]
        if type(self.fields) is not dict or set(self.fields) != NATIVE_FIELDS:
            raise DependencyInputError("CUTOFF_NATIVE_FIELDS_INVALID")
        evidence = self.fields["company_evidence"]
        if type(evidence) is not dict or set(evidence) != {
            "exposure_rows",
            "fundamental_source",
            "macro_risk",
        }:
            raise DependencyInputError("CUTOFF_COMPANY_FIELDS_INVALID")
        if (
            self.fields["expected_trade_date"] != journal.trade_date
            or self.fields["strategy_id"] != self.recipe["strategy_id"]
            or self.fields["expected_factor_pointer_sha256"]
            != self.handoff["factor_pointer_ref"]["sha256"]
            or self.fields["policy"] != self.read(self.recipe["policy_refs"]["research"])
        ):
            raise ContractError("CUTOFF_NATIVE_CONTEXT_INVALID")
        self._verify_origins()

    def read(self, reference):
        _, raw, _ = self.files.source_file(reference, code="CUTOFF_SOURCE_REF_INVALID")
        return parse_research_json(raw, label="cutoff native source", canonical=False)

    def source_document(self, reference, *, code):
        return self.read(reference)

    def _verify_origins(self):
        value, fields, recipe = self.document, self.fields, self.recipe
        core = self.read(value["core_handoff_ref"])
        from .core_handoff import FIELDS as CORE_FIELDS
        from .core_pool import CORE_NODES

        if (
            type(core) is not dict
            or set(core) != CORE_FIELDS
            or core["schema_version"] != "cn-daily-core-handoff.v1"
            or core["trade_date"] != self.journal.trade_date
            or core["release_ref"] != self.handoff["release_ref"]
            or core["graph_sha256"] != self.handoff["graph_sha256"]
            or not _false_authority(core["authority"])
            or type(core["node_refs"]) is not dict
            or set(core["node_refs"]) != set(CORE_NODES)
        ):
            raise ContractError("CUTOFF_CORE_HANDOFF_INVALID")
        self.core_times = [self.handoff["sealed_at"]]
        for ref in core["node_refs"].values():
            terminal = self.read(ref)
            if terminal["state"] != "SUCCEEDED":
                raise DependencyInputError("CUTOFF_CORE_TERMINAL_INCOMPLETE")
            self.core_times.append(terminal["finished_at"])
        if value["core_handoff_ref"]["path"] != str(self.journal.root / "core-handoff.v1.json"):
            raise DependencyInputError("CUTOFF_CORE_PATH_INVALID")
        top = self.read(core["node_refs"]["top100"])
        if (
            top["state"] != "SUCCEEDED"
            or top["output_refs"]["manifest.json"] != value["pool_manifest_ref"]
        ):
            raise ContractError("CUTOFF_POOL_ORIGIN_INVALID")
        for alias, node in (("low", "low_observation"), ("w80", "w80_observation")):
            terminal = self.read(core["node_refs"][node])
            ref = {
                "path": fields[alias + "_observation_path"],
                "sha256": fields[alias + "_observation_sha256"],
            }
            if terminal["state"] != "SUCCEEDED" or terminal["output_refs"][alias.upper()] != ref:
                raise ContractError("CUTOFF_OBSERVATION_ORIGIN_INVALID")
            self.read(ref)
        industry_ref = recipe["research_sources"]["industry_source_ref"]
        expected_industry = None if industry_ref is None else self.read(industry_ref)
        if fields["industry_source"] != expected_industry:
            raise ContractError("CUTOFF_DECLARED_SOURCE_CHANGED")
        if recipe["theme_acquisition_ref"] is None:
            if (
                value["theme_source_handoff_ref"] is not None
                or self.read(recipe["research_sources"]["theme_source_ref"])
                != fields["theme_source"]
            ):
                raise DependencyInputError("CUTOFF_PINNED_THEME_MISMATCH")
        else:
            from .theme_handoff_readback import read_theme_handoff
            from .theme_handoff import SCHEMA_V3

            replay = read_theme_handoff(
                journal=self.journal,
                request_ref=value["request_ref"],
                core_handoff_ref=value["core_handoff_ref"],
                handoff_ref=value["theme_source_handoff_ref"],
            )
            if (
                replay["handoff"]["schema_version"] != SCHEMA_V3
                or replay["descriptor"] != fields["theme_source"]
            ):
                raise DependencyInputError("CUTOFF_ACQUIRED_THEME_MISMATCH")
            self.read(value["theme_source_handoff_ref"])
        _, _, enabled = split_theme_source(fields["theme_source"])
        if not enabled:
            raise DependencyInputError("CUTOFF_FOCUS_SOURCE_DECLARATION_REQUIRED")
        exposure_ref = recipe["research_sources"]["exposure_rows_ref"]
        expected_exposure = select_exposure_rows(
            None if exposure_ref is None else self.read(exposure_ref),
            recipe=recipe,
            workspace=str(self.workspace),
            pool_ref=value["pool_manifest_ref"],
            theme_source=fields["theme_source"],
            files=self.files,
        )
        if fields["company_evidence"]["exposure_rows"] != expected_exposure:
            raise ContractError("CUTOFF_DECLARED_SOURCE_CHANGED")
        stages = value["auxiliary_stage_refs"]
        if type(stages) is not dict or set(stages) != {"fundamental", "macro"}:
            raise DependencyInputError("CUTOFF_AUXILIARY_REFS_INVALID")
        for name, field in (("fundamental", "fundamental_source"), ("macro", "macro_risk")):
            declaration = recipe["research_sources"][name]
            if declaration["mode"] == "PINNED":
                if stages[name] is not None:
                    raise ContractError("CUTOFF_PINNED_STAGE_REF_FORBIDDEN")
                expected = self.read(declaration["source_ref"])
            else:
                stage_name = "FUNDAMENTAL" if name == "fundamental" else "MACRO_RELEASE"
                ref = validate_ref(stages[name])
                from pathlib import PurePosixPath

                attempt = PurePosixPath(self.handoff["maintenance_core_ref"]["path"]).parent
                if ref["path"] != str(attempt / ("stage-" + stage_name + ".json")):
                    raise DependencyInputError("CUTOFF_STAGE_ATTEMPT_MISMATCH")
                stage = self.read(ref)
                result = stage["result"]
                if (
                    stage["state"] != "STAGE_COMPLETED"
                    or result["stage"] != stage_name
                    or result["status"] not in {"READY", "NO_ACTION"}
                    or result["blockers"]
                ):
                    raise DependencyInputError("CUTOFF_AUXILIARY_INCOMPLETE")
                expected = self.read(result["evidence"]["research_source_ref"])
            if name == "fundamental":
                if type(expected) is not dict or set(expected) != {
                    "available_at",
                    "pointer",
                    "daily_parquet",
                }:
                    raise DependencyInputError("CUTOFF_FUNDAMENTAL_DESCRIPTOR_INVALID")
                expected = {
                    **expected,
                    "pointer": {
                        "path": str(
                            self.execution
                            / "inputs"
                            / ("fundamental-pointer-" + expected["pointer"]["sha256"] + ".json")
                        ),
                        "sha256": expected["pointer"]["sha256"],
                    },
                }
            if fields["company_evidence"][field] != expected:
                raise DependencyInputError("CUTOFF_AUXILIARY_SOURCE_MISMATCH")
        self.files.recheck()

    def payload(self, cutoff):
        utc_stamp(cutoff)
        value = {"as_of": cutoff, **self.fields}
        raw = canonical_json_bytes(value)
        digest = hashlib.sha256(raw).hexdigest()
        return (
            value,
            raw,
            {"path": str(self.execution / "inputs" / f"research-{digest}.json"), "sha256": digest},
        )

    def project(self, cutoff, *, current_macro):
        """Only native source projections; no context prepare/adapter publication methods."""
        if type(current_macro) is not bool:
            raise ContractError("CUTOFF_MACRO_MODE_INVALID")
        if self.timing_policy["mode"] == CURRENT and any(
            utc_stamp(stamp) > utc_stamp(cutoff) for stamp in self.core_times
        ):
            raise ContractError("CUTOFF_CORE_NOT_AVAILABLE")
        payload, raw, ref = self.payload(cutoff)
        context = ResearchSources(
            workspace=str(self.workspace),
            journal=self.journal,
            native_request_ref=ref,
            pool_manifest_ref=self.document["pool_manifest_ref"],
            release_ref=self.handoff["release_ref"],
            _request_bytes=raw,
        )
        for ref in (context.pool_ref, context.release_ref):
            self.read(ref)
        if context.focus_context is not None:
            for ref in context.focus_context["source_refs"]:
                self.files.source_file(ref, code="CUTOFF_FOCUS_PIT_REF_INVALID")
        values, hashes = {}, {}
        for node, field in (
            ("industry", "industry_source"),
            ("theme", "theme_source"),
            ("exposure", "exposure_rows"),
            ("fundamental", "fundamental_source"),
        ):
            recipe = source_recipe(
                node=node,
                field=field,
                request=payload,
                pool_ref=context.pool_ref,
                focus_context=context.focus_context,
            )
            if node == "exposure":
                recipe["source_completion_policy"] = POLICY
            projected = project_research_source(
                node=node,
                recipe=recipe,
                companies=context.companies,
                source_document=self.source_document,
                source_file=self.files.source_file,
                workspace=str(self.workspace),
                theme=values.get("theme", [None])[0],
                focus_membership=next(
                    (a for a in values.get("theme", []) if a["kind"] == MEMBERSHIP_KIND), None
                ),
            )
            if (
                projected is None
                or projected.state != NodeState.SUCCEEDED
                or not projected.artifacts
            ):
                raise DependencyInputError("CUTOFF_CRITICAL_SOURCE_INCOMPLETE:" + node)
            artifact_output_names(projected.artifacts)
            values[node] = projected.artifacts
            hashes[node] = hashlib.sha256(canonical_json_bytes(projected.artifacts)).hexdigest()
        macro, admission = self._macro(payload, context.rank, current=current_macro)
        artifact_output_names(macro)
        values["macro"] = macro
        hashes["macro"] = hashlib.sha256(
            canonical_json_bytes({"native_admission": admission, "artifacts": macro})
        ).hexdigest()
        context._verify_pool()
        from .prospective_sources import native_cutoff_source_times

        times = native_cutoff_source_times(
            request=payload,
            source_document=self.read,
            source_file=self.files.source_file,
            workspace=self.workspace,
            macro_admission=admission,
        )
        if self.document["theme_source_handoff_ref"] is not None:
            ref = self.document["theme_source_handoff_ref"]
            times.append(
                {
                    "role": "THEME_HANDOFF",
                    "subject_id": "ALL",
                    "source_ref": ref,
                    "original_time": self.read(ref)["sealed_at"],
                    "time_semantics": "LOCAL_CLOSURE",
                }
            )
        self.files.recheck()
        return {
            "artifacts": values,
            "projection_sha256s": hashes,
            "macro_admission": admission,
            "source_times": times,
        }

    def _macro(self, payload, rank, *, current):
        risk = payload["company_evidence"]["macro_risk"]
        if (
            type(risk) is not dict
            or set(risk) != {"classification", "source"}
            or risk["classification"] != "CANONICAL_MACRO_READY"
        ):
            raise DependencyInputError("CUTOFF_CRITICAL_SOURCE_INCOMPLETE:macro")
        closure = validate_macro_readiness_closure(
            workspace_root=self.workspace, closure=self.read(risk["source"])
        )
        from .completion_macro import _original_availability

        _original_availability(closure, trade_date=self.journal.trade_date, cutoff=payload["as_of"])
        if current:
            artifact, _, admission = unified._daily_macro_evidence(
                risk, self.files.source_file, self.workspace, rank, payload
            )
        else:
            admission = _readiness_projection(closure)
            artifact = build_market_risk_evidence(
                source_path=risk["source"]["path"],
                source_sha256=risk["source"]["sha256"],
                blocker_codes=[],
                classification="CANONICAL_MACRO_READY",
                as_of=payload["as_of"],
            )
        if artifact is None or admission is None:
            raise DependencyInputError("CUTOFF_MACRO_ADMISSION_MISSING")
        report = macro_freshness_from_closure(
            workspace=str(self.workspace),
            as_of=payload["as_of"],
            closure_ref=risk["source"],
            source_file=self.files.source_file,
            closure=closure,
        )
        if report["payload"]["critical_missing_codes"]:
            raise DependencyInputError("CUTOFF_CRITICAL_SOURCE_INCOMPLETE:macro")
        return [artifact, report], admission
