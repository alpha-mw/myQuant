"""Versioned existing corporate DAG node with immutable full-window reconciliation."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .corporate_actions import CorporateActionAdapter, CorporateActionEvidence
from .corporate_reconciliation import ReconciliationSources
from .registered_transition import RegisteredTransitionSources
from .daily_contract import ContractError, NodeState, utc_stamp, validate_ref
from .daily_journal import DailyJournal
from .daily_runner import Probe, NativeOutcome

SCHEMA = "cn-corporate-recipe.v2"
SCHEMA_V3 = "cn-corporate-recipe.v3"
EXTRA_FIELDS = {
    "corporate_action_context_ref",
    "decision_recipe_ref",
    "research_request_ref",
    "store_plan_ref",
}
LEGACY_FIELDS = {
    "event_pointer_ref",
    "previous_trade_date",
    "market_refs",
    "calendar_ref",
    "market_snapshot_ref",
}


def derive_corporate_projection(*, evidence, recipe):
    registered = recipe.get("schema_version") == SCHEMA_V3
    extra = {"registered_event_declaration_ref"} if registered else set()
    if set(recipe) != LEGACY_FIELDS | EXTRA_FIELDS | {
        "schema_version",
        "custody_at",
    } | extra or recipe["schema_version"] not in {SCHEMA, SCHEMA_V3}:
        raise ContractError("CORPORATE_RECIPE_V2_SHAPE_INVALID")
    custody = utc_stamp(recipe["custody_at"])
    if custody > datetime.now(timezone.utc):
        raise ContractError("CORPORATE_CUSTODY_IN_FUTURE")
    projection = (
        _registered_projection(evidence)
        if registered
        else CorporateActionEvidence.project(evidence)
    )
    if projection is None:
        return None
    sources = ReconciliationSources(
        evidence,
        context_ref=recipe["corporate_action_context_ref"],
        **{
            name: recipe[name]
            for name in ("decision_recipe_ref", "research_request_ref", "store_plan_ref")
        },
    )
    report = sources.report(recipe["custody_at"])
    raw = canonical_json_bytes(report)
    sha = hashlib.sha256(raw).hexdigest()
    journal = DailyJournal(str(evidence.workspace), evidence.trade_date)
    ref = {
        "path": str(journal.root / "corporate-actions" / f"reconciliation-{sha}.json"),
        "sha256": sha,
    }
    projection.update(
        schema_version="cn-corporate-financial-projection.v2",
        reconciliation_ref=ref,
        reconciliation_state=report["payload"]["summary_state"],
    )
    if registered:
        transition = RegisteredTransitionSources(
            sources, declaration_ref=recipe["registered_event_declaration_ref"]
        ).report(recipe["custody_at"])
        body = transition["payload"]
        if set(evidence.market_refs) != {r["symbol"] for r in body["position_rows"]}:
            raise ContractError("REGISTERED_TRANSITION_MARKET_UNION_MISMATCH")
        transition_sha = hashlib.sha256(canonical_json_bytes(transition)).hexdigest()
        transition_ref = {
            "path": str(
                journal.root / "corporate-actions" / f"registered-transition-{transition_sha}.json"
            ),
            "sha256": transition_sha,
        }
        projection.update(
            schema_version="cn-corporate-financial-projection.v3",
            registered_transition_ref=transition_ref,
            registered_transition_state="VALIDATED_REGISTERED_TRANSITION",
            registered_evidence_level=body["evidence_level"],
            broker_statement_verified=body["broker_statement_verified"],
            financial_admission_state=body["financial_admission_state"],
            risk_readiness_state=body["risk_readiness_state"],
        )
        return projection, report, ref, transition, transition_ref
    return projection, report, ref


def _registered_projection(evidence):
    from .corporate_actions import EVENT_ROOT
    from .daily_journal import FALSE_AUTHORITY

    events = evidence._events()
    if any(r["trade_date"].replace("-", "") == evidence.trade_date for r in events["closures"]):
        raise ContractError("REGISTERED_TRANSITION_EMPTY_EVENT_CONFLICT")
    generation = events["pointer"]["generation"]
    return {
        "trade_date": evidence.trade_date,
        "event_pointer_sha256": evidence.event_sha,
        "event_closure": None,
        "event_generation_ref": {
            "path": EVENT_ROOT + "/" + generation["path"],
            "sha256": events["generation_sha256"],
        },
        "financial_event_state": "REGISTERED_OWNER_DECLARED_FACTS",
        "adjustment_checks": evidence._adjustments(),
        "threshold_state": "SEE_PER_SYMBOL",
        "threshold_anchor_mutation": False,
        "authority": dict(FALSE_AUTHORITY),
    }


class CorporateReconciliationAdapter(CorporateActionAdapter):
    def __init__(
        self,
        *,
        corporate_action_context_ref,
        decision_recipe_ref,
        research_request_ref,
        store_plan_ref,
        registered_event_declaration_ref=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.binding = {
            "corporate_action_context_ref": validate_ref(corporate_action_context_ref),
            "decision_recipe_ref": validate_ref(decision_recipe_ref),
            "research_request_ref": validate_ref(research_request_ref),
            "store_plan_ref": validate_ref(store_plan_ref),
        }
        self.schema = SCHEMA
        if registered_event_declaration_ref is not None:
            self.binding["registered_event_declaration_ref"] = validate_ref(
                registered_event_declaration_ref
            )
            self.schema = SCHEMA_V3
        self.corporate_recipe = None

    def prepare(self):
        self.journal._require_lock()
        pointer_path = str(self.journal.root / "inputs" / f"event-pointer-{self.event_sha}.json")
        retained = self.journal.storage.read(pointer_path)
        if retained is None:
            super().prepare()
        else:
            if retained.byte_sha256 != self.event_sha:
                raise ContractError("CORPORATE_RETAINED_POINTER_SHA_MISMATCH")
            self.event_pointer_ref = {"path": pointer_path, "sha256": self.event_sha}
            self._events()
        legacy = {
            "event_pointer_ref": self.event_pointer_ref,
            "previous_trade_date": self.previous,
            "market_refs": self.market_refs,
            "calendar_ref": self.calendar_ref,
            "market_snapshot_ref": self.market_snapshot_ref,
        }
        identity = {**legacy, **self.binding, "schema_version": self.schema}
        key = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        version = 3 if self.schema == SCHEMA_V3 else 2
        path = str(self.journal.root / "inputs" / f"corporate-recipe-v{version}-{key}.json")
        # Read every source before sampling actual custody; no caller timestamp input.
        if version == 3:
            _registered_projection(self)
        else:
            CorporateActionEvidence.project(self)
        sources = ReconciliationSources(
            self,
            context_ref=self.binding["corporate_action_context_ref"],
            **{
                k: self.binding[k]
                for k in ("decision_recipe_ref", "research_request_ref", "store_plan_ref")
            },
        )
        transition = (
            RegisteredTransitionSources(
                sources, declaration_ref=self.binding["registered_event_declaration_ref"]
            )
            if version == 3
            else None
        )
        stored = self.journal.storage.read(path)
        if stored is None:
            self.corporate_recipe = {
                **identity,
                "custody_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            sources.report(self.corporate_recipe["custody_at"])
            if transition is not None:
                transition.report(self.corporate_recipe["custody_at"])
            raw = canonical_json_bytes(self.corporate_recipe)
            stored = self.journal.storage.write(path, raw)
        else:
            self.corporate_recipe = parse_canonical_json_bytes(stored.data)
            if set(self.corporate_recipe) != {*identity, "custody_at"} or any(
                self.corporate_recipe[k] != v for k, v in identity.items()
            ):
                raise ContractError("CORPORATE_RECIPE_V2_BINDING_INVALID")
            sources.report(self.corporate_recipe["custody_at"])
            if transition is not None:
                transition.report(self.corporate_recipe["custody_at"])
        self.recipe_ref = {"path": path, "sha256": stored.byte_sha256}

    def template(self):
        value = super().template()
        value["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return value

    def _documents(self, request):
        observed = {
            **request,
            "input_refs": {
                k: v for k, v in request["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != self.template() or self.corporate_recipe is None:
            raise ContractError("CORPORATE_REQUEST_MISMATCH")
        if self._bytes(self.recipe_ref) != canonical_json_bytes(self.corporate_recipe):
            raise ContractError("CORPORATE_RECIPE_BINDING_MISMATCH")
        self._bytes(self.release_ref)
        return derive_corporate_projection(evidence=self, recipe=self.corporate_recipe)

    def probe(self, request):
        documents = self._documents(request)
        if documents is None:
            return Probe(NativeOutcome(NodeState.BLOCKED, {}, "INPUT_MISSING"))
        projection, report, report_ref = documents[:3]
        raw = canonical_json_bytes(projection)
        sha = hashlib.sha256(raw).hexdigest()
        path = str(self.journal.root / "corporate-actions" / (sha + ".json"))
        stored = self.journal.storage.read(path)
        if stored is None:
            return Probe(None, safe_to_execute=True)
        retained = self.journal.storage.read(report_ref["path"])
        if stored.data != raw or retained is None or retained.data != canonical_json_bytes(report):
            raise ContractError("CORPORATE_RECONCILIATION_READBACK_CONFLICT")
        extra_outputs = {}
        if self.schema == SCHEMA_V3:
            transition, transition_ref = documents[3:]
            retained_transition = self.journal.storage.read(transition_ref["path"])
            if retained_transition is None or retained_transition.data != canonical_json_bytes(
                transition
            ):
                raise ContractError("REGISTERED_TRANSITION_READBACK_CONFLICT")
            extra_outputs["registered_transition"] = transition_ref
        blocked = projection["reconciliation_state"] == "CURRENT_FINANCIAL_CONFLICT"
        return Probe(
            NativeOutcome(
                NodeState.BLOCKED if blocked else NodeState.SUCCEEDED,
                {
                    "financial_events": {"path": path, "sha256": sha},
                    "event_generation": projection["event_generation_ref"],
                    "reconciliation": report_ref,
                    **extra_outputs,
                },
                "CORPORATE_ACTION_UNRECONCILED" if blocked else None,
            )
        )

    def execute(self, request):
        self.journal._require_lock()
        documents = self._documents(request)
        if documents is None:
            raise ContractError("CORPORATE_EVENT_CLOSURE_MISSING")
        projection, report, report_ref = documents[:3]
        if self.schema == SCHEMA_V3:
            transition, transition_ref = documents[3:]
            self.journal.storage.write(transition_ref["path"], canonical_json_bytes(transition))
        self.journal.storage.write(report_ref["path"], canonical_json_bytes(report))
        raw = canonical_json_bytes(projection)
        sha = hashlib.sha256(raw).hexdigest()
        self.journal.storage.write(
            str(self.journal.root / "corporate-actions" / (sha + ".json")), raw
        )


def corporate_report_is_late(*, inputs, report):
    """Ledger caller also replays the native node; this validates its timing binding."""
    from quant_investor.contracts import validate_artifact
    from .native_input_contract import validate_native_input_shape
    from quant_investor.intelligence.corporate_reconciliation import KIND, BLOCKERS
    from quant_investor.intelligence.corporate_report_contract import validate_company_rows

    validate_native_input_shape(inputs)
    if inputs["schema_version"] not in {
        "cn-daily-native-inputs.v4",
        "cn-daily-native-inputs.v5",
        "cn-daily-native-inputs.v6",
        "cn-daily-native-inputs.v7",
    }:
        if report is not None:
            raise ContractError("CORPORATE_UNEXPECTED_LEGACY_REPORT")
        return False
    value = validate_artifact(report, expected_kind=KIND)
    body = value["payload"]
    validate_company_rows(body["company_rows"], BLOCKERS)
    if any(
        body[name] != inputs[source]
        for name, source in (
            ("trade_date", "trade_date"),
            ("context_ref", "corporate_action_context_ref"),
            ("decision_recipe_ref", "decision_recipe_ref"),
            ("store_plan_ref", "store_plan_ref"),
        )
    ):
        raise ContractError("CORPORATE_LEDGER_REPORT_BINDING_INVALID")
    late = utc_stamp(body["custody_at"]) > utc_stamp(body["as_of"])
    if (
        body["timing_status"] != ("LATE_RECORDED" if late else "ON_TIME")
        or value["created_at"] != body["custody_at"]
        or body["prospective"] is not False
    ):
        raise ContractError("CORPORATE_LEDGER_TIMING_INVALID")
    return late
