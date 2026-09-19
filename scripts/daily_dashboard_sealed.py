"""Capture-only v5 Dashboard node; serving publication requires a separate EOD seal."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.dashboard_evidence import DashboardEvidenceSources, DOMAINS
from quant_investor.operations.daily_contract import (
    ContractError,
    NodeState,
    validate_ref,
    utc_stamp,
)
from quant_investor.operations.daily_runner import NativeOutcome, Probe
from scripts.daily_dashboard_adapter import HistoricalDashboardAdapter

POLICY = "native-eod-first.v1"


class SealedDashboardAdapter(HistoricalDashboardAdapter):
    def __init__(
        self,
        *,
        authority_terminal_refs,
        publish_current_dashboard,
        corporate_terminal_ref=None,
        registered_event_declaration_ref=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(publish_current_dashboard) is not bool or set(authority_terminal_refs) != set(
            DOMAINS
        ):
            raise ContractError("DASHBOARD_SEALED_PROFILE_INVALID")
        self.historical_mode = not publish_current_dashboard
        self.authority_refs = {k: validate_ref(v) for k, v in authority_terminal_refs.items()}
        self.evidence_recipe_ref = None
        self.evidence_recipe = None
        if (corporate_terminal_ref is None) != (registered_event_declaration_ref is None):
            raise ContractError("DASHBOARD_REGISTERED_PROFILE_INCOMPLETE")
        self.registered_binding = (
            None
            if corporate_terminal_ref is None
            else {
                "corporate_terminal_ref": validate_ref(corporate_terminal_ref),
                "store_plan_ref": self.refs["store_plan"],
                "registered_event_declaration_ref": validate_ref(registered_event_declaration_ref),
            }
        )

    def registered_sources(self):
        if self.registered_binding is None:
            return None
        from quant_investor.operations.registered_dashboard import RegisteredDashboardSources

        return RegisteredDashboardSources(
            workspace=self.workspace,
            trade_date=self.journal.trade_date,
            release_ref=self.release_ref,
            store_terminal_ref=self.authority_refs["store"],
            **self.registered_binding,
        )

    def sources(self):
        return DashboardEvidenceSources(
            workspace=self.workspace,
            trade_date=self.journal.trade_date,
            release_ref=self.release_ref,
            terminal_refs=self.authority_refs,
        )

    def prepare(self):
        import json
        from quant_investor.strategy_records.close_plan_contracts import validate_plan

        plan = json.loads(self._source(self.refs["store_plan"]))
        if validate_plan(plan, path=self.refs["store_plan"]["path"]) != (
            2 if self.registered_binding else 1
        ):
            raise ContractError("DASHBOARD_REGISTERED_PLAN_PROFILE_MISMATCH")
        super().prepare()
        source = self.sources()
        registered = self.registered_sources()
        identity = {
            "schema_version": (
                "cn-daily-dashboard-evidence-recipe.v2"
                if registered
                else "cn-daily-dashboard-evidence-recipe.v1"
            ),
            **(self.registered_binding or {}),
            "trade_date": self.journal.trade_date,
            "release_ref": self.release_ref,
            "terminal_refs": self.authority_refs,
            "publication_policy": POLICY,
        }
        key = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        path = str(self.journal.root / "inputs" / f"dashboard-evidence-{key}.json")
        stored = self.journal.storage.read(path)
        if stored is None:
            value = {
                **identity,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            source.build(value["created_at"])
            if registered is not None:
                registered.result(value["created_at"])
            stored = self.journal.storage.write(path, canonical_json_bytes(value))
        else:
            value = parse_canonical_json_bytes(stored.data)
            if set(value) != {*identity, "created_at"} or any(
                value[k] != v for k, v in identity.items()
            ):
                raise ContractError("DASHBOARD_EVIDENCE_RECIPE_CONFLICT")
            if utc_stamp(value["created_at"]) > datetime.now(timezone.utc):
                raise ContractError("DASHBOARD_EVIDENCE_CUSTODY_FUTURE")
            if registered is not None:
                registered.result(value["created_at"])
        self.evidence_recipe = value
        self.evidence_recipe_ref = {"path": path, "sha256": stored.byte_sha256}

    def template(self):
        value = super().template()
        if self.evidence_recipe_ref is None:
            raise ContractError("DASHBOARD_EVIDENCE_NOT_PREPARED")
        value["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        value["input_refs"].update(
            {
                "daily_evidence_recipe": self.evidence_recipe_ref,
                **{"authority." + k: v for k, v in self.authority_refs.items()},
            }
        )
        if self.registered_binding is not None:
            value["input_refs"]["registered.corporate_terminal"] = self.registered_binding[
                "corporate_terminal_ref"
            ]
        return value

    def expected_registered(self):
        if self.registered_binding is None:
            return {}
        if self.evidence_recipe is None or self._source(
            self.evidence_recipe_ref
        ) != canonical_json_bytes(self.evidence_recipe):
            raise ContractError("DASHBOARD_EVIDENCE_RECIPE_CHANGED")
        value = self.registered_sources().result(self.evidence_recipe["created_at"])
        return {"registered_transition": value["registered_transition_ref"]}

    def expected_evidence(self):
        if self.evidence_recipe is None or self._source(
            self.evidence_recipe_ref
        ) != canonical_json_bytes(self.evidence_recipe):
            raise ContractError("DASHBOARD_EVIDENCE_RECIPE_CHANGED")
        value = self.sources().build(self.evidence_recipe["created_at"])
        raw = canonical_json_bytes(value)
        sha = hashlib.sha256(raw).hexdigest()
        return raw, {
            "path": str(self.journal.root / "dashboard" / f"daily-evidence-{sha}.json"),
            "sha256": sha,
        }

    def probe(self, request):
        extra = self.expected_registered()
        base = super().probe(request)
        if base.outcome is None:
            return base
        raw, ref = self.expected_evidence()
        stored = self.journal.storage.read(ref["path"])
        if stored is None:
            return Probe(None, recovery_only=True)
        if stored.data != raw:
            raise ContractError("DASHBOARD_EVIDENCE_CAPTURE_CONFLICT")
        return Probe(
            NativeOutcome(
                NodeState.SUCCEEDED, {**base.outcome.output_refs, "daily_evidence": ref, **extra}
            )
        )

    def execute(self, request):
        self.expected_registered()
        super().execute(request)
        raw, ref = self.expected_evidence()
        self.journal.storage.write(ref["path"], raw)
        self.expected_registered()
