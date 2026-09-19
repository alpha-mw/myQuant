"""Immutable Decision v2 report/capture around the unchanged native compilation."""

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import os

from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.intelligence.decision_report import REPORT_KIND, build_decision_report
from quant_investor.intelligence.portfolio_state import STRATEGY_ID
from quant_investor.intelligence.storage import _publish_exact_policy_artifact
from quant_investor.system.storage import SecureSystemStorage, SystemNotFound
from .daily_contract import ContractError, utc_stamp, validate_ref
from .daily_journal import FALSE_AUTHORITY, _false_authority, request_identity
from .decision_recipe import read_decision_recipe
from .decision_sources import DecisionSources
from .research_capture import ResearchCapture

CAPTURE_SCHEMA = "cn-daily-decision-report-capture.v2"
CAPTURE_FIELDS = frozenset(
    {
        "schema_version",
        "request_key",
        "decision_recipe_ref",
        "native_capture_ref",
        "native_result_ref",
        "portfolio_state_ref",
        "report_ref",
        "research_cutoff",
        "captured_at",
        "authority",
    }
)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DecisionReportPublication:
    def __init__(self, *, journal, recipe_ref, research_request_ref, store_plan_ref, context=None):
        self.journal = journal
        self.workspace = journal.storage._io.workspace_root
        self.recipe_ref = validate_ref(recipe_ref)
        self.request_ref = validate_ref(research_request_ref)
        self.plan_ref = validate_ref(store_plan_ref)
        self.context = context
        self.reader = SecureSystemStorage(self.workspace)
        self.path = (
            f"results/intelligence/decision/{STRATEGY_ID}/{journal.trade_date}/decision.v2.json"
        )

    def _context(self, request, captured):
        if request["input_refs"].get("decision_recipe") != self.recipe_ref:
            raise ContractError("DECISION_REPORT_REQUEST_PROFILE_MISMATCH")
        bound = read_decision_recipe(
            workspace=self.workspace,
            trade_date=self.journal.trade_date,
            recipe_ref=self.recipe_ref,
            research_request_ref=self.request_ref,
            store_plan_ref=self.plan_ref,
        )
        sources = DecisionSources(
            workspace=self.workspace,
            trade_date=self.journal.trade_date,
            request=request,
            result=captured[1],
            portfolio_state_ref=bound["recipe"]["portfolio_source_ref"],
            portfolio=bound["portfolio"],
            context=self.context,
        )
        inputs = sources.collect()
        kwargs = dict(
            result=captured[1],
            result_ref=captured[0]["result_ref"],
            portfolio_state=bound["portfolio"],
            portfolio_state_ref=bound["recipe"]["portfolio_source_ref"],
            **inputs,
        )
        return bound, sources, kwargs

    def _report(self):
        path = PurePosixPath(self.path)
        try:
            parent = self.reader._open_source_directory(path.parts[:-1])
        except SystemNotFound:
            return None
        try:
            self.reader._reject_casefold_alias(parent, path.name)
            stored = self.reader._read_leaf(parent, path.name, relative_path=path, optional=True)
        finally:
            os.close(parent)
        if stored is None:
            return None
        value = validate_artifact(stored.data, expected_kind=REPORT_KIND)
        return value, {"path": self.path, "sha256": stored.byte_sha256}

    def capture_path(self, request):
        key = request_identity(request)[1]
        return str(self.journal.root / "research" / "decision-reports" / key / "capture.v2.json")

    def _capture_binding(self, request, captured, bound, report_ref):
        capture_path = ResearchCapture(self.journal)._root(self.request_ref) + "/capture.v1.json"
        native = self.journal.storage.read(capture_path)
        if native is None or parse_canonical_json_bytes(native.data) != captured[0]:
            raise ContractError("DECISION_REPORT_NATIVE_CAPTURE_MISMATCH")
        return {
            "schema_version": CAPTURE_SCHEMA,
            "request_key": request_identity(request)[1],
            "decision_recipe_ref": self.recipe_ref,
            "native_capture_ref": {"path": capture_path, "sha256": native.byte_sha256},
            "native_result_ref": captured[0]["result_ref"],
            "portfolio_state_ref": bound["recipe"]["portfolio_source_ref"],
            "report_ref": report_ref,
            "research_cutoff": bound["recipe"]["as_of"],
            "authority": FALSE_AUTHORITY,
        }

    def probe(self, request, captured):
        report = self._report()
        side = self.journal.storage.read(self.capture_path(request))
        if report is None:
            if side is not None:
                raise ContractError("DECISION_REPORT_CAPTURE_WITHOUT_REPORT")
            return None
        bound, sources, kwargs = self._context(request, captured)
        rebuilt = build_decision_report(created_at=report[0]["created_at"], **kwargs)
        if rebuilt != report[0]:
            raise ContractError("DECISION_REPORT_IMMUTABLE_CONFLICT")
        if side is None:
            return None
        value = parse_canonical_json_bytes(side.data)
        expected = self._capture_binding(request, captured, bound, report[1])
        if (
            type(value) is not dict
            or set(value) != CAPTURE_FIELDS
            or {k: v for k, v in value.items() if k != "captured_at"} != expected
            or not _false_authority(value["authority"])
        ):
            raise ContractError("DECISION_REPORT_CAPTURE_BINDING_INVALID")
        if utc_stamp(value["captured_at"]) < max(
            utc_stamp(report[0]["created_at"]),
            utc_stamp(bound["portfolio"]["created_at"]),
            utc_stamp(captured[0]["captured_at"]),
        ):
            raise ContractError("DECISION_REPORT_CAPTURE_TIME_INVALID")
        terminal = self.journal.readonly_inspect(request).get("terminal")
        if terminal is not None and utc_stamp(value["captured_at"]) > utc_stamp(
            terminal["finished_at"]
        ):
            raise ContractError("DECISION_REPORT_CAPTURE_AFTER_TERMINAL")
        sources.recheck()
        return {
            "decision.v2.json": report[1],
            "decision_report_capture": {
                "path": self.capture_path(request),
                "sha256": side.byte_sha256,
            },
        }

    def execute(self, request, captured):
        self.journal._require_lock()
        if self.probe(request, captured) is not None:
            return
        bound, sources, kwargs = self._context(request, captured)
        previous = self._report()
        report = build_decision_report(
            created_at=_now() if previous is None else previous[0]["created_at"], **kwargs
        )
        published = _publish_exact_policy_artifact(
            root=Path(self.workspace),
            relative_path=self.path,
            artifact=report,
            validator=lambda raw: validate_artifact(raw, expected_kind=REPORT_KIND),
            parent_parts=("intelligence", "decision", STRATEGY_ID, self.journal.trade_date),
        )
        report_ref = {"path": published["path"], "sha256": published["sha256"]}
        sources.recheck()
        side = {
            **self._capture_binding(request, captured, bound, report_ref),
            "captured_at": _now(),
        }
        self.journal.storage.write(self.capture_path(request), canonical_json_bytes(side))
        if self.probe(request, captured) is None:
            raise ContractError("DECISION_REPORT_POSTWRITE_REPLAY_MISSING")
