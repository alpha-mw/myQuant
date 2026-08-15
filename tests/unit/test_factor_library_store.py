from __future__ import annotations

import json

import pytest

from quant_investor.factors.library import FactorLibraryPolicy, audit_factor_library
from quant_investor.factors.report import (
    build_factor_governance_dashboard_payload,
    render_factor_library_audit_markdown,
)
from quant_investor.factors.store import FactorLibraryAuditStore
from tests.unit.test_factor_library import _decision, _definition, _library, _validation_report


def _audit_report():
    definition = _definition()
    report = _validation_report(definition)
    decision = _decision(definition, report)
    library = _library(definition=definition, report=report, decision=decision)
    audit = audit_factor_library(
        library=library,
        definitions=[definition],
        admission_decisions=[decision],
        validation_reports=[report],
        policy=FactorLibraryPolicy(require_incremental_review=False),
        as_of="2026-04-28",
        generated_at="2026-04-28",
    )
    return library, audit


def test_append_and_read_audit_report(tmp_path) -> None:
    _library_obj, audit = _audit_report()
    store = FactorLibraryAuditStore(tmp_path / "audit")

    store.append_audit_report(audit)

    assert store.read_audit_reports()[0].to_dict() == audit.to_dict()
    assert store.get_audit_report_ids() == {audit.report_id}


def test_duplicate_audit_report_id_raises(tmp_path) -> None:
    _library_obj, audit = _audit_report()
    store = FactorLibraryAuditStore(tmp_path / "audit")
    store.append_audit_report(audit)

    with pytest.raises(ValueError, match="Duplicate report_id"):
        store.append_audit_report(audit)


def test_malformed_json_raises_clear_error(tmp_path) -> None:
    store = FactorLibraryAuditStore(tmp_path / "audit")
    store.audit_reports_path.parent.mkdir(parents=True, exist_ok=True)
    store.audit_reports_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        store.read_audit_reports()


def test_save_and_load_markdown_and_dashboard(tmp_path) -> None:
    library, audit = _audit_report()
    markdown = render_factor_library_audit_markdown(audit)
    dashboard = build_factor_governance_dashboard_payload(
        library=library,
        audit_report=audit,
    )
    store = FactorLibraryAuditStore(tmp_path / "audit")

    markdown_path = store.save_audit_markdown(markdown)
    dashboard_path = store.save_dashboard_payload(dashboard)

    assert markdown_path == store.audit_markdown_path
    assert dashboard_path == store.dashboard_payload_path
    assert store.load_audit_markdown() == markdown
    assert store.load_dashboard_payload()["verdict"] == audit.verdict
    json.dumps(store.load_dashboard_payload(), ensure_ascii=False, sort_keys=True)


def test_store_creates_directories_on_demand(tmp_path) -> None:
    root = tmp_path / "missing" / "audit"
    store = FactorLibraryAuditStore(root)
    _library_obj, audit = _audit_report()

    assert not root.exists()
    store.append_audit_report(audit)

    assert root.exists()
    assert store.audit_reports_path.exists()
