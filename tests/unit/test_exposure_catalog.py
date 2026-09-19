"""Catalog grammar and native source replay, with explicit core/PIT fixture seams.

The separately recorded retained-D preview covers genuine native Factor/PIT/pool
inputs. These tests exercise collector/source-bundle and failure boundaries.
"""

from copy import deepcopy
import json

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.output import CommandError
from quant_investor.cli.unified import _daily_exposure_evidence
from quant_investor.intelligence._common import IntelligenceError
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.exposure_catalog import (
    SCHEMA,
    select_exposure_rows,
    validate_catalog,
)
from quant_investor.operations.research_timing import CURRENT, HISTORICAL
from quant_investor.operations.research_file_readback import ResearchFileReadback
from quant_investor.operations.dependency_diagnostics import DependencyInputError


def row(company="000001.SZ"):
    return {
        "company_code": company,
        "available_at": "2026-08-20T08:00:00Z",
        "source": {"path": "never-opened.json", "sha256": "a" * 64},
        "primary_theme_id": "TUSHARE_DC:synthetic",
        "source_type": "ANNUAL_REPORT",
        "source_page": 1,
        "theme_revenue_share": "0.5",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "envelope_extra",
        "rows_not_list",
        "row_extra",
        "duplicate",
        "company",
        "path",
        "sha",
        "time",
        "source_type",
        "page",
        "infinite",
        "range",
    ],
)
def test_bad_catalog_declarations_reject_without_a_source_reader(fault):
    value = {"schema_version": SCHEMA, "rows": [row()]}
    if fault == "version":
        value["schema_version"] = "unknown"
    elif fault == "envelope_extra":
        value["authority"] = False
    elif fault == "rows_not_list":
        value["rows"] = {}
    elif fault == "row_extra":
        value["rows"][0]["extra"] = 1
    elif fault == "duplicate":
        value["rows"].append(row())
    elif fault == "company":
        value["rows"][0]["company_code"] = "not-a-company"
    elif fault == "path":
        value["rows"][0]["source"]["path"] = "../escape.json"
    elif fault == "sha":
        value["rows"][0]["source"]["sha256"] = "bad"
    elif fault == "time":
        value["rows"][0]["available_at"] = "2026-08-20"
    elif fault == "source_type":
        value["rows"][0]["source_type"] = []
    elif fault == "page":
        value["rows"][0]["source_page"] = True
    elif fault == "infinite":
        value["rows"][0]["theme_revenue_share"] = "Infinity"
    else:
        value["rows"][0]["theme_revenue_share"] = "1.1"
    with pytest.raises(DependencyInputError) as caught:
        validate_catalog(value)
    assert caught.value.failure_code == "SCHEMA_MISMATCH"


def test_catalog_grammar_retains_unselected_future_and_missing_physical_source():
    value = {"schema_version": SCHEMA, "rows": [{**row(), "available_at": "2099-01-01T00:00:00Z"}]}
    assert validate_catalog(value) == value["rows"]


@pytest.mark.parametrize(
    "schema,mode,theme",
    [
        ("cn-daily-execute-recipe.v4", CURRENT, None),
        ("cn-daily-execute-recipe.v5", HISTORICAL, None),
        ("cn-daily-execute-recipe.v6", HISTORICAL, None),
        ("cn-daily-execute-recipe.v5", CURRENT, None),
    ],
)
def test_unsupported_profile_stops_before_cohort_or_source_read(schema, mode, theme):
    class NoRead:
        def source_file(self, *a, **k):
            pytest.fail("unsupported profile read a source")

    with pytest.raises(DependencyInputError):
        select_exposure_rows(
            {"schema_version": SCHEMA, "rows": [row()]},
            recipe={"schema_version": schema, "research_timing": {"mode": mode}},
            workspace="unused",
            pool_ref={},
            theme_source=theme,
            files=NoRead(),
        )


def test_legacy_list_is_not_silently_filtered_or_reinterpreted():
    value = [row(), row("999999.SZ")]
    assert (
        select_exposure_rows(
            value,
            recipe={},
            workspace="unused",
            pool_ref={},
            theme_source=None,
            files=None,
        )
        is value
    )


@pytest.mark.parametrize("registered", [False, True])
def test_collection_and_frozen_bundle_replay_bind_catalog_and_selected_times(
    tmp_path, monkeypatch, registered
):
    from _native_cutoff_sources_fixture import build
    from quant_investor.operations.research_source_bundle import SourceBundle, build_source_bundle

    case = build(tmp_path, monkeypatch, exposure_catalog=True, registered_buy=registered)
    journal, workspace = case["journal"], case["workspace"]
    reference = case["recovered"]["recipe"]["research_sources"]["exposure_rows_ref"]
    original_catalog = (workspace / reference["path"]).read_bytes()
    with journal.locked():
        bundle = build_source_bundle(
            journal=journal, recovered=case["recovered"], auxiliary={"stages": {}}
        )
    selected = bundle["native_request_fields"]["company_evidence"]["exposure_rows"]
    assert [r["company_code"] for r in selected] == sorted(case["ctx"].companies)
    sources = SourceBundle(journal=journal, document=bundle)
    result = sources.project("2026-08-28T13:30:00Z", current_macro=True)
    assert set(result["projection_sha256s"]) == {
        "theme",
        "industry",
        "exposure",
        "fundamental",
        "macro",
    }
    assert reference["path"] in sources.files.observed
    assert "unread-future-catalog-source.json" not in sources.files.observed
    assert all(
        r["source_ref"]["path"] != "unread-future-catalog-source.json"
        for r in result["source_times"]
    )
    assert all(r["original_time"] != "2099-01-01T00:00:00Z" for r in result["source_times"])
    assert (workspace / reference["path"]).read_bytes() == original_catalog
    assert not (workspace / sources.reference["path"]).exists()
    altered = deepcopy(bundle)
    altered["native_request_fields"]["company_evidence"]["exposure_rows"].pop()
    with pytest.raises(ContractError, match="CUTOFF_DECLARED_SOURCE_CHANGED"):
        SourceBundle(journal=journal, document=altered)
    for fault in ("future", "sha", "missing"):
        candidate = json.loads(original_catalog)
        used = next(r for r in candidate["rows"] if r["company_code"] in case["ctx"].companies)
        if fault == "future":
            used["available_at"] = "2099-01-01T00:00:00Z"
        elif fault == "sha":
            used["source"]["sha256"] = "0" * 64
        else:
            used["source"]["path"] = "missing-selected-source.json"
        files = ResearchFileReadback(str(workspace))
        chosen = select_exposure_rows(
            candidate,
            recipe=case["recovered"]["recipe"],
            workspace=str(workspace),
            pool_ref=case["ctx"].pool_ref,
            theme_source=bundle["native_request_fields"]["theme_source"],
            files=files,
        )
        assert used in chosen  # Never hide a bad selected declaration by filtering it out.
        with pytest.raises(IntelligenceError if fault == "future" else CommandError):
            _daily_exposure_evidence(chosen, {"as_of": "2026-08-28T13:30:00Z"}, files.source_file)
    pool_path = workspace / case["ctx"].pool_ref["path"]
    original_pool = pool_path.read_bytes()
    try:
        pool_path.write_bytes(original_pool + b"\n")
        with pytest.raises(CommandError, match="RESEARCH_FILE_CHANGED_DURING_READBACK"):
            sources.files.recheck()
    finally:
        pool_path.write_bytes(original_pool)
    changed = json.loads(original_catalog)
    changed["rows"][0]["available_at"] = "2098-01-01T00:00:00Z"
    (workspace / reference["path"]).write_bytes(canonical_json_bytes(changed))
    with pytest.raises(CommandError, match="RESEARCH_FILE_CHANGED_DURING_READBACK"):
        sources.files.recheck()
    with pytest.raises(DependencyInputError, match="RESEARCH_NATIVE_SOURCE_SHA_MISMATCH"):
        SourceBundle(journal=journal, document=bundle)
