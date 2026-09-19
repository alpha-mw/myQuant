"""Real native two-scope sources and journals; outer Factor/PIT selection is isolated.

The PIT owner reader has separate native retained-source acceptance. These tests
exercise source projection, revenue qualification, adapters and immutable refs.
"""

from copy import deepcopy
from functools import partial
import hashlib

import pytest

from quant_investor.cli import unified
from quant_investor.intelligence.pcb_ai_hardware import (
    FOCUS_COMPANIES,
    FOCUS_SHA,
    MEMBERSHIP_KIND,
    EVIDENCE_KIND,
    build_focus_membership,
    build_focus_evidence,
    partition_exposure_evidence,
)
from quant_investor.intelligence.theme_sources import (
    THEME_SOURCE_V2,
    split_theme_source,
    descriptor_refs,
)
from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
from quant_investor.market.pit_universe import PITUniverseRecord, filter_symbols_by_pit_status
from quant_investor.market.tushare import (
    build_theme_provider_execution_plan,
    build_theme_partition_capture,
    build_theme_provider_capture,
    build_industry_membership_partition_capture,
    build_industry_membership_capture,
)
from quant_investor.operations.research_sources import ResearchSources, ResearchSourceAdapter
from quant_investor.operations.research_recipes import source_recipe, artifact_output_names
from quant_investor.operations.research_projection import project_research_source
from test_daily_evidence_research_sources import context, put
from test_unified_daily_intelligence_storage import approved_theme_policy_v2
from test_tushare_industry_capture_stable import _membership_plan, _member_row

DAY = "20260828"
AS_OF = "2026-08-28T13:30:00Z"


def theme_source(root, companies, prefix="focus", *, theme=None):
    theme = theme or TECHNOLOGY_THEME_IDS[0].split(":", 1)[1]
    plan = build_theme_provider_execution_plan(
        provider="TUSHARE_DC",
        trade_date=DAY,
        company_keyset=sorted(companies),
        document_observed_at="2026-08-28T13:00:00Z",
        created_at="2026-08-28T13:00:00Z",
    )
    parts = [
        build_theme_partition_capture(
            plan=plan,
            partition_ordinal=0,
            provider_request_id=prefix + "-registry",
            reported_count=1,
            rows=[
                {
                    "idx_type": "概念板块",
                    "level": "1",
                    "name": "synthetic PCB evidence",
                    "trade_date": DAY,
                    "ts_code": theme,
                }
            ],
            blocker_codes=[],
            captured_at="2026-08-28T13:01:00Z",
        )
    ]
    for index, company in enumerate(sorted(companies), 1):
        parts.append(
            build_theme_partition_capture(
                plan=plan,
                partition_ordinal=index,
                provider_request_id=prefix + "-" + company,
                reported_count=1,
                rows=[
                    {
                        "con_code": company,
                        "name": "synthetic company",
                        "trade_date": DAY,
                        "ts_code": theme,
                    }
                ],
                blocker_codes=[],
                captured_at="2026-08-28T13:01:00Z",
            )
        )
    capture = build_theme_provider_capture(
        plan=plan, partition_documents=parts, completed_at="2026-08-28T13:02:00Z"
    )
    return {
        "dc_plan": put(root, prefix + "/plan.json", plan),
        "dc_capture": put(root, prefix + "/capture.json", capture),
        "dc_partitions": [put(root, f"{prefix}/part-{i}.json", p) for i, p in enumerate(parts)],
        "tdx_plan": None,
        "tdx_capture": None,
        "tdx_partitions": [],
    }


def industry_source(root, companies):
    taxonomy, tax_capture, plan = _membership_plan()
    keys = plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    chosen = next(key for key in keys if key.endswith("=Y"))
    parts = []
    for index, key in enumerate(keys):
        code = key.split("|", 1)[0].split("=", 1)[1]
        flag = key.rsplit("=", 1)[1]
        rows = (
            [
                {**_member_row(l3_code=code, flag=flag), "ts_code": company}
                for company in sorted(companies)
            ]
            if key == chosen
            else []
        )
        parts.append(
            build_industry_membership_partition_capture(
                membership_plan=plan,
                taxonomy_plan=taxonomy,
                taxonomy_capture=tax_capture,
                partition_key=key,
                partition_ordinal=index,
                provider_request_id="focus-industry-" + str(index),
                reported_count=len(rows),
                rows=rows,
                captured_at="2026-08-11T07:31:00Z",
            )
        )
    capture = build_industry_membership_capture(
        membership_plan=plan,
        taxonomy_plan=taxonomy,
        taxonomy_capture=tax_capture,
        partition_documents=parts,
        completed_at="2026-08-11T07:32:00Z",
    )
    return {
        "taxonomy_plan": put(root, "industry/tax-plan.json", taxonomy),
        "taxonomy_capture": put(root, "industry/tax-capture.json", tax_capture),
        "membership_plan": put(root, "industry/plan.json", plan),
        "membership_capture": put(root, "industry/capture.json", capture),
        "membership_partitions": [
            put(root, f"industry/part-{i}.json", p) for i, p in enumerate(parts)
        ],
    }


def pit_context(root, trade_date=DAY):
    records = [
        PITUniverseRecord(
            symbol=company,
            list_date="20000101",
            source_list_status="L",
            observed_at="2026-08-20T07:00:00Z",
        )
        for company in FOCUS_COMPANIES
    ]
    statuses = filter_symbols_by_pit_status(
        FOCUS_COMPANIES, as_of=trade_date, records=records, required=True
    ).metadata["statuses"]
    ref = put(root, "fixture-pit/context.json", {"synthetic": True, "statuses": statuses})
    return {
        "trade_date": trade_date,
        "company_keyset": list(FOCUS_COMPANIES),
        "company_set_sha256": FOCUS_SHA,
        "pit_selection_ref": ref,
        "pit_generation_manifest_ref": ref,
        "pit_membership_ref": ref,
        "source_refs": [ref],
        "company_statuses": [statuses[c] for c in FOCUS_COMPANIES],
    }


def exposure_rows(root, companies):
    return [
        {
            "company_code": company,
            "available_at": "2026-08-27T08:00:00Z",
            "primary_theme_id": TECHNOLOGY_THEME_IDS[0],
            "source": put(root, f"facts/{company}.json", {"synthetic": True, "share": "0.5"}),
            "source_page": 1,
            "source_type": "ANNUAL_REPORT",
            "theme_revenue_share": "0.5",
        }
        for company in companies
    ]


def inventory(root):
    return {
        str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("missing_focus", [False, True])
def test_native_theme_and_exposure_emit_focus_report_and_replay(
    tmp_path, monkeypatch, missing_focus
):
    original, journal = context(tmp_path)
    pit = pit_context(tmp_path)
    request = deepcopy(original.request)
    request["theme_source"] = {
        "schema_version": THEME_SOURCE_V2,
        "pool": request["theme_source"],
        "pcb_ai_hardware": None if missing_focus else theme_source(tmp_path, FOCUS_COMPANIES),
    }
    request["industry_source"] = industry_source(tmp_path, [*original.companies, *FOCUS_COMPANIES])
    request["company_evidence"] = {
        "exposure_rows": exposure_rows(tmp_path, [*original.companies, *FOCUS_COMPANIES]),
        "fundamental_source": None,
    }
    owner_calls = []

    def native_pit(**kwargs):
        assert kwargs["generation_ref"] == original.rank["payload"]["factor_generation_ref"]
        assert kwargs["trade_date"] == DAY
        owner_calls.append(kwargs)
        return pit

    monkeypatch.setattr(
        "quant_investor.operations.research_recipes.read_bound_focus_pit", native_pit
    )
    ctx = ResearchSources(
        workspace=str(tmp_path),
        journal=journal,
        native_request_ref=put(tmp_path, "focus-request.json", request),
        pool_manifest_ref=original.pool_ref,
        release_ref=original.release_ref,
    )
    before_pool = {
        p.name: p.read_bytes() for p in (tmp_path / ctx.pool_ref["path"]).parent.iterdir()
    }
    with journal.locked():
        ctx.prepare()
        theme = ResearchSourceAdapter(ctx, "theme")
        t = ctx.template("theme")
        journal.begin(t)
        theme.execute(t)
        theme_out = theme.probe(t).outcome
        assert theme_out.state.value == "SUCCEEDED"
        assert MEMBERSHIP_KIND in theme_out.output_refs
        terminal = journal.finish(t, state=theme_out.state, output_refs=theme_out.output_refs)
        ordinary_theme = ctx._read(theme_out.output_refs["artifact"])
        source_doc = partial(unified._daily_source_document, str(tmp_path))
        original_theme = unified._daily_theme_projection(
            {
                "as_of": AS_OF,
                "policy": request["policy"],
                "theme_source": request["theme_source"]["pool"],
            },
            ctx.companies,
            source_doc,
        )
        assert ordinary_theme == original_theme
        e = ctx.template("exposure")
        e["input_refs"]["upstream.theme"] = terminal["terminal_ref"]
        exposure = ResearchSourceAdapter(ctx, "exposure")
        journal.begin(e)
        exposure.execute(e)
        output = exposure.probe(e).outcome
        report = ctx._read(output.output_refs[EVIDENCE_KIND])
        assert output.state.value == ("PARTIAL" if missing_focus else "SUCCEEDED")
        assert report["payload"]["completion_state"] == (
            "PARTIAL_WITH_EXPLICIT_MISSING" if missing_focus else "SUCCEEDED"
        )
        rows = report["payload"]["company_rows"]
        assert [row["company_code"] for row in rows] == list(FOCUS_COMPANIES)
        assert not any(row["in_top100"] for row in rows)
        assert all(
            row["confidence"]["category"]
            == ("SOURCE_ONLY" if missing_focus else "COMPLETE_SOURCE_BOUND")
            for row in rows
        )
        assert all(
            (
                row["economic_exposure"] is None
                if missing_focus
                else row["economic_exposure"]["economic_exposure_state"] == "HIGH"
            )
            for row in rows
        )
        assert all(not v for v in report["payload"]["authority"].values())
        # Exact same recipe/projection implementation is used by completed replay.
        recipe = source_recipe(
            node="exposure",
            field="exposure_rows",
            request=request,
            pool_ref=ctx.pool_ref,
            focus_context=pit,
        )
        replay = project_research_source(
            node="exposure",
            recipe=recipe,
            companies=ctx.companies,
            source_document=source_doc,
            source_file=partial(unified._daily_source_file, tmp_path),
            workspace=str(tmp_path),
            theme=ordinary_theme,
            focus_membership=ctx._read(theme_out.output_refs[MEMBERSHIP_KIND]),
        )
        assert artifact_output_names(replay.artifacts) == [
            name for name in output.output_refs if name != "capture"
        ]
        for value, name in zip(replay.artifacts, artifact_output_names(replay.artifacts)):
            assert value == ctx._read(output.output_refs[name])
    before = inventory(tmp_path)
    monkeypatch.setattr(
        journal.storage, "write", lambda *a, **k: pytest.fail("replay writer called")
    )
    assert exposure.probe(e).outcome == output
    assert inventory(tmp_path) == before
    assert before_pool == {
        p.name: p.read_bytes() for p in (tmp_path / ctx.pool_ref["path"]).parent.iterdir()
    }
    assert len(owner_calls) == 1


def test_source_wrapper_is_exact_and_union_rejects_duplicate_or_unknown(tmp_path):
    source = theme_source(tmp_path, list(FOCUS_COMPANIES))
    wrapper = {"schema_version": THEME_SOURCE_V2, "pool": source, "pcb_ai_hardware": None}
    assert split_theme_source(wrapper) == (source, None, True)
    with pytest.raises(ValueError):
        split_theme_source({**wrapper, "unexpected": True})
    with pytest.raises(ValueError):
        split_theme_source({**wrapper, "schema_version": "unknown"})
    values = {"as_of": AS_OF}
    facts = unified._daily_exposure_evidence(
        exposure_rows(tmp_path, FOCUS_COMPANIES),
        values,
        partial(unified._daily_source_file, tmp_path),
    )
    with pytest.raises(ValueError):
        partition_exposure_evidence([facts[0], facts[0]], [])
    unknown = unified._daily_exposure_evidence(
        exposure_rows(tmp_path, ["000999.SZ"]),
        values,
        partial(unified._daily_source_file, tmp_path),
    )
    with pytest.raises(ValueError):
        partition_exposure_evidence(unknown, [])


def test_focus_overlap_and_missing_pit_do_not_become_qualified_exposure(tmp_path):
    policy = approved_theme_policy_v2()
    source_doc = partial(unified._daily_source_document, str(tmp_path))
    focus_source = theme_source(tmp_path, list(FOCUS_COMPANIES), "same")
    native = unified._daily_theme_projection(
        {"as_of": AS_OF, "policy": policy, "theme_source": focus_source},
        list(FOCUS_COMPANIES),
        source_doc,
    )
    pit = pit_context(tmp_path)
    ref = put(tmp_path, "pool.json", {"synthetic": True})
    member = build_focus_membership(
        as_of=AS_OF,
        pool_manifest_ref=ref,
        pit=pit,
        pool_theme=native,
        focus_theme=native,
        pool_source_refs=descriptor_refs(focus_source),
        focus_source_refs=descriptor_refs(focus_source),
    )
    assert member["payload"]["completion_state"] == "SUCCEEDED"
    assert all(row["in_top100"] for row in member["payload"]["company_rows"])
    conflicting_source = theme_source(
        tmp_path, list(FOCUS_COMPANIES), "different", theme="BK3001.DC"
    )
    conflicting = unified._daily_theme_projection(
        {"as_of": AS_OF, "policy": policy, "theme_source": conflicting_source},
        list(FOCUS_COMPANIES),
        source_doc,
    )
    conflict_report = build_focus_membership(
        as_of=AS_OF,
        pool_manifest_ref=ref,
        pit=pit,
        pool_theme=native,
        focus_theme=conflicting,
        pool_source_refs=descriptor_refs(focus_source),
        focus_source_refs=descriptor_refs(conflicting_source),
    )
    assert conflict_report["payload"]["completion_state"] == "PARTIAL_WITH_EXPLICIT_MISSING"
    assert all(
        row["membership"] is None and "FOCUS_POOL_MEMBERSHIP_CONFLICT" in row["missing_codes"]
        for row in conflict_report["payload"]["company_rows"]
    )
    pit["company_statuses"][0].update(
        in_universe=False, research_eligible=False, reason="missing_pit_record"
    )
    member = build_focus_membership(
        as_of=AS_OF,
        pool_manifest_ref=ref,
        pit=pit,
        pool_theme=native,
        focus_theme=native,
        pool_source_refs=descriptor_refs(focus_source),
        focus_source_refs=descriptor_refs(focus_source),
    )
    facts = unified._daily_exposure_evidence(
        exposure_rows(tmp_path, FOCUS_COMPANIES),
        {"as_of": AS_OF},
        partial(unified._daily_source_file, tmp_path),
    )
    report = build_focus_evidence(
        membership=member,
        pit=pit,
        focus_theme=native,
        industry=None,
        evidence=facts,
        industry_source_refs=[],
        daily_policy=policy,
    )
    row = report["payload"]["company_rows"][0]
    assert row["membership"] is None and row["economic_exposure"] is None
    assert row["source_evidence"]["unqualified_company_evidence_refs"]


def test_focus_facts_preserve_ordinary_exposure_and_compiler_input(tmp_path, monkeypatch):
    from quant_investor.operations.daily_contract import NodeState

    policy = approved_theme_policy_v2()
    pool = ["000001.SZ"]
    pool_source = theme_source(tmp_path, pool, "ordinary")
    focus_source = theme_source(tmp_path, list(FOCUS_COMPANIES), "special")
    wrapper = {
        "schema_version": THEME_SOURCE_V2,
        "pool": pool_source,
        "pcb_ai_hardware": focus_source,
    }
    source_doc = partial(unified._daily_source_document, str(tmp_path))
    source_file = partial(unified._daily_source_file, tmp_path)
    theme = unified._daily_theme_projection(
        {"as_of": AS_OF, "policy": policy, "theme_source": pool_source}, pool, source_doc
    )
    focus = unified._daily_theme_projection(
        {"as_of": AS_OF, "policy": policy, "theme_source": focus_source},
        list(FOCUS_COMPANIES),
        source_doc,
    )
    pit = pit_context(tmp_path)
    ref = put(tmp_path, "ordinary-pool.json", {"synthetic": True})
    member = build_focus_membership(
        as_of=AS_OF,
        pool_manifest_ref=ref,
        pit=pit,
        pool_theme=theme,
        focus_theme=focus,
        pool_source_refs=descriptor_refs(pool_source),
        focus_source_refs=descriptor_refs(focus_source),
    )
    rows = exposure_rows(tmp_path, [*pool, *FOCUS_COMPANIES])
    base = {"as_of": AS_OF, "policy": policy, "pool_ref": ref, "exposure_rows": rows[:1]}
    kwargs = {
        "node": "exposure",
        "companies": pool,
        "source_document": source_doc,
        "source_file": source_file,
        "workspace": str(tmp_path),
        "theme": theme,
    }
    original = project_research_source(**kwargs, recipe=base)
    combined = project_research_source(
        **kwargs,
        recipe={
            **base,
            "exposure_rows": rows,
            "focus_context": pit,
            "focus_theme_source": wrapper,
            "focus_industry_source": None,
        },
        focus_membership=member,
    )
    assert original.state == NodeState.SUCCEEDED
    assert combined.artifacts[: len(original.artifacts)] == original.artifacts
    assert combined.state == NodeState.PARTIAL  # Only focus Industry is missing.
    monkeypatch.setattr(unified, "_daily_fundamental_source", lambda *a, **k: (None, None))
    monkeypatch.setattr(unified, "_daily_macro_evidence", lambda *a, **k: (None, None, None))
    rank = {"payload": {"pool_rows": [{"symbol": pool[0]}]}}
    values = {"as_of": AS_OF}
    compiled = unified._daily_company_evidence(
        {"exposure_rows": rows, "fundamental_source": None},
        values,
        source_file,
        tmp_path,
        rank,
        focus_scope=True,
    )
    assert compiled[0] == original.artifacts[1:]
