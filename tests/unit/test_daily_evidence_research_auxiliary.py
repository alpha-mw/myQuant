"""Native exposure/Fundamental/Macro input semantics on generated fixtures."""

from copy import deepcopy
import hashlib
import pandas as pd
import pytest

from quant_investor.operations.research_sources import ResearchSources, ResearchSourceAdapter
from quant_investor.operations.journal_revisions import append_revision
from test_daily_evidence_research_sources import context, put


def changed(ctx, root, evidence, name):
    raw = deepcopy(ctx.request)
    raw["company_evidence"] = evidence
    return ResearchSources(
        workspace=str(root),
        journal=ctx.journal,
        native_request_ref=put(root, name, raw),
        pool_manifest_ref=ctx.pool_ref,
        release_ref=ctx.release_ref,
    )


def test_exposure_requires_sources_and_can_revise_partial_without_rewriting_theme(tmp_path):
    ctx, journal = context(tmp_path)
    with journal.locked():
        ctx.prepare()
        theme = ResearchSourceAdapter(ctx, "theme")
        t = ctx.template("theme")
        journal.begin(t)
        theme.execute(t)
        out = theme.probe(t).outcome
        terminal = journal.finish(t, state=out.state, output_refs=out.output_refs)
        exposure = ResearchSourceAdapter(ctx, "exposure")
        request = ctx.template("exposure")
        request["input_refs"]["upstream.theme"] = terminal["terminal_ref"]
        journal.begin(request)
        exposure.execute(request)
        missing = exposure.probe(request).outcome
        assert missing.state.value == "PARTIAL"
        prior = journal.finish(
            request,
            state=missing.state,
            output_refs=missing.output_refs,
            failure_code="INPUT_MISSING",
        )
        rows = []
        for company in ctx.companies:
            rows.append(
                {
                    "company_code": company,
                    "available_at": "2026-08-27T08:00:00Z",
                    "primary_theme_id": ctx.request["policy"]["payload"]["technology_theme_ids"][0],
                    "source": put(
                        tmp_path, f"company/{company}.json", {"synthetic": True, "ratio": "0.5"}
                    ),
                    "source_page": 1,
                    "source_type": "ANNUAL_REPORT",
                    "theme_revenue_share": "0.5",
                }
            )
        successor = changed(
            ctx, tmp_path, {"exposure_rows": rows, "fundamental_source": None}, "with-exposure.json"
        )
        successor.prepare()
        new = successor.template("exposure")
        new["input_refs"]["upstream.theme"] = terminal["terminal_ref"]
        append_revision(journal, new, expected_request_key=prior["request_key"])
        node = ResearchSourceAdapter(successor, "exposure")
        journal.begin(new)
        node.execute(new)
        assert node.probe(new).outcome.state.value == "SUCCEEDED"
        assert successor.template("theme") == t


def test_valid_fundamental_lag_is_accepted_but_future_availability_is_not(tmp_path):
    ctx, journal = context(tmp_path)
    columns = [
        "fin_roe",
        "fin_roa",
        "fin_debt_to_assets",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_fcf_to_profit",
        "fcf_to_price",
        "forecast_revision",
    ]
    frame = pd.DataFrame(
        [
            {"ts_code": company, "trade_date": "20260814", **{k: 0.1 + i / 1000 for k in columns}}
            for i, company in enumerate(ctx.companies)
        ]
    )
    path = tmp_path / "fundamental/fund-g1/daily.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    path.chmod(0o600)
    source = {
        "available_at": "2026-08-14T08:00:00Z",
        "daily_parquet": {
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "pointer": put(
            tmp_path,
            "fundamental/pointer.json",
            {
                "generation_id": "fund-g1",
                "status": "OK",
                "metadata": {"binding_aware_research_ready": True, "gate2_passed": True},
            },
        ),
    }
    successor = changed(
        ctx, tmp_path, {"exposure_rows": [], "fundamental_source": source}, "fund-request.json"
    )
    with journal.locked():
        successor.prepare()
        node = ResearchSourceAdapter(successor, "fundamental")
        request = successor.template("fundamental")
        node.execute(request)
        out = node.probe(request).outcome
        assert out.state.value == "SUCCEEDED" and len(out.output_refs) == 102
        source["available_at"] = "2026-08-29T08:00:00Z"
        future = changed(
            ctx, tmp_path, {"exposure_rows": [], "fundamental_source": source}, "future-fund.json"
        )
        future.prepare()
        with pytest.raises(Exception, match="NOT_AVAILABLE_AT_DECISION"):
            ResearchSourceAdapter(future, "fundamental").probe(future.template("fundamental"))


def test_native_macro_pipeline_veto_remains_partial(tmp_path):
    ctx, journal = context(tmp_path)
    ref = put(
        tmp_path,
        "macro-veto.json",
        {
            "schema_version": "cn-daily-maintenance-macro-write-veto.v1",
            "blockers": ["MACRO_RELEASE_CONTRACT_BLOCKED"],
        },
    )
    successor = changed(
        ctx,
        tmp_path,
        {
            "exposure_rows": [],
            "fundamental_source": None,
            "macro_risk": {"source": ref, "classification": "PIPELINE_DATA_VETO"},
        },
        "macro-request.json",
    )
    with journal.locked():
        successor.prepare()
        node = ResearchSourceAdapter(successor, "macro")
        request = successor.template("macro")
        node.execute(request)
        assert node.probe(request).outcome.state.value == "PARTIAL"
    assert (tmp_path / "macro-veto.json").exists()
