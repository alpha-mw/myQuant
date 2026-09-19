"""Native source-projector adapters; Factor input is a synthetic sealed rank fixture."""

import hashlib

from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence.storage import (
    DailyResearchPoolStore,
    publish_theme_policy_v2,
    approved_theme_policy_v2,
)
from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
from quant_investor.market.tushare import (
    build_theme_provider_execution_plan,
    build_theme_partition_capture,
    build_theme_provider_capture,
)
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.research_sources import ResearchSources, ResearchSourceAdapter
from test_unified_daily_intelligence_storage import _pool_rank


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def context(root, *, pool_manifest_ref=None, release_ref=None):
    publish_theme_policy_v2(root)
    policy = approved_theme_policy_v2()
    if pool_manifest_ref is None:
        signal_date = "20260828"
        rank = _pool_rank(root, policy, signal_date=signal_date)
        policy_sha = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
        pool = DailyResearchPoolStore(root).publish(
            rank=rank,
            expected_policy_sha256=policy_sha,
            policy_path="results/policies/research/aggressive_tech_manufacturing/v2.json",
            before_publish=lambda: None,
        )
    else:
        import json

        manifest_path = root / pool_manifest_ref["path"]
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pool_manifest_ref["sha256"]:
            raise ValueError("native fixture pool manifest SHA differs")
        manifest = json.loads(raw)
        rank_raw = (manifest_path.parent / "factor_research_rank.json").read_bytes()
        if hashlib.sha256(rank_raw).hexdigest() != manifest["payload"]["rank_byte_sha256"]:
            raise ValueError("native fixture rank SHA differs")
        rank = json.loads(rank_raw)
        signal_date = manifest["payload"]["signal_date"]
        pool = {
            "manifest_path": pool_manifest_ref["path"],
            "manifest_sha256": pool_manifest_ref["sha256"],
        }
    iso = f"{signal_date[:4]}-{signal_date[4:6]}-{signal_date[6:]}"
    prefix = "" if pool_manifest_ref is None else "synthetic-research/" + signal_date + "/"

    def emit(name, value):
        return put(root, prefix + name, value)

    companies = sorted(r["symbol"] for r in rank["payload"]["pool_rows"])
    plan = build_theme_provider_execution_plan(
        provider="TUSHARE_DC",
        trade_date=signal_date,
        company_keyset=companies,
        document_observed_at=iso + "T13:00:00Z",
        created_at=iso + "T13:00:00Z",
    )
    theme = TECHNOLOGY_THEME_IDS[0].split(":", 1)[1]
    parts = [
        build_theme_partition_capture(
            plan=plan,
            partition_ordinal=0,
            provider_request_id="synthetic-registry",
            reported_count=1,
            rows=[
                {
                    "idx_type": "概念板块",
                    "level": "1",
                    "name": "fixture theme",
                    "trade_date": signal_date,
                    "ts_code": theme,
                }
            ],
            blocker_codes=[],
            captured_at=iso + "T13:01:00Z",
        )
    ]
    for index, company in enumerate(companies, 1):
        parts.append(
            build_theme_partition_capture(
                plan=plan,
                partition_ordinal=index,
                provider_request_id="synthetic-" + company,
                reported_count=1,
                rows=[
                    {
                        "con_code": company,
                        "name": "fixture company",
                        "trade_date": signal_date,
                        "ts_code": theme,
                    }
                ],
                blocker_codes=[],
                captured_at=iso + "T13:01:00Z",
            )
        )
    capture = build_theme_provider_capture(
        plan=plan, partition_documents=parts, completed_at=iso + "T13:02:00Z"
    )
    request = {
        "as_of": iso + "T13:30:00Z",
        "strategy_id": "aggressive_tech_manufacturing",
        "policy": policy,
        "expected_factor_pointer_sha256": rank["payload"]["factor_pointer_sha256"],
        "industry_source": None,
        "theme_source": {
            "dc_plan": emit("sources/plan.json", plan),
            "dc_capture": emit("sources/capture.json", capture),
            "dc_partitions": [emit(f"sources/part-{i}.json", p) for i, p in enumerate(parts)],
            "tdx_plan": None,
            "tdx_capture": None,
            "tdx_partitions": [],
        },
    }
    if pool_manifest_ref is not None:
        for alias in ("LOW", "W80"):
            path = (
                f"results/factors/observations/{signal_date[:4]}/"
                f"{signal_date[4:6]}/{signal_date[6:]}/{alias}.json"
            )
            request[alias.lower() + "_observation_path"] = path
            request[alias.lower() + "_observation_sha256"] = hashlib.sha256(
                (root / path).read_bytes()
            ).hexdigest()
    journal = DailyJournal(str(root), signal_date)
    ctx = ResearchSources(
        workspace=str(root),
        journal=journal,
        native_request_ref=emit("source-request.json", request),
        pool_manifest_ref={"path": pool["manifest_path"], "sha256": pool["manifest_sha256"]},
        release_ref=release_ref or emit("release.json", {"synthetic": True}),
    )
    return ctx, journal


def test_native_dc_projection_is_separate_from_missing_industry(tmp_path):
    ctx, journal = context(tmp_path)
    with journal.locked():
        ctx.prepare()
        industry = ResearchSourceAdapter(ctx, "industry")
        assert industry.probe(ctx.template("industry")).outcome.state.value == "BLOCKED"
        theme = ResearchSourceAdapter(ctx, "theme")
        request = ctx.template("theme")
        assert theme.probe(request).safe_to_execute
        theme.execute(request)
        result = theme.probe(request)
        assert result.outcome.state.value == "SUCCEEDED"
        refs = result.outcome.output_refs
        before = {r["path"]: (tmp_path / r["path"]).read_bytes() for r in refs.values()}
        assert theme.probe(request).outcome.output_refs == refs
        assert before == {r["path"]: (tmp_path / r["path"]).read_bytes() for r in refs.values()}


def test_native_industry_adapter_replays_complete_membership_sources(tmp_path):
    from quant_investor.market.tushare import (
        build_industry_membership_partition_capture,
        build_industry_membership_capture,
    )
    from test_tushare_industry_capture_stable import _membership_plan, _member_row

    ctx, journal = context(tmp_path)
    taxonomy_plan, taxonomy_capture, membership_plan = _membership_plan()
    parts = []
    for ordinal, key in enumerate(
        membership_plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    ):
        code = key.split("|", 1)[0].split("=", 1)[1]
        flag = key.rsplit("=", 1)[1]
        rows = (
            [
                {**_member_row(l3_code=code, flag=flag), "ts_code": company}
                for company in ctx.companies
            ]
            if ordinal == 0
            else []
        )
        parts.append(
            build_industry_membership_partition_capture(
                membership_plan=membership_plan,
                taxonomy_plan=taxonomy_plan,
                taxonomy_capture=taxonomy_capture,
                partition_key=key,
                partition_ordinal=ordinal,
                provider_request_id=f"synthetic-industry-{ordinal}",
                reported_count=len(rows),
                rows=rows,
                captured_at="2026-08-11T07:31:00Z",
            )
        )
    captured = build_industry_membership_capture(
        membership_plan=membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_documents=parts,
        completed_at="2026-08-11T07:32:00Z",
    )
    ctx.request["industry_source"] = {
        "taxonomy_plan": put(tmp_path, "industry/tax-plan.json", taxonomy_plan),
        "taxonomy_capture": put(tmp_path, "industry/tax-capture.json", taxonomy_capture),
        "membership_plan": put(tmp_path, "industry/plan.json", membership_plan),
        "membership_capture": put(tmp_path, "industry/capture.json", captured),
        "membership_partitions": [
            put(tmp_path, f"industry/part-{i}.json", p) for i, p in enumerate(parts)
        ],
    }
    ctx = ResearchSources(
        workspace=str(tmp_path),
        journal=journal,
        native_request_ref=put(tmp_path, "source-request-with-industry.json", ctx.request),
        pool_manifest_ref=ctx.pool_ref,
        release_ref=ctx.release_ref,
    )
    with journal.locked():
        ctx.prepare()
        adapter = ResearchSourceAdapter(ctx, "industry")
        request = ctx.template("industry")
        adapter.execute(request)
        outcome = adapter.probe(request).outcome
        assert outcome.state.value == "SUCCEEDED"
