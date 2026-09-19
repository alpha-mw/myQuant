"""Native missing-only evidence classification; all source and financial data are synthetic."""

from copy import deepcopy
from functools import partial

import pytest

from quant_investor.cli import unified
from quant_investor.cli.output import CommandError
from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.intelligence.daily_evidence import (
    build_source_bound_economic_exposure_projection,
)
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.intelligence.theme_governance import (
    build_unverified_economic_exposure_projection,
)
from quant_investor.intelligence.pcb_ai_hardware import (
    FOCUS_COMPANIES,
    EVIDENCE_KIND,
    build_focus_membership,
    build_focus_evidence,
)
from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2, descriptor_refs
from quant_investor.operations.daily_contract import ContractError, NodeState
from quant_investor.operations.exposure_completion import (
    POLICY,
    optional_exposure_completion,
    validate_optional_exposure_completion,
    validate_optional_focus_completion,
)
from quant_investor.operations.research_projection import project_research_source
from test_unified_daily_evidence import _theme_projection, _exposure, COMPANY, OTHER, NOW
from test_daily_evidence_focus_sources import (
    AS_OF,
    theme_source,
    pit_context,
    exposure_rows,
    put,
    inventory,
    TECHNOLOGY_THEME_IDS,
)


def ordinary(share=None):
    theme = _theme_projection()
    evidence = [] if share is None else [_exposure(share)]
    args = dict(as_of=NOW, daily_policy=approved_theme_policy_v2(), theme_projection=theme)
    projection = (
        build_source_bound_economic_exposure_projection(**args, evidence=evidence)[0]
        if evidence
        else build_unverified_economic_exposure_projection(**args)
    )
    return dict(
        companies=[COMPANY, OTHER],
        theme=theme,
        projection=projection,
        evidence=evidence,
        daily_policy=approved_theme_policy_v2(),
    )


@pytest.mark.parametrize("value", [None, False, True, 0, [], {}, "", "unknown"])
def test_selector_is_exact_and_absence_retains_legacy(value):
    assert optional_exposure_completion(recipe={}) is False
    assert optional_exposure_completion(recipe={"source_completion_policy": POLICY}) is True
    with pytest.raises(ContractError, match="POLICY_INVALID"):
        optional_exposure_completion(recipe={"source_completion_policy": value})


@pytest.mark.parametrize("share", [None, "0.01", "0.10", "0.30", "0"])
def test_only_missing_unverified_is_tolerated_and_bytes_are_unchanged(share):
    args = ordinary(share)
    before = canonical_json_bytes(args)
    if share == "0":
        with pytest.raises(ContractError, match="NOT_OPTIONAL_MISSING"):
            validate_optional_exposure_completion(**args)
    else:
        expected = ("ECONOMIC_EXPOSURE_UNVERIFIED:" + COMPANY,) if share is None else ()
        assert validate_optional_exposure_completion(**args) == expected
        assert args["projection"]["payload"]["status"] == ("BLOCKED" if expected else "READY")
    assert canonical_json_bytes(args) == before


@pytest.mark.parametrize(
    "fault",
    ["missing_company", "duplicate_company", "company_sha", "reason", "extra_blocker", "status"],
)
def test_rehashed_projection_mutations_fail(fault):
    args = ordinary()
    value = deepcopy(args["projection"])
    body = value["payload"]
    if fault == "missing_company":
        body["company_rows"].pop()
    elif fault == "duplicate_company":
        body["company_rows"].append(deepcopy(body["company_rows"][0]))
    elif fault == "company_sha":
        body["company_set_sha256"] = "f" * 64
    elif fault == "reason":
        body["company_rows"][0]["reason_codes"] = ["UNREGISTERED_MISSING"]
    elif fault == "extra_blocker":
        body["blocker_codes"].append("ARBITRARY_BLOCKER")
    else:
        body["status"] = "READY"
    args["projection"] = seal_artifact(value["kind"], body, created_at=value["created_at"])
    with pytest.raises(ValueError):
        validate_optional_exposure_completion(**args)


@pytest.mark.parametrize(
    "fault",
    [
        "expected_duplicate",
        "expected_wrong",
        "theme_blocked",
        "theme_unmapped",
        "evidence_duplicate",
        "evidence_extra",
    ],
)
def test_company_theme_and_evidence_closure_cannot_be_bypassed(fault):
    args = ordinary("0.30")
    if fault == "expected_duplicate":
        args["companies"] = [COMPANY, COMPANY]
    elif fault == "expected_wrong":
        args["companies"] = [COMPANY, "000001.SZ"]
    elif fault.startswith("theme_"):
        t = deepcopy(args["theme"])
        if fault == "theme_blocked":
            t["payload"]["blocker_codes"] = ["SOURCE_UNAVAILABLE"]
            t["payload"]["status"] = "PARTIAL"
        else:
            t["payload"]["company_rows"][0]["status"] = "UNMAPPED"
        args["theme"] = seal_artifact(t["kind"], t["payload"], created_at=t["created_at"])
    elif fault == "evidence_duplicate":
        args["evidence"] *= 2
    else:
        from quant_investor.intelligence.daily_evidence import build_company_source_evidence

        args["evidence"].append(
            build_company_source_evidence(
                company="000001.SZ",
                source_type="ANNUAL_REPORT",
                source_path="fixture.pdf",
                source_sha256="b" * 64,
                source_page=None,
                available_at="2026-08-20T00:00:00Z",
                created_at=NOW,
                metrics={"primary_theme_id": TECHNOLOGY_THEME_IDS[0], "theme_revenue_share": "0.5"},
            )
        )
    with pytest.raises(ValueError):
        validate_optional_exposure_completion(**args)


def source_case(root, *, focus=False, conflict=False, zero=False, missing_focus=False):
    pool = [FOCUS_COMPANIES[0]] if conflict else ["000001.SZ"]
    policy = approved_theme_policy_v2()
    primary = theme_source(root, pool, "ordinary")
    read = partial(unified._daily_source_document, str(root))
    fields = dict(as_of=AS_OF, policy=policy)
    theme = unified._daily_theme_projection({**fields, "theme_source": primary}, pool, read)
    recipe = {
        **fields,
        "pool_ref": put(root, "pool.json", {"synthetic": True}),
        "exposure_rows": None,
    }
    args = dict(
        node="exposure",
        companies=pool,
        theme=theme,
        source_document=read,
        source_file=partial(unified._daily_source_file, root),
        workspace=str(root),
    )
    if focus:
        selected = TECHNOLOGY_THEME_IDS[1] if conflict else None
        descriptor = (
            None
            if missing_focus
            else theme_source(root, list(FOCUS_COMPANIES), "focus", theme=selected)
        )
        projection = unified._daily_theme_projection(
            {**fields, "theme_source": descriptor}, list(FOCUS_COMPANIES), read
        )
        pit = pit_context(root)
        membership = build_focus_membership(
            as_of=AS_OF,
            pool_manifest_ref=recipe["pool_ref"],
            pit=pit,
            pool_theme=theme,
            focus_theme=projection,
            pool_source_refs=descriptor_refs(primary),
            focus_source_refs=descriptor_refs(descriptor),
        )
        recipe.update(
            focus_context=pit,
            focus_theme_source={
                "schema_version": THEME_SOURCE_V2,
                "pool": primary,
                "pcb_ai_hardware": descriptor,
            },
            focus_industry_source=None,
        )
        args["focus_membership"] = membership
        if zero:
            rows = exposure_rows(root, list(FOCUS_COMPANIES))
            rows[0]["theme_revenue_share"] = "0"
            recipe["exposure_rows"] = rows
    return args, recipe


def test_valid_raw_focus_facts_without_membership_remain_unqualified(tmp_path):
    args, recipe = source_case(tmp_path, focus=True, missing_focus=True)
    recipe["exposure_rows"] = exposure_rows(tmp_path, list(FOCUS_COMPANIES))
    legacy = project_research_source(**args, recipe=recipe)
    selected = project_research_source(
        **args, recipe={**recipe, "source_completion_policy": POLICY}
    )
    assert legacy.state == NodeState.PARTIAL and selected.state == NodeState.SUCCEEDED
    assert canonical_json_bytes(legacy.artifacts) == canonical_json_bytes(selected.artifacts)
    report = next(v["payload"] for v in selected.artifacts if v["kind"] == EVIDENCE_KIND)
    for row in report["company_rows"]:
        assert row["economic_exposure"] is None
        assert "FOCUS_SOURCE_MISSING" in row["missing_codes"]
        assert "FOCUS_EXPOSURE_NOT_QUALIFIED" in row["missing_codes"]
        assert row["source_evidence"]["company_evidence_refs"] == []
        assert row["source_evidence"]["unqualified_company_evidence_refs"]


@pytest.mark.parametrize("fault", ["code", "company", "state", "pit_ref"])
def test_rehashed_focus_report_cannot_change_native_missingness(tmp_path, fault):
    args, recipe = source_case(tmp_path, focus=True)
    focus = unified._daily_theme_projection(
        {
            "as_of": AS_OF,
            "policy": recipe["policy"],
            "theme_source": recipe["focus_theme_source"]["pcb_ai_hardware"],
        },
        list(FOCUS_COMPANIES),
        args["source_document"],
    )
    bound = dict(
        membership=args["focus_membership"],
        pit=recipe["focus_context"],
        focus_theme=focus,
        industry=None,
        evidence=[],
        industry_source_refs=[],
        daily_policy=recipe["policy"],
    )
    report = build_focus_evidence(**bound)
    original = canonical_json_bytes(report)
    assert validate_optional_focus_completion(report=report, **bound) == tuple(
        report["payload"]["missing_codes"]
    )
    assert canonical_json_bytes(report) == original
    if fault == "code":
        report["payload"]["company_rows"][0]["missing_codes"] = ["FOCUS_UNKNOWN_MISSING"]
    elif fault == "company":
        report["payload"]["company_rows"].pop()
    elif fault == "state":
        report["payload"]["completion_state"] = "SUCCEEDED"
    else:
        report["payload"]["pit_membership_ref"]["sha256"] = "f" * 64
    forged = seal_artifact(report["kind"], report["payload"], created_at=report["created_at"])
    with pytest.raises(ContractError, match="NATIVE_REPLAY_MISMATCH"):
        validate_optional_focus_completion(report=forged, **bound)


@pytest.mark.parametrize("fault", ["source_sha", "pit_binding"])
def test_invalid_focus_inputs_do_not_reach_successful_classification(tmp_path, fault):
    args, recipe = source_case(tmp_path, focus=True)
    if fault == "source_sha":
        recipe["focus_theme_source"]["pcb_ai_hardware"]["dc_partitions"][1]["sha256"] = "f" * 64
    else:
        recipe["focus_context"]["pit_membership_ref"]["sha256"] = "f" * 64
    with pytest.raises((ValueError, CommandError)):
        project_research_source(**args, recipe={**recipe, "source_completion_policy": POLICY})


@pytest.mark.parametrize("focus", [False, True])
def test_opt_in_changes_only_lifecycle_and_retains_missing_focus_reports(tmp_path, focus):
    args, recipe = source_case(tmp_path, focus=focus)
    before = inventory(tmp_path)
    legacy = project_research_source(**args, recipe=recipe)
    selected = project_research_source(
        **args, recipe={**recipe, "source_completion_policy": POLICY}
    )
    assert legacy.state == NodeState.PARTIAL and selected.state == NodeState.SUCCEEDED
    assert canonical_json_bytes(legacy.artifacts) == canonical_json_bytes(selected.artifacts)
    assert selected.artifacts[0]["payload"]["status"] == "BLOCKED"
    if focus:
        report = next(v["payload"] for v in selected.artifacts if v["kind"] == EVIDENCE_KIND)
        assert report["completion_state"] == "PARTIAL_WITH_EXPLICIT_MISSING"
        assert [row["company_code"] for row in report["company_rows"]] == list(FOCUS_COMPANIES)
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("fault", ["conflict", "zero"])
def test_focus_conflict_and_evidenced_zero_share_are_hard_failures(tmp_path, fault):
    args, recipe = source_case(
        tmp_path, focus=True, conflict=fault == "conflict", zero=fault == "zero"
    )
    legacy = project_research_source(**args, recipe=recipe)
    assert legacy.state == NodeState.PARTIAL
    with pytest.raises(ContractError, match="FOCUS_COMPLETION_NOT_OPTIONAL_MISSING"):
        project_research_source(**args, recipe={**recipe, "source_completion_policy": POLICY})


@pytest.mark.parametrize(
    "fault", ["missing", "sha", "future", "duplicate", "wrong_company", "wrong_theme"]
)
def test_declared_source_errors_still_fail_before_success(tmp_path, fault):
    args, recipe = source_case(tmp_path)
    rows = exposure_rows(tmp_path, args["companies"])
    if fault == "missing":
        (tmp_path / rows[0]["source"]["path"]).unlink()
    elif fault == "sha":
        rows[0]["source"]["sha256"] = "f" * 64
    elif fault == "future":
        rows[0]["available_at"] = "2026-08-29T00:00:00Z"
    elif fault == "duplicate":
        rows *= 2
    elif fault == "wrong_company":
        rows[0]["company_code"] = "000002.SZ"
    else:
        rows[0]["primary_theme_id"] = TECHNOLOGY_THEME_IDS[1]
    recipe.update(exposure_rows=rows, source_completion_policy=POLICY)
    with pytest.raises((ValueError, CommandError)):
        project_research_source(**args, recipe=recipe)


def test_removing_exposure_reduces_an_actually_admitted_native_decision():
    """Native compiler and Decision execute; rank/source artifacts are explicit fixtures."""
    import hashlib
    import pandas as pd
    from quant_investor.intelligence import compile_daily_intelligence
    from quant_investor.intelligence._common import artifact_ref, build_artifact
    from quant_investor.intelligence.daily_evidence import (
        FUNDAMENTAL_METRICS,
        build_company_source_evidence,
        build_market_risk_evidence,
    )
    from test_unified_daily_intelligence_storage import _rank

    policy = approved_theme_policy_v2()
    rank = _rank(policy, signal_date="20260827")
    as_of = rank["created_at"]
    companies = [row["symbol"] for row in rank["payload"]["pool_rows"]]
    first = companies[0]
    company_sha = hashlib.sha256(canonical_json_bytes(sorted(companies))).hexdigest()
    theme = build_artifact(
        kind="theme_membership_projection",
        identity_field="projection_id",
        identity="synthetic-admission-theme",
        created_at=as_of,
        fields={
            "as_of": as_of,
            "blocker_codes": [],
            "company_set_sha256": company_sha,
            "company_rows": [
                {
                    "company_code": company,
                    "provider": "TUSHARE_DC",
                    "status": "MEMBERSHIP_ONLY",
                    "technology_theme_ids": [TECHNOLOGY_THEME_IDS[0]] if company == first else [],
                    "theme_ids": (
                        [TECHNOLOGY_THEME_IDS[0]] if company == first else ["TUSHARE_DC:BK0001.DC"]
                    ),
                }
                for company in companies
            ],
            "fallback_company_keyset": [],
            "policy_ref": artifact_ref(policy),
            "source_refs": [],
            "status": "READY",
            "trade_date": "20260827",
        },
    )
    industry = build_artifact(
        kind="industry_source_projection",
        identity_field="projection_id",
        identity="synthetic-admission-industry",
        created_at=as_of,
        fields={
            "as_of": as_of,
            "blocker_codes": [],
            "company_set_sha256": company_sha,
            "company_rows": [
                {
                    "company_code": company,
                    "industry_ids": ["TUSHARE_SW2021:850811.SI"],
                    "status": "AVAILABLE",
                }
                for company in companies
            ],
            "provider": "TUSHARE_SW2021",
            "source_refs": [],
            "status": "READY",
        },
    )
    evidence = build_company_source_evidence(
        company=first,
        source_type="ANNUAL_REPORT",
        source_path="synthetic/revenue.pdf",
        source_sha256="b" * 64,
        available_at="2026-08-20T00:00:00Z",
        source_page=1,
        created_at=as_of,
        metrics={"primary_theme_id": TECHNOLOGY_THEME_IDS[0], "theme_revenue_share": "0.5"},
    )
    frame = pd.DataFrame(
        [{"ts_code": first, "trade_date": "20260826", **dict.fromkeys(FUNDAMENTAL_METRICS, 0.2)}]
    )
    risk = build_market_risk_evidence(
        source_path=f"results/intelligence/macro_readiness/20260827/{'c' * 64}.json",
        source_sha256="c" * 64,
        blocker_codes=[],
        classification="CANONICAL_MACRO_READY",
        as_of=as_of,
    )
    args = dict(
        as_of=as_of,
        strategy_id="aggressive_tech_manufacturing",
        rank=rank,
        policy=policy,
        industry_projection=industry,
        theme_projection=theme,
        fundamental_frame=frame,
        fundamental_source={
            "available_at": "2026-08-26T00:00:00Z",
            "path": "synthetic/fundamental.parquet",
            "sha256": "d" * 64,
        },
        market_risk_evidence=risk,
    )
    complete = compile_daily_intelligence(**args, exposure_evidence=[evidence])
    missing = compile_daily_intelligence(**args, exposure_evidence=[])
    decisions = [
        {r["company_code"]: r for r in result["decisions"]} for result in (complete, missing)
    ]
    assert decisions[0][first]["state"] == "PAPER_CANDIDATE"
    assert decisions[1][first]["state"] == "INSUFFICIENT_EVIDENCE"
    assert decisions[1][first]["economic_exposure_state"] == "UNVERIFIED"
    assert all(decisions[0][c]["state"] == decisions[1][c]["state"] for c in companies[1:])
    projected = next(
        a for a in missing["artifacts"] if a["kind"] == "theme_economic_exposure_projection"
    )
    before = canonical_json_bytes(missing)
    assert validate_optional_exposure_completion(
        companies=companies, theme=theme, projection=projected, daily_policy=policy, evidence=[]
    ) == ("ECONOMIC_EXPOSURE_UNVERIFIED:" + first,)
    assert canonical_json_bytes(missing) == before
    assert all(
        value is False for result in (complete, missing) for value in result["authority"].values()
    )
