from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence import (
    IntelligenceError,
    build_daily_research_policy,
    compile_daily_intelligence,
    rank_factor_signals,
)
from quant_investor.intelligence._common import artifact_ref, build_artifact, business_identity
from quant_investor.cli.main import main

NOW = "2026-08-21T13:30:00Z"
STRATEGY = "cn-tech-research"
TECH = "TUSHARE_DC:BK1001.DC"


def policy() -> dict:
    return build_daily_research_policy(
        strategy_id=STRATEGY,
        effective_from="2026-08-21T00:00:00Z",
        effective_signal_date="20260821",
        effective_to=None,
        factor_rows=[
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": "LOW",
                "factor_id": "pv_low_dollar_volume_5d",
                "weight": "0.5",
            },
            {
                "direction": "HIGHER_IS_BETTER",
                "factor_alias": "W80",
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                "weight": "0.5",
            },
        ],
        pool_policy={
            "minimum_cohort": 2,
            "missing_rule": "BLOCK_ON_ANY_MISSING_OR_NONFINITE",
            "normalization": "AVERAGE_TIE_PERCENTILE_ASCENDING_ZERO_ONE",
            "pool_boundary_rule": "EXACT_LIMIT_ASCII_SYMBOL_TIEBREAK",
            "pool_size": 2,
            "sort_key": "DESC_COMBINED_PERCENTILE_ASCII_SYMBOL",
            "tie_rule": "AVERAGE_ORDINAL_PERCENTILE",
        },
        decision_thresholds={
            "paper_candidate": "0.9",
            "research_approved": "0.7",
        },
        technology_theme_ids=[TECH],
        technology_policy_state="ACTIVE",
        theme_provider_precedence=["TUSHARE_DC", "TUSHARE_TDX"],
        fundamental_freshness={"policy": "ADVISORY_NO_FIXED_MAXIMUM"},
        created_at=NOW,
    )


def rank_artifact(policy_artifact: dict) -> dict:
    rows = [
        {
            "combined_percentile": "0.950000000000",
            "factor_percentiles": {"LOW": "1.000000000000", "W80": "0.900000000000"},
            "symbol": "000001.SZ",
        },
        {
            "combined_percentile": "0.800000000000",
            "factor_percentiles": {"LOW": "0.800000000000", "W80": "0.800000000000"},
            "symbol": "000002.SZ",
        },
    ]
    return build_artifact(
        kind="factor_research_rank",
        identity_field="rank_id",
        identity=business_identity(
            kind="factor_research_rank",
            identity_inputs={"policy_id": policy_artifact["artifact_id"], "pointer": "a" * 64},
        ),
        created_at=NOW,
        fields={
            "as_of": NOW,
            "blocker_codes": [],
            "common_symbol_count": 2,
            "common_symbol_set_sha256": hashlib.sha256(
                canonical_json_bytes(["000001.SZ", "000002.SZ"])
            ).hexdigest(),
            "factor_generation_ref": {
                "artifact_id": "generation-a",
                "byte_sha256": "b" * 64,
                "contract_sha256": "c" * 64,
                "kind": "factor.production_generation",
                "semantic_sha256": "d" * 64,
            },
            "factor_pointer_sha256": "a" * 64,
            "observation_refs": [
                {
                    "artifact_id": f"observation-{alias}",
                    "byte_sha256": character * 64,
                    "contract_sha256": "e" * 64,
                    "kind": "factor.production_observation",
                    "semantic_sha256": "f" * 64,
                }
                for alias, character in (("low", "1"), ("w80", "2"))
            ],
            "policy_ref": artifact_ref(policy_artifact),
            "pool_rows": rows,
            "signal_date": "20260821",
            "status": "READY",
            "strategy_id": STRATEGY,
        },
    )


def projections(policy_artifact: dict) -> tuple[dict, dict]:
    companies = ["000001.SZ", "000002.SZ"]
    company_sha = hashlib.sha256(canonical_json_bytes(companies)).hexdigest()
    industry = build_artifact(
        kind="industry_source_projection",
        identity_field="projection_id",
        identity="industry-projection",
        created_at=NOW,
        fields={
            "as_of": NOW,
            "blocker_codes": [],
            "company_rows": [
                {
                    "company_code": company,
                    "industry_ids": [f"TUSHARE_SW2021:L3-{index}"],
                    "status": "AVAILABLE",
                }
                for index, company in enumerate(companies, 1)
            ],
            "company_set_sha256": company_sha,
            "provider": "TUSHARE_SW2021",
            "source_refs": [],
            "status": "READY",
        },
    )
    theme = build_artifact(
        kind="theme_membership_projection",
        identity_field="projection_id",
        identity="theme-projection",
        created_at=NOW,
        fields={
            "as_of": NOW,
            "blocker_codes": [],
            "company_rows": [
                {
                    "company_code": "000001.SZ",
                    "provider": "TUSHARE_DC",
                    "status": "MEMBERSHIP_ONLY",
                    "technology_theme_ids": [TECH],
                    "theme_ids": [TECH],
                },
                {
                    "company_code": "000002.SZ",
                    "provider": "TUSHARE_DC",
                    "status": "MEMBERSHIP_ONLY",
                    "technology_theme_ids": [],
                    "theme_ids": ["TUSHARE_DC:UTILITY"],
                },
            ],
            "company_set_sha256": company_sha,
            "fallback_company_keyset": [],
            "policy_ref": artifact_ref(policy_artifact),
            "source_refs": [],
            "status": "READY",
            "trade_date": "20260821",
        },
    )
    return industry, theme


def test_rank_policy_is_deterministic_for_ties_and_input_order() -> None:
    policy_artifact = policy()
    values = {
        "pv_low_dollar_volume_5d": {
            "000002.SZ": float(2).hex(),
            "000001.SZ": float(2).hex(),
            "000003.SZ": float(1).hex(),
        },
        "pv_blend_volstab19x2_mom90_amihud5_w80": {
            "000003.SZ": float(3).hex(),
            "000001.SZ": float(1).hex(),
            "000002.SZ": float(2).hex(),
        },
    }
    result = rank_factor_signals(signal_values=values, policy=policy_artifact)
    reordered = {factor: dict(reversed(list(rows.items()))) for factor, rows in values.items()}
    assert result == rank_factor_signals(signal_values=reordered, policy=policy_artifact)
    assert result["common_symbol_count"] == 3
    assert [row["symbol"] for row in result["pool_rows"]] == ["000002.SZ", "000003.SZ"]
    assert result["pool_rows"][0]["factor_percentiles"]["LOW"] == "0.750000000000"


def test_compile_daily_keeps_membership_only_theme_insufficient() -> None:
    policy_artifact = policy()
    industry, theme = projections(policy_artifact)
    result = compile_daily_intelligence(
        as_of=NOW,
        strategy_id=STRATEGY,
        rank=rank_artifact(policy_artifact),
        policy=policy_artifact,
        industry_projection=industry,
        theme_projection=theme,
    )
    decisions = {row["company_code"]: row for row in result["decisions"]}
    assert result["status"] == "PARTIAL"
    assert decisions["000001.SZ"]["technology_gate"] == "PASS"
    assert decisions["000001.SZ"]["theme_assessment_ref"] is None
    assert decisions["000001.SZ"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert decisions["000002.SZ"]["technology_gate"] == "REJECT_NON_TECH"
    assert decisions["000002.SZ"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert result["evidence_bundle"]["payload"]["status"] == "BLOCKED"
    assert all(value is False for value in result["authority"].values())


def test_compile_daily_returns_bounded_replayable_inline_artifacts() -> None:
    policy_artifact = policy()
    industry, theme_projection = projections(policy_artifact)
    result = compile_daily_intelligence(
        as_of=NOW,
        strategy_id=STRATEGY,
        rank=rank_artifact(policy_artifact),
        policy=policy_artifact,
        industry_projection=industry,
        theme_projection=theme_projection,
    )
    decisions = {row["company_code"]: row for row in result["decisions"]}
    assert decisions["000001.SZ"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert decisions["000002.SZ"]["state"] == "INSUFFICIENT_EVIDENCE"
    assert result["status"] == "PARTIAL"
    assert result["production"] is False
    assert result["run_state"] == "INACTIVE"
    assert len(canonical_json_bytes(result)) < 8 * 1024 * 1024
    inline = {tuple(sorted(artifact_ref(artifact).items())) for artifact in result["artifacts"]}
    for row in result["decisions"]:
        for field in ("decision_ref", "industry_ref"):
            assert tuple(sorted(row[field].items())) in inline
    assert tuple(sorted(artifact_ref(result["evaluation"]).items())) in inline
    assert tuple(sorted(artifact_ref(result["evidence_bundle"]).items())) in inline


def test_policy_mutations_fail_closed() -> None:
    import quant_investor.intelligence.daily as daily

    with pytest.raises(IntelligenceError, match="provider-qualified"):
        build_daily_research_policy(
            strategy_id=STRATEGY,
            effective_from="2026-08-21T00:00:00Z",
            effective_signal_date="20260821",
            effective_to=None,
            factor_rows=policy()["payload"]["factor_rows"],
            pool_policy=policy()["payload"]["pool_policy"],
            decision_thresholds=policy()["payload"]["decision_thresholds"],
            technology_theme_ids=["UNQUALIFIED"],
            technology_policy_state="ACTIVE",
            theme_provider_precedence=["TUSHARE_DC", "TUSHARE_TDX"],
            fundamental_freshness=policy()["payload"]["fundamental_freshness"],
            created_at=NOW,
        )

    changed_rows = [dict(row) for row in policy()["payload"]["factor_rows"]]
    changed_rows[0]["weight"] = "0.9"
    changed_rows[1]["weight"] = "0.1"
    with pytest.raises(IntelligenceError, match="50/50"):
        build_daily_research_policy(
            strategy_id=STRATEGY,
            effective_from="2026-08-21T00:00:00Z",
            effective_signal_date="20260821",
            effective_to=None,
            factor_rows=changed_rows,
            pool_policy=policy()["payload"]["pool_policy"],
            decision_thresholds=policy()["payload"]["decision_thresholds"],
            technology_theme_ids=[TECH],
            technology_policy_state="ACTIVE",
            theme_provider_precedence=["TUSHARE_DC", "TUSHARE_TDX"],
            fundamental_freshness=policy()["payload"]["fundamental_freshness"],
            created_at=NOW,
        )

    active_rows = [
        {
            "direction": "HIGHER_IS_BETTER",
            "factor_id": "pv_low_dollar_volume_5d",
            "role": "BOOTSTRAP",
            "selectable": True,
            "weight": "0.500000000000",
        },
        {
            "direction": "HIGHER_IS_BETTER",
            "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
            "role": "BOOTSTRAP",
            "selectable": True,
            "weight": "0.500000000000",
        },
    ]
    daily._validate_active_factor_policy(
        active_rows,
        policy_payload=policy()["payload"],
    )
    changed_active = [dict(row) for row in active_rows]
    changed_active[0]["weight"] = "0.900000000000"
    with pytest.raises(IntelligenceError, match="differs from active"):
        daily._validate_active_factor_policy(
            changed_active,
            policy_payload=policy()["payload"],
        )


def _write_request(path: Path, document: dict) -> str:
    raw = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def test_compile_daily_cli_is_one_read_only_exact_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import quant_investor.factors.production_authority as production
    import quant_investor.intelligence as intelligence

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        production,
        "read_factor_production_research_inputs",
        lambda workspace_root, expected_pointer_sha256: calls.append(
            ("read", expected_pointer_sha256)
        )
        or {"snapshot": True},
    )
    monkeypatch.setattr(
        production,
        "assert_factor_production_pointer",
        lambda workspace_root, expected_pointer_sha256: calls.append(
            ("assert", expected_pointer_sha256)
        ),
    )
    monkeypatch.setattr(
        intelligence,
        "build_factor_research_rank",
        lambda **kwargs: calls.append(("rank", kwargs["snapshot"]))
        or {"payload": {"pool_rows": [{"symbol": "000001.SZ"}]}},
    )
    result = {
        "authority": {},
        "production": False,
        "research_only": True,
        "run_state": "INACTIVE",
        "status": "PARTIAL",
    }
    monkeypatch.setattr(
        intelligence,
        "compile_daily_intelligence",
        lambda **kwargs: calls.append(("compile", kwargs["strategy_id"])) or result,
    )
    low_sha = _write_request(tmp_path / "low.json", {"alias": "LOW"})
    w80_sha = _write_request(tmp_path / "w80.json", {"alias": "W80"})
    request = {
        "as_of": NOW,
        "expected_factor_pointer_sha256": "a" * 64,
        "industry_source": None,
        "low_observation_path": "low.json",
        "low_observation_sha256": low_sha,
        "policy": {},
        "strategy_id": STRATEGY,
        "theme_source": None,
        "w80_observation_path": "w80.json",
        "w80_observation_sha256": w80_sha,
    }
    request_sha = _write_request(tmp_path / "daily.json", request)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    main(
        [
            "research",
            "compile-daily",
            "--workspace-root",
            str(tmp_path),
            "--request",
            "daily.json",
            "--expected-request-sha256",
            request_sha,
        ]
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == result
    assert captured.out.count("\n") == 1
    assert calls == [
        ("read", "a" * 64),
        ("rank", {"snapshot": True}),
        ("compile", STRATEGY),
        ("assert", "a" * 64),
    ]
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == before


def test_stable_tushare_source_projections_replay_exact_captures() -> None:
    from quant_investor.intelligence import (
        project_tushare_industry_source,
        project_tushare_theme_source,
    )
    from quant_investor.market.tushare import (
        build_industry_membership_capture,
        build_industry_membership_partition_capture,
        build_theme_partition_capture,
        build_theme_provider_capture,
        build_theme_provider_execution_plan,
    )
    from tests.unit.test_tushare_industry_capture_stable import (
        _member_row,
        _membership_plan,
    )

    taxonomy_plan, taxonomy_capture, membership_plan = _membership_plan()
    partitions = []
    for ordinal, key in enumerate(
        membership_plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    ):
        code = key.split("|", 1)[0].split("=", 1)[1]
        flag = key.rsplit("=", 1)[1]
        rows = [_member_row(l3_code=code, flag=flag)] if ordinal == 0 else []
        partitions.append(
            build_industry_membership_partition_capture(
                membership_plan=membership_plan,
                taxonomy_plan=taxonomy_plan,
                taxonomy_capture=taxonomy_capture,
                partition_key=key,
                partition_ordinal=ordinal,
                provider_request_id=f"request-{ordinal}",
                reported_count=len(rows),
                rows=rows,
                captured_at="2026-08-11T07:31:00Z",
            )
        )
    membership_capture = build_industry_membership_capture(
        membership_plan=membership_plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_documents=partitions,
        completed_at="2026-08-11T07:32:00Z",
    )
    industry = project_tushare_industry_source(
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        membership_plan=membership_plan,
        membership_capture=membership_capture,
        partition_documents=partitions,
        companies=["000001.SZ"],
        as_of=NOW,
    )
    assert industry["payload"]["company_rows"] == [
        {
            "company_code": "000001.SZ",
            "industry_ids": ["TUSHARE_SW2021:L3000.SI"],
            "status": "AVAILABLE",
        }
    ]
    assert len(industry["payload"]["source_refs"]) == 4

    dc_plan = build_theme_provider_execution_plan(
        provider="TUSHARE_DC",
        trade_date="20260821",
        company_keyset=["000001.SZ"],
        document_observed_at="2026-08-21T13:00:00Z",
        created_at="2026-08-21T13:00:00Z",
    )
    dc_partitions = [
        build_theme_partition_capture(
            plan=dc_plan,
            partition_ordinal=0,
            provider_request_id="registry-request",
            reported_count=1,
            rows=[
                {
                    "idx_type": "概念板块",
                    "level": "1",
                    "name": "机器人",
                    "trade_date": "20260821",
                    "ts_code": "BK1001.DC",
                }
            ],
            blocker_codes=[],
            captured_at="2026-08-21T13:01:00Z",
        ),
        build_theme_partition_capture(
            plan=dc_plan,
            partition_ordinal=1,
            provider_request_id="member-request",
            reported_count=1,
            rows=[
                {
                    "con_code": "000001.SZ",
                    "name": "公司",
                    "trade_date": "20260821",
                    "ts_code": "BK1001.DC",
                }
            ],
            blocker_codes=[],
            captured_at="2026-08-21T13:01:00Z",
        ),
    ]
    dc_capture = build_theme_provider_capture(
        plan=dc_plan,
        partition_documents=dc_partitions,
        completed_at="2026-08-21T13:02:00Z",
    )
    theme = project_tushare_theme_source(
        dc_plan=dc_plan,
        dc_capture=dc_capture,
        dc_partitions=dc_partitions,
        tdx_plan=None,
        tdx_capture=None,
        tdx_partitions=[],
        policy=policy(),
        as_of=NOW,
    )
    assert theme["payload"]["company_rows"] == [
        {
            "company_code": "000001.SZ",
            "provider": "TUSHARE_DC",
            "status": "MEMBERSHIP_ONLY",
            "technology_theme_ids": [TECH],
            "theme_ids": [TECH],
        }
    ]
    assert theme["payload"]["fallback_company_keyset"] == []
