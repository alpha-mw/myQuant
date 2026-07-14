from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

from quant_investor.fundamental_research import (
    ClaimKind,
    ClaimV1,
    Dimension,
    DimensionAssessmentV1,
    DimensionSignal,
    FundamentalResearchDossierV1,
    FundamentalResearchResponseV1,
    SourceRecordV1,
    SourceTier,
    atomic_write_json_model,
    compute_base_score_sha256,
    load_json_model,
    model_sha256,
)
from quant_investor.fundamental_research.models import FundamentalResearchRequestV1
from quant_investor.fundamental_research.storage import canonical_json_bytes
from quant_investor.fundamental_research.workflow import (
    WorkflowInputError,
    generate_activation_gate_evidence,
    import_research_response,
    prepare_research_requests,
    research_status,
)

UTC = timezone.utc
CN = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 14, 8, tzinfo=UTC)
AS_OF = datetime(2026, 7, 14, 15, tzinfo=CN)


def _sha(value: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_analysis(path: Path, *, git_sha: str | None = None) -> Path:
    bases = {
        "000001.SZ": {
            "company_name": "平安银行",
            "base_score": 0.2,
            "base_score_sha256": compute_base_score_sha256(0.2),
            "status": "SUCCESS",
            "data_generation": "fundamental-gen-20260714-a",
            "industry": "银行",
            "peer_symbols": ["600036.SH"],
            "peer_set_status": "confirmed",
            "base_confidence": 0.7,
            "valuation_price": 12.5,
            "valuation_price_as_of": "2026-07-14",
            "available_modules": ["financial_quality", "valuation"],
            "missing_modules": ["management_quality"],
            "runtime_audit": {"source": "deterministic_fundamental"},
        },
        "000002.SZ": {
            "company_name": "万科A",
            "base_score": -0.1,
            "base_score_sha256": compute_base_score_sha256(-0.1),
            "status": "SUCCESS",
            "data_generation": "fundamental-gen-20260714-b",
            "industry": "房地产",
            "peer_symbols": [],
            "peer_set_status": "unconfirmed",
            "base_confidence": 0.6,
            "available_modules": ["financial_quality"],
            "missing_modules": ["valuation"],
            "runtime_audit": {"source": "deterministic_fundamental"},
        },
    }
    meta = {
        "market": "CN",
        "shortlist": [
            {"symbol": "000001.SZ", "company_name": "平安银行"},
            {"symbol": "000003.SZ", "company_name": "国农科技"},
        ],
        "fundamental_deterministic_bases": bases,
        "global_context": {
            "latest_trade_date": "20260714",
            "metadata": {
                "data_snapshot": {
                    "local_latest_trade_date": "20260714",
                    "snapshot_id": "snapshot-20260714",
                }
            },
        },
        "private_account": {"capital": 1_000_000, "cost_basis": 9.9},
    }
    payload = {
        "schema_version": "analysis-run-manifest.v1",
        "run_id": "CN_20260714T080000Z",
        "generated_at": NOW.isoformat(),
        "market": "CN",
        "git_sha": git_sha or _git_sha(),
        "analysis_meta_sha256": _sha(meta),
        "analysis_meta": meta,
    }
    payload["manifest_sha256"] = _sha(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return path


def _write_holdings(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = path.parent / "ledger_after_manual_switch.parquet"
    pd.DataFrame(
        [
            {"symbol": "000002.SZ", "name": "万科A", "shares": 100, "avg_cost": 12.3},
            {"symbol": "000004.SZ", "name": "国华网安", "shares": 200, "avg_cost": 8.8},
            {"symbol": "000099.SZ", "name": "零持仓", "shares": 0, "avg_cost": 1.0},
        ]
    ).to_parquet(ledger, index=False)
    ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema_version": "cn_aggressive_manual_execution.v2",
                "status": "no_action_carry_forward",
                "recorded_at": datetime(2026, 7, 14, 14, tzinfo=CN).isoformat(),
                "next_ledger_path": ledger.name,
                "ledger_after_manual_switch_parquet": ledger.name,
                "ledger_after_manual_switch_parquet_sha256": ledger_hash,
                "cash_after": 123_456,
                "total_value_after": 999_999,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_binds_lineage_prioritizes_holdings_and_sanitizes_requests(tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / "analysis" / "analysis_run_manifest.v1.json")
    holdings = _write_holdings(tmp_path / "holdings" / "manual_execution_manifest.json")
    root = tmp_path / "private"

    manifest = prepare_research_requests(
        market="CN",
        as_of=AS_OF,
        analysis_run=analysis.parent,
        holdings_manifest=holdings,
        root=root,
        now=NOW,
    )

    assert [item.symbol for item in manifest.requested] == ["000002.SZ", "000001.SZ"]
    assert manifest.requested[0].selection_reasons == ["current_holding"]
    assert manifest.requested[1].selection_reasons == ["analysis_shortlist"]
    assert {(item.symbol, item.code) for item in manifest.blockers} >= {
        ("000004.SZ", "deterministic_fundamental_base_missing"),
        ("000003.SZ", "deterministic_fundamental_base_missing"),
    }
    request_path = root / manifest.requested[0].request_path
    request = load_json_model(root, request_path, FundamentalResearchRequestV1)
    assert request.decision_cutoff == AS_OF
    assert request.base_score == -0.1
    assert request.base_score_sha256 == compute_base_score_sha256(-0.1)
    assert request.data_generation == "fundamental-gen-20260714-b"
    assert request.local_context.industry == "房地产"
    assert request.local_context.peer_set_status == "unconfirmed"
    assert request.local_context.peer_symbols == []
    assert request.local_context.valuation_price is None
    assert ("000002.SZ", "valuation_price_unconfirmed") in {
        (item.symbol, item.code) for item in manifest.blockers
    }
    request_text = request_path.read_text(encoding="utf-8")
    assert "capital" not in request_text
    assert "cost_basis" not in request_text
    assert str(tmp_path) not in request_text
    task_path = root / manifest.requested[0].task_path
    task_text = task_path.read_text(encoding="utf-8")
    assert "20 searches" in task_text
    assert "25 deduplicated documents" in task_text
    assert "never select web substitutes" in task_text
    assert "FundamentalResearchResponseV1" in task_text
    task_payload = json.loads(task_text)
    schema_text = json.dumps(task_payload["response_json_schema"], ensure_ascii=False)
    for schema_name in ("dossier", "sources", "claims", "dimensions"):
        assert schema_name in schema_text
    assert "primary_hostname_allowlist" in task_text
    for private_key in ("123456", "999999", "12.3", "8.8", "avg_cost", str(tmp_path)):
        assert private_key not in request_text
        assert private_key not in task_text
    assert request_path.stat().st_mode & 0o777 == 0o600
    assert (root / "CN/2026-07-14/CN_20260714T080000Z/manifest.v1.json").exists()
    status = research_status(root=root, symbol="000002.SZ", state="EXPORTED")
    assert status["count"] == 1
    assert status["jobs"][0]["run_id"] == "CN_20260714T080000Z"
    assert status["jobs"][0]["expires_at"]
    assert status["jobs"][0]["derived_state"] == "EXPORTED"
    expired = research_status(
        root=root,
        symbol="000002.SZ",
        state="EXPIRED",
        now=NOW + timedelta(days=31),
    )
    assert expired["count"] == 1
    assert expired["jobs"][0]["ledger_state"] == "EXPORTED"
    assert expired["jobs"][0]["derived_state"] == "EXPIRED"

    rerun = prepare_research_requests(
        market="CN",
        as_of=AS_OF,
        analysis_run=analysis.parent,
        holdings_manifest=holdings,
        root=root,
        now=NOW + timedelta(minutes=1),
    )
    assert [item.request_sha256 for item in rerun.requested] == [
        item.request_sha256 for item in manifest.requested
    ]


@pytest.mark.parametrize("tamper", ["meta", "manifest", "git"])
def test_prepare_rejects_tampered_analysis_lineage(tmp_path: Path, tamper: str) -> None:
    analysis = _write_analysis(tmp_path / "analysis_run_manifest.v1.json")
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    if tamper == "meta":
        payload["analysis_meta"]["market"] = "US"
    elif tamper == "manifest":
        payload["run_id"] = "different"
    else:
        payload["git_sha"] = "deadbee"
        payload["manifest_sha256"] = _sha(
            {key: value for key, value in payload.items() if key != "manifest_sha256"}
        )
    analysis.write_bytes(canonical_json_bytes(payload))
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")
    with pytest.raises(WorkflowInputError):
        prepare_research_requests(
            market="CN",
            as_of=AS_OF,
            analysis_run=analysis,
            holdings_manifest=holdings,
            root=tmp_path / "root",
            now=NOW,
        )


def test_prepare_rejects_future_asof_and_unsafe_manual_ledger(tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / "analysis_run_manifest.v1.json")
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")
    with pytest.raises(WorkflowInputError, match="analysis decision cutoff"):
        prepare_research_requests(
            market="CN",
            as_of="2026-07-15T00:00:00+08:00",
            analysis_run=analysis,
            holdings_manifest=holdings,
            root=tmp_path / "root-a",
            now=NOW,
        )
    with pytest.raises(WorkflowInputError, match="analysis decision cutoff"):
        prepare_research_requests(
            market="CN",
            as_of="2026-07-13T15:00:00+08:00",
            analysis_run=analysis,
            holdings_manifest=holdings,
            root=tmp_path / "root-prior",
            now=NOW,
        )
    payload = json.loads(holdings.read_text(encoding="utf-8"))
    payload["ledger_after_manual_switch_parquet"] = "other.csv"
    (holdings.parent / "other.csv").write_text("symbol,shares\n000001.SZ,1\n", encoding="utf-8")
    payload["ledger_after_manual_switch_parquet_sha256"] = hashlib.sha256(
        (holdings.parent / "other.csv").read_bytes()
    ).hexdigest()
    holdings.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkflowInputError, match="ledger_after_manual_switch"):
        prepare_research_requests(
            market="CN",
            as_of=AS_OF,
            analysis_run=analysis,
            holdings_manifest=holdings,
            root=tmp_path / "root-b",
            now=NOW,
        )


def test_prepare_accepts_legacy_cn_cst_manual_timestamp(tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / "analysis_run_manifest.v1.json")
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")
    payload = json.loads(holdings.read_text(encoding="utf-8"))
    payload["recorded_at"] = "2026-07-14 14:00:00 CST"
    holdings.write_text(json.dumps(payload), encoding="utf-8")

    manifest = prepare_research_requests(
        market="CN",
        as_of=AS_OF,
        analysis_run=analysis,
        holdings_manifest=holdings,
        root=tmp_path / "root",
        now=NOW,
    )

    assert manifest.requested


def _response(request: FundamentalResearchRequestV1) -> FundamentalResearchResponseV1:
    hostnames = ("sse.com.cn", "szse.cn")
    sources = [
        SourceRecordV1(
            source_id=f"src-{index}",
            publisher=publisher,
            document_kind="annual_report",
            canonical_url=f"https://{hostnames[index]}/disclosure/{index}",
            published_at=AS_OF - timedelta(days=2),
            first_available_at=AS_OF - timedelta(days=2),
            retrieved_at=NOW + timedelta(minutes=5),
            source_tier=SourceTier.PRIMARY,
            content_sha256=str(index + 1) * 64,
            locator="p.1",
            evidence_extract="verified evidence",
        )
        for index, publisher in enumerate(("Exchange", "Company"))
    ]
    claims = []
    dimensions = []
    for index, dimension in enumerate(Dimension):
        claim_id = f"claim-{index}"
        claims.append(
            ClaimV1(
                claim_id=claim_id,
                kind=ClaimKind.FACT,
                dimension=dimension,
                statement="Verified fundamental evidence",
                direction="neutral",
                materiality=0.5,
                supporting_source_ids=[sources[index % 2].source_id],
            )
        )
        dimensions.append(
            DimensionAssessmentV1(
                dimension=dimension,
                signal=DimensionSignal.NEUTRAL,
                claim_ids=[claim_id],
            )
        )
    dossier = FundamentalResearchDossierV1(
        dossier_id="dossier-1",
        request_id=request.request_id,
        symbol=request.symbol,
        company_name=request.company_name,
        decision_cutoff=request.decision_cutoff,
        produced_at=NOW + timedelta(minutes=10),
        model_name="codex",
        prompt_version=request.prompt_version,
        sources=sources,
        claims=claims,
        dimensions=dimensions,
    )
    return FundamentalResearchResponseV1(
        request_id=request.request_id,
        request_sha256=model_sha256(request),
        dossier=dossier,
    )


def test_import_validate_only_then_persist_and_reject_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _write_analysis(tmp_path / "analysis" / "analysis_run_manifest.v1.json")
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")
    root = tmp_path / "private"
    manifest = prepare_research_requests(
        market="CN",
        as_of=AS_OF,
        analysis_run=analysis,
        holdings_manifest=holdings,
        root=root,
        now=NOW,
    )
    record = manifest.requested[0]
    request_path = root / record.request_path
    request = load_json_model(root, request_path, FundamentalResearchRequestV1)
    response = _response(request)
    response_path = request_path.parent.parent / "responses" / f"{request.symbol}.response.v1.json"
    atomic_write_json_model(root, response_path, response)

    validated = import_research_response(
        root=root,
        request_path=request_path,
        response_path=response_path,
        validate_only=True,
        now=NOW + timedelta(minutes=20),
    )
    assert validated["status"] == "VALIDATED"
    assert research_status(root=root, symbol=request.symbol)["jobs"][0]["state"] == "EXPORTED"
    monkeypatch.setenv("FUNDAMENTAL_RESEARCH_EXTRA_PRIMARY_HOSTNAMES", "company.example")
    with pytest.raises(ValueError, match="source policy hash"):
        import_research_response(
            root=root,
            request_path=request_path,
            response_path=response_path,
            validate_only=True,
            now=NOW + timedelta(minutes=20),
        )
    monkeypatch.delenv("FUNDAMENTAL_RESEARCH_EXTRA_PRIMARY_HOSTNAMES")

    imported = import_research_response(
        root=root,
        request_path=request_path,
        response_path=response_path,
        now=NOW + timedelta(minutes=20),
    )
    assert imported["status"] == "VALIDATED"
    assert (root / imported["dossier_path"]).stat().st_mode & 0o777 == 0o600
    imported_status = research_status(root=root, symbol=request.symbol, state="VALIDATED")
    assert imported_status["count"] == 1
    assert imported_status["jobs"][0]["import_status"] == "VALIDATED"
    assert imported_status["jobs"][0]["overlay_eligible"] is True
    assert imported_status["jobs"][0]["overlay_blockers"] == []
    assert str(tmp_path) not in json.dumps(imported_status, ensure_ascii=False)

    second_record = manifest.requested[1]
    second_request_path = root / second_record.request_path
    second_request = load_json_model(root, second_request_path, FundamentalResearchRequestV1)
    bad_response = _response(second_request).model_copy(update={"request_sha256": "0" * 64})
    bad_path = second_request_path.parent.parent / "responses" / "bad.response.v1.json"
    atomic_write_json_model(root, bad_path, bad_response)
    with pytest.raises(WorkflowInputError, match="response rejected"):
        import_research_response(
            root=root,
            request_path=second_request_path,
            response_path=bad_path,
            now=NOW + timedelta(minutes=20),
        )
    assert research_status(root=root, symbol=second_request.symbol, state="REJECTED")["count"] == 1


def test_import_rejects_request_not_bound_to_prepare_manifest(tmp_path: Path) -> None:
    analysis = _write_analysis(tmp_path / "analysis" / "analysis_run_manifest.v1.json")
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")
    root = tmp_path / "private"
    manifest = prepare_research_requests(
        market="CN",
        as_of=AS_OF,
        analysis_run=analysis,
        holdings_manifest=holdings,
        root=root,
        now=NOW,
    )
    prepared_path = root / manifest.requested[0].request_path
    request = load_json_model(root, prepared_path, FundamentalResearchRequestV1)
    fake_run = root / "CN" / "2026-07-14" / "hand-written-run"
    fake_request_path = fake_run / "requests" / f"{request.symbol}.request.v1.json"
    fake_response_path = fake_run / "responses" / f"{request.symbol}.response.v1.json"
    atomic_write_json_model(root, fake_request_path, request)
    atomic_write_json_model(root, fake_response_path, _response(request))
    ledger_path = root / "state" / "jobs.v1.jsonl"
    ledger_before = ledger_path.read_bytes()

    with pytest.raises(WorkflowInputError, match="prepare manifest"):
        import_research_response(
            root=root,
            request_path=fake_request_path,
            response_path=fake_response_path,
            validate_only=True,
            now=NOW + timedelta(minutes=20),
        )
    with pytest.raises(WorkflowInputError, match="prepare manifest"):
        import_research_response(
            root=root,
            request_path=fake_request_path,
            response_path=fake_response_path,
            validate_only=False,
            now=NOW + timedelta(minutes=20),
        )
    assert ledger_path.read_bytes() == ledger_before


def test_cli_prepare_dispatches_explicit_asof_and_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_investor.cli import main as cli

    captured: dict[str, object] = {}

    def _prepare(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(model_dump=lambda **_: {"requested": [], "blockers": []})

    monkeypatch.setattr(cli, "run_fundamental_research_prepare", _prepare)
    cli.main(
        [
            "market",
            "fundamental-research-prepare",
            "--market",
            "CN",
            "--as-of",
            "2026-07-14T15:00:00+08:00",
            "--analysis-run",
            "analysis",
            "--holdings-manifest",
            "manual.json",
            "--root",
            "private",
        ]
    )
    assert captured["as_of"] == "2026-07-14T15:00:00+08:00"
    assert captured["analysis_run"] == "analysis"
    assert captured["holdings_manifest"] == "manual.json"
    assert captured["root"] == "private"


def test_cli_dispatches_canonical_longitudinal_producers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_investor.cli import main as cli

    target: dict[str, object] = {}
    nav: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_fundamental_research_target_weight_produce",
        lambda **kwargs: target.update(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        cli,
        "run_fundamental_research_nav_produce",
        lambda **kwargs: nav.update(kwargs) or {"ok": True},
    )

    cli.main(
        [
            "market",
            "fundamental-research-target-weight-produce",
            "--request",
            "request.json",
            "--dossier-id",
            "dossier-1",
            "--actual-analysis",
            "actual.json",
            "--counterfactual-analysis",
            "counter.json",
            "--root",
            "private",
        ]
    )
    cli.main(
        [
            "market",
            "fundamental-research-nav-produce",
            "--target-weight-observation",
            "target.json",
            "--attribution-date",
            "2026-07-15",
            "--data-root",
            "data",
            "--root",
            "private",
        ]
    )

    assert target["dossier_id"] == "dossier-1"
    assert target["counterfactual_analysis_manifest"] == "counter.json"
    assert nav["attribution_date"] == "2026-07-15"
    assert nav["target_weight_observation"] == "target.json"


def test_gate_evidence_command_uses_canonical_holdings_and_status_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("quant_investor.fundamental_research.governance.REPO_ROOT", tmp_path)
    root = tmp_path / "private"
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")

    result = generate_activation_gate_evidence(
        holdings_manifest=holdings,
        root=root,
        now=NOW,
    )

    assert result["eligible_modes"] == ["shadow"]
    assert result["holdings_coverage_passed"] is False
    assert "validated_dossiers_below_30" in result["limited_blockers"]
    evidence_path = root / result["evidence_path"]
    snapshot_path = root / result["holdings_snapshot_path"]
    assert evidence_path.stat().st_mode & 0o777 == 0o600
    assert snapshot_path.stat().st_mode & 0o777 == 0o600
    status = research_status(root=root, now=NOW + timedelta(seconds=1))
    assert status["activation_gate"]["available"] is True
    assert status["activation_gate"]["recomputed"] is True
    assert status["activation_gate"]["eligible_modes"] == ["shadow"]

    stale = research_status(root=root, now=NOW + timedelta(days=8))
    assert stale["activation_gate"]["recomputed"] is False

    ledger = holdings.parent / "ledger_after_manual_switch.parquet"
    pd.DataFrame([{"symbol": "000099.SZ", "shares": 1}]).to_parquet(ledger, index=False)
    drifted = research_status(root=root, now=NOW + timedelta(seconds=2))
    assert drifted["activation_gate"]["recomputed"] is False


@pytest.mark.parametrize("invalid", ["blocked_status", "future_timestamp", "wrong_schema"])
def test_gate_evidence_rejects_ineligible_holdings_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setattr("quant_investor.fundamental_research.governance.REPO_ROOT", tmp_path)
    holdings = _write_holdings(tmp_path / "manual" / "manual_execution_manifest.json")
    payload = json.loads(holdings.read_text(encoding="utf-8"))
    if invalid == "blocked_status":
        payload["status"] = "BLOCKED"
    elif invalid == "wrong_schema":
        payload["schema_version"] = "cn_aggressive_manual_execution.v1"
    else:
        payload["recorded_at"] = (NOW + timedelta(days=1)).isoformat()
    holdings.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        generate_activation_gate_evidence(
            holdings_manifest=holdings,
            root=tmp_path / "private",
            now=NOW,
        )
