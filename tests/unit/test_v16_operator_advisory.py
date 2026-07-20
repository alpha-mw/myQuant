from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.v16.operator_advisory import contracts
from quant_investor.v16.operator_advisory.contracts import (
    LLM_REQUEST_SCHEMA,
    LLM_RESPONSE_SCHEMA,
    AdvisoryError,
    centered_average_rank,
    validate_llm_response,
)
from quant_investor.v16.operator_advisory.factor_scoring import (
    FORMULA_FACTOR_NAME,
    QUALITY_FACTOR_NAME,
    compute_fundamental_factor_signals,
)
from quant_investor.v16.operator_advisory import provider, runtime


def _request() -> dict:
    return {
        "schema_version": LLM_REQUEST_SCHEMA,
        "model_id": provider.OPENAI_MODEL,
        "prompt_sha256": provider.PROMPT_SHA256,
        "response_schema_sha256": provider.RESPONSE_SCHEMA_SHA256,
        "evidence_file_sha256": "e" * 64,
        "items": [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "industry": "银行",
                "quant_raw": 0.2,
                "fundamental_raw": 0.1,
                "macro_raw": -0.1,
                "fact_ids": ["000001.SZ:quant:a"],
                "facts": [
                    {
                        "id": "000001.SZ:quant:a",
                        "branch": "quant",
                        "metric": "a",
                        "value": 0.2,
                    }
                ],
            }
        ],
    }


def _response(
    request_sha256: str,
    *,
    rationale: str = "Supported by sealed facts.",
    model_id: str = provider.OPENAI_MODEL,
) -> dict:
    return {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_sha256,
        "model_id": model_id,
        "prompt_sha256": provider.PROMPT_SHA256,
        "response_schema_sha256": provider.RESPONSE_SCHEMA_SHA256,
        "reviews": [
            {
                "symbol": "000001.SZ",
                "raw_score": 0.25,
                "confidence": 0.7,
                "rationale": rationale,
                "evidence_ids": ["000001.SZ:quant:a"],
                "risks": ["Source data can change."],
            }
        ],
    }


def test_average_rank_ties_and_constant_policy():
    values = pd.Series({"a": 3.0, "b": 1.0, "c": 3.0, "d": float("nan")})
    ranked = centered_average_rank(values)
    assert ranked["b"] == -1.0
    assert ranked["a"] == ranked["c"] == 0.5
    assert pd.isna(ranked["d"])
    assert centered_average_rank(pd.Series([4.0, 4.0])).tolist() == [0.0, 0.0]


def test_advisory_root_rejects_symlink_to_formal_v16(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    formal = repository / "results" / "v16"
    formal.mkdir(parents=True)
    advisory = repository / "results" / "v16_operator_advisory"
    advisory.symlink_to(formal, target_is_directory=True)
    monkeypatch.setattr(contracts, "REPO_ROOT", repository)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", advisory)
    with pytest.raises(AdvisoryError, match="symlink rejected"):
        contracts.advisory_root()


def test_advisory_root_rejects_inverse_formal_alias(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    advisory = repository / "results" / "v16_operator_advisory"
    advisory.mkdir(parents=True)
    formal = repository / "results" / "v16"
    formal.symlink_to(advisory, target_is_directory=True)
    monkeypatch.setattr(contracts, "REPO_ROOT", repository)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", advisory)
    with pytest.raises(AdvisoryError, match="aliases forbidden control tree"):
        contracts.advisory_root()


@pytest.mark.parametrize("max_candidates", [0, 51])
def test_invalid_candidate_count_is_rejected_before_run_creation(
    tmp_path,
    monkeypatch,
    max_candidates,
):
    root = tmp_path / "results" / "v16_operator_advisory"
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", root)
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    run_id = f"cn-v16-advisory-invalid-{max_candidates:02d}"
    with pytest.raises(AdvisoryError, match="max_candidates must be within 1..50"):
        runtime.prepare_advisory(
            run_id=run_id,
            max_candidates=max_candidates,
            top_k=12,
        )
    assert not (root / run_id).exists()


def test_fundamental_candidate_semantics_match_mining_rank_blends():
    frame = pd.DataFrame(
        {
            "fin_roe": [3.0, 1.0, 2.0],
            "fin_ocf_to_profit": [1.0, 3.0, 2.0],
            "fin_debt_to_assets": [0.4, 0.2, 0.3],
        },
        index=["a", "b", "c"],
    )
    actual = compute_fundamental_factor_signals(frame)
    expected_formula = (
        frame["fin_ocf_to_profit"]
        .rank(pct=True)
        .mul(0.65)
        .add((-frame["fin_debt_to_assets"]).rank(pct=True).mul(0.35))
    )
    expected_quality = (
        frame["fin_roe"].rank(pct=True).add(frame["fin_ocf_to_profit"].rank(pct=True))
    )
    pd.testing.assert_series_equal(actual[FORMULA_FACTOR_NAME], expected_formula)
    pd.testing.assert_series_equal(actual[QUALITY_FACTOR_NAME], expected_quality)


def test_llm_response_requires_exact_bindings_and_rejects_action_labels():
    request = _request()
    request_sha = "a" * 64
    validated = validate_llm_response(
        _response(request_sha),
        request=request,
        request_file_sha256=request_sha,
        model_id=provider.OPENAI_MODEL,
        prompt_sha256=provider.PROMPT_SHA256,
        response_schema_sha256=provider.RESPONSE_SCHEMA_SHA256,
    )
    assert list(validated) == ["000001.SZ"]
    with pytest.raises(AdvisoryError, match="prohibited publishable text"):
        validate_llm_response(
            _response(request_sha, rationale="BUY because the factor is strong."),
            request=request,
            request_file_sha256=request_sha,
            model_id=provider.OPENAI_MODEL,
            prompt_sha256=provider.PROMPT_SHA256,
            response_schema_sha256=provider.RESPONSE_SCHEMA_SHA256,
        )


def test_llm_request_supports_explicit_codex_identity():
    request = provider.build_llm_request(
        evidence={"items": _request()["items"]},
        evidence_file_sha256="e" * 64,
        model_id=provider.CODEX_DELEGATED_MODEL,
    )
    assert request["model_id"] == provider.CODEX_DELEGATED_MODEL
    with pytest.raises(AdvisoryError, match="unsupported advisory LLM request model"):
        provider.build_llm_request(
            evidence={"items": _request()["items"]},
            evidence_file_sha256="e" * 64,
            model_id="unbound-model",
        )


def test_openai_transport_is_fixed_https_no_tools(monkeypatch):
    request = _request()
    request_sha = "b" * 64
    model_output = _response(request_sha)
    api_response = {
        "id": "resp_test",
        "status": "completed",
        "model": provider.OPENAI_MODEL,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(model_output, ensure_ascii=False),
                    }
                ],
            }
        ],
    }
    encoded_response = json.dumps(api_response).encode()
    captured: dict = {}

    class FakeResponse:
        status = 200

        def __init__(self):
            self.offset = 0

        def getheader(self, name):
            return "application/json" if name == "Content-Type" else None

        def read(self, size):
            chunk = encoded_response[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    class FakeConnection:
        def __init__(self, host, port, **kwargs):
            captured["host"] = host
            captured["port"] = port
            captured["kwargs"] = kwargs

        def request(self, method, path, body, headers):
            captured.update(method=method, path=path, body=body, headers=headers)

        def getresponse(self):
            return FakeResponse()

        def close(self):
            captured["closed"] = True

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setattr(provider.http.client, "HTTPSConnection", FakeConnection)
    normalized, receipt = provider.call_openai_responses(
        request=request,
        request_file_sha256=request_sha,
    )
    sent = json.loads(captured["body"])
    assert (captured["host"], captured["port"], captured["path"]) == (
        "api.openai.com",
        443,
        "/v1/responses",
    )
    assert sent["store"] is False
    assert sent["tools"] == []
    assert sent["tool_choice"] == "none"
    assert sent["max_output_tokens"] == provider.MAX_OUTPUT_TOKENS
    assert "test-secret" not in captured["body"].decode()
    assert normalized == model_output
    assert receipt["provider_response_id"] == "resp_test"
    assert receipt["max_output_tokens"] == provider.MAX_OUTPUT_TOKENS


def test_openai_transport_rejects_oversized_request_before_network(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setattr(provider, "MAX_PROVIDER_REQUEST_BYTES", 1)
    monkeypatch.setattr(
        provider.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network must not be initialized")
        ),
    )
    with pytest.raises(contracts.AdvisoryProviderError, match="request exceeded size limit"):
        provider.call_openai_responses(
            request=_request(),
            request_file_sha256="b" * 64,
        )


def test_provider_interval_side_effect_is_rejected(monkeypatch):
    before = {
        "policy": "test",
        "entry_count": 1,
        "inventory_sha256": "1" * 64,
        "git": {"status": "a"},
        "entries": {"results/v16/formal.json": {"identity": "sha256:a"}},
    }
    after = {
        **before,
        "inventory_sha256": "2" * 64,
        "entries": {"results/v16/formal.json": {"identity": "sha256:b"}},
    }
    inventories = iter([before, after])
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: next(inventories))
    monkeypatch.setattr(
        runtime,
        "call_openai_responses",
        lambda **_kwargs: ({"reviews": []}, {"schema_version": "receipt"}),
    )
    with pytest.raises(
        contracts.AdvisorySideEffectError,
        match="results/v16/formal.json",
    ):
        runtime._call_openai_guarded(
            request=_request(),
            request_file_sha256="a" * 64,
        )


def test_bound_artifact_missing_fails_as_advisory_state_error(tmp_path):
    state = {
        "artifacts": {
            "branch_evidence": {
                "path": "branch_evidence.json",
                "sha256": "a" * 64,
            }
        }
    }
    with pytest.raises(
        contracts.AdvisoryStateError,
        match="advisory artifact unavailable: branch_evidence",
    ):
        runtime._read_bound_artifact(tmp_path, state, "branch_evidence")


def test_provider_failure_still_checks_side_effect_guard(monkeypatch):
    guard = {
        "policy": "test",
        "entry_count": 0,
        "inventory_sha256": "0" * 64,
        "git": {},
        "entries": {},
    }
    inventories = iter([guard, guard])
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: next(inventories))
    monkeypatch.setattr(
        runtime,
        "call_openai_responses",
        lambda **_kwargs: (_ for _ in ()).throw(
            contracts.AdvisoryProviderError("provider timeout")
        ),
    )
    with pytest.raises(contracts.AdvisoryProviderError, match="provider timeout"):
        runtime._call_openai_guarded(
            request=_request(),
            request_file_sha256="a" * 64,
        )


def _fake_inputs() -> tuple[dict, dict]:
    symbols = [f"00000{index}.SZ" for index in range(1, 7)]
    rows = []
    items = []
    for index, symbol in enumerate(symbols, start=1):
        quant = 0.8 - index * 0.05
        fundamental = 0.65 - index * 0.02
        rows.append(
            {
                "symbol": symbol,
                "quant_raw": quant,
                "fundamental_raw": fundamental,
                "macro_raw": -0.1,
            }
        )
        fact_id = f"{symbol}:quant:fixture"
        items.append(
            {
                "symbol": symbol,
                "name": f"公司{index}",
                "industry": "测试行业",
                "quant_raw": quant,
                "fundamental_raw": fundamental,
                "macro_raw": -0.1,
                "fact_ids": [fact_id],
                "facts": [
                    {
                        "id": fact_id,
                        "branch": "quant",
                        "metric": "fixture",
                        "value": quant,
                    }
                ],
            }
        )
    bindings = {"trade_date": "20260717", "fixture_sha256": "f" * 64}
    return (
        {
            "schema_version": contracts.FACTOR_BUNDLE_SCHEMA,
            "source_bindings": bindings,
            "factor_family_count": 5,
            "common_domain_count": 600,
            "rows": rows,
        },
        {
            "schema_version": contracts.PREPARED_EVIDENCE_SCHEMA,
            "source_bindings": bindings,
            "items": items,
        },
    )


def test_runtime_reaches_human_decision_without_execution_authority(tmp_path, monkeypatch):
    root = tmp_path / "results" / "v16_operator_advisory"
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", root)
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    fake_inputs = _fake_inputs()
    monkeypatch.setattr(runtime, "build_deterministic_inputs", lambda **_kwargs: fake_inputs)
    monkeypatch.setattr(
        runtime,
        "load_input_manifest",
        lambda: {
            "schema_version": contracts.INPUT_MANIFEST_SCHEMA,
            "research_waivers": ["calibration_not_run"],
            "non_waivable_gates": ["source_artifact_integrity"],
        },
    )
    guard = {
        "policy": "test",
        "entry_count": 0,
        "inventory_sha256": "0" * 64,
        "git": {},
        "entries": {},
    }
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: guard)

    prepared = runtime.prepare_advisory(
        run_id="cn-v16-advisory-test-0001",
        max_candidates=6,
        top_k=6,
    )
    run_dir = Path(prepared["run_directory"])
    request = json.loads((run_dir / "llm_request.json").read_text())
    request_sha = hashlib.sha256((run_dir / "llm_request.json").read_bytes()).hexdigest()
    response = {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_sha,
        "model_id": provider.OPENAI_MODEL,
        "prompt_sha256": provider.PROMPT_SHA256,
        "response_schema_sha256": provider.RESPONSE_SCHEMA_SHA256,
        "reviews": [
            {
                "symbol": item["symbol"],
                "raw_score": float(item["quant_raw"]),
                "confidence": 0.8,
                "rationale": "Supported by the sealed fact.",
                "evidence_ids": item["fact_ids"],
                "risks": ["Source data can change."],
            }
            for item in request["items"]
        ],
    }
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    received = runtime.receive_advisory_response(
        run_id=prepared["run_id"],
        response_path=response_path,
        expected_state_sha256=prepared["state_sha256"],
    )
    completed = runtime.finalize_advisory(
        run_id=prepared["run_id"],
        expected_state_sha256=received["state_sha256"],
    )
    report = json.loads(Path(completed["report_path"]).read_text())
    shares = [row["research_share"] for row in report["ranked_candidates"]]
    assert completed["state"] == contracts.STATE_ADVISORY_COMPLETE
    assert completed["provider_mode"] == "external_file"
    assert completed["provider_receipt_present"] is False
    assert max(shares) <= 0.2
    assert round(sum(shares) + report["allocation_summary"]["unallocated_cash_share"], 8) == 1.0
    assert report["branch_policy"]["branch_shares"] == contracts.BRANCH_SHARES
    assert report["llm_response_provenance"] == {
        "requested_provider_mode": "openai",
        "provider_mode": "external_file",
        "model_id": provider.OPENAI_MODEL,
        "provider_receipt_present": False,
        "provider_receipt_sha256": "",
    }
    assert report["authority"] == {
        "broker_enabled": False,
        "dashboard_activation_changed": False,
        "factor_registry_changed": False,
        "formal_v16_activation_changed": False,
        "new_risk_authorized": False,
        "production_authority": False,
        "production_pointer_changed": False,
    }
    decided = runtime.record_advisory_decision(
        run_id=prepared["run_id"],
        decision="DEFERRED",
        expected_state_sha256=completed["state_sha256"],
    )
    assert decided["state"] == contracts.STATE_DECISION_RECORDED


def test_provider_resume_reuses_prepared_run_after_failure(tmp_path, monkeypatch):
    root = tmp_path / "results" / "v16_operator_advisory"
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", root)
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    fake_inputs = _fake_inputs()
    monkeypatch.setattr(runtime, "build_deterministic_inputs", lambda **_kwargs: fake_inputs)
    monkeypatch.setattr(
        runtime,
        "load_input_manifest",
        lambda: {
            "schema_version": contracts.INPUT_MANIFEST_SCHEMA,
            "research_waivers": ["calibration_not_run"],
            "non_waivable_gates": ["source_artifact_integrity"],
        },
    )
    guard = {
        "policy": "test",
        "entry_count": 0,
        "inventory_sha256": "0" * 64,
        "git": {},
        "entries": {},
    }
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: guard)

    prepared = runtime.prepare_advisory(
        run_id="cn-v16-advisory-resume-0001",
        max_candidates=6,
        top_k=6,
    )
    monkeypatch.setattr(
        runtime,
        "_call_openai_guarded",
        lambda **_kwargs: (_ for _ in ()).throw(
            contracts.AdvisoryProviderError("provider timeout")
        ),
    )
    with pytest.raises(contracts.AdvisoryProviderError, match="provider timeout"):
        runtime.resume_advisory_provider(
            run_id=prepared["run_id"],
            expected_state_sha256=prepared["state_sha256"],
        )
    unchanged = runtime.advisory_status(run_id=prepared["run_id"])
    assert unchanged["state"] == contracts.STATE_LLM_REQUEST_READY
    assert unchanged["state_sha256"] == prepared["state_sha256"]

    run_dir = Path(prepared["run_directory"])
    request = json.loads((run_dir / "llm_request.json").read_text())
    request_sha = hashlib.sha256((run_dir / "llm_request.json").read_bytes()).hexdigest()
    response = {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_sha,
        "model_id": provider.OPENAI_MODEL,
        "prompt_sha256": provider.PROMPT_SHA256,
        "response_schema_sha256": provider.RESPONSE_SCHEMA_SHA256,
        "reviews": [
            {
                "symbol": item["symbol"],
                "raw_score": float(item["quant_raw"]),
                "confidence": 0.8,
                "rationale": "Supported by the sealed fact.",
                "evidence_ids": item["fact_ids"],
                "risks": ["Source data can change."],
            }
            for item in request["items"]
        ],
    }
    monkeypatch.setattr(
        runtime,
        "_call_openai_guarded",
        lambda **_kwargs: (
            response,
            {"schema_version": "v16.operator-advisory-provider-receipt.v1"},
        ),
    )
    completed = runtime.resume_advisory_provider(
        run_id=prepared["run_id"],
        expected_state_sha256=prepared["state_sha256"],
    )
    report = json.loads(Path(completed["report_path"]).read_text())
    assert completed["state"] == contracts.STATE_ADVISORY_COMPLETE
    assert completed["provider_mode"] == "openai"
    assert completed["provider_receipt_present"] is True
    assert report["llm_response_provenance"] == {
        "requested_provider_mode": "openai",
        "provider_mode": "openai",
        "model_id": provider.OPENAI_MODEL,
        "provider_receipt_present": True,
        "provider_receipt_sha256": completed["provider_receipt_sha256"],
    }


def test_codex_delegated_response_is_bound_and_receipted(tmp_path, monkeypatch):
    root = tmp_path / "results" / "v16_operator_advisory"
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", root)
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "build_deterministic_inputs", lambda **_kwargs: _fake_inputs())
    monkeypatch.setattr(
        runtime,
        "load_input_manifest",
        lambda: {
            "schema_version": contracts.INPUT_MANIFEST_SCHEMA,
            "research_waivers": ["calibration_not_run"],
            "non_waivable_gates": ["source_artifact_integrity"],
        },
    )
    guard = {
        "policy": "test",
        "entry_count": 0,
        "inventory_sha256": "0" * 64,
        "git": {},
        "entries": {},
    }
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: guard)
    monkeypatch.setattr(
        runtime,
        "call_openai_responses",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("external API forbidden")),
    )

    prepared = runtime.prepare_advisory(
        run_id="cn-v16-advisory-codex-0001",
        max_candidates=6,
        top_k=6,
        llm_backend="codex",
    )
    run_dir = Path(prepared["run_directory"])
    request = json.loads((run_dir / "llm_request.json").read_text())
    request_sha = hashlib.sha256((run_dir / "llm_request.json").read_bytes()).hexdigest()
    response = {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_sha,
        "model_id": provider.CODEX_DELEGATED_MODEL,
        "prompt_sha256": provider.PROMPT_SHA256,
        "response_schema_sha256": provider.RESPONSE_SCHEMA_SHA256,
        "reviews": [
            {
                "symbol": item["symbol"],
                "raw_score": float(item["quant_raw"]),
                "confidence": 0.8,
                "rationale": "Supported by the sealed fact.",
                "evidence_ids": item["fact_ids"],
                "risks": ["Source data can change."],
            }
            for item in request["items"]
        ],
    }
    response_path = tmp_path / "codex-response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    received = runtime.receive_advisory_response(
        run_id=prepared["run_id"],
        response_path=response_path,
        response_source="codex_delegated",
        expected_state_sha256=prepared["state_sha256"],
    )
    receipt = json.loads((run_dir / "provider_receipt.json").read_text())
    response_sha = hashlib.sha256((run_dir / "llm_response.json").read_bytes()).hexdigest()
    assert received["provider_mode"] == "codex_delegated"
    assert received["requested_provider_mode"] == "codex_delegated"
    assert receipt["reviewer"] == "codex_delegated_reviewer"
    assert receipt["request_sha256"] == request_sha
    assert receipt["response_sha256"] == response_sha
    assert receipt["external_provider_api_called"] is False
    assert receipt["tools"] == []

    completed = runtime.finalize_advisory(
        run_id=prepared["run_id"],
        expected_state_sha256=received["state_sha256"],
    )
    report = json.loads(Path(completed["report_path"]).read_text())
    assert report["llm_response_provenance"] == {
        "requested_provider_mode": "codex_delegated",
        "provider_mode": "codex_delegated",
        "model_id": provider.CODEX_DELEGATED_MODEL,
        "provider_receipt_present": True,
        "provider_receipt_sha256": completed["provider_receipt_sha256"],
    }


def test_codex_source_rejects_openai_bound_request(tmp_path, monkeypatch):
    root = tmp_path / "results" / "v16_operator_advisory"
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", root)
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "build_deterministic_inputs", lambda **_kwargs: _fake_inputs())
    monkeypatch.setattr(
        runtime,
        "load_input_manifest",
        lambda: {
            "schema_version": contracts.INPUT_MANIFEST_SCHEMA,
            "research_waivers": ["calibration_not_run"],
            "non_waivable_gates": ["source_artifact_integrity"],
        },
    )
    guard = {
        "policy": "test",
        "entry_count": 0,
        "inventory_sha256": "0" * 64,
        "git": {},
        "entries": {},
    }
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: guard)
    prepared = runtime.prepare_advisory(
        run_id="cn-v16-advisory-codex-reject-0001",
        max_candidates=6,
        top_k=6,
    )
    response_path = tmp_path / "wrong-source-response.json"
    response_path.write_text(json.dumps(_response(prepared["state_sha256"])), encoding="utf-8")
    with pytest.raises(
        contracts.AdvisoryStateError,
        match="Codex response does not match prepared request",
    ):
        runtime.receive_advisory_response(
            run_id=prepared["run_id"],
            response_path=response_path,
            response_source="codex_delegated",
            expected_state_sha256=prepared["state_sha256"],
        )


def test_codex_bound_request_rejects_external_file_source(tmp_path, monkeypatch):
    root = tmp_path / "results" / "v16_operator_advisory"
    monkeypatch.setattr(contracts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(contracts, "ADVISORY_ROOT", root)
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "build_deterministic_inputs", lambda **_kwargs: _fake_inputs())
    monkeypatch.setattr(
        runtime,
        "load_input_manifest",
        lambda: {
            "schema_version": contracts.INPUT_MANIFEST_SCHEMA,
            "research_waivers": ["calibration_not_run"],
            "non_waivable_gates": ["source_artifact_integrity"],
        },
    )
    guard = {
        "policy": "test",
        "entry_count": 0,
        "inventory_sha256": "0" * 64,
        "git": {},
        "entries": {},
    }
    monkeypatch.setattr(runtime, "_guard_inventory", lambda: guard)
    prepared = runtime.prepare_advisory(
        run_id="cn-v16-advisory-codex-reject-0002",
        max_candidates=6,
        top_k=6,
        llm_backend="codex",
    )
    run_dir = Path(prepared["run_directory"])
    request = json.loads((run_dir / "llm_request.json").read_text())
    request_sha = hashlib.sha256((run_dir / "llm_request.json").read_bytes()).hexdigest()
    response = {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_sha,
        "model_id": provider.CODEX_DELEGATED_MODEL,
        "prompt_sha256": provider.PROMPT_SHA256,
        "response_schema_sha256": provider.RESPONSE_SCHEMA_SHA256,
        "reviews": [
            {
                "symbol": item["symbol"],
                "raw_score": float(item["quant_raw"]),
                "confidence": 0.8,
                "rationale": "Supported by the sealed fact.",
                "evidence_ids": item["fact_ids"],
                "risks": ["Source data can change."],
            }
            for item in request["items"]
        ],
    }
    response_path = tmp_path / "codex-external-response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(
        contracts.AdvisoryStateError,
        match="Codex-bound request requires codex_delegated response source",
    ):
        runtime.receive_advisory_response(
            run_id=prepared["run_id"],
            response_path=response_path,
            expected_state_sha256=prepared["state_sha256"],
        )
