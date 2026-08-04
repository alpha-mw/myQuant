from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from quant_investor.v17_v4_contract.canonical import (
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_runtime.provisional_forward import (
    NO_AUTHORITY,
    ProvisionalForwardError,
    build_provisional_evaluation_receipt,
    build_provisional_forward_label,
    build_provisional_input,
    build_provisional_request,
    run_provisional_forward,
    validate_provisional_artifact,
)
from quant_investor.v17_v4_runtime.run_profiles import RunProfile, profile_definition

CUTOFF = "2026-08-04T07:00:00Z"
SESSION = "2026-08-04"
STRATEGY = "v17-provisional-test"


def _write(path: Path, document: dict[str, object]) -> tuple[str, dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_resource_bytes(document)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), document


def _input(
    tmp_path: Path,
    *,
    role: str,
    payload: dict[str, object],
    available_at: str = CUTOFF,
) -> tuple[dict[str, str], dict[str, object]]:
    document = build_provisional_input(
        role=role,
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        available_at=available_at,
        payload=payload,
    )
    artifact_id = str(document["artifact_id"])
    relative_path = f"data/inputs/{role.lower()}.json"
    byte_sha, _ = _write(tmp_path / relative_path, document)
    return (
        {
            "artifact_id": artifact_id,
            "artifact_version": str(document["version"]),
            "byte_sha256": byte_sha,
            "cutoff": CUTOFF,
            "relative_path": relative_path,
            "semantic_sha256": str(document["semantic_sha256"]),
            "strategy_id": STRATEGY,
        },
        document,
    )


def _rebuild_input(
    document: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    return build_provisional_input(
        role=str(document["role"]),
        strategy_id=str(document["strategy_id"]),
        decision_session=str(document["decision_session"]),
        cutoff=str(document["cutoff"]),
        available_at=str(document["available_at"]),
        payload=payload,
    )


def _fixture(
    tmp_path: Path,
    *,
    industry: bool = False,
    theme: bool = False,
    missing_sparse_value: bool = False,
) -> tuple[str, str, list[dict[str, str]]]:
    interval = {
        "available_at": "2026-08-03T07:00:00Z",
        "interval_id": "listing-interval-v1",
        "valid_from": "2020-01-01",
        "valid_to": None,
    }
    securities = [
        {
            "exchange": exchange,
            "listing_interval_ref": {**interval, "interval_id": f"interval-{ticker}-{exchange}"},
            "pit_ticker": ticker,
            "symbol": f"{ticker}.{exchange}",
        }
        for ticker, exchange in (("000001", "SZ"), ("600000", "SH"), ("920001", "BJ"))
    ]
    factors = [
        {"family": "value", "implementation_sha256": "1" * 64, "name": "book_to_price"},
        {"family": "momentum", "implementation_sha256": "2" * 64, "name": "momentum_20"},
    ]
    symbols = [row["symbol"] for row in securities]
    values: dict[str, dict[str, float]] = {
        "book_to_price": dict(zip(symbols, (1.0, 2.0, 3.0), strict=True)),
        "momentum_20": dict(zip(symbols, (3.0, 1.0, 2.0), strict=True)),
    }
    if missing_sparse_value:
        values["momentum_20"].pop(symbols[-1])
    neutralizers = {
        symbol: {
            "amihud_20d": {"available_at": CUTOFF, "value": str(index + 1)},
            "beta_252d": {"available_at": CUTOFF, "value": str(index + 2)},
            "industry": {"available_at": CUTOFF, "value": f"industry-{index % 2}"},
            "log_market_cap": {"available_at": CUTOFF, "value": str(index + 10)},
        }
        for index, symbol in enumerate(symbols)
    }
    base_inputs = [
        _input(
            tmp_path,
            role="MARKET_POINTER",
            payload={"latest_complete_trade_date": SESSION},
        )[0],
        _input(tmp_path, role="MARKET_MANIFEST", payload={"row_count": 3})[0],
        _input(tmp_path, role="PIT_MEMBERSHIP_POINTER", payload={"generation": "g1"})[0],
        _input(tmp_path, role="PIT_MEMBERSHIP_MANIFEST", payload={"row_count": 3})[0],
    ]
    universe_ref = _input(tmp_path, role="RESEARCH_UNIVERSE", payload={"securities": securities})[0]
    factor_set_ref = _input(tmp_path, role="FACTOR_SET", payload={"selected_factors": factors})[0]
    inputs = [
        *base_inputs,
        universe_ref,
        factor_set_ref,
        _input(
            tmp_path,
            role="QUANT_INPUT",
            payload={
                "factor_implementation_sha256": {
                    "book_to_price": "1" * 64,
                    "momentum_20": "2" * 64,
                },
                "factor_set_artifact_id": factor_set_ref["artifact_id"],
                "factor_values": values,
                "neutralizer_inputs": neutralizers,
                "research_universe_artifact_id": universe_ref["artifact_id"],
            },
        )[0],
    ]
    if industry:
        inputs.append(
            _input(
                tmp_path,
                role="INDUSTRY_CONTEXT",
                payload={"scores": {symbol: str(index) for index, symbol in enumerate(symbols)}},
            )[0]
        )
    if theme:
        inputs.append(
            _input(
                tmp_path,
                role="THEME_EXPOSURE",
                payload={
                    "scores": {symbol: str(index + 1) for index, symbol in enumerate(symbols)}
                },
            )[0]
        )
    quant_ref = next(row for row in inputs if row["relative_path"].endswith("quant_input.json"))
    request = build_provisional_request(
        request_id="provisional-request-test",
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        input_refs=inputs,
        quant_input_ref=quant_ref,
    )
    request_path = "data/requests/provisional.json"
    request_sha, _ = _write(tmp_path / request_path, request)
    return request_path, request_sha, inputs


def _run(tmp_path: Path, **kwargs: bool) -> dict[str, object]:
    request_path, request_sha, _ = _fixture(tmp_path, **kwargs)
    return run_provisional_forward(
        str(tmp_path),
        request_path=request_path,
        request_sha256=request_sha,
    )


def test_core_completes_without_optional_context(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result["run_state"] == "PROVISIONAL_FORWARD_EVIDENCE_ACTIVE"
    assert result["variant_statuses"] == {
        "v17-quant-core": "COMPLETE",
        "v17-quant-plus-industry": "PARTIAL",
        "v17-quant-plus-industry-theme": "UNAVAILABLE",
    }
    assert result["factor_evidence_status"] == "ACCUMULATING"
    assert result["regime_conditioned_status"] == "UNAVAILABLE"
    assert result["research_runtime_available"] is True
    assert result["research_runtime_default"] is False
    assert result["security_master_status"] == "UNAVAILABLE"
    assert result["provisional_identity_status"] == "RUN_SCOPED_REPLAYABLE"
    assert result["source_admissibility_status"] == "DEGRADED_BUT_REPLAYABLE"
    assert result["authority"] == NO_AUTHORITY
    assert result["production_governance_eligible"] is False


def test_three_variants_are_independent(tmp_path: Path) -> None:
    assert _run(tmp_path / "industry", industry=True)["variant_statuses"] == {
        "v17-quant-core": "COMPLETE",
        "v17-quant-plus-industry": "COMPLETE",
        "v17-quant-plus-industry-theme": "PARTIAL",
    }
    assert _run(tmp_path / "all", industry=True, theme=True)["variant_statuses"] == {
        "v17-quant-core": "COMPLETE",
        "v17-quant-plus-industry": "COMPLETE",
        "v17-quant-plus-industry-theme": "COMPLETE",
    }


def test_sparse_factor_does_not_drop_full_universe(tmp_path: Path) -> None:
    result = _run(tmp_path, missing_sparse_value=True)
    manifest_ref = result["artifact_manifest_ref"]
    manifest = json.loads((tmp_path / manifest_ref["relative_path"]).read_bytes())
    observation_ref = next(
        row for row in manifest["artifact_refs"] if "quant-observation" in row["relative_path"]
    )
    observation = json.loads((tmp_path / observation_ref["relative_path"]).read_bytes())
    rows = json.loads(observation["payload"])["observation_rows"]
    assert len({row["symbol"] for row in rows}) == 3
    missing = [row for row in rows if row["availability"] == "MISSING_VALUE"]
    assert len(missing) == 1
    assert missing[0]["raw_exposure"] is None
    assert missing[0]["missing_reason"] == "MISSING_VALUE"


def test_run_is_idempotent_and_request_conflict_fails(tmp_path: Path) -> None:
    request_path, request_sha, _ = _fixture(tmp_path)
    first = run_provisional_forward(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    second = run_provisional_forward(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    assert first == second
    request = json.loads((tmp_path / request_path).read_bytes())
    request["created_at"] = "2026-08-04T07:00:01Z"
    request.pop("semantic_sha256")
    request = seal_semantic(request)
    changed_sha, _ = _write(tmp_path / request_path, request)
    with pytest.raises(ProvisionalForwardError, match="PROVISIONAL_EXACT_ONCE_REQUEST_CONFLICT"):
        run_provisional_forward(
            str(tmp_path), request_path=request_path, request_sha256=changed_sha
        )


def test_sha_future_symlink_and_hardlink_fail_closed(tmp_path: Path) -> None:
    request_path, request_sha, refs = _fixture(tmp_path / "sha")
    with pytest.raises(ProvisionalForwardError, match="PROVISIONAL_REQUEST_SHA_MISMATCH"):
        run_provisional_forward(
            str(tmp_path / "sha"), request_path=request_path, request_sha256="0" * 64
        )
    with pytest.raises(ProvisionalForwardError, match="PROVISIONAL_PIT_CUTOFF_VIOLATION"):
        build_provisional_input(
            role="MARKET_POINTER",
            strategy_id=STRATEGY,
            decision_session=SESSION,
            cutoff=CUTOFF,
            available_at="2026-08-04T07:00:01Z",
            payload={"latest_complete_trade_date": SESSION},
        )
    symlink_root = tmp_path / "symlink"
    request_path, request_sha, refs = _fixture(symlink_root)
    target = symlink_root / refs[0]["relative_path"]
    moved = target.with_suffix(".target")
    target.rename(moved)
    target.symlink_to(moved.name)
    with pytest.raises(ProvisionalForwardError, match="PROVISIONAL_SYMLINK_REJECTED"):
        run_provisional_forward(
            str(symlink_root), request_path=request_path, request_sha256=request_sha
        )
    hardlink_root = tmp_path / "hardlink"
    request_path, request_sha, refs = _fixture(hardlink_root)
    target = hardlink_root / refs[0]["relative_path"]
    os.link(target, target.with_suffix(".alias"))
    with pytest.raises(
        ProvisionalForwardError,
        match="PROVISIONAL_HARDLINK_OR_NONREGULAR_REJECTED",
    ):
        run_provisional_forward(
            str(hardlink_root), request_path=request_path, request_sha256=request_sha
        )


def test_invalid_downstream_context_preserves_quant(tmp_path: Path) -> None:
    request_path, request_sha, refs = _fixture(tmp_path, industry=True)
    industry_ref = next(
        row for row in refs if row["relative_path"].endswith("industry_context.json")
    )
    industry_path = tmp_path / industry_ref["relative_path"]
    industry = json.loads(industry_path.read_bytes())
    industry_payload = json.loads(industry["payload"])
    industry_payload["scores"].pop("920001.BJ")
    industry = _rebuild_input(industry, industry_payload)
    industry_ref["artifact_id"] = industry["artifact_id"]
    industry_ref["semantic_sha256"] = industry["semantic_sha256"]
    industry_ref["byte_sha256"], _ = _write(industry_path, industry)
    request = json.loads((tmp_path / request_path).read_bytes())
    request["input_refs"] = sorted(refs, key=lambda row: (row["relative_path"], row["byte_sha256"]))
    request.pop("semantic_sha256")
    request = seal_semantic(request)
    request_sha, _ = _write(tmp_path / request_path, request)
    with pytest.raises(ProvisionalForwardError) as caught:
        run_provisional_forward(
            str(tmp_path), request_path=request_path, request_sha256=request_sha
        )
    assert caught.value.code == "PROVISIONAL_INDUSTRY_CONTEXT_UNIVERSE_MISMATCH"
    assert any(
        "quant-observation" in row["relative_path"] for row in caught.value.preserved_artifact_refs
    )
    assert not list(tmp_path.glob("**/manifest-*.json"))


def test_label_and_research_evaluation_remain_nonproduction() -> None:
    ref = {
        "artifact_id": "observation",
        "artifact_version": "myquant.v17.v4.provisional-forward-artifact.v1",
        "byte_sha256": "1" * 64,
        "cutoff": CUTOFF,
        "relative_path": "results/v17_v4_shadow/observation.json",
        "semantic_sha256": "2" * 64,
        "strategy_id": STRATEGY,
    }
    pending = build_provisional_forward_label(
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        horizon_sessions=20,
        observation_ref=ref,
    )
    assert json.loads(validate_provisional_artifact(pending)["payload"])["status"] == "PENDING"
    receipt = build_provisional_evaluation_receipt(
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        receipt_kind="VARIANT_COMPARISON",
        subject_id="core-vs-industry",
        evidence_refs=[ref],
        metrics={"rank_ic": "0.12", "coverage": "0.95"},
    )
    assert receipt["research_evaluation_eligible"] is True
    assert receipt["production_governance_eligible"] is False
    receipt_payload = json.loads(receipt["payload"])
    assert receipt_payload["factor_governance_write"] is False
    assert receipt_payload["formal_activation_eligible"] is False
    assert receipt_payload["promotion_eligible"] is False


def test_matured_label_requires_exact_future_session_closure() -> None:
    observation_ref = {
        "artifact_id": "observation",
        "artifact_version": "myquant.v17.v4.provisional-forward-artifact.v1",
        "byte_sha256": "1" * 64,
        "cutoff": CUTOFF,
        "relative_path": "results/v17_v4_shadow/observation.json",
        "semantic_sha256": "2" * 64,
        "strategy_id": STRATEGY,
    }
    calendar_ref = {
        **observation_ref,
        "artifact_id": "calendar",
        "relative_path": "data/calendar.json",
    }
    future_sessions = [f"2026-08-{day:02d}" for day in range(5, 25)]
    matured = build_provisional_forward_label(
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        horizon_sessions=20,
        observation_ref=observation_ref,
        label_session="2026-08-24",
        future_sessions=future_sessions,
        calendar_ref=calendar_ref,
        rows=[{"provisional_security_key": "provisional-key", "market_excess_return": "0.1"}],
    )
    payload = json.loads(matured["payload"])
    assert payload["future_sessions"] == future_sessions
    assert payload["status"] == "MATURED"
    assert payload["label_rows"][0]["industry_excess_return"] == {
        "status": "UNAVAILABLE",
        "value": None,
    }
    with pytest.raises(
        ProvisionalForwardError,
        match="PROVISIONAL_LABEL_FUTURE_WINDOW_VIOLATION",
    ):
        build_provisional_forward_label(
            strategy_id=STRATEGY,
            decision_session=SESSION,
            cutoff=CUTOFF,
            created_at=CUTOFF,
            horizon_sessions=20,
            observation_ref=observation_ref,
            label_session="2026-08-24",
            future_sessions=future_sessions[:-1],
            calendar_ref=calendar_ref,
        )


def test_factor_implementation_and_market_session_mismatch_fail_closed(tmp_path: Path) -> None:
    request_path, request_sha, refs = _fixture(tmp_path / "implementation")
    quant_ref = next(row for row in refs if row["relative_path"].endswith("quant_input.json"))
    quant_path = tmp_path / "implementation" / quant_ref["relative_path"]
    quant = json.loads(quant_path.read_bytes())
    quant_payload = json.loads(quant["payload"])
    quant_payload["factor_implementation_sha256"]["momentum_20"] = "3" * 64
    quant = _rebuild_input(quant, quant_payload)
    quant_ref["artifact_id"] = quant["artifact_id"]
    quant_ref["semantic_sha256"] = quant["semantic_sha256"]
    quant_ref["byte_sha256"], _ = _write(quant_path, quant)
    request = json.loads((tmp_path / "implementation" / request_path).read_bytes())
    request["input_refs"] = sorted(refs, key=lambda row: (row["relative_path"], row["byte_sha256"]))
    request["quant_input_ref"] = quant_ref
    request.pop("semantic_sha256")
    request = seal_semantic(request)
    request_sha, _ = _write(tmp_path / "implementation" / request_path, request)
    with pytest.raises(
        ProvisionalForwardError,
        match="PROVISIONAL_FACTOR_IMPLEMENTATION_SHA_MISMATCH",
    ):
        run_provisional_forward(
            str(tmp_path / "implementation"),
            request_path=request_path,
            request_sha256=request_sha,
        )
    request_path, request_sha, refs = _fixture(tmp_path / "session")
    market_ref = next(row for row in refs if row["relative_path"].endswith("market_pointer.json"))
    market_path = tmp_path / "session" / market_ref["relative_path"]
    market = json.loads(market_path.read_bytes())
    market_payload = json.loads(market["payload"])
    market_payload["latest_complete_trade_date"] = "2026-08-01"
    market = _rebuild_input(market, market_payload)
    market_ref["artifact_id"] = market["artifact_id"]
    market_ref["semantic_sha256"] = market["semantic_sha256"]
    market_ref["byte_sha256"], _ = _write(market_path, market)
    request = json.loads((tmp_path / "session" / request_path).read_bytes())
    request["input_refs"] = sorted(refs, key=lambda row: (row["relative_path"], row["byte_sha256"]))
    request.pop("semantic_sha256")
    request = seal_semantic(request)
    request_sha, _ = _write(tmp_path / "session" / request_path, request)
    with pytest.raises(
        ProvisionalForwardError,
        match="PROVISIONAL_DECISION_SESSION_MARKET_MISMATCH",
    ):
        run_provisional_forward(
            str(tmp_path / "session"),
            request_path=request_path,
            request_sha256=request_sha,
        )


def test_profile_and_cli_have_no_mutable_overrides() -> None:
    from quant_investor.v17_v4_runtime.cli_provisional import _parser

    assert profile_definition(RunProfile.PROVISIONAL_FORWARD_EVIDENCE).profile.value == (
        "PROVISIONAL_FORWARD_EVIDENCE"
    )
    parser = _parser()
    flags = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }
    assert flags == {"--workspace-root", "--request-path", "--request-sha256"}


def test_pointer_drift_is_a_limitation_not_observation_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.v17_v4_runtime.provisional_forward as runtime

    request_path, request_sha, refs = _fixture(tmp_path)
    pointer_ref = next(row for row in refs if row["relative_path"].endswith("market_pointer.json"))
    pointer_path = tmp_path / pointer_ref["relative_path"]
    original = runtime._stable_workspace_read
    reads = 0

    def drifting_read(root: Path, relative_path: str) -> bytes:
        nonlocal reads
        if relative_path == pointer_ref["relative_path"]:
            reads += 1
            if reads == 2:
                changed = json.loads(pointer_path.read_bytes())
                changed_payload = json.loads(changed["payload"])
                changed_payload["generation"] = "changed-after-freeze"
                _write(pointer_path, _rebuild_input(changed, changed_payload))
        return original(root, relative_path)

    monkeypatch.setattr(runtime, "_stable_workspace_read", drifting_read)
    result = run_provisional_forward(
        str(tmp_path), request_path=request_path, request_sha256=request_sha
    )
    manifest = json.loads(
        (tmp_path / result["artifact_manifest_ref"]["relative_path"]).read_bytes()
    )
    limitation_ref = next(
        row for row in manifest["artifact_refs"] if "/limitations-" in row["relative_path"]
    )
    limitation = json.loads((tmp_path / limitation_ref["relative_path"]).read_bytes())
    assert (
        "CURRENT_POINTER_CHANGED_DURING_RUN"
        in json.loads(limitation["payload"])["limitation_codes"]
    )
    assert result["factor_evidence_status"] == "ACCUMULATING"


def test_outputs_never_contain_permanent_identity_or_authority_escalation(tmp_path: Path) -> None:
    result = _run(tmp_path)
    manifest = json.loads(
        (tmp_path / result["artifact_manifest_ref"]["relative_path"]).read_bytes()
    )
    for reference in manifest["artifact_refs"]:
        raw = (tmp_path / reference["relative_path"]).read_text(encoding="utf-8")
        assert '"security_id"' not in raw
        assert '"factor_tier"' not in raw
        assert '"lifecycle_action"' not in raw
        document = json.loads(raw)
        assert document["authority"] == NO_AUTHORITY
