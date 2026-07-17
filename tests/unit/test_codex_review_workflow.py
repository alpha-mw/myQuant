from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from quant_investor.codex_review import (
    DifferentBytesError,
    ReviewState,
    StateConflictError,
    StrictJSONError,
    export_review_request,
    prepare_stage1_run,
    receive_review_response,
    resume_review,
    review_status,
    seal_json_payload,
    symbol_set_sha256,
    validate_review_response,
)
from quant_investor.codex_review.storage import parse_strict_json_bytes, sha256_file
from quant_investor.codex_review.models import MenuSeal
from quant_investor.v16.stage1_contract import PITFactRow, build_stage1_fact_package

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_TIME = datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc)


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, expires_at: datetime | None = None) -> dict[str, object]:
    root = tmp_path / "review"
    config = _write(tmp_path / "config.json", '{"decision_protocol":"v16"}\n')
    prompt = _write(tmp_path / "prompt.md", "strict v16 review\n")
    pointer = _write(tmp_path / "_latest.json", '{"generation":"g1"}\n')
    expiry = expires_at or BASE_TIME + timedelta(days=1)
    rows = [
        PITFactRow(
            symbol=symbol,
            stratum="funnel" if symbol in {"AAA", "BBB"} else "outside_funnel",
            eligibility_receipt_sha256=character * 64,
            formal_quant_score=score,
            quant_facts={"fact_id": f"{symbol}-q"},
            fundamental_facts={"fact_id": f"{symbol}-f"},
            macro_facts={"fact_id": f"{symbol}-m"},
        )
        for symbol, character, score in (
            ("AAA", "a", 0.4),
            ("BBB", "b", 0.2),
            ("CCC", "c", 0.1),
        )
    ]
    package = build_stage1_fact_package(
        rows=rows,
        funnel_symbols=["AAA", "BBB"],
        cutoff_at=BASE_TIME.isoformat(),
        expires_at=expiry.isoformat(),
        pit_pointer_sha256=sha256_file(pointer),
    )
    prepared = prepare_stage1_run(
        root=root,
        run_id="run-1",
        payload=package,
        config_path=config,
        prompt_path=prompt,
        model_id="gpt-v16-review",
        pit_pointer_path=pointer,
        repo_path=REPO_ROOT,
        expected_state_sha256="EMPTY",
        now=BASE_TIME,
    )
    return {
        "root": root,
        "config": config,
        "prompt": prompt,
        "pointer": pointer,
        "package": package,
        "state": prepared,
    }


def _read_private_json(path: Path) -> dict[str, object]:
    assert path.stat().st_mode & 0o777 == 0o600
    return json.loads(path.read_text(encoding="utf-8"))


def _stage1_response(request: dict[str, object]) -> dict[str, object]:
    final_symbols = ["AAA", "BBB", "CCC"]
    common = {
        key: request[key]
        for key in (
            "run_id",
            "git_sha",
            "config_path",
            "config_sha256",
            "prompt_path",
            "prompt_sha256",
            "model_id",
            "model_sha256",
            "pit_pointer_path",
            "pit_pointer_sha256",
            "predecessor_sha256",
            "decision_cutoff_at",
            "expires_at",
            "request_sha256",
        )
    }
    payload: dict[str, object] = {
        "schema_version": "codex-review-stage1-response.v1",
        **common,
        "stage": 1,
        "symbol_set": final_symbols,
        "symbol_set_sha256": symbol_set_sha256(final_symbols),
        "supplemental_candidates": [
            {"symbol": "CCC", "retrieval_reason": "补充覆盖漏斗外合格标的。"}
        ],
        "retrieval_evidence": [
            {
                "symbol": symbol,
                "branch": branch,
                "supporting_fact_ids": [f"{symbol}-{branch[0]}"],
                "contradicting_fact_ids": [],
                "conflict_note": "",
            }
            for symbol in final_symbols
            for branch in ("quant", "fundamental", "macro")
        ],
        "llm_verdicts": [
            {
                "symbol": symbol,
                "raw_score": score,
                "confidence": 0.7,
                "supporting_fact_ids": [f"{symbol}-q"],
                "contradicting_fact_ids": [],
                "rationale": "基于封存事实形成第四分支原始判断。",
            }
            for symbol, score in (("AAA", 0.4), ("BBB", 0.3), ("CCC", 0.2))
        ],
    }
    return seal_json_payload(payload, digest_field="response_sha256")


def _menu(state: dict[str, object], *, sealed_at: datetime) -> dict[str, object]:
    def branch_evidence(symbol: str) -> list[dict[str, object]]:
        return [
            {
                "branch": branch,
                "raw_score": 0.5,
                "confidence": 0.8,
                "calibrated_probability": 0.65,
                "evidence_ids": [f"{symbol}-{branch}-formal"],
            }
            for branch in ("quant", "fundamental", "macro", "llm")
        ]

    items = [
        {
            "symbol": "AAA",
            "posterior_win_rate": 0.7,
            "posterior_expected_alpha": 0.04,
            "posterior_edge_after_costs": 0.03,
            "branch_evidence": branch_evidence("AAA"),
            "retrieval_advisory": [],
            "risk_advisory": {
                "severity": "low",
                "flags": [],
                "scenarios": [],
                "suggestions": [],
                "rationale": "波动可控。",
            },
            "existing_weight": 0.2,
            "reference_price": 100.0,
            "existing_shares": 2.0,
        },
        {
            "symbol": "BBB",
            "posterior_win_rate": 0.6,
            "posterior_expected_alpha": 0.03,
            "posterior_edge_after_costs": 0.02,
            "branch_evidence": branch_evidence("BBB"),
            "retrieval_advisory": [],
            "risk_advisory": {
                "severity": "extreme",
                "flags": ["审计风险"],
                "scenarios": ["审计事项恶化"],
                "suggestions": ["持续复核"],
                "rationale": "存在严重审计风险。",
            },
            "existing_weight": 0.0,
            "reference_price": 100.0,
            "existing_shares": 0.0,
        },
    ]
    return seal_json_payload(
        {
            "schema_version": "codex-review-menu.v1",
            "run_id": "run-1",
            "stage1_response_sha256": state["stage1_response_sha256"],
            "symbols": ["AAA", "BBB"],
            "items": items,
            "existing_weights": {"AAA": 0.2, "BBB": 0.0},
            "sealed_at": sealed_at.isoformat().replace("+00:00", "Z"),
        },
        digest_field="menu_sha256",
    )


def _stage2_response(request: dict[str, object]) -> dict[str, object]:
    common = {
        key: request[key]
        for key in (
            "run_id",
            "git_sha",
            "config_path",
            "config_sha256",
            "prompt_path",
            "prompt_sha256",
            "model_id",
            "model_sha256",
            "pit_pointer_path",
            "pit_pointer_sha256",
            "symbol_set",
            "symbol_set_sha256",
            "predecessor_sha256",
            "decision_cutoff_at",
            "expires_at",
            "request_sha256",
            "menu_sha256",
        )
    }
    return seal_json_payload(
        {
            "schema_version": "codex-review-stage2-response.v1",
            **common,
            "stage": 2,
            "verdicts": [
                {
                    "symbol": "AAA",
                    "action": "HOLD",
                    "selected_for_portfolio": True,
                    "target_weight": 0.2,
                    "rationale": "保持已有仓位。",
                    "severe_risks": [],
                    "risk_acceptance_rationale": "",
                },
                {
                    "symbol": "BBB",
                    "action": "BUY",
                    "selected_for_portfolio": True,
                    "target_weight": 0.3,
                    "rationale": "后验净边际支持买入。",
                    "severe_risks": ["审计风险"],
                    "risk_acceptance_rationale": "已审阅并明确接受该研究风险。",
                },
            ],
            "cash_ratio": 0.5,
        },
        digest_field="response_sha256",
    )


def _advance_to_stage2_validated(
    tmp_path: Path, *, invalid_hold_weight: bool = False
) -> tuple[dict[str, object], dict[str, object]]:
    fixture = _fixture(tmp_path)
    state = fixture["state"]
    state = export_review_request(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=1),
    )
    request = _read_private_json(Path(state["export_path"]))
    response_path = tmp_path / "stage1.response.json"
    response_path.write_text(
        json.dumps(_stage1_response(request), ensure_ascii=False), encoding="utf-8"
    )
    state = receive_review_response(
        root=fixture["root"],
        run_id="run-1",
        response_path=response_path,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=2),
    )
    state = validate_review_response(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=3),
    )
    assert state["state"] == ReviewState.S1_VALIDATED.value, state
    menu_path = tmp_path / "menu.json"
    menu_path.write_text(
        json.dumps(_menu(state, sealed_at=BASE_TIME + timedelta(seconds=4)), ensure_ascii=False),
        encoding="utf-8",
    )
    state = resume_review(
        root=fixture["root"],
        run_id="run-1",
        menu_path=menu_path,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=4),
    )
    assert state["state"] == ReviewState.MENU_SEALED.value
    state = resume_review(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=5),
    )
    state = export_review_request(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=6),
    )
    stage2_request = _read_private_json(Path(state["export_path"]))
    response2_path = tmp_path / "stage2.response.json"
    stage2_response = _stage2_response(stage2_request)
    if invalid_hold_weight:
        stage2_response["verdicts"][0]["target_weight"] = 0.25
        stage2_response["cash_ratio"] = 0.45
        stage2_response = seal_json_payload(stage2_response, digest_field="response_sha256")
    response2_path.write_text(
        json.dumps(stage2_response, ensure_ascii=False),
        encoding="utf-8",
    )
    state = receive_review_response(
        root=fixture["root"],
        run_id="run-1",
        response_path=response2_path,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=7),
    )
    state = validate_review_response(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=8),
    )
    if invalid_hold_weight:
        assert state["state"] == ReviewState.BLOCKED.value
    else:
        assert state["state"] == ReviewState.S2_VALIDATED.value
    return fixture, state


def test_full_two_stage_handoff_to_human_authorization(tmp_path: Path) -> None:
    fixture, state = _advance_to_stage2_validated(tmp_path)
    state = resume_review(
        root=fixture["root"],
        run_id="run-1",
        total_capital=1000.0,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=9),
    )
    assert state["state"] == ReviewState.CAPITAL_MAPPED.value
    capital = _read_private_json(Path(fixture["root"]) / "run-1" / str(state["capital_map_path"]))
    assert capital["positions"][0]["raw_target_shares"] == 2.0
    assert capital["positions"][0]["target_shares"] == 2.0
    assert capital["positions"][1]["raw_target_shares"] == 3.0
    assert capital["positions"][1]["target_shares"] == 3.0

    state = resume_review(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=10),
    )
    assert state["state"] == ReviewState.AWAITING_HUMAN_AUTH.value
    receipt = seal_json_payload(
        {
            "schema_version": "codex-review-human-authorization.v1",
            "run_id": "run-1",
            "stage2_response_sha256": state["stage2_response_sha256"],
            "capital_map_sha256": state["capital_map_sha256"],
            "decision": "AUTHORIZED",
            "authorized_by": "Maxwell",
            "authorized_at": (BASE_TIME + timedelta(seconds=11)).isoformat().replace("+00:00", "Z"),
            "expires_at": (BASE_TIME + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "rationale": "人工复核同意该研究资金映射。",
        },
        digest_field="receipt_sha256",
    )
    receipt_path = tmp_path / "authorization.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
    state = resume_review(
        root=fixture["root"],
        run_id="run-1",
        authorization_path=receipt_path,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=11),
    )
    assert state["state"] == ReviewState.AUTHORIZED.value
    assert state["accepted"] is True
    assert os.stat(Path(fixture["root"]) / "run-1" / "state.json").st_mode & 0o777 == 0o600


def test_strict_json_rejects_duplicate_keys_nan_and_unknown_response_fields(
    tmp_path: Path,
) -> None:
    with pytest.raises(StrictJSONError, match="duplicate"):
        parse_strict_json_bytes(b'{"a":1,"a":2}', max_bytes=100)
    with pytest.raises(StrictJSONError, match="non-finite"):
        parse_strict_json_bytes(b'{"a":NaN}', max_bytes=100)

    fixture = _fixture(tmp_path)
    state = export_review_request(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=fixture["state"]["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=1),
    )
    request = _read_private_json(Path(state["export_path"]))
    response = _stage1_response(request)
    response["unknown"] = True
    response = seal_json_payload(response, digest_field="response_sha256")
    path = tmp_path / "bad.response.json"
    path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        receive_review_response(
            root=fixture["root"],
            run_id="run-1",
            response_path=path,
            expected_state_sha256=state["state_file_sha256"],
            now=BASE_TIME + timedelta(seconds=2),
        )


def test_response_same_bytes_is_idempotent_and_different_bytes_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state = export_review_request(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=fixture["state"]["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=1),
    )
    request = _read_private_json(Path(state["export_path"]))
    path = tmp_path / "response.json"
    path.write_text(json.dumps(_stage1_response(request)), encoding="utf-8")
    received = receive_review_response(
        root=fixture["root"],
        run_id="run-1",
        response_path=path,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=2),
    )
    repeated = receive_review_response(
        root=fixture["root"],
        run_id="run-1",
        response_path=path,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=3),
    )
    assert repeated["state_file_sha256"] == received["state_file_sha256"]
    different = tmp_path / "different.json"
    different.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DifferentBytesError, match="new run_id"):
        receive_review_response(
            root=fixture["root"],
            run_id="run-1",
            response_path=different,
            expected_state_sha256=received["state_file_sha256"],
            now=BASE_TIME + timedelta(seconds=4),
        )


def test_invalid_stage2_semantics_transitions_to_blocked(tmp_path: Path) -> None:
    _, state = _advance_to_stage2_validated(tmp_path, invalid_hold_weight=True)
    assert state["state"] == ReviewState.BLOCKED.value
    assert "HOLD must preserve" in state["blockers"][0]


def test_stage2_menu_requires_exact_ordered_four_branch_evidence() -> None:
    value = _menu(
        {"stage1_response_sha256": "a" * 64},
        sealed_at=BASE_TIME,
    )
    MenuSeal.model_validate(value)
    value["items"][0]["branch_evidence"][0:2] = reversed(value["items"][0]["branch_evidence"][0:2])
    with pytest.raises(ValueError, match="exact Q/F/M/LLM order"):
        MenuSeal.model_validate(value)


def test_pit_pointer_drift_transitions_to_blocked(tmp_path: Path) -> None:
    fixture, state = _advance_to_stage2_validated(tmp_path)
    # The good response has already validated; create a fresh run and corrupt its Stage2
    # response through the public schemas would require a new run.  Exercise the common
    # fail-closed guard via PIT drift before the next mutation instead.
    Path(fixture["pointer"]).write_text('{"generation":"g2"}\n', encoding="utf-8")
    blocked = resume_review(
        root=fixture["root"],
        run_id="run-1",
        total_capital=1000.0,
        expected_state_sha256=state["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=9),
    )
    assert blocked["state"] == ReviewState.BLOCKED.value
    assert "pit_pointer_sha256_drift" in blocked["blockers"]


def test_expiry_and_cas_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, expires_at=BASE_TIME + timedelta(seconds=2))
    with pytest.raises(StateConflictError, match="CAS mismatch"):
        export_review_request(
            root=fixture["root"],
            run_id="run-1",
            expected_state_sha256="f" * 64,
            now=BASE_TIME + timedelta(seconds=1),
        )
    expired = export_review_request(
        root=fixture["root"],
        run_id="run-1",
        expected_state_sha256=fixture["state"]["state_file_sha256"],
        now=BASE_TIME + timedelta(seconds=3),
    )
    assert expired["state"] == ReviewState.EXPIRED.value
    assert review_status(root=fixture["root"], run_id="run-1")["state"] == "EXPIRED"
