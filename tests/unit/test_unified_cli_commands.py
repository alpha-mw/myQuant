from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.cli.main import main


def _canonical_request(path: Path, document: dict) -> str:
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


def _line(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    document = json.loads(captured.out)
    assert (
        captured.out
        == json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return document


def _ref(kind: str, artifact_id: str) -> dict[str, str]:
    return {
        "kind": kind,
        "contract_sha256": "1" * 64,
        "artifact_id": artifact_id,
        "semantic_sha256": "2" * 64,
        "byte_sha256": "3" * 64,
    }


def _install_factor_store_boundary(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    import quant_investor.factors.governance as governance
    import quant_investor.system as system

    calls: list[tuple] = []
    composite = {
        "payload": {
            "blockers": [],
            "cycle_state": "PREREGISTERED",
            "terminal": False,
        }
    }
    status = {
        "payload": {
            "blockers": [],
            "readiness": "READY",
        }
    }

    class FactorStoreBoundary:
        def __init__(self, *, system_store: object) -> None:
            assert system_store is not None

        def mine(self, **kwargs: object) -> dict:
            calls.append(("mine", kwargs))
            return composite

        def observe_signal(self, **kwargs: object) -> dict:
            calls.append(("observe_signal", kwargs))
            return composite

        def observe_label(self, **kwargs: object) -> dict:
            calls.append(("observe_label", kwargs))
            return composite

        def evaluate(self, **kwargs: object) -> dict:
            calls.append(("evaluate", kwargs))
            return composite

        def build_status(self, **kwargs: object) -> dict:
            calls.append(("build_status", kwargs))
            return status

    monkeypatch.setattr(
        governance,
        "FactorValidationStore",
        FactorStoreBoundary,
        raising=False,
    )
    monkeypatch.setattr(
        governance,
        "validate_composite_state",
        lambda artifact: artifact,
        raising=False,
    )
    monkeypatch.setattr(
        governance,
        "validate_factor_status",
        lambda artifact: artifact,
        raising=False,
    )
    monkeypatch.setattr(
        system,
        "object_ref_for_artifact",
        lambda artifact: (
            _ref("factor.status", "status-a")
            if artifact is status
            else _ref("factor.composite_state", "composite-a")
        ),
    )
    return calls


def _install_contextual_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_state_ref: dict[str, str] | None,
    result_request_ref: dict[str, str] | None = None,
) -> list[tuple]:
    import quant_investor.system as system

    calls: list[tuple] = []
    request_ref = _ref("system.validation_run_request", "validation-request-a")

    class SystemStoreBoundary:
        def __init__(self, workspace_root: object) -> None:
            calls.append(("init", workspace_root))

        def get_object(self, reference: dict[str, str]) -> dict:
            calls.append(("get_object", reference))
            return {
                "kind": "system.validation_run_request",
                "payload": {"candidate_state_ref": candidate_state_ref},
            }

        def run_validation(self, reference: dict[str, str]) -> dict:
            calls.append(("run_validation", reference))
            return {
                "completion_sha256": "4" * 64,
                "contextual_result_ref": _ref("factor.contextual_validation_result", "context-a"),
                "internal_path": "/must/not/be/projected",
                "outcome": "VALIDATED",
                "validation_attestation_ref": _ref(
                    "system.validation_attestation", "attestation-a"
                ),
                "validation_request_ref": result_request_ref or request_ref,
            }

    monkeypatch.setattr(system, "SystemStore", SystemStoreBoundary)
    monkeypatch.setattr(
        system,
        "validate_object_ref",
        lambda value, **kwargs: dict(value),
    )
    return calls


def test_system_status_complete_when_uninitialized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["system", "status", "--workspace-root", str(tmp_path)])
    system = _line(capsys)

    assert set(system) == {
        "active_generation_id",
        "blockers",
        "calendar_authority_confidence",
        "calendar_authority_route",
        "calendar_source_limitations",
        "capabilities",
        "external_routing_state",
        "fundamental_advisory",
        "status",
    }
    assert system["status"] == "OK"
    assert system["capabilities"]["system"] == "UNINITIALIZED"
    assert system["fundamental_advisory"] is None


def test_system_bootstrap_assemble_uses_exact_request_and_explicit_input_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.factors.governance.production as production

    input_root = tmp_path / "sealed-inputs"
    input_root.mkdir(mode=0o700)
    request_path = tmp_path / "bootstrap-request.json"
    digest = _canonical_request(request_path, {"sealed": True})
    calls: list[dict[str, object]] = []

    def assemble(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {
            "status": "OFFLINE_VERIFIED",
            "generation_id": "1" * 64,
            "active_pointer_write_count": 0,
            "marker_write_count": 0,
        }

    monkeypatch.setattr(production, "assemble_production_bootstrap", assemble)
    main(
        [
            "system",
            "bootstrap-assemble",
            "--workspace-root",
            str(tmp_path),
            "--input-root",
            "sealed-inputs",
            "--request",
            "bootstrap-request.json",
            "--expected-request-sha256",
            digest,
        ]
    )

    assert _line(capsys)["status"] == "OFFLINE_VERIFIED"
    assert calls == [
        {
            "workspace_root": str(tmp_path),
            "input_root": input_root,
            "request_raw": request_path.read_bytes(),
        }
    ]


def test_system_bootstrap_admission_preflight_is_explicit_and_nonauthorizing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.factors.governance.production as production

    input_root = tmp_path / "sealed-inputs"
    input_root.mkdir(mode=0o700)
    request_path = tmp_path / "bootstrap-request.json"
    digest = _canonical_request(request_path, {"sealed": True})
    calls: list[dict[str, object]] = []

    def preflight(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {
            "status": "ADMISSION_PREFLIGHT_ONLY",
            "generation_write_count": 0,
            "active_pointer_write_count": 0,
            "marker_write_count": 0,
        }

    monkeypatch.setattr(production, "prepare_production_bootstrap_admission", preflight)
    main(
        [
            "system",
            "bootstrap-admission-preflight",
            "--workspace-root",
            str(tmp_path),
            "--input-root",
            "sealed-inputs",
            "--request",
            "bootstrap-request.json",
            "--expected-request-sha256",
            digest,
        ]
    )
    assert _line(capsys)["status"] == "ADMISSION_PREFLIGHT_ONLY"
    assert calls == [
        {
            "workspace_root": str(tmp_path),
            "input_root": input_root,
            "request_raw": request_path.read_bytes(),
        }
    ]


def test_factor_status_builds_only_from_exact_refs_and_preserves_active_pointer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_factor_store_boundary(monkeypatch)
    active_path = tmp_path / "results/system/_active.json"
    active_path.parent.mkdir(parents=True)
    active_bytes = b'{"exact":"active-pointer-sentinel"}'
    active_path.write_bytes(active_bytes)
    request = {
        "active_contextual_result_ref": _ref("factor.contextual_validation_result", "context-a"),
        "active_factor_set_ref": _ref("factor.bootstrap_set", "bootstrap-set-a"),
        "active_validation_attestation_ref": _ref("system.validation_attestation", "attestation-a"),
        "active_validation_receipt_ref": _ref("factor.validation_receipt", "receipt-a"),
        "observed_composite_state_ref": None,
    }
    digest = _canonical_request(tmp_path / "factor-status.json", request)

    main(
        [
            "factor",
            "status",
            "--workspace-root",
            str(tmp_path),
            "--request",
            "factor-status.json",
            "--expected-request-sha256",
            digest,
        ]
    )

    assert _line(capsys) == {
        "blockers": [],
        "readiness": "READY",
        "status_ref": _ref("factor.status", "status-a"),
    }
    assert calls[-1] == ("build_status", request)
    assert active_path.read_bytes() == active_bytes


def test_factor_status_has_no_legacy_active_read_route(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main(["factor", "status", "--workspace-root", str(tmp_path)])

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }


def test_factor_history_projects_authoritative_composite_cycle_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.system as system

    factor_set_ref = _ref("factor.bootstrap_set", "bootstrap-set-a")
    factor_status_ref = _ref("factor.status", "status-a")

    class SystemStoreBoundary:
        def __init__(self, workspace_root: object) -> None:
            assert workspace_root == str(tmp_path)

        def read_active(self) -> dict:
            return {
                "factor_active_set": {
                    "payload": {
                        "factor_rows": [
                            {"factor_id": "factor-b"},
                            {"factor_id": "factor-a"},
                        ]
                    }
                },
                "factor_status": {
                    "payload": {
                        "observed": {
                            "blockers": [],
                            "composite_state_ref": _ref("factor.composite_state", "composite-a"),
                            "cycle_state": "INTRINSIC_VALIDATED",
                            "terminal": True,
                        }
                    }
                },
                "factor_status_ref": factor_status_ref,
                "generation_id": "a" * 64,
                "manifest": {"payload": {"factor_active_set_ref": factor_set_ref}},
            }

    monkeypatch.setattr(system, "SystemStore", SystemStoreBoundary)

    main(["factor", "history", "--workspace-root", str(tmp_path)])

    assert _line(capsys) == {
        "active_generation_id": "a" * 64,
        "blockers": [],
        "entries": [
            {
                "factor_ids": ["factor-a", "factor-b"],
                "factor_set_ref": factor_set_ref,
                "factor_status_ref": factor_status_ref,
                "generation_id": "a" * 64,
                "observed_candidate_state": "INTRINSIC_VALIDATED",
            }
        ],
        "status": "OK",
    }


def test_factor_history_blocks_historical_unconfirmed_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_investor.system as system

    class SystemStoreBoundary:
        def __init__(self, workspace_root: object) -> None:
            assert workspace_root == str(tmp_path)

        def read_active(self) -> dict:
            return {
                "deployed_release_verified": False,
                "generation_id": "a" * 64,
                "generation_state": "OPERATIONAL",
                "historical_release_verified": True,
            }

    monkeypatch.setattr(system, "SystemStore", SystemStoreBoundary)

    main(["factor", "history", "--workspace-root", str(tmp_path)])

    assert _line(capsys) == {
        "active_generation_id": "a" * 64,
        "blockers": ["SYSTEM_DEPLOYED_RELEASE_UNCONFIRMED"],
        "entries": [],
        "status": "BLOCKED",
    }


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "blockers",
        "callback",
        "created_at",
        "readiness",
        "validation_namespace_id",
    ],
)
def test_factor_status_rejects_caller_authority_before_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    forbidden_field: str,
) -> None:
    import quant_investor.factors.governance as governance

    class StoreMustNotBeConstructed:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            raise AssertionError("invalid request reached Factor store")

    monkeypatch.setattr(
        governance,
        "FactorValidationStore",
        StoreMustNotBeConstructed,
        raising=False,
    )
    request = {
        "active_contextual_result_ref": _ref("factor.contextual_validation_result", "context-a"),
        "active_factor_set_ref": _ref("factor.bootstrap_set", "bootstrap-set-a"),
        "active_validation_attestation_ref": _ref("system.validation_attestation", "attestation-a"),
        "active_validation_receipt_ref": _ref("factor.validation_receipt", "receipt-a"),
        "observed_composite_state_ref": None,
        forbidden_field: {"caller": "authoritative"},
    }
    digest = _canonical_request(tmp_path / "invalid-factor-status.json", request)

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "factor",
                "status",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "invalid-factor-status.json",
                "--expected-request-sha256",
                digest,
            ]
        )

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "FACTOR_STATUS_REQUEST_INVALID",
        "status": "BLOCKED",
    }


def test_contextual_validation_has_no_second_system_cli_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main(["system", "validate"])

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }


def test_research_run_is_exact_blocked_uninitialized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as captured:
        main(
            [
                "research",
                "run",
                "--workspace-root",
                str(tmp_path),
                "--strategy-id",
                "paper-strategy",
            ]
        )

    assert captured.value.code == 2
    state = _line(capsys)
    assert state == {
        "active_generation_id": None,
        "blockers": ["ACTIVE_GENERATION_ABSENT"],
        "investment_state": "BLOCKED",
        "mainline_state": "UNINITIALIZED",
        "result": None,
        "status": "BLOCKED",
    }


def test_research_forward_is_inactive_and_cannot_create_pointer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    digest = _canonical_request(
        tmp_path / "forward.json",
        {
            "created_at": "2026-08-14T00:00:00Z",
            "request": {
                "as_of": "2026-08-14T00:00:00Z",
                "input_refs": [],
                "stages": ["decision_context"],
                "strategy_id": "paper-strategy",
            },
            "request_id": "request-owner-id",
        },
    )

    main(
        [
            "research",
            "forward",
            "--workspace-root",
            str(tmp_path),
            "--request",
            "forward.json",
            "--expected-request-sha256",
            digest,
        ]
    )
    artifact = _line(capsys)

    assert artifact["kind"] == "research_request"
    assert artifact["payload"]["run_state"] == "INACTIVE"
    assert artifact["payload"]["production"] is False
    assert not (tmp_path / "results" / "system" / "_active.json").exists()


def test_research_readiness_is_base_only_and_rejects_candidate_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = {
        "assessed_at": "2026-08-14T00:00:00Z",
        "factor_status": None,
        "producer_identity": "readiness-producer",
        "readiness_id": None,
        "source_blockers": ["FUNDAMENTAL_SOURCE_STALE"],
    }
    digest = _canonical_request(tmp_path / "readiness.json", request)

    main(
        [
            "research",
            "readiness",
            "--workspace-root",
            str(tmp_path),
            "--request",
            "readiness.json",
            "--expected-request-sha256",
            digest,
        ]
    )
    readiness = _line(capsys)

    assert readiness["kind"] == "intelligence_readiness"
    assert readiness["payload"]["mainline_candidate_ref"] is None
    assert readiness["payload"]["mainline_state"] == "UNINITIALIZED"
    assert readiness["payload"]["investment_state"] == "BLOCKED"
    assert "MAINLINE_CANDIDATE_ABSENT" in readiness["payload"]["blockers"]
    assert not (tmp_path / "results" / "system" / "_active.json").exists()

    request["mainline_candidate"] = None
    digest = _canonical_request(tmp_path / "candidate-readiness.json", request)
    with pytest.raises(SystemExit) as captured:
        main(
            [
                "research",
                "readiness",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "candidate-readiness.json",
                "--expected-request-sha256",
                digest,
            ]
        )
    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "RESEARCH_READINESS_REQUEST_INVALID",
        "status": "BLOCKED",
    }


@pytest.mark.parametrize(
    ("command", "method", "request_document"),
    [
        (
            "mine",
            "mine",
            {
                "exchange_calendar_ref": _ref("system.source_object", "calendar"),
                "expected_composite_state_ref": None,
                "factor_validator_manifest_ref": _ref("factor.validator_manifest", "validator"),
                "implementation_manifest_ref": _ref("system.source_object", "implementation"),
            },
        ),
        (
            "observe",
            "observe_signal",
            {
                "action": "SIGNAL",
                "expected_composite_state_ref": _ref(
                    "factor.composite_state", "composite-before-signal"
                ),
                "market_history_ref": _ref("system.source_object", "market"),
                "pit_universe_ref": _ref("system.source_object", "universe"),
                "preregistration_ref": _ref("factor.preregistration", "preregistration"),
                "selection_ref": None,
                "sparse_weights_ref": _ref("system.source_object", "weights"),
            },
        ),
        (
            "observe",
            "observe_label",
            {
                "action": "LABEL",
                "expected_composite_state_ref": _ref(
                    "factor.composite_state", "composite-before-label"
                ),
                "matured_label_prices_ref": _ref("system.source_object", "matured-label-prices"),
                "preregistration_ref": _ref("factor.preregistration", "preregistration"),
                "selection_ref": _ref("factor.configuration_selection", "selection"),
                "signal_capture_ref": _ref("factor.signal_capture", "capture"),
            },
        ),
        (
            "evaluate",
            "evaluate",
            {
                "action": "FINALIZE_EXECUTION",
                "expected_composite_state_ref": _ref(
                    "factor.composite_state", "composite-before-evaluation"
                ),
                "preregistration_ref": _ref("factor.preregistration", "preregistration"),
                "selection_ref": _ref("factor.configuration_selection", "selection"),
            },
        ),
    ],
)
def test_factor_candidates_dispatch_only_refs_and_preserve_active_pointer_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    method: str,
    request_document: dict,
) -> None:
    calls = _install_factor_store_boundary(monkeypatch)
    active_path = tmp_path / "results/system/_active.json"
    active_path.parent.mkdir(parents=True)
    active_bytes = b'{"exact":"active-pointer-sentinel"}'
    active_path.write_bytes(active_bytes)
    digest = _canonical_request(tmp_path / "factor-request.json", request_document)

    main(
        [
            "factor",
            command,
            "--workspace-root",
            str(tmp_path),
            "--request",
            "factor-request.json",
            "--expected-request-sha256",
            digest,
        ]
    )

    projected = _line(capsys)
    assert projected == {
        "blockers": [],
        "composite_state_ref": _ref("factor.composite_state", "composite-a"),
        "cycle_state": "PREREGISTERED",
        "terminal": False,
    }
    assert calls[-1][0] == method
    expected_call = dict(request_document)
    if method in {"observe_signal", "observe_label"}:
        expected_call.pop("action")
    assert calls[-1][1] == expected_call
    assert active_path.read_bytes() == active_bytes


def test_factor_contextual_validation_uses_only_stored_request_and_safe_projection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_state_ref = _ref("factor.composite_state", "composite-ready-for-contextual-validation")
    request_ref = _ref("system.validation_run_request", "validation-request-a")
    calls = _install_contextual_validation_boundary(
        monkeypatch,
        candidate_state_ref=expected_state_ref,
    )
    active_path = tmp_path / "results/system/_active.json"
    active_path.parent.mkdir(parents=True)
    active_bytes = b'{"exact":"active-pointer-sentinel"}'
    active_path.write_bytes(active_bytes)
    digest = _canonical_request(
        tmp_path / "contextual-validation.json",
        {
            "action": "REQUEST_CONTEXTUAL_VALIDATION",
            "expected_composite_state_ref": expected_state_ref,
            "validation_run_request_ref": request_ref,
        },
    )

    main(
        [
            "factor",
            "evaluate",
            "--workspace-root",
            str(tmp_path),
            "--request",
            "contextual-validation.json",
            "--expected-request-sha256",
            digest,
        ]
    )

    assert _line(capsys) == {
        "completion_sha256": "4" * 64,
        "contextual_result_ref": _ref("factor.contextual_validation_result", "context-a"),
        "outcome": "VALIDATED",
        "validation_attestation_ref": _ref("system.validation_attestation", "attestation-a"),
        "validation_request_ref": request_ref,
    }
    assert [call[0] for call in calls] == ["init", "get_object", "run_validation"]
    assert calls[-1] == ("run_validation", request_ref)
    assert active_path.read_bytes() == active_bytes


def test_factor_contextual_validation_fails_closed_on_expected_state_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ref = _ref("system.validation_run_request", "validation-request-a")
    calls = _install_contextual_validation_boundary(
        monkeypatch,
        candidate_state_ref=_ref("factor.composite_state", "stored-state"),
    )
    digest = _canonical_request(
        tmp_path / "contextual-validation-mismatch.json",
        {
            "action": "REQUEST_CONTEXTUAL_VALIDATION",
            "expected_composite_state_ref": _ref("factor.composite_state", "caller-state"),
            "validation_run_request_ref": request_ref,
        },
    )

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "factor",
                "evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "contextual-validation-mismatch.json",
                "--expected-request-sha256",
                digest,
            ]
        )

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "FACTOR_CONTEXTUAL_VALIDATION_STATE_MISMATCH",
        "status": "BLOCKED",
    }
    assert [call[0] for call in calls] == ["init", "get_object"]


def test_factor_contextual_validation_rejects_result_for_another_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ref = _ref("system.validation_run_request", "validation-request-a")
    calls = _install_contextual_validation_boundary(
        monkeypatch,
        candidate_state_ref=None,
        result_request_ref=_ref("system.validation_run_request", "different-validation-request"),
    )
    digest = _canonical_request(
        tmp_path / "contextual-result-mismatch.json",
        {
            "action": "REQUEST_CONTEXTUAL_VALIDATION",
            "expected_composite_state_ref": None,
            "validation_run_request_ref": request_ref,
        },
    )

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "factor",
                "evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "contextual-result-mismatch.json",
                "--expected-request-sha256",
                digest,
            ]
        )

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "FACTOR_CONTEXTUAL_VALIDATION_RESULT_INVALID",
        "status": "BLOCKED",
    }
    assert [call[0] for call in calls] == ["init", "get_object", "run_validation"]


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "callback",
        "created_at",
        "metrics",
        "result",
        "success",
        "validation_namespace_id",
    ],
)
def test_factor_contextual_validation_rejects_caller_authoritative_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    forbidden_field: str,
) -> None:
    import quant_investor.system as system

    class SystemStoreMustNotBeConstructed:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("invalid request reached System store")

    monkeypatch.setattr(system, "SystemStore", SystemStoreMustNotBeConstructed)
    request = {
        "action": "REQUEST_CONTEXTUAL_VALIDATION",
        "expected_composite_state_ref": None,
        "validation_run_request_ref": _ref(
            "system.validation_run_request", "bootstrap-validation-request"
        ),
        forbidden_field: {"caller": "authoritative"},
    }
    digest = _canonical_request(tmp_path / "invalid-contextual-request.json", request)

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "factor",
                "evaluate",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "invalid-contextual-request.json",
                "--expected-request-sha256",
                digest,
            ]
        )

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "FACTOR_EVALUATE_REQUEST_INVALID",
        "status": "BLOCKED",
    }


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "callback",
        "candidates",
        "coverage",
        "created_at",
        "gross",
        "metrics",
        "net",
        "nonce",
        "result",
        "signals_by_configuration",
        "statistics",
        "success",
        "turnover",
        "validation_namespace_id",
    ],
)
def test_factor_candidate_request_rejects_caller_authoritative_fields_before_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    forbidden_field: str,
) -> None:
    import quant_investor.factors.governance as governance

    class StoreMustNotBeConstructed:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            raise AssertionError("invalid request reached Factor store")

    monkeypatch.setattr(
        governance,
        "FactorValidationStore",
        StoreMustNotBeConstructed,
        raising=False,
    )
    request = {
        "exchange_calendar_ref": _ref("system.source_object", "calendar"),
        "expected_composite_state_ref": None,
        "factor_validator_manifest_ref": _ref("factor.validator_manifest", "validator"),
        "implementation_manifest_ref": _ref("system.source_object", "implementation"),
        forbidden_field: {"caller": "authoritative"},
    }
    active_path = tmp_path / "results/system/_active.json"
    active_path.parent.mkdir(parents=True)
    active_bytes = b'{"exact":"active-pointer-sentinel"}'
    active_path.write_bytes(active_bytes)
    digest = _canonical_request(tmp_path / "invalid-factor-request.json", request)

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "factor",
                "mine",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "invalid-factor-request.json",
                "--expected-request-sha256",
                digest,
            ]
        )

    assert captured.value.code == 2
    assert _line(capsys) == {
        "blocker_code": "FACTOR_MINE_REQUEST_INVALID",
        "status": "BLOCKED",
    }
    assert active_path.read_bytes() == active_bytes


def test_request_hash_failure_uses_expected_exit_two_without_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _canonical_request(tmp_path / "forward.json", {"value": 1})

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "research",
                "forward",
                "--workspace-root",
                str(tmp_path),
                "--request",
                "forward.json",
                "--expected-request-sha256",
                "0" * 64,
            ]
        )

    assert captured.value.code == 2
    error = _line(capsys)
    assert error == {
        "blocker_code": "REQUEST_SHA256_MISMATCH",
        "status": "BLOCKED",
    }
    assert str(tmp_path) not in json.dumps(error)
