from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pytest

import quant_investor.factors.production_control_v1 as production_control
from quant_investor.factors.production_control_v1 import (
    ProductionControlStore,
)
from quant_investor.v17_v4_contract import seal_semantic
from quant_investor.v17_v4_contract.schema_validation import (
    load_canonical_artifact,
)
from quant_investor.v17_v4_runtime.formal_activation import (
    FormalActivationCrash,
    FormalActivationError,
    FormalActivationService,
    factor_artifact_ref,
    manifest_refs,
)
from quant_investor.v17_v4_runtime.source_storage import (
    ExactReferenceReader,
    SourceCASMismatch,
    SourceStorageSecurityError,
)
from tests.unit.test_v17_v4_contract import _formal_intent
from tests.unit.test_factor_production_control_v1 import _artifacts


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FormalActivationService:
    service = FormalActivationService(tmp_path)
    monkeypatch.setattr(
        service,
        "_revalidate_closure",
        lambda intent: None,
    )
    return service


def _intent(intent_id: str = "formal-activation-1") -> dict[str, Any]:
    document = dict(_formal_intent())
    document["intent_id"] = intent_id
    document.pop("semantic_sha256")
    return seal_semantic(document)


def _replacement_intent(
    intent_id: str,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    document = dict(_intent(intent_id))
    document["expected_pointer_sha256"] = expected_pointer_sha256
    document["from_state"] = "FORMAL_ACTIVE"
    document.pop("semantic_sha256")
    return seal_semantic(document)


def _root(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "results/v17_v4_formal_research/strategies/quant-first"
    )


def test_formal_activation_is_intent_pointer_completion_and_not_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    result = service.activate(_intent())
    state = service.resolve("quant-first")

    assert result.status == state.status == "FORMAL_ACTIVE"
    assert result.recovered is False
    assert state.pointer is not None
    assert state.pointer["state"] == "PENDING_COMPLETION"
    assert state.pointer["authority"] == {
        "broker": False,
        "execution": False,
        "formal_research_publication": False,
        "order": False,
        "research_runtime_default": False,
        "trade": False,
    }
    assert state.completion is not None
    assert state.completion["authority"][
        "formal_research_publication"
    ] is True
    assert state.completion["authority"]["research_runtime_default"] is False
    assert not (tmp_path / "results/research_runtime_control").exists()

    root = _root(tmp_path)
    assert sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    ) == [
        ".active.lock",
        "_active.json",
        "completion_receipts/formal-activation-1.json",
        "intents/formal-activation-1.json",
    ]
    for path in root.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


def test_manifest_refs_bind_exact_verified_package_and_runtime() -> None:
    package_ref, runtime_ref = manifest_refs(
        strategy_id="quant-first",
        cutoff="2026-07-27T08:00:00Z",
    )
    assert package_ref["artifact_id"] == (
        "myquant.v17.v4.package-manifest.v1"
    )
    assert runtime_ref["artifact_id"] == (
        "myquant.v17.v4.runtime-build-manifest.v1"
    )
    assert package_ref["relative_path"].endswith(
        "resources/package_manifest.v1.json"
    )
    assert runtime_ref["relative_path"].endswith(
        "resources/runtime_build_manifest.v1.json"
    )


@pytest.mark.parametrize("boundary", ["intent", "cas", "readback", "completion"])
def test_formal_activation_recovers_every_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    service = _service(tmp_path, monkeypatch)
    with pytest.raises(FormalActivationCrash):
        service.activate(_intent(), crash_after=boundary)

    state = service.resolve("quant-first")
    if boundary == "intent":
        assert state.status == "V15_DEFAULT"
    elif boundary in {"cas", "readback"}:
        assert state.status == "PENDING_COMPLETION"
    else:
        assert state.status == "FORMAL_ACTIVE"

    recovered = service.activate(_intent())
    assert recovered.status == "FORMAL_ACTIVE"
    assert recovered.recovered is (boundary != "intent")
    assert service.resolve("quant-first").status == "FORMAL_ACTIVE"


def test_closure_rejection_writes_no_pointer_or_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FormalActivationService(tmp_path)

    def reject(intent: dict[str, Any]) -> None:
        raise FormalActivationError("invalid closure")

    monkeypatch.setattr(service, "_revalidate_closure", reject)
    with pytest.raises(FormalActivationError):
        service.activate(_intent())

    root = _root(tmp_path)
    assert not (root / "_active.json").exists()
    rejection = root / "rejection_receipts/formal-activation-1.json"
    artifact = load_canonical_artifact(rejection.read_bytes())
    assert artifact.version == (
        "myquant.v17.v4.formal-activation-rejection.v1"
    )
    for forbidden in (
        "results/v17_v4_canary",
        "results/research_runtime_control",
    ):
        assert not (tmp_path / forbidden).exists()

    monkeypatch.setattr(
        service,
        "_revalidate_closure",
        lambda intent: None,
    )
    with pytest.raises(
        FormalActivationError,
        match="INTENT_PREVIOUSLY_REJECTED",
    ):
        service.activate(_intent())
    assert not (root / "_active.json").exists()


def test_competing_intents_have_one_cas_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)

    def activate(intent_id: str) -> str:
        try:
            return service.activate(_intent(intent_id)).status
        except FormalActivationError:
            return "BLOCKED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                activate,
                ("formal-activation-a", "formal-activation-b"),
            )
        )
    assert sorted(outcomes) == ["BLOCKED", "FORMAL_ACTIVE"]
    pointer = (_root(tmp_path) / "_active.json").read_bytes()
    assert hashlib.sha256(pointer).hexdigest() in {
        service.resolve("quant-first").completion[
            "post_readback_sha256"
        ]
    }
    assert not (tmp_path / "results/research_runtime_control").exists()


def test_formal_active_can_be_replaced_only_from_exact_previous_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    first = service.activate(_intent("formal-activation-first"))
    second = service.activate(
        _replacement_intent(
            "formal-activation-second",
            first.pointer_ref["byte_sha256"],
        )
    )
    assert second.status == "FORMAL_ACTIVE"
    assert service.resolve("quant-first").intent["intent_id"] == (
        "formal-activation-second"
    )

    with pytest.raises(FormalActivationError):
        service.activate(
            _replacement_intent(
                "formal-activation-stale",
                first.pointer_ref["byte_sha256"],
            )
        )


def test_exact_reference_reader_cannot_open_lock_or_write_neutral_control(
    tmp_path: Path,
) -> None:
    neutral = "results/research_runtime_control/default_protocol.json"
    reader = ExactReferenceReader(tmp_path)
    with pytest.raises(SourceStorageSecurityError):
        reader.read(neutral)
    with pytest.raises(SourceStorageSecurityError):
        reader.write_exact_once(neutral, b"x")
    with pytest.raises(SourceStorageSecurityError):
        with reader.locked(neutral):
            pass
    assert not (tmp_path / "results/research_runtime_control").exists()


def test_formal_writer_rejects_other_v4_governed_paths(
    tmp_path: Path,
) -> None:
    service = FormalActivationService(tmp_path)
    for path in (
        "data/private/v17_v4_runs/run-1/output.json",
        "results/v17_v4_canary/strategies/quant-first/_current.json",
        (
            "results/v17_v4_formal_research/strategies/"
            "quant-first/runs/run-1/formal.json"
        ),
        "results/research_runtime_control/default_protocol.json",
    ):
        with pytest.raises(SourceStorageSecurityError):
            service._writer.write_exact_once(path, b"x")


def test_formal_factor_closure_replays_real_production_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def readback(evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "complete_chain_hash_binding_verified": True,
            "context_bindings_readback_verified": True,
            "evidence": deepcopy(evidence),
            "local_bytes_readback_verified": True,
            "quantitative_evidence_hash_binding_verified": True,
            "replay": {
                "replay_semantic_sha256": evidence[
                    "replay_semantic_sha256"
                ],
            },
            "replay_file_sha256": evidence["replay_file_sha256"],
        }

    monkeypatch.setattr(
        production_control,
        "readback_v4_evidence",
        readback,
    )
    (
        registry,
        eligibility,
        authorization,
        transaction,
        source_artifacts,
    ) = _artifacts()
    root = (
        tmp_path
        / "data/private/factor_governance_production_control_v1"
    )
    store = ProductionControlStore(root.resolve())
    receipt = store.apply(
        transaction,
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
        source_artifacts=source_artifacts,
    )
    active = json.loads(store.active_set_path.read_bytes())
    active_path = store.active_set_path.relative_to(tmp_path).as_posix()
    receipt_path = (
        root
        / "receipts/control_activations"
        / f"{receipt['receipt_id']}.json"
    ).relative_to(tmp_path).as_posix()
    refs = {
        "factor_control_active_set_ref": factor_artifact_ref(
            active,
            relative_path=active_path,
            strategy_id="quant-first",
            cutoff="2026-07-27T09:00:00Z",
        ),
        "factor_control_activation_receipt_ref": factor_artifact_ref(
            receipt,
            relative_path=receipt_path,
            strategy_id="quant-first",
            cutoff="2026-07-27T09:00:00Z",
        ),
    }

    service = FormalActivationService(tmp_path)
    service._revalidate_factor_closure(refs)

    drifted = dict(refs)
    drifted["factor_control_active_set_ref"] = {
        **refs["factor_control_active_set_ref"],
        "byte_sha256": "f" * 64,
    }
    with pytest.raises(SourceCASMismatch):
        service._revalidate_factor_closure(drifted)


def test_missing_completion_is_never_formal_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    with pytest.raises(FormalActivationCrash):
        service.activate(_intent(), crash_after="readback")
    assert service.resolve("quant-first").status == "PENDING_COMPLETION"
    assert not (
        _root(tmp_path)
        / "completion_receipts/formal-activation-1.json"
    ).exists()
