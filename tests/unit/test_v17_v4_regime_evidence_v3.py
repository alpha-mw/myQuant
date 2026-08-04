from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes, seal_semantic
from quant_investor.v17_v4_contract.resources import (
    load_packaged_json,
    read_packaged_asset,
    verify_package,
    verify_runtime_build,
)
from quant_investor.v17_v4_contract.schema_validation import validate_artifact
from quant_investor.v17_v4_contract.validators import (
    ArtifactContractError,
    regime_artifact_identity,
)
from quant_investor.v17_v4_runtime.source_storage import SourceStore
import quant_investor.v17_v4_runtime.regime_evidence_v3 as subject

_V2_TEST_PATH = Path(__file__).with_name("test_v17_v4_regime_evidence_v2_producer.py")
_V2_SPEC = importlib.util.spec_from_file_location("_v2_regime_test_support", _V2_TEST_PATH)
assert _V2_SPEC is not None and _V2_SPEC.loader is not None
_V2_SUPPORT = importlib.util.module_from_spec(_V2_SPEC)
sys.modules[_V2_SPEC.name] = _V2_SUPPORT
_V2_SPEC.loader.exec_module(_V2_SUPPORT)

_V1_SCHEMA_SHA = "49d006413465d7304c621f74f9732cc0d5636c400989d25b170ee4337ff229a3"
_V2_SCHEMA_SHA = "1d2b624d63808038240d29cf27a48b62d1f3d3da32b757be8bd196916f22de8c"
_POLICY_V1_SHA = "006773e24f47f0b7f28d6f7707ff6f570066cb212bd83ebd9566512fda7734ef"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _business_sessions(start: str, count: int) -> list[str]:
    cursor = date.fromisoformat(start)
    rows: list[str] = []
    while len(rows) < count:
        if cursor.weekday() < 5:
            rows.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return rows


class V3Factory:
    def __init__(self, workspace: Path, sessions: list[str]) -> None:
        self.workspace = workspace
        self.sessions = sessions
        self.policy_raw = read_packaged_asset(subject.INFERENCE_POLICY_V2_PATH)
        self.policy = load_packaged_json(subject.INFERENCE_POLICY_V2_PATH)
        self.policy_ref = {
            "byte_sha256": _sha(self.policy_raw),
            "relative_path": subject.INFERENCE_POLICY_V2_PATH,
            "semantic_sha256": self.policy["semantic_sha256"],
            "version": subject.INFERENCE_POLICY_V2_VERSION,
        }

    def build(
        self,
        *,
        observed: str,
        effective: str,
        created_at: str,
        prior: subject.RegimeEvidenceV3BuildResult | None = None,
        chain_anchor_ref: dict[str, str] | None = None,
    ) -> tuple[subject.RegimeEvidenceV3BuildResult, dict[str, Any]]:
        fixture = _V2_SUPPORT.make_regime_fixture(
            self.workspace,
            observed=observed,
            effective=effective,
            created_at=created_at,
            open_sessions=self.sessions,
        )
        store = SourceStore(self.workspace)
        transition = json.loads(
            store.read(
                fixture.kwargs["transition_matrix_path"],
                fixture.kwargs["transition_matrix_sha256"],
            )
        )
        transition.pop("semantic_sha256")
        transition["version"] = subject.TRANSITION_SNAPSHOT_V2_VERSION
        transition["inference_policy_ref"] = self.policy_ref
        transition["transition_snapshot_id"] = "0" * 64
        transition["transition_snapshot_id"] = regime_artifact_identity(
            transition,
            identity_field="transition_snapshot_id",
        )
        transition = seal_semantic(transition)
        transition_path = (
            "data/private/v17_v4_sources/regime_inputs/" f"{effective}-transition-v3.json"
        )
        transition_raw = canonical_resource_bytes(transition)
        store.write_exact_once(transition_path, transition_raw)
        transition_ref = subject._artifact_ref(
            transition,
            transition_raw,
            relative_path=transition_path,
        )

        model = json.loads(
            store.read(
                fixture.kwargs["model_snapshot_path"],
                fixture.kwargs["model_snapshot_sha256"],
            )
        )
        model.pop("semantic_sha256")
        model.pop("predecessor_evidence_ref")
        model.pop("model_implementation_sha256")
        model["version"] = subject.MODEL_SNAPSHOT_V2_VERSION
        model["inference_policy_ref"] = self.policy_ref
        model["transition_matrix_ref"] = transition_ref
        model["model_helper_sha256"] = self.policy["model_helper_sha256"]
        model["producer_sha256"] = self.policy["producer_sha256"]
        model["model_snapshot_id"] = "0" * 64
        model["model_snapshot_id"] = regime_artifact_identity(
            model,
            identity_field="model_snapshot_id",
        )
        model = seal_semantic(model)
        model_path = "data/private/v17_v4_sources/regime_inputs/" f"{effective}-model-v3.json"
        model_raw = canonical_resource_bytes(model)
        store.write_exact_once(model_path, model_raw)

        kwargs = dict(fixture.kwargs)
        kwargs.update(
            evidence_id="0" * 64,
            inference_policy_path=subject.INFERENCE_POLICY_V2_PATH,
            inference_policy_sha256=_sha(self.policy_raw),
            transition_matrix_path=transition_path,
            transition_matrix_sha256=_sha(transition_raw),
            model_snapshot_path=model_path,
            model_snapshot_sha256=_sha(model_raw),
        )
        kwargs.pop("prior_evidence_path")
        kwargs.pop("prior_evidence_sha256")
        if prior is not None:
            kwargs.update(
                prior_evidence_path=prior.evidence_path,
                prior_evidence_sha256=prior.evidence_sha256,
                prior_checkpoint_path=prior.chain_checkpoint_path,
                prior_checkpoint_sha256=prior.chain_checkpoint_sha256,
            )
        elif chain_anchor_ref is not None:
            kwargs.update(
                chain_anchor_path=chain_anchor_ref["relative_path"],
                chain_anchor_sha256=chain_anchor_ref["byte_sha256"],
            )
        for _ in range(2):
            try:
                return (
                    subject.build_regime_evidence_v3(
                        **kwargs,
                        _now_fn=_V2_SUPPORT._clock(created_at),
                    ),
                    kwargs,
                )
            except subject.RegimeEvidenceV3Error as exc:
                marker = "evidence_id mismatch; expected "
                if not exc.detail.startswith(marker):
                    raise
                kwargs["evidence_id"] = exc.detail.removeprefix(marker)
        raise AssertionError("identity probe did not converge")


@pytest.fixture
def sessions() -> list[str]:
    return _business_sessions("2026-07-29", 12)


def test_additive_contract_keeps_v1_v2_and_policy_v1_bytes() -> None:
    root = Path("quant_investor/v17_v4_contract")
    assert _sha((root / "schemas/regime_evidence.v1.schema.json").read_bytes()) == _V1_SCHEMA_SHA
    assert _sha((root / "schemas/regime_evidence.v2.schema.json").read_bytes()) == _V2_SCHEMA_SHA
    assert _sha((root / "resources/regime_inference_policy.v1.json").read_bytes()) == _POLICY_V1_SHA
    assert len(verify_package()) == 117
    assert len(verify_runtime_build()) == 34


def test_policy_v2_is_authority_false_and_bound_to_implementations() -> None:
    policy = load_packaged_json(subject.INFERENCE_POLICY_V2_PATH)
    assert policy["version"] == subject.INFERENCE_POLICY_V2_VERSION
    assert policy["producer_sha256"] == subject.implementation_sha256()
    assert policy["model_helper_sha256"] == _sha(Path(subject.v2.__file__).read_bytes())
    assert policy["authority"] == subject.regime_evidence_v3_authority_attestation()["authority"]
    assert policy["no_retroactive_causal_backfill"] is True
    assert policy["audit_limits"]["segment_length"] == 64
    assert policy["audit_limits"]["max_recovery_sessions"] == 260


def test_genesis_exact_once_replay_and_no_recursive_predecessor(
    tmp_path: Path,
    sessions: list[str],
) -> None:
    factory = V3Factory(tmp_path, sessions)
    result, kwargs = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    assert result.document["phase"] == "GENESIS"
    assert result.document["finalized_evidence_ordinal"] == 0
    assert result.document["segment_index"] == 0
    assert result.document["segment_position"] == 0
    assert "predecessor_evidence_ref" not in result.document
    replayed = subject.replay_regime_evidence_v3(
        workspace_root=tmp_path,
        evidence_path=result.evidence_path,
        evidence_sha256=result.evidence_sha256,
    )
    assert replayed == result.document
    reused = subject.build_regime_evidence_v3(
        **kwargs,
        _now_fn=_V2_SUPPORT._clock("2030-01-01T00:00:00Z"),
    )
    assert reused.reused is True
    assert reused.evidence_sha256 == result.evidence_sha256


def test_contiguous_and_missed_session_recovery(
    tmp_path: Path,
    sessions: list[str],
) -> None:
    factory = V3Factory(tmp_path, sessions)
    first, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    second, _ = factory.build(
        observed=sessions[1],
        effective=sessions[2],
        created_at="2026-07-30T15:21:00Z",
        prior=first,
    )
    recovered, _ = factory.build(
        observed=sessions[4],
        effective=sessions[5],
        created_at=f"{sessions[4]}T08:21:00Z",
        prior=second,
    )
    assert second.document["phase"] == "CONTIGUOUS"
    assert second.document["finalized_evidence_ordinal"] == 1
    assert recovered.document["phase"] == "RECOVERY"
    assert recovered.document["missing_sessions"] == sessions[3:5]
    assert recovered.document["finalized_evidence_ordinal"] == 2
    assert recovered.document["segment_index"] == 1
    assert recovered.document["segment_position"] == 0


def test_segment_rollover_is_fixed_at_64_finalized_evidence_records() -> None:
    prior = subject._Prior(
        document={"version": subject.REGIME_EVIDENCE_V3_VERSION},
        checkpoint={"segment_index": 0, "segment_position": 63},
        evidence_ref=None,
        checkpoint_ref={},
        chain_anchor_ref=None,
        finalized_ordinal=63,
        effective_session="2026-10-28",
        state_probabilities={state: "0.200000000000" for state in subject.v2.STATE_ORDER},
        chain_id="a" * 64,
        global_accumulator="b" * 64,
        segment_id="c" * 64,
        segment_index=0,
        segment_position=63,
        segment_accumulator="d" * 64,
    )
    phase = subject._phase(
        prior_document=prior.document,
        prior_ordinal=prior.finalized_ordinal,
        missing_sessions=[],
    )
    assert phase == "ROLLOVER"
    assert subject._segment_location(phase=phase, prior=prior) == (1, 0)


def test_anchor_only_crash_recovery_starts_new_chain_segment_without_backfill(
    tmp_path: Path,
    sessions: list[str],
) -> None:
    factory = V3Factory(tmp_path, sessions)
    first, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    anchor_ref = first.document["chain_anchor_ref"]
    for relative_path in (
        first.evidence_path,
        first.chain_checkpoint_path,
        first.document["segment_anchor_ref"]["relative_path"],
    ):
        (tmp_path / relative_path).unlink()

    recovered, _ = factory.build(
        observed=sessions[3],
        effective=sessions[4],
        created_at=f"{sessions[3]}T08:21:00Z",
        chain_anchor_ref=anchor_ref,
    )
    assert recovered.document["phase"] == "RECOVERY"
    assert recovered.document["missing_sessions"] == sessions[1:4]
    assert recovered.document["finalized_evidence_ordinal"] == 0
    assert recovered.document["prior_finality"] == {
        "prior_checkpoint_byte_sha256": None,
        "prior_checkpoint_id": None,
        "prior_checkpoint_semantic_sha256": None,
        "prior_effective_session": None,
        "prior_evidence_byte_sha256": None,
        "prior_evidence_id": None,
        "prior_evidence_semantic_sha256": None,
        "prior_finalized_evidence_ordinal": None,
        "prior_global_accumulator": None,
        "prior_segment_id": None,
        "prior_segment_index": None,
        "prior_segment_position": None,
    }


def test_gap_over_260_sessions_fails_closed(tmp_path: Path) -> None:
    sessions = _business_sessions("2026-07-29", 264)
    factory = V3Factory(tmp_path, sessions)
    first, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    with pytest.raises(subject.RegimeEvidenceV3Error, match="max260"):
        factory.build(
            observed=sessions[-2],
            effective=sessions[-1],
            created_at=f"{sessions[-2]}T08:21:00Z",
            prior=first,
        )


def test_replay_rejects_resealed_accumulator_forgery(
    tmp_path: Path,
    sessions: list[str],
) -> None:
    result, _ = V3Factory(tmp_path, sessions).build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    path = tmp_path / result.evidence_path
    forged = deepcopy(result.document)
    forged.pop("semantic_sha256")
    forged["global_accumulator"] = "f" * 64
    forged["evidence_id"] = "0" * 64
    forged["evidence_id"] = regime_artifact_identity(
        forged,
        identity_field="evidence_id",
    )
    forged = seal_semantic(forged)
    raw = canonical_resource_bytes(forged)
    path.write_bytes(raw)
    with pytest.raises(subject.RegimeEvidenceV3Error):
        subject.replay_regime_evidence_v3(
            workspace_root=tmp_path,
            evidence_path=result.evidence_path,
            evidence_sha256=_sha(raw),
        )


def test_contract_validator_recursively_reads_v3_direct_source_closure(
    tmp_path: Path,
    sessions: list[str],
) -> None:
    result, _ = V3Factory(tmp_path, sessions).build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    store = SourceStore(tmp_path)
    calls: list[str] = []

    def loader(reference: dict[str, str]) -> bytes:
        calls.append(reference["artifact_version"])
        return store.read(reference["relative_path"], reference["byte_sha256"])

    validate_artifact(result.document, artifact_loader=loader)
    assert "myquant.v17.v4.regime-feature-snapshot.v1" in calls
    assert "myquant.v17.v4.regime-model-snapshot.v2" in calls
    assert "myquant.v17.v4.regime-transition-matrix-snapshot.v2" in calls

    def missing_feature(reference: dict[str, str]) -> bytes:
        if reference["artifact_version"] == "myquant.v17.v4.regime-feature-snapshot.v1":
            raise FileNotFoundError("feature deliberately absent")
        return store.read(reference["relative_path"], reference["byte_sha256"])

    with pytest.raises(ArtifactContractError, match="feature_snapshot_ref readback failed"):
        validate_artifact(result.document, artifact_loader=missing_feature)


def test_daily_replay_uses_a_constant_bounded_direct_read_set(
    tmp_path: Path,
    sessions: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = V3Factory(tmp_path, sessions)
    first, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    second, _ = factory.build(
        observed=sessions[1],
        effective=sessions[2],
        created_at="2026-07-30T15:21:00Z",
        prior=first,
    )
    original_read = SourceStore.read
    reads: list[str] = []

    def tracked_read(
        store: SourceStore,
        relative_path: str | PurePosixPath,
        expected_sha256: str | None = None,
    ) -> bytes:
        reads.append(str(relative_path))
        return original_read(store, relative_path, expected_sha256)

    monkeypatch.setattr(SourceStore, "read", tracked_read)
    subject.replay_regime_evidence_v3(
        workspace_root=tmp_path,
        evidence_path=second.evidence_path,
        evidence_sha256=second.evidence_sha256,
    )
    assert len(set(reads)) <= 13
    assert first.document["chain_anchor_ref"]["relative_path"] in reads
    assert second.document["current_checkpoint_ref"]["relative_path"] in reads
    assert first.document["current_checkpoint_ref"]["relative_path"] in reads
    assert first.evidence_path in reads


def test_explicit_audit_rejects_omitted_tail(
    tmp_path: Path,
    sessions: list[str],
) -> None:
    factory = V3Factory(tmp_path, sessions)
    first, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    second, _ = factory.build(
        observed=sessions[1],
        effective=sessions[2],
        created_at="2026-07-30T15:21:00Z",
        prior=first,
    )
    refs = [
        subject._artifact_ref(
            result.document,
            SourceStore(tmp_path).read(result.evidence_path, result.evidence_sha256),
            relative_path=result.evidence_path,
        )
        for result in (first, second)
    ]
    audit = subject.audit_regime_chain_v3(
        workspace_root=tmp_path,
        evidence_refs=refs,
        expected_head_path=second.evidence_path,
        expected_head_sha256=second.evidence_sha256,
        audit_as_of_session=sessions[2],
    )
    assert audit["record_count"] == 2
    with pytest.raises(subject.RegimeEvidenceV3Error, match="head or as-of"):
        subject.audit_regime_chain_v3(
            workspace_root=tmp_path,
            evidence_refs=refs[:1],
            expected_head_path=second.evidence_path,
            expected_head_sha256=second.evidence_sha256,
            audit_as_of_session=sessions[2],
        )


@pytest.mark.parametrize("count", [1, 20, 60, 63, 64, 65, 128, 129, 260, 1000])
def test_capacity_probe_is_constant_bounded_through_1000_sessions(count: int) -> None:
    probe = subject.regime_chain_capacity_probe_v3(session_count=count)
    assert subject.regime_chain_capacity_probe_v3(session_count=count) == probe
    assert probe["session_count"] == count
    assert probe["hash_records_processed"] == count
    assert probe["bounded"] is True
    assert probe["daily_replay_unique_node_upper_bound"] == 13
    assert probe["daily_replay_unique_node_upper_bound"] < 16
    assert probe["daily_replay_depth_upper_bound"] <= 5
    assert probe["segment_count"] == (count + 63) // 64
    assert probe["finalized_evidence_ordinal"] == count - 1
    assert len(probe["final_global_accumulator"]) == 64
    assert len(probe["final_segment_accumulator"]) == 64


def test_no_authority_or_weight_surface() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "factor_weight",
        "portfolio_weight",
        "target_weight",
        "factor_tier",
        "lifecycle_action",
    ):
        assert forbidden not in source
    attestation = subject.regime_evidence_v3_authority_attestation()
    assert set(attestation["authority"].values()) == {False}
    assert attestation["formal_activation_eligible"] is False
    assert attestation["promotion_eligible"] is False
