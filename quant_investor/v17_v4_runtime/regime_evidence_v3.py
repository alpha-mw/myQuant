"""V17 v4 regime-evidence v3 producer, replay, and explicit-chain audit.

The frozen v2 module remains unchanged.  V3 reuses only v2's pure Decimal
normalization, score, posterior, path-safe strategy/session helpers, and fixed
constants; it owns its own direct closure, chain, segment, and checkpoint
envelopes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Callable, Final, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    artifact_identity_field,
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
)
from quant_investor.v17_v4_contract.canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
)
from quant_investor.v17_v4_contract.schema_validation import (
    SchemaValidationError,
    schema_path_for_version,
)
from quant_investor.v17_v4_contract.validators import (
    ArtifactContractError,
    regime_artifact_identity,
)
from quant_investor.v17_v4_contract.resources import (
    PackageResourceError,
    load_packaged_json,
    read_packaged_asset,
)
from quant_investor.v17_v4_runtime import regime_evidence_v2 as v2
from quant_investor.v17_v4_runtime.source_storage import (
    SourceCASMismatch,
    SourceExactOnceConflict,
    SourceNotFoundError,
    SourceStorageError,
    SourceStorageSecurityError,
    SourceStore,
)

REGIME_EVIDENCE_V3_VERSION: Final = "myquant.v17.v4.regime-evidence.v3"
CHAIN_ANCHOR_VERSION: Final = "myquant.v17.v4.regime-chain-anchor.v1"
SEGMENT_ANCHOR_VERSION: Final = "myquant.v17.v4.regime-segment-anchor.v1"
STATE_CHECKPOINT_VERSION: Final = "myquant.v17.v4.regime-state-checkpoint.v1"
MODEL_SNAPSHOT_V2_VERSION: Final = "myquant.v17.v4.regime-model-snapshot.v2"
TRANSITION_SNAPSHOT_V2_VERSION: Final = "myquant.v17.v4.regime-transition-matrix-snapshot.v2"
INFERENCE_POLICY_V2_VERSION: Final = "myquant.v17.v4.regime-inference-policy.v2"
INFERENCE_POLICY_V2_PATH: Final = "resources/regime_inference_policy.v2.json"

REGIME_CHAIN_V3_ROOT: Final = PurePosixPath("data/private/v17_v4_sources/regime_v3")
REGIME_EVIDENCE_V3_ROOT: Final = PurePosixPath("data/private/v17_v4_sources/regime_evidence")
_EVIDENCE_NAME: Final = "regime_evidence.v3.json"
_ANCHOR_NAME: Final = "chain_anchor.v1.json"
_LOCK_NAME: Final = ".producer.lock"

SEGMENT_LENGTH: Final = 64
MAX_GAP_SESSIONS: Final = 260
MAX_AUDIT_RECORDS: Final = 1000
MAX_AUDIT_EVIDENCE_BYTES: Final = 128 * 1024 * 1024
MAX_AUDIT_SECONDS: Final = 120
NEW_PUBLICATION_TOLERANCE_SECONDS: Final = v2.NEW_PUBLICATION_TOLERANCE_SECONDS

AUTHORITY_STATUS: Final = "SHADOW_ONLY_NO_AUTHORITY"
TRUE_CURRENT_CANONICAL_INPUT_GAP: Final = "TRUE_CURRENT_CANONICAL_INPUT_GAP"
INPUT_TAMPER_BLOCKER: Final = "REGIME_EVIDENCE_V3_INPUT_TAMPER"
SEMANTIC_BLOCKER: Final = "REGIME_EVIDENCE_V3_SEMANTIC_MISMATCH"
IDENTITY_BLOCKER: Final = "REGIME_EVIDENCE_V3_IDENTITY_MISMATCH"
TEMPORAL_BLOCKER: Final = "REGIME_EVIDENCE_V3_TEMPORAL_VIOLATION"
CONFLICT_BLOCKER: Final = "REGIME_EVIDENCE_V3_EXACT_ONCE_CONFLICT"
STALE_PRIOR_BLOCKER: Final = "REGIME_EVIDENCE_V3_STALE_PRIOR"
CHECKPOINT_BLOCKER: Final = "REGIME_EVIDENCE_V3_CHECKPOINT_MISMATCH"

_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
_REF_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
)
_SELF_FIELDS: Final = frozenset(
    {
        "anchor_commitment",
        "calendar_prefix_commitment",
        "checkpoint_id",
        "global_accumulator",
        "record_commitment",
        "segment_anchor_id",
        "semantic_sha256",
    }
)
_EVIDENCE_ID_EXCLUDED: Final = frozenset(
    {
        "available_at",
        "blocker_codes",
        "computed_at",
        "created_at",
        "evidence_id",
        "published_at",
        "record_commitment",
        "semantic_sha256",
    }
)
_UTC: Final = timezone.utc
_SHANGHAI: Final = ZoneInfo("Asia/Shanghai")


class RegimeEvidenceV3Error(RuntimeError):
    """Typed fail-closed error for V3 callers."""

    exit_code = 2

    def __init__(self, blocker_code: str, detail: str, *, status: str = "BLOCKED") -> None:
        self.blocker_code = blocker_code
        self.blocker_codes = (blocker_code,)
        self.detail = detail
        self.status = status
        super().__init__(f"{blocker_code}: {detail}")

    def to_status(self) -> dict[str, Any]:
        return {
            "authority": dict(_NO_AUTHORITY),
            "authority_status": AUTHORITY_STATUS,
            "blocker_codes": list(self.blocker_codes),
            "detail": self.detail,
            "status": self.status,
        }


class RegimeEvidenceV3InputGap(RegimeEvidenceV3Error):
    """Missing current canonical input, not a synthetic fallback."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            TRUE_CURRENT_CANONICAL_INPUT_GAP,
            detail,
            status=TRUE_CURRENT_CANONICAL_INPUT_GAP,
        )


class RegimeEvidenceV3Conflict(RegimeEvidenceV3Error):
    def __init__(self, detail: str) -> None:
        super().__init__(CONFLICT_BLOCKER, detail)


@dataclass(frozen=True)
class RegimeEvidenceV3BuildResult:
    status: str
    authority: dict[str, bool]
    authority_status: str
    evidence_id: str
    evidence_path: str
    evidence_sha256: str
    chain_checkpoint_path: str
    chain_checkpoint_sha256: str
    record_commitment: str
    checkpoint_commitment: str
    created: bool
    reused: bool
    document: dict[str, Any]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Loaded:
    document: dict[str, Any]
    raw: bytes
    ref: dict[str, str]


@dataclass(frozen=True)
class _Prior:
    document: dict[str, Any] | None
    checkpoint: dict[str, Any]
    evidence_ref: dict[str, str] | None
    checkpoint_ref: dict[str, str]
    chain_anchor_ref: dict[str, str] | None
    finalized_ordinal: int
    effective_session: str
    state_probabilities: dict[str, str]
    chain_id: str
    global_accumulator: str
    segment_id: str | None
    segment_index: int
    segment_position: int
    segment_accumulator: str | None


def regime_evidence_v3_authority_attestation() -> dict[str, Any]:
    return {
        "authority": dict(_NO_AUTHORITY),
        "authority_status": AUTHORITY_STATUS,
        "formal_activation_eligible": False,
        "no_retroactive_causal_backfill": True,
        "performance_evidence_eligible": False,
        "promotion_eligible": False,
        "same_session_execution_eligible": False,
        "shadow_only": True,
    }


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def regime_evidence_v3_path(*, strategy_id: str, effective_session: str) -> str:
    return str(
        REGIME_EVIDENCE_V3_ROOT
        / _strategy_id(strategy_id)
        / _session(effective_session, label="effective_session")
        / _EVIDENCE_NAME
    )


def chain_anchor_v1_path(*, strategy_id: str) -> str:
    return str(REGIME_CHAIN_V3_ROOT / _strategy_id(strategy_id) / _ANCHOR_NAME)


def segment_anchor_v1_path(*, strategy_id: str, segment_commitment: str) -> str:
    return str(
        REGIME_CHAIN_V3_ROOT
        / _strategy_id(strategy_id)
        / "segments"
        / f"{_sha(segment_commitment, label='segment_commitment')}.v1.json"
    )


def checkpoint_v1_path(*, strategy_id: str, checkpoint_commitment: str) -> str:
    """Compatibility helper when only the content id is known."""

    return str(
        REGIME_CHAIN_V3_ROOT
        / _strategy_id(strategy_id)
        / "checkpoints"
        / "unknown"
        / f"{_sha(checkpoint_commitment, label='checkpoint_commitment')}.v1.json"
    )


def state_checkpoint_v1_path(
    *,
    strategy_id: str,
    effective_session: str,
    checkpoint_id: str,
) -> str:
    return str(
        REGIME_CHAIN_V3_ROOT
        / _strategy_id(strategy_id)
        / "checkpoints"
        / _session(effective_session, label="effective_session")
        / f"{_sha(checkpoint_id, label='checkpoint_id')}.v1.json"
    )


def regime_evidence_v3_id(document: Mapping[str, Any]) -> str:
    return regime_artifact_identity(document, identity_field="evidence_id")


def record_commitment_v1(record: Mapping[str, Any]) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-record-commitment.v1",
        {"record_core": _strip_self(record)},
    )


def segment_anchor_commitment_v1(segment: Mapping[str, Any]) -> str:
    return regime_artifact_identity(
        segment,
        identity_field="segment_anchor_id",
    )


def calendar_prefix_commitment_v1(
    *,
    strategy_id: str,
    through_session: str,
    open_sessions: Sequence[str],
    policy_byte_sha256: str,
) -> str:
    through = _session(through_session, label="through_session")
    prefix = [_session(item, label="open_session") for item in open_sessions if item <= through]
    if prefix != sorted(set(prefix)) or not prefix or prefix[-1] != through:
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER,
            "calendar prefix is not exact historic open-session coverage through D",
        )
    return _domain_hash(
        "myquant.v17.v4.regime-calendar-prefix.v1",
        {
            "open_sessions": prefix,
            "policy_byte_sha256": _sha(policy_byte_sha256, label="policy_byte_sha256"),
            "prefix_end_session": through,
            "prefix_length": len(prefix),
            "strategy_id": _strategy_id(strategy_id),
        },
    )


def calendar_prefix_v1(
    *,
    through_session: str,
    open_sessions: Sequence[str],
    policy_byte_sha256: str,
    strategy_id: str,
) -> dict[str, Any]:
    through = _session(through_session, label="through_session")
    prefix = [_session(item, label="open_session") for item in open_sessions if item <= through]
    if prefix != sorted(set(prefix)) or not prefix or prefix[-1] != through:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "calendar prefix is not exact through D")
    return {
        "prefix_end_session": through,
        "prefix_length": len(prefix),
        "prefix_sha256": calendar_prefix_commitment_v1(
            strategy_id=strategy_id,
            through_session=through,
            open_sessions=open_sessions,
            policy_byte_sha256=policy_byte_sha256,
        ),
    }


def chain_checkpoint_commitment_v1(checkpoint: Mapping[str, Any]) -> str:
    return regime_artifact_identity(
        checkpoint,
        identity_field="checkpoint_id",
    )


def global_accumulator_v1(
    *,
    prior_accumulator: str,
    record_commitment: str,
    finalized_ordinal: int,
    chain_id: str,
) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-global-accumulator.v1",
        {
            "chain_id": _sha(chain_id, label="chain_id"),
            "evidence_ordinal": _ordinal(finalized_ordinal),
            "previous_accumulator": _sha(prior_accumulator, label="prior_accumulator"),
            "record_commitment": _sha(record_commitment, label="record_commitment"),
        },
    )


def chain_id_v1(*, policy_ref: Mapping[str, str], strategy_id: str) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-chain-id.v1",
        {
            "policy_byte_sha256": _sha(policy_ref["byte_sha256"], label="policy_byte_sha256"),
            "policy_semantic_sha256": _sha(
                policy_ref["semantic_sha256"],
                label="policy_semantic_sha256",
            ),
            "strategy_id": _strategy_id(strategy_id),
        },
    )


def segment_id_v1(
    *,
    chain_id: str,
    segment_index: int,
    segment_start_session: str,
    start_phase: str,
) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-segment-id.v1",
        {
            "chain_id": _sha(chain_id, label="chain_id"),
            "segment_index": segment_index,
            "segment_start_session": _session(
                segment_start_session,
                label="segment_start_session",
            ),
            "start_phase": start_phase,
        },
    )


def global_seed_v1(*, chain_id: str, policy_ref: Mapping[str, str], strategy_id: str) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-global-seed.v1",
        {
            "chain_id": _sha(chain_id, label="chain_id"),
            "policy_byte_sha256": _sha(policy_ref["byte_sha256"], label="policy_byte_sha256"),
            "strategy_id": _strategy_id(strategy_id),
        },
    )


def segment_seed_v1(
    *,
    chain_id: str,
    policy_ref: Mapping[str, str],
    previous_global_accumulator: str | None,
    segment_id: str,
    segment_index: int,
    segment_start_session: str,
    start_phase: str,
    strategy_id: str,
) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-segment-seed.v1",
        {
            "chain_id": _sha(chain_id, label="chain_id"),
            "policy_byte_sha256": _sha(policy_ref["byte_sha256"], label="policy_byte_sha256"),
            "previous_global_accumulator": (
                None
                if previous_global_accumulator is None
                else _sha(previous_global_accumulator, label="previous_global_accumulator")
            ),
            "segment_id": _sha(segment_id, label="segment_id"),
            "segment_index": segment_index,
            "segment_start_session": _session(segment_start_session, label="segment_start_session"),
            "start_phase": start_phase,
            "strategy_id": _strategy_id(strategy_id),
        },
    )


def segment_accumulator_v1(
    *,
    previous_accumulator: str,
    record_commitment: str,
    segment_id: str,
    segment_position: int,
) -> str:
    return _domain_hash(
        "myquant.v17.v4.regime-segment-accumulator.v1",
        {
            "previous_accumulator": _sha(previous_accumulator, label="previous_accumulator"),
            "record_commitment": _sha(record_commitment, label="record_commitment"),
            "segment_id": _sha(segment_id, label="segment_id"),
            "segment_position": segment_position,
        },
    )


class _V3Store(SourceStore):
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        strategy_id: str,
        allowed_paths: Sequence[str],
    ) -> None:
        super().__init__(workspace_root)
        self._strategy_id = _strategy_id(strategy_id)
        self._allowed = {PurePosixPath(path) for path in allowed_paths} | {
            REGIME_CHAIN_V3_ROOT / self._strategy_id / _ANCHOR_NAME,
            REGIME_CHAIN_V3_ROOT / self._strategy_id / _LOCK_NAME,
        }

    def _canonical_path(self, value: str | PurePosixPath) -> PurePosixPath:
        path = PurePosixPath(value)
        if path not in self._allowed:
            raise SourceStorageSecurityError(
                "regime evidence v3 writer path is not an exact permitted slot"
            )
        return super()._canonical_path(path)

    @contextmanager
    def producer_locked(self) -> Iterator[None]:
        with super().locked(REGIME_CHAIN_V3_ROOT / self._strategy_id / _LOCK_NAME):
            yield


def build_regime_evidence_v3(
    *,
    workspace_root: str | Path,
    evidence_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    inference_policy_path: str,
    inference_policy_sha256: str,
    model_snapshot_path: str,
    model_snapshot_sha256: str,
    transition_matrix_path: str,
    transition_matrix_sha256: str,
    feature_snapshot_path: str,
    feature_snapshot_sha256: str,
    prior_evidence_path: str | None = None,
    prior_evidence_sha256: str | None = None,
    prior_checkpoint_path: str | None = None,
    prior_checkpoint_sha256: str | None = None,
    chain_anchor_path: str | None = None,
    chain_anchor_sha256: str | None = None,
    _now_fn: Callable[[], datetime] | None = None,
) -> RegimeEvidenceV3BuildResult:
    """Build or exactly reuse one fixed strategy/effective v3 evidence slot."""

    strategy = _strategy_id(strategy_id)
    decision = _session(decision_session, label="decision_session")
    _opaque(evidence_id, label="evidence_id")
    _timestamp(cutoff, label="cutoff")
    _timestamp(created_at, label="created_at")
    evidence_path = regime_evidence_v3_path(strategy_id=strategy, effective_session=decision)
    build_args = {
        "chain_anchor_path": chain_anchor_path,
        "chain_anchor_sha256": chain_anchor_sha256,
        "created_at": created_at,
        "cutoff": cutoff,
        "decision_session": decision,
        "evidence_id": evidence_id,
        "feature_snapshot_path": feature_snapshot_path,
        "feature_snapshot_sha256": feature_snapshot_sha256,
        "inference_policy_path": inference_policy_path,
        "inference_policy_sha256": inference_policy_sha256,
        "model_snapshot_path": model_snapshot_path,
        "model_snapshot_sha256": model_snapshot_sha256,
        "prior_checkpoint_path": prior_checkpoint_path,
        "prior_checkpoint_sha256": prior_checkpoint_sha256,
        "prior_evidence_path": prior_evidence_path,
        "prior_evidence_sha256": prior_evidence_sha256,
        "strategy_id": strategy,
        "transition_matrix_path": transition_matrix_path,
        "transition_matrix_sha256": transition_matrix_sha256,
    }
    reader = SourceStore(workspace_root)
    writer = _V3Store(workspace_root, strategy_id=strategy, allowed_paths=(evidence_path,))
    with writer.producer_locked():
        existing = writer.read_optional(evidence_path)
        if existing is not None:
            replayed = _validate_evidence(
                store=reader,
                evidence_path=evidence_path,
                evidence_sha256=existing.byte_sha256,
                raw=existing.data,
                enforce_build_arguments=build_args,
            )
            return _result(
                document=replayed,
                evidence_path=evidence_path,
                evidence_sha256=existing.byte_sha256,
                created=False,
                reused=True,
            )

        policy = _load_policy(
            inference_policy_path=inference_policy_path,
            inference_policy_sha256=inference_policy_sha256,
        )
        _, publication_time = _timestamp(created_at, label="created_at")
        _, policy_not_before = _timestamp(
            policy.document["not_before"],
            label="policy.not_before",
        )
        if publication_time < policy_not_before:
            raise RegimeEvidenceV3Error(
                TEMPORAL_BLOCKER,
                "publication predates additive v3 policy deployment",
            )
        feature = _load_snapshot(
            store=reader,
            path=feature_snapshot_path,
            sha256=feature_snapshot_sha256,
            expected_version=v2.FEATURE_SNAPSHOT_VERSION,
            strategy_id=strategy,
            label="feature_snapshot",
        )
        transition = _load_snapshot(
            store=reader,
            path=transition_matrix_path,
            sha256=transition_matrix_sha256,
            expected_version=TRANSITION_SNAPSHOT_V2_VERSION,
            strategy_id=strategy,
            label="transition_matrix",
        )
        model = _load_snapshot(
            store=reader,
            path=model_snapshot_path,
            sha256=model_snapshot_sha256,
            expected_version=MODEL_SNAPSHOT_V2_VERSION,
            strategy_id=strategy,
            label="model_snapshot",
        )
        closure_loader = v2._ClosureLoader(reader)
        try:
            closure_loader(feature.ref)
            closure_loader(model.ref)
            closure_loader(transition.ref)
        except v2.RegimeEvidenceV2Error as exc:
            raise _translate_v2(exc) from exc
        observed, open_sessions = _validate_current_closure(
            strategy_id=strategy,
            decision_session=decision,
            cutoff=cutoff,
            policy=policy.document,
            policy_ref=policy.ref,
            feature=feature.document,
            transition=transition.document,
            model=model.document,
            transition_ref=transition.ref,
        )
        _publication_clock(
            created_at=created_at,
            cutoff=cutoff,
            observed_session=observed,
            decision_session=decision,
            now=(_now_fn or (lambda: datetime.now(_UTC)))(),
        )
        anchor_to_publish: tuple[str, bytes] | None = None
        if (
            prior_evidence_path is None
            and prior_checkpoint_path is None
            and chain_anchor_path is None
        ):
            anchor_path = chain_anchor_v1_path(strategy_id=strategy)
            anchor_existing = writer.read_optional(anchor_path)
            if anchor_existing is None:
                anchor = _make_chain_anchor(
                    strategy_id=strategy,
                    created_at=created_at,
                    cutoff=cutoff,
                    policy=policy.document,
                    policy_ref=policy.ref,
                    model_ref=model.ref,
                    transition_ref=transition.ref,
                    open_sessions=open_sessions,
                )
                anchor_raw = canonical_resource_bytes(anchor)
                chain_anchor_path = anchor_path
                chain_anchor_sha256 = hashlib.sha256(anchor_raw).hexdigest()
                anchor_to_publish = (anchor_path, anchor_raw)
            else:
                chain_anchor_path = anchor_path
                chain_anchor_sha256 = anchor_existing.byte_sha256
            build_args["chain_anchor_path"] = chain_anchor_path
            build_args["chain_anchor_sha256"] = chain_anchor_sha256
        if anchor_to_publish is not None:
            prior = _prior_from_anchor(
                anchor=anchor,
                anchor_path=anchor_path,
                anchor_raw=anchor_raw,
                observed_session=observed,
            )
        else:
            prior = _load_prior(
                store=reader,
                strategy_id=strategy,
                observed_session=observed,
                prior_evidence_path=prior_evidence_path,
                prior_evidence_sha256=prior_evidence_sha256,
                prior_checkpoint_path=prior_checkpoint_path,
                prior_checkpoint_sha256=prior_checkpoint_sha256,
                chain_anchor_path=chain_anchor_path,
                chain_anchor_sha256=chain_anchor_sha256,
            )
        missing = _missing_sessions(
            open_sessions=open_sessions,
            after_session=prior.effective_session,
            before_session=decision,
        )
        if len(missing) > MAX_GAP_SESSIONS:
            raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "missing-open propagation exceeds max260")
        _reject_stale_prior(
            store=reader,
            strategy_id=strategy,
            missing_sessions=missing,
        )
        phase = _phase(
            prior_document=prior.document,
            prior_ordinal=prior.finalized_ordinal,
            missing_sessions=missing,
        )
        propagated = _propagate_missing(
            start_probabilities=prior.state_probabilities,
            missing_sessions=missing,
            transition_matrix=transition.document["transition_matrix"],
        )
        probabilities = v2._posterior(
            previous=propagated,
            transition_matrix=transition.document["transition_matrix"],
            likelihoods=feature.document["state_likelihoods"],
        )
        ordinal = 0 if prior.document is None else prior.finalized_ordinal + 1
        segment_index, segment_position = _segment_location(
            phase=phase,
            prior=prior,
        )
        calendar_ref = _normalize_ref(
            feature.document["calendar_ref"], label="feature.calendar_ref"
        )
        calendar_binding = {
            "calendar_ref": calendar_ref,
            "decision_session": decision,
            "effective_session": decision,
            "observed_through_session": observed,
        }
        calendar_prefix = calendar_prefix_v1(
            through_session=decision,
            open_sessions=open_sessions,
            policy_byte_sha256=policy.ref["byte_sha256"],
            strategy_id=strategy,
        )
        prior_finality = _prior_finality(prior)
        segment_id = (
            segment_id_v1(
                chain_id=prior.chain_id,
                segment_index=segment_index,
                segment_start_session=decision,
                start_phase=phase,
            )
            if phase in {"GENESIS", "RECOVERY", "ROLLOVER"}
            else str(prior.segment_id)
        )
        record = _checkpoint_record(
            strategy_id=strategy,
            effective_session=decision,
            observed_session=observed,
            ordinal=ordinal,
            phase=phase,
            probabilities=probabilities,
            calendar_binding=calendar_binding,
            calendar_prefix=calendar_prefix,
            chain_id=prior.chain_id,
            chain_anchor_ref=prior.chain_anchor_ref,
            segment_id=segment_id,
            segment_index=segment_index,
            segment_position=segment_position,
            feature_ref=feature.ref,
            model_ref=model.ref,
            transition_ref=transition.ref,
            policy_ref=policy.ref,
            missing_sessions=missing,
            prior_finality=prior_finality,
        )
        rec_commitment = record_commitment_v1(record)
        current_accumulator = global_accumulator_v1(
            prior_accumulator=(
                global_seed_v1(chain_id=prior.chain_id, policy_ref=policy.ref, strategy_id=strategy)
                if ordinal == 0
                else prior.global_accumulator
            ),
            record_commitment=rec_commitment,
            finalized_ordinal=ordinal,
            chain_id=prior.chain_id,
        )
        prior_segment_acc = (
            segment_seed_v1(
                chain_id=prior.chain_id,
                policy_ref=policy.ref,
                previous_global_accumulator=(None if ordinal == 0 else prior.global_accumulator),
                segment_id=segment_id,
                segment_index=segment_index,
                segment_start_session=decision,
                start_phase=phase,
                strategy_id=strategy,
            )
            if segment_position == 0
            else str(prior.segment_accumulator)
        )
        segment_accumulator = segment_accumulator_v1(
            previous_accumulator=prior_segment_acc,
            record_commitment=rec_commitment,
            segment_id=segment_id,
            segment_position=segment_position,
        )
        if phase == "CONTIGUOUS":
            segment_ref = prior.checkpoint["segment_anchor_ref"]
            segment_path = segment_ref["relative_path"]
            segment = _read_ref(
                store=reader,
                ref=segment_ref,
                expected_version=SEGMENT_ANCHOR_VERSION,
                label="prior_segment_anchor",
                schema_optional=False,
            ).document
        else:
            segment = _make_segment(
                strategy_id=strategy,
                finalized_evidence_ordinal=ordinal,
                segment_index=segment_index,
                segment_position=segment_position,
                phase=phase,
                calendar_binding=calendar_binding,
                calendar_prefix=calendar_prefix,
                chain_anchor_ref=prior.chain_anchor_ref,
                chain_id=prior.chain_id,
                global_accumulator=current_accumulator,
                segment_id=segment_id,
                segment_accumulator=segment_accumulator,
                inference_policy_ref=policy.ref,
                missing_sessions=missing,
                prior_finality=prior_finality,
                record_commitment=rec_commitment,
                created_at=created_at,
                cutoff=cutoff,
            )
            segment = _seal_identity(
                segment,
                identity_field="segment_anchor_id",
            )
            segment_path = segment_anchor_v1_path(
                strategy_id=strategy,
                segment_commitment=str(segment["segment_anchor_id"]),
            )
        checkpoint = _make_checkpoint(
            strategy_id=strategy,
            effective_session=decision,
            finalized_evidence_ordinal=ordinal,
            segment_index=segment_index,
            segment_position=segment_position,
            phase=phase,
            record_commitment=rec_commitment,
            segment_anchor_ref=_artifact_ref(
                segment,
                canonical_resource_bytes(segment),
                relative_path=segment_path,
            ),
            calendar_binding=calendar_binding,
            calendar_prefix=calendar_prefix,
            global_accumulator=current_accumulator,
            segment_accumulator=segment_accumulator,
            chain_anchor_ref=prior.chain_anchor_ref,
            chain_id=prior.chain_id,
            segment_id=segment_id,
            prior_finality=prior_finality,
            inference_policy_ref=policy.ref,
            feature_snapshot_ref=feature.ref,
            transition_matrix_ref=transition.ref,
            model_snapshot_ref=model.ref,
            missing_sessions=missing,
            state_probabilities=probabilities,
            created_at=created_at,
            cutoff=cutoff,
        )
        checkpoint = _seal_identity(
            checkpoint,
            identity_field="checkpoint_id",
        )
        checkpoint_path = state_checkpoint_v1_path(
            strategy_id=strategy,
            effective_session=decision,
            checkpoint_id=str(checkpoint["checkpoint_id"]),
        )
        checkpoint_ref = _artifact_ref(
            checkpoint,
            canonical_resource_bytes(checkpoint),
            relative_path=checkpoint_path,
        )
        segment_ref = _artifact_ref(
            segment,
            canonical_resource_bytes(segment),
            relative_path=segment_path,
        )
        scope_ref = _normalize_ref(
            feature.document["pit_membership_ref"], label="feature.pit_membership_ref"
        )
        source_refs = _sorted_refs(
            [
                prior.chain_anchor_ref,
                checkpoint_ref,
                feature.ref,
                model.ref,
                scope_ref,
                segment_ref,
                transition.ref,
            ]
        )
        document = _make_evidence(
            evidence_id=evidence_id,
            strategy_id=strategy,
            decision_session=decision,
            observed_session=observed,
            cutoff=cutoff,
            created_at=created_at,
            phase=phase,
            ordinal=ordinal,
            probabilities=probabilities,
            record=record,
            record_commitment=rec_commitment,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            segment=segment,
            segment_path=segment_path,
            feature=feature.document,
            scope_ref=scope_ref,
            source_refs=source_refs,
            build_arguments=build_args,
        )
        raw = canonical_resource_bytes(document)
        _prepublication_validate(
            anchor=(anchor if anchor_to_publish is not None else None),
            segment=segment,
            checkpoint=checkpoint,
            evidence=document,
        )
        scoped_writer = _V3Store(
            workspace_root,
            strategy_id=strategy,
            allowed_paths=(
                evidence_path,
                segment_path,
                checkpoint_path,
                chain_anchor_v1_path(strategy_id=strategy),
            ),
        )
        try:
            if anchor_to_publish is not None:
                scoped_writer.write_exact_once(*anchor_to_publish)
            if phase != "CONTIGUOUS":
                scoped_writer.write_exact_once(segment_path, canonical_resource_bytes(segment))
            checkpoint_write = scoped_writer.write_exact_once(
                checkpoint_path,
                canonical_resource_bytes(checkpoint),
            )
            evidence_write = scoped_writer.write_exact_once(evidence_path, raw)
        except SourceExactOnceConflict as exc:
            raise RegimeEvidenceV3Conflict("v3 fixed slot or content object conflict") from exc
        except SourceStorageError as exc:
            raise RegimeEvidenceV3Error(
                CONFLICT_BLOCKER, "immutable v3 publication failed"
            ) from exc
        replayed = replay_regime_evidence_v3(
            workspace_root=workspace_root,
            evidence_path=evidence_path,
            evidence_sha256=evidence_write.byte_sha256,
        )
        return _result(
            document=replayed,
            evidence_path=evidence_path,
            evidence_sha256=evidence_write.byte_sha256,
            checkpoint_path=checkpoint_write.relative_path,
            checkpoint_sha256=checkpoint_write.byte_sha256,
            created=evidence_write.created,
            reused=not evidence_write.created,
        )


def replay_regime_evidence_v3(
    *,
    workspace_root: str | Path,
    evidence_path: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    store = SourceStore(workspace_root)
    try:
        raw = store.read(evidence_path, evidence_sha256)
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV3InputGap(f"missing regime evidence {evidence_path}") from exc
    except (SourceCASMismatch, SourceStorageSecurityError) as exc:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, "evidence exact readback failed") from exc
    return _validate_evidence(
        store=store,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        raw=raw,
    )


def audit_regime_chain_v3(
    *,
    workspace_root: str | Path,
    evidence_refs: Sequence[Mapping[str, str]],
    expected_head_path: str,
    expected_head_sha256: str,
    audit_as_of_session: str,
    max_records: int = MAX_AUDIT_RECORDS,
) -> dict[str, Any]:
    if type(max_records) is not int or max_records <= 0 or max_records > MAX_AUDIT_RECORDS:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "audit max_records must be in 1..1000")
    if len(evidence_refs) > max_records:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "explicit audit stream exceeds max_records")
    if not evidence_refs:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "explicit audit stream is empty")
    started = monotonic()
    as_of = _session(audit_as_of_session, label="audit_as_of_session")
    expected_head_sha = _sha(expected_head_sha256, label="expected_head_sha256")
    store = SourceStore(workspace_root)
    rows: list[dict[str, Any]] = []
    previous_checkpoint_ref: dict[str, str] | None = None
    previous_session: str | None = None
    total_evidence_bytes = 0
    seen_refs: set[bytes] = set()
    for index, ref in enumerate(evidence_refs):
        if monotonic() - started > MAX_AUDIT_SECONDS:
            raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "explicit audit time budget exceeded")
        normalized = _normalize_ref(ref, label=f"evidence_refs[{index}]")
        ref_key = canonical_bytes(normalized)
        if ref_key in seen_refs:
            raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "duplicate evidence in audit stream")
        seen_refs.add(ref_key)
        if normalized["artifact_version"] != REGIME_EVIDENCE_V3_VERSION:
            raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "audit stream contains non-v3 evidence")
        raw = store.read(normalized["relative_path"], normalized["byte_sha256"])
        total_evidence_bytes += len(raw)
        if total_evidence_bytes > MAX_AUDIT_EVIDENCE_BYTES:
            raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "explicit audit byte budget exceeded")
        document = replay_regime_evidence_v3(
            workspace_root=workspace_root,
            evidence_path=normalized["relative_path"],
            evidence_sha256=normalized["byte_sha256"],
        )
        checkpoint_ref = _normalize_ref(
            document["current_checkpoint_ref"], label="current_checkpoint_ref"
        )
        checkpoint = _read_ref(
            store=store,
            ref=checkpoint_ref,
            expected_version=STATE_CHECKPOINT_VERSION,
            label="state_checkpoint",
            schema_optional=False,
        )
        if (
            previous_checkpoint_ref is not None
            and checkpoint.document.get("prior_finality", {}).get("prior_checkpoint_id")
            != previous_checkpoint_ref["artifact_id"]
        ):
            raise RegimeEvidenceV3Error(
                CHECKPOINT_BLOCKER,
                "audit stream predecessor checkpoint mismatch",
            )
        session = str(document["effective_session"])
        if previous_session is not None and session <= previous_session:
            raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "audit stream is not session ordered")
        if document["finalized_evidence_ordinal"] != index:
            raise RegimeEvidenceV3Error(
                CHECKPOINT_BLOCKER, "audit stream ordinal is not contiguous"
            )
        rows.append(
            {
                "checkpoint_id": checkpoint.document["checkpoint_id"],
                "effective_session": document["effective_session"],
                "evidence_id": document["evidence_id"],
                "finalized_evidence_ordinal": document["finalized_evidence_ordinal"],
                "global_accumulator": checkpoint.document["global_accumulator"],
                "record_commitment": document["record_commitment"],
            }
        )
        previous_checkpoint_ref = checkpoint_ref
        previous_session = session
    head = _normalize_ref(evidence_refs[-1], label="expected_head")
    if (
        head["relative_path"] != expected_head_path
        or head["byte_sha256"] != expected_head_sha
        or rows[-1]["effective_session"] != as_of
    ):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "audit head or as-of tail mismatch")
    return {
        "authority": dict(_NO_AUTHORITY),
        "authority_status": AUTHORITY_STATUS,
        "record_count": len(rows),
        "records": rows,
        "audit_as_of_session": as_of,
        "expected_head_path": expected_head_path,
        "expected_head_sha256": expected_head_sha,
        "evidence_bytes": total_evidence_bytes,
        "replay_seconds": f"{monotonic() - started:.6f}",
        "status": "AVAILABLE",
    }


def regime_chain_capacity_probe_v3(*, session_count: int) -> dict[str, Any]:
    """Exercise the production accumulator formulas with O(1) retained state."""

    if type(session_count) is not int or session_count < 1 or session_count > 1000:
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER,
            "capacity probe session_count must be in 1..1000",
        )
    strategy_id = "capacity-probe"
    policy_ref = {
        "byte_sha256": "a" * 64,
        "semantic_sha256": "b" * 64,
    }
    chain_id = chain_id_v1(policy_ref=policy_ref, strategy_id=strategy_id)
    global_accumulator = global_seed_v1(
        chain_id=chain_id,
        policy_ref=policy_ref,
        strategy_id=strategy_id,
    )
    segment_accumulator = ""
    segment_id = ""
    segment_index = 0
    segment_position = 0
    rollover_count = 0
    for ordinal in range(session_count):
        if ordinal and segment_position == SEGMENT_LENGTH - 1:
            segment_index += 1
            segment_position = 0
            rollover_count += 1
        elif ordinal:
            segment_position += 1
        record_commitment = _domain_hash(
            "myquant.v17.v4.regime-capacity-record.v1",
            {"finalized_evidence_ordinal": ordinal},
        )
        previous_global = global_accumulator
        global_accumulator = global_accumulator_v1(
            prior_accumulator=previous_global,
            record_commitment=record_commitment,
            finalized_ordinal=ordinal,
            chain_id=chain_id,
        )
        if segment_position == 0:
            segment_id = segment_id_v1(
                chain_id=chain_id,
                segment_index=segment_index,
                segment_start_session="2026-07-30",
                start_phase=("GENESIS" if ordinal == 0 else "ROLLOVER"),
            )
            segment_accumulator = segment_seed_v1(
                chain_id=chain_id,
                policy_ref=policy_ref,
                previous_global_accumulator=(None if ordinal == 0 else previous_global),
                segment_id=segment_id,
                segment_index=segment_index,
                segment_start_session="2026-07-30",
                start_phase=("GENESIS" if ordinal == 0 else "ROLLOVER"),
                strategy_id=strategy_id,
            )
        segment_accumulator = segment_accumulator_v1(
            previous_accumulator=segment_accumulator,
            record_commitment=record_commitment,
            segment_id=segment_id,
            segment_position=segment_position,
        )
    return {
        "authority": dict(_NO_AUTHORITY),
        "bounded": True,
        "daily_replay_depth_upper_bound": 5,
        "daily_replay_reference_upper_bound": 32,
        "daily_replay_unique_node_upper_bound": 13,
        "finalized_evidence_ordinal": session_count - 1,
        "final_global_accumulator": global_accumulator,
        "final_segment_accumulator": segment_accumulator,
        "final_segment_index": segment_index,
        "final_segment_position": segment_position,
        "hash_records_processed": session_count,
        "rollover_count": rollover_count,
        "segment_count": segment_index + 1,
        "session_count": session_count,
        "status": "DEPLOYABLE_WITHIN_EXISTING_DAILY_BUDGET",
    }


def _result(
    *,
    document: Mapping[str, Any],
    evidence_path: str,
    evidence_sha256: str,
    created: bool,
    reused: bool,
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
) -> RegimeEvidenceV3BuildResult:
    checkpoint_ref = _normalize_ref(
        document["current_checkpoint_ref"], label="current_checkpoint_ref"
    )
    return RegimeEvidenceV3BuildResult(
        status=str(document["status"]),
        authority=dict(document["authority"]),
        authority_status=AUTHORITY_STATUS,
        evidence_id=str(document["evidence_id"]),
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        chain_checkpoint_path=checkpoint_path or checkpoint_ref["relative_path"],
        chain_checkpoint_sha256=checkpoint_sha256 or checkpoint_ref["byte_sha256"],
        record_commitment=str(document["record_commitment"]),
        checkpoint_commitment=str(checkpoint_ref["artifact_id"]),
        created=created,
        reused=reused,
        document=dict(document),
    )


def _load_policy(*, inference_policy_path: str, inference_policy_sha256: str) -> _Loaded:
    try:
        expected = _sha(inference_policy_sha256, label="inference_policy_sha256")
        if inference_policy_path != INFERENCE_POLICY_V2_PATH:
            raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, "policy path is not v2 packaged path")
        raw = read_packaged_asset(INFERENCE_POLICY_V2_PATH)
        document = dict(load_packaged_json(INFERENCE_POLICY_V2_PATH))
        validate_semantic_sha(document)
    except PackageResourceError as exc:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, "packaged policy v2 unavailable") from exc
    except CanonicalContractError as exc:
        raise RegimeEvidenceV3Error(
            INPUT_TAMPER_BLOCKER, "packaged policy v2 is not sealed"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, "policy v2 byte SHA mismatch")
    if (
        document.get("version") != INFERENCE_POLICY_V2_VERSION
        or document.get("protocol_version") != PROTOCOL_VERSION
        or document.get("authority") != _NO_AUTHORITY
        or document.get("publication_mode") != v2.TEMPORAL_MODE
        or document.get("inference_kind") != v2.INFERENCE_KIND
        or document.get("state_order") != list(v2.STATE_ORDER)
        or document.get("audit_limits", {}).get("max_recovery_sessions") != MAX_GAP_SESSIONS
        or document.get("audit_limits", {}).get("segment_length") != SEGMENT_LENGTH
        or document.get("model_helper_sha256") != v2.implementation_sha256()
        or document.get("producer_sha256") != implementation_sha256()
    ):
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "policy v2 semantic mismatch")
    ref = {
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "relative_path": INFERENCE_POLICY_V2_PATH,
        "semantic_sha256": str(document["semantic_sha256"]),
        "version": INFERENCE_POLICY_V2_VERSION,
    }
    return _Loaded(document, raw, ref)


def _load_snapshot(
    *,
    store: SourceStore,
    path: str,
    sha256: str,
    expected_version: str,
    strategy_id: str,
    label: str,
) -> _Loaded:
    expected = _sha(sha256, label=f"{label}_sha256")
    try:
        raw = store.read(path, expected)
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV3InputGap(f"missing current input {path}") from exc
    except (SourceCASMismatch, SourceStorageSecurityError) as exc:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, f"{label} exact readback failed") from exc
    document = _load_canonical(
        raw, expected_version=expected_version, label=label, schema_optional=False
    )
    if document.get("version") != expected_version or document.get("strategy_id") != strategy_id:
        raise RegimeEvidenceV3Error(IDENTITY_BLOCKER, f"{label} identity binding mismatch")
    ref = _artifact_ref(document, raw, relative_path=path)
    return _Loaded(document, raw, ref)


def _read_ref(
    *,
    store: SourceStore,
    ref: Mapping[str, str],
    expected_version: str,
    label: str,
    schema_optional: bool,
) -> _Loaded:
    normalized = _normalize_ref(ref, label=label)
    if normalized["artifact_version"] != expected_version:
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, f"{label} version mismatch")
    try:
        raw = store.read(normalized["relative_path"], normalized["byte_sha256"])
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV3InputGap(f"missing referenced {label}") from exc
    except (SourceCASMismatch, SourceStorageSecurityError) as exc:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, f"{label} exact read failed") from exc
    document = _load_canonical(
        raw,
        expected_version=expected_version,
        label=label,
        schema_optional=schema_optional,
    )
    observed_ref = _artifact_ref(document, raw, relative_path=normalized["relative_path"])
    if observed_ref != normalized:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, f"{label} reference binding mismatch")
    return _Loaded(document, raw, normalized)


def _load_canonical(
    raw: bytes,
    *,
    expected_version: str,
    label: str,
    schema_optional: bool,
) -> dict[str, Any]:
    try:
        schema_path_for_version(expected_version)
    except SchemaValidationError:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, f"{label} schema is not registered")
    try:
        loaded = load_canonical_artifact(raw, expected_version=expected_version, label=label)
        return dict(loaded.payload)
    except ArtifactContractError as exc:
        raise RegimeEvidenceV3Error(
            SEMANTIC_BLOCKER, f"{label} contract validation failed: {exc}"
        ) from exc
    except (SchemaValidationError, CanonicalContractError) as exc:
        raise RegimeEvidenceV3Error(
            INPUT_TAMPER_BLOCKER, f"{label} canonical validation failed"
        ) from exc


def _validate_current_closure(
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    policy: Mapping[str, Any],
    policy_ref: Mapping[str, str],
    feature: Mapping[str, Any],
    transition: Mapping[str, Any],
    model: Mapping[str, Any],
    transition_ref: Mapping[str, str],
) -> tuple[str, list[str]]:
    observed = _session(feature.get("observed_through_session"), label="observed_through_session")
    effective = _session(feature.get("effective_session"), label="effective_session")
    if effective != decision_session:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "feature effective session is not D")
    open_sessions = _open_sessions(feature=feature)
    if observed not in open_sessions or decision_session not in open_sessions:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "sealed feature calendar lacks S or D")
    if open_sessions.index(decision_session) != open_sessions.index(observed) + 1:
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER,
            "observed session is not the immediate prior sealed open session",
        )
    first_eligible = _session(
        policy.get("bootstrap", {}).get("first_eligible_decision_session"),
        label="policy.bootstrap.first_eligible_decision_session",
    )
    if decision_session < first_eligible:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "decision session predates v3 policy")
    for label, snapshot in (("feature", feature), ("transition", transition), ("model", model)):
        if (
            snapshot.get("strategy_id") != strategy_id
            or snapshot.get("observed_through_session") != observed
            or snapshot.get("effective_session") != decision_session
        ):
            raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, f"{label} session binding mismatch")
        _, available = _timestamp(snapshot.get("available_at"), label=f"{label}.available_at")
        _, decision_cutoff = _timestamp(cutoff, label="cutoff")
        if available > decision_cutoff:
            raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, f"{label} available after cutoff")
    risk_on, volatility, pressure, likelihoods = v2._derive_scores_and_likelihoods(feature)
    if (
        feature.get("risk_on_score") != risk_on
        or feature.get("volatility_score") != volatility
        or feature.get("pressure_score") != pressure
        or feature.get("state_likelihoods") != likelihoods
    ):
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "feature fixed-rule replay mismatch")
    if (
        transition.get("transition_matrix") != policy.get("transition_matrix")
        or transition.get("inference_policy_ref") != policy_ref
        or model.get("inference_policy_ref") != policy_ref
        or model.get("transition_matrix_ref") != transition_ref
        or model.get("model_helper_sha256") != policy.get("model_helper_sha256")
        or model.get("producer_sha256") != policy.get("producer_sha256")
        or model.get("model_training_end_session") is not None
        or model.get("training_source_refs") != []
    ):
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "current direct closure mismatch")
    return observed, open_sessions


def _open_sessions(*, feature: Mapping[str, Any]) -> list[str]:
    raw = feature.get("open_sessions")
    if type(raw) is not list or any(type(item) is not str for item in raw):
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "sealed open calendar is missing")
    sessions = [_session(item, label="open_session") for item in raw]
    if sessions != sorted(set(sessions)):
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "sealed open calendar is not sorted unique")
    return sessions


def _load_prior(
    *,
    store: SourceStore,
    strategy_id: str,
    observed_session: str,
    prior_evidence_path: str | None,
    prior_evidence_sha256: str | None,
    prior_checkpoint_path: str | None,
    prior_checkpoint_sha256: str | None,
    chain_anchor_path: str | None,
    chain_anchor_sha256: str | None,
) -> _Prior:
    if (prior_evidence_path is None) != (prior_evidence_sha256 is None):
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "prior evidence path/SHA pair is incomplete")
    if (prior_checkpoint_path is None) != (prior_checkpoint_sha256 is None):
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER, "prior checkpoint path/SHA pair is incomplete"
        )
    if (chain_anchor_path is None) != (chain_anchor_sha256 is None):
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "chain anchor path/SHA pair is incomplete")
    if prior_evidence_path is not None:
        if prior_checkpoint_path is None:
            raise RegimeEvidenceV3Error(
                TEMPORAL_BLOCKER, "explicit prior requires explicit checkpoint"
            )
        prior = _read_path_sha(
            store=store,
            path=prior_evidence_path,
            sha256=str(prior_evidence_sha256),
            expected_version=REGIME_EVIDENCE_V3_VERSION,
            label="prior_evidence",
            strategy_id=strategy_id,
            schema_optional=False,
        )
        checkpoint = _read_path_sha(
            store=store,
            path=prior_checkpoint_path,
            sha256=str(prior_checkpoint_sha256),
            expected_version=STATE_CHECKPOINT_VERSION,
            label="prior_checkpoint",
            strategy_id=strategy_id,
            schema_optional=False,
        )
        _validate_prior(prior=prior.document, prior_ref=prior.ref, checkpoint=checkpoint.document)
        if prior.document["effective_session"] > observed_session:
            raise RegimeEvidenceV3Error(
                STALE_PRIOR_BLOCKER, "prior is after current observed session"
            )
        return _Prior(
            document=prior.document,
            checkpoint=checkpoint.document,
            evidence_ref=prior.ref,
            checkpoint_ref=checkpoint.ref,
            chain_anchor_ref=prior.document.get("chain_anchor_ref"),
            finalized_ordinal=_ordinal(checkpoint.document["finalized_evidence_ordinal"]),
            effective_session=str(prior.document["effective_session"]),
            state_probabilities=dict(prior.document["state_probabilities"]),
            chain_id=str(checkpoint.document["chain_id"]),
            global_accumulator=str(checkpoint.document["global_accumulator"]),
            segment_id=str(checkpoint.document["segment_id"]),
            segment_index=int(checkpoint.document["segment_index"]),
            segment_position=int(checkpoint.document["segment_position"]),
            segment_accumulator=str(checkpoint.document["segment_accumulator"]),
        )
    if chain_anchor_path is None:
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER, "anchor-only recovery requires explicit anchor"
        )
    if chain_anchor_path != chain_anchor_v1_path(strategy_id=strategy_id):
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "chain anchor is not fixed strategy slot")
    anchor = _read_path_sha(
        store=store,
        path=chain_anchor_path,
        sha256=str(chain_anchor_sha256),
        expected_version=CHAIN_ANCHOR_VERSION,
        label="chain_anchor",
        strategy_id=strategy_id,
        schema_optional=False,
    )
    if anchor.document.get("chain_anchor_id") != regime_artifact_identity(
        anchor.document,
        identity_field="chain_anchor_id",
    ):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "chain anchor commitment mismatch")
    bootstrap_observed = _session(
        anchor.document.get("bootstrap_observed_through_session"),
        label="anchor.bootstrap_observed_through_session",
    )
    if bootstrap_observed > observed_session:
        raise RegimeEvidenceV3Error(STALE_PRIOR_BLOCKER, "anchor is after observed session")
    probabilities = v2._serialized_probabilities(
        anchor.document.get("bootstrap_prior"),
        label="anchor.bootstrap_prior",
    )
    return _Prior(
        document=None,
        checkpoint=anchor.document,
        evidence_ref=None,
        checkpoint_ref=anchor.ref,
        chain_anchor_ref=anchor.ref,
        finalized_ordinal=0,
        effective_session=bootstrap_observed,
        state_probabilities=probabilities,
        chain_id=str(anchor.document["chain_id"]),
        global_accumulator=str(anchor.document["global_seed"]),
        segment_id=None,
        segment_index=0,
        segment_position=0,
        segment_accumulator=None,
    )


def _prior_from_anchor(
    *,
    anchor: Mapping[str, Any],
    anchor_path: str,
    anchor_raw: bytes,
    observed_session: str,
) -> _Prior:
    anchor_ref = _artifact_ref(anchor, anchor_raw, relative_path=anchor_path)
    bootstrap_observed = _session(
        anchor.get("bootstrap_observed_through_session"),
        label="anchor.bootstrap_observed_through_session",
    )
    if bootstrap_observed > observed_session:
        raise RegimeEvidenceV3Error(STALE_PRIOR_BLOCKER, "anchor is after observed session")
    probabilities = v2._serialized_probabilities(
        anchor.get("bootstrap_prior"),
        label="anchor.bootstrap_prior",
    )
    return _Prior(
        document=None,
        checkpoint=dict(anchor),
        evidence_ref=None,
        checkpoint_ref=anchor_ref,
        chain_anchor_ref=anchor_ref,
        finalized_ordinal=0,
        effective_session=bootstrap_observed,
        state_probabilities=probabilities,
        chain_id=str(anchor["chain_id"]),
        global_accumulator=str(anchor["global_seed"]),
        segment_id=None,
        segment_index=0,
        segment_position=0,
        segment_accumulator=None,
    )


def _validate_prior(
    *,
    prior: Mapping[str, Any],
    prior_ref: Mapping[str, str],
    checkpoint: Mapping[str, Any],
) -> None:
    if (
        prior.get("version") != REGIME_EVIDENCE_V3_VERSION
        or prior.get("status") != "AVAILABLE"
        or prior.get("blocker_codes") != []
        or prior.get("evidence_id") != regime_evidence_v3_id(prior)
        or prior.get("current_checkpoint_ref") is None
        or checkpoint.get("version") != STATE_CHECKPOINT_VERSION
        or checkpoint.get("record_commitment") != prior.get("record_commitment")
        or checkpoint.get("checkpoint_id") != chain_checkpoint_commitment_v1(checkpoint)
    ):
        raise RegimeEvidenceV3Error(
            CHECKPOINT_BLOCKER, "prior evidence/checkpoint is not finalized"
        )
    expected = regime_evidence_v3_path(
        strategy_id=str(prior["strategy_id"]),
        effective_session=str(prior["effective_session"]),
    )
    if prior_ref["relative_path"] != expected:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "prior evidence path is not deterministic")


def _missing_sessions(
    *,
    open_sessions: Sequence[str],
    after_session: str,
    before_session: str,
) -> list[str]:
    sessions = list(open_sessions)
    if after_session not in sessions or before_session not in sessions:
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER, "prior/current sessions absent from sealed calendar"
        )
    start = sessions.index(after_session)
    end = sessions.index(before_session)
    if start >= end:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "prior must precede D")
    return sessions[start + 1 : end]


def _reject_stale_prior(
    *,
    store: SourceStore,
    strategy_id: str,
    missing_sessions: Sequence[str],
) -> None:
    for session in missing_sessions:
        path = regime_evidence_v3_path(strategy_id=strategy_id, effective_session=session)
        if store.read_optional(path) is not None:
            raise RegimeEvidenceV3Error(
                STALE_PRIOR_BLOCKER, "missing-open slot already has finalized evidence"
            )


def _phase(
    *,
    prior_document: Mapping[str, Any] | None,
    prior_ordinal: int,
    missing_sessions: Sequence[str],
) -> str:
    if missing_sessions:
        return "RECOVERY"
    if prior_document is None and prior_ordinal == 0:
        return "GENESIS"
    if prior_ordinal % SEGMENT_LENGTH == 63:
        return "ROLLOVER"
    return "CONTIGUOUS"


def _propagate_missing(
    *,
    start_probabilities: Mapping[str, Any],
    missing_sessions: Sequence[str],
    transition_matrix: Mapping[str, Any],
) -> dict[str, str]:
    probabilities = v2._serialized_probabilities(
        start_probabilities, label="prior.state_probabilities"
    )
    neutral_likelihood = {state: "0.200000000000" for state in v2.STATE_ORDER}
    for _session_item in missing_sessions:
        probabilities = v2._posterior(
            previous=probabilities,
            transition_matrix=transition_matrix,
            likelihoods=neutral_likelihood,
        )
    return probabilities


def _segment_location(*, phase: str, prior: _Prior) -> tuple[int, int]:
    if prior.document is None:
        return 0, 0
    prior_index = int(prior.checkpoint.get("segment_index", 0))
    prior_position = int(prior.checkpoint.get("segment_position", 0))
    if phase in {"RECOVERY", "ROLLOVER"}:
        return prior_index + 1, 0
    return prior_index, prior_position + 1


def _checkpoint_record(
    *,
    strategy_id: str,
    effective_session: str,
    observed_session: str,
    ordinal: int,
    phase: str,
    probabilities: Mapping[str, str],
    calendar_binding: Mapping[str, Any],
    calendar_prefix: Mapping[str, Any],
    chain_id: str,
    chain_anchor_ref: Mapping[str, str],
    segment_id: str,
    segment_index: int,
    segment_position: int,
    feature_ref: Mapping[str, str],
    model_ref: Mapping[str, str],
    transition_ref: Mapping[str, str],
    policy_ref: Mapping[str, str],
    missing_sessions: Sequence[str],
    prior_finality: Mapping[str, Any],
) -> dict[str, Any]:
    missing = list(missing_sessions)
    return {
        "calendar_binding": dict(calendar_binding),
        "calendar_prefix": dict(calendar_prefix),
        "chain_anchor_ref": dict(chain_anchor_ref),
        "chain_id": _sha(chain_id, label="chain_id"),
        "effective_session": _session(effective_session, label="effective_session"),
        "evidence_ordinal": _ordinal(ordinal),
        "feature_snapshot_ref": dict(feature_ref),
        "hard_state": v2._hard_state(probabilities),
        "inference_policy_ref": dict(policy_ref),
        "missing_sessions": missing,
        "missing_sessions_digest": _domain_hash(
            "myquant.v17.v4.regime-missing-sessions.v1",
            {"sessions": missing},
        ),
        "model_snapshot_ref": dict(model_ref),
        "observed_through_session": _session(
            observed_session,
            label="observed_through_session",
        ),
        "phase": phase,
        "prior_finality": dict(prior_finality),
        "segment_id": _sha(segment_id, label="segment_id"),
        "segment_index": segment_index,
        "segment_position": segment_position,
        "state_probabilities": dict(probabilities),
        "strategy_id": _strategy_id(strategy_id),
        "transition_matrix_ref": dict(transition_ref),
        "version": "myquant.v17.v4.regime-checkpoint-record.v1",
    }


def _prior_finality(prior: _Prior) -> dict[str, Any]:
    if prior.document is None:
        return {
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
    return {
        "prior_checkpoint_byte_sha256": prior.checkpoint_ref["byte_sha256"],
        "prior_checkpoint_id": prior.checkpoint["checkpoint_id"],
        "prior_checkpoint_semantic_sha256": prior.checkpoint_ref["semantic_sha256"],
        "prior_effective_session": prior.document.get("effective_session"),
        "prior_evidence_byte_sha256": (
            prior.evidence_ref["byte_sha256"] if prior.evidence_ref else None
        ),
        "prior_evidence_id": prior.document["evidence_id"],
        "prior_evidence_semantic_sha256": (
            prior.evidence_ref["semantic_sha256"] if prior.evidence_ref else None
        ),
        "prior_finalized_evidence_ordinal": prior.checkpoint.get("finalized_evidence_ordinal"),
        "prior_global_accumulator": prior.checkpoint.get("global_accumulator"),
        "prior_segment_id": prior.checkpoint.get("segment_id"),
        "prior_segment_index": prior.checkpoint.get("segment_index"),
        "prior_segment_position": prior.checkpoint.get("segment_position"),
    }


def _make_segment(
    *,
    strategy_id: str,
    finalized_evidence_ordinal: int,
    segment_index: int,
    segment_position: int,
    phase: str,
    calendar_binding: Mapping[str, Any],
    calendar_prefix: Mapping[str, Any],
    chain_anchor_ref: Mapping[str, str] | None,
    chain_id: str,
    global_accumulator: str,
    segment_id: str,
    segment_accumulator: str,
    inference_policy_ref: Mapping[str, str],
    missing_sessions: Sequence[str],
    prior_finality: Mapping[str, Any],
    record_commitment: str,
    created_at: str,
    cutoff: str,
) -> dict[str, Any]:
    authority = regime_evidence_v3_authority_attestation()
    return {
        "authority": authority["authority"],
        "calendar_binding": dict(calendar_binding),
        "calendar_prefix": dict(calendar_prefix),
        "chain_anchor_ref": dict(chain_anchor_ref or {}),
        "chain_id": _sha(chain_id, label="chain_id"),
        "created_at": created_at,
        "cutoff": cutoff,
        "finalized_evidence_ordinal": finalized_evidence_ordinal,
        "formal_activation_eligible": authority["formal_activation_eligible"],
        "global_accumulator": _sha_state(global_accumulator),
        "inference_policy_ref": dict(inference_policy_ref),
        "missing_sessions": list(missing_sessions),
        "performance_evidence_eligible": authority["performance_evidence_eligible"],
        "phase": phase,
        "prior_finality": dict(prior_finality),
        "promotion_eligible": authority["promotion_eligible"],
        "protocol_version": PROTOCOL_VERSION,
        "record_commitment": record_commitment,
        "segment_accumulator": _sha_state(segment_accumulator),
        "segment_anchor_id": "0" * 64,
        "segment_id": _sha(segment_id, label="segment_id"),
        "segment_index": segment_index,
        "segment_position": segment_position,
        "semantic_sha256": "0" * 64,
        "shadow_only": authority["shadow_only"],
        "state_order": list(v2.STATE_ORDER),
        "strategy_id": strategy_id,
        "version": SEGMENT_ANCHOR_VERSION,
    }


def _make_chain_anchor(
    *,
    strategy_id: str,
    created_at: str,
    cutoff: str,
    policy: Mapping[str, Any],
    policy_ref: Mapping[str, str],
    model_ref: Mapping[str, str],
    transition_ref: Mapping[str, str],
    open_sessions: Sequence[str],
) -> dict[str, Any]:
    authority = regime_evidence_v3_authority_attestation()
    first_eligible = _session(
        policy["bootstrap"]["first_eligible_decision_session"],
        label="policy.bootstrap.first_eligible_decision_session",
    )
    observed = _session(
        policy["bootstrap"]["first_observed_through_session"],
        label="policy.bootstrap.first_observed_through_session",
    )
    chain_id = chain_id_v1(policy_ref=policy_ref, strategy_id=strategy_id)
    body = {
        "audit_limits": {
            "closure_due_time_local": "23:59:59+08:00",
            "max_recovery_sessions": MAX_GAP_SESSIONS,
            "segment_length": SEGMENT_LENGTH,
        },
        "authority": authority["authority"],
        "bootstrap_observed_through_session": observed,
        "bootstrap_prior": dict(policy["bootstrap"]["prior"]),
        "calendar_prefix": calendar_prefix_v1(
            through_session=first_eligible,
            open_sessions=open_sessions,
            policy_byte_sha256=policy_ref["byte_sha256"],
            strategy_id=strategy_id,
        ),
        "chain_anchor_id": "0" * 64,
        "chain_id": chain_id,
        "created_at": created_at,
        "cutoff": cutoff,
        "first_eligible_session": first_eligible,
        "formal_activation_eligible": authority["formal_activation_eligible"],
        "global_seed": global_seed_v1(
            chain_id=chain_id,
            policy_ref=policy_ref,
            strategy_id=strategy_id,
        ),
        "inference_policy_ref": dict(policy_ref),
        "model_snapshot_ref": dict(model_ref),
        "performance_evidence_eligible": authority["performance_evidence_eligible"],
        "phase": "GENESIS",
        "promotion_eligible": authority["promotion_eligible"],
        "protocol_version": PROTOCOL_VERSION,
        "semantic_sha256": "0" * 64,
        "shadow_only": authority["shadow_only"],
        "state_order": list(v2.STATE_ORDER),
        "strategy_id": strategy_id,
        "transition_matrix_ref": dict(transition_ref),
        "version": CHAIN_ANCHOR_VERSION,
    }
    return _seal_identity(body, identity_field="chain_anchor_id")


def _make_checkpoint(
    *,
    strategy_id: str,
    effective_session: str,
    finalized_evidence_ordinal: int,
    segment_index: int,
    segment_position: int,
    phase: str,
    record_commitment: str,
    segment_anchor_ref: Mapping[str, str],
    calendar_binding: Mapping[str, Any],
    calendar_prefix: Mapping[str, Any],
    global_accumulator: str,
    segment_accumulator: str,
    chain_anchor_ref: Mapping[str, str] | None,
    chain_id: str,
    segment_id: str,
    prior_finality: Mapping[str, Any],
    inference_policy_ref: Mapping[str, str],
    feature_snapshot_ref: Mapping[str, str],
    transition_matrix_ref: Mapping[str, str],
    model_snapshot_ref: Mapping[str, str],
    missing_sessions: Sequence[str],
    state_probabilities: Mapping[str, str],
    created_at: str,
    cutoff: str,
) -> dict[str, Any]:
    authority = regime_evidence_v3_authority_attestation()
    return {
        "authority": authority["authority"],
        "calendar_binding": dict(calendar_binding),
        "calendar_prefix": dict(calendar_prefix),
        "chain_anchor_ref": dict(chain_anchor_ref or {}),
        "chain_id": _sha(chain_id, label="chain_id"),
        "checkpoint_id": "0" * 64,
        "created_at": created_at,
        "cutoff": cutoff,
        "feature_snapshot_ref": dict(feature_snapshot_ref),
        "finalized_evidence_ordinal": finalized_evidence_ordinal,
        "formal_activation_eligible": authority["formal_activation_eligible"],
        "global_accumulator": _sha_state(global_accumulator),
        "hard_state": v2._hard_state(state_probabilities),
        "hard_state_derivation": v2.HARD_STATE_DERIVATION,
        "inference_policy_ref": dict(inference_policy_ref),
        "missing_sessions": list(missing_sessions),
        "model_snapshot_ref": dict(model_snapshot_ref),
        "performance_evidence_eligible": authority["performance_evidence_eligible"],
        "phase": phase,
        "prior_finality": dict(prior_finality),
        "promotion_eligible": authority["promotion_eligible"],
        "protocol_version": PROTOCOL_VERSION,
        "record_commitment": record_commitment,
        "segment_accumulator": _sha_state(segment_accumulator),
        "segment_anchor_ref": dict(segment_anchor_ref),
        "segment_id": _sha(segment_id, label="segment_id"),
        "segment_index": segment_index,
        "segment_position": segment_position,
        "semantic_sha256": "0" * 64,
        "shadow_only": authority["shadow_only"],
        "state_order": list(v2.STATE_ORDER),
        "state_probabilities": dict(state_probabilities),
        "strategy_id": strategy_id,
        "transition_matrix_ref": dict(transition_matrix_ref),
        "version": STATE_CHECKPOINT_VERSION,
    }


def _make_evidence(
    *,
    evidence_id: str,
    strategy_id: str,
    decision_session: str,
    observed_session: str,
    cutoff: str,
    created_at: str,
    phase: str,
    ordinal: int,
    probabilities: Mapping[str, str],
    record: Mapping[str, Any],
    record_commitment: str,
    checkpoint: Mapping[str, Any],
    checkpoint_path: str,
    segment: Mapping[str, Any],
    segment_path: str,
    feature: Mapping[str, Any],
    scope_ref: Mapping[str, str],
    source_refs: Sequence[Mapping[str, str]],
    build_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    del build_arguments
    authority = regime_evidence_v3_authority_attestation()
    checkpoint_ref = _artifact_ref(
        checkpoint,
        canonical_resource_bytes(checkpoint),
        relative_path=checkpoint_path,
    )
    segment_ref = _artifact_ref(
        segment,
        canonical_resource_bytes(segment),
        relative_path=segment_path,
    )
    body: dict[str, Any] = {
        "authority": authority["authority"],
        "available_at": created_at,
        "blocker_codes": [],
        "calendar_binding": dict(record["calendar_binding"]),
        "calendar_prefix": dict(record["calendar_prefix"]),
        "chain_anchor_ref": dict(record["chain_anchor_ref"]),
        "chain_id": _sha(record["chain_id"], label="chain_id"),
        "computed_at": created_at,
        "created_at": created_at,
        "current_checkpoint_ref": checkpoint_ref,
        "coverage_ratio": str(feature["coverage_ratio"]),
        "cutoff": cutoff,
        "decision_session": decision_session,
        "evidence_id": evidence_id,
        "effective_session": decision_session,
        "feature_cutoff": str(feature["cutoff"]),
        "feature_snapshot_ref": dict(record["feature_snapshot_ref"]),
        "finalized_evidence_ordinal": ordinal,
        "formal_activation_eligible": authority["formal_activation_eligible"],
        "global_accumulator": _sha_state(checkpoint["global_accumulator"]),
        "hard_state": v2._hard_state(probabilities),
        "hard_state_derivation": v2.HARD_STATE_DERIVATION,
        "inference_kind": v2.INFERENCE_KIND,
        "inference_policy_ref": dict(record["inference_policy_ref"]),
        "market_sample_count": int(feature["market_sample_count"]),
        "minimum_market_sample": int(feature["minimum_market_sample"]),
        "missing_sessions": list(record["missing_sessions"]),
        "model_snapshot_ref": dict(record["model_snapshot_ref"]),
        "no_retroactive_causal_backfill": authority["no_retroactive_causal_backfill"],
        "observed_through_session": observed_session,
        "performance_evidence_eligible": authority["performance_evidence_eligible"],
        "phase": phase,
        "prior_finality": dict(record["prior_finality"]),
        "promotion_eligible": authority["promotion_eligible"],
        "protocol_version": PROTOCOL_VERSION,
        "publication_phase": v2.TEMPORAL_MODE,
        "published_at": created_at,
        "record_commitment": record_commitment,
        "same_session_execution_eligible": authority["same_session_execution_eligible"],
        "scope_kind": str(feature["scope_kind"]),
        "scope_ref": dict(scope_ref),
        "segment_accumulator": _sha_state(checkpoint["segment_accumulator"]),
        "segment_anchor_ref": segment_ref,
        "segment_id": _sha(record["segment_id"], label="segment_id"),
        "segment_index": record["segment_index"],
        "segment_position": record["segment_position"],
        "shadow_only": authority["shadow_only"],
        "smoothing_used": False,
        "source_refs": [dict(ref) for ref in source_refs],
        "state_order": list(v2.STATE_ORDER),
        "state_probabilities": dict(probabilities),
        "status": "AVAILABLE",
        "strategy_id": strategy_id,
        "transition_matrix_ref": dict(record["transition_matrix_ref"]),
        "version": REGIME_EVIDENCE_V3_VERSION,
    }
    expected = regime_evidence_v3_id(body)
    if evidence_id != expected:
        raise RegimeEvidenceV3Error(IDENTITY_BLOCKER, f"evidence_id mismatch; expected {expected}")
    return seal_semantic(body)


def _validate_evidence(
    *,
    store: SourceStore,
    evidence_path: str,
    evidence_sha256: str,
    raw: bytes,
    enforce_build_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _sha(evidence_sha256, label="evidence_sha256")
    if hashlib.sha256(raw).hexdigest() != evidence_sha256:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, "evidence SHA mismatch")
    document = _load_canonical(
        raw,
        expected_version=REGIME_EVIDENCE_V3_VERSION,
        label="regime_evidence_v3",
        schema_optional=False,
    )
    strategy = _strategy_id(document.get("strategy_id"))
    effective = _session(document.get("effective_session"), label="effective_session")
    if evidence_path != regime_evidence_v3_path(strategy_id=strategy, effective_session=effective):
        raise RegimeEvidenceV3Error(
            INPUT_TAMPER_BLOCKER, "evidence path is not fixed strategy/effective slot"
        )
    if (
        document.get("authority") != _NO_AUTHORITY
        or document.get("blocker_codes") != []
        or document.get("status") != "AVAILABLE"
        or document.get("evidence_id") != regime_evidence_v3_id(document)
    ):
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "evidence self closure mismatch")
    probabilities = v2._serialized_probabilities(
        document.get("state_probabilities"),
        label="state_probabilities",
    )
    if document.get("hard_state") != v2._hard_state(probabilities):
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "sealed hard state is not argmax")
    checkpoint = _read_ref(
        store=store,
        ref=document["current_checkpoint_ref"],
        expected_version=STATE_CHECKPOINT_VERSION,
        label="state_checkpoint",
        schema_optional=False,
    )
    segment = _read_ref(
        store=store,
        ref=document["segment_anchor_ref"],
        expected_version=SEGMENT_ANCHOR_VERSION,
        label="segment_anchor",
        schema_optional=False,
    )
    if (
        checkpoint.document.get("record_commitment") != document["record_commitment"]
        or checkpoint.document.get("checkpoint_id")
        != chain_checkpoint_commitment_v1(checkpoint.document)
        or segment.document.get("segment_anchor_id")
        != checkpoint.document.get("segment_anchor_ref", {}).get("artifact_id")
        or segment.document.get("segment_anchor_id")
        != segment_anchor_commitment_v1(segment.document)
    ):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "checkpoint/segment commitment mismatch")
    _validate_daily_replay_closure(
        store=store,
        evidence=document,
        checkpoint=checkpoint,
        segment=segment,
    )
    if enforce_build_arguments is not None:
        _validate_retry_arguments(document=document, expected=enforce_build_arguments)
    return document


def _validate_daily_replay_closure(
    *,
    store: SourceStore,
    evidence: Mapping[str, Any],
    checkpoint: _Loaded,
    segment: _Loaded,
) -> None:
    strategy = str(evidence["strategy_id"])
    policy = _load_policy(
        inference_policy_path=str(evidence["inference_policy_ref"]["relative_path"]),
        inference_policy_sha256=str(evidence["inference_policy_ref"]["byte_sha256"]),
    )
    anchor = _read_ref(
        store=store,
        ref=evidence["chain_anchor_ref"],
        expected_version=CHAIN_ANCHOR_VERSION,
        label="chain_anchor",
        schema_optional=False,
    )
    feature = _read_ref(
        store=store,
        ref=evidence["feature_snapshot_ref"],
        expected_version=v2.FEATURE_SNAPSHOT_VERSION,
        label="feature_snapshot",
        schema_optional=False,
    )
    model = _read_ref(
        store=store,
        ref=evidence["model_snapshot_ref"],
        expected_version=MODEL_SNAPSHOT_V2_VERSION,
        label="model_snapshot",
        schema_optional=False,
    )
    transition = _read_ref(
        store=store,
        ref=evidence["transition_matrix_ref"],
        expected_version=TRANSITION_SNAPSHOT_V2_VERSION,
        label="transition_matrix",
        schema_optional=False,
    )
    scope = _read_ref(
        store=store,
        ref=evidence["scope_ref"],
        expected_version=v2.PIT_TERMINAL_VERSION,
        label="scope_ref",
        schema_optional=False,
    )
    closure_loader = v2._ClosureLoader(store)
    try:
        closure_loader(feature.ref)
        closure_loader(model.ref)
        closure_loader(transition.ref)
    except v2.RegimeEvidenceV2Error as exc:
        raise _translate_v2(exc) from exc
    observed, open_sessions = _validate_current_closure(
        strategy_id=strategy,
        decision_session=str(evidence["decision_session"]),
        cutoff=str(evidence["cutoff"]),
        policy=policy.document,
        policy_ref=policy.ref,
        feature=feature.document,
        transition=transition.document,
        model=model.document,
        transition_ref=transition.ref,
    )
    if (
        observed != evidence["observed_through_session"]
        or scope.ref != evidence["scope_ref"]
        or anchor.document["chain_id"] != evidence["chain_id"]
        or anchor.document["inference_policy_ref"] != policy.ref
    ):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "daily direct closure binding mismatch")
    expected_refs = _sorted_refs(
        [
            anchor.ref,
            checkpoint.ref,
            feature.ref,
            model.ref,
            scope.ref,
            segment.ref,
            transition.ref,
        ]
    )
    observed_refs = _sorted_refs(evidence["source_refs"])
    if observed_refs != expected_refs:
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "daily direct reference set mismatch")
    checkpoint_fields = (
        "calendar_binding",
        "calendar_prefix",
        "chain_anchor_ref",
        "chain_id",
        "feature_snapshot_ref",
        "finalized_evidence_ordinal",
        "global_accumulator",
        "hard_state",
        "inference_policy_ref",
        "missing_sessions",
        "model_snapshot_ref",
        "phase",
        "prior_finality",
        "record_commitment",
        "segment_accumulator",
        "segment_anchor_ref",
        "segment_id",
        "segment_index",
        "segment_position",
        "state_order",
        "state_probabilities",
        "transition_matrix_ref",
    )
    if any(checkpoint.document[field] != evidence[field] for field in checkpoint_fields):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "evidence/checkpoint binding mismatch")

    ordinal = _ordinal(evidence["finalized_evidence_ordinal"])
    prior_finality = evidence["prior_finality"]
    if ordinal == 0:
        if any(value is not None for value in prior_finality.values()):
            raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "first evidence has prior finality")
        prior_document = None
        prior_checkpoint = None
        prior_effective = str(anchor.document["bootstrap_observed_through_session"])
        prior_probabilities = dict(anchor.document["bootstrap_prior"])
        prior_global = str(anchor.document["global_seed"])
        prior_segment_accumulator = None
        expected_phase = "RECOVERY" if evidence["missing_sessions"] else "GENESIS"
        expected_segment_index = 0
        expected_segment_position = 0
    else:
        prior_session = str(prior_finality["prior_effective_session"])
        prior_evidence_path = regime_evidence_v3_path(
            strategy_id=strategy,
            effective_session=prior_session,
        )
        prior_checkpoint_path = state_checkpoint_v1_path(
            strategy_id=strategy,
            effective_session=prior_session,
            checkpoint_id=str(prior_finality["prior_checkpoint_id"]),
        )
        prior_document = _read_path_sha(
            store=store,
            path=prior_evidence_path,
            sha256=str(prior_finality["prior_evidence_byte_sha256"]),
            expected_version=REGIME_EVIDENCE_V3_VERSION,
            label="prior_evidence",
            strategy_id=strategy,
            schema_optional=False,
        )
        prior_checkpoint = _read_path_sha(
            store=store,
            path=prior_checkpoint_path,
            sha256=str(prior_finality["prior_checkpoint_byte_sha256"]),
            expected_version=STATE_CHECKPOINT_VERSION,
            label="prior_checkpoint",
            strategy_id=strategy,
            schema_optional=False,
        )
        _validate_prior(
            prior=prior_document.document,
            prior_ref=prior_document.ref,
            checkpoint=prior_checkpoint.document,
        )
        prior_state = _Prior(
            document=prior_document.document,
            checkpoint=prior_checkpoint.document,
            evidence_ref=prior_document.ref,
            checkpoint_ref=prior_checkpoint.ref,
            chain_anchor_ref=prior_document.document["chain_anchor_ref"],
            finalized_ordinal=int(prior_checkpoint.document["finalized_evidence_ordinal"]),
            effective_session=prior_session,
            state_probabilities=dict(prior_document.document["state_probabilities"]),
            chain_id=str(prior_checkpoint.document["chain_id"]),
            global_accumulator=str(prior_checkpoint.document["global_accumulator"]),
            segment_id=str(prior_checkpoint.document["segment_id"]),
            segment_index=int(prior_checkpoint.document["segment_index"]),
            segment_position=int(prior_checkpoint.document["segment_position"]),
            segment_accumulator=str(prior_checkpoint.document["segment_accumulator"]),
        )
        if _prior_finality(prior_state) != prior_finality:
            raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "prior finality scalar mismatch")
        prior_effective = prior_session
        prior_probabilities = dict(prior_document.document["state_probabilities"])
        prior_global = str(prior_checkpoint.document["global_accumulator"])
        prior_segment_accumulator = str(prior_checkpoint.document["segment_accumulator"])
        expected_phase = (
            "RECOVERY"
            if evidence["missing_sessions"]
            else (
                "ROLLOVER"
                if int(prior_checkpoint.document["segment_position"]) == SEGMENT_LENGTH - 1
                else "CONTIGUOUS"
            )
        )
        if expected_phase in {"RECOVERY", "ROLLOVER"}:
            expected_segment_index = int(prior_checkpoint.document["segment_index"]) + 1
            expected_segment_position = 0
        else:
            expected_segment_index = int(prior_checkpoint.document["segment_index"])
            expected_segment_position = int(prior_checkpoint.document["segment_position"]) + 1

    expected_missing = _missing_sessions(
        open_sessions=open_sessions,
        after_session=prior_effective,
        before_session=str(evidence["effective_session"]),
    )
    if (
        expected_missing != evidence["missing_sessions"]
        or expected_phase != evidence["phase"]
        or expected_segment_index != evidence["segment_index"]
        or expected_segment_position != evidence["segment_position"]
    ):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "phase or segment transition mismatch")
    expected_probabilities = v2._posterior(
        previous=_propagate_missing(
            start_probabilities=prior_probabilities,
            missing_sessions=expected_missing,
            transition_matrix=transition.document["transition_matrix"],
        ),
        transition_matrix=transition.document["transition_matrix"],
        likelihoods=feature.document["state_likelihoods"],
    )
    if expected_probabilities != evidence["state_probabilities"]:
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "filtered posterior replay mismatch")

    record = _checkpoint_record(
        strategy_id=strategy,
        effective_session=str(evidence["effective_session"]),
        observed_session=str(evidence["observed_through_session"]),
        ordinal=ordinal,
        phase=str(evidence["phase"]),
        probabilities=expected_probabilities,
        calendar_binding=evidence["calendar_binding"],
        calendar_prefix=evidence["calendar_prefix"],
        chain_id=str(evidence["chain_id"]),
        chain_anchor_ref=evidence["chain_anchor_ref"],
        segment_id=str(evidence["segment_id"]),
        segment_index=int(evidence["segment_index"]),
        segment_position=int(evidence["segment_position"]),
        feature_ref=feature.ref,
        model_ref=model.ref,
        transition_ref=transition.ref,
        policy_ref=policy.ref,
        missing_sessions=expected_missing,
        prior_finality=prior_finality,
    )
    expected_record = record_commitment_v1(record)
    expected_global = global_accumulator_v1(
        prior_accumulator=prior_global,
        record_commitment=expected_record,
        finalized_ordinal=ordinal,
        chain_id=str(evidence["chain_id"]),
    )
    if expected_segment_position == 0:
        segment_prior = segment_seed_v1(
            chain_id=str(evidence["chain_id"]),
            policy_ref=policy.ref,
            previous_global_accumulator=(None if ordinal == 0 else prior_global),
            segment_id=str(evidence["segment_id"]),
            segment_index=expected_segment_index,
            segment_start_session=str(evidence["effective_session"]),
            start_phase=str(evidence["phase"]),
            strategy_id=strategy,
        )
    else:
        if prior_segment_accumulator is None:
            raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "missing prior segment accumulator")
        segment_prior = prior_segment_accumulator
    expected_segment = segment_accumulator_v1(
        previous_accumulator=segment_prior,
        record_commitment=expected_record,
        segment_id=str(evidence["segment_id"]),
        segment_position=expected_segment_position,
    )
    if (
        expected_record != evidence["record_commitment"]
        or expected_global != evidence["global_accumulator"]
        or expected_segment != evidence["segment_accumulator"]
    ):
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "accumulator replay mismatch")


def _publication_clock(
    *,
    created_at: str,
    cutoff: str,
    observed_session: str,
    decision_session: str,
    now: datetime,
) -> None:
    _, created = _timestamp(created_at, label="created_at")
    _, decision_cutoff = _timestamp(cutoff, label="cutoff")
    if now.tzinfo is None:
        raise RegimeEvidenceV3Error(TEMPORAL_BLOCKER, "publisher clock must be timezone-aware")
    observed_close = datetime.combine(
        date.fromisoformat(observed_session),
        time(hour=15),
        tzinfo=_SHANGHAI,
    ).astimezone(_UTC)
    normalized_now = now.astimezone(_UTC).replace(microsecond=0)
    cutoff_session = decision_cutoff.astimezone(_SHANGHAI).date().isoformat()
    if (
        abs((normalized_now - created).total_seconds()) > NEW_PUBLICATION_TOLERANCE_SECONDS
        or created < observed_close
        or created > decision_cutoff
        or cutoff_session != decision_session
    ):
        raise RegimeEvidenceV3Error(
            TEMPORAL_BLOCKER,
            "new publication clock is stale, future, pre-close, post-cutoff, or outside D",
        )


def _validate_retry_arguments(
    *,
    document: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    prior_finality = document["prior_finality"]
    observed = {
        "chain_anchor_path": document["chain_anchor_ref"]["relative_path"],
        "chain_anchor_sha256": document["chain_anchor_ref"]["byte_sha256"],
        "created_at": document["created_at"],
        "cutoff": document["cutoff"],
        "decision_session": document["calendar_binding"]["decision_session"],
        "evidence_id": document["evidence_id"],
        "feature_snapshot_path": document["feature_snapshot_ref"]["relative_path"],
        "feature_snapshot_sha256": document["feature_snapshot_ref"]["byte_sha256"],
        "inference_policy_path": document["inference_policy_ref"]["relative_path"],
        "inference_policy_sha256": document["inference_policy_ref"]["byte_sha256"],
        "model_snapshot_path": document["model_snapshot_ref"]["relative_path"],
        "model_snapshot_sha256": document["model_snapshot_ref"]["byte_sha256"],
        "prior_checkpoint_path": None,
        "prior_checkpoint_sha256": None,
        "prior_evidence_path": None,
        "prior_evidence_sha256": None,
        "strategy_id": document["strategy_id"],
        "transition_matrix_path": document["transition_matrix_ref"]["relative_path"],
        "transition_matrix_sha256": document["transition_matrix_ref"]["byte_sha256"],
    }
    if prior_finality["prior_checkpoint_id"] is not None:
        prior_session = str(prior_finality["prior_effective_session"])
        observed["prior_checkpoint_path"] = state_checkpoint_v1_path(
            strategy_id=str(document["strategy_id"]),
            effective_session=prior_session,
            checkpoint_id=str(prior_finality["prior_checkpoint_id"]),
        )
        observed["prior_checkpoint_sha256"] = prior_finality["prior_checkpoint_byte_sha256"]
        observed["prior_evidence_path"] = regime_evidence_v3_path(
            strategy_id=str(document["strategy_id"]),
            effective_session=prior_session,
        )
        observed["prior_evidence_sha256"] = prior_finality["prior_evidence_byte_sha256"]
    normalized_expected = dict(expected)
    if (
        normalized_expected.get("chain_anchor_path") is None
        and normalized_expected.get("chain_anchor_sha256") is None
        and normalized_expected.get("prior_evidence_path") is None
        and normalized_expected.get("prior_checkpoint_path") is None
    ):
        normalized_expected["chain_anchor_path"] = observed["chain_anchor_path"]
        normalized_expected["chain_anchor_sha256"] = observed["chain_anchor_sha256"]
    if observed != normalized_expected:
        raise RegimeEvidenceV3Conflict("fixed slot retry arguments differ from sealed evidence")


def _read_path_sha(
    *,
    store: SourceStore,
    path: str,
    sha256: str,
    expected_version: str,
    label: str,
    strategy_id: str,
    schema_optional: bool,
) -> _Loaded:
    expected = _sha(sha256, label=f"{label}_sha256")
    try:
        raw = store.read(path, expected)
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV3InputGap(f"missing referenced {label}") from exc
    except (SourceCASMismatch, SourceStorageSecurityError) as exc:
        raise RegimeEvidenceV3Error(INPUT_TAMPER_BLOCKER, f"{label} exact read failed") from exc
    document = _load_canonical(
        raw,
        expected_version=expected_version,
        label=label,
        schema_optional=schema_optional,
    )
    if document.get("strategy_id") != strategy_id:
        raise RegimeEvidenceV3Error(IDENTITY_BLOCKER, f"{label} strategy mismatch")
    return _Loaded(document, raw, _artifact_ref(document, raw, relative_path=path))


def _seal_identity(
    document: Mapping[str, Any],
    *,
    identity_field: str,
) -> dict[str, Any]:
    body = {str(key): value for key, value in document.items() if key != "semantic_sha256"}
    body[identity_field] = "0" * 64
    body[identity_field] = regime_artifact_identity(
        body,
        identity_field=identity_field,
    )
    return seal_semantic(body)


def _sorted_refs(refs: Sequence[Mapping[str, str] | None]) -> list[dict[str, str]]:
    normalized = [_normalize_ref(ref, label="source_ref") for ref in refs if ref is not None]
    ordered = sorted(
        normalized,
        key=lambda row: (
            row["relative_path"],
            row["byte_sha256"],
            row["artifact_id"],
        ),
    )
    if len({canonical_bytes(row) for row in ordered}) != len(ordered):
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, "duplicate direct source reference")
    return ordered


def _prepublication_validate(
    *,
    anchor: Mapping[str, Any] | None,
    segment: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    candidates = [
        (anchor, CHAIN_ANCHOR_VERSION, "chain_anchor"),
        (segment, SEGMENT_ANCHOR_VERSION, "segment_anchor"),
        (checkpoint, STATE_CHECKPOINT_VERSION, "state_checkpoint"),
        (evidence, REGIME_EVIDENCE_V3_VERSION, "regime_evidence_v3"),
    ]
    for document, version, label in candidates:
        if document is None:
            continue
        _load_canonical(
            canonical_resource_bytes(document),
            expected_version=version,
            label=label,
            schema_optional=False,
        )


def _artifact_ref(document: Mapping[str, Any], raw: bytes, *, relative_path: str) -> dict[str, str]:
    version = str(document.get("version") or "")
    try:
        identity_field = artifact_identity_field(version)
    except SchemaValidationError:
        candidates = [
            key for key in document if key.endswith("_id") and type(document.get(key)) is str
        ]
        identity_field = candidates[0] if len(candidates) == 1 else ""
    if not identity_field:
        raise RegimeEvidenceV3Error(IDENTITY_BLOCKER, "cannot infer artifact identity field")
    return {
        "artifact_id": _opaque(document.get(identity_field), label=identity_field),
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document.get("cutoff") or ""),
        "relative_path": relative_path,
        "semantic_sha256": _sha(document.get("semantic_sha256"), label="semantic_sha256"),
        "strategy_id": _strategy_id(document.get("strategy_id")),
    }


def _normalize_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        raise RegimeEvidenceV3Error(SEMANTIC_BLOCKER, f"{label} artifact reference shape mismatch")
    ref = {key: str(value[key]) for key in sorted(_REF_FIELDS)}
    _opaque(ref["artifact_id"], label=f"{label}.artifact_id")
    _opaque(ref["artifact_version"], label=f"{label}.artifact_version")
    _sha(ref["byte_sha256"], label=f"{label}.byte_sha256")
    _sha(ref["semantic_sha256"], label=f"{label}.semantic_sha256")
    _timestamp(ref["cutoff"], label=f"{label}.cutoff")
    _strategy_id(ref["strategy_id"])
    return ref


def _strategy_id(value: Any) -> str:
    try:
        return v2._strategy_id(value)
    except v2.RegimeEvidenceV2Error as exc:
        raise _translate_v2(exc) from exc


def _session(value: Any, *, label: str) -> str:
    try:
        return v2._session(value, label=label)
    except v2.RegimeEvidenceV2Error as exc:
        raise _translate_v2(exc) from exc


def _timestamp(value: Any, *, label: str) -> tuple[str, datetime]:
    try:
        return v2._timestamp(value, label=label)
    except v2.RegimeEvidenceV2Error as exc:
        raise _translate_v2(exc) from exc


def _opaque(value: Any, *, label: str) -> str:
    try:
        return require_opaque_id(value, label=label)
    except IdentityContractError as exc:
        raise RegimeEvidenceV3Error(IDENTITY_BLOCKER, f"{label} is not canonical") from exc


def _sha(value: Any, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except IdentityContractError as exc:
        raise RegimeEvidenceV3Error(IDENTITY_BLOCKER, f"{label} is not canonical SHA-256") from exc


def _ordinal(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise RegimeEvidenceV3Error(CHECKPOINT_BLOCKER, "finalized ordinal must be non-negative")
    return value


def _strip_self(document: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in document.items() if key not in _SELF_FIELDS}


def _sha_state(value: Any) -> str:
    return _sha(value, label="accumulator")


def _domain_hash(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes({"domain": domain, "payload": dict(payload)})).hexdigest()


def _translate_v2(exc: v2.RegimeEvidenceV2Error) -> RegimeEvidenceV3Error:
    if isinstance(exc, v2.RegimeEvidenceV2InputGap):
        return RegimeEvidenceV3InputGap(exc.detail)
    mapping = {
        v2.CONFLICT_BLOCKER: CONFLICT_BLOCKER,
        v2.IDENTITY_BLOCKER: IDENTITY_BLOCKER,
        v2.INPUT_TAMPER_BLOCKER: INPUT_TAMPER_BLOCKER,
        v2.SEMANTIC_BLOCKER: SEMANTIC_BLOCKER,
        v2.TEMPORAL_BLOCKER: TEMPORAL_BLOCKER,
    }
    return RegimeEvidenceV3Error(mapping.get(exc.blocker_code, INPUT_TAMPER_BLOCKER), exc.detail)


__all__ = [
    "AUTHORITY_STATUS",
    "CHAIN_ANCHOR_VERSION",
    "INFERENCE_POLICY_V2_PATH",
    "INFERENCE_POLICY_V2_VERSION",
    "MAX_AUDIT_RECORDS",
    "MAX_GAP_SESSIONS",
    "MODEL_SNAPSHOT_V2_VERSION",
    "REGIME_EVIDENCE_V3_VERSION",
    "RegimeEvidenceV3BuildResult",
    "RegimeEvidenceV3Conflict",
    "RegimeEvidenceV3Error",
    "RegimeEvidenceV3InputGap",
    "SEGMENT_ANCHOR_VERSION",
    "STATE_CHECKPOINT_VERSION",
    "TRANSITION_SNAPSHOT_V2_VERSION",
    "audit_regime_chain_v3",
    "calendar_prefix_commitment_v1",
    "chain_anchor_v1_path",
    "chain_checkpoint_commitment_v1",
    "checkpoint_v1_path",
    "global_accumulator_v1",
    "implementation_sha256",
    "record_commitment_v1",
    "regime_chain_capacity_probe_v3",
    "regime_evidence_v3_authority_attestation",
    "regime_evidence_v3_id",
    "regime_evidence_v3_path",
    "replay_regime_evidence_v3",
    "segment_anchor_commitment_v1",
    "segment_anchor_v1_path",
    "state_checkpoint_v1_path",
]
