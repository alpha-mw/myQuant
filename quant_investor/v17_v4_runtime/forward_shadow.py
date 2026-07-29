"""Atomic forward-only Shadow publication for Fusion v2 and Deep v3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v4_contract.identities import (
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)

from .deep_v3 import (
    DEEP_BUNDLE_V3,
    FUSION_TOP24_V2,
    revalidate_deep_v3_bundle,
)
from .source_storage import (
    SHADOW_ROOT,
    ExactReferenceReader,
    GovernedStore,
    SourceStorageSecurityError,
    canonical_governed_path,
)

READINESS_VERSION: Final = "myquant.v17.v4.shadow-readiness.v2"
SHADOW_RUN_VERSION: Final = "myquant.v17.v4.shadow-run.v3"
SESSION_REF_VERSION: Final = "myquant.v17.v4.shadow-session-ref.v3"
FACTOR_ASSERTION_VERSION: Final = "myquant.v17.v4.research-factor-shadow-assertion.v2"
FACTOR_SET_VERSION: Final = "myquant.v17.v4.research-shadow-factor-set.v1"
FACTOR_SET_POINTER_VERSION: Final = "myquant.v17.v4.research-shadow-factor-set-pointer.v1"
INITIAL_POOL_VERSION: Final = "myquant.v17.v4.research-initial-pool-output.v2"
SOURCE_LOCATOR_VERSION: Final = "myquant.v17.v4.research-source-locator.v2"
RESEARCH_QUANT_VERSION: Final = "myquant.v17.v4.research-quant-branch-output.v2"
FUNDAMENTAL_BRANCH_VERSION: Final = "myquant.v17.v4.research-fundamental-branch-output.v2"
HOLDINGS_VERSION: Final = "myquant.v17.v4.holdings-snapshot.v1"
FUSION_OBSERVATION_VERSION: Final = "myquant.v17.v4.shadow-fusion-observation.v1"
FACTOR_EVIDENCE_MODE: Final = "DYNAMIC_RESEARCH_FACTOR_SET_SHADOW_ONLY"
FACTOR_ASSERTION_SCOPE: Final = "ONE_RUN_V17_V4_DYNAMIC_FACTOR_SET_SHADOW_ONLY"
FUSION_STATE: Final = "UNCALIBRATED_FORWARD_ACCUMULATING"
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
FORWARD_ONLY_VERSIONS: Final = frozenset(
    {
        DEEP_BUNDLE_V3,
        FACTOR_ASSERTION_VERSION,
        FUSION_OBSERVATION_VERSION,
        FUSION_TOP24_V2,
        READINESS_VERSION,
        SESSION_REF_VERSION,
        SHADOW_RUN_VERSION,
    }
)
_STRATEGY_PATH_RE: Final = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    re.ASCII,
)


class ForwardShadowError(RuntimeError):
    """Raised when a forward-only Shadow publication fails closed."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise ForwardShadowError(f"V17_V4_FORWARD_SHADOW_BLOCKED:{reason}")


def _strategy(value: Any) -> str:
    strategy = require_opaque_id(value, label="strategy_id")
    if _STRATEGY_PATH_RE.fullmatch(strategy) is None:
        _blocked("strategy_id_path")
    return strategy


def _session(value: Any) -> str:
    if type(value) is not str:
        _blocked("decision_session")
    try:
        normalized = date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ForwardShadowError("V17_V4_FORWARD_SHADOW_BLOCKED:decision_session") from exc
    if normalized != value:
        _blocked("decision_session")
    return normalized


def _artifact_ref(
    artifact: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    version = str(artifact.get("version") or "")
    try:
        identity_field = artifact_identity_field(version)
        identity = require_opaque_id(
            artifact.get(identity_field),
            label=identity_field,
        )
    except Exception as exc:
        raise ForwardShadowError("V17_V4_FORWARD_SHADOW_BLOCKED:artifact_ref_identity") from exc
    raw = canonical_resource_bytes(artifact)
    return {
        "artifact_id": identity,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(artifact.get("cutoff") or ""),
        "relative_path": str(canonical_governed_path(relative_path)),
        "semantic_sha256": str(artifact.get("semantic_sha256") or ""),
        "strategy_id": str(artifact.get("strategy_id") or ""),
    }


def _load_ref(
    reference: Mapping[str, Any],
    *,
    expected_version: str,
    reader: ExactReferenceReader,
) -> dict[str, Any]:
    try:
        raw = reader.read(
            str(reference["relative_path"]),
            require_sha256(
                reference["byte_sha256"],
                label="artifact_ref.byte_sha256",
            ),
        )
        loader = lambda child: reader.read(
            child["relative_path"],
            child["byte_sha256"],
        )
        artifact = load_canonical_artifact(
            raw,
            expected_version=expected_version,
            artifact_loader=loader,
        )
        document = load_canonical_resource(
            raw,
            label=expected_version,
        )
        identity_field = artifact_identity_field(expected_version)
    except Exception as exc:
        raise ForwardShadowError("V17_V4_FORWARD_SHADOW_BLOCKED:artifact_readback") from exc
    if (
        type(document) is not dict
        or artifact.payload != document
        or document.get(identity_field) != reference.get("artifact_id")
        or document.get("version") != reference.get("artifact_version")
        or document.get("strategy_id") != reference.get("strategy_id")
        or document.get("cutoff") != reference.get("cutoff")
        or document.get("semantic_sha256") != reference.get("semantic_sha256")
        or hashlib.sha256(raw).hexdigest() != reference.get("byte_sha256")
    ):
        _blocked("artifact_ref_mismatch")
    return dict(document)


def _forward_flags(document: Mapping[str, Any]) -> None:
    if (
        document.get("authority") != NO_AUTHORITY
        or document.get("shadow_only") is not True
        or document.get("formal_activation_eligible") is not False
        or document.get("canary_evidence_eligible") is not False
        or document.get("performance_evidence_eligible") is not False
    ):
        _blocked("authority_or_eligibility")
    for field in (
        "formal_research_publication_eligible",
        "policy_promotion_eligible",
        "promotion_eligible",
    ):
        if field in document and document[field] is not False:
            _blocked("authority_or_eligibility")


def _factor_names(factor_set: Mapping[str, Any]) -> list[str]:
    rows = factor_set.get("selected_factors")
    if type(rows) is not list or not rows:
        _blocked("factor_set_names")
    names: list[str] = []
    for row in rows:
        if type(row) is str:
            name = row
        elif type(row) is dict:
            name = row.get("factor_name") or row.get("name")
        else:
            name = None
        names.append(require_opaque_id(name, label="factor_name"))
    if len(names) != len(set(names)) or len(names) > 8:
        _blocked("factor_set_names")
    return names


def _factor_policy_sha(factor_set: Mapping[str, Any]) -> str:
    for field in (
        "selection_policy_sha256",
        "factor_selection_policy_sha256",
        "policy_sha256",
    ):
        value = factor_set.get(field)
        if value is not None:
            return require_sha256(
                value,
                label="factor_selection_policy_sha256",
            )
    _blocked("factor_set_policy")


class _ForwardShadowWriter(GovernedStore):
    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        if parts[:3] != ("results", "v17_v4_shadow", "strategies") or len(parts) != 6:
            raise SourceStorageSecurityError("path is outside forward Shadow roots")
        try:
            _strategy(parts[3])
        except ForwardShadowError as exc:
            raise SourceStorageSecurityError("forward Shadow strategy path is invalid") from exc
        if parts[4] not in {
            "assertions",
            "readiness",
            "runs",
            "sessions",
        }:
            raise SourceStorageSecurityError("forward Shadow category is invalid")
        if not parts[5].endswith(".json"):
            raise SourceStorageSecurityError("forward Shadow filename is invalid")
        return path


def _shadow_path(
    strategy_id: str,
    category: str,
    identity: str,
) -> str:
    return str(
        SHADOW_ROOT
        / "strategies"
        / _strategy(strategy_id)
        / category
        / f"{require_opaque_id(identity, label='identity')}.json"
    )


def build_shadow_readiness_v2(
    workspace_root: str,
    *,
    readiness_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    created_at: str,
    blocker_codes: list[str],
    factor_refs_present: bool,
) -> dict[str, Any]:
    """Write blocker-only readiness without fabricating model output."""

    strategy = _strategy(strategy_id)
    cutoff_value = require_utc_timestamp(cutoff, label="cutoff")
    created = require_utc_timestamp(created_at, label="created_at")
    session = _session(decision_session)
    blockers = sorted(set(blocker_codes))
    if (
        not blockers
        or any(type(value) is not str for value in blockers)
        or created < cutoff_value
        or session != cutoff_value[:10]
        or type(factor_refs_present) is not bool
    ):
        _blocked("readiness_inputs")
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "blocker_codes": blockers,
            "canary_evidence_eligible": False,
            "created_at": created,
            "cutoff": cutoff_value,
            "decision_session": session,
            "factor_refs_present": factor_refs_present,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "model_output_present": False,
            "performance_evidence_eligible": False,
            "policy_promotion_eligible": False,
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "readiness_id": require_opaque_id(
                readiness_id,
                label="readiness_id",
            ),
            "shadow_only": True,
            "state": "FORWARD_SHADOW_BLOCKED",
            "strategy_id": strategy,
            "version": READINESS_VERSION,
        }
    )
    validate_artifact(document)
    path = _shadow_path(strategy, "readiness", readiness_id)
    writer = _ForwardShadowWriter(workspace_root)
    writer.initialize()
    result = writer.write_exact_once(
        path,
        canonical_resource_bytes(document),
    )
    return {
        "created": result.created,
        "model_output_present": False,
        "readiness_ref": _artifact_ref(
            document,
            relative_path=path,
        ),
        "state": "FORWARD_SHADOW_BLOCKED",
    }


def _load_forward_inputs(
    *,
    reader: ExactReferenceReader,
    source_locator_ref: Mapping[str, Any],
    initial_pool_ref: Mapping[str, Any],
    quant_branch_ref: Mapping[str, Any],
    fundamental_branch_ref: Mapping[str, Any],
    fusion_top24_ref: Mapping[str, Any],
    deep_bundle_ref: Mapping[str, Any],
    holdings_snapshot_ref: Mapping[str, Any],
    factor_set_pointer_ref: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    fusion_observation_ref: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    specifications = {
        "source": (source_locator_ref, SOURCE_LOCATOR_VERSION),
        "initial": (initial_pool_ref, INITIAL_POOL_VERSION),
        "quant": (quant_branch_ref, RESEARCH_QUANT_VERSION),
        "fundamental": (
            fundamental_branch_ref,
            FUNDAMENTAL_BRANCH_VERSION,
        ),
        "fusion": (fusion_top24_ref, FUSION_TOP24_V2),
        "deep": (deep_bundle_ref, DEEP_BUNDLE_V3),
        "holdings": (holdings_snapshot_ref, HOLDINGS_VERSION),
        "factor_pointer": (
            factor_set_pointer_ref,
            FACTOR_SET_POINTER_VERSION,
        ),
        "factor_set": (factor_set_ref, FACTOR_SET_VERSION),
        "observation": (
            fusion_observation_ref,
            FUSION_OBSERVATION_VERSION,
        ),
    }
    return {
        name: _load_ref(
            reference,
            expected_version=version,
            reader=reader,
        )
        for name, (reference, version) in specifications.items()
    }


def _validate_closure(
    *,
    loaded: Mapping[str, Mapping[str, Any]],
    refs: Mapping[str, Mapping[str, Any]],
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    reader: ExactReferenceReader,
) -> tuple[list[str], str]:
    for name in (
        "source",
        "initial",
        "quant",
        "fundamental",
        "fusion",
        "deep",
        "factor_pointer",
        "factor_set",
        "observation",
    ):
        _forward_flags(loaded[name])
    if any(document.get("strategy_id") != strategy_id for document in loaded.values()):
        _blocked("strategy_binding")
    for name in (
        "source",
        "initial",
        "quant",
        "fundamental",
        "fusion",
        "deep",
        "holdings",
        "observation",
    ):
        if loaded[name].get("cutoff") != cutoff:
            _blocked("cutoff_binding")
    if (
        loaded["fusion"].get("decision_session") != decision_session
        or loaded["observation"].get("decision_session") != decision_session
        or loaded["holdings"].get("as_of_session") > decision_session
        or loaded["initial"].get("research_source_locator_ref") != refs["source"]
        or loaded["initial"].get("factor_set_ref") != refs["factor_set"]
        or loaded["quant"].get("initial_pool_ref") != refs["initial"]
        or loaded["fundamental"].get("initial_pool_ref") != refs["initial"]
        or loaded["fundamental"].get("factor_set_ref") != refs["factor_set"]
        or loaded["factor_pointer"].get("factor_set_ref") != refs["factor_set"]
        or loaded["quant"].get("factor_set_ref") != refs["factor_set"]
        or loaded["initial"].get("input_bundle_ref") != loaded["quant"].get("input_bundle_ref")
        or loaded["initial"].get("input_bundle_ref")
        != loaded["fundamental"].get("input_bundle_ref")
        or loaded["observation"].get("fusion_top24_ref") != refs["fusion"]
        or loaded["deep"].get("fusion_top24_ref") != refs["fusion"]
        or loaded["fusion"].get("factor_set_byte_sha256") != refs["factor_set"]["byte_sha256"]
        or loaded["observation"].get("factor_set_byte_sha256") != refs["factor_set"]["byte_sha256"]
        or loaded["fusion"].get("input_bundle_sha256")
        != loaded["initial"].get("input_bundle_ref", {}).get("byte_sha256")
        or loaded["observation"].get("input_bundle_sha256")
        != loaded["fusion"].get("input_bundle_sha256")
        or loaded["fusion"].get("source_locator_semantic_sha256")
        != refs["source"]["semantic_sha256"]
        or loaded["observation"].get("source_locator_semantic_sha256")
        != refs["source"]["semantic_sha256"]
        or loaded["observation"].get("policy_semantic_sha256")
        != loaded["fusion"].get("policy_semantic_sha256")
        or loaded["fusion"].get("state") != FUSION_STATE
        or loaded["observation"].get("state") != FUSION_STATE
    ):
        _blocked("transitive_binding")
    loader = lambda reference: reader.read(
        reference["relative_path"],
        reference["byte_sha256"],
    )
    replayed_deep, replayed_fusion, _ = revalidate_deep_v3_bundle(
        refs["deep"],
        artifact_loader=loader,
    )
    if replayed_deep != loaded["deep"] or replayed_fusion != loaded["fusion"]:
        _blocked("deep_replay")
    names = _factor_names(loaded["factor_set"])
    policy_sha = _factor_policy_sha(loaded["factor_set"])
    initial_names = loaded["initial"].get("selected_factor_names")
    quant_names = loaded["quant"].get("selected_factor_names")
    if initial_names != names or quant_names != names:
        _blocked("quant_factor_names")
    return names, policy_sha


def _normalize_reread(
    value: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    if value is None or value is True:
        return None
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, Mapping) for item in value)
    ):
        return value[0], value[1]
    if isinstance(value, Mapping):
        pointer_ref = value.get(
            "factor_set_pointer_ref",
            value.get("pointer_ref"),
        )
        factor_ref = value.get("factor_set_ref")
        if isinstance(pointer_ref, Mapping) and isinstance(
            factor_ref,
            Mapping,
        ):
            return pointer_ref, factor_ref
    pointer_ref = getattr(value, "pointer_ref", None)
    factor_ref = getattr(value, "factor_set_ref", None)
    if isinstance(pointer_ref, Mapping) and isinstance(
        factor_ref,
        Mapping,
    ):
        return pointer_ref, factor_ref
    _blocked("factor_reread_result")


def publish_forward_shadow(
    workspace_root: str,
    *,
    shadow_run_id: str,
    override_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    created_at: str,
    source_locator_ref: Mapping[str, Any],
    initial_pool_ref: Mapping[str, Any],
    quant_branch_ref: Mapping[str, Any],
    fundamental_branch_ref: Mapping[str, Any],
    fusion_top24_ref: Mapping[str, Any],
    deep_bundle_ref: Mapping[str, Any],
    holdings_snapshot_ref: Mapping[str, Any],
    factor_set_pointer_ref: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    fusion_observation_ref: Mapping[str, Any],
    factor_state_reread: Callable[[], Any],
) -> dict[str, Any]:
    """Publish assertion/run, replay them, then commit the sole session ref."""

    strategy = _strategy(strategy_id)
    cutoff_value = require_utc_timestamp(cutoff, label="cutoff")
    created = require_utc_timestamp(created_at, label="created_at")
    session = _session(decision_session)
    run_id = require_opaque_id(shadow_run_id, label="shadow_run_id")
    assertion_id = require_opaque_id(override_id, label="override_id")
    if session != cutoff_value[:10] or created < cutoff_value or not callable(factor_state_reread):
        _blocked("publication_inputs")
    refs = {
        "source": dict(source_locator_ref),
        "initial": dict(initial_pool_ref),
        "quant": dict(quant_branch_ref),
        "fundamental": dict(fundamental_branch_ref),
        "fusion": dict(fusion_top24_ref),
        "deep": dict(deep_bundle_ref),
        "holdings": dict(holdings_snapshot_ref),
        "factor_pointer": dict(factor_set_pointer_ref),
        "factor_set": dict(factor_set_ref),
        "observation": dict(fusion_observation_ref),
    }
    reader = ExactReferenceReader(workspace_root)
    loaded = _load_forward_inputs(
        reader=reader,
        source_locator_ref=refs["source"],
        initial_pool_ref=refs["initial"],
        quant_branch_ref=refs["quant"],
        fundamental_branch_ref=refs["fundamental"],
        fusion_top24_ref=refs["fusion"],
        deep_bundle_ref=refs["deep"],
        holdings_snapshot_ref=refs["holdings"],
        factor_set_pointer_ref=refs["factor_pointer"],
        factor_set_ref=refs["factor_set"],
        fusion_observation_ref=refs["observation"],
    )
    names, policy_sha = _validate_closure(
        loaded=loaded,
        refs=refs,
        strategy_id=strategy,
        cutoff=cutoff_value,
        decision_session=session,
        reader=reader,
    )
    assertion = seal_semantic(
        {
            "assertion_scope": FACTOR_ASSERTION_SCOPE,
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": created,
            "cutoff": cutoff_value,
            "decision_session": session,
            "factor_evidence_mode": FACTOR_EVIDENCE_MODE,
            "factor_names": names,
            "factor_selection_policy_sha256": policy_sha,
            "factor_set_pointer_ref": refs["factor_pointer"],
            "factor_set_ref": refs["factor_set"],
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "operator_asserted": True,
            "override_id": assertion_id,
            "performance_evidence_eligible": False,
            "policy_promotion_eligible": False,
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "shadow_only": True,
            "shadow_run_id": run_id,
            "strategy_id": strategy,
            "version": FACTOR_ASSERTION_VERSION,
        }
    )
    validate_artifact(assertion)
    assertion_path = _shadow_path(
        strategy,
        "assertions",
        assertion_id,
    )
    assertion_ref = _artifact_ref(
        assertion,
        relative_path=assertion_path,
    )
    run = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": created,
            "cutoff": cutoff_value,
            "decision_session": session,
            "deep_bundle_ref": refs["deep"],
            "factor_evidence_mode": FACTOR_EVIDENCE_MODE,
            "factor_set_pointer_ref": refs["factor_pointer"],
            "factor_set_ref": refs["factor_set"],
            "formal_activation_eligible": False,
            "fundamental_branch_ref": refs["fundamental"],
            "fusion_observation_ref": refs["observation"],
            "fusion_state": FUSION_STATE,
            "fusion_top24_ref": refs["fusion"],
            "holdings_snapshot_ref": refs["holdings"],
            "initial_pool_ref": refs["initial"],
            "model_output_present": True,
            "performance_evidence_eligible": False,
            "policy_promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "quant_branch_ref": refs["quant"],
            "research_factor_shadow_assertion_ref": assertion_ref,
            "shadow_only": True,
            "shadow_run_id": run_id,
            "source_locator_ref": refs["source"],
            "state": "SHADOW_COMPLETE",
            "strategy_id": strategy,
            "version": SHADOW_RUN_VERSION,
        }
    )
    validate_artifact(run)
    run_path = _shadow_path(strategy, "runs", run_id)
    run_ref = _artifact_ref(run, relative_path=run_path)
    writer = _ForwardShadowWriter(workspace_root)
    writer.initialize()
    assertion_write = writer.write_exact_once(
        assertion_path,
        canonical_resource_bytes(assertion),
    )
    run_write = writer.write_exact_once(
        run_path,
        canonical_resource_bytes(run),
    )
    replayed = revalidate_forward_shadow_run(
        run_ref,
        reader=reader,
    )
    if replayed != run:
        _blocked("run_readback")
    session_ref_id = "shadow-session-" + session.replace("-", "")
    session_document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": created,
            "cutoff": cutoff_value,
            "decision_session": session,
            "factor_evidence_mode": FACTOR_EVIDENCE_MODE,
            "factor_set_pointer_ref": refs["factor_pointer"],
            "factor_set_ref": refs["factor_set"],
            "formal_activation_eligible": False,
            "performance_evidence_eligible": False,
            "policy_promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "research_factor_shadow_assertion_ref": assertion_ref,
            "session_ref_id": session_ref_id,
            "shadow_only": True,
            "shadow_run_ref": run_ref,
            "state": "SHADOW_COMPLETE",
            "strategy_id": strategy,
            "version": SESSION_REF_VERSION,
        }
    )
    validate_artifact(session_document)
    session_path = _shadow_path(
        strategy,
        "sessions",
        session,
    )
    reread = _normalize_reread(factor_state_reread())
    if reread is not None and (
        dict(reread[0]) != refs["factor_pointer"] or dict(reread[1]) != refs["factor_set"]
    ):
        _blocked("factor_pointer_drift")
    session_write = writer.write_exact_once(
        session_path,
        canonical_resource_bytes(session_document),
    )
    return {
        "assertion_created": assertion_write.created,
        "model_output_present": True,
        "run_created": run_write.created,
        "session_created": session_write.created,
        "session_ref": _artifact_ref(
            session_document,
            relative_path=session_path,
        ),
        "shadow_run_ref": run_ref,
        "state": "SHADOW_COMPLETE",
    }


def revalidate_forward_shadow_run(
    run_ref: Mapping[str, Any],
    *,
    reader: ExactReferenceReader,
) -> dict[str, Any]:
    """Reopen one run and replay its complete forward-only closure."""

    run = _load_ref(
        run_ref,
        expected_version=SHADOW_RUN_VERSION,
        reader=reader,
    )
    _forward_flags(run)
    assertion = _load_ref(
        run["research_factor_shadow_assertion_ref"],
        expected_version=FACTOR_ASSERTION_VERSION,
        reader=reader,
    )
    _forward_flags(assertion)
    refs = {
        "source": run["source_locator_ref"],
        "initial": run["initial_pool_ref"],
        "quant": run["quant_branch_ref"],
        "fundamental": run["fundamental_branch_ref"],
        "fusion": run["fusion_top24_ref"],
        "deep": run["deep_bundle_ref"],
        "holdings": run["holdings_snapshot_ref"],
        "factor_pointer": run["factor_set_pointer_ref"],
        "factor_set": run["factor_set_ref"],
        "observation": run["fusion_observation_ref"],
    }
    loaded = _load_forward_inputs(
        reader=reader,
        source_locator_ref=refs["source"],
        initial_pool_ref=refs["initial"],
        quant_branch_ref=refs["quant"],
        fundamental_branch_ref=refs["fundamental"],
        fusion_top24_ref=refs["fusion"],
        deep_bundle_ref=refs["deep"],
        holdings_snapshot_ref=refs["holdings"],
        factor_set_pointer_ref=refs["factor_pointer"],
        factor_set_ref=refs["factor_set"],
        fusion_observation_ref=refs["observation"],
    )
    names, policy_sha = _validate_closure(
        loaded=loaded,
        refs=refs,
        strategy_id=run["strategy_id"],
        cutoff=run["cutoff"],
        decision_session=run["decision_session"],
        reader=reader,
    )
    if (
        assertion["factor_set_pointer_ref"] != refs["factor_pointer"]
        or assertion["factor_set_ref"] != refs["factor_set"]
        or assertion["factor_names"] != names
        or assertion["factor_selection_policy_sha256"] != policy_sha
        or assertion["shadow_run_id"] != run["shadow_run_id"]
        or assertion["decision_session"] != run["decision_session"]
    ):
        _blocked("assertion_binding")
    return run


def read_forward_shadow_session(
    workspace_root: str,
    *,
    session_ref_path: str,
    expected_session_ref_sha256: str,
) -> dict[str, Any]:
    """Resolve the sole discovery artifact and replay its run."""

    reader = ExactReferenceReader(workspace_root)
    expected = require_sha256(
        expected_session_ref_sha256,
        label="expected_session_ref_sha256",
    )
    try:
        raw = reader.read(session_ref_path, expected)
        session_document = load_canonical_resource(
            raw,
            label=SESSION_REF_VERSION,
        )
    except Exception as exc:
        raise ForwardShadowError("V17_V4_FORWARD_SHADOW_BLOCKED:session_readback") from exc
    if type(session_document) is not dict:
        _blocked("session_root")
    session_ref = _artifact_ref(
        session_document,
        relative_path=session_ref_path,
    )
    if session_ref["byte_sha256"] != expected:
        _blocked("session_sha")
    session = _load_ref(
        session_ref,
        expected_version=SESSION_REF_VERSION,
        reader=reader,
    )
    run = revalidate_forward_shadow_run(
        session["shadow_run_ref"],
        reader=reader,
    )
    if (
        session["strategy_id"] != run["strategy_id"]
        or session["cutoff"] != run["cutoff"]
        or session["decision_session"] != run["decision_session"]
        or session["factor_set_pointer_ref"] != run["factor_set_pointer_ref"]
        or session["factor_set_ref"] != run["factor_set_ref"]
        or session["research_factor_shadow_assertion_ref"]
        != run["research_factor_shadow_assertion_ref"]
    ):
        _blocked("session_binding")
    return {
        "run": run,
        "session": session,
        "state": "SHADOW_COMPLETE",
    }


def reject_forward_artifact_for_formal(
    artifact_or_ref: Mapping[str, Any],
) -> None:
    """Fail closed if a forward-only artifact is offered to formal code."""

    version = artifact_or_ref.get(
        "artifact_version",
        artifact_or_ref.get("version"),
    )
    if version in FORWARD_ONLY_VERSIONS:
        _blocked("formal_artifact_rejected")


def forward_artifact_formal_eligible(
    artifact_or_ref: Mapping[str, Any],
) -> bool:
    """Return false for the entire new artifact family."""

    version = artifact_or_ref.get(
        "artifact_version",
        artifact_or_ref.get("version"),
    )
    if version not in FORWARD_ONLY_VERSIONS:
        _blocked("unknown_forward_artifact")
    return False


__all__ = [
    "FACTOR_ASSERTION_VERSION",
    "FORWARD_ONLY_VERSIONS",
    "READINESS_VERSION",
    "SESSION_REF_VERSION",
    "SHADOW_RUN_VERSION",
    "ForwardShadowError",
    "build_shadow_readiness_v2",
    "forward_artifact_formal_eligible",
    "publish_forward_shadow",
    "read_forward_shadow_session",
    "reject_forward_artifact_for_formal",
    "revalidate_forward_shadow_run",
]
