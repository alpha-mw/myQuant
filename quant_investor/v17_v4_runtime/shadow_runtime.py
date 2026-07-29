"""Factor-gated, shadow-only V17 v4 publication and replay."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Final, NoReturn

from quant_investor.factors.production_control_v1 import (
    ACTIVE_SET_SCHEMA_VERSION,
    CONTROL_RECEIPT_SCHEMA_VERSION,
    canonical_file_bytes as factor_file_bytes,
    validate_active_set_pointer,
    validate_authorization_receipt,
    validate_control_receipt,
    validate_pre_activation_eligibility,
    validate_production_control_transaction,
    validate_production_registry,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
)
from quant_investor.v17_v4_contract.identities import (
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)

from .deep_v2 import (
    DEEP_BUNDLE_V2,
    revalidate_deep_v2_bundle,
)
from .research_quant import (
    RESEARCH_FACTOR_NAMES,
    RESEARCH_FACTOR_POLICY_SHA256,
    RESEARCH_QUANT_BRANCH_VERSION,
    revalidate_research_quant_branch,
)
from .source_storage import (
    SHADOW_ROOT,
    ExactReferenceReader,
    GovernedStore,
    SourceStorageError,
    SourceStorageSecurityError,
    canonical_governed_path,
)

READINESS_VERSION: Final = "myquant.v17.v4.shadow-readiness.v1"
SHADOW_RUN_VERSION: Final = "myquant.v17.v4.shadow-run.v1"
SHADOW_RUN_RESEARCH_VERSION: Final = "myquant.v17.v4.shadow-run.v2"
SESSION_REF_VERSION: Final = "myquant.v17.v4.shadow-session-ref.v1"
SESSION_REF_RESEARCH_VERSION: Final = "myquant.v17.v4.shadow-session-ref.v2"
RESEARCH_FACTOR_SHADOW_ASSERTION_VERSION: Final = (
    "myquant.v17.v4.research-factor-shadow-assertion.v1"
)
INITIAL_POOL_VERSION: Final = "myquant.v17.v4.initial-pool-output.v1"
BRANCH_VERSION: Final = "myquant.v17.v4.branch-output.v1"
FUSION_VERSION: Final = "myquant.v17.v4.fusion-top24.v1"
PROMOTION_VERSION: Final = "myquant.v17.v4.fusion-promotion-receipt.v1"
LOCATOR_VERSION: Final = "myquant.v17.v4.preselect-locator.v1"
PIT_CATALOG_VERSION: Final = "myquant.v17.v4.pit-generation-catalog.v1"
HOLDINGS_VERSION: Final = "myquant.v17.v4.holdings-snapshot.v1"
RESEARCH_FACTOR_EVIDENCE_MODE: Final = "RESEARCH_TRIO_SHADOW_ONLY"
RESEARCH_FACTOR_ASSERTION_SCOPE: Final = (
    "ONE_RUN_V17_V4_SHADOW_RESEARCH_TRIO"
)
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
_STRATEGY_PATH_RE: Final = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    re.ASCII,
)


class ShadowRuntimeError(RuntimeError):
    """Raised when V17 v4 Shadow publication fails closed."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise ShadowRuntimeError(f"V17_V4_SHADOW_BLOCKED:{reason}")


def require_strategy_path_id(value: Any) -> str:
    """Return the single-component, lower-case strategy path identity."""

    strategy = require_opaque_id(value, label="strategy_id")
    if _STRATEGY_PATH_RE.fullmatch(strategy) is None:
        _blocked("strategy_id_path")
    return strategy


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) not in {str, int, float, Decimal} or type(value) is bool:
        _blocked(f"{label}_decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_decimal")
    if not result.is_finite():
        _blocked(f"{label}_nonfinite")
    return result


def _artifact_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    version = str(document.get("version") or "")
    try:
        identity_field = artifact_identity_field(version)
        identity = require_opaque_id(
            document.get(identity_field),
            label=identity_field,
        )
    except (TypeError, ValueError):
        _blocked("artifact_ref_identity")
    path = str(canonical_governed_path(relative_path))
    raw = canonical_resource_bytes(document)
    return {
        "artifact_id": identity,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document.get("cutoff") or ""),
        "relative_path": path,
        "semantic_sha256": str(document.get("semantic_sha256") or ""),
        "strategy_id": str(document.get("strategy_id") or ""),
    }


def _factor_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
    strategy_id: str,
    cutoff: str,
) -> dict[str, str]:
    identity = document.get("active_set_id") or document.get("receipt_id")
    if type(identity) is not str:
        _blocked("factor_identity")
    raw = factor_file_bytes(document)
    return {
        "artifact_id": require_opaque_id(
            identity,
            label="factor_artifact_id",
        ),
        "artifact_version": str(document["schema_version"]),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": cutoff,
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": strategy_id,
    }


class _ShadowWriter(GovernedStore):
    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        prefix = ("results", "v17_v4_shadow", "strategies")
        if parts[:3] != prefix or len(parts) != 6:
            raise SourceStorageSecurityError("path is outside V17 v4 shadow writer roots")
        try:
            require_strategy_path_id(parts[3])
        except ShadowRuntimeError as exc:
            raise SourceStorageSecurityError("shadow strategy path is invalid") from exc
        if parts[4] not in {"assertions", "readiness", "runs", "sessions"}:
            raise SourceStorageSecurityError("shadow writer category is invalid")
        if not parts[5].endswith(".json"):
            raise SourceStorageSecurityError("shadow writer filename is invalid")
        return path


def _shadow_path(
    strategy_id: str,
    category: str,
    identity: str,
) -> str:
    strategy = require_strategy_path_id(strategy_id)
    value = require_opaque_id(identity, label=f"{category}_id")
    return str(SHADOW_ROOT / "strategies" / strategy / category / f"{value}.json")


def _load_v4_ref(
    reference: Mapping[str, Any],
    *,
    expected_version: str,
    reader: ExactReferenceReader,
    transitive: bool = False,
) -> dict[str, Any]:
    try:
        raw = reader.read(
            str(reference["relative_path"]),
            str(reference["byte_sha256"]),
        )
        loader = lambda child: reader.read(
            child["relative_path"],
            child["byte_sha256"],
        )
        validated = load_canonical_artifact(
            raw,
            expected_version=expected_version,
            artifact_loader=loader if transitive else None,
        )
        document = load_canonical_resource(
            raw,
            label=expected_version,
        )
        identity_field = artifact_identity_field(expected_version)
    except Exception as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:v4_artifact_readback") from exc
    if (
        type(document) is not dict
        or validated.payload != document
        or document.get(identity_field) != reference.get("artifact_id")
        or document.get("semantic_sha256") != reference.get("semantic_sha256")
        or document.get("strategy_id") != reference.get("strategy_id")
        or document.get("cutoff") != reference.get("cutoff")
        or hashlib.sha256(raw).hexdigest() != reference.get("byte_sha256")
    ):
        _blocked("v4_artifact_reference")
    return dict(document)


def _load_v4_path(
    *,
    path: str,
    sha256: str,
    expected_version: str,
    reader: ExactReferenceReader,
    transitive: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = require_sha256(sha256, label=f"{expected_version}.sha256")
    try:
        raw = reader.read(path, expected)
        document = load_canonical_resource(raw, label=expected_version)
    except Exception as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:v4_input_readback") from exc
    if type(document) is not dict:
        _blocked("v4_input_root")
    reference = _artifact_ref(document, relative_path=path)
    if reference["byte_sha256"] != expected:
        _blocked("v4_input_sha")
    return (
        _load_v4_ref(
            reference,
            expected_version=expected_version,
            reader=reader,
            transitive=transitive,
        ),
        reference,
    )


def _read_factor_common(
    *,
    reader: ExactReferenceReader,
    path: str,
    sha256: str,
    expected_schema: str,
    strategy_id: str,
    cutoff: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = require_sha256(sha256, label=f"{expected_schema}.sha256")
    try:
        raw = reader.read(path, expected)
        document = json.loads(raw)
    except Exception as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:factor_common_readback") from exc
    if (
        type(document) is not dict
        or factor_file_bytes(document) != raw
        or document.get("schema_version") != expected_schema
    ):
        _blocked("factor_common_binding")
    reference = _factor_ref(
        document,
        relative_path=path,
        strategy_id=strategy_id,
        cutoff=cutoff,
    )
    if reference["byte_sha256"] != expected:
        _blocked("factor_common_sha")
    return document, reference


def _read_factor_native(
    reference: Mapping[str, Any],
    *,
    reader: ExactReferenceReader,
) -> dict[str, Any]:
    try:
        raw = reader.read(
            str(reference["relative_path"]),
            str(reference["byte_sha256"]),
        )
        document = json.loads(raw)
    except Exception as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:factor_native_readback") from exc
    if (
        type(document) is not dict
        or factor_file_bytes(document) != raw
        or document.get("semantic_sha256") != reference.get("semantic_sha256")
        or document.get("schema_version") != reference.get("artifact_schema")
    ):
        _blocked("factor_native_binding")
    return document


def revalidate_factor_v4_closure(
    *,
    reader: ExactReferenceReader,
    active_set: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
    active_set_ref: Mapping[str, Any],
    control_receipt_ref: Mapping[str, Any],
    cutoff: str,
    decision_session: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the formal Factor v4 production-control transaction."""

    active = validate_active_set_pointer(active_set)
    receipt = dict(control_receipt)
    transaction = _read_factor_native(
        receipt["transaction_ref"],
        reader=reader,
    )
    registry = _read_factor_native(
        transaction["proposed_registry_ref"],
        reader=reader,
    )
    eligibility = _read_factor_native(
        transaction["v4_pre_activation_eligibility_ref"],
        reader=reader,
    )
    authorization = _read_factor_native(
        transaction["production_control_authorization_receipt_ref"],
        reader=reader,
    )
    validate_production_registry(registry)
    validate_pre_activation_eligibility(
        eligibility,
        registry=registry,
    )
    validate_authorization_receipt(
        authorization,
        observed_at=transaction["activated_at"],
    )
    normalized = validate_production_control_transaction(
        transaction,
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
    )
    validated_receipt = validate_control_receipt(
        receipt,
        transaction=normalized,
    )
    expected_native_active_ref = {
        "artifact_schema": active_set_ref["artifact_version"],
        "byte_sha256": active_set_ref["byte_sha256"],
        "relative_path": active_set_ref["relative_path"],
        "schema_version": ("factor-governance-production-control.artifact-ref.v1"),
        "semantic_sha256": active_set_ref["semantic_sha256"],
    }
    if (
        normalized["proposed_active_set"] != active
        or validated_receipt["active_set_ref"] != expected_native_active_ref
        or validated_receipt["active_set_readback_sha256"] != active_set_ref["byte_sha256"]
        or validated_receipt["transaction_id"] != active["transaction_id"]
        or active["activated_at"] > cutoff
        or active["as_of"] > decision_session
        or control_receipt_ref["artifact_id"] != validated_receipt["receipt_id"]
    ):
        _blocked("factor_closure_mismatch")
    return dict(active), dict(validated_receipt)


def _percentiles(
    scores: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    count = Decimal(len(scores))
    ordered = sorted(scores.values())
    return {
        symbol: Decimal(sum(candidate <= score for candidate in ordered)) / count
        for symbol, score in scores.items()
    }


def _replay_fusion(
    *,
    initial_pool: Mapping[str, Any],
    quant_branch: Mapping[str, Any],
    fundamental_branch: Mapping[str, Any],
    fusion: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> None:
    pool = list(initial_pool["ordered_pool"])
    if (
        quant_branch["branch_kind"] != "QUANT"
        or fundamental_branch["branch_kind"] != "FUNDAMENTAL"
        or [row["symbol"] for row in quant_branch["score_rows"]] != pool
        or [row["symbol"] for row in fundamental_branch["score_rows"]] != pool
        or promotion["accepted"] is not True
        or promotion["status"] != "PROMOTED"
    ):
        _blocked("same_pool_or_promotion")
    quant_scores = {
        row["symbol"]: _decimal(
            row["score"],
            label=f"quant.{row['symbol']}",
        )
        for row in quant_branch["score_rows"]
    }
    fundamental_scores = {
        row["symbol"]: _decimal(
            row["score"],
            label=f"fundamental.{row['symbol']}",
        )
        for row in fundamental_branch["score_rows"]
    }
    quant_percentiles = _percentiles(quant_scores)
    fundamental_percentiles = _percentiles(fundamental_scores)
    weight = _decimal(
        promotion["active_quant_weight"],
        label="active_quant_weight",
    )
    with localcontext() as context:
        context.prec = 40
        fused = {
            symbol: (
                weight * quant_percentiles[symbol]
                + (Decimal("1") - weight) * fundamental_percentiles[symbol]
            )
            for symbol in pool
        }
    expected_symbols = sorted(
        pool,
        key=lambda symbol: (-fused[symbol], symbol),
    )[:24]
    observed_symbols = [row["symbol"] for row in fusion["rows"]]
    if observed_symbols != expected_symbols:
        _blocked("fusion_top24_order")
    for row in fusion["rows"]:
        if (
            _decimal(
                row["fused_score"],
                label=f"fusion.{row['symbol']}",
            )
            != fused[row["symbol"]]
        ):
            _blocked("fusion_score")


def _write_readiness(
    *,
    workspace_root: str,
    readiness_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    created_at: str,
    blockers: list[str],
    factor_refs_present: bool,
) -> dict[str, Any]:
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "blocker_codes": sorted(set(blockers)),
            "canary_evidence_eligible": False,
            "created_at": created_at,
            "cutoff": cutoff,
            "decision_session": decision_session,
            "factor_refs_present": factor_refs_present,
            "formal_activation_eligible": False,
            "model_output_present": False,
            "protocol_version": PROTOCOL_VERSION,
            "readiness_id": require_opaque_id(
                readiness_id,
                label="readiness_id",
            ),
            "shadow_only": True,
            "state": "FACTOR_V4_BLOCKED",
            "strategy_id": strategy_id,
            "version": READINESS_VERSION,
        }
    )
    validate_artifact(document)
    path = _shadow_path(strategy_id, "readiness", readiness_id)
    raw = canonical_resource_bytes(document)
    writer = _ShadowWriter(workspace_root)
    writer.initialize()
    result = writer.write_exact_once(path, raw)
    return {
        "authority": dict(NO_AUTHORITY),
        "created": result.created,
        "model_output_present": False,
        "readiness_ref": _artifact_ref(
            document,
            relative_path=path,
        ),
        "state": "FACTOR_V4_BLOCKED",
    }


def publish_shadow_run(
    workspace_root: str,
    *,
    readiness_id: str,
    shadow_run_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    created_at: str,
    factor_active_set_path: str | None,
    factor_active_set_sha256: str | None,
    factor_control_receipt_path: str | None,
    factor_control_receipt_sha256: str | None,
    research_factor_shadow_only_override_id: str | None = None,
    source_locator_path: str | None = None,
    source_locator_sha256: str | None = None,
    initial_pool_path: str | None = None,
    initial_pool_sha256: str | None = None,
    quant_branch_path: str | None = None,
    quant_branch_sha256: str | None = None,
    fundamental_branch_path: str | None = None,
    fundamental_branch_sha256: str | None = None,
    fusion_top24_path: str | None = None,
    fusion_top24_sha256: str | None = None,
    deep_bundle_path: str | None = None,
    deep_bundle_sha256: str | None = None,
    holdings_snapshot_path: str | None = None,
    holdings_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish one immutable v4 Shadow session or blocker-only readiness."""

    strategy = require_strategy_path_id(strategy_id)
    cutoff_value = require_utc_timestamp(cutoff, label="cutoff")
    created = require_utc_timestamp(created_at, label="created_at")
    try:
        session = date.fromisoformat(decision_session).isoformat()
    except (TypeError, ValueError) as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:decision_session") from exc
    if session != decision_session or session != cutoff_value[:10]:
        _blocked("decision_session_cutoff")
    reader = ExactReferenceReader(workspace_root)
    factor_pairs = (
        factor_active_set_path,
        factor_active_set_sha256,
        factor_control_receipt_path,
        factor_control_receipt_sha256,
    )
    factor_refs_present = any(value is not None for value in factor_pairs)
    factor_refs_missing = any(value is None for value in factor_pairs)
    research_factor_shadow = research_factor_shadow_only_override_id is not None
    if research_factor_shadow:
        try:
            override_id = require_opaque_id(
                research_factor_shadow_only_override_id,
                label="research_factor_shadow_only_override_id",
            )
        except (TypeError, ValueError) as exc:
            raise ShadowRuntimeError(
                "V17_V4_SHADOW_BLOCKED:research_factor_shadow_override_id"
            ) from exc
        if factor_refs_present:
            _blocked("research_factor_shadow_mixed_formal_refs")
    else:
        override_id = None
    if not research_factor_shadow and factor_refs_missing:
        blockers = []
        if factor_active_set_path is None or factor_active_set_sha256 is None:
            blockers.append("factor_v4_active_set_missing")
        if factor_control_receipt_path is None or factor_control_receipt_sha256 is None:
            blockers.append("factor_v4_control_receipt_missing")
        return _write_readiness(
            workspace_root=workspace_root,
            readiness_id=readiness_id,
            strategy_id=strategy,
            cutoff=cutoff_value,
            decision_session=session,
            created_at=created,
            blockers=blockers,
            factor_refs_present=factor_refs_present,
        )
    production_factor_names: list[str] = []
    factor_set_sha256: str | None = None
    if research_factor_shadow:
        active_ref = receipt_ref = None
    else:
        try:
            active, active_ref = _read_factor_common(
                reader=reader,
                path=str(factor_active_set_path),
                sha256=str(factor_active_set_sha256),
                expected_schema=ACTIVE_SET_SCHEMA_VERSION,
                strategy_id=strategy,
                cutoff=cutoff_value,
            )
            receipt, receipt_ref = _read_factor_common(
                reader=reader,
                path=str(factor_control_receipt_path),
                sha256=str(factor_control_receipt_sha256),
                expected_schema=CONTROL_RECEIPT_SCHEMA_VERSION,
                strategy_id=strategy,
                cutoff=cutoff_value,
            )
            active, receipt = revalidate_factor_v4_closure(
                reader=reader,
                active_set=active,
                control_receipt=receipt,
                active_set_ref=active_ref,
                control_receipt_ref=receipt_ref,
                cutoff=cutoff_value,
                decision_session=session,
            )
        except Exception:
            return _write_readiness(
                workspace_root=workspace_root,
                readiness_id=readiness_id,
                strategy_id=strategy,
                cutoff=cutoff_value,
                decision_session=session,
                created_at=created,
                blockers=["factor_v4_closure_invalid"],
                factor_refs_present=True,
            )
        production_factor_names = list(active["production_factor_names"])
        factor_set_sha256 = active["production_factor_set_sha256"]
    input_pairs = {
        "source_locator": (
            source_locator_path,
            source_locator_sha256,
            LOCATOR_VERSION,
        ),
        "initial_pool": (
            initial_pool_path,
            initial_pool_sha256,
            INITIAL_POOL_VERSION,
        ),
        "quant_branch": (
            quant_branch_path,
            quant_branch_sha256,
            RESEARCH_QUANT_BRANCH_VERSION,
        ),
        "fundamental_branch": (
            fundamental_branch_path,
            fundamental_branch_sha256,
            BRANCH_VERSION,
        ),
        "fusion_top24": (
            fusion_top24_path,
            fusion_top24_sha256,
            FUSION_VERSION,
        ),
        "deep_bundle": (
            deep_bundle_path,
            deep_bundle_sha256,
            DEEP_BUNDLE_V2,
        ),
        "holdings_snapshot": (
            holdings_snapshot_path,
            holdings_snapshot_sha256,
            HOLDINGS_VERSION,
        ),
    }
    if any(path is None or digest is None for path, digest, _ in input_pairs.values()):
        _blocked("shadow_model_input_missing")
    loaded: dict[str, dict[str, Any]] = {}
    refs: dict[str, dict[str, str]] = {}
    for name, (path, digest, version) in input_pairs.items():
        loaded[name], refs[name] = _load_v4_path(
            path=str(path),
            sha256=str(digest),
            expected_version=version,
            reader=reader,
        )
    locator = loaded["source_locator"]
    initial = loaded["initial_pool"]
    quant = loaded["quant_branch"]
    fundamental = loaded["fundamental_branch"]
    fusion = loaded["fusion_top24"]
    deep = loaded["deep_bundle"]
    holdings = loaded["holdings_snapshot"]
    promotion = _load_v4_ref(
        fusion["promotion_receipt_ref"],
        expected_version=PROMOTION_VERSION,
        reader=reader,
        transitive=True,
    )
    pit_catalog = _load_v4_ref(
        locator["pit_catalog_ref"],
        expected_version=PIT_CATALOG_VERSION,
        reader=reader,
    )
    loader = lambda reference: reader.read(
        reference["relative_path"],
        reference["byte_sha256"],
    )
    replayed_deep, replayed_fusion, _ = revalidate_deep_v2_bundle(
        refs["deep_bundle"],
        artifact_loader=loader,
    )
    if (
        any(
            document["strategy_id"] != strategy or document["cutoff"] != cutoff_value
            for document in loaded.values()
        )
        or promotion["strategy_id"] != strategy
        or pit_catalog["strategy_id"] != strategy
    ):
        _blocked("shadow_strategy_cutoff")
    if (
        initial["preselect_locator_ref"] != refs["source_locator"]
        or quant["initial_pool_ref"] != refs["initial_pool"]
        or fundamental["initial_pool_ref"] != refs["initial_pool"]
        or fusion["promotion_receipt_ref"]
        != _artifact_ref(
            promotion,
            relative_path=fusion["promotion_receipt_ref"]["relative_path"],
        )
        or deep != replayed_deep
        or fusion != replayed_fusion
        or deep["fusion_top24_ref"] != refs["fusion_top24"]
        or fusion["run_id"] != shadow_run_id
        or deep["run_id"] != shadow_run_id
        or pit_catalog["decision_session"] != session
        or holdings["as_of_session"] > session
    ):
        _blocked("shadow_transitive_binding")
    _replay_fusion(
        initial_pool=initial,
        quant_branch=quant,
        fundamental_branch=fundamental,
        fusion=fusion,
        promotion=promotion,
    )
    revalidate_research_quant_branch(
        quant,
        initial_pool=initial,
        initial_pool_ref=refs["initial_pool"],
        reader=reader,
    )
    calendar_ref = dict(pit_catalog["dataset_refs"]["cn_open_day_calendar"])
    market_bars_ref = dict(pit_catalog["dataset_refs"]["market_bars"])
    try:
        reader.verify_sha256(
            calendar_ref["relative_path"],
            calendar_ref["byte_sha256"],
        )
        reader.verify_sha256(
            market_bars_ref["relative_path"],
            market_bars_ref["byte_sha256"],
        )
    except SourceStorageError as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:comparison_input_readback") from exc
    source_closure_ref = dict(locator["pit_catalog_ref"])
    canonical_shadow_run_id = require_opaque_id(
        shadow_run_id,
        label="shadow_run_id",
    )
    assertion: dict[str, Any] | None = None
    assertion_ref: dict[str, str] | None = None
    assertion_path: str | None = None
    if research_factor_shadow:
        assert override_id is not None
        assertion = seal_semantic(
            {
                "assertion_scope": RESEARCH_FACTOR_ASSERTION_SCOPE,
                "authority": dict(NO_AUTHORITY),
                "canary_evidence_eligible": False,
                "created_at": created,
                "cutoff": cutoff_value,
                "decision_session": session,
                "factor_evidence_mode": RESEARCH_FACTOR_EVIDENCE_MODE,
                "factor_names": list(RESEARCH_FACTOR_NAMES),
                "factor_policy_sha256": RESEARCH_FACTOR_POLICY_SHA256,
                "formal_activation_eligible": False,
                "operator_asserted": True,
                "override_id": override_id,
                "protocol_version": PROTOCOL_VERSION,
                "shadow_only": True,
                "shadow_run_id": canonical_shadow_run_id,
                "strategy_id": strategy,
                "version": RESEARCH_FACTOR_SHADOW_ASSERTION_VERSION,
            }
        )
        validate_artifact(assertion)
        assertion_path = _shadow_path(strategy, "assertions", override_id)
        assertion_ref = _artifact_ref(
            assertion,
            relative_path=assertion_path,
        )
    run_payload: dict[str, Any] = {
        "authority": dict(NO_AUTHORITY),
        "canary_evidence_eligible": False,
        "comparison_inputs": {
            "calendar_ref": calendar_ref,
            "holdings_ref": refs["holdings_snapshot"],
            "market_bars_ref": market_bars_ref,
            "source_closure_ref": source_closure_ref,
        },
        "created_at": created,
        "cutoff": cutoff_value,
        "decision_session": session,
        "deep_bundle_ref": refs["deep_bundle"],
        "formal_activation_eligible": False,
        "fundamental_branch_ref": refs["fundamental_branch"],
        "fusion_top24_ref": refs["fusion_top24"],
        "initial_pool_ref": refs["initial_pool"],
        "model_output_present": True,
        "protocol_version": PROTOCOL_VERSION,
        "quant_branch_ref": refs["quant_branch"],
        "research_quant_factor_names": list(RESEARCH_FACTOR_NAMES),
        "research_quant_factor_policy_sha256": (
            RESEARCH_FACTOR_POLICY_SHA256
        ),
        "shadow_only": True,
        "shadow_run_id": canonical_shadow_run_id,
        "source_locator_ref": refs["source_locator"],
        "state": "SHADOW_COMPLETE",
        "strategy_id": strategy,
    }
    if research_factor_shadow:
        assert assertion_ref is not None
        run_payload.update(
            {
                "factor_evidence_mode": RESEARCH_FACTOR_EVIDENCE_MODE,
                "research_factor_shadow_assertion_ref": assertion_ref,
                "version": SHADOW_RUN_RESEARCH_VERSION,
            }
        )
    else:
        run_payload.update(
            {
                "factor_control_active_set_ref": active_ref,
                "factor_control_receipt_ref": receipt_ref,
                "factor_set_sha256": factor_set_sha256,
                "production_factor_names": list(production_factor_names),
                "version": SHADOW_RUN_VERSION,
            }
        )
    run = seal_semantic(run_payload)
    validate_artifact(run)
    run_path = _shadow_path(strategy, "runs", shadow_run_id)
    run_ref = _artifact_ref(run, relative_path=run_path)
    session_ref_id = "shadow-session-" + session.replace("-", "")
    session_payload: dict[str, Any] = {
        "authority": dict(NO_AUTHORITY),
        "canary_evidence_eligible": False,
        "created_at": created,
        "cutoff": cutoff_value,
        "decision_session": session,
        "formal_activation_eligible": False,
        "protocol_version": PROTOCOL_VERSION,
        "session_ref_id": session_ref_id,
        "shadow_only": True,
        "shadow_run_ref": run_ref,
        "state": "SHADOW_COMPLETE",
        "strategy_id": strategy,
        "version": SESSION_REF_VERSION,
    }
    if research_factor_shadow:
        assert assertion_ref is not None
        session_payload.update(
            {
                "factor_evidence_mode": RESEARCH_FACTOR_EVIDENCE_MODE,
                "research_factor_shadow_assertion_ref": assertion_ref,
                "version": SESSION_REF_RESEARCH_VERSION,
            }
        )
    session_document = seal_semantic(session_payload)
    validate_artifact(session_document)
    session_path = _shadow_path(
        strategy,
        "sessions",
        session,
    )
    writer = _ShadowWriter(workspace_root)
    writer.initialize()
    assertion_created: bool | None = None
    if assertion is not None and assertion_path is not None:
        assertion_write = writer.write_exact_once(
            assertion_path,
            canonical_resource_bytes(assertion),
        )
        assertion_created = assertion_write.created
    run_write = writer.write_exact_once(
        run_path,
        canonical_resource_bytes(run),
    )
    session_write = writer.write_exact_once(
        session_path,
        canonical_resource_bytes(session_document),
    )
    result = {
        "authority": dict(NO_AUTHORITY),
        "canary_evidence_eligible": False,
        "formal_activation_eligible": False,
        "run_created": run_write.created,
        "session_created": session_write.created,
        "session_ref": _artifact_ref(
            session_document,
            relative_path=session_path,
        ),
        "shadow_run_ref": run_ref,
        "state": "SHADOW_COMPLETE",
    }
    if assertion_ref is not None:
        result.update(
            {
                "factor_evidence_mode": RESEARCH_FACTOR_EVIDENCE_MODE,
                "research_factor_shadow_assertion_created": assertion_created,
                "research_factor_shadow_assertion_ref": assertion_ref,
            }
        )
    return result


def revalidate_shadow_run(
    run_ref: Mapping[str, Any],
    *,
    reader: ExactReferenceReader,
) -> dict[str, Any]:
    """Replay a published Shadow run through Factor, Fusion and Deep."""

    run_version = str(run_ref.get("artifact_version") or "")
    if run_version == SHADOW_RUN_VERSION:
        run = _load_v4_ref(
            run_ref,
            expected_version=SHADOW_RUN_VERSION,
            reader=reader,
        )
        research_mode = False
    elif run_version == SHADOW_RUN_RESEARCH_VERSION:
        run = _load_v4_ref(
            run_ref,
            expected_version=SHADOW_RUN_RESEARCH_VERSION,
            reader=reader,
        )
        research_mode = True
    else:
        _blocked("shadow_run_version")
    if research_mode:
        assertion = _load_v4_ref(
            run["research_factor_shadow_assertion_ref"],
            expected_version=RESEARCH_FACTOR_SHADOW_ASSERTION_VERSION,
            reader=reader,
        )
        if (
            run["factor_evidence_mode"] != RESEARCH_FACTOR_EVIDENCE_MODE
            or assertion["assertion_scope"]
            != RESEARCH_FACTOR_ASSERTION_SCOPE
            or assertion["factor_evidence_mode"]
            != RESEARCH_FACTOR_EVIDENCE_MODE
            or assertion["strategy_id"] != run["strategy_id"]
            or assertion["cutoff"] != run["cutoff"]
            or assertion["created_at"] != run["created_at"]
            or assertion["decision_session"] != run["decision_session"]
            or assertion["shadow_run_id"] != run["shadow_run_id"]
            or assertion["factor_names"] != list(RESEARCH_FACTOR_NAMES)
            or assertion["factor_policy_sha256"]
            != RESEARCH_FACTOR_POLICY_SHA256
            or assertion["operator_asserted"] is not True
        ):
            _blocked("published_research_factor_assertion_binding")
    else:
        active, active_ref = _read_factor_common(
            reader=reader,
            path=run["factor_control_active_set_ref"]["relative_path"],
            sha256=run["factor_control_active_set_ref"]["byte_sha256"],
            expected_schema=ACTIVE_SET_SCHEMA_VERSION,
            strategy_id=run["strategy_id"],
            cutoff=run["cutoff"],
        )
        receipt, receipt_ref = _read_factor_common(
            reader=reader,
            path=run["factor_control_receipt_ref"]["relative_path"],
            sha256=run["factor_control_receipt_ref"]["byte_sha256"],
            expected_schema=CONTROL_RECEIPT_SCHEMA_VERSION,
            strategy_id=run["strategy_id"],
            cutoff=run["cutoff"],
        )
        active, _ = revalidate_factor_v4_closure(
            reader=reader,
            active_set=active,
            control_receipt=receipt,
            active_set_ref=active_ref,
            control_receipt_ref=receipt_ref,
            cutoff=run["cutoff"],
            decision_session=run["decision_session"],
        )
        if (
            active_ref != run["factor_control_active_set_ref"]
            or receipt_ref != run["factor_control_receipt_ref"]
            or active["production_factor_names"] != run["production_factor_names"]
            or active["production_factor_set_sha256"] != run["factor_set_sha256"]
        ):
            _blocked("published_factor_binding")
    loaded = {
        "source_locator": _load_v4_ref(
            run["source_locator_ref"],
            expected_version=LOCATOR_VERSION,
            reader=reader,
        ),
        "initial_pool": _load_v4_ref(
            run["initial_pool_ref"],
            expected_version=INITIAL_POOL_VERSION,
            reader=reader,
        ),
        "quant_branch": _load_v4_ref(
            run["quant_branch_ref"],
            expected_version=RESEARCH_QUANT_BRANCH_VERSION,
            reader=reader,
        ),
        "fundamental_branch": _load_v4_ref(
            run["fundamental_branch_ref"],
            expected_version=BRANCH_VERSION,
            reader=reader,
        ),
        "fusion_top24": _load_v4_ref(
            run["fusion_top24_ref"],
            expected_version=FUSION_VERSION,
            reader=reader,
        ),
        "deep_bundle": _load_v4_ref(
            run["deep_bundle_ref"],
            expected_version=DEEP_BUNDLE_V2,
            reader=reader,
        ),
        "holdings_snapshot": _load_v4_ref(
            run["comparison_inputs"]["holdings_ref"],
            expected_version=HOLDINGS_VERSION,
            reader=reader,
        ),
    }
    locator = loaded["source_locator"]
    initial = loaded["initial_pool"]
    quant = loaded["quant_branch"]
    fundamental = loaded["fundamental_branch"]
    fusion = loaded["fusion_top24"]
    deep = loaded["deep_bundle"]
    holdings = loaded["holdings_snapshot"]
    promotion = _load_v4_ref(
        fusion["promotion_receipt_ref"],
        expected_version=PROMOTION_VERSION,
        reader=reader,
        transitive=True,
    )
    catalog = _load_v4_ref(
        locator["pit_catalog_ref"],
        expected_version=PIT_CATALOG_VERSION,
        reader=reader,
    )
    loader = lambda reference: reader.read(
        reference["relative_path"],
        reference["byte_sha256"],
    )
    replayed_deep, replayed_fusion, _ = revalidate_deep_v2_bundle(
        run["deep_bundle_ref"],
        artifact_loader=loader,
    )
    if (
        any(
            document["strategy_id"] != run["strategy_id"] or document["cutoff"] != run["cutoff"]
            for document in loaded.values()
        )
        or initial["preselect_locator_ref"] != run["source_locator_ref"]
        or quant["initial_pool_ref"] != run["initial_pool_ref"]
        or fundamental["initial_pool_ref"] != run["initial_pool_ref"]
        or fusion["promotion_receipt_ref"]
        != _artifact_ref(
            promotion,
            relative_path=fusion["promotion_receipt_ref"]["relative_path"],
        )
        or deep != replayed_deep
        or fusion != replayed_fusion
        or deep["fusion_top24_ref"] != run["fusion_top24_ref"]
        or fusion["run_id"] != run["shadow_run_id"]
        or deep["run_id"] != run["shadow_run_id"]
        or catalog["decision_session"] != run["decision_session"]
        or holdings["as_of_session"] > run["decision_session"]
    ):
        _blocked("published_transitive_binding")
    _replay_fusion(
        initial_pool=initial,
        quant_branch=quant,
        fundamental_branch=fundamental,
        fusion=fusion,
        promotion=promotion,
    )
    revalidate_research_quant_branch(
        quant,
        initial_pool=initial,
        initial_pool_ref=run["initial_pool_ref"],
        reader=reader,
    )
    if (
        run["research_quant_factor_names"] != list(RESEARCH_FACTOR_NAMES)
        or run["research_quant_factor_policy_sha256"]
        != RESEARCH_FACTOR_POLICY_SHA256
    ):
        _blocked("published_research_quant_binding")
    expected_comparison = {
        "calendar_ref": dict(catalog["dataset_refs"]["cn_open_day_calendar"]),
        "holdings_ref": dict(run["comparison_inputs"]["holdings_ref"]),
        "market_bars_ref": dict(catalog["dataset_refs"]["market_bars"]),
        "source_closure_ref": dict(locator["pit_catalog_ref"]),
    }
    if run["comparison_inputs"] != expected_comparison:
        _blocked("published_comparison_binding")
    for reference in (
        expected_comparison["calendar_ref"],
        expected_comparison["market_bars_ref"],
    ):
        reader.verify_sha256(
            reference["relative_path"],
            reference["byte_sha256"],
        )
    return run


def read_shadow_session(
    workspace_root: str,
    *,
    strategy_id: str,
    decision_session: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and replay one immutable Shadow session reference."""

    path = _shadow_path(
        strategy_id,
        "sessions",
        decision_session,
    )
    reader = ExactReferenceReader(workspace_root)
    try:
        raw = reader.read(path, expected_sha256)
        session = load_canonical_resource(raw, label=SESSION_REF_VERSION)
        validate_artifact(session)
    except Exception as exc:
        raise ShadowRuntimeError("V17_V4_SHADOW_BLOCKED:session_readback") from exc
    if type(session) is not dict:
        _blocked("session_root")
    run = revalidate_shadow_run(
        session["shadow_run_ref"],
        reader=reader,
    )
    if session["version"] == SESSION_REF_VERSION:
        if run["version"] != SHADOW_RUN_VERSION:
            _blocked("session_run_version_binding")
    elif session["version"] == SESSION_REF_RESEARCH_VERSION:
        if (
            run["version"] != SHADOW_RUN_RESEARCH_VERSION
            or session["factor_evidence_mode"]
            != RESEARCH_FACTOR_EVIDENCE_MODE
            or session["research_factor_shadow_assertion_ref"]
            != run["research_factor_shadow_assertion_ref"]
        ):
            _blocked("research_session_run_binding")
    else:
        _blocked("session_version")
    return {
        "session": session,
        "session_path": path,
        "session_sha256": hashlib.sha256(raw).hexdigest(),
        "shadow_run": run,
    }


__all__ = [
    "READINESS_VERSION",
    "RESEARCH_FACTOR_EVIDENCE_MODE",
    "RESEARCH_FACTOR_SHADOW_ASSERTION_VERSION",
    "SESSION_REF_RESEARCH_VERSION",
    "SESSION_REF_VERSION",
    "SHADOW_RUN_RESEARCH_VERSION",
    "SHADOW_RUN_VERSION",
    "ShadowRuntimeError",
    "publish_shadow_run",
    "read_shadow_session",
    "revalidate_shadow_run",
    "require_strategy_path_id",
    "revalidate_factor_v4_closure",
]
