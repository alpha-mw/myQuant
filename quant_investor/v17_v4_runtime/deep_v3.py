"""Offline Deep v3 compiler for the forward-only Fusion v2 chain."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
from pathlib import PurePosixPath
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.resources import (
    load_packaged_json,
    read_packaged_asset,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)

from .source_storage import (
    ExactReferenceReader,
    GovernedStore,
    RUN_ROOT,
    SourceStorageError,
    canonical_governed_path,
)

ASSESSMENT_VERSION: Final = "myquant.v17.v4.deep-assessment-manifest.v2"
OFFICIAL_EVIDENCE_V3: Final = "myquant.v17.v4.official-evidence.v3"
ISSUER_DOSSIER_V3: Final = "myquant.v17.v4.issuer-dossier.v3"
EVENT_SCAN_V3: Final = "myquant.v17.v4.event-scan.v3"
DEEP_BUNDLE_V3: Final = "myquant.v17.v4.deep-evidence-bundle.v3"
FUSION_TOP24_V2: Final = "myquant.v17.v4.fusion-top24.v2"
POLICY_VERSION: Final = "myquant.v17.v4.deep-scoring-policy.v1"
POLICY_PATH: Final = "resources/deep_scoring_policy.v1.json"
TOP_N: Final = 24
MAX_DOSSIER_AGE_DAYS: Final = 30
MAX_EVENT_SCAN_AGE_DAYS: Final = 7
MODULE_ORDER: Final = (
    "financial_reconciliation",
    "business_model",
    "industry",
    "competition",
    "management",
    "valuation",
    "catalysts",
    "contrary_evidence",
    "falsification_conditions",
    "monitoring",
)
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}


class DeepV3Error(RuntimeError):
    """Raised when Deep v3 cannot prove an exact owner-staged closure."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise DeepV3Error(f"V17_V4_DEEP_V3_BLOCKED:{reason}")


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


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


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
    except (IdentityContractError, TypeError, ValueError) as exc:
        raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:artifact_ref_identity") from exc
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


def _load_exact_artifact(
    reference: Mapping[str, Any],
    *,
    expected_version: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    try:
        normalized = {str(key): str(value) for key, value in reference.items()}
        raw = artifact_loader(normalized)
        document = load_canonical_resource(raw, label=expected_version)
        validated = load_canonical_artifact(
            raw,
            expected_version=expected_version,
            artifact_loader=artifact_loader,
        )
        identity_field = artifact_identity_field(expected_version)
    except Exception as exc:
        raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:artifact_readback") from exc
    if (
        type(document) is not dict
        or document.get(identity_field) != reference.get("artifact_id")
        or document.get("strategy_id") != reference.get("strategy_id")
        or document.get("cutoff") != reference.get("cutoff")
        or document.get("semantic_sha256") != reference.get("semantic_sha256")
        or hashlib.sha256(raw).hexdigest() != reference.get("byte_sha256")
        or validated.payload != document
    ):
        _blocked("artifact_reference_mismatch")
    return dict(document)


def _policy() -> tuple[dict[str, Any], dict[str, str]]:
    try:
        raw = read_packaged_asset(POLICY_PATH)
        policy = load_packaged_json(POLICY_PATH)
    except Exception as exc:
        raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:scoring_policy") from exc
    order = policy.get("module_order")
    weights = policy.get("module_weights")
    if (
        policy.get("version") != POLICY_VERSION
        or policy.get("protocol_version") != PROTOCOL_VERSION
        or policy.get("authority") != NO_AUTHORITY
        or order != list(MODULE_ORDER)
        or type(weights) is not dict
        or set(weights) != set(MODULE_ORDER)
        or sum(
            (_decimal(weights[name], label=f"weight.{name}") for name in MODULE_ORDER),
            Decimal("0"),
        )
        != Decimal("1")
        or policy.get("module_red_flag_conclusion_veto") is not True
        or policy.get("missing_evidence_action") != "BUY_VETO_ZERO_TARGET"
    ):
        _blocked("scoring_policy")
    return policy, {
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "relative_path": POLICY_PATH,
        "semantic_sha256": str(policy["semantic_sha256"]),
        "version": POLICY_VERSION,
    }


def _module_documents(
    modules: Sequence[Mapping[str, Any]],
    *,
    evidence_by_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    if [row.get("module_id") for row in modules] != list(MODULE_ORDER):
        _blocked("module_order")
    result: list[dict[str, Any]] = []
    for module in modules:
        try:
            references = sorted(
                (dict(evidence_by_id[evidence_id]) for evidence_id in module["evidence_ids"]),
                key=lambda row: (
                    row["relative_path"],
                    row["byte_sha256"],
                    row["artifact_id"],
                ),
            )
        except (KeyError, TypeError) as exc:
            raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:module_evidence") from exc
        result.append(
            {
                "conclusion": module["conclusion"],
                "evidence_refs": references,
                "finding": module["finding"],
                "module_id": module["module_id"],
                "score": _decimal_text(
                    _decimal(
                        module["score"],
                        label=f"module.{module['module_id']}",
                    )
                ),
            }
        )
    return result


def _module_states(status: str) -> list[dict[str, str]]:
    return [{"module_id": module_id, "status": status} for module_id in MODULE_ORDER]


def score_dossier_v3(
    dossier: Mapping[str, Any],
    event_scan: Mapping[str, Any],
) -> tuple[Decimal, bool]:
    """Recompute the fixed Deep score without any judgment surface."""

    policy, policy_ref = _policy()
    if dossier.get("scoring_policy_ref") != policy_ref or dossier.get("symbol") != event_scan.get(
        "symbol"
    ):
        _blocked("dossier_policy_or_symbol")
    modules = dossier.get("modules")
    if type(modules) is not list or [row.get("module_id") for row in modules] != list(MODULE_ORDER):
        _blocked("dossier_module_order")
    with localcontext() as context:
        context.prec = 40
        signal = sum(
            (
                _decimal(
                    policy["module_weights"][row["module_id"]],
                    label="module_weight",
                )
                * _decimal(row["score"], label="module_score")
                for row in modules
            ),
            Decimal("0"),
        )
    flags = set(dossier.get("red_flags") or ()) | set(event_scan.get("flags") or ())
    buy_veto = bool(
        flags & set(policy["severe_red_flags"])
        or any(row["conclusion"] == "RED_FLAG" for row in modules)
        or signal
        <= _decimal(
            policy["buy_veto_signal_threshold"],
            label="buy_veto_signal_threshold",
        )
    )
    return signal, buy_veto


def _target_after_deep(
    *,
    base_target: Decimal,
    signal: Decimal,
    buy_veto: bool,
) -> Decimal:
    policy, _ = _policy()
    if buy_veto:
        return Decimal("0")
    with localcontext() as context:
        context.prec = 40
        penalty = min(
            _decimal(
                policy["max_deep_penalty"],
                label="max_deep_penalty",
            ),
            max(
                Decimal("0"),
                _decimal(
                    policy["deep_penalty_scale"],
                    label="deep_penalty_scale",
                )
                * max(-signal, Decimal("0")),
            ),
        )
        return base_target * (Decimal("1") - penalty)


class _DeepV3Writer(GovernedStore):
    def __init__(self, workspace_root: str, *, run_id: str) -> None:
        super().__init__(workspace_root)
        self._root = RUN_ROOT / require_opaque_id(run_id, label="run_id") / "deep_v3"

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = super()._canonical_path(value)
        if path != self._root and self._root not in path.parents:
            raise SourceStorageError("Deep v3 writer path is outside its immutable run root")
        return path


def compile_deep_v3(
    workspace_root: str,
    *,
    assessment_manifest_path: str,
    expected_assessment_manifest_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Compile exact Fusion v2 assessments with no collection or inference."""

    expected = require_sha256(
        expected_assessment_manifest_sha256,
        label="expected_assessment_manifest_sha256",
    )
    created = require_utc_timestamp(created_at, label="created_at")
    reader = ExactReferenceReader(workspace_root)
    try:
        manifest_raw = reader.read(
            assessment_manifest_path,
            expected,
        )
        manifest = load_canonical_resource(
            manifest_raw,
            label=ASSESSMENT_VERSION,
        )
        validate_artifact(manifest)
    except Exception as exc:
        raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:assessment_manifest") from exc
    if type(manifest) is not dict or manifest.get("version") != ASSESSMENT_VERSION:
        _blocked("assessment_manifest_version")
    manifest_ref = _artifact_ref(
        manifest,
        relative_path=assessment_manifest_path,
    )
    if manifest_ref["byte_sha256"] != expected:
        _blocked("assessment_manifest_sha")

    def artifact_loader(reference: Mapping[str, str]) -> bytes:
        return reader.read(
            reference["relative_path"],
            reference["byte_sha256"],
        )

    fusion = _load_exact_artifact(
        manifest["fusion_top24_ref"],
        expected_version=FUSION_TOP24_V2,
        artifact_loader=artifact_loader,
    )
    if created < manifest["cutoff"] or [row["symbol"] for row in manifest["rows"]] != [
        row["symbol"] for row in fusion["rows"]
    ]:
        _blocked("assessment_fusion_binding")
    _, policy_ref = _policy()
    staged: list[tuple[str, bytes]] = []
    deep_rows: list[dict[str, Any]] = []
    in_memory: dict[str, bytes] = {
        manifest_ref["byte_sha256"]: manifest_raw,
        manifest["fusion_top24_ref"]["byte_sha256"]: canonical_resource_bytes(fusion),
    }

    def staged_loader(reference: Mapping[str, str]) -> bytes:
        digest = reference["byte_sha256"]
        if digest in in_memory:
            return in_memory[digest]
        return artifact_loader(reference)

    for assessment, top_row in zip(
        manifest["rows"],
        fusion["rows"],
        strict=True,
    ):
        symbol = assessment["symbol"]
        if assessment["status"] == "UNAVAILABLE":
            deep_rows.append(
                {
                    "blocker_codes": list(assessment["blocker_codes"]),
                    "buy_veto": True,
                    "event_scan_ref": None,
                    "issuer_dossier_ref": None,
                    "modules": _module_states("UNAVAILABLE"),
                    "official_evidence_refs": [],
                    "signal": None,
                    "status": "UNAVAILABLE",
                    "symbol": symbol,
                    "target_after_deep": "0",
                }
            )
            continue
        official_refs: list[dict[str, str]] = []
        evidence_by_id: dict[str, dict[str, str]] = {}
        for raw_document in assessment["raw_documents"]:
            try:
                raw_content = reader.read(
                    raw_document["raw_relative_path"],
                    raw_document["raw_byte_sha256"],
                )
            except SourceStorageError as exc:
                raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:raw_evidence_readback") from exc
            if len(raw_content) != raw_document["raw_size"]:
                _blocked("raw_evidence_size")
            evidence = seal_semantic(
                {
                    "assessment_manifest_ref": manifest_ref,
                    "authority": dict(NO_AUTHORITY),
                    "available_at": raw_document["available_at"],
                    "captured_at": raw_document["captured_at"],
                    "content_relative_path": raw_document["raw_relative_path"],
                    "content_sha256": raw_document["raw_byte_sha256"],
                    "content_size": raw_document["raw_size"],
                    "cutoff": manifest["cutoff"],
                    "evidence_id": raw_document["evidence_id"],
                    "evidence_kind": raw_document["evidence_kind"],
                    "official_source_id": raw_document["official_source_id"],
                    "parser_id": raw_document["parser_id"],
                    "parser_sha256": raw_document["parser_sha256"],
                    "parser_version": raw_document["parser_version"],
                    "protocol_version": PROTOCOL_VERSION,
                    "published_at": raw_document["published_at"],
                    "strategy_id": manifest["strategy_id"],
                    "symbol": symbol,
                    "version": OFFICIAL_EVIDENCE_V3,
                }
            )
            validate_artifact(evidence)
            path = (
                f"{RUN_ROOT}/{manifest['run_id']}/deep_v3/"
                f"{symbol}/official/"
                f"{raw_document['evidence_id']}.json"
            )
            reference = _artifact_ref(
                evidence,
                relative_path=path,
            )
            raw = canonical_resource_bytes(evidence)
            in_memory[reference["byte_sha256"]] = raw
            staged.append((path, raw))
            official_refs.append(reference)
            evidence_by_id[raw_document["evidence_id"]] = reference
        official_refs.sort(
            key=lambda row: (
                row["relative_path"],
                row["byte_sha256"],
                row["artifact_id"],
            )
        )
        modules = _module_documents(
            assessment["modules"],
            evidence_by_id=evidence_by_id,
        )
        red_flags = sorted(
            {
                *assessment["event_flags"],
                *(
                    f"module_red_flag.{row['module_id']}"
                    for row in modules
                    if row["conclusion"] == "RED_FLAG"
                ),
            }
        )
        stable_id = hashlib.sha256(f"{manifest['run_id']}:{symbol}".encode("ascii")).hexdigest()[
            :24
        ]
        dossier = seal_semantic(
            {
                "as_of": manifest["cutoff"][:10],
                "assessment_manifest_ref": manifest_ref,
                "authority": dict(NO_AUTHORITY),
                "created_at": created,
                "cutoff": manifest["cutoff"],
                "dossier_id": f"deep-v3-dossier-{stable_id}",
                "modules": modules,
                "official_evidence_refs": official_refs,
                "protocol_version": PROTOCOL_VERSION,
                "red_flags": red_flags,
                "scoring_policy_ref": policy_ref,
                "strategy_id": manifest["strategy_id"],
                "summary_sha256": hashlib.sha256(canonical_bytes(modules)).hexdigest(),
                "symbol": symbol,
                "version": ISSUER_DOSSIER_V3,
            }
        )
        validate_artifact(dossier)
        dossier_path = f"{RUN_ROOT}/{manifest['run_id']}/deep_v3/" f"{symbol}/issuer_dossier.json"
        dossier_ref = _artifact_ref(
            dossier,
            relative_path=dossier_path,
        )
        dossier_raw = canonical_resource_bytes(dossier)
        in_memory[dossier_ref["byte_sha256"]] = dossier_raw
        staged.append((dossier_path, dossier_raw))
        event = seal_semantic(
            {
                "as_of": manifest["cutoff"][:10],
                "assessment_manifest_ref": manifest_ref,
                "authority": dict(NO_AUTHORITY),
                "created_at": created,
                "cutoff": manifest["cutoff"],
                "flags": list(assessment["event_flags"]),
                "official_evidence_refs": official_refs,
                "protocol_version": PROTOCOL_VERSION,
                "scan_id": f"deep-v3-event-{stable_id}",
                "strategy_id": manifest["strategy_id"],
                "symbol": symbol,
                "version": EVENT_SCAN_V3,
            }
        )
        validate_artifact(event)
        event_path = f"{RUN_ROOT}/{manifest['run_id']}/deep_v3/" f"{symbol}/event_scan.json"
        event_ref = _artifact_ref(event, relative_path=event_path)
        event_raw = canonical_resource_bytes(event)
        in_memory[event_ref["byte_sha256"]] = event_raw
        staged.append((event_path, event_raw))
        signal, buy_veto = score_dossier_v3(dossier, event)
        target = _target_after_deep(
            base_target=_decimal(
                top_row["base_target"],
                label=f"{symbol}.base_target",
            ),
            signal=signal,
            buy_veto=buy_veto,
        )
        deep_rows.append(
            {
                "blocker_codes": [],
                "buy_veto": buy_veto,
                "event_scan_ref": event_ref,
                "issuer_dossier_ref": dossier_ref,
                "modules": _module_states("COMPLETE"),
                "official_evidence_refs": official_refs,
                "signal": _decimal_text(signal),
                "status": "COMPLETE",
                "symbol": symbol,
                "target_after_deep": _decimal_text(target),
            }
        )
    bundle_id = (
        "deep-v3-bundle-"
        + hashlib.sha256(
            (f"{manifest['run_id']}:" f"{manifest['request_id']}").encode("ascii")
        ).hexdigest()[:24]
    )
    bundle = seal_semantic(
        {
            "assessment_manifest_ref": manifest_ref,
            "authority": dict(NO_AUTHORITY),
            "bundle_id": bundle_id,
            "created_at": created,
            "cutoff": manifest["cutoff"],
            "fusion_top24_ref": dict(manifest["fusion_top24_ref"]),
            "performance_evidence_eligible": False,
            "policy_promotion_eligible": False,
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "rows": deep_rows,
            "run_id": manifest["run_id"],
            "scoring_policy_ref": policy_ref,
            "shadow_only": True,
            "state": "DEEP_COMPLETE_SHADOW",
            "strategy_id": manifest["strategy_id"],
            "version": DEEP_BUNDLE_V3,
        }
    )
    validate_artifact(bundle)
    bundle_path = f"{RUN_ROOT}/{manifest['run_id']}/deep_v3/" "deep_evidence_bundle.v3.json"
    bundle_ref = _artifact_ref(bundle, relative_path=bundle_path)
    bundle_raw = canonical_resource_bytes(bundle)
    in_memory[bundle_ref["byte_sha256"]] = bundle_raw
    replayed, replayed_fusion, _ = revalidate_deep_v3_bundle(
        bundle_ref,
        artifact_loader=staged_loader,
    )
    if replayed != bundle or replayed_fusion != fusion:
        _blocked("prepublication_replay")
    writer = _DeepV3Writer(
        workspace_root,
        run_id=manifest["run_id"],
    )
    writer.initialize()
    for path, raw in staged:
        writer.write_exact_once(path, raw)
    write_result = writer.write_exact_once(
        bundle_path,
        bundle_raw,
    )
    return {
        "authority": dict(NO_AUTHORITY),
        "bundle_ref": bundle_ref,
        "created": write_result.created,
        "performance_evidence_eligible": False,
        "policy_promotion_eligible": False,
        "promotion_eligible": False,
        "run_id": manifest["run_id"],
        "shadow_only": True,
        "status": "DEEP_V3_COMPILED",
        "strategy_id": manifest["strategy_id"],
    }


def revalidate_deep_v3_bundle(
    bundle_ref: Mapping[str, Any],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay Deep v3 through the manifest, children and official bytes."""

    bundle = _load_exact_artifact(
        bundle_ref,
        expected_version=DEEP_BUNDLE_V3,
        artifact_loader=artifact_loader,
    )
    fusion = _load_exact_artifact(
        bundle["fusion_top24_ref"],
        expected_version=FUSION_TOP24_V2,
        artifact_loader=artifact_loader,
    )
    manifest = _load_exact_artifact(
        bundle["assessment_manifest_ref"],
        expected_version=ASSESSMENT_VERSION,
        artifact_loader=artifact_loader,
    )
    _, policy_ref = _policy()
    if (
        bundle["scoring_policy_ref"] != policy_ref
        or manifest["fusion_top24_ref"] != bundle["fusion_top24_ref"]
        or manifest["run_id"] != bundle["run_id"]
        or [row["symbol"] for row in bundle["rows"]] != [row["symbol"] for row in fusion["rows"]]
        or [row["symbol"] for row in manifest["rows"]] != [row["symbol"] for row in fusion["rows"]]
    ):
        _blocked("bundle_transitive_binding")
    for bundle_row, assessment, top_row in zip(
        bundle["rows"],
        manifest["rows"],
        fusion["rows"],
        strict=True,
    ):
        if assessment["status"] == "UNAVAILABLE":
            expected = {
                "blocker_codes": list(assessment["blocker_codes"]),
                "buy_veto": True,
                "event_scan_ref": None,
                "issuer_dossier_ref": None,
                "modules": _module_states("UNAVAILABLE"),
                "official_evidence_refs": [],
                "signal": None,
                "status": "UNAVAILABLE",
                "symbol": assessment["symbol"],
                "target_after_deep": "0",
            }
            if bundle_row != expected:
                _blocked("unavailable_row_replay")
            continue
        evidence_by_id: dict[str, dict[str, str]] = {}
        for reference in bundle_row["official_evidence_refs"]:
            document = _load_exact_artifact(
                reference,
                expected_version=OFFICIAL_EVIDENCE_V3,
                artifact_loader=artifact_loader,
            )
            try:
                raw = artifact_loader(
                    {
                        "relative_path": document["content_relative_path"],
                        "byte_sha256": document["content_sha256"],
                    }
                )
            except Exception as exc:
                raise DeepV3Error("V17_V4_DEEP_V3_BLOCKED:raw_replay") from exc
            if (
                type(raw) is not bytes
                or len(raw) != document["content_size"]
                or hashlib.sha256(raw).hexdigest() != document["content_sha256"]
                or document["symbol"] != bundle_row["symbol"]
                or document["assessment_manifest_ref"] != bundle["assessment_manifest_ref"]
            ):
                _blocked("official_evidence_replay")
            evidence_by_id[document["evidence_id"]] = dict(reference)
        dossier = _load_exact_artifact(
            bundle_row["issuer_dossier_ref"],
            expected_version=ISSUER_DOSSIER_V3,
            artifact_loader=artifact_loader,
        )
        event = _load_exact_artifact(
            bundle_row["event_scan_ref"],
            expected_version=EVENT_SCAN_V3,
            artifact_loader=artifact_loader,
        )
        cutoff_date = date.fromisoformat(bundle["cutoff"][:10])
        dossier_date = date.fromisoformat(dossier["as_of"])
        event_date = date.fromisoformat(event["as_of"])
        expected_modules = _module_documents(
            assessment["modules"],
            evidence_by_id=evidence_by_id,
        )
        if (
            not 0 <= (cutoff_date - dossier_date).days <= MAX_DOSSIER_AGE_DAYS
            or not 0 <= (cutoff_date - event_date).days <= MAX_EVENT_SCAN_AGE_DAYS
            or dossier["modules"] != expected_modules
            or event["flags"] != assessment["event_flags"]
            or dossier["official_evidence_refs"] != bundle_row["official_evidence_refs"]
            or event["official_evidence_refs"] != bundle_row["official_evidence_refs"]
            or dossier["assessment_manifest_ref"] != bundle["assessment_manifest_ref"]
            or event["assessment_manifest_ref"] != bundle["assessment_manifest_ref"]
        ):
            _blocked("dossier_event_replay")
        signal, buy_veto = score_dossier_v3(dossier, event)
        target = _target_after_deep(
            base_target=_decimal(
                top_row["base_target"],
                label="base_target",
            ),
            signal=signal,
            buy_veto=buy_veto,
        )
        expected_row = {
            "blocker_codes": [],
            "buy_veto": buy_veto,
            "event_scan_ref": dict(bundle_row["event_scan_ref"]),
            "issuer_dossier_ref": dict(bundle_row["issuer_dossier_ref"]),
            "modules": _module_states("COMPLETE"),
            "official_evidence_refs": [
                dict(reference) for reference in bundle_row["official_evidence_refs"]
            ],
            "signal": _decimal_text(signal),
            "status": "COMPLETE",
            "symbol": bundle_row["symbol"],
            "target_after_deep": _decimal_text(target),
        }
        if bundle_row != expected_row:
            _blocked("complete_row_replay")
    return bundle, fusion, manifest


__all__ = [
    "ASSESSMENT_VERSION",
    "DEEP_BUNDLE_V3",
    "EVENT_SCAN_V3",
    "FUSION_TOP24_V2",
    "ISSUER_DOSSIER_V3",
    "MODULE_ORDER",
    "OFFICIAL_EVIDENCE_V3",
    "DeepV3Error",
    "compile_deep_v3",
    "revalidate_deep_v3_bundle",
    "score_dossier_v3",
]
