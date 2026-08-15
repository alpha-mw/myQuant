"""Fixed contextual validation callbacks invoked only by the System runner.

Callbacks accept a stored validation-request envelope and a System-owned time,
replay the bound Factor closure, and return an exact non-authorizing payload.
They never seal, publish, mutate candidate state, or activate a generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any, Final

import pandas as pd

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    COMPONENT_REGISTRY_SHA256,
    PROSPECTIVE_VALIDATION_PROFILE,
    SystemError,
    SystemStore,
    object_ref_for_artifact,
    validate_installed_component_manifest,
)

from .bootstrap import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    compute_bootstrap_signals,
    validate_bootstrap_factor_set,
)
from .bootstrap_evidence import (
    build_bootstrap_exception_evidence,
    validate_bootstrap_exception_evidence,
)
from .common import (
    artifact_ref,
    business_identity,
    canonical_timestamp,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
)
from .custody import replay_custody_chain
from .errors import FactorGovernanceError
from .execution import validate_execution_turnover_evidence
from .manifest import validate_validator_manifest
from .prospective import (
    _build_preregistration,
    validate_configuration_selection,
    validate_preregistration,
)
from .receipt import _build_factor_validation_receipt, validate_factor_validation_receipt
from .source import (
    DecodedSource,
    build_source_decode_attestation,
    decode_source_role,
    validate_source_decode_attestation,
)
from .store import FactorValidationStore, prospective_validation_namespace_id

CONTEXTUAL_RESULT_KIND: Final = "factor.contextual_validation_result"
_MAXIMUM_JSON_BYTES: Final = 16 * 1024 * 1024
_BOOTSTRAP_DECISION_PATH: Final = "operations/unified_cutover/bootstrap-decision.json"
_CONTEXT_FIELDS: Final = {
    "contextual_result_id",
    "validation_namespace_id",
    "lane",
    "intrinsic_receipt_ref",
    "policy_ref",
    "evidence_refs",
    "active_set_ref",
    "composite_state_ref",
    "factor_validator_manifest_ref",
    "contextual_validator_component_ref",
    "source_decoder_component_ref",
    "implementation_component_refs",
    "source_attestation_refs",
    "source_object_refs",
    "custody_record_refs",
    "custody_tree_sha256",
    "custody_head_ref",
    "validated",
    "blockers",
    "authority",
}
_REF_SORT_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)
_BOOTSTRAP_ROLE_MATRIX: Final = {
    "decision_source": ("bootstrap_decision", "JSON", None),
    "exchange_calendar": ("calendar", "PARQUET", "exchange_calendar"),
    "implementation": ("implementation_tree_manifest", "JSON", None),
    "market": ("market", "PARQUET", "market_history"),
    "pit_universe": ("pit", "PARQUET", "pit_universe"),
    "recomputation": ("recomputation", "JSON", None),
    "source_generation": ("source_generation", "JSON", None),
}


def _ref_key(value: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(value[field] for field in _REF_SORT_FIELDS)


def _sorted_refs(values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = [validate_artifact_ref(value, label="context reference") for value in values]
    rows.sort(key=_ref_key)
    keys = [_ref_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise FactorGovernanceError("context reference closure contains duplicates")
    return rows


def _context_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(values)
    if set(body) != _CONTEXT_FIELDS - {"contextual_result_id"}:
        raise FactorGovernanceError("contextual result fields are not exact")
    return {
        "contextual_result_id": business_identity("factor-contextual-result", body),
        **body,
    }


def validate_contextual_result(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate the inert contextual envelope without claiming protected custody."""

    envelope, payload = exact_payload(
        document,
        kind=CONTEXTUAL_RESULT_KIND,
        fields=_CONTEXT_FIELDS,
    )
    for field, kind, nullable in (
        ("intrinsic_receipt_ref", "factor.validation_receipt", False),
        ("policy_ref", None, False),
        ("active_set_ref", None, False),
        ("composite_state_ref", "factor.composite_state", True),
        ("factor_validator_manifest_ref", "factor.validator_manifest", False),
        (
            "contextual_validator_component_ref",
            "system.installed_component_manifest",
            False,
        ),
        ("source_decoder_component_ref", "system.installed_component_manifest", False),
        ("custody_head_ref", "factor.custody_record", True),
    ):
        value = payload[field]
        if nullable and value is None:
            continue
        validate_artifact_ref(value, label=field, expected_kind=kind)
    for field, kind, sorted_unique in (
        ("evidence_refs", None, True),
        ("implementation_component_refs", "system.installed_component_manifest", True),
        ("source_attestation_refs", "factor.source_decode_attestation", True),
        ("source_object_refs", "system.source_object", True),
        ("custody_record_refs", "factor.custody_record", False),
    ):
        raw = payload[field]
        if type(raw) is not list:
            raise FactorGovernanceError(f"{field} must be a list")
        refs = [
            validate_artifact_ref(value, label=f"{field}[{index}]", expected_kind=kind)
            for index, value in enumerate(raw)
        ]
        keys = [_ref_key(row) for row in refs]
        if len(keys) != len(set(keys)) or (sorted_unique and keys != sorted(keys)):
            raise FactorGovernanceError(f"{field} is not canonical")
    require_sha256(payload["custody_tree_sha256"], label="custody_tree_sha256")
    if (
        payload["lane"] not in {"BOOTSTRAP", "PROSPECTIVE"}
        or type(payload["validation_namespace_id"]) is not str
        or not payload["validation_namespace_id"]
        or payload["validated"] is not True
        or payload["blockers"] != []
        or payload["authority"] != "NON_AUTHORIZING"
        or payload
        != _context_payload(
            {field: payload[field] for field in _CONTEXT_FIELDS if field != "contextual_result_id"}
        )
    ):
        raise FactorGovernanceError("contextual validation result does not replay")
    return envelope


def _stored(
    system_store: SystemStore,
    value: Mapping[str, Any],
    *,
    label: str,
    expected_kind: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    ref = validate_artifact_ref(value, label=label, expected_kind=expected_kind)
    try:
        artifact = system_store.get_object(ref)
    except SystemError as exc:
        raise FactorGovernanceError(
            f"{label} cannot be resolved",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        ) from exc
    if object_ref_for_artifact(artifact) != ref:
        raise FactorGovernanceError(
            f"{label} exact ref differs",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    return ref, artifact


def _request_payload(
    validation_request: Mapping[str, Any],
    *,
    expected_profile: str,
) -> dict[str, Any]:
    try:
        request = validate_artifact(
            validation_request,
            expected_kind="system.validation_run_request",
        )
    except ContractError as exc:
        raise FactorGovernanceError("validation request envelope is invalid") from exc
    payload = request["payload"]
    if (
        payload["validation_profile_id"] != expected_profile
        or payload["component_registry_sha256"] != COMPONENT_REGISTRY_SHA256
    ):
        raise FactorGovernanceError("validation request profile differs")
    return payload


def _read_json_source(
    system_store: SystemStore,
    source_ref: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], Any, bytes]:
    try:
        payload, raw = system_store.read_source_object_bytes(
            source_ref,
            maximum_bytes=_MAXIMUM_JSON_BYTES,
        )
    except SystemError as exc:
        raise FactorGovernanceError(
            f"{label} cannot be read securely",
            code="SOURCE_VALIDATION_FAILED",
        ) from exc
    if payload["source_format"] != "JSON" or payload["media_type"] != "application/json":
        raise FactorGovernanceError(f"{label} is not canonical JSON")
    try:
        document = parse_canonical_json_bytes(raw, label=label)
    except ContractError as exc:
        raise FactorGovernanceError(f"{label} is not canonical JSON") from exc
    return payload, document, raw


def _bundle_source(
    system_store: SystemStore,
    bundle_ref: Mapping[str, Any],
    *,
    outer_role: str,
    inner_role: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    _, bundle = _stored(
        system_store,
        bundle_ref,
        label=f"{outer_role} bundle",
        expected_kind="system.source_bundle",
    )
    rows = bundle["payload"].get("sources")
    if (
        type(rows) is not list
        or len(rows) != 1
        or type(rows[0]) is not dict
        or set(rows[0]) != {"role", "source_ref"}
        or rows[0]["role"] != inner_role
    ):
        raise FactorGovernanceError(f"{outer_role} bundle closure is not exact")
    ref = validate_artifact_ref(
        rows[0]["source_ref"],
        label=f"{outer_role}.{inner_role}",
        expected_kind="system.source_object",
    )
    return bundle, ref


def _bootstrap_sources(  # noqa: C901
    system_store: SystemStore,
    policy: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    rows = policy["payload"]["source_refs"]
    by_role = {row["role"]: row["ref"] for row in rows}
    if set(by_role) != {"code", *_BOOTSTRAP_ROLE_MATRIX}:
        raise FactorGovernanceError("Bootstrap policy source roles are not exact")
    bundles: dict[str, dict[str, Any]] = {}
    source_refs: dict[str, dict[str, str]] = {}
    decoded: dict[str, dict[str, Any]] = {}
    json_documents: dict[str, Any] = {}

    def project_bootstrap_market(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        frame = table.to_pandas()
        try:
            frames = {
                symbol: group.drop(columns=["symbol"]).reset_index(drop=True)
                for symbol, group in frame.groupby("symbol", sort=True)
            }
            signals = compute_bootstrap_signals(frames, source_format="PARQUET")
            return {
                "signals": {
                    factor_id: {
                        symbol: (None if pd.isna(number) else float(number).hex())
                        for symbol, number in series.sort_index().items()
                    }
                    for factor_id, series in signals.items()
                    if factor_id in {LOW_DOLLAR_VOLUME, BLEND_W80}
                }
            }
        finally:
            del frame

    def project_bootstrap_pit(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        frame = table.to_pandas()
        try:
            eligible = frame.loc[
                frame["tradable"].eq(True) & frame["total_mv"].gt(0),  # noqa: E712
                "symbol",
            ].tolist()
            if (
                not eligible
                or any(type(symbol) is not str for symbol in eligible)
                or eligible != sorted(set(eligible))
            ):
                raise FactorGovernanceError("Bootstrap PIT eligible cohort is not exact")
            return {"eligible_symbols": eligible}
        finally:
            del frame

    for outer_role, (inner_role, source_format, decoder_role) in _BOOTSTRAP_ROLE_MATRIX.items():
        bundle, source_ref = _bundle_source(
            system_store,
            by_role[outer_role],
            outer_role=outer_role,
            inner_role=inner_role,
        )
        bundles[outer_role] = bundle
        source_refs[outer_role] = source_ref
        if source_format == "JSON":
            payload, document, raw = _read_json_source(
                system_store,
                source_ref,
                label=inner_role,
            )
            json_documents[outer_role] = {
                "document": document,
                "payload": payload,
                "raw": raw,
            }
        else:
            assert decoder_role is not None
            value = decode_source_role(
                system_store=system_store,
                source_object_ref=source_ref,
                role=decoder_role,
                projector=(
                    project_bootstrap_market
                    if decoder_role == "market_history"
                    else (
                        project_bootstrap_pit
                        if decoder_role == "pit_universe"
                        else lambda table, binding: {}
                    )
                ),
            )
            if decoder_role == "market_history":
                decoded["signals"] = value.projection["signals"]
            if decoder_role == "pit_universe":
                decoded["eligible_symbols"] = value.projection["eligible_symbols"]
            decoded[decoder_role] = {
                "binding": dict(value.binding),
            }
    return bundles, source_refs, decoded, json_documents


def _signal_hashes(signals: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        factor_id: hashlib.sha256(
            canonical_json_bytes(
                [{"symbol": symbol, "signal": values[symbol]} for symbol in sorted(values)]
            )
        ).hexdigest()
        for factor_id, values in sorted(signals.items())
    }


def _signal_statistics(  # noqa: C901
    signals: Mapping[str, Mapping[str, Any]],
    *,
    eligible_symbols: Sequence[str],
    implementation_sha256s: Mapping[str, str],
    source_bundle_sha256: str,
) -> list[dict[str, Any]]:
    """Seal non-empty, nonconstant exact-replay evidence for active factors."""

    cohort = list(eligible_symbols)
    if not cohort or cohort != sorted(set(cohort)):
        raise FactorGovernanceError("Bootstrap PIT eligible cohort is not canonical")
    cohort_sha = hashlib.sha256(canonical_json_bytes(cohort)).hexdigest()
    signal_hashes = _signal_hashes(signals)
    if set(signals) != {LOW_DOLLAR_VOLUME, BLEND_W80}:
        raise FactorGovernanceError("Bootstrap active signal set is not exact")
    rows: list[dict[str, Any]] = []
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        values = signals[factor_id]
        symbols = sorted(values)
        if symbols != cohort:
            raise FactorGovernanceError("Bootstrap signal/PIT cohort differs")
        finite_values: list[str] = []
        for symbol in symbols:
            value = values[symbol]
            if value is None:
                continue
            if type(value) is not str:
                raise FactorGovernanceError("Bootstrap signal is not canonical float hex")
            try:
                number = float.fromhex(value)
            except ValueError as exc:
                raise FactorGovernanceError("Bootstrap signal is not canonical float hex") from exc
            if not math.isfinite(number) or number.hex() != value:
                raise FactorGovernanceError("Bootstrap signal is not finite canonical float hex")
            finite_values.append(value)
        finite_count = len(finite_values)
        distinct_finite_count = len(set(finite_values))
        if finite_count <= 0:
            raise FactorGovernanceError("Bootstrap active signal is empty or all null")
        if distinct_finite_count <= 1:
            raise FactorGovernanceError("Bootstrap active signal is constant")
        implementation_sha = require_sha256(
            implementation_sha256s.get(factor_id),
            label=f"{factor_id}.implementation_sha256",
        )
        rows.append(
            {
                "coverage_denominator": len(cohort),
                "coverage_numerator": finite_count,
                "coverage_rate": f"{finite_count / len(cohort):.12f}",
                "distinct_finite_count": distinct_finite_count,
                "factor_id": factor_id,
                "finite_count": finite_count,
                "finite_signal_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        [
                            {"symbol": symbol, "signal": values[symbol]}
                            for symbol in symbols
                            if values[symbol] is not None
                        ]
                    )
                ).hexdigest(),
                "full_signal_sha256": signal_hashes[factor_id],
                "implementation_sha256": implementation_sha,
                "nonfinite_count": 0,
                "output_symbol_count": len(symbols),
                "pit_eligible_symbol_count": len(cohort),
                "sealed_pit_eligible_cohort": cohort,
                "signal_sha256": signal_hashes[factor_id],
                "signal_symbol_set_sha256": cohort_sha,
                "source_bundle_sha256": require_sha256(
                    source_bundle_sha256, label="source_bundle_sha256"
                ),
                "source_symbol_count": len(symbols),
            }
        )
    return rows


def _validate_bootstrap_json_documents(
    *,
    manifest: Mapping[str, Any],
    active: Mapping[str, Any],
    policy: Mapping[str, Any],
    source_refs: Mapping[str, Mapping[str, Any]],
    decoded: Mapping[str, Any],
    documents: Mapping[str, Any],
) -> None:
    if documents["decision_source"]["payload"]["relative_path"] != _BOOTSTRAP_DECISION_PATH:
        raise FactorGovernanceError("Bootstrap decision source path differs")
    implementation_rows = manifest["payload"]["implementation_rows"]
    expected_tree = {
        "domain": "myquant-bootstrap-implementation-tree-manifest",
        "implementation_rows": implementation_rows,
    }
    if documents["implementation"]["document"] != expected_tree:
        raise FactorGovernanceError("Bootstrap implementation tree manifest differs")
    implementation_sha = hashlib.sha256(documents["implementation"]["raw"]).hexdigest()
    if any(
        row["implementation_sha256"] != implementation_sha
        for row in policy["payload"]["factor_rows"]
    ):
        raise FactorGovernanceError("Bootstrap implementation raw SHA differs")
    normalized = {
        role: decoded[role]["binding"]["normalized_sha256"]
        for role in ("exchange_calendar", "market_history", "pit_universe")
    }
    weights = [
        {"factor_id": row["factor_id"], "weight": row["weight"]}
        for row in active["payload"]["factor_rows"]
    ]
    implementation_sha256s = {
        row["factor_id"]: row["implementation_sha256"] for row in policy["payload"]["factor_rows"]
    }
    signal_statistics = _signal_statistics(
        decoded["signals"],
        eligible_symbols=decoded["eligible_symbols"],
        implementation_sha256s=implementation_sha256s,
        source_bundle_sha256=policy["payload"]["source_refs"][
            [row["role"] for row in policy["payload"]["source_refs"]].index("market")
        ]["ref"]["byte_sha256"],
    )
    expected_recomputation = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-recomputation",
        "factor_set_sha256": active["payload"]["factor_set_sha256"],
        "factor_weights": weights,
        "implementation_rows": implementation_rows,
        "normalized_source_sha256s": normalized,
        "result": "EXACT_MATCH",
        "signal_sha256s": _signal_hashes(decoded["signals"]),
        "signal_statistics": signal_statistics,
    }
    if documents["recomputation"]["document"] != expected_recomputation:
        raise FactorGovernanceError("Bootstrap recomputation document differs")
    source_rows = []
    decoder_role_by_outer = {
        "exchange_calendar": "exchange_calendar",
        "market": "market_history",
        "pit_universe": "pit_universe",
    }
    for role in ("exchange_calendar", "market", "pit_universe"):
        ref = dict(source_refs[role])
        source_rows.append(
            {
                "role": role,
                "source_ref": ref,
                "source_byte_sha256": decoded[decoder_role_by_outer[role]]["binding"][
                    "source_byte_sha256"
                ],
            }
        )
    source_rows.sort(key=lambda row: row["role"])
    source_generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": policy["payload"]["reader_contract"],
        "source_rows": source_rows,
    }
    expected_generation = {
        **source_generation_body,
        "generation_sha256": hashlib.sha256(
            canonical_json_bytes(source_generation_body)
        ).hexdigest(),
    }
    if documents["source_generation"]["document"] != expected_generation:
        raise FactorGovernanceError("Bootstrap source-generation document differs")


def _bootstrap_context_values(
    *,
    system_store: SystemStore,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_ref, receipt = _stored(
        system_store,
        request["intrinsic_receipt_ref"],
        label="intrinsic receipt",
        expected_kind="factor.validation_receipt",
    )
    receipt = validate_factor_validation_receipt(receipt)
    policy_ref, policy = _stored(
        system_store,
        receipt["payload"]["policy_ref"],
        label="Bootstrap policy",
        expected_kind="factor.bootstrap_exception_evidence",
    )
    active_ref, active = _stored(
        system_store,
        receipt["payload"]["active_set_ref"],
        label="Bootstrap active set",
        expected_kind="factor.bootstrap_set",
    )
    policy = validate_bootstrap_exception_evidence(policy)
    active = validate_bootstrap_factor_set(active)
    evidence: list[dict[str, Any]] = []
    for index, ref in enumerate(receipt["payload"]["evidence_refs"]):
        _, artifact = _stored(system_store, ref, label=f"receipt evidence[{index}]")
        evidence.append(artifact)
    replayed_receipt = _build_factor_validation_receipt(
        policy=policy,
        active_set=active,
        evidence_artifacts=evidence,
        trusted_at=receipt["created_at"],
    )
    if replayed_receipt != receipt:
        raise FactorGovernanceError("Bootstrap intrinsic receipt does not replay")
    release_ref = validate_artifact_ref(
        request["release_manifest_ref"],
        label="release_manifest_ref",
        expected_kind="system.release",
    )
    if release_ref not in receipt["payload"]["evidence_refs"]:
        raise FactorGovernanceError("Bootstrap receipt does not bind request release")
    manifest_ref, manifest = _stored(
        system_store,
        request["factor_validator_manifest_ref"],
        label="Factor validator manifest",
        expected_kind="factor.validator_manifest",
    )
    manifest = validate_validator_manifest(manifest)
    if manifest["payload"]["release_manifest_ref"] != release_ref:
        raise FactorGovernanceError("Factor validator manifest release differs")
    bundles, source_refs_by_role, decoded, documents = _bootstrap_sources(system_store, policy)
    decision = documents["decision_source"]["raw"]
    rebuilt_policy = build_bootstrap_exception_evidence(
        decision_source_bytes=decision,
        source_artifacts={
            "code": system_store.get_object(release_ref),
            **bundles,
        },
        implementation_source_sha256=hashlib.sha256(documents["implementation"]["raw"]).hexdigest(),
        created_at=policy["created_at"],
    )
    if rebuilt_policy != policy:
        raise FactorGovernanceError("Bootstrap policy raw replay differs")
    _validate_bootstrap_json_documents(
        manifest=manifest,
        active=active,
        policy=policy,
        source_refs=source_refs_by_role,
        decoded=decoded,
        documents=documents,
    )
    implementation_refs = _sorted_refs(
        [row["implementation_component_ref"] for row in manifest["payload"]["implementation_rows"]]
    )
    source_refs = _sorted_refs(list(source_refs_by_role.values()))
    return _context_payload(
        {
            "validation_namespace_id": request["validation_namespace_id"],
            "lane": "BOOTSTRAP",
            "intrinsic_receipt_ref": receipt_ref,
            "policy_ref": policy_ref,
            "evidence_refs": list(receipt["payload"]["evidence_refs"]),
            "active_set_ref": active_ref,
            "composite_state_ref": None,
            "factor_validator_manifest_ref": manifest_ref,
            "contextual_validator_component_ref": manifest["payload"][
                "contextual_validator_component_ref"
            ],
            "source_decoder_component_ref": manifest["payload"]["source_decoder_component_ref"],
            "implementation_component_refs": implementation_refs,
            "source_attestation_refs": [],
            "source_object_refs": source_refs,
            "custody_record_refs": [],
            "custody_tree_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
            "custody_head_ref": None,
            "validated": True,
            "blockers": [],
            "authority": "NON_AUTHORIZING",
        }
    )


def validate_bootstrap_contextual_run(
    *,
    system_store: SystemStore,
    validation_request: dict[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    """Replay the complete Bootstrap exception closure and return payload only."""

    canonical_timestamp(trusted_at, label="trusted_at")
    request = _request_payload(
        validation_request,
        expected_profile=BOOTSTRAP_VALIDATION_PROFILE,
    )
    if request["candidate_state_ref"] is not None:
        raise FactorGovernanceError("Bootstrap contextual request cannot bind candidate state")
    return _bootstrap_context_values(system_store=system_store, request=request)


def _prospective_manifest_closure(
    *,
    system_store: SystemStore,
    request: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    manifest_ref, manifest = _stored(
        system_store,
        request["factor_validator_manifest_ref"],
        label="Factor validator manifest",
        expected_kind="factor.validator_manifest",
    )
    manifest = validate_validator_manifest(manifest)
    payload = manifest["payload"]
    if payload["release_manifest_ref"] != request["release_manifest_ref"]:
        raise FactorGovernanceError("prospective Factor manifest release differs")

    _, contextual = _stored(
        system_store,
        payload["contextual_validator_component_ref"],
        label="prospective contextual component",
        expected_kind="system.installed_component_manifest",
    )
    _, decoder = _stored(
        system_store,
        payload["source_decoder_component_ref"],
        label="prospective source decoder component",
        expected_kind="system.installed_component_manifest",
    )
    try:
        contextual = validate_installed_component_manifest(contextual)
        decoder = validate_installed_component_manifest(decoder)
    except SystemError as exc:
        raise FactorGovernanceError(
            "prospective validator component identity differs",
            code="IMPLEMENTATION_IDENTITY_MISMATCH",
        ) from exc
    if (
        contextual["payload"]["component_role"] != "CONTEXTUAL_VALIDATOR"
        or contextual["payload"]["component_id"] != f"{PROSPECTIVE_VALIDATION_PROFILE}-component"
        or decoder["payload"]["component_role"] != "SOURCE_DECODER"
        or decoder["payload"]["allowed_source_formats"] != ["PARQUET"]
        or decoder["payload"]["fallback_allowed"] is not False
    ):
        raise FactorGovernanceError(
            "prospective validator component roles differ",
            code="IMPLEMENTATION_IDENTITY_MISMATCH",
        )

    implementations: list[dict[str, Any]] = []
    for index, row in enumerate(payload["implementation_rows"]):
        _, component = _stored(
            system_store,
            row["implementation_component_ref"],
            label=f"implementation component[{index}]",
            expected_kind="system.installed_component_manifest",
        )
        try:
            component = validate_installed_component_manifest(component)
        except SystemError as exc:
            raise FactorGovernanceError(
                "installed implementation component identity differs",
                code="IMPLEMENTATION_IDENTITY_MISMATCH",
            ) from exc
        expected_entrypoint = {
            "module_name": row["module_name"],
            "qualified_name": row["qualified_name"],
            "code_sha256": row["code_sha256"],
        }
        if (
            component["payload"]["component_role"] != "SOURCE_IMPLEMENTATION"
            or component["payload"]["component_id"] != row["implementation_id"]
            or component["payload"]["release_manifest_ref"] != request["release_manifest_ref"]
            or component["payload"]["entrypoints"] != [expected_entrypoint]
            or component["payload"]["allowed_source_formats"] != []
            or component["payload"]["fallback_allowed"] is not False
        ):
            raise FactorGovernanceError(
                "installed implementation component differs from Factor manifest",
                code="IMPLEMENTATION_IDENTITY_MISMATCH",
            )
        implementations.append(component)
    return manifest_ref, manifest, contextual, decoder, implementations


def _prospective_intrinsic_closure(
    *,
    system_store: SystemStore,
    request: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    list[dict[str, Any]],
]:
    receipt_ref, receipt = _stored(
        system_store,
        request["intrinsic_receipt_ref"],
        label="prospective intrinsic receipt",
        expected_kind="factor.validation_receipt",
    )
    receipt = validate_factor_validation_receipt(receipt)
    policy_ref, preregistration = _stored(
        system_store,
        receipt["payload"]["policy_ref"],
        label="prospective preregistration",
        expected_kind="factor.preregistration",
    )
    preregistration = validate_preregistration(preregistration)
    active_ref, active = _stored(
        system_store,
        receipt["payload"]["active_set_ref"],
        label="prospective admitted set",
        expected_kind="factor.admitted_set",
    )
    evidence: list[dict[str, Any]] = []
    for index, ref in enumerate(receipt["payload"]["evidence_refs"]):
        _, artifact = _stored(
            system_store,
            ref,
            label=f"prospective receipt evidence[{index}]",
        )
        evidence.append(artifact)
    replayed_receipt = _build_factor_validation_receipt(
        policy=preregistration,
        active_set=active,
        evidence_artifacts=evidence,
        trusted_at=receipt["created_at"],
    )
    candidate_payload = candidate["payload"]
    if (
        replayed_receipt != receipt
        or candidate_payload["intrinsic_receipt_ref"] != receipt_ref
        or candidate_payload["preregistration_ref"] != policy_ref
        or candidate_payload["admitted_set_ref"] != active_ref
    ):
        raise FactorGovernanceError(
            "prospective intrinsic closure differs from final composite",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    return (
        receipt_ref,
        receipt,
        policy_ref,
        preregistration,
        active_ref,
        active,
        evidence,
    )


def _replay_preregistration_sources(
    *,
    factor_store: FactorValidationStore,
    preregistration: Mapping[str, Any],
    manifest_ref: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contextual: Mapping[str, Any],
    decoder: Mapping[str, Any],
    implementations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _, attestation = factor_store._resolve(
        preregistration["payload"]["source_decode_attestation_ref"],
        label="preregistration source attestation",
        expected_kind="factor.source_decode_attestation",
    )
    attestation = validate_source_decode_attestation(attestation)
    payload = attestation["payload"]
    bindings = {row["role"]: row["source_object_ref"] for row in payload["source_bindings"]}
    if (
        payload["purpose"] != "PREREGISTRATION"
        or any(
            payload[field] is not None
            for field in (
                "preregistration_id",
                "selection_id",
                "ordinal",
                "signal_session",
                "maturity_session",
            )
        )
        or set(bindings) != {"exchange_calendar", "implementation_manifest"}
        or attestation["created_at"] != preregistration["created_at"]
    ):
        raise FactorGovernanceError(
            "preregistration source-attestation identity differs",
            code="SOURCE_VALIDATION_FAILED",
        )

    decoded: dict[str, DecodedSource] = {}

    def project_calendar(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        sessions, windows = factor_store._calendar_projection(
            table,
            trusted_at=preregistration["created_at"],
        )
        return {"sessions": sessions, "windows": windows}

    calendar = decode_source_role(
        system_store=factor_store._system_store,
        source_object_ref=bindings["exchange_calendar"],
        role="exchange_calendar",
        projector=project_calendar,
    )
    sessions = calendar.projection["sessions"]
    windows = calendar.projection["windows"]
    decoded["exchange_calendar"] = factor_store._decoded_summary(calendar)

    implementation = decode_source_role(
        system_store=factor_store._system_store,
        source_object_ref=bindings["implementation_manifest"],
        role="implementation_manifest",
        projector=lambda table, binding: {
            "rows": factor_store._implementation_manifest_projection(
                table,
                factor_manifest=manifest,
            )
        },
    )
    implementation_rows = implementation.projection["rows"]
    decoded["implementation_manifest"] = factor_store._decoded_summary(implementation)

    rebuilt_attestation = build_source_decode_attestation(
        purpose="PREREGISTRATION",
        preregistration_id=None,
        selection_id=None,
        ordinal=None,
        signal_session=None,
        maturity_session=None,
        decoded_sources=decoded,
        factor_validator_manifest=manifest,
        contextual_validator_component=contextual,
        source_decoder_component=decoder,
        implementation_components=implementations,
        trusted_at=attestation["created_at"],
    )
    candidates = [factor_store._candidate_from_manifest_row(row) for row in implementation_rows]
    rebuilt_preregistration = _build_preregistration(
        open_sessions=sessions,
        session_windows=windows,
        candidates=candidates,
        exchange_calendar_ref=bindings["exchange_calendar"],
        implementation_manifest_ref=bindings["implementation_manifest"],
        source_decode_attestation_ref=artifact_ref(attestation),
        factor_validator_manifest_ref=manifest_ref,
        trusted_at=preregistration["created_at"],
    )
    if rebuilt_attestation != attestation or rebuilt_preregistration != preregistration:
        raise FactorGovernanceError(
            "preregistration does not replay from its exact raw sources",
            code="SOURCE_VALIDATION_FAILED",
        )
    return attestation


def _prospective_stage_closure(
    *,
    factor_store: FactorValidationStore,
    candidate: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    _, selection = factor_store._resolve(
        candidate["payload"]["selection_ref"],
        label="prospective configuration selection",
        expected_kind="factor.configuration_selection",
    )
    selection = validate_configuration_selection(
        selection,
        preregistration=preregistration,
    )
    captures, observations = factor_store._resolve_prospective_stage_closure(
        composite=candidate,
        preregistration=preregistration,
        selection=selection,
    )
    _, execution = factor_store._resolve(
        candidate["payload"]["execution_evidence_ref"],
        label="prospective execution evidence",
        expected_kind="factor.execution_turnover_evidence",
    )
    execution = validate_execution_turnover_evidence(
        execution,
        preregistration=preregistration,
        selection=selection,
        signal_captures=captures,
        observations=observations,
    )
    configuration_rows, blockers = factor_store._execution_configuration_rows(
        preregistration=preregistration,
        selection=selection,
        captures=captures,
        observations=observations,
        manifest=manifest,
    )
    if (
        execution["payload"]["configuration_rows"] != configuration_rows
        or execution["payload"]["execution_state"] != "COMPLETE"
        or execution["payload"]["blockers"] != []
        or blockers
    ):
        raise FactorGovernanceError(
            "execution evidence does not replay from exact captured weights and labels",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    return selection, captures, observations, execution


def _prospective_source_closure(
    *,
    factor_store: FactorValidationStore,
    replay_source_refs: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    captures: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    artifacts = factor_store._source_attestation_artifacts(
        preregistration=preregistration,
        captures=captures,
        observations=observations,
    )
    attestation_refs = _sorted_refs([artifact_ref(value) for value in artifacts])
    if attestation_refs != list(replay_source_refs):
        raise FactorGovernanceError(
            "custody and prospective source-attestation closures differ",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    source_refs: list[dict[str, Any]] = []
    for artifact in artifacts:
        source_refs.extend(
            row["source_object_ref"] for row in artifact["payload"]["source_bindings"]
        )
    canonical_sources = _sorted_refs(source_refs)
    if len(canonical_sources) != 1_442:
        raise FactorGovernanceError(
            "prospective raw source-object closure is not exactly 1,442 objects",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    return attestation_refs, canonical_sources


def _prospective_context_values(
    *,
    system_store: SystemStore,
    request: Mapping[str, Any],
    candidate_ref: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    replay = replay_custody_chain(system_store=system_store, final_composite=candidate)
    if (
        replay.final_composite_ref != candidate_ref
        or replay.final_composite["payload"]["cycle_state"] != "INTRINSIC_VALIDATED"
        or replay.final_composite["payload"]["terminal"] is not True
        or replay.final_composite["payload"]["blockers"] != []
        or replay.transaction_count != 725
        or len(replay.custody_record_refs) != 726
        or len(replay.source_attestation_refs) != 721
        or len(replay.stage_slots) != 720
    ):
        raise FactorGovernanceError(
            "prospective custody closure is incomplete",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    (
        receipt_ref,
        receipt,
        policy_ref,
        preregistration,
        active_ref,
        _,
        _,
    ) = _prospective_intrinsic_closure(
        system_store=system_store,
        request=request,
        candidate=candidate,
    )
    (
        manifest_ref,
        manifest,
        contextual,
        decoder,
        implementations,
    ) = _prospective_manifest_closure(system_store=system_store, request=request)
    expected_namespace = prospective_validation_namespace_id(
        exchange_calendar_ref=preregistration["payload"]["exchange_calendar_ref"],
        implementation_manifest_ref=preregistration["payload"]["implementation_manifest_ref"],
        factor_validator_manifest_ref=manifest_ref,
    )
    if (
        request["validation_namespace_id"] != expected_namespace
        or candidate["payload"]["custody_namespace_id"] != expected_namespace
        or preregistration["payload"]["factor_validator_manifest_ref"] != manifest_ref
        or trusted_at < candidate["payload"]["last_stored_at"]
        or trusted_at < receipt["created_at"]
    ):
        raise FactorGovernanceError(
            "prospective validation namespace or trusted time differs",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    try:
        current = system_store.read_candidate_state(expected_namespace)
    except SystemError as exc:
        raise FactorGovernanceError(
            "prospective candidate pointer cannot be read",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        ) from exc
    if current is None or current["candidate_state_ref"] != candidate_ref:
        raise FactorGovernanceError(
            "prospective contextual run is not bound to the current candidate",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )

    factor_store = FactorValidationStore(system_store=system_store)
    factor_store._slot_cache[_ref_key(candidate_ref)] = tuple(
        dict(row) for row in replay.stage_slots
    )
    _replay_preregistration_sources(
        factor_store=factor_store,
        preregistration=preregistration,
        manifest_ref=manifest_ref,
        manifest=manifest,
        contextual=contextual,
        decoder=decoder,
        implementations=implementations,
    )
    selection, captures, observations, _ = _prospective_stage_closure(
        factor_store=factor_store,
        candidate=candidate,
        preregistration=preregistration,
        manifest=manifest,
    )
    if candidate["payload"]["selection_ref"] != artifact_ref(selection):
        raise FactorGovernanceError(
            "prospective selection ref differs after raw replay",
            code="CONTEXT_CLOSURE_INCOMPLETE",
        )
    source_attestation_refs, source_object_refs = _prospective_source_closure(
        factor_store=factor_store,
        replay_source_refs=replay.source_attestation_refs,
        preregistration=preregistration,
        captures=captures,
        observations=observations,
    )
    implementation_refs = _sorted_refs(
        [row["implementation_component_ref"] for row in manifest["payload"]["implementation_rows"]]
    )
    return _context_payload(
        {
            "validation_namespace_id": expected_namespace,
            "lane": "PROSPECTIVE",
            "intrinsic_receipt_ref": receipt_ref,
            "policy_ref": policy_ref,
            "evidence_refs": list(receipt["payload"]["evidence_refs"]),
            "active_set_ref": active_ref,
            "composite_state_ref": dict(candidate_ref),
            "factor_validator_manifest_ref": manifest_ref,
            "contextual_validator_component_ref": manifest["payload"][
                "contextual_validator_component_ref"
            ],
            "source_decoder_component_ref": manifest["payload"]["source_decoder_component_ref"],
            "implementation_component_refs": implementation_refs,
            "source_attestation_refs": source_attestation_refs,
            "source_object_refs": source_object_refs,
            "custody_record_refs": list(replay.custody_record_refs),
            "custody_tree_sha256": replay.custody_tree_sha256,
            "custody_head_ref": replay.custody_head_ref,
            "validated": True,
            "blockers": [],
            "authority": "NON_AUTHORIZING",
        }
    )


def validate_prospective_contextual_run(
    *,
    system_store: SystemStore,
    validation_request: dict[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    """Replay the full prospective raw, custody, policy, and receipt closure."""

    stamp = canonical_timestamp(trusted_at, label="trusted_at")
    request = _request_payload(
        validation_request,
        expected_profile=PROSPECTIVE_VALIDATION_PROFILE,
    )
    candidate_ref = request["candidate_state_ref"]
    if candidate_ref is None:
        raise FactorGovernanceError("Prospective contextual request lacks candidate state")
    normalized_candidate_ref, candidate = _stored(
        system_store,
        candidate_ref,
        label="candidate composite state",
        expected_kind="factor.composite_state",
    )
    return _prospective_context_values(
        system_store=system_store,
        request=request,
        candidate_ref=normalized_candidate_ref,
        candidate=candidate,
        trusted_at=stamp,
    )


__all__ = [
    "CONTEXTUAL_RESULT_KIND",
    "validate_bootstrap_contextual_run",
    "validate_contextual_result",
    "validate_prospective_contextual_run",
]
