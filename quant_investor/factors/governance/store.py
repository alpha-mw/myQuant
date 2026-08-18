"""Trusted Factor validation store over immutable System objects.

The public surface accepts only exact, already-stored object references.  It
owns timestamps and candidate-state transitions, but it never writes the
System active pointer and none of the artifacts produced here authorize
activation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import math
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_investor.contracts import canonical_json_bytes, get_contract
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    EMPTY_POINTER_SHA256,
    PROSPECTIVE_VALIDATION_PROFILE,
    SystemCASMismatch,
    SystemError,
    SystemStore,
    object_ref_for_artifact,
    validate_installed_component_manifest,
)

from .bootstrap import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    build_bootstrap_factor_set,
)
from .bootstrap_evidence import build_bootstrap_exception_evidence
from .admission import (
    _build_admitted_factor_set,
    _evaluate_preregistration,
    validate_admitted_factor_set,
    validate_preregistration_evaluation,
)
from .common import (
    artifact_ref,
    business_identity,
    canonical_timestamp,
    decimal_text,
    validate_artifact_ref,
)
from .custody import (
    build_composite_state,
    build_custody_record,
    build_stage_slot,
    custody_slot_tree_sha256,
    custody_transaction_id,
    operation_request_sha256,
    replay_custody_chain,
    validate_composite_state,
)
from .errors import FactorGovernanceError
from .execution import (
    _build_execution_turnover_evidence,
    validate_execution_turnover_evidence,
)
from .implementations import compute_installed_signals, installed_semantic_row
from .manifest import build_validator_manifest, validate_validator_manifest
from .prospective import (
    _build_configuration_selection,
    _build_observation,
    _build_preregistration,
    _build_signal_capture,
    _validate_configuration_selection_prevalidated,
    _validate_observation_prevalidated,
    _validate_signal_capture_prevalidated,
    validate_preregistration,
)
from .receipt import _build_factor_validation_receipt
from .source import (
    DecodedSource,
    build_source_decode_attestation,
    decode_source_role,
    validate_source_decode_attestation,
)
from .status import _build_factor_status, validate_factor_status

_MAXIMUM_JSON_SOURCE_BYTES: Final = 512 * 1024 * 1024
_FACTOR_VALIDATED_KINDS: Final = (
    "factor.admitted_set",
    "factor.bootstrap_exception_evidence",
    "factor.bootstrap_set",
    "factor.composite_state",
    "factor.configuration_selection",
    "factor.contextual_validation_result",
    "factor.custody_record",
    "factor.execution_turnover_evidence",
    "factor.preregistration",
    "factor.prospective_evaluation",
    "factor.prospective_observation",
    "factor.signal_capture",
    "factor.source_decode_attestation",
    "factor.status",
    "factor.validation_receipt",
    "factor.validator_manifest",
)
_REFERENCE_SORT_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)
_COVERAGE_MINIMUM: Final = Decimal("0.8")
_EXECUTION_COST_RATE: Final = Decimal("0.00005")


@dataclass(frozen=True)
class BootstrapClosure:
    """Exact inert Bootstrap roots produced by the trusted store."""

    policy_ref: dict[str, str]
    active_set_ref: dict[str, str]
    intrinsic_receipt_ref: dict[str, str]


@dataclass(frozen=True)
class _CandidateTransition:
    previous_ref: dict[str, str]
    previous: dict[str, Any]
    pointer_sha256: str
    operation_request_sha256: str
    transaction_id: str
    transaction_sequence: int


def _reference_key(value: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(value[field] for field in _REFERENCE_SORT_FIELDS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clock_stamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if type(value) is not datetime or value.utcoffset() is None:
        raise FactorGovernanceError("Factor validation clock is not timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_value(value: str) -> datetime:
    return datetime.strptime(
        canonical_timestamp(value, label="trusted timestamp"), "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)


def _sha256_document(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _finite_hex(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number.hex() if math.isfinite(number) else None


def _signal_values_sha256(
    *,
    configuration_id: str,
    factor_id: str,
    symbols: Sequence[str],
    values: Mapping[str, Any],
) -> str:
    return _sha256_document(
        {
            "domain": "myquant-factor-signal-values",
            "configuration_id": configuration_id,
            "factor_id": factor_id,
            "rows": [
                {"symbol": symbol, "signal_value": _finite_hex(values.get(symbol))}
                for symbol in symbols
            ],
        }
    )


def _normalized_input_sha256(
    *,
    candidate: Mapping[str, Any],
    market_normalized_sha256: str,
) -> str:
    return _sha256_document(
        {
            "domain": "myquant-factor-normalized-implementation-input",
            "configuration_id": candidate["configuration_id"],
            "factor_id": candidate["factor_id"],
            "input_fields": candidate["input_fields"],
            "market_history_normalized_sha256": market_normalized_sha256,
        }
    )


def _portfolio_weights_sha256(
    *,
    configuration_id: str,
    signal_session: str,
    weights: Mapping[str, Decimal],
) -> str:
    return _sha256_document(
        {
            "domain": "myquant-factor-sparse-portfolio-weights",
            "configuration_id": configuration_id,
            "signal_session": signal_session,
            "unlisted_universe_weight": "EXACT_ZERO",
            "rows": [
                {"symbol": symbol, "weight": decimal_text(weights[symbol])}
                for symbol in sorted(weights, key=lambda value: value.encode("utf-8"))
            ],
        }
    )


def _label_values_sha256(
    *,
    ordinal: int,
    signal_session: str,
    label_start_session: str,
    label_end_session: str,
    symbols: Sequence[str],
    values: Mapping[str, float | None],
) -> str:
    return _sha256_document(
        {
            "domain": "myquant-factor-matured-label-values",
            "ordinal": ordinal,
            "signal_session": signal_session,
            "label_start_session": label_start_session,
            "label_end_session": label_end_session,
            "formula": "adj_close[t+30]/adj_close[t+1]-1",
            "rows": [
                {"symbol": symbol, "label_value": _finite_hex(values.get(symbol))}
                for symbol in symbols
            ],
        }
    )


def _selection_rows(
    candidates: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coverage = {row["configuration_id"]: Decimal(row["selection_coverage"]) for row in summaries}
    primaries = {row["configuration_id"]: row for row in candidates if row["role"] == "PRIMARY"}
    alternates = {
        row["role"].split(":", 1)[1]: row
        for row in candidates
        if row["role"].startswith("ALTERNATE_FOR:")
    }
    rows: list[dict[str, Any]] = []
    for primary_id in sorted(primaries, key=lambda value: value.encode("utf-8")):
        candidate = primaries[primary_id]
        used_alternate = False
        if coverage[primary_id] < _COVERAGE_MINIMUM:
            candidate = alternates.get(primary_id)  # type: ignore[assignment]
            if candidate is None or coverage[candidate["configuration_id"]] < _COVERAGE_MINIMUM:
                raise FactorGovernanceError(
                    "primary and registered alternate fail initial coverage",
                    code="SIGNAL_COVERAGE_BELOW_MINIMUM",
                )
            used_alternate = True
        rows.append(
            {
                "primary_configuration_id": primary_id,
                "selected_configuration_id": candidate["configuration_id"],
                "selected_factor_id": candidate["factor_id"],
                "used_alternate": used_alternate,
            }
        )
    return rows


def prospective_validation_namespace_id(
    *,
    exchange_calendar_ref: Mapping[str, Any],
    implementation_manifest_ref: Mapping[str, Any],
    factor_validator_manifest_ref: Mapping[str, Any],
) -> str:
    """Derive the sole prospective namespace from immutable mining roots."""

    return business_identity(
        "factor-validation-namespace",
        {
            "validation_profile_id": PROSPECTIVE_VALIDATION_PROFILE,
            "exchange_calendar_ref": validate_artifact_ref(
                exchange_calendar_ref,
                label="exchange_calendar_ref",
                expected_kind="system.source_object",
            ),
            "implementation_manifest_ref": validate_artifact_ref(
                implementation_manifest_ref,
                label="implementation_manifest_ref",
                expected_kind="system.source_object",
            ),
            "factor_validator_manifest_ref": validate_artifact_ref(
                factor_validator_manifest_ref,
                label="factor_validator_manifest_ref",
                expected_kind="factor.validator_manifest",
            ),
        },
    )


def bootstrap_validation_namespace_id(*, intrinsic_receipt_ref: Mapping[str, Any]) -> str:
    """Derive the sole Bootstrap namespace from its intrinsic closure root."""

    return business_identity(
        "factor-validation-namespace",
        {
            "validation_profile_id": BOOTSTRAP_VALIDATION_PROFILE,
            "intrinsic_receipt_ref": validate_artifact_ref(
                intrinsic_receipt_ref,
                label="intrinsic_receipt_ref",
                expected_kind="factor.validation_receipt",
            ),
        },
    )


class FactorValidationStore:
    """Trusted writer for non-authorizing Factor validation artifacts."""

    def __init__(self, *, system_store: SystemStore) -> None:
        if not isinstance(system_store, SystemStore):
            raise FactorGovernanceError("system_store must be a SystemStore")
        self._system_store = system_store
        self._clock: Callable[[], datetime] = _utc_now
        self._slot_cache: dict[tuple[str, ...], tuple[dict[str, Any], ...]] = {}

    @classmethod
    def _for_testing(
        cls,
        *,
        system_store: SystemStore,
        clock: Callable[[], datetime],
    ) -> FactorValidationStore:
        """Private deterministic-clock seam used only by isolated tests."""

        if not callable(clock):
            raise FactorGovernanceError("test clock must be callable")
        instance = cls(system_store=system_store)
        instance._clock = clock
        _clock_stamp(instance._clock)
        return instance

    @classmethod
    def for_sealed_operation(
        cls,
        *,
        system_store: SystemStore,
        trusted_at: str,
    ) -> FactorValidationStore:
        """Bind one production operation to its already-sealed UTC timestamp."""

        instant = _timestamp_value(trusted_at)
        if instant > _utc_now():
            raise FactorGovernanceError("sealed operation timestamp may not be in the future")
        instance = cls(system_store=system_store)
        instance._clock = lambda: instant
        _clock_stamp(instance._clock)
        return instance

    def _trusted_at(self) -> str:
        return _clock_stamp(self._clock)

    def _resolve(
        self,
        value: Mapping[str, Any],
        *,
        label: str,
        expected_kind: str | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        try:
            ref = validate_artifact_ref(value, label=label, expected_kind=expected_kind)
            artifact = self._system_store.get_object(ref)
        except FactorGovernanceError:
            raise
        except SystemError as exc:
            raise FactorGovernanceError(
                f"{label} cannot be resolved",
                code="FACTOR_STORAGE_RESOLUTION_FAILED",
            ) from exc
        if object_ref_for_artifact(artifact) != ref:
            raise FactorGovernanceError(
                f"{label} exact object binding differs",
                code="FACTOR_STORAGE_RESOLUTION_FAILED",
            )
        return ref, artifact

    def _put(self, artifact: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        try:
            ref = self._system_store.put_object(artifact)
            readback = self._system_store.get_object(ref)
        except SystemError as exc:
            raise FactorGovernanceError(
                "Factor immutable object publication failed",
                code="FACTOR_STORAGE_WRITE_FAILED",
            ) from exc
        if readback != artifact or object_ref_for_artifact(readback) != ref:
            raise FactorGovernanceError(
                "Factor immutable object readback differs",
                code="FACTOR_STORAGE_WRITE_FAILED",
            )
        return ref, readback

    def _manifest_components(
        self, manifest: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        payload = manifest["payload"]
        try:
            contextual = self._system_store.get_object(
                payload["contextual_validator_component_ref"]
            )
            decoder = self._system_store.get_object(payload["source_decoder_component_ref"])
            implementations = [
                self._system_store.get_object(row["implementation_component_ref"])
                for row in payload["implementation_rows"]
            ]
        except SystemError as exc:
            raise FactorGovernanceError(
                "Factor validator component closure cannot be resolved",
                code="IMPLEMENTATION_IDENTITY_MISMATCH",
            ) from exc
        return contextual, decoder, implementations

    def _transition_head(
        self,
        *,
        operation: str,
        input_refs: Mapping[str, Mapping[str, Any] | None],
        expected_composite_state_ref: Mapping[str, Any],
    ) -> tuple[_CandidateTransition | None, dict[str, Any] | None]:
        previous_ref, previous = self._resolve(
            expected_composite_state_ref,
            label="expected_composite_state_ref",
            expected_kind="factor.composite_state",
        )
        previous = validate_composite_state(previous)
        payload = previous["payload"]
        if payload["terminal"]:
            raise FactorGovernanceError(
                "candidate cycle is terminal",
                code="STAGE_SLOT_ALREADY_RESOLVED",
            )
        request_sha = operation_request_sha256(
            operation=operation,
            expected_composite_state_ref=previous_ref,
            input_refs=dict(input_refs),
        )
        sequence = payload["transaction_sequence"] + 1
        transaction_id = custody_transaction_id(
            custody_namespace_id=payload["custody_namespace_id"],
            transaction_sequence=sequence,
            previous_composite_state_ref=previous_ref,
            operation_request_sha256_value=request_sha,
        )
        try:
            current = self._system_store.read_candidate_state(payload["custody_namespace_id"])
        except SystemError as exc:
            raise FactorGovernanceError(
                "candidate state cannot be read",
                code="CANDIDATE_STATE_CONFLICT",
            ) from exc
        if current is None:
            raise FactorGovernanceError(
                "candidate state pointer is absent",
                code="CANDIDATE_STATE_CONFLICT",
            )
        if current["candidate_state_ref"] == previous_ref:
            return (
                _CandidateTransition(
                    previous_ref=previous_ref,
                    previous=previous,
                    pointer_sha256=current["pointer_byte_sha256"],
                    operation_request_sha256=request_sha,
                    transaction_id=transaction_id,
                    transaction_sequence=sequence,
                ),
                None,
            )
        observed = validate_composite_state(current["candidate_state"])
        if (
            observed["payload"]["previous_composite_state_ref"] == previous_ref
            and observed["payload"]["transaction_id"] == transaction_id
        ):
            return None, observed
        raise FactorGovernanceError(
            "candidate state differs from expected composite",
            code="CANDIDATE_STATE_CONFLICT",
        )

    def _begin_transition(
        self,
        transition: _CandidateTransition,
        *,
        operation: str,
        input_refs: Mapping[str, Mapping[str, Any] | None],
        transaction_record_count: int,
    ) -> str:
        plan = {
            "operation": operation,
            "operation_request_sha256": transition.operation_request_sha256,
            "input_refs": [
                {"role": role, "ref": input_refs[role]}
                for role in sorted(input_refs, key=lambda value: value.encode("utf-8"))
            ],
            "transaction_record_count": transaction_record_count,
        }
        try:
            intent = self._system_store.begin_candidate_transaction(
                transition.previous["payload"]["custody_namespace_id"],
                transition.transaction_id,
                expected_pointer_sha256=transition.pointer_sha256,
                transaction_plan=plan,
            )
        except SystemCASMismatch as exc:
            raise FactorGovernanceError(
                "candidate state changed before transaction begin",
                code="CANDIDATE_STATE_CONFLICT",
            ) from exc
        except SystemError as exc:
            code = "CLOCK_ROLLBACK" if exc.code == "SYSTEM_CLOCK_ROLLBACK" else None
            raise FactorGovernanceError(
                "candidate transaction intent could not be fixed",
                code=code or "CUSTODY_TRANSACTION_INCOMPLETE",
            ) from exc
        if intent["previous_candidate_state_ref"] != transition.previous_ref:
            raise FactorGovernanceError(
                "candidate transaction predecessor differs",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        return canonical_timestamp(intent["trusted_at"], label="candidate transaction trusted_at")

    def _stage_slots(self, composite: Mapping[str, Any]) -> list[dict[str, Any]]:
        ref = artifact_ref(composite)
        key = _reference_key(ref)
        cached = self._slot_cache.get(key)
        if cached is not None:
            rows = [dict(row) for row in cached]
            if custody_slot_tree_sha256(rows) == composite["payload"]["slot_tree_sha256"]:
                return rows
        replay = replay_custody_chain(system_store=self._system_store, final_composite=composite)
        rows = [dict(row) for row in replay.stage_slots]
        self._slot_cache[key] = tuple(dict(row) for row in rows)
        return rows

    def _commit_candidate(
        self,
        transition: _CandidateTransition,
        composite: Mapping[str, Any],
        *,
        stage_slots: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        composite_ref, _ = self._put(composite)
        namespace = transition.previous["payload"]["custody_namespace_id"]
        try:
            committed = self._system_store.compare_and_swap_candidate_state(
                namespace,
                composite_ref,
                expected_pointer_sha256=transition.pointer_sha256,
            )
        except SystemCASMismatch as exc:
            observed = self._system_store.read_candidate_state(namespace)
            if observed is not None and observed["candidate_state_ref"] == composite_ref:
                committed = observed
            else:
                raise FactorGovernanceError(
                    "candidate state changed concurrently",
                    code="CANDIDATE_STATE_CONFLICT",
                ) from exc
        except SystemError as exc:
            raise FactorGovernanceError(
                "candidate state publication failed",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            ) from exc
        candidate = validate_composite_state(committed["candidate_state"])
        self._slot_cache[_reference_key(artifact_ref(candidate))] = tuple(
            dict(row) for row in stage_slots
        )
        return candidate

    @staticmethod
    def _next_composite(
        transition: _CandidateTransition,
        *,
        custody_head_ref: Mapping[str, Any],
        custody_record_count: int,
        stage_slots: Sequence[Mapping[str, Any]],
        trusted_at: str,
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        previous = transition.previous["payload"]
        values: dict[str, Any] = {
            "selection_ref": previous["selection_ref"],
            "signal_capture_count": previous["signal_capture_count"],
            "signal_capture_head_ref": previous["signal_capture_head_ref"],
            "observation_count": previous["observation_count"],
            "observation_head_ref": previous["observation_head_ref"],
            "execution_evidence_ref": previous["execution_evidence_ref"],
            "evaluation_ref": previous["evaluation_ref"],
            "admitted_set_ref": previous["admitted_set_ref"],
            "intrinsic_receipt_ref": previous["intrinsic_receipt_ref"],
            "resolved_signal_slot_count": previous["resolved_signal_slot_count"],
            "resolved_label_slot_count": previous["resolved_label_slot_count"],
            "cycle_state": previous["cycle_state"],
            "terminal": previous["terminal"],
            "blockers": previous["blockers"],
        }
        values.update(dict(overrides))
        return build_composite_state(
            custody_namespace_id=previous["custody_namespace_id"],
            preregistration_ref=previous["preregistration_ref"],
            cycle_state=values["cycle_state"],
            transaction_sequence=transition.transaction_sequence,
            previous_composite_state_ref=transition.previous_ref,
            transaction_id=transition.transaction_id,
            custody_record_count=custody_record_count,
            custody_head_ref=custody_head_ref,
            selection_ref=values["selection_ref"],
            signal_capture_count=values["signal_capture_count"],
            signal_capture_head_ref=values["signal_capture_head_ref"],
            observation_count=values["observation_count"],
            observation_head_ref=values["observation_head_ref"],
            execution_evidence_ref=values["execution_evidence_ref"],
            evaluation_ref=values["evaluation_ref"],
            admitted_set_ref=values["admitted_set_ref"],
            intrinsic_receipt_ref=values["intrinsic_receipt_ref"],
            resolved_signal_slot_count=values["resolved_signal_slot_count"],
            resolved_label_slot_count=values["resolved_label_slot_count"],
            slot_tree_sha256=custody_slot_tree_sha256(stage_slots),
            terminal=values["terminal"],
            blockers=values["blockers"],
            last_stored_at=trusted_at,
        )

    @staticmethod
    def _validated_contract_rows() -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for kind in _FACTOR_VALIDATED_KINDS:
            definition = get_contract(kind)
            rows.append(
                {
                    "kind": definition.kind,
                    "contract_sha256": definition.contract_sha256,
                    "json_schema_sha256": definition.json_schema_sha256,
                    "validator_code_sha256": definition.validator_code_sha256,
                }
            )
        return rows

    def build_validator_manifest(
        self,
        *,
        release_manifest_ref: Mapping[str, Any],
        contextual_validator_component_ref: Mapping[str, Any],
        source_decoder_component_ref: Mapping[str, Any],
        implementation_component_refs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build and store one exact finite installed Factor manifest."""

        _, release = self._resolve(
            release_manifest_ref,
            label="release_manifest_ref",
            expected_kind="system.release",
        )
        _, contextual = self._resolve(
            contextual_validator_component_ref,
            label="contextual_validator_component_ref",
            expected_kind="system.installed_component_manifest",
        )
        _, decoder = self._resolve(
            source_decoder_component_ref,
            label="source_decoder_component_ref",
            expected_kind="system.installed_component_manifest",
        )
        validate_installed_component_manifest(contextual)
        validate_installed_component_manifest(decoder)
        if type(implementation_component_refs) is not dict or set(
            implementation_component_refs
        ) != {LOW_DOLLAR_VOLUME, BLEND_W80}:
            raise FactorGovernanceError("implementation component refs are not exact")
        implementations: dict[str, dict[str, Any]] = {}
        for factor_id in sorted(implementation_component_refs, key=lambda value: value.encode()):
            _, component = self._resolve(
                implementation_component_refs[factor_id],
                label=f"implementation_component_refs[{factor_id}]",
                expected_kind="system.installed_component_manifest",
            )
            normalized = validate_installed_component_manifest(component)
            payload = normalized["payload"]
            expected = installed_semantic_row(factor_id)
            if (
                payload["component_role"] != "SOURCE_IMPLEMENTATION"
                or payload["component_id"] != expected["implementation_id"]
                or payload["release_manifest_ref"] != artifact_ref(release)
                or len(payload["entrypoints"]) != 1
                or payload["entrypoints"][0]
                != {
                    "module_name": expected["module_name"],
                    "qualified_name": expected["qualified_name"],
                    "code_sha256": expected["code_sha256"],
                }
                or payload["allowed_source_formats"] != []
                or payload["fallback_allowed"] is not False
            ):
                raise FactorGovernanceError("installed implementation component differs")
            implementations[factor_id] = normalized
        manifest = build_validator_manifest(
            release_manifest=release,
            contextual_validator_component=contextual,
            source_decoder_component=decoder,
            implementation_components=implementations,
            validated_contracts=self._validated_contract_rows(),
            trusted_at=self._trusted_at(),
        )
        _, stored = self._put(manifest)
        return validate_validator_manifest(stored)

    @staticmethod
    def _bundle_source_ref(
        bundle: Mapping[str, Any],
        *,
        outer_role: str,
        inner_role: str,
    ) -> dict[str, str]:
        payload = bundle.get("payload")
        rows = payload.get("sources") if type(payload) is dict else None
        if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
            raise FactorGovernanceError(f"Bootstrap {outer_role} bundle is not exact")
        row = rows[0]
        if set(row) != {"role", "source_ref"} or row["role"] != inner_role:
            raise FactorGovernanceError(f"Bootstrap {outer_role} inner role differs")
        return validate_artifact_ref(
            row["source_ref"],
            label=f"{outer_role}.{inner_role}",
            expected_kind="system.source_object",
        )

    def initialize_bootstrap(
        self,
        *,
        release_ref: Mapping[str, Any],
        decision_source_bundle_ref: Mapping[str, Any],
        exchange_calendar_bundle_ref: Mapping[str, Any],
        implementation_bundle_ref: Mapping[str, Any],
        market_bundle_ref: Mapping[str, Any],
        pit_universe_bundle_ref: Mapping[str, Any],
        recomputation_bundle_ref: Mapping[str, Any],
        source_generation_bundle_ref: Mapping[str, Any],
    ) -> BootstrapClosure:
        """Build the exact inert Bootstrap policy/set/receipt closure."""

        _, release = self._resolve(
            release_ref,
            label="release_ref",
            expected_kind="system.release",
        )
        bundle_inputs = {
            "decision_source": (decision_source_bundle_ref, "bootstrap_decision"),
            "exchange_calendar": (exchange_calendar_bundle_ref, "calendar"),
            "implementation": (implementation_bundle_ref, "implementation_tree_manifest"),
            "market": (market_bundle_ref, "market"),
            "pit_universe": (pit_universe_bundle_ref, "pit"),
            "recomputation": (recomputation_bundle_ref, "recomputation"),
            "source_generation": (source_generation_bundle_ref, "source_generation"),
        }
        bundles: dict[str, dict[str, Any]] = {}
        source_refs: dict[str, dict[str, str]] = {}
        for role, (value, inner_role) in bundle_inputs.items():
            _, bundle = self._resolve(
                value,
                label=f"{role}_bundle_ref",
                expected_kind="system.source_bundle",
            )
            bundles[role] = bundle
            source_refs[role] = self._bundle_source_ref(
                bundle,
                outer_role=role,
                inner_role=inner_role,
            )
        try:
            decision_payload, decision_bytes = self._system_store.read_source_object_bytes(
                source_refs["decision_source"],
                maximum_bytes=_MAXIMUM_JSON_SOURCE_BYTES,
            )
        except SystemError as exc:
            raise FactorGovernanceError(
                "Bootstrap decision source cannot be read",
                code="SOURCE_VALIDATION_FAILED",
            ) from exc
        if (
            decision_payload["source_format"] != "JSON"
            or decision_payload["media_type"] != "application/json"
        ):
            raise FactorGovernanceError("Bootstrap decision source format differs")
        _, implementation_object = self._resolve(
            source_refs["implementation"],
            label="implementation_tree_manifest_ref",
            expected_kind="system.source_object",
        )
        trusted_at = self._trusted_at()
        source_artifacts = {"code": release, **bundles}
        evidence = build_bootstrap_exception_evidence(
            decision_source_bytes=decision_bytes,
            source_artifacts=source_artifacts,
            implementation_source_sha256=implementation_object["payload"]["byte_sha256"],
            created_at=trusted_at,
        )
        evidence_ref, evidence = self._put(evidence)
        active_set = build_bootstrap_factor_set(
            bootstrap_exception_evidence=evidence,
            created_at=trusted_at,
        )
        active_ref, active_set = self._put(active_set)
        receipt = _build_factor_validation_receipt(
            policy=evidence,
            active_set=active_set,
            evidence_artifacts=[release, *bundles.values()],
            trusted_at=trusted_at,
        )
        receipt_ref, _ = self._put(receipt)
        return BootstrapClosure(
            policy_ref=evidence_ref,
            active_set_ref=active_ref,
            intrinsic_receipt_ref=receipt_ref,
        )

    @staticmethod
    def _candidate_from_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "configuration_id": row["factor_id"],
            "factor_id": row["factor_id"],
            "implementation_id": row["implementation_id"],
            "implementation_component_ref": dict(row["implementation_component_ref"]),
            "implementation_sha256": row["code_sha256"],
            "family": row["family"],
            "primitive": row["primitive"],
            "direction": row["direction"],
            "formula": row["formula"],
            "normalized_expression": row["normalized_expression"],
            "parameters_json": row["parameters_json"],
            "input_fields": list(row["input_fields"]),
            "role": "PRIMARY",
        }
        return {"candidate_spec_id": business_identity("factor-candidate-spec", body), **body}

    @staticmethod
    def _calendar_projection(
        table: Any, *, trusted_at: str
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if table is None:
            raise FactorGovernanceError("decoded exchange calendar is absent")
        rows = table.to_pylist()
        now = _timestamp_value(trusted_at)
        first = next(
            (index for index, row in enumerate(rows) if row["opens_at_utc"] > now),
            None,
        )
        if first is None or len(rows) - first < 391:
            raise FactorGovernanceError(
                "calendar cannot provide 390 planned sessions and final deadline",
                code="PREREGISTRATION_WINDOW_UNAVAILABLE",
            )
        selected = rows[first : first + 391]  # noqa: E203
        sessions = [row["open_session"].isoformat() for row in selected[:390]]
        windows = [
            {
                "ordinal": ordinal,
                "open_session": row["open_session"].isoformat(),
                "opens_at_utc": row["opens_at_utc"]
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "closes_at_utc": row["closes_at_utc"]
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "next_opens_at_utc": selected[ordinal + 1]["opens_at_utc"]
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            for ordinal, row in enumerate(selected[:390])
        ]
        return sessions, windows

    @staticmethod
    def _implementation_manifest_projection(
        table: Any,
        *,
        factor_manifest: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if table is None:
            raise FactorGovernanceError("decoded implementation manifest is absent")
        source_rows = table.to_pylist()
        observed: list[dict[str, Any]] = []
        for row in source_rows:
            observed.append(
                {
                    "factor_id": row["factor_id"],
                    "implementation_id": row["implementation_id"],
                    "implementation_component_ref": {
                        "kind": row["implementation_component_kind"],
                        "contract_sha256": row["implementation_component_contract_sha256"],
                        "artifact_id": row["implementation_component_artifact_id"],
                        "semantic_sha256": row["implementation_component_semantic_sha256"],
                        "byte_sha256": row["implementation_component_byte_sha256"],
                    },
                    "module_name": row["module_name"],
                    "qualified_name": row["qualified_name"],
                    "code_sha256": row["code_sha256"],
                    "family": row["family"],
                    "primitive": row["primitive"],
                    "direction": row["direction"],
                    "formula": row["formula"],
                    "normalized_expression": row["normalized_expression"],
                    "parameters_json": row["parameters_json"],
                    "input_fields": list(row["input_fields"]),
                    "required_source_roles": list(row["required_source_roles"]),
                }
            )
        if observed != factor_manifest["payload"]["implementation_rows"]:
            raise FactorGovernanceError(
                "implementation source differs from Factor validator manifest",
                code="IMPLEMENTATION_IDENTITY_MISMATCH",
            )
        return observed

    @staticmethod
    def _decoded_summary(decoded: DecodedSource) -> DecodedSource:
        return DecodedSource(
            role=decoded.role,
            source_object_ref=dict(decoded.source_object_ref),
            projection=None,
            binding=dict(decoded.binding),
        )

    @staticmethod
    def _signal_window_state(
        preregistration: Mapping[str, Any],
        ordinal: int,
        trusted_at: str,
    ) -> str:
        window = preregistration["payload"]["session_windows"][ordinal]
        now = _timestamp_value(trusted_at)
        if now < _timestamp_value(window["closes_at_utc"]):
            return "NOT_OPEN"
        if now >= _timestamp_value(window["next_opens_at_utc"]):
            return "MISSED"
        return "OPEN"

    @staticmethod
    def _label_window_state(
        preregistration: Mapping[str, Any],
        ordinal: int,
        trusted_at: str,
    ) -> str:
        window = preregistration["payload"]["session_windows"][ordinal + 30]
        now = _timestamp_value(trusted_at)
        if now < _timestamp_value(window["closes_at_utc"]):
            return "NOT_OPEN"
        if now >= _timestamp_value(window["next_opens_at_utc"]):
            return "MISSED"
        return "OPEN"

    @staticmethod
    def _expected_signal_transaction(ordinal: int) -> int:
        return ordinal + 2 if ordinal < 30 else 2 * ordinal - 28

    @staticmethod
    def _expected_label_transaction(ordinal: int) -> int:
        return 2 * ordinal + 33 if ordinal < 330 else ordinal + 362

    @staticmethod
    def _latest_input_completeness(
        frames: Mapping[str, pd.DataFrame],
        *,
        symbols: Sequence[str],
        input_fields: Sequence[str],
        signal_session: str,
    ) -> dict[str, bool]:
        target = datetime.strptime(signal_session, "%Y-%m-%d").date()
        result: dict[str, bool] = {}
        for symbol in symbols:
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                result[symbol] = False
                continue
            ordered = frame.sort_values("trade_date", kind="mergesort")
            latest = ordered.iloc[-1]
            trade_date = pd.Timestamp(latest["trade_date"]).date()
            result[symbol] = trade_date == target and all(
                _finite_hex(latest.get(field)) is not None for field in input_fields
            )
        return result

    def _decode_signal_projection(  # noqa: C901
        self,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any] | None,
        ordinal: int,
        pit_universe_ref: Mapping[str, Any],
        market_history_ref: Mapping[str, Any],
        sparse_weights_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        signal_session = preregistration["payload"]["signal_sessions"][ordinal]
        candidates = preregistration["payload"]["candidates"]
        decoded: dict[str, DecodedSource] = {}

        pit = decode_source_role(
            system_store=self._system_store,
            source_object_ref=pit_universe_ref,
            role="pit_universe",
            projector=lambda table, binding: {
                "rows": [
                    {
                        "symbol": row["symbol"],
                        "industry": row["industry"],
                        "total_mv": row["total_mv"],
                        "tradable": row["tradable"],
                    }
                    for row in table.to_pylist()
                ]
            },
        )
        if (
            pit.binding["minimum_session"] != signal_session
            or pit.binding["maximum_session"] != signal_session
        ):
            raise FactorGovernanceError(
                "PIT universe does not bind the signal session",
                code="SOURCE_SESSION_AFTER_SIGNAL_CUTOFF",
            )
        pit_rows = pit.projection["rows"]
        symbols = [row["symbol"] for row in pit_rows]
        pit_by_symbol = {row["symbol"]: row for row in pit_rows}
        pit_sha256 = pit.binding["normalized_sha256"]
        decoded["pit_universe"] = self._decoded_summary(pit)
        if not symbols:
            raise FactorGovernanceError("PIT universe is empty", code="SOURCE_VALIDATION_FAILED")

        def project_market(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
            if binding["maximum_session"] != signal_session:
                raise FactorGovernanceError(
                    "market history cutoff differs from the signal session",
                    code="SOURCE_SESSION_AFTER_SIGNAL_CUTOFF",
                )
            market_frame = table.to_pandas()
            try:
                if not set(market_frame["symbol"]).issubset(pit_by_symbol):
                    raise FactorGovernanceError(
                        "market history contains a symbol outside the PIT universe",
                        code="SOURCE_VALIDATION_FAILED",
                    )
                frames = {
                    str(symbol): frame.drop(columns=["symbol"]).reset_index(drop=True)
                    for symbol, frame in market_frame.groupby("symbol", sort=True)
                }
                factor_ids = sorted(
                    {row["factor_id"] for row in candidates},
                    key=lambda value: value.encode("utf-8"),
                )
                computed = compute_installed_signals(frames, factor_ids=factor_ids)
                all_signal_values: dict[str, dict[str, float | None]] = {}
                summaries: list[dict[str, Any]] = []
                for candidate in candidates:
                    factor_id = candidate["factor_id"]
                    configuration_id = candidate["configuration_id"]
                    completeness = self._latest_input_completeness(
                        frames,
                        symbols=symbols,
                        input_fields=candidate["input_fields"],
                        signal_session=signal_session,
                    )
                    values = {
                        symbol: (
                            float(computed[factor_id].get(symbol))
                            if _finite_hex(computed[factor_id].get(symbol)) is not None
                            and completeness[symbol]
                            else None
                        )
                        for symbol in symbols
                    }
                    all_signal_values[configuration_id] = values
                    finite_count = sum(value is not None for value in values.values())
                    required_count = sum(completeness.values())
                    complete_count = sum(
                        values[symbol] is not None
                        and completeness[symbol]
                        and pit_by_symbol[symbol]["industry"] is not None
                        and pit_by_symbol[symbol]["total_mv"] is not None
                        and pit_by_symbol[symbol]["tradable"] is not None
                        for symbol in symbols
                    )
                    denominator = len(symbols)
                    selection_coverage = Decimal(complete_count) / Decimal(denominator)
                    summaries.append(
                        {
                            "configuration_id": configuration_id,
                            "factor_id": factor_id,
                            "normalized_input_sha256": _normalized_input_sha256(
                                candidate=candidate,
                                market_normalized_sha256=binding["normalized_sha256"],
                            ),
                            "signal_sha256": _signal_values_sha256(
                                configuration_id=configuration_id,
                                factor_id=factor_id,
                                symbols=symbols,
                                values=values,
                            ),
                            "finite_signal_count": finite_count,
                            "required_input_complete_count": required_count,
                            "selection_complete_count": complete_count,
                            "denominator_count": denominator,
                            "signal_coverage": decimal_text(
                                Decimal(finite_count) / Decimal(denominator)
                            ),
                            "selection_coverage": decimal_text(selection_coverage),
                            "coverage_gate": (
                                "PASSED" if selection_coverage >= _COVERAGE_MINIMUM else "FAILED"
                            ),
                        }
                    )
                selected_rows = (
                    _selection_rows(candidates, summaries)
                    if selection is None
                    else [dict(row) for row in selection["payload"]["selected_configurations"]]
                )
                selected_ids = {row["selected_configuration_id"] for row in selected_rows}
                return {
                    "signal_values": {
                        configuration_id: all_signal_values[configuration_id]
                        for configuration_id in sorted(
                            selected_ids, key=lambda value: value.encode("utf-8")
                        )
                    },
                    "configuration_summary_rows": summaries,
                    "selected_configurations": selected_rows,
                }
            finally:
                del market_frame

        market = decode_source_role(
            system_store=self._system_store,
            source_object_ref=market_history_ref,
            role="market_history",
            projector=project_market,
        )
        signal_values = market.projection["signal_values"]
        summaries = market.projection["configuration_summary_rows"]
        selected_rows = market.projection["selected_configurations"]
        decoded["market_history"] = self._decoded_summary(market)
        selected_factor_by_configuration = {
            row["selected_configuration_id"]: row["selected_factor_id"] for row in selected_rows
        }

        weights = decode_source_role(
            system_store=self._system_store,
            source_object_ref=sparse_weights_ref,
            role="sparse_weights",
            projector=lambda table, binding: {
                "rows": [
                    {
                        "configuration_id": row["configuration_id"],
                        "symbol": row["symbol"],
                        "weight": format(row["weight"], ".12f"),
                    }
                    for row in table.to_pylist()
                ]
            },
        )
        if (
            weights.binding["minimum_session"] != signal_session
            or weights.binding["maximum_session"] != signal_session
        ):
            raise FactorGovernanceError(
                "sparse weights do not bind the signal session",
                code="SOURCE_VALIDATION_FAILED",
            )
        weights_by_configuration: dict[str, dict[str, Decimal]] = {
            configuration_id: {} for configuration_id in selected_factor_by_configuration
        }
        for row in weights.projection["rows"]:
            configuration_id = row["configuration_id"]
            symbol = row["symbol"]
            if configuration_id not in weights_by_configuration or symbol not in pit_by_symbol:
                raise FactorGovernanceError(
                    "sparse weight lies outside selected PIT closure",
                    code="SOURCE_VALIDATION_FAILED",
                )
            weights_by_configuration[configuration_id][symbol] = Decimal(row["weight"])
        if any(not rows for rows in weights_by_configuration.values()):
            raise FactorGovernanceError(
                "every selected configuration requires one sparse weight row",
                code="SOURCE_VALIDATION_FAILED",
            )
        decoded["sparse_weights"] = self._decoded_summary(weights)

        capture_rows: list[dict[str, Any]] = []
        summary_by_configuration = {row["configuration_id"]: row for row in summaries}
        for configuration_id in sorted(
            selected_factor_by_configuration, key=lambda value: value.encode("utf-8")
        ):
            factor_id = selected_factor_by_configuration[configuration_id]
            summary = summary_by_configuration[configuration_id]
            configuration_weights = weights_by_configuration[configuration_id]
            long_weight = sum(
                (value for value in configuration_weights.values() if value > 0), Decimal("0")
            )
            short_weight = sum(
                (value for value in configuration_weights.values() if value < 0), Decimal("0")
            )
            gross_weight = long_weight - short_weight
            net_weight = long_weight + short_weight
            if gross_weight > Decimal("1") or abs(net_weight) > Decimal("1"):
                raise FactorGovernanceError(
                    "sparse portfolio weights exceed their normalized bound",
                    code="SOURCE_VALIDATION_FAILED",
                )
            capture_rows.append(
                {
                    "configuration_id": configuration_id,
                    "factor_id": factor_id,
                    "signal_values_sha256": summary["signal_sha256"],
                    "finite_signal_count": summary["finite_signal_count"],
                    "coverage_numerator_count": summary["finite_signal_count"],
                    "coverage_denominator_count": len(symbols),
                    "coverage": summary["signal_coverage"],
                    "coverage_gate": (
                        "PASSED"
                        if Decimal(summary["signal_coverage"]) >= _COVERAGE_MINIMUM
                        else "FAILED"
                    ),
                    "portfolio_weights_sha256": _portfolio_weights_sha256(
                        configuration_id=configuration_id,
                        signal_session=signal_session,
                        weights=configuration_weights,
                    ),
                    "nonzero_weight_count": len(configuration_weights),
                    "long_weight": decimal_text(long_weight),
                    "short_weight": decimal_text(short_weight),
                    "gross_weight": decimal_text(gross_weight),
                    "net_weight": decimal_text(net_weight),
                }
            )
        return {
            "decoded_sources": decoded,
            "pit_universe_count": len(symbols),
            "pit_universe_sha256": pit_sha256,
            "pit_rows": pit_rows,
            "symbols": symbols,
            "signal_values": signal_values,
            "weights_by_configuration": weights_by_configuration,
            "configuration_summary_rows": summaries,
            "selected_configurations": selected_rows,
            "configuration_rows": capture_rows,
        }

    @staticmethod
    def _neutralized_rankic(
        *,
        pit_rows: Sequence[Mapping[str, Any]],
        signal_values: Mapping[str, float | None],
        label_values: Mapping[str, float | None],
    ) -> tuple[int, float | None, float | None]:
        complete: list[tuple[float, float, str, float]] = []
        for row in pit_rows:
            symbol = row["symbol"]
            signal_value = signal_values.get(symbol)
            label_value = label_values.get(symbol)
            industry = row["industry"]
            total_mv = row["total_mv"]
            if signal_value is None or label_value is None or industry is None or total_mv is None:
                continue
            complete.append(
                (float(signal_value), float(label_value), str(industry), float(total_mv))
            )
        count = len(complete)
        if count < 20:
            return count, None, None
        industries = sorted({row[2] for row in complete}, key=lambda value: value.encode("utf-8"))
        design = np.asarray(
            [
                [
                    1.0,
                    math.log(row[3]),
                    *[1.0 if row[2] == industry else 0.0 for industry in industries[1:]],
                ]
                for row in complete
            ],
            dtype=float,
        )
        signals = np.asarray([row[0] for row in complete], dtype=float)
        labels = np.asarray([row[1] for row in complete], dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, signals, rcond=None)
        residuals = signals - design @ coefficients
        if np.ptp(residuals) == 0 or np.ptp(labels) == 0:
            return count, None, None
        result = spearmanr(residuals, labels)
        rank_ic = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(rank_ic) or not math.isfinite(p_value):
            return count, None, None
        return count, rank_ic, p_value

    def _replay_signal_capture_projection(  # noqa: C901
        self,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any],
        capture: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        attestation_ref = capture["payload"]["source_decode_attestation_ref"]
        _, attestation = self._resolve(
            attestation_ref,
            label="capture.source_decode_attestation_ref",
            expected_kind="factor.source_decode_attestation",
        )
        attestation = validate_source_decode_attestation(attestation)
        ordinal = capture["payload"]["ordinal"]
        attestation_payload = attestation["payload"]
        expected_selection_id = None if ordinal == 0 else selection["payload"]["selection_id"]
        if (
            attestation_payload["purpose"] != "SIGNAL_CAPTURE"
            or attestation_payload["preregistration_id"]
            != preregistration["payload"]["preregistration_id"]
            or attestation_payload["selection_id"] != expected_selection_id
            or attestation_payload["ordinal"] != ordinal
            or attestation_payload["signal_session"] != capture["payload"]["signal_session"]
            or attestation_payload["maturity_session"] is not None
        ):
            raise FactorGovernanceError(
                "signal capture source attestation identity differs",
                code="SOURCE_VALIDATION_FAILED",
            )
        bindings = {
            row["role"]: row["source_object_ref"] for row in attestation_payload["source_bindings"]
        }
        if set(bindings) != {"pit_universe", "market_history", "sparse_weights"}:
            raise FactorGovernanceError(
                "signal capture source role closure differs",
                code="SOURCE_VALIDATION_FAILED",
            )
        projection = self._decode_signal_projection(
            preregistration=preregistration,
            selection=selection,
            ordinal=ordinal,
            pit_universe_ref=bindings["pit_universe"],
            market_history_ref=bindings["market_history"],
            sparse_weights_ref=bindings["sparse_weights"],
        )
        contextual, decoder, implementations = self._manifest_components(manifest)
        rebuilt = build_source_decode_attestation(
            purpose="SIGNAL_CAPTURE",
            preregistration_id=preregistration["payload"]["preregistration_id"],
            selection_id=expected_selection_id,
            ordinal=ordinal,
            signal_session=capture["payload"]["signal_session"],
            maturity_session=None,
            decoded_sources=projection["decoded_sources"],
            factor_validator_manifest=manifest,
            contextual_validator_component=contextual,
            source_decoder_component=decoder,
            implementation_components=implementations,
            trusted_at=attestation["created_at"],
        )
        selection_differs = ordinal == 0 and (
            selection["payload"]["source_decode_attestation_ref"] != attestation_ref
            or selection["payload"]["configuration_summary_rows"]
            != projection["configuration_summary_rows"]
            or selection["payload"]["selected_configurations"]
            != _selection_rows(
                preregistration["payload"]["candidates"],
                projection["configuration_summary_rows"],
            )
        )
        if (
            rebuilt != attestation
            or selection_differs
            or (
                projection["pit_universe_count"] != capture["payload"]["pit_universe_count"]
                or projection["pit_universe_sha256"] != capture["payload"]["pit_universe_sha256"]
                or projection["configuration_rows"] != capture["payload"]["configuration_rows"]
            )
        ):
            raise FactorGovernanceError(
                "signal capture does not replay from its exact raw sources",
                code="SOURCE_VALIDATION_FAILED",
            )
        return projection

    def _decode_label_projection(  # noqa: C901
        self,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any],
        capture: Mapping[str, Any],
        signal_projection: Mapping[str, Any],
        matured_label_prices_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        ordinal = capture["payload"]["ordinal"]
        signal_session = capture["payload"]["signal_session"]
        label_start_session = preregistration["payload"]["open_sessions"][ordinal + 1]
        label_end_session = preregistration["payload"]["open_sessions"][ordinal + 30]
        symbols = list(signal_projection["symbols"])
        symbol_set = set(symbols)

        decoded = decode_source_role(
            system_store=self._system_store,
            source_object_ref=matured_label_prices_ref,
            role="matured_label_prices",
            projector=lambda table, binding: {
                "rows": [
                    {
                        "price_date": row["price_date"].isoformat(),
                        "symbol": row["symbol"],
                        "adj_close": row["adj_close"],
                    }
                    for row in table.to_pylist()
                ]
            },
        )
        if (
            decoded.binding["minimum_session"] != label_start_session
            or decoded.binding["maximum_session"] != label_end_session
        ):
            raise FactorGovernanceError(
                "matured labels do not bind exact t+1 and t+30 sessions",
                code="SOURCE_VALIDATION_FAILED",
            )
        prices: dict[tuple[str, str], float | None] = {}
        observed_sessions: set[str] = set()
        for row in decoded.projection["rows"]:
            price_session = row["price_date"]
            symbol = row["symbol"]
            if symbol not in symbol_set or price_session not in {
                label_start_session,
                label_end_session,
            }:
                raise FactorGovernanceError(
                    "matured label row lies outside the capture closure",
                    code="SOURCE_VALIDATION_FAILED",
                )
            observed_sessions.add(price_session)
            prices[(price_session, symbol)] = row["adj_close"]
        if observed_sessions != {label_start_session, label_end_session}:
            raise FactorGovernanceError(
                "matured label endpoints are incomplete",
                code="SOURCE_VALIDATION_FAILED",
            )
        decoded_summary = self._decoded_summary(decoded)

        label_values: dict[str, float | None] = {}
        for symbol in symbols:
            start = prices.get((label_start_session, symbol))
            end = prices.get((label_end_session, symbol))
            label_values[symbol] = (
                None if start is None or end is None else float(end) / float(start) - 1.0
            )
        configuration_rows: list[dict[str, Any]] = []
        capture_rows = {
            row["configuration_id"]: row for row in capture["payload"]["configuration_rows"]
        }
        selected = {
            row["selected_configuration_id"]: row["selected_factor_id"]
            for row in selection["payload"]["selected_configurations"]
        }
        for configuration_id in sorted(selected, key=lambda value: value.encode("utf-8")):
            factor_id = selected[configuration_id]
            capture_row = capture_rows[configuration_id]
            values = signal_projection["signal_values"][configuration_id]
            weights = signal_projection["weights_by_configuration"][configuration_id]
            complete_count, rank_ic, rank_p = self._neutralized_rankic(
                pit_rows=signal_projection["pit_rows"],
                signal_values=values,
                label_values=label_values,
            )
            missing_held = sum(label_values.get(symbol) is None for symbol in weights)
            gross_count = len(weights) - missing_held
            gross_return = None
            if missing_held == 0:
                gross_return = sum(
                    (
                        weight * Decimal(str(label_values[symbol]))
                        for symbol, weight in weights.items()
                    ),
                    Decimal("0"),
                )
            valid = (
                capture_row["coverage_gate"] == "PASSED"
                and complete_count >= 20
                and rank_ic is not None
                and rank_p is not None
                and missing_held == 0
            )
            configuration_rows.append(
                {
                    "configuration_id": configuration_id,
                    "factor_id": factor_id,
                    "signal_values_sha256": capture_row["signal_values_sha256"],
                    "coverage_numerator_count": capture_row["coverage_numerator_count"],
                    "coverage_denominator_count": capture_row["coverage_denominator_count"],
                    "coverage": capture_row["coverage"],
                    "coverage_gate": capture_row["coverage_gate"],
                    "complete_case_count": complete_count,
                    "held_nonzero_symbol_count": len(weights),
                    "held_missing_label_count": missing_held,
                    "gross_labeled_return_symbol_count": gross_count,
                    "gross_labeled_return": (
                        None if gross_return is None else decimal_text(gross_return)
                    ),
                    "rank_ic": (None if rank_ic is None else decimal_text(Decimal(str(rank_ic)))),
                    "rank_ic_p_value": (
                        None if rank_p is None else decimal_text(Decimal(str(rank_p)))
                    ),
                    "valid_daily_rankic": valid,
                }
            )
        return {
            "decoded_source": decoded_summary,
            "label_values": label_values,
            "label_values_sha256": _label_values_sha256(
                ordinal=ordinal,
                signal_session=signal_session,
                label_start_session=label_start_session,
                label_end_session=label_end_session,
                symbols=symbols,
                values=label_values,
            ),
            "label_finite_pair_count": sum(value is not None for value in label_values.values()),
            "configuration_rows": configuration_rows,
        }

    def mine(  # noqa: C901
        self,
        *,
        exchange_calendar_ref: Mapping[str, Any],
        implementation_manifest_ref: Mapping[str, Any],
        factor_validator_manifest_ref: Mapping[str, Any],
        expected_composite_state_ref: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Preregister one new deterministic prospective namespace."""

        if expected_composite_state_ref is not None:
            raise FactorGovernanceError(
                "mine requires an empty expected composite state",
                code="COMPOSITE_STATE_CONFLICT",
            )
        calendar_ref = validate_artifact_ref(
            exchange_calendar_ref,
            label="exchange_calendar_ref",
            expected_kind="system.source_object",
        )
        implementation_ref = validate_artifact_ref(
            implementation_manifest_ref,
            label="implementation_manifest_ref",
            expected_kind="system.source_object",
        )
        manifest_ref, manifest = self._resolve(
            factor_validator_manifest_ref,
            label="factor_validator_manifest_ref",
            expected_kind="factor.validator_manifest",
        )
        manifest = validate_validator_manifest(manifest)
        contextual_component = validate_installed_component_manifest(
            self._system_store.get_object(manifest["payload"]["contextual_validator_component_ref"])
        )
        if (
            contextual_component["payload"]["component_id"]
            != f"{PROSPECTIVE_VALIDATION_PROFILE}-component"
        ):
            raise FactorGovernanceError(
                "mine requires the compiled prospective contextual validator",
                code="IMPLEMENTATION_IDENTITY_MISMATCH",
            )
        namespace = prospective_validation_namespace_id(
            exchange_calendar_ref=calendar_ref,
            implementation_manifest_ref=implementation_ref,
            factor_validator_manifest_ref=manifest_ref,
        )
        existing = self._system_store.read_candidate_state(namespace)
        request_sha = operation_request_sha256(
            operation="PREREGISTER",
            expected_composite_state_ref=None,
            input_refs={
                "exchange_calendar": calendar_ref,
                "factor_validator_manifest": manifest_ref,
                "implementation_manifest": implementation_ref,
            },
        )
        tx_id = custody_transaction_id(
            custody_namespace_id=namespace,
            transaction_sequence=1,
            previous_composite_state_ref=None,
            operation_request_sha256_value=request_sha,
        )
        if existing is not None:
            current = validate_composite_state(existing["candidate_state"])
            if (
                current["payload"]["transaction_id"] == tx_id
                and current["payload"]["previous_composite_state_ref"] is None
            ):
                return current
            raise FactorGovernanceError(
                "prospective namespace already has a different candidate",
                code="COMPOSITE_STATE_CONFLICT",
            )

        transaction_plan = {
            "operation": "PREREGISTER",
            "operation_request_sha256": request_sha,
            "input_refs": [
                {"role": "exchange_calendar", "ref": calendar_ref},
                {"role": "factor_validator_manifest", "ref": manifest_ref},
                {"role": "implementation_manifest", "ref": implementation_ref},
            ],
            "transaction_record_count": 1,
        }
        try:
            intent = self._system_store.begin_candidate_transaction(
                namespace,
                tx_id,
                expected_pointer_sha256=EMPTY_POINTER_SHA256,
                transaction_plan=transaction_plan,
            )
        except SystemCASMismatch as exc:
            raise FactorGovernanceError(
                "candidate state changed before transaction begin",
                code="CANDIDATE_STATE_CONFLICT",
            ) from exc
        except SystemError as exc:
            code = "CLOCK_ROLLBACK" if exc.code == "SYSTEM_CLOCK_ROLLBACK" else None
            raise FactorGovernanceError(
                "candidate transaction intent could not be fixed",
                code=code or "CUSTODY_TRANSACTION_INCOMPLETE",
            ) from exc
        if intent["previous_candidate_state_ref"] is not None:
            raise FactorGovernanceError(
                "preregistration transaction has an unexpected predecessor",
                code="CUSTODY_TRANSACTION_INCOMPLETE",
            )
        trusted_at = canonical_timestamp(
            intent["trusted_at"], label="candidate transaction trusted_at"
        )
        decoded: dict[str, DecodedSource] = {}

        def project_calendar(table: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
            del binding
            sessions, windows = self._calendar_projection(table, trusted_at=trusted_at)
            return {"sessions": sessions, "windows": windows}

        calendar = decode_source_role(
            system_store=self._system_store,
            source_object_ref=calendar_ref,
            role="exchange_calendar",
            projector=project_calendar,
        )
        sessions = calendar.projection["sessions"]
        windows = calendar.projection["windows"]
        decoded["exchange_calendar"] = self._decoded_summary(calendar)

        implementation = decode_source_role(
            system_store=self._system_store,
            source_object_ref=implementation_ref,
            role="implementation_manifest",
            projector=lambda table, binding: {
                "rows": self._implementation_manifest_projection(
                    table,
                    factor_manifest=manifest,
                )
            },
        )
        implementation_rows = implementation.projection["rows"]
        decoded["implementation_manifest"] = self._decoded_summary(implementation)
        contextual = self._system_store.get_object(
            manifest["payload"]["contextual_validator_component_ref"]
        )
        decoder = self._system_store.get_object(manifest["payload"]["source_decoder_component_ref"])
        implementation_components = [
            self._system_store.get_object(row["implementation_component_ref"])
            for row in implementation_rows
        ]
        attestation = build_source_decode_attestation(
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
            implementation_components=implementation_components,
            trusted_at=trusted_at,
        )
        attestation_ref, _ = self._put(attestation)
        candidates = [self._candidate_from_manifest_row(row) for row in implementation_rows]
        preregistration = _build_preregistration(
            open_sessions=sessions,
            session_windows=windows,
            candidates=candidates,
            exchange_calendar_ref=calendar_ref,
            implementation_manifest_ref=implementation_ref,
            source_decode_attestation_ref=attestation_ref,
            factor_validator_manifest_ref=manifest_ref,
            trusted_at=trusted_at,
        )
        preregistration_ref, preregistration = self._put(preregistration)
        custody = build_custody_record(
            custody_namespace_id=namespace,
            preregistration_id=preregistration["payload"]["preregistration_id"],
            sequence=0,
            previous_custody_ref=None,
            previous_composite_state_ref=None,
            transaction_id=tx_id,
            transaction_sequence=1,
            transaction_record_index=0,
            transaction_record_count=1,
            operation_request_sha256=request_sha,
            operation="PREREGISTER",
            subject_refs=[preregistration_ref],
            source_attestation_refs=[attestation_ref],
            stage_slot=None,
            blockers=[],
            trusted_at=trusted_at,
        )
        custody_ref, _ = self._put(custody)
        composite = build_composite_state(
            custody_namespace_id=namespace,
            preregistration_ref=preregistration_ref,
            cycle_state="PREREGISTERED",
            transaction_sequence=1,
            previous_composite_state_ref=None,
            transaction_id=tx_id,
            custody_record_count=1,
            custody_head_ref=custody_ref,
            selection_ref=None,
            signal_capture_count=0,
            signal_capture_head_ref=None,
            observation_count=0,
            observation_head_ref=None,
            execution_evidence_ref=None,
            evaluation_ref=None,
            admitted_set_ref=None,
            intrinsic_receipt_ref=None,
            resolved_signal_slot_count=0,
            resolved_label_slot_count=0,
            slot_tree_sha256=custody_slot_tree_sha256([]),
            terminal=False,
            blockers=[],
            last_stored_at=trusted_at,
        )
        composite_ref, _ = self._put(composite)
        try:
            committed = self._system_store.compare_and_swap_candidate_state(
                namespace,
                composite_ref,
                expected_pointer_sha256=EMPTY_POINTER_SHA256,
            )
        except SystemCASMismatch as exc:
            observed = self._system_store.read_candidate_state(namespace)
            if observed is not None:
                current = validate_composite_state(observed["candidate_state"])
                if current["payload"]["transaction_id"] == tx_id:
                    return current
            raise FactorGovernanceError(
                "candidate state changed concurrently",
                code="CANDIDATE_STATE_CONFLICT",
            ) from exc
        candidate = validate_composite_state(committed["candidate_state"])
        self._slot_cache[_reference_key(artifact_ref(candidate))] = ()
        return candidate

    def _commit_missed_signal(
        self,
        transition: _CandidateTransition,
        *,
        preregistration: Mapping[str, Any],
        ordinal: int,
        trusted_at: str,
    ) -> dict[str, Any]:
        previous = transition.previous["payload"]
        slot = build_stage_slot(
            stage="SIGNAL",
            ordinal=ordinal,
            signal_session=preregistration["payload"]["signal_sessions"][ordinal],
            maturity_session=None,
            state="MISSED",
            subject_ref=None,
            blocker="SIGNAL_WINDOW_MISSED",
        )
        custody = build_custody_record(
            custody_namespace_id=previous["custody_namespace_id"],
            preregistration_id=preregistration["payload"]["preregistration_id"],
            sequence=previous["custody_record_count"],
            previous_custody_ref=previous["custody_head_ref"],
            previous_composite_state_ref=transition.previous_ref,
            transaction_id=transition.transaction_id,
            transaction_sequence=transition.transaction_sequence,
            transaction_record_index=0,
            transaction_record_count=1,
            operation_request_sha256=transition.operation_request_sha256,
            operation="OBSERVE_SIGNAL",
            subject_refs=[],
            source_attestation_refs=[],
            stage_slot=slot,
            blockers=["SIGNAL_WINDOW_MISSED"],
            trusted_at=trusted_at,
        )
        custody_ref, _ = self._put(custody)
        slots = sorted(
            [*self._stage_slots(transition.previous), slot],
            key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"),
        )
        composite = self._next_composite(
            transition,
            custody_head_ref=custody_ref,
            custody_record_count=previous["custody_record_count"] + 1,
            stage_slots=slots,
            trusted_at=trusted_at,
            overrides={
                "cycle_state": "SIGNAL_CAPTURE_MISSED",
                "resolved_signal_slot_count": previous["resolved_signal_slot_count"] + 1,
                "terminal": True,
                "blockers": ["SIGNAL_WINDOW_MISSED"],
            },
        )
        return self._commit_candidate(transition, composite, stage_slots=slots)

    def _commit_captured_signal(  # noqa: C901
        self,
        transition: _CandidateTransition,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any] | None,
        manifest: Mapping[str, Any],
        ordinal: int,
        projection: Mapping[str, Any],
        trusted_at: str,
    ) -> dict[str, Any]:
        previous = transition.previous["payload"]
        contextual, decoder, implementations = self._manifest_components(manifest)
        if ordinal > 0 and selection is None:
            raise FactorGovernanceError("later signal capture requires selection")
        attestation_selection_id = (
            None if selection is None else selection["payload"]["selection_id"]
        )
        attestation = build_source_decode_attestation(
            purpose="SIGNAL_CAPTURE",
            preregistration_id=preregistration["payload"]["preregistration_id"],
            selection_id=attestation_selection_id,
            ordinal=ordinal,
            signal_session=preregistration["payload"]["signal_sessions"][ordinal],
            maturity_session=None,
            decoded_sources=projection["decoded_sources"],
            factor_validator_manifest=manifest,
            contextual_validator_component=contextual,
            source_decoder_component=decoder,
            implementation_components=implementations,
            trusted_at=trusted_at,
        )
        attestation_ref, _ = self._put(attestation)
        selection_ref: dict[str, str]
        selection_record_ref: dict[str, str] | None = None
        if ordinal == 0:
            selection = _build_configuration_selection(
                preregistration=preregistration,
                source_decode_attestation_ref=attestation_ref,
                configuration_summary_rows=projection["configuration_summary_rows"],
                selected_configurations=projection["selected_configurations"],
                trusted_at=trusted_at,
            )
            selection_ref, selection = self._put(selection)
        else:
            assert selection is not None
            selection_ref = artifact_ref(selection)
        previous_capture = None
        if previous["signal_capture_head_ref"] is not None:
            _, previous_capture = self._resolve(
                previous["signal_capture_head_ref"],
                label="previous_signal_capture_ref",
                expected_kind="factor.signal_capture",
            )
        capture = _build_signal_capture(
            preregistration=preregistration,
            selection=selection,
            previous_signal_capture=previous_capture,
            source_decode_attestation_ref=attestation_ref,
            ordinal=ordinal,
            pit_universe_count=projection["pit_universe_count"],
            pit_universe_sha256=projection["pit_universe_sha256"],
            configuration_rows=projection["configuration_rows"],
            trusted_at=trusted_at,
        )
        capture_ref, capture = self._put(capture)
        record_count = 2 if ordinal == 0 else 1
        previous_custody_ref = previous["custody_head_ref"]
        if ordinal == 0:
            selection_record = build_custody_record(
                custody_namespace_id=previous["custody_namespace_id"],
                preregistration_id=preregistration["payload"]["preregistration_id"],
                sequence=previous["custody_record_count"],
                previous_custody_ref=previous_custody_ref,
                previous_composite_state_ref=transition.previous_ref,
                transaction_id=transition.transaction_id,
                transaction_sequence=transition.transaction_sequence,
                transaction_record_index=0,
                transaction_record_count=2,
                operation_request_sha256=transition.operation_request_sha256,
                operation="OBSERVE_SIGNAL",
                subject_refs=[selection_ref],
                source_attestation_refs=[attestation_ref],
                stage_slot=None,
                blockers=[],
                trusted_at=trusted_at,
            )
            selection_record_ref, _ = self._put(selection_record)
            previous_custody_ref = selection_record_ref
        slot = build_stage_slot(
            stage="SIGNAL",
            ordinal=ordinal,
            signal_session=capture["payload"]["signal_session"],
            maturity_session=None,
            state="CAPTURED",
            subject_ref=capture_ref,
            blocker=None,
        )
        blockers = (
            ["SIGNAL_COVERAGE_BELOW_MINIMUM"]
            if any(row["coverage_gate"] != "PASSED" for row in projection["configuration_rows"])
            else []
        )
        capture_record = build_custody_record(
            custody_namespace_id=previous["custody_namespace_id"],
            preregistration_id=preregistration["payload"]["preregistration_id"],
            sequence=previous["custody_record_count"] + (1 if ordinal == 0 else 0),
            previous_custody_ref=previous_custody_ref,
            previous_composite_state_ref=transition.previous_ref,
            transaction_id=transition.transaction_id,
            transaction_sequence=transition.transaction_sequence,
            transaction_record_index=1 if ordinal == 0 else 0,
            transaction_record_count=record_count,
            operation_request_sha256=transition.operation_request_sha256,
            operation="OBSERVE_SIGNAL",
            subject_refs=[capture_ref],
            source_attestation_refs=[attestation_ref],
            stage_slot=slot,
            blockers=blockers,
            trusted_at=trusted_at,
        )
        capture_record_ref, _ = self._put(capture_record)
        slots = sorted(
            [*self._stage_slots(transition.previous), slot],
            key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"),
        )
        composite = self._next_composite(
            transition,
            custody_head_ref=capture_record_ref,
            custody_record_count=previous["custody_record_count"] + record_count,
            stage_slots=slots,
            trusted_at=trusted_at,
            overrides={
                "selection_ref": selection_ref,
                "signal_capture_count": previous["signal_capture_count"] + 1,
                "signal_capture_head_ref": capture_ref,
                "resolved_signal_slot_count": previous["resolved_signal_slot_count"] + 1,
                "cycle_state": "TERMINAL_INCOMPLETE" if blockers else "OBSERVING",
                "terminal": bool(blockers),
                "blockers": blockers,
            },
        )
        return self._commit_candidate(transition, composite, stage_slots=slots)

    def observe_signal(
        self,
        *,
        preregistration_ref: Mapping[str, Any],
        selection_ref: Mapping[str, Any] | None,
        pit_universe_ref: Mapping[str, Any],
        market_history_ref: Mapping[str, Any],
        sparse_weights_ref: Mapping[str, Any],
        expected_composite_state_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Capture one exact signal slot and atomically select at ordinal zero."""

        prereg_ref = validate_artifact_ref(
            preregistration_ref,
            label="preregistration_ref",
            expected_kind="factor.preregistration",
        )
        selected_ref = (
            None
            if selection_ref is None
            else validate_artifact_ref(
                selection_ref,
                label="selection_ref",
                expected_kind="factor.configuration_selection",
            )
        )
        pit_ref = validate_artifact_ref(
            pit_universe_ref, label="pit_universe_ref", expected_kind="system.source_object"
        )
        market_ref = validate_artifact_ref(
            market_history_ref,
            label="market_history_ref",
            expected_kind="system.source_object",
        )
        weights_ref = validate_artifact_ref(
            sparse_weights_ref,
            label="sparse_weights_ref",
            expected_kind="system.source_object",
        )
        input_refs = {
            "market_history": market_ref,
            "pit_universe": pit_ref,
            "preregistration": prereg_ref,
            "selection": selected_ref,
            "sparse_weights": weights_ref,
        }
        transition, already_committed = self._transition_head(
            operation="OBSERVE_SIGNAL",
            input_refs=input_refs,
            expected_composite_state_ref=expected_composite_state_ref,
        )
        if already_committed is not None:
            return already_committed
        assert transition is not None
        previous = transition.previous["payload"]
        if previous["preregistration_ref"] != prereg_ref:
            raise FactorGovernanceError("signal preregistration root differs")
        ordinal = previous["signal_capture_count"]
        if ordinal >= 360 or previous["cycle_state"] not in {"PREREGISTERED", "OBSERVING"}:
            raise FactorGovernanceError(
                "signal stage is already complete",
                code="STAGE_SLOT_ALREADY_RESOLVED",
            )
        if (ordinal == 0) != (selected_ref is None) or (
            ordinal > 0 and selected_ref != previous["selection_ref"]
        ):
            raise FactorGovernanceError(
                "signal selection ref differs from immutable cycle selection",
                code="SUBSTITUTION_FORBIDDEN",
            )
        if transition.transaction_sequence != self._expected_signal_transaction(ordinal):
            raise FactorGovernanceError(
                "signal capture does not follow the frozen signal/label interleave",
                code="STAGE_SLOT_ALREADY_RESOLVED",
            )
        _, preregistration = self._resolve(
            prereg_ref,
            label="preregistration_ref",
            expected_kind="factor.preregistration",
        )
        preregistration = validate_preregistration(preregistration)
        preflight = self._signal_window_state(preregistration, ordinal, self._trusted_at())
        if preflight == "NOT_OPEN":
            raise FactorGovernanceError(
                "signal capture window is not open",
                code="SIGNAL_WINDOW_NOT_OPEN",
            )
        record_count = 1 if preflight == "MISSED" or ordinal > 0 else 2
        trusted_at = self._begin_transition(
            transition,
            operation="OBSERVE_SIGNAL",
            input_refs=input_refs,
            transaction_record_count=record_count,
        )
        window_state = self._signal_window_state(preregistration, ordinal, trusted_at)
        if window_state == "NOT_OPEN":
            raise FactorGovernanceError(
                "System trusted time precedes the signal window",
                code="CLOCK_ROLLBACK",
            )
        if window_state == "MISSED":
            return self._commit_missed_signal(
                transition,
                preregistration=preregistration,
                ordinal=ordinal,
                trusted_at=trusted_at,
            )
        selection = None
        if selected_ref is not None:
            _, selection = self._resolve(
                selected_ref,
                label="selection_ref",
                expected_kind="factor.configuration_selection",
            )
            selection = _validate_configuration_selection_prevalidated(
                selection,
                preregistration=preregistration,
            )
        manifest_ref = preregistration["payload"]["factor_validator_manifest_ref"]
        _, manifest = self._resolve(
            manifest_ref,
            label="factor_validator_manifest_ref",
            expected_kind="factor.validator_manifest",
        )
        manifest = validate_validator_manifest(manifest)
        projection = self._decode_signal_projection(
            preregistration=preregistration,
            selection=selection,
            ordinal=ordinal,
            pit_universe_ref=pit_ref,
            market_history_ref=market_ref,
            sparse_weights_ref=weights_ref,
        )
        return self._commit_captured_signal(
            transition,
            preregistration=preregistration,
            selection=selection,
            manifest=manifest,
            ordinal=ordinal,
            projection=projection,
            trusted_at=trusted_at,
        )

    def _commit_missed_label(
        self,
        transition: _CandidateTransition,
        *,
        preregistration: Mapping[str, Any],
        ordinal: int,
        trusted_at: str,
    ) -> dict[str, Any]:
        previous = transition.previous["payload"]
        slot = build_stage_slot(
            stage="LABEL",
            ordinal=ordinal,
            signal_session=preregistration["payload"]["signal_sessions"][ordinal],
            maturity_session=preregistration["payload"]["open_sessions"][ordinal + 30],
            state="MISSED",
            subject_ref=None,
            blocker="LABEL_WINDOW_MISSED",
        )
        custody = build_custody_record(
            custody_namespace_id=previous["custody_namespace_id"],
            preregistration_id=preregistration["payload"]["preregistration_id"],
            sequence=previous["custody_record_count"],
            previous_custody_ref=previous["custody_head_ref"],
            previous_composite_state_ref=transition.previous_ref,
            transaction_id=transition.transaction_id,
            transaction_sequence=transition.transaction_sequence,
            transaction_record_index=0,
            transaction_record_count=1,
            operation_request_sha256=transition.operation_request_sha256,
            operation="OBSERVE_LABEL",
            subject_refs=[],
            source_attestation_refs=[],
            stage_slot=slot,
            blockers=["LABEL_WINDOW_MISSED"],
            trusted_at=trusted_at,
        )
        custody_ref, _ = self._put(custody)
        slots = sorted(
            [*self._stage_slots(transition.previous), slot],
            key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"),
        )
        composite = self._next_composite(
            transition,
            custody_head_ref=custody_ref,
            custody_record_count=previous["custody_record_count"] + 1,
            stage_slots=slots,
            trusted_at=trusted_at,
            overrides={
                "cycle_state": "LABEL_OBSERVATION_MISSED",
                "resolved_label_slot_count": previous["resolved_label_slot_count"] + 1,
                "terminal": True,
                "blockers": ["LABEL_WINDOW_MISSED"],
            },
        )
        return self._commit_candidate(transition, composite, stage_slots=slots)

    def _commit_captured_label(
        self,
        transition: _CandidateTransition,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any],
        capture: Mapping[str, Any],
        manifest: Mapping[str, Any],
        projection: Mapping[str, Any],
        trusted_at: str,
    ) -> dict[str, Any]:
        previous = transition.previous["payload"]
        ordinal = capture["payload"]["ordinal"]
        contextual, decoder, implementations = self._manifest_components(manifest)
        attestation = build_source_decode_attestation(
            purpose="LABEL_OBSERVATION",
            preregistration_id=preregistration["payload"]["preregistration_id"],
            selection_id=selection["payload"]["selection_id"],
            ordinal=ordinal,
            signal_session=capture["payload"]["signal_session"],
            maturity_session=preregistration["payload"]["open_sessions"][ordinal + 30],
            decoded_sources={"matured_label_prices": projection["decoded_source"]},
            factor_validator_manifest=manifest,
            contextual_validator_component=contextual,
            source_decoder_component=decoder,
            implementation_components=implementations,
            trusted_at=trusted_at,
        )
        attestation_ref, _ = self._put(attestation)
        previous_observation = None
        if previous["observation_head_ref"] is not None:
            _, previous_observation = self._resolve(
                previous["observation_head_ref"],
                label="previous_observation_ref",
                expected_kind="factor.prospective_observation",
            )
        observation = _build_observation(
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
            previous_observation=previous_observation,
            source_decode_attestation_ref=attestation_ref,
            pit_universe_sha256=capture["payload"]["pit_universe_sha256"],
            label_values_sha256=projection["label_values_sha256"],
            label_finite_pair_count=projection["label_finite_pair_count"],
            configuration_rows=projection["configuration_rows"],
            trusted_at=trusted_at,
        )
        observation_ref, observation = self._put(observation)
        slot = build_stage_slot(
            stage="LABEL",
            ordinal=ordinal,
            signal_session=capture["payload"]["signal_session"],
            maturity_session=observation["payload"]["label_end_session"],
            state="CAPTURED",
            subject_ref=observation_ref,
            blocker=None,
        )
        custody = build_custody_record(
            custody_namespace_id=previous["custody_namespace_id"],
            preregistration_id=preregistration["payload"]["preregistration_id"],
            sequence=previous["custody_record_count"],
            previous_custody_ref=previous["custody_head_ref"],
            previous_composite_state_ref=transition.previous_ref,
            transaction_id=transition.transaction_id,
            transaction_sequence=transition.transaction_sequence,
            transaction_record_index=0,
            transaction_record_count=1,
            operation_request_sha256=transition.operation_request_sha256,
            operation="OBSERVE_LABEL",
            subject_refs=[observation_ref],
            source_attestation_refs=[attestation_ref],
            stage_slot=slot,
            blockers=[],
            trusted_at=trusted_at,
        )
        custody_ref, _ = self._put(custody)
        slots = sorted(
            [*self._stage_slots(transition.previous), slot],
            key=lambda row: (row["ordinal"], row["stage"] != "SIGNAL"),
        )
        observation_count = previous["observation_count"] + 1
        cycle_state = (
            "OBSERVATIONS_MATURED"
            if previous["signal_capture_count"] == 360 and observation_count == 360
            else "OBSERVING"
        )
        composite = self._next_composite(
            transition,
            custody_head_ref=custody_ref,
            custody_record_count=previous["custody_record_count"] + 1,
            stage_slots=slots,
            trusted_at=trusted_at,
            overrides={
                "observation_count": observation_count,
                "observation_head_ref": observation_ref,
                "resolved_label_slot_count": previous["resolved_label_slot_count"] + 1,
                "cycle_state": cycle_state,
                "terminal": False,
                "blockers": [],
            },
        )
        return self._commit_candidate(transition, composite, stage_slots=slots)

    def observe_label(  # noqa: C901
        self,
        *,
        preregistration_ref: Mapping[str, Any],
        selection_ref: Mapping[str, Any],
        signal_capture_ref: Mapping[str, Any],
        matured_label_prices_ref: Mapping[str, Any],
        expected_composite_state_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append one exact matured t+1/t+30 observation without backfill."""

        prereg_ref = validate_artifact_ref(
            preregistration_ref,
            label="preregistration_ref",
            expected_kind="factor.preregistration",
        )
        selected_ref = validate_artifact_ref(
            selection_ref,
            label="selection_ref",
            expected_kind="factor.configuration_selection",
        )
        capture_ref = validate_artifact_ref(
            signal_capture_ref,
            label="signal_capture_ref",
            expected_kind="factor.signal_capture",
        )
        label_ref = validate_artifact_ref(
            matured_label_prices_ref,
            label="matured_label_prices_ref",
            expected_kind="system.source_object",
        )
        input_refs = {
            "matured_label_prices": label_ref,
            "preregistration": prereg_ref,
            "selection": selected_ref,
            "signal_capture": capture_ref,
        }
        transition, already_committed = self._transition_head(
            operation="OBSERVE_LABEL",
            input_refs=input_refs,
            expected_composite_state_ref=expected_composite_state_ref,
        )
        if already_committed is not None:
            return already_committed
        assert transition is not None
        previous = transition.previous["payload"]
        ordinal = previous["observation_count"]
        if (
            ordinal >= 360
            or previous["cycle_state"] != "OBSERVING"
            or previous["signal_capture_count"] <= ordinal
        ):
            raise FactorGovernanceError(
                "label stage is not the next resolvable stage",
                code="STAGE_SLOT_ALREADY_RESOLVED",
            )
        if (
            previous["preregistration_ref"] != prereg_ref
            or previous["selection_ref"] != selected_ref
        ):
            raise FactorGovernanceError(
                "label immutable cycle roots differ",
                code="SUBSTITUTION_FORBIDDEN",
            )
        if transition.transaction_sequence != self._expected_label_transaction(ordinal):
            raise FactorGovernanceError(
                "label observation does not follow the frozen signal/label interleave",
                code="STAGE_SLOT_ALREADY_RESOLVED",
            )
        signal_slots = {
            row["ordinal"]: row
            for row in self._stage_slots(transition.previous)
            if row["stage"] == "SIGNAL"
        }
        signal_slot = signal_slots.get(ordinal)
        if (
            signal_slot is None
            or signal_slot["state"] != "CAPTURED"
            or signal_slot["subject_ref"] != capture_ref
        ):
            raise FactorGovernanceError(
                "label signal capture is not the authoritative captured slot",
                code="SUBSTITUTION_FORBIDDEN",
            )
        _, preregistration = self._resolve(
            prereg_ref,
            label="preregistration_ref",
            expected_kind="factor.preregistration",
        )
        preregistration = validate_preregistration(preregistration)
        _, selection = self._resolve(
            selected_ref,
            label="selection_ref",
            expected_kind="factor.configuration_selection",
        )
        selection = _validate_configuration_selection_prevalidated(
            selection,
            preregistration=preregistration,
        )
        _, capture = self._resolve(
            capture_ref,
            label="signal_capture_ref",
            expected_kind="factor.signal_capture",
        )
        previous_capture = None
        if capture["payload"]["previous_signal_capture_ref"] is not None:
            _, previous_capture = self._resolve(
                capture["payload"]["previous_signal_capture_ref"],
                label="capture.previous_signal_capture_ref",
                expected_kind="factor.signal_capture",
            )
        capture = _validate_signal_capture_prevalidated(
            capture,
            preregistration=preregistration,
            selection=selection,
            previous_signal_capture=previous_capture,
        )
        if capture["payload"]["ordinal"] != ordinal:
            raise FactorGovernanceError(
                "label capture ordinal differs",
                code="SUBSTITUTION_FORBIDDEN",
            )
        preflight = self._label_window_state(preregistration, ordinal, self._trusted_at())
        if preflight == "NOT_OPEN":
            raise FactorGovernanceError(
                "label maturity window is not open",
                code="LABEL_WINDOW_NOT_OPEN",
            )
        trusted_at = self._begin_transition(
            transition,
            operation="OBSERVE_LABEL",
            input_refs=input_refs,
            transaction_record_count=1,
        )
        window_state = self._label_window_state(preregistration, ordinal, trusted_at)
        if window_state == "NOT_OPEN":
            raise FactorGovernanceError(
                "System trusted time precedes the label window",
                code="CLOCK_ROLLBACK",
            )
        if window_state == "MISSED":
            return self._commit_missed_label(
                transition,
                preregistration=preregistration,
                ordinal=ordinal,
                trusted_at=trusted_at,
            )
        _, manifest = self._resolve(
            preregistration["payload"]["factor_validator_manifest_ref"],
            label="factor_validator_manifest_ref",
            expected_kind="factor.validator_manifest",
        )
        manifest = validate_validator_manifest(manifest)
        signal_projection = self._replay_signal_capture_projection(
            preregistration=preregistration,
            selection=selection,
            capture=capture,
            manifest=manifest,
        )
        projection = self._decode_label_projection(
            preregistration=preregistration,
            selection=selection,
            capture=capture,
            signal_projection=signal_projection,
            matured_label_prices_ref=label_ref,
        )
        return self._commit_captured_label(
            transition,
            preregistration=preregistration,
            selection=selection,
            capture=capture,
            manifest=manifest,
            projection=projection,
            trusted_at=trusted_at,
        )

    def _resolve_prospective_stage_closure(
        self,
        *,
        composite: Mapping[str, Any],
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        slots = self._stage_slots(composite)
        signal_slots = [row for row in slots if row["stage"] == "SIGNAL"]
        label_slots = [row for row in slots if row["stage"] == "LABEL"]
        if (
            len(signal_slots) != 360
            or len(label_slots) != 360
            or any(
                row["ordinal"] != ordinal or row["state"] != "CAPTURED"
                for ordinal, row in enumerate(signal_slots)
            )
            or any(
                row["ordinal"] != ordinal or row["state"] != "CAPTURED"
                for ordinal, row in enumerate(label_slots)
            )
        ):
            raise FactorGovernanceError(
                "prospective stage closure is not 360 captured signal/label pairs",
                code="MATURITY_INCOMPLETE",
            )
        captures: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        previous_capture = None
        previous_observation = None
        for ordinal, (signal_slot, label_slot) in enumerate(
            zip(signal_slots, label_slots, strict=True)
        ):
            _, capture = self._resolve(
                signal_slot["subject_ref"],
                label=f"signal_capture[{ordinal}]",
                expected_kind="factor.signal_capture",
            )
            capture = _validate_signal_capture_prevalidated(
                capture,
                preregistration=preregistration,
                selection=selection,
                previous_signal_capture=previous_capture,
            )
            _, observation = self._resolve(
                label_slot["subject_ref"],
                label=f"observation[{ordinal}]",
                expected_kind="factor.prospective_observation",
            )
            observation = _validate_observation_prevalidated(
                observation,
                preregistration=preregistration,
                selection=selection,
                signal_capture=capture,
                previous_observation=previous_observation,
            )
            if (
                capture["payload"]["ordinal"] != ordinal
                or observation["payload"]["ordinal"] != ordinal
            ):
                raise FactorGovernanceError(
                    "prospective stage artifact ordinals differ",
                    code="CONTEXT_CLOSURE_INCOMPLETE",
                )
            captures.append(capture)
            observations.append(observation)
            previous_capture = capture
            previous_observation = observation
        return captures, observations

    def _replay_observation_projection(
        self,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any],
        capture: Mapping[str, Any],
        observation: Mapping[str, Any],
        manifest: Mapping[str, Any],
        signal_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        _, attestation = self._resolve(
            observation["payload"]["source_decode_attestation_ref"],
            label="observation.source_decode_attestation_ref",
            expected_kind="factor.source_decode_attestation",
        )
        attestation = validate_source_decode_attestation(attestation)
        payload = attestation["payload"]
        ordinal = observation["payload"]["ordinal"]
        if (
            payload["purpose"] != "LABEL_OBSERVATION"
            or payload["preregistration_id"] != preregistration["payload"]["preregistration_id"]
            or payload["selection_id"] != selection["payload"]["selection_id"]
            or payload["ordinal"] != ordinal
            or payload["signal_session"] != observation["payload"]["signal_session"]
            or payload["maturity_session"] != observation["payload"]["label_end_session"]
            or len(payload["source_bindings"]) != 1
            or payload["source_bindings"][0]["role"] != "matured_label_prices"
        ):
            raise FactorGovernanceError(
                "observation source attestation identity differs",
                code="SOURCE_VALIDATION_FAILED",
            )
        projection = self._decode_label_projection(
            preregistration=preregistration,
            selection=selection,
            capture=capture,
            signal_projection=signal_projection,
            matured_label_prices_ref=payload["source_bindings"][0]["source_object_ref"],
        )
        contextual, decoder, implementations = self._manifest_components(manifest)
        rebuilt = build_source_decode_attestation(
            purpose="LABEL_OBSERVATION",
            preregistration_id=preregistration["payload"]["preregistration_id"],
            selection_id=selection["payload"]["selection_id"],
            ordinal=ordinal,
            signal_session=observation["payload"]["signal_session"],
            maturity_session=observation["payload"]["label_end_session"],
            decoded_sources={"matured_label_prices": projection["decoded_source"]},
            factor_validator_manifest=manifest,
            contextual_validator_component=contextual,
            source_decoder_component=decoder,
            implementation_components=implementations,
            trusted_at=attestation["created_at"],
        )
        if rebuilt != attestation or (
            observation["payload"]["pit_universe_sha256"]
            != capture["payload"]["pit_universe_sha256"]
            or observation["payload"]["label_values_sha256"] != projection["label_values_sha256"]
            or observation["payload"]["label_finite_pair_count"]
            != projection["label_finite_pair_count"]
            or observation["payload"]["configuration_rows"] != projection["configuration_rows"]
        ):
            raise FactorGovernanceError(
                "observation does not replay from its exact raw sources",
                code="SOURCE_VALIDATION_FAILED",
            )
        return projection

    def _execution_configuration_rows(  # noqa: C901
        self,
        *,
        preregistration: Mapping[str, Any],
        selection: Mapping[str, Any],
        captures: Sequence[Mapping[str, Any]],
        observations: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        selected = {
            row["selected_configuration_id"]: row["selected_factor_id"]
            for row in selection["payload"]["selected_configurations"]
        }
        summaries: dict[str, list[dict[str, Any]]] = {
            configuration_id: [] for configuration_id in selected
        }
        initial: dict[str, Decimal] = {key: Decimal("0") for key in selected}
        rebalance: dict[str, Decimal] = {key: Decimal("0") for key in selected}
        terminal: dict[str, Decimal] = {key: Decimal("0") for key in selected}
        gross_values: dict[str, list[Decimal]] = {key: [] for key in selected}
        previous_weights: dict[str, dict[str, Decimal]] = {key: {} for key in selected}
        blockers: set[str] = set()
        for ordinal, (capture, observation) in enumerate(zip(captures, observations, strict=True)):
            signal_projection = self._replay_signal_capture_projection(
                preregistration=preregistration,
                selection=selection,
                capture=capture,
                manifest=manifest,
            )
            self._replay_observation_projection(
                preregistration=preregistration,
                selection=selection,
                capture=capture,
                observation=observation,
                manifest=manifest,
                signal_projection=signal_projection,
            )
            observation_rows = {
                row["configuration_id"]: row for row in observation["payload"]["configuration_rows"]
            }
            capture_rows = {
                row["configuration_id"]: row for row in capture["payload"]["configuration_rows"]
            }
            for configuration_id in sorted(selected, key=lambda value: value.encode("utf-8")):
                weights = signal_projection["weights_by_configuration"][configuration_id]
                prior = previous_weights[configuration_id]
                change = sum(
                    (
                        abs(weights.get(symbol, Decimal("0")) - prior.get(symbol, Decimal("0")))
                        for symbol in set(weights) | set(prior)
                    ),
                    Decimal("0"),
                )
                if ordinal == 0:
                    initial[configuration_id] = change
                else:
                    rebalance[configuration_id] += change
                exit_turnover = (
                    sum((abs(value) for value in weights.values()), Decimal("0"))
                    if ordinal == 359
                    else Decimal("0")
                )
                if ordinal == 359:
                    terminal[configuration_id] = exit_turnover
                session_turnover = change + exit_turnover
                session_cost = session_turnover * _EXECUTION_COST_RATE
                observation_row = observation_rows[configuration_id]
                gross_text = observation_row["gross_labeled_return"]
                gross = None if gross_text is None else Decimal(gross_text)
                if observation_row["held_missing_label_count"]:
                    blockers.add("EXECUTION_HELD_LABEL_MISSING")
                if gross is None:
                    blockers.add("EXECUTION_EVIDENCE_INCOMPLETE")
                else:
                    gross_values[configuration_id].append(gross)
                summaries[configuration_id].append(
                    {
                        "ordinal": ordinal,
                        "signal_session": capture["payload"]["signal_session"],
                        "portfolio_weights_sha256": capture_rows[configuration_id][
                            "portfolio_weights_sha256"
                        ],
                        "nonzero_weight_count": capture_rows[configuration_id][
                            "nonzero_weight_count"
                        ],
                        "entry_or_rebalance_turnover": decimal_text(change),
                        "terminal_exit_turnover": decimal_text(exit_turnover),
                        "total_turnover": decimal_text(session_turnover),
                        "estimated_cost": decimal_text(session_cost),
                        "gross_labeled_return": gross_text,
                        "net_labeled_return": (
                            None if gross is None else decimal_text(gross - session_cost)
                        ),
                        "held_missing_label_count": observation_row["held_missing_label_count"],
                    }
                )
                previous_weights[configuration_id] = dict(weights)
        rows: list[dict[str, Any]] = []
        for configuration_id in sorted(selected, key=lambda value: value.encode("utf-8")):
            total = (
                initial[configuration_id] + rebalance[configuration_id] + terminal[configuration_id]
            )
            cost = total * _EXECUTION_COST_RATE
            gross_count = len(gross_values[configuration_id])
            gross_sum = (
                None if gross_count == 0 else sum(gross_values[configuration_id], Decimal("0"))
            )
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "factor_id": selected[configuration_id],
                    "session_summary_sha256": _sha256_document(
                        {
                            "domain": "myquant-factor-execution-session-summary",
                            "configuration_id": configuration_id,
                            "factor_id": selected[configuration_id],
                            "rows": summaries[configuration_id],
                        }
                    ),
                    "session_summary_count": 360,
                    "initial_entry_turnover": decimal_text(initial[configuration_id]),
                    "rebalance_turnover": decimal_text(rebalance[configuration_id]),
                    "terminal_exit_turnover": decimal_text(terminal[configuration_id]),
                    "total_turnover": decimal_text(total),
                    "annualized_turnover": decimal_text(total * Decimal(252) / Decimal(360)),
                    "total_estimated_cost": decimal_text(cost),
                    "gross_labeled_return_count": gross_count,
                    "gross_labeled_return_sum": (
                        None if gross_sum is None else decimal_text(gross_sum)
                    ),
                    "net_labeled_return_sum": (
                        None if gross_sum is None else decimal_text(gross_sum - cost)
                    ),
                }
            )
        return rows, sorted(blockers)

    def _commit_final_artifact(
        self,
        transition: _CandidateTransition,
        *,
        preregistration: Mapping[str, Any],
        operation: str,
        artifact: Mapping[str, Any],
        composite_field: str,
        cycle_state: str,
        trusted_at: str,
        terminal: bool,
        blockers: Sequence[str],
    ) -> dict[str, Any]:
        previous = transition.previous["payload"]
        subject_ref, _ = self._put(artifact)
        custody = build_custody_record(
            custody_namespace_id=previous["custody_namespace_id"],
            preregistration_id=preregistration["payload"]["preregistration_id"],
            sequence=previous["custody_record_count"],
            previous_custody_ref=previous["custody_head_ref"],
            previous_composite_state_ref=transition.previous_ref,
            transaction_id=transition.transaction_id,
            transaction_sequence=transition.transaction_sequence,
            transaction_record_index=0,
            transaction_record_count=1,
            operation_request_sha256=transition.operation_request_sha256,
            operation=operation,
            subject_refs=[subject_ref],
            source_attestation_refs=[],
            stage_slot=None,
            blockers=[],
            trusted_at=trusted_at,
        )
        custody_ref, _ = self._put(custody)
        slots = self._stage_slots(transition.previous)
        composite = self._next_composite(
            transition,
            custody_head_ref=custody_ref,
            custody_record_count=previous["custody_record_count"] + 1,
            stage_slots=slots,
            trusted_at=trusted_at,
            overrides={
                composite_field: subject_ref,
                "cycle_state": cycle_state,
                "terminal": terminal,
                "blockers": sorted(set(blockers)),
            },
        )
        return self._commit_candidate(transition, composite, stage_slots=slots)

    def _source_attestation_artifacts(
        self,
        *,
        preregistration: Mapping[str, Any],
        captures: Sequence[Mapping[str, Any]],
        observations: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        refs = [preregistration["payload"]["source_decode_attestation_ref"]]
        refs.extend(row["payload"]["source_decode_attestation_ref"] for row in captures)
        refs.extend(row["payload"]["source_decode_attestation_ref"] for row in observations)
        artifacts: list[dict[str, Any]] = []
        for index, ref in enumerate(refs):
            _, artifact = self._resolve(
                ref,
                label=f"source_attestation[{index}]",
                expected_kind="factor.source_decode_attestation",
            )
            artifacts.append(validate_source_decode_attestation(artifact))
        if (
            len(artifacts) != 721
            or len({_reference_key(artifact_ref(row)) for row in artifacts}) != 721
        ):
            raise FactorGovernanceError(
                "prospective source-attestation closure is not exact",
                code="CONTEXT_CLOSURE_INCOMPLETE",
            )
        return artifacts

    def evaluate(  # noqa: C901
        self,
        *,
        action: str,
        preregistration_ref: Mapping[str, Any],
        selection_ref: Mapping[str, Any],
        expected_composite_state_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Advance one exact final prospective stage without activation authority."""

        if action not in {
            "FINALIZE_EXECUTION",
            "EVALUATE_PREREGISTRATION",
            "BUILD_ADMITTED_SET",
            "BUILD_INTRINSIC_RECEIPT",
        }:
            raise FactorGovernanceError("prospective evaluation action is invalid")
        prereg_ref = validate_artifact_ref(
            preregistration_ref,
            label="preregistration_ref",
            expected_kind="factor.preregistration",
        )
        selected_ref = validate_artifact_ref(
            selection_ref,
            label="selection_ref",
            expected_kind="factor.configuration_selection",
        )
        input_refs = {"preregistration": prereg_ref, "selection": selected_ref}
        transition, already_committed = self._transition_head(
            operation=action,
            input_refs=input_refs,
            expected_composite_state_ref=expected_composite_state_ref,
        )
        if already_committed is not None:
            return already_committed
        assert transition is not None
        previous = transition.previous["payload"]
        required = {
            "FINALIZE_EXECUTION": ("OBSERVATIONS_MATURED", 722),
            "EVALUATE_PREREGISTRATION": ("EXECUTION_FINALIZED", 723),
            "BUILD_ADMITTED_SET": ("EVALUATED_ELIGIBLE", 724),
            "BUILD_INTRINSIC_RECEIPT": ("ADMITTED", 725),
        }[action]
        if (
            previous["cycle_state"] != required[0]
            or transition.transaction_sequence != required[1]
            or previous["preregistration_ref"] != prereg_ref
            or previous["selection_ref"] != selected_ref
        ):
            raise FactorGovernanceError(
                "prospective final stage does not follow the frozen DAG",
                code="STAGE_SLOT_ALREADY_RESOLVED",
            )
        if _timestamp_value(self._trusted_at()) < _timestamp_value(previous["last_stored_at"]):
            raise FactorGovernanceError(
                "Factor validation clock moved backwards",
                code="CLOCK_ROLLBACK",
            )
        trusted_at = self._begin_transition(
            transition,
            operation=action,
            input_refs=input_refs,
            transaction_record_count=1,
        )
        _, preregistration = self._resolve(
            prereg_ref,
            label="preregistration_ref",
            expected_kind="factor.preregistration",
        )
        preregistration = validate_preregistration(preregistration)
        _, selection = self._resolve(
            selected_ref,
            label="selection_ref",
            expected_kind="factor.configuration_selection",
        )
        selection = _validate_configuration_selection_prevalidated(
            selection,
            preregistration=preregistration,
        )
        captures, observations = self._resolve_prospective_stage_closure(
            composite=transition.previous,
            preregistration=preregistration,
            selection=selection,
        )
        if action == "FINALIZE_EXECUTION":
            _, manifest = self._resolve(
                preregistration["payload"]["factor_validator_manifest_ref"],
                label="factor_validator_manifest_ref",
                expected_kind="factor.validator_manifest",
            )
            manifest = validate_validator_manifest(manifest)
            configuration_rows, blockers = self._execution_configuration_rows(
                preregistration=preregistration,
                selection=selection,
                captures=captures,
                observations=observations,
                manifest=manifest,
            )
            execution = _build_execution_turnover_evidence(
                preregistration=preregistration,
                selection=selection,
                signal_captures=captures,
                observations=observations,
                configuration_rows=configuration_rows,
                execution_state="INCOMPLETE" if blockers else "COMPLETE",
                blockers=blockers,
                trusted_at=trusted_at,
            )
            validate_execution_turnover_evidence(
                execution,
                preregistration=preregistration,
                selection=selection,
                signal_captures=captures,
                observations=observations,
            )
            return self._commit_final_artifact(
                transition,
                preregistration=preregistration,
                operation=action,
                artifact=execution,
                composite_field="execution_evidence_ref",
                cycle_state="EXECUTION_FINALIZED",
                trusted_at=trusted_at,
                terminal=False,
                blockers=[],
            )
        _, execution = self._resolve(
            previous["execution_evidence_ref"],
            label="execution_evidence_ref",
            expected_kind="factor.execution_turnover_evidence",
        )
        execution = validate_execution_turnover_evidence(
            execution,
            preregistration=preregistration,
            selection=selection,
            signal_captures=captures,
            observations=observations,
        )
        if action == "EVALUATE_PREREGISTRATION":
            evaluation = _evaluate_preregistration(
                preregistration=preregistration,
                selection=selection,
                signal_captures=captures,
                observations=observations,
                execution_turnover_evidence=execution,
                created_at=trusted_at,
            )
            eligible = evaluation["payload"]["admission_eligible"] is True
            return self._commit_final_artifact(
                transition,
                preregistration=preregistration,
                operation=action,
                artifact=evaluation,
                composite_field="evaluation_ref",
                cycle_state="EVALUATED_ELIGIBLE" if eligible else "EVALUATED_REJECTED",
                trusted_at=trusted_at,
                terminal=not eligible,
                blockers=[] if eligible else evaluation["payload"]["blockers"],
            )
        _, evaluation = self._resolve(
            previous["evaluation_ref"],
            label="evaluation_ref",
            expected_kind="factor.prospective_evaluation",
        )
        evaluation = validate_preregistration_evaluation(
            evaluation,
            preregistration=preregistration,
            selection=selection,
            signal_captures=captures,
            observations=observations,
            execution_turnover_evidence=execution,
        )
        if action == "BUILD_ADMITTED_SET":
            admitted = _build_admitted_factor_set(
                evaluation=evaluation,
                preregistration=preregistration,
                selection=selection,
                signal_captures=captures,
                observations=observations,
                execution_turnover_evidence=execution,
                created_at=trusted_at,
            )
            validate_admitted_factor_set(
                admitted,
                evaluation=evaluation,
                preregistration=preregistration,
                selection=selection,
                signal_captures=captures,
                observations=observations,
                execution_turnover_evidence=execution,
            )
            return self._commit_final_artifact(
                transition,
                preregistration=preregistration,
                operation=action,
                artifact=admitted,
                composite_field="admitted_set_ref",
                cycle_state="ADMITTED",
                trusted_at=trusted_at,
                terminal=False,
                blockers=[],
            )
        _, admitted = self._resolve(
            previous["admitted_set_ref"],
            label="admitted_set_ref",
            expected_kind="factor.admitted_set",
        )
        admitted = validate_admitted_factor_set(
            admitted,
            evaluation=evaluation,
            preregistration=preregistration,
            selection=selection,
            signal_captures=captures,
            observations=observations,
            execution_turnover_evidence=execution,
        )
        source_attestations = self._source_attestation_artifacts(
            preregistration=preregistration,
            captures=captures,
            observations=observations,
        )
        evidence_artifacts = [
            selection,
            *captures,
            *source_attestations,
            *observations,
            execution,
            evaluation,
            transition.previous,
        ]
        receipt = _build_factor_validation_receipt(
            policy=preregistration,
            active_set=admitted,
            evidence_artifacts=evidence_artifacts,
            trusted_at=trusted_at,
        )
        return self._commit_final_artifact(
            transition,
            preregistration=preregistration,
            operation=action,
            artifact=receipt,
            composite_field="intrinsic_receipt_ref",
            cycle_state="INTRINSIC_VALIDATED",
            trusted_at=trusted_at,
            terminal=True,
            blockers=[],
        )

    def build_status(
        self,
        *,
        active_factor_set_ref: Mapping[str, Any] | None,
        active_validation_receipt_ref: Mapping[str, Any] | None,
        active_contextual_result_ref: Mapping[str, Any] | None,
        active_validation_attestation_ref: Mapping[str, Any] | None,
        observed_composite_state_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an inert status after protected System attestation readback."""

        active_values = (
            active_factor_set_ref,
            active_validation_receipt_ref,
            active_contextual_result_ref,
            active_validation_attestation_ref,
        )
        if any(value is None for value in active_values) and not all(
            value is None for value in active_values
        ):
            raise FactorGovernanceError("active Factor validation refs are incomplete")
        active = receipt = context = attestation = None
        if all(value is not None for value in active_values):
            assert active_factor_set_ref is not None
            assert active_validation_receipt_ref is not None
            assert active_contextual_result_ref is not None
            assert active_validation_attestation_ref is not None
            _, active = self._resolve(active_factor_set_ref, label="active_factor_set_ref")
            _, receipt = self._resolve(
                active_validation_receipt_ref,
                label="active_validation_receipt_ref",
                expected_kind="factor.validation_receipt",
            )
            context_ref, context = self._resolve(
                active_contextual_result_ref,
                label="active_contextual_result_ref",
                expected_kind="factor.contextual_validation_result",
            )
            attestation_ref, attestation = self._resolve(
                active_validation_attestation_ref,
                label="active_validation_attestation_ref",
                expected_kind="system.validation_attestation",
            )
            try:
                protected = self._system_store.resolve_validation_attestation(
                    attestation_ref,
                    verification_level="stat",
                )
            except SystemError as exc:
                raise FactorGovernanceError(
                    "protected System validation custody is incomplete",
                    code="VALIDATION_ATTESTATION_INVALID",
                ) from exc
            if (
                protected["validation_attestation"] != attestation
                or protected["contextual_result_ref"] != context_ref
                or protected["contextual_result"] != context
            ):
                raise FactorGovernanceError("protected System validation readback differs")
        observed = None
        if observed_composite_state_ref is not None:
            _, observed = self._resolve(
                observed_composite_state_ref,
                label="observed_composite_state_ref",
                expected_kind="factor.composite_state",
            )
        status = _build_factor_status(
            active_factor_set=active,
            active_validation_receipt=receipt,
            active_contextual_result=context,
            active_validation_attestation=attestation,
            observed_composite_state=observed,
            trusted_at=self._trusted_at(),
        )
        _, stored = self._put(status)
        return validate_factor_status(stored)


__all__ = [
    "BootstrapClosure",
    "FactorValidationStore",
    "bootstrap_validation_namespace_id",
    "prospective_validation_namespace_id",
]
