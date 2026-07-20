"""Hermetic runtime and frozen model-bundle identity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    EvidenceRef,
    EvidenceV2Error,
    seal_semantic,
    validate_semantic_seal,
)

RUNTIME_CAPSULE_SCHEMA = "v16.hermetic-runtime-capsule.v2"
MODEL_BUNDLE_SCHEMA = "v16.frozen-model-bundle.v2"
LLM_PROVIDER_BUILD_SCHEMA = "v16.llm-provider-build-identity.v2"
FORMAL_BRANCHES = ("quant", "fundamental", "macro", "llm")
RUNTIME_COMPONENT_ORDER = (
    "python_interpreter",
    "source_tree",
    "dependency_lock",
    "installed_distributions",
    "platform_manifest",
    "pyarrow_backend",
    "scipy_backend",
    "openssl_backend",
)
PINNED_OPENSSL_PATH = "/opt/homebrew/opt/openssl@3/bin/openssl"
REQUIRED_ENVIRONMENT_CONTROLS = {
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+"
    if not text or text != text.strip() or len(text) > 256:
        raise EvidenceV2Error(f"{label} is not a bounded identifier")
    if any(character not in allowed for character in text):
        raise EvidenceV2Error(f"{label} is not a bounded identifier")
    return text


def _text(value: Any, *, label: str, maximum: int = 512) -> str:
    text = str(value or "")
    if not text or text != text.strip() or len(text.encode("utf-8")) > maximum:
        raise EvidenceV2Error(f"{label} must be nonempty, trimmed, and bounded")
    return text


@dataclass(frozen=True)
class RuntimeComponent:
    component_id: str
    component_kind: str
    version: str
    build_id: str
    absolute_runtime_path: str
    artifact_ref: EvidenceRef

    def __post_init__(self) -> None:
        _identifier(self.component_id, label="component_id")
        _identifier(self.component_kind, label="component_kind")
        _text(self.version, label="component version")
        _text(self.build_id, label="component build_id")
        if not self.absolute_runtime_path.startswith("/") or "\x00" in self.absolute_runtime_path:
            raise EvidenceV2Error("component runtime path must be absolute and NUL-free")
        if self.artifact_ref.absolute_path != self.absolute_runtime_path:
            raise EvidenceV2Error("component runtime path is not its byte-bound artifact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_kind": self.component_kind,
            "version": self.version,
            "build_id": self.build_id,
            "absolute_runtime_path": self.absolute_runtime_path,
            "artifact_ref": self.artifact_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeComponent":
        fields = {
            "component_id",
            "component_kind",
            "version",
            "build_id",
            "absolute_runtime_path",
            "artifact_ref",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise EvidenceV2Error("runtime component fields mismatch")
        return cls(
            component_id=str(value["component_id"]),
            component_kind=str(value["component_kind"]),
            version=str(value["version"]),
            build_id=str(value["build_id"]),
            absolute_runtime_path=str(value["absolute_runtime_path"]),
            artifact_ref=EvidenceRef.from_dict(value["artifact_ref"]),
        )


@dataclass(frozen=True)
class LLMProviderBuildIdentity:
    provider_id: str
    model_id: str
    immutable_model_build_id: str
    endpoint_contract_id: str
    tokenizer_ref: EvidenceRef
    inference_config_ref: EvidenceRef
    provider_attestation_ref: EvidenceRef

    def __post_init__(self) -> None:
        provider = _identifier(self.provider_id, label="provider_id")
        model = _text(self.model_id, label="model_id")
        build = _text(
            self.immutable_model_build_id,
            label="immutable_model_build_id",
        )
        _identifier(self.endpoint_contract_id, label="endpoint_contract_id")
        forbidden = {"latest", "current", "default", "auto", model, provider}
        if build.lower() in {item.lower() for item in forbidden}:
            raise EvidenceV2Error("LLM model build identity is mutable or non-specific")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LLM_PROVIDER_BUILD_SCHEMA,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "immutable_model_build_id": self.immutable_model_build_id,
            "endpoint_contract_id": self.endpoint_contract_id,
            "tokenizer_ref": self.tokenizer_ref.to_dict(),
            "inference_config_ref": self.inference_config_ref.to_dict(),
            "provider_attestation_ref": self.provider_attestation_ref.to_dict(),
            "immutable_build_verified": True,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LLMProviderBuildIdentity":
        fields = {
            "schema_version",
            "provider_id",
            "model_id",
            "immutable_model_build_id",
            "endpoint_contract_id",
            "tokenizer_ref",
            "inference_config_ref",
            "provider_attestation_ref",
            "immutable_build_verified",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise EvidenceV2Error("LLM provider build identity fields mismatch")
        if (
            value["schema_version"] != LLM_PROVIDER_BUILD_SCHEMA
            or value["immutable_build_verified"] is not True
        ):
            raise EvidenceV2Error("LLM provider build identity is not immutable")
        return cls(
            provider_id=str(value["provider_id"]),
            model_id=str(value["model_id"]),
            immutable_model_build_id=str(value["immutable_model_build_id"]),
            endpoint_contract_id=str(value["endpoint_contract_id"]),
            tokenizer_ref=EvidenceRef.from_dict(value["tokenizer_ref"]),
            inference_config_ref=EvidenceRef.from_dict(value["inference_config_ref"]),
            provider_attestation_ref=EvidenceRef.from_dict(value["provider_attestation_ref"]),
        )


def _validate_components(
    components: Sequence[RuntimeComponent],
) -> list[RuntimeComponent]:
    normalized = list(components)
    if [component.component_id for component in normalized] != list(RUNTIME_COMPONENT_ORDER):
        raise EvidenceV2Error("runtime components must preserve the required order")
    if len({component.artifact_ref.byte_sha256 for component in normalized}) != len(normalized):
        raise EvidenceV2Error("runtime components must bind distinct byte artifacts")
    by_id = {component.component_id: component for component in normalized}
    openssl = by_id["openssl_backend"]
    if (
        openssl.component_kind != "openssl-rfc3161-cli"
        or openssl.absolute_runtime_path != PINNED_OPENSSL_PATH
        or not openssl.version.startswith("OpenSSL 3.")
    ):
        raise EvidenceV2Error("runtime capsule does not bind pinned OpenSSL 3")
    pyarrow = by_id["pyarrow_backend"]
    if pyarrow.component_kind != "python-parquet-backend" or not pyarrow.version:
        raise EvidenceV2Error("runtime capsule does not bind the PyArrow backend")
    scipy = by_id["scipy_backend"]
    if scipy.component_kind != "python-statistics-backend" or not scipy.version:
        raise EvidenceV2Error("runtime capsule does not bind the SciPy backend")
    interpreter = by_id["python_interpreter"]
    if interpreter.component_kind != "cpython-interpreter":
        raise EvidenceV2Error("runtime capsule requires a CPython interpreter")
    return normalized


def build_runtime_capsule(
    *,
    protocol_attempt_id: str,
    capsule_id: str,
    components: Sequence[RuntimeComponent],
    environment_controls: Mapping[str, str],
) -> dict[str, Any]:
    normalized = _validate_components(components)
    controls = {str(key): str(value) for key, value in environment_controls.items()}
    if controls != REQUIRED_ENVIRONMENT_CONTROLS:
        raise EvidenceV2Error("runtime environment controls are not hermetic v2 controls")
    return seal_semantic(
        {
            "schema_version": RUNTIME_CAPSULE_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "capsule_id": _identifier(capsule_id, label="capsule_id"),
            "components": [component.to_dict() for component in normalized],
            "environment_controls": {
                key: controls[key] for key in sorted(REQUIRED_ENVIRONMENT_CONTROLS)
            },
            "network_access": "forbidden_during_recompute",
            "provider_discovery": "forbidden",
            "runtime_discovery": "forbidden",
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_runtime_capsule(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "capsule_id",
        "components",
        "environment_controls",
        "network_access",
        "provider_discovery",
        "runtime_discovery",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != RUNTIME_CAPSULE_SCHEMA:
        raise EvidenceV2Error("runtime capsule envelope mismatch")
    _identifier(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _identifier(payload["capsule_id"], label="capsule_id")
    raw_components = payload["components"]
    if not isinstance(raw_components, list):
        raise EvidenceV2Error("runtime capsule components must be a list")
    _validate_components([RuntimeComponent.from_dict(item) for item in raw_components])
    if payload["environment_controls"] != {
        key: REQUIRED_ENVIRONMENT_CONTROLS[key] for key in sorted(REQUIRED_ENVIRONMENT_CONTROLS)
    }:
        raise EvidenceV2Error("runtime capsule environment controls drift")
    if (
        payload["network_access"] != "forbidden_during_recompute"
        or payload["provider_discovery"] != "forbidden"
        or payload["runtime_discovery"] != "forbidden"
    ):
        raise EvidenceV2Error("runtime capsule purity controls drift")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("runtime capsule must be permanently nonauthorizing")
    return payload


def build_frozen_model_bundle(
    *,
    protocol_attempt_id: str,
    branch: str,
    bundle_id: str,
    training_schedule_ref: EvidenceRef,
    training_capture_ref: EvidenceRef,
    feature_contract_ref: EvidenceRef,
    hyperparameter_ref: EvidenceRef,
    serialized_model_ref: EvidenceRef,
    deterministic_inference_entrypoint: str,
    llm_provider_build: LLMProviderBuildIdentity | None = None,
) -> dict[str, Any]:
    if branch not in FORMAL_BRANCHES:
        raise EvidenceV2Error("frozen model bundle branch is not formal v16")
    if (branch == "llm") != (llm_provider_build is not None):
        raise EvidenceV2Error("only the LLM branch requires an immutable provider build identity")
    entrypoint = _text(
        deterministic_inference_entrypoint,
        label="deterministic_inference_entrypoint",
    )
    if ":" not in entrypoint or any(character.isspace() for character in entrypoint):
        raise EvidenceV2Error("deterministic inference entrypoint is not module:function")
    return seal_semantic(
        {
            "schema_version": MODEL_BUNDLE_SCHEMA,
            "protocol_attempt_id": _identifier(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "branch": branch,
            "bundle_id": _identifier(bundle_id, label="bundle_id"),
            "training_epoch": "A",
            "training_schedule_ref": training_schedule_ref.to_dict(),
            "training_capture_ref": training_capture_ref.to_dict(),
            "feature_contract_ref": feature_contract_ref.to_dict(),
            "hyperparameter_ref": hyperparameter_ref.to_dict(),
            "serialized_model_ref": serialized_model_ref.to_dict(),
            "deterministic_inference_entrypoint": entrypoint,
            "llm_provider_build": (
                None if llm_provider_build is None else llm_provider_build.to_dict()
            ),
            "frozen_after_epoch_a": True,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_frozen_model_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "branch",
        "bundle_id",
        "training_epoch",
        "training_schedule_ref",
        "training_capture_ref",
        "feature_contract_ref",
        "hyperparameter_ref",
        "serialized_model_ref",
        "deterministic_inference_entrypoint",
        "llm_provider_build",
        "frozen_after_epoch_a",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != MODEL_BUNDLE_SCHEMA:
        raise EvidenceV2Error("frozen model bundle envelope mismatch")
    branch = str(payload["branch"])
    if branch not in FORMAL_BRANCHES:
        raise EvidenceV2Error("frozen model bundle branch is invalid")
    _identifier(payload["protocol_attempt_id"], label="protocol_attempt_id")
    _identifier(payload["bundle_id"], label="bundle_id")
    if payload["training_epoch"] != "A" or payload["frozen_after_epoch_a"] is not True:
        raise EvidenceV2Error("model bundle is not frozen after epoch A")
    for field in (
        "training_schedule_ref",
        "training_capture_ref",
        "feature_contract_ref",
        "hyperparameter_ref",
        "serialized_model_ref",
    ):
        EvidenceRef.from_dict(payload[field])
    entrypoint = _text(
        payload["deterministic_inference_entrypoint"],
        label="deterministic_inference_entrypoint",
    )
    if ":" not in entrypoint or any(character.isspace() for character in entrypoint):
        raise EvidenceV2Error("deterministic inference entrypoint is not module:function")
    if branch == "llm":
        if not isinstance(payload["llm_provider_build"], Mapping):
            raise EvidenceV2Error("LLM model bundle lacks immutable provider identity")
        LLMProviderBuildIdentity.from_dict(payload["llm_provider_build"])
    elif payload["llm_provider_build"] is not None:
        raise EvidenceV2Error("non-LLM model bundle carries provider identity")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("model bundle must be permanently nonauthorizing")
    return payload


__all__ = [
    "FORMAL_BRANCHES",
    "LLM_PROVIDER_BUILD_SCHEMA",
    "LLMProviderBuildIdentity",
    "MODEL_BUNDLE_SCHEMA",
    "PINNED_OPENSSL_PATH",
    "REQUIRED_ENVIRONMENT_CONTROLS",
    "RUNTIME_CAPSULE_SCHEMA",
    "RUNTIME_COMPONENT_ORDER",
    "RuntimeComponent",
    "build_frozen_model_bundle",
    "build_runtime_capsule",
    "validate_frozen_model_bundle",
    "validate_runtime_capsule",
]
