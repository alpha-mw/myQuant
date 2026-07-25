"""Versioned, content-addressed source bindings for v17 shadow runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from .contracts import (
    Availability,
    V17ContractError,
    coerce_enum,
    parse_utc_timestamp,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_identifier,
    require_nonempty_string,
    require_ratio,
)
from .holdings import validate_holdings_snapshot
from .pretrade import validate_execution_cost_policy
from .risk_policy import validate_portfolio_risk_policy_snapshot
from .semantic import require_sha256, validate_semantic_seal
from .storage import file_sha256, read_json

SOURCE_PLAN_VERSION = "myquant.v17.source-maintenance-plan.v1"
SOURCE_MANIFEST_VERSION = "myquant.v17.source-manifest.v1"

SOURCE_PLAN_KEYS = frozenset(
    {
        "version",
        "manifest_id",
        "market",
        "cutoff",
        "created_at",
        "sources",
        "authority",
        "semantic_sha256",
    }
)
SOURCE_MANIFEST_KEYS = SOURCE_PLAN_KEYS
AVAILABLE_PLAN_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "role",
        "availability",
        "path",
        "expected_byte_sha256",
    }
)
UNAVAILABLE_SOURCE_KEYS = frozenset({"source_id", "role", "availability", "reason"})
AVAILABLE_MANIFEST_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "role",
        "availability",
        "object_path",
        "byte_sha256",
        "size_bytes",
    }
)

RANK_REQUIRED_ROLES = frozenset(
    {
        "market_pointer",
        "market_snapshot_manifest",
        "market_snapshot",
        "cn_open_day_calendar",
        "pit_membership_generation_manifest",
        "pit_membership_canonical",
        "pit_membership",
        "fundamental_generation_pointer",
        "fundamental_generation_manifest",
        "fundamental_snapshot",
        "fundamental_metric_history",
        "fundamental_forward_calibration",
        "H00300_total_return",
        "dividend_total_return",
        "official_delisting_cash",
        "rank_projection_verification_receipt",
        "fundamental_calibration_input_manifest",
        "fundamental_calibration_verification_receipt",
        "fundamental_calibration_raw_predictors",
        "calibration_pit_membership_history",
        "quant_timing_calibration",
        "quant_calibration_input_manifest",
        "quant_calibration_verification_receipt",
        "quant_calibration_raw_bars",
        "sealed_deep_evidence",
    }
)
PORTFOLIO_REQUIRED_ROLES = frozenset(
    {
        "macro_overlay_mapping",
        "macro_overlay_input",
        "markov_overlay_mapping",
        "markov_overlay_input",
        "holdings",
        "holdings_pointer",
        "risk_policy",
        "execution_cost_policy",
        "tradability_evidence",
        "risk_model_input",
        "cluster_mapping",
        "portfolio_projection_verification_receipt",
    }
)
REQUIRED_SOURCE_ROLES = RANK_REQUIRED_ROLES | PORTFOLIO_REQUIRED_ROLES


class V17SourceBindingError(V17ContractError):
    """A source plan, manifest, or content-addressed object is invalid."""


def _validate_source_items(
    raw: Any,
    *,
    manifest: bool,
) -> tuple[dict[str, Any], ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise V17SourceBindingError("sources must be a nonempty array")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_roles: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise V17SourceBindingError(f"sources[{index}] must be an object")
        source_id = require_identifier(item.get("source_id"), label=f"sources[{index}].source_id")
        role = require_identifier(item.get("role"), label=f"sources[{index}].role")
        if source_id in seen_ids:
            raise V17SourceBindingError(f"duplicate source_id: {source_id}")
        if role in seen_roles:
            raise V17SourceBindingError(f"duplicate source role: {role}")
        seen_ids.add(source_id)
        seen_roles.add(role)
        availability = coerce_enum(
            item.get("availability"),
            Availability,
            label=f"sources[{index}].availability",
        )
        if availability is Availability.UNAVAILABLE:
            require_exact_keys(
                item,
                UNAVAILABLE_SOURCE_KEYS,
                label=f"UNAVAILABLE sources[{index}]",
            )
            normalized.append(
                {
                    "source_id": source_id,
                    "role": role,
                    "availability": Availability.UNAVAILABLE.value,
                    "reason": require_nonempty_string(
                        item.get("reason"),
                        label=f"sources[{index}].reason",
                        max_chars=512,
                    ),
                }
            )
            continue

        expected_keys = AVAILABLE_MANIFEST_SOURCE_KEYS if manifest else AVAILABLE_PLAN_SOURCE_KEYS
        require_exact_keys(
            item,
            expected_keys,
            label=f"AVAILABLE sources[{index}]",
        )
        if manifest:
            size = item.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise V17SourceBindingError(f"sources[{index}].size_bytes must be positive")
            normalized.append(
                {
                    "source_id": source_id,
                    "role": role,
                    "availability": Availability.AVAILABLE.value,
                    "object_path": require_nonempty_string(
                        item.get("object_path"),
                        label=f"sources[{index}].object_path",
                        max_chars=2048,
                    ),
                    "byte_sha256": require_sha256(
                        item.get("byte_sha256"),
                        label=f"sources[{index}].byte_sha256",
                    ),
                    "size_bytes": size,
                }
            )
        else:
            normalized.append(
                {
                    "source_id": source_id,
                    "role": role,
                    "availability": Availability.AVAILABLE.value,
                    "path": require_nonempty_string(
                        item.get("path"),
                        label=f"sources[{index}].path",
                        max_chars=4096,
                    ),
                    "expected_byte_sha256": require_sha256(
                        item.get("expected_byte_sha256"),
                        label=f"sources[{index}].expected_byte_sha256",
                    ),
                }
            )
    expected_order = sorted(
        normalized,
        key=lambda item: (str(item["role"]), str(item["source_id"])),
    )
    if normalized != expected_order:
        raise V17SourceBindingError("sources must be canonically ordered by role then source_id")
    missing = sorted(REQUIRED_SOURCE_ROLES - seen_roles)
    if missing:
        raise V17SourceBindingError(f"required source roles missing: {missing}")
    return tuple(normalized)


def validate_source_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, SOURCE_PLAN_KEYS, label="source maintenance plan")
    if sealed.get("version") != SOURCE_PLAN_VERSION:
        raise V17SourceBindingError("source maintenance plan version mismatch")
    require_identifier(sealed.get("manifest_id"), label="manifest_id")
    if sealed.get("market") != "CN":
        raise V17SourceBindingError("source maintenance plan market must be CN")
    cutoff = parse_utc_timestamp(sealed.get("cutoff"), label="cutoff")
    created = parse_utc_timestamp(sealed.get("created_at"), label="created_at")
    if created < cutoff:
        raise V17SourceBindingError("source plan created_at precedes cutoff")
    require_authority_false(sealed.get("authority"))
    _validate_source_items(sealed.get("sources"), manifest=False)
    return sealed


def _fixed_object_path(repo_root: Path, raw_path: Any, byte_sha: str) -> Path:
    relative = PurePosixPath(str(raw_path or ""))
    expected = PurePosixPath("data/private/v17_sources/objects") / byte_sha[:2] / f"{byte_sha}.bin"
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise V17SourceBindingError("source object path mismatch")
    return repo_root / Path(*relative.parts)


def validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    revalidate_objects: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).absolute()
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, SOURCE_MANIFEST_KEYS, label="source manifest")
    if sealed.get("version") != SOURCE_MANIFEST_VERSION:
        raise V17SourceBindingError("source manifest version mismatch")
    require_identifier(sealed.get("manifest_id"), label="manifest_id")
    if sealed.get("market") != "CN":
        raise V17SourceBindingError("source manifest market must be CN")
    cutoff = parse_utc_timestamp(sealed.get("cutoff"), label="cutoff")
    created = parse_utc_timestamp(sealed.get("created_at"), label="created_at")
    if created < cutoff:
        raise V17SourceBindingError("source manifest created_at precedes cutoff")
    require_authority_false(sealed.get("authority"))
    sources = _validate_source_items(sealed.get("sources"), manifest=True)
    if revalidate_objects:
        for item in sources:
            if item["availability"] != Availability.AVAILABLE.value:
                continue
            declared_sha = str(item["byte_sha256"])
            object_path = _fixed_object_path(
                root,
                item["object_path"],
                declared_sha,
            )
            if file_sha256(object_path) != declared_sha:
                raise V17SourceBindingError(f"source object byte SHA mismatch: {item['role']}")
            entry = os.stat(object_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(entry.st_mode)
                or entry.st_nlink != 1
                or stat.S_IMODE(entry.st_mode) != 0o600
                or entry.st_size != item["size_bytes"]
            ):
                raise V17SourceBindingError(f"source object identity mismatch: {item['role']}")
    return sealed


def _resolve_manifest_path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    expected_parent = repo_root / "data" / "private" / "v17_sources" / "manifests"
    if candidate.parent.absolute() != expected_parent.absolute():
        raise V17SourceBindingError("source manifest path is outside fixed manifest root")
    if candidate.suffix != ".json" or candidate.name != Path(candidate.name).name:
        raise V17SourceBindingError("source manifest filename invalid")
    return candidate


@dataclass(frozen=True)
class SourceBindingBundle:
    manifest: Mapping[str, Any]
    manifest_path: Path
    manifest_byte_sha256: str
    by_role: Mapping[str, Mapping[str, Any]]
    effective_availability_by_role: Mapping[str, str]

    @property
    def rank_unavailable_roles(self) -> tuple[str, ...]:
        return tuple(
            role
            for role in sorted(RANK_REQUIRED_ROLES)
            if self.effective_availability_by_role[role] != Availability.AVAILABLE.value
        )

    @property
    def portfolio_unavailable_roles(self) -> tuple[str, ...]:
        blocked: set[str] = {
            role
            for role in (
                "holdings",
                "risk_policy",
                "execution_cost_policy",
                "tradability_evidence",
                "risk_model_input",
                "cluster_mapping",
                "holdings_pointer",
                "portfolio_projection_verification_receipt",
            )
            if self.effective_availability_by_role[role] != Availability.AVAILABLE.value
        }
        for name in ("macro", "markov"):
            input_role = f"{name}_overlay_input"
            mapping_role = f"{name}_overlay_mapping"
            input_state = self.effective_availability_by_role[input_role]
            if input_state == "DISABLED":
                continue
            if input_state != Availability.AVAILABLE.value:
                blocked.add(input_role)
            elif self.effective_availability_by_role[mapping_role] != Availability.AVAILABLE.value:
                blocked.add(mapping_role)
        return tuple(sorted(blocked))

    def object_path(self, role: str) -> Path:
        item = self.by_role[role]
        if item["availability"] != Availability.AVAILABLE.value:
            raise V17SourceBindingError(f"source role unavailable: {role}")
        return _fixed_object_path(
            self.manifest_path.parents[4],
            item["object_path"],
            str(item["byte_sha256"]),
        )


def _validate_overlay_source_payload(
    payload: Mapping[str, Any],
    *,
    expected_name: str,
    expected_cutoff: str,
) -> tuple[str, str | None]:
    if not isinstance(payload, Mapping):
        raise V17SourceBindingError(f"{expected_name} overlay input must be an object")
    sealed = validate_semantic_seal(payload)
    require_exact_keys(
        sealed,
        frozenset(
            {
                "version",
                "market",
                "cutoff",
                "name",
                "enabled",
                "availability",
                "state",
                "reason",
                "authority",
                "semantic_sha256",
            }
        ),
        label=f"{expected_name} regime source input",
    )
    if sealed.get("version") != "myquant.v17.regime-input-source.v1":
        raise V17SourceBindingError(f"{expected_name} regime source version mismatch")
    if sealed.get("market") != "CN" or sealed.get("cutoff") != expected_cutoff:
        raise V17SourceBindingError(f"{expected_name} regime source market/cutoff mismatch")
    parse_utc_timestamp(sealed.get("cutoff"), label=f"{expected_name}.cutoff")
    require_authority_false(sealed.get("authority"))
    if sealed.get("name") != expected_name:
        raise V17SourceBindingError(f"overlay input name must be {expected_name}")
    enabled = require_bool(sealed.get("enabled"), label=f"{expected_name}.enabled")
    availability = sealed.get("availability")
    state = sealed.get("state")
    reason = sealed.get("reason")
    if not enabled:
        if availability is not None or state is not None or reason is not None:
            raise V17SourceBindingError(
                f"disabled {expected_name} regime source must not carry values"
            )
        return "DISABLED", None
    if availability == Availability.UNAVAILABLE.value:
        if state is not None:
            raise V17SourceBindingError(
                f"unavailable {expected_name} regime source cannot carry state"
            )
        require_nonempty_string(
            reason,
            label=f"{expected_name}.reason",
            max_chars=512,
        )
        return Availability.UNAVAILABLE.value, None
    if availability != Availability.AVAILABLE.value or reason is not None:
        raise V17SourceBindingError(f"available {expected_name} regime source shape invalid")
    canonical_state = require_identifier(state, label=f"{expected_name}.state")
    return Availability.AVAILABLE.value, canonical_state


def _validate_overlay_mapping_payload(
    payload: Mapping[str, Any],
    *,
    expected_name: str,
    expected_cutoff: str,
    required_state: str | None,
) -> None:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(
        sealed,
        frozenset(
            {
                "version",
                "market",
                "cutoff",
                "name",
                "states",
                "authority",
                "semantic_sha256",
            }
        ),
        label=f"{expected_name} regime mapping",
    )
    if sealed.get("version") != "myquant.v17.regime-mapping-source.v1":
        raise V17SourceBindingError(f"{expected_name} regime mapping version mismatch")
    if (
        sealed.get("market") != "CN"
        or sealed.get("cutoff") != expected_cutoff
        or sealed.get("name") != expected_name
    ):
        raise V17SourceBindingError(f"{expected_name} regime mapping identity mismatch")
    parse_utc_timestamp(sealed.get("cutoff"), label=f"{expected_name}.cutoff")
    require_authority_false(sealed.get("authority"))
    states = sealed.get("states")
    if not isinstance(states, Mapping) or not states:
        raise V17SourceBindingError(f"{expected_name} regime states must be nonempty")
    for state, cell in states.items():
        require_identifier(state, label=f"{expected_name}.state")
        if not isinstance(cell, Mapping):
            raise V17SourceBindingError(f"{expected_name}.{state} must be an object")
        require_exact_keys(
            cell,
            frozenset({"gross_cap", "cash_floor"}),
            label=f"{expected_name}.{state}",
        )
        require_ratio(cell.get("gross_cap"), label=f"{expected_name}.{state}.gross_cap")
        require_ratio(cell.get("cash_floor"), label=f"{expected_name}.{state}.cash_floor")
    if required_state is not None and required_state not in states:
        raise V17SourceBindingError(f"{expected_name} enabled state missing from sealed mapping")


def load_source_manifest_binding(
    repo_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    validate_authority_payloads: bool = True,
) -> SourceBindingBundle:
    root = Path(repo_root).absolute()
    expected = require_sha256(expected_manifest_sha256, label="expected source manifest SHA-256")
    path = _resolve_manifest_path(root, manifest_path)
    before = file_sha256(path)
    if before != expected:
        raise V17SourceBindingError("source manifest byte SHA mismatch")
    payload = read_json(path)
    after = file_sha256(path)
    if after != before:
        raise V17SourceBindingError("source manifest changed during read")
    manifest = validate_source_manifest(payload, repo_root=root, revalidate_objects=True)
    if path.name != f"{manifest['manifest_id']}.json":
        raise V17SourceBindingError("source manifest identity/filename mismatch")
    by_role = {str(item["role"]): item for item in manifest["sources"]}
    effective = {role: str(item["availability"]) for role, item in by_role.items()}
    bundle = SourceBindingBundle(
        manifest=manifest,
        manifest_path=path,
        manifest_byte_sha256=before,
        by_role=by_role,
        effective_availability_by_role=effective,
    )
    if validate_authority_payloads:
        if by_role["holdings"]["availability"] == Availability.AVAILABLE.value:
            holdings = validate_holdings_snapshot(
                read_json(bundle.object_path("holdings")),
                cutoff=str(manifest["cutoff"]),
            )
            effective["holdings"] = str(holdings["availability"])
        if by_role["risk_policy"]["availability"] == Availability.AVAILABLE.value:
            risk_policy = validate_portfolio_risk_policy_snapshot(
                read_json(bundle.object_path("risk_policy")),
                cutoff=str(manifest["cutoff"]),
            )
            effective["risk_policy"] = str(risk_policy["availability"])
        if by_role["execution_cost_policy"]["availability"] == Availability.AVAILABLE.value:
            validate_execution_cost_policy(read_json(bundle.object_path("execution_cost_policy")))
        for name in ("macro", "markov"):
            role = f"{name}_overlay_input"
            required_state: str | None = None
            if by_role[role]["availability"] == Availability.AVAILABLE.value:
                effective[role], required_state = _validate_overlay_source_payload(
                    read_json(bundle.object_path(role)),
                    expected_name=name,
                    expected_cutoff=str(manifest["cutoff"]),
                )
            mapping_role = f"{name}_overlay_mapping"
            if by_role[mapping_role]["availability"] == Availability.AVAILABLE.value:
                _validate_overlay_mapping_payload(
                    read_json(bundle.object_path(mapping_role)),
                    expected_name=name,
                    expected_cutoff=str(manifest["cutoff"]),
                    required_state=required_state,
                )
    return bundle


__all__ = [
    "AVAILABLE_MANIFEST_SOURCE_KEYS",
    "AVAILABLE_PLAN_SOURCE_KEYS",
    "PORTFOLIO_REQUIRED_ROLES",
    "RANK_REQUIRED_ROLES",
    "REQUIRED_SOURCE_ROLES",
    "SOURCE_MANIFEST_KEYS",
    "SOURCE_MANIFEST_VERSION",
    "SOURCE_PLAN_KEYS",
    "SOURCE_PLAN_VERSION",
    "SourceBindingBundle",
    "UNAVAILABLE_SOURCE_KEYS",
    "V17SourceBindingError",
    "load_source_manifest_binding",
    "validate_source_manifest",
    "validate_source_plan",
]
