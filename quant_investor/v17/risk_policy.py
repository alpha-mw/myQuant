"""Non-authorizing, owner-sealed portfolio risk policy snapshots for v17."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import (
    Availability,
    V17ContractError,
    coerce_enum,
    parse_iso_date,
    parse_utc_timestamp,
    require_authority_false,
    require_exact_keys,
    require_identifier,
    require_nonempty_string,
    require_number,
    require_ratio,
)
from .semantic import require_sha256, seal_semantic, validate_semantic_seal
from .storage import atomic_write_json_exact_once, file_sha256, read_json

RISK_POLICY_VERSION = "myquant.v17.portfolio-risk-policy-snapshot.v1"
OWNER_MANDATE_VERSION = "myquant.v17.owner-risk-mandate.v1"

RISK_IDENTITY_KEYS = frozenset(
    {
        "version",
        "policy_id",
        "strategy_id",
        "market",
        "availability",
        "authority",
        "semantic_sha256",
    }
)
RISK_AVAILABLE_KEYS = RISK_IDENTITY_KEYS | frozenset(
    {
        "pit_cutoff",
        "as_of",
        "expires_at",
        "gross_cap",
        "cash_floor",
        "single_name_cap",
        "industry_cap",
        "cluster_cap",
        "beta_cap",
        "stress_loss_cap",
        "adv20_participation_cap",
        "turnover_cap",
        "stress_scenario",
        "source_refs",
    }
)
RISK_UNAVAILABLE_KEYS = RISK_IDENTITY_KEYS | frozenset({"reason"})

SOURCE_REF_KEYS = frozenset({"source_id", "path", "byte_sha256", "semantic_sha256"})


def _parse_cutoff(value: Any, *, label: str) -> datetime:
    if isinstance(value, str) and len(value) == 10:
        parsed_date = parse_iso_date(value, label=label)
        return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
    return parse_utc_timestamp(value, label=label)


def _validate_identity(payload: Mapping[str, Any]) -> Availability:
    if payload.get("version") != RISK_POLICY_VERSION:
        raise V17ContractError("risk policy version mismatch")
    require_identifier(payload.get("policy_id"), label="policy_id")
    require_identifier(payload.get("strategy_id"), label="strategy_id")
    if payload.get("market") != "CN":
        raise V17ContractError("v17 risk policy market must be CN")
    require_authority_false(payload.get("authority"))
    return coerce_enum(
        payload.get("availability"), Availability, label="availability"
    )  # type: ignore[return-value]


def _validate_source_refs(value: Any) -> tuple[dict[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise V17ContractError("source_refs must be a nonempty array")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise V17ContractError(f"source_refs[{index}] must be an object")
        require_exact_keys(item, SOURCE_REF_KEYS, label=f"source_refs[{index}]")
        source_id = require_identifier(item.get("source_id"), label="source_id")
        if source_id in seen:
            raise V17ContractError(f"duplicate source_ref source_id: {source_id}")
        seen.add(source_id)
        source_path = require_nonempty_string(
            item.get("path"), label="source_ref.path", max_chars=2048
        )
        path_parts = PurePosixPath(source_path.replace("\\", "/")).parts
        if ".." in path_parts:
            raise V17ContractError("source_ref.path cannot traverse parent directories")
        result.append(
            {
                "source_id": source_id,
                "path": source_path,
                "byte_sha256": require_sha256(
                    item.get("byte_sha256"), label="source_ref.byte_sha256"
                ),
                "semantic_sha256": require_sha256(
                    item.get("semantic_sha256"), label="source_ref.semantic_sha256"
                ),
            }
        )
    return tuple(result)


def validate_portfolio_risk_policy_snapshot(
    payload: Mapping[str, Any],
    *,
    cutoff: str | None,
) -> dict[str, Any]:
    """Validate exact AVAILABLE/UNAVAILABLE mutual-exclusion shapes.

    ``cutoff`` is explicit: pass ``None`` only for shape-only validation.  A
    supplied cutoff additionally verifies PIT availability and expiry.
    """

    if not isinstance(payload, Mapping):
        raise V17ContractError("risk policy snapshot must be an object")
    sealed = validate_semantic_seal(payload)
    availability = _validate_identity(sealed)
    if availability is Availability.UNAVAILABLE:
        require_exact_keys(sealed, RISK_UNAVAILABLE_KEYS, label="UNAVAILABLE risk policy")
        require_nonempty_string(sealed.get("reason"), label="reason", max_chars=512)
        return sealed

    require_exact_keys(sealed, RISK_AVAILABLE_KEYS, label="AVAILABLE risk policy")
    pit_cutoff = _parse_cutoff(sealed.get("pit_cutoff"), label="pit_cutoff")
    as_of = parse_utc_timestamp(sealed.get("as_of"), label="as_of")
    expires_at = parse_utc_timestamp(sealed.get("expires_at"), label="expires_at")
    if pit_cutoff > as_of:
        raise V17ContractError("risk policy PIT cutoff is later than as_of")
    if expires_at <= as_of:
        raise V17ContractError("risk policy expires_at must be later than as_of")
    if cutoff is not None:
        validation_time = _parse_cutoff(cutoff, label="validation cutoff")
        if pit_cutoff > validation_time:
            raise V17ContractError("risk policy contains post-cutoff evidence")
        if as_of > validation_time:
            raise V17ContractError("risk policy as_of is later than validation cutoff")
        if validation_time >= expires_at:
            raise V17ContractError("risk policy is expired at validation cutoff")

    require_ratio(sealed.get("gross_cap"), label="gross_cap")
    require_ratio(sealed.get("cash_floor"), label="cash_floor")
    for field in (
        "single_name_cap",
        "industry_cap",
        "cluster_cap",
        "adv20_participation_cap",
        "turnover_cap",
    ):
        require_ratio(sealed.get(field), label=field, allow_zero=False)
    require_number(
        sealed.get("beta_cap"),
        label="beta_cap",
        minimum=0.0,
        minimum_exclusive=True,
    )
    require_ratio(sealed.get("stress_loss_cap"), label="stress_loss_cap")
    require_nonempty_string(sealed.get("stress_scenario"), label="stress_scenario", max_chars=256)
    _validate_source_refs(sealed.get("source_refs"))
    return sealed


def build_available_risk_policy_snapshot(
    *,
    policy_id: str,
    strategy_id: str,
    market: str,
    pit_cutoff: str,
    as_of: str,
    expires_at: str,
    gross_cap: float,
    cash_floor: float,
    single_name_cap: float,
    industry_cap: float,
    cluster_cap: float,
    beta_cap: float,
    stress_loss_cap: float,
    adv20_participation_cap: float,
    turnover_cap: float,
    stress_scenario: str,
    source_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = seal_semantic(
        {
            "version": RISK_POLICY_VERSION,
            "policy_id": policy_id,
            "strategy_id": strategy_id,
            "market": market,
            "availability": Availability.AVAILABLE.value,
            "pit_cutoff": pit_cutoff,
            "as_of": as_of,
            "expires_at": expires_at,
            "gross_cap": gross_cap,
            "cash_floor": cash_floor,
            "single_name_cap": single_name_cap,
            "industry_cap": industry_cap,
            "cluster_cap": cluster_cap,
            "beta_cap": beta_cap,
            "stress_loss_cap": stress_loss_cap,
            "adv20_participation_cap": adv20_participation_cap,
            "turnover_cap": turnover_cap,
            "stress_scenario": stress_scenario,
            "source_refs": [dict(item) for item in source_refs],
            "authority": False,
        }
    )
    return validate_portfolio_risk_policy_snapshot(payload, cutoff=None)


def build_unavailable_risk_policy_snapshot(
    *,
    policy_id: str,
    strategy_id: str,
    market: str,
    reason: str,
) -> dict[str, Any]:
    payload = seal_semantic(
        {
            "version": RISK_POLICY_VERSION,
            "policy_id": policy_id,
            "strategy_id": strategy_id,
            "market": market,
            "availability": Availability.UNAVAILABLE.value,
            "reason": reason,
            "authority": False,
        }
    )
    return validate_portfolio_risk_policy_snapshot(payload, cutoff=None)


@dataclass(frozen=True)
class PortfolioRiskPolicySnapshot:
    """Validated immutable view over one risk policy payload."""

    _payload: Mapping[str, Any]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        cutoff: str | None,
    ) -> "PortfolioRiskPolicySnapshot":
        validated = validate_portfolio_risk_policy_snapshot(payload, cutoff=cutoff)
        return cls(_payload=validated)

    @property
    def availability(self) -> Availability:
        return Availability(self._payload["availability"])

    @property
    def semantic_sha256(self) -> str:
        return str(self._payload["semantic_sha256"])

    def to_dict(self) -> dict[str, Any]:
        # The contract is JSON-only; a round-trip prevents nested caller mutation.
        import copy

        return copy.deepcopy(dict(self._payload))


def seal_risk_policy_from_owner_mandate(
    owner_mandate_path: str | Path,
    output_path: str | Path,
    *,
    expected_owner_mandate_sha256: str,
    output_root: str | Path,
    validation_cutoff: str | None,
) -> tuple[dict[str, Any], str]:
    """Validate a byte-bound owner mandate and durably emit a 0600 snapshot.

    The owner mandate is an exact two-field envelope containing an *unsealed*
    ``risk_policy`` object.  Validation completes before the output path is
    touched, so missing/invalid input produces zero writes.
    """

    expected = require_sha256(expected_owner_mandate_sha256, label="expected owner mandate SHA-256")
    before = file_sha256(owner_mandate_path)
    if before != expected:
        raise V17ContractError("owner mandate byte SHA-256 mismatch")
    envelope = read_json(owner_mandate_path)
    after = file_sha256(owner_mandate_path)
    if after != before:
        raise V17ContractError("owner mandate changed during validation")
    require_exact_keys(
        envelope,
        frozenset({"version", "risk_policy"}),
        label="owner risk mandate",
    )
    if envelope.get("version") != OWNER_MANDATE_VERSION:
        raise V17ContractError("owner mandate version mismatch")
    raw_snapshot = envelope.get("risk_policy")
    if not isinstance(raw_snapshot, Mapping):
        raise V17ContractError("owner mandate risk_policy must be an object")
    if "semantic_sha256" in raw_snapshot:
        raise V17ContractError("owner mandate risk_policy must be unsealed")
    snapshot = seal_semantic(raw_snapshot)
    snapshot = validate_portfolio_risk_policy_snapshot(snapshot, cutoff=validation_cutoff)
    output_sha = atomic_write_json_exact_once(output_path, snapshot, root=output_root)
    return snapshot, output_sha


__all__ = [
    "OWNER_MANDATE_VERSION",
    "PortfolioRiskPolicySnapshot",
    "RISK_AVAILABLE_KEYS",
    "RISK_IDENTITY_KEYS",
    "RISK_POLICY_VERSION",
    "RISK_UNAVAILABLE_KEYS",
    "SOURCE_REF_KEYS",
    "build_available_risk_policy_snapshot",
    "build_unavailable_risk_policy_snapshot",
    "seal_risk_policy_from_owner_mandate",
    "validate_portfolio_risk_policy_snapshot",
]
