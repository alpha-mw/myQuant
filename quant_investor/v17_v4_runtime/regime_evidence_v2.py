"""Causal, immutable V17 v4 regime-evidence v2 producer and replay.

This module is deliberately independent of the legacy mutable Markov history.
It consumes only caller-pinned canonical artifacts, performs fixed Decimal
inference, and publishes one exact slot for a strategy/effective session.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
import hashlib
from pathlib import Path, PurePosixPath
import re
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
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.resources import (
    PackageResourceError,
    load_packaged_json,
    read_packaged_asset,
)
from quant_investor.v17_v4_contract.schema_validation import (
    SchemaValidationError,
    schema_path_for_version,
)
from quant_investor.v17_v4_contract.validators import ArtifactContractError
from quant_investor.v17_v4_contract.validators import ArtifactContractError
from quant_investor.v17_v4_runtime.source_storage import (
    SourceCASMismatch,
    SourceExactOnceConflict,
    SourceNotFoundError,
    SourceStorageError,
    SourceStorageSecurityError,
    SourceStore,
    WriteResult,
)

REGIME_EVIDENCE_VERSION: Final = "myquant.v17.v4.regime-evidence.v2"
FEATURE_SNAPSHOT_VERSION: Final = "myquant.v17.v4.regime-feature-snapshot.v1"
TRANSITION_SNAPSHOT_VERSION: Final = "myquant.v17.v4.regime-transition-matrix-snapshot.v1"
MODEL_SNAPSHOT_VERSION: Final = "myquant.v17.v4.regime-model-snapshot.v1"
INFERENCE_POLICY_VERSION: Final = "myquant.v17.v4.regime-inference-policy.v1"
INFERENCE_POLICY_PATH: Final = "resources/regime_inference_policy.v1.json"

REGIME_SOURCE_ROOT: Final = PurePosixPath("data/private/v17_v4_sources/regime_evidence")
_PRODUCER_LOCK: Final = REGIME_SOURCE_ROOT / ".producer.lock"
_OUTPUT_NAME: Final = "regime_evidence.v2.json"

STATE_ORDER: Final = (
    "趋势上涨",
    "震荡低波",
    "震荡高波",
    "趋势下跌",
    "未知",
)
TEMPORAL_MODE: Final = "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION"
INFERENCE_KIND: Final = "FILTERED_CAUSAL"
MODEL_KIND: Final = "PINNED_RULE_BASED_NO_TRAINING"
TRANSITION_SOURCE: Final = "PINNED_NATIVE_DEFAULT_V1"
MODEL_ID: Final = "v17-v4-native-regime-filtered-model"
MODEL_VERSION: Final = "PINNED_RULE_BASED_NO_TRAINING_V1"
FORMULA_VERSION: Final = "NATIVE_HEURISTIC_LIKELIHOOD_DECIMAL_V1"
HARD_STATE_DERIVATION: Final = "SEALED_ARGMAX_POLICY_V1"
BOOTSTRAP_OBSERVED_SESSION: Final = "2026-07-29"
BOOTSTRAP_DECISION_SESSION: Final = "2026-07-30"

DECIMAL_PLACES: Final = 12
DECIMAL_QUANTUM: Final = Decimal("0.000000000001")
DECIMAL_ONE: Final = Decimal("1.000000000000")
MINIMUM_MARKET_SAMPLE: Final = 30
NEW_PUBLICATION_TOLERANCE_SECONDS: Final = 300
MAX_CLOSURE_DEPTH: Final = 5
MAX_CLOSURE_NODES: Final = 16
MAX_JSON_BYTES_PER_NODE: Final = 2 * 1024 * 1024
MAX_TOTAL_JSON_BYTES: Final = 8 * 1024 * 1024
MAX_RAW_BYTES_PER_FILE: Final = 64 * 1024 * 1024
MAX_RAW_TOTAL_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_REFERENCES: Final = 128
MAX_VALIDATION_SECONDS: Final = 10

_TEMPORAL_CONTRACT_MARKERS: Final = (
    "after observed",
    "available_at",
    "before",
    "contiguous",
    "created_at",
    "cutoff",
    "effective_session",
    "future",
    "observed_through_session",
    "predates",
    "precedes",
    "published_at",
    "session",
    "time closure",
)

CALENDAR_TERMINAL_VERSION: Final = "myquant.v17.v4.regime-calendar-terminal.v1"
PIT_TERMINAL_VERSION: Final = "myquant.v17.v4.regime-pit-membership-terminal.v1"
MARKET_TERMINAL_VERSION: Final = "myquant.v17.v4.regime-market-terminal.v1"
LOCATOR_TERMINAL_VERSION: Final = "myquant.v17.v4.regime-source-locator-terminal.v1"

TRUE_CURRENT_CANONICAL_INPUT_GAP: Final = "TRUE_CURRENT_CANONICAL_INPUT_GAP"
INPUT_TAMPER_BLOCKER: Final = "REGIME_EVIDENCE_V2_INPUT_TAMPER"
SEMANTIC_BLOCKER: Final = "REGIME_EVIDENCE_V2_SEMANTIC_MISMATCH"
IDENTITY_BLOCKER: Final = "REGIME_EVIDENCE_V2_IDENTITY_MISMATCH"
TEMPORAL_BLOCKER: Final = "REGIME_EVIDENCE_V2_TEMPORAL_VIOLATION"
CONFLICT_BLOCKER: Final = "REGIME_EVIDENCE_V2_EXACT_ONCE_CONFLICT"
SCOPE_BLOCKER: Final = "REGIME_EVIDENCE_V2_SCOPE_VIOLATION"
IMPLEMENTATION_BLOCKER: Final = "REGIME_EVIDENCE_V2_IMPLEMENTATION_SHA_MISMATCH"

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
_POLICY_REF_FIELDS: Final = frozenset(
    {"byte_sha256", "relative_path", "semantic_sha256", "version"}
)
_EVIDENCE_ID_EXCLUDED: Final = frozenset(
    {
        "available_at",
        "blocker_codes",
        "computed_at",
        "created_at",
        "evidence_id",
        "published_at",
        "semantic_sha256",
    }
)
_STRATEGY_RE: Final = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    re.ASCII,
)
_DECIMAL_TEXT_RE: Final = re.compile(
    r"^-?(?:0|[1-9][0-9]*)\.[0-9]{12}$",
    re.ASCII,
)
_SHANGHAI: Final = ZoneInfo("Asia/Shanghai")
_UTC: Final = timezone.utc


class RegimeEvidenceV2Error(RuntimeError):
    """Typed fail-closed error for producer/replay callers."""

    exit_code = 2

    def __init__(self, blocker_code: str, detail: str) -> None:
        self.blocker_code = blocker_code
        self.blocker_codes = (blocker_code,)
        self.status = "BLOCKED"
        self.detail = detail
        super().__init__(f"{blocker_code}: {detail}")

    def to_status(self) -> dict[str, Any]:
        return {
            "blocker_codes": list(self.blocker_codes),
            "detail": self.detail,
            "status": self.status,
        }


class RegimeEvidenceV2InputGap(RegimeEvidenceV2Error):
    """A real missing owner-prepositioned input, never a synthetic fallback."""

    def __init__(self, detail: str) -> None:
        super().__init__(TRUE_CURRENT_CANONICAL_INPUT_GAP, detail)
        self.status = TRUE_CURRENT_CANONICAL_INPUT_GAP


class RegimeEvidenceV2Conflict(RegimeEvidenceV2Error):
    """The fixed output slot already contains different canonical bytes."""

    def __init__(self, detail: str) -> None:
        super().__init__(CONFLICT_BLOCKER, detail)


def _contract_validation_error(
    exc: ArtifactContractError,
    *,
    context: str,
) -> RegimeEvidenceV2Error:
    """Keep semantic-validator failures on the typed runtime blocker surface."""

    detail = str(exc)
    blocker = (
        TEMPORAL_BLOCKER
        if any(marker in detail.lower() for marker in _TEMPORAL_CONTRACT_MARKERS)
        else INPUT_TAMPER_BLOCKER
    )
    return RegimeEvidenceV2Error(
        blocker,
        f"{context}: {detail}",
    )


@dataclass(frozen=True)
class RegimeEvidenceV2BuildResult:
    status: str
    evidence_id: str
    evidence_path: str
    evidence_sha256: str
    created: bool
    reused: bool
    document: dict[str, Any]


def regime_evidence_v2_authority_attestation() -> dict[str, Any]:
    """Return the immutable non-authority claims carried by every v2."""

    return {
        "authority": dict(_NO_AUTHORITY),
        "formal_activation_eligible": False,
        "no_retroactive_causal_backfill": True,
        "performance_evidence_eligible": False,
        "promotion_eligible": False,
        "same_session_execution_eligible": False,
        "shadow_only": True,
    }


def implementation_sha256() -> str:
    """Hash the exact implementation bytes used for this inference."""

    try:
        raw = Path(__file__).read_bytes()
    except OSError as exc:
        raise RegimeEvidenceV2Error(
            IMPLEMENTATION_BLOCKER,
            "runtime source is unreadable",
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def regime_evidence_v2_path(
    *,
    strategy_id: str,
    effective_session: str,
) -> str:
    strategy = _strategy_id(strategy_id)
    session = _session(effective_session, label="effective_session")
    return str(REGIME_SOURCE_ROOT / strategy / session / _OUTPUT_NAME)


def regime_evidence_v2_id(document: Mapping[str, Any]) -> str:
    """Derive the caller-precomputable evidence identity."""

    if not isinstance(document, Mapping):
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "evidence identity source must be an object",
        )
    stable = {
        str(key): value for key, value in document.items() if key not in _EVIDENCE_ID_EXCLUDED
    }
    try:
        return hashlib.sha256(canonical_bytes(stable)).hexdigest()
    except CanonicalContractError as exc:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "evidence identity source is noncanonical",
        ) from exc


def _snapshot_id(
    document: Mapping[str, Any],
    *,
    identity_field: str,
) -> str:
    stable = {
        str(key): value
        for key, value in document.items()
        if key not in {identity_field, "semantic_sha256"}
    }
    return hashlib.sha256(canonical_bytes(stable)).hexdigest()


def _strategy_id(value: Any) -> str:
    try:
        normalized = require_opaque_id(value, label="strategy_id")
    except IdentityContractError as exc:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "strategy_id is not canonical",
        ) from exc
    if _STRATEGY_RE.fullmatch(normalized) is None:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "strategy_id is not a safe single path component",
        )
    return normalized


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise RegimeEvidenceV2Error(TEMPORAL_BLOCKER, f"{label} is not text")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            f"{label} is not a real ISO date",
        ) from exc
    if parsed.isoformat() != value:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            f"{label} is noncanonical",
        )
    return value


def _timestamp(value: Any, *, label: str) -> tuple[str, datetime]:
    try:
        text = require_utc_timestamp(value, label=label)
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except (IdentityContractError, ValueError) as exc:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            f"{label} is not canonical UTC seconds",
        ) from exc
    return text, parsed


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is not str:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} must be Decimal text",
        )
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} is not Decimal text",
        ) from exc
    if not result.is_finite():
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} is nonfinite",
        )
    return result


def _probability(value: Any, *, label: str) -> Decimal:
    result = _decimal(value, label=label)
    if result < 0 or result > 1:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} is outside [0,1]",
        )
    return result


def _decimal_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 60
        context.rounding = ROUND_HALF_EVEN
        quantized = value.quantize(DECIMAL_QUANTUM)
    text = format(quantized, ".12f")
    if _DECIMAL_TEXT_RE.fullmatch(text) is None:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "Decimal serialization is noncanonical",
        )
    return text


def _clamp(value: Decimal) -> Decimal:
    return min(Decimal(1), max(Decimal(0), value))


def _normalize_largest_remainder(
    values: Mapping[str, Decimal],
    *,
    label: str,
) -> dict[str, str]:
    if set(values) != set(STATE_ORDER):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} state domain mismatch",
        )
    if any(not value.is_finite() or value < 0 for value in values.values()):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} contains negative or nonfinite values",
        )
    total = sum((values[state] for state in STATE_ORDER), Decimal(0))
    if total <= 0:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} total is not positive",
        )
    units_total = 10**DECIMAL_PLACES
    with localcontext() as context:
        context.prec = 60
        context.rounding = ROUND_HALF_EVEN
        exact_units = {state: values[state] / total * Decimal(units_total) for state in STATE_ORDER}
    floor_units = {
        state: int(exact_units[state].to_integral_value(rounding=ROUND_FLOOR))
        for state in STATE_ORDER
    }
    remaining = units_total - sum(floor_units.values())
    if remaining < 0 or remaining > len(STATE_ORDER):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} largest-remainder accounting failed",
        )
    ranked = sorted(
        STATE_ORDER,
        key=lambda state: (
            -(exact_units[state] - Decimal(floor_units[state])),
            STATE_ORDER.index(state),
        ),
    )
    for state in ranked[:remaining]:
        floor_units[state] += 1
    result = {state: f"{floor_units[state] / units_total:.12f}" for state in STATE_ORDER}
    if sum((_decimal(value, label=label) for value in result.values()), Decimal(0)) != (
        DECIMAL_ONE
    ):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} does not sum exactly to one",
        )
    return result


def _serialized_probabilities(
    value: Any,
    *,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(STATE_ORDER):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} must use the exact native state domain",
        )
    result = {
        state: _decimal_text(_probability(value[state], label=f"{label}.{state}"))
        for state in STATE_ORDER
    }
    if result != value:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} is not fixed 12-place Decimal text",
        )
    if sum((_decimal(item, label=label) for item in result.values()), Decimal(0)) != (DECIMAL_ONE):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} does not sum exactly to one",
        )
    return result


def _derive_scores_and_likelihoods(
    feature: Mapping[str, Any],
) -> tuple[str, str, str, dict[str, str]]:
    average_return = _decimal(
        feature.get("average_return"),
        label="feature.average_return",
    )
    average_volatility = _decimal(
        feature.get("average_volatility"),
        label="feature.average_volatility",
    )
    breadth = _clamp(_decimal(feature.get("breadth"), label="feature.breadth"))
    momentum = _clamp(
        _decimal(
            feature.get("momentum_share"),
            label="feature.momentum_share",
        )
    )
    fake_breakout = _clamp(
        _decimal(
            feature.get("fake_breakout_share"),
            label="feature.fake_breakout_share",
        )
    )
    drawdown = _clamp(
        _decimal(
            feature.get("median_drawdown"),
            label="feature.median_drawdown",
        )
    )
    liquidity = _clamp(
        _decimal(
            feature.get("average_liquidity"),
            label="feature.average_liquidity",
        )
    )
    macro = _decimal(feature.get("macro_score"), label="feature.macro_score")
    with localcontext() as context:
        context.prec = 60
        context.rounding = ROUND_HALF_EVEN
        normalized_return = _clamp((average_return + Decimal("0.015")) / Decimal("0.030"))
        macro_normalized = _clamp((macro + Decimal(1)) / Decimal(2))
        volatility = _clamp(average_volatility / Decimal("0.035"))
        pressure = _clamp(
            Decimal("0.40") * drawdown / Decimal("0.20")
            + Decimal("0.35") * fake_breakout
            + Decimal("0.25") * (Decimal(1) - liquidity)
        )
        risk_on = _clamp(
            Decimal("0.30") * normalized_return
            + Decimal("0.25") * breadth
            + Decimal("0.20") * momentum
            + Decimal("0.15") * liquidity
            + Decimal("0.10") * macro_normalized
            - Decimal("0.25") * volatility
            - Decimal("0.20") * pressure
        )
        neutral = Decimal(1) - min(
            abs(risk_on - Decimal("0.50")) * Decimal(2),
            Decimal(1),
        )
        raw = {
            "趋势上涨": (
                Decimal("0.05")
                + Decimal("0.55") * risk_on
                + Decimal("0.20") * breadth
                + Decimal("0.15") * (Decimal(1) - volatility)
                + Decimal("0.05") * momentum
            ),
            "震荡低波": (
                Decimal("0.05")
                + Decimal("0.35") * (Decimal(1) - volatility)
                + Decimal("0.30") * neutral
                + Decimal("0.20") * (Decimal(1) - pressure)
                + Decimal("0.10") * breadth
            ),
            "震荡高波": (
                Decimal("0.05")
                + Decimal("0.35") * volatility
                + Decimal("0.35") * pressure
                + Decimal("0.15") * neutral
                + Decimal("0.05") * fake_breakout
            ),
            "趋势下跌": (
                Decimal("0.05")
                + Decimal("0.45") * (Decimal(1) - risk_on)
                + Decimal("0.20") * (Decimal(1) - breadth)
                + Decimal("0.20") * pressure
                + Decimal("0.10") * volatility
            ),
            "未知": Decimal("0.04"),
        }
    return (
        _decimal_text(risk_on),
        _decimal_text(volatility),
        _decimal_text(pressure),
        _normalize_largest_remainder(raw, label="state_likelihoods"),
    )


def _posterior(
    *,
    previous: Mapping[str, Any],
    transition_matrix: Mapping[str, Any],
    likelihoods: Mapping[str, Any],
) -> dict[str, str]:
    previous_values = {
        state: _probability(
            previous.get(state),
            label=f"previous.{state}",
        )
        for state in STATE_ORDER
    }
    likelihood_values = {
        state: _probability(
            likelihoods.get(state),
            label=f"likelihoods.{state}",
        )
        for state in STATE_ORDER
    }
    if type(transition_matrix) is not dict or set(transition_matrix) != set(STATE_ORDER):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "transition matrix row order mismatch",
        )
    matrix: dict[str, dict[str, Decimal]] = {}
    for source in STATE_ORDER:
        row = transition_matrix.get(source)
        if type(row) is not dict or set(row) != set(STATE_ORDER):
            raise RegimeEvidenceV2Error(
                SEMANTIC_BLOCKER,
                f"transition matrix column order mismatch for {source}",
            )
        matrix[source] = {
            target: _probability(
                row[target],
                label=f"transition.{source}.{target}",
            )
            for target in STATE_ORDER
        }
        if sum(matrix[source].values(), Decimal(0)) != DECIMAL_ONE:
            raise RegimeEvidenceV2Error(
                SEMANTIC_BLOCKER,
                f"transition row {source} does not sum to one",
            )
    with localcontext() as context:
        context.prec = 60
        context.rounding = ROUND_HALF_EVEN
        prior = {
            target: sum(
                (previous_values[source] * matrix[source][target] for source in STATE_ORDER),
                Decimal(0),
            )
            for target in STATE_ORDER
        }
        unnormalized = {state: prior[state] * likelihood_values[state] for state in STATE_ORDER}
    return _normalize_largest_remainder(
        unnormalized,
        label="state_probabilities",
    )


def _hard_state(probabilities: Mapping[str, str]) -> str:
    return max(
        STATE_ORDER,
        key=lambda state: (
            _probability(
                probabilities[state],
                label=f"state_probabilities.{state}",
            ),
            -STATE_ORDER.index(state),
        ),
    )


def _artifact_ref(
    document: Mapping[str, Any],
    raw: bytes,
    *,
    relative_path: str,
) -> dict[str, str]:
    version = document.get("version")
    try:
        identity_field = artifact_identity_field(version)
        identity = require_opaque_id(
            document.get(identity_field),
            label=identity_field,
        )
        byte_sha = require_sha256(
            hashlib.sha256(raw).hexdigest(),
            label="artifact byte SHA-256",
        )
        semantic_sha = require_sha256(
            document.get("semantic_sha256"),
            label="artifact semantic SHA-256",
        )
    except (IdentityContractError, SchemaValidationError) as exc:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "cannot construct exact artifact reference",
        ) from exc
    return {
        "artifact_id": identity,
        "artifact_version": str(version),
        "byte_sha256": byte_sha,
        "cutoff": str(document.get("cutoff") or ""),
        "relative_path": relative_path,
        "semantic_sha256": semantic_sha,
        "strategy_id": str(document.get("strategy_id") or ""),
    }


def _normalize_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} artifact reference shape mismatch",
        )
    result = {key: str(value[key]) for key in sorted(_REF_FIELDS)}
    try:
        require_opaque_id(result["artifact_id"], label=f"{label}.artifact_id")
        require_opaque_id(
            result["artifact_version"],
            label=f"{label}.artifact_version",
        )
        require_sha256(
            result["byte_sha256"],
            label=f"{label}.byte_sha256",
        )
        require_sha256(
            result["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
        require_utc_timestamp(
            result["cutoff"],
            label=f"{label}.cutoff",
        )
        _strategy_id(result["strategy_id"])
    except IdentityContractError as exc:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            f"{label} artifact reference is noncanonical",
        ) from exc
    return result


def _policy_ref(
    policy: Mapping[str, Any],
    raw: bytes,
) -> dict[str, str]:
    return {
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "relative_path": INFERENCE_POLICY_PATH,
        "semantic_sha256": str(policy.get("semantic_sha256") or ""),
        "version": INFERENCE_POLICY_VERSION,
    }


def _load_policy(
    *,
    inference_policy_path: str,
    inference_policy_sha256: str,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    if inference_policy_path != INFERENCE_POLICY_PATH:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "inference policy path is not the packaged fixed path",
        )
    try:
        expected_sha = require_sha256(
            inference_policy_sha256,
            label="inference_policy_sha256",
        )
        raw = read_packaged_asset(INFERENCE_POLICY_PATH)
        policy = load_packaged_json(INFERENCE_POLICY_PATH)
        validate_semantic_sha(policy)
    except IdentityContractError as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "inference policy SHA is noncanonical",
        ) from exc
    except PackageResourceError as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "packaged inference policy is unavailable or drifted",
        ) from exc
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != expected_sha:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "inference policy byte SHA mismatch",
        )
    if (
        policy.get("version") != INFERENCE_POLICY_VERSION
        or policy.get("protocol_version") != PROTOCOL_VERSION
        or policy.get("authority") != _NO_AUTHORITY
        or policy.get("publication_mode") != TEMPORAL_MODE
        or policy.get("inference_kind") != INFERENCE_KIND
        or policy.get("model_kind") != MODEL_KIND
        or policy.get("transition_source") != TRANSITION_SOURCE
        or policy.get("hard_state_derivation") != HARD_STATE_DERIVATION
        or policy.get("state_order") != list(STATE_ORDER)
        or policy.get("smoothing_used") is not False
        or policy.get("no_retroactive_causal_backfill") is not True
        or policy.get("authority") != _NO_AUTHORITY
        or policy.get("scope_policy")
        != {
            "accepted_scope_kind": "FULL_MARKET",
            "minimum_market_sample": MINIMUM_MARKET_SAMPLE,
            "sampled": False,
            "source_scope": "FULL_PIT_MARKET",
            "symbol_binding": "EXACT_ACTIVE_PIT_EQUALITY",
        }
        or policy.get("decimal_policy")
        != {
            "argmax_input": "SERIALIZED_NORMALIZED_POSTERIOR",
            "largest_remainder_tie_break": "STATE_ORDER",
            "precision": DECIMAL_PLACES,
            "rounding": "ROUND_HALF_EVEN",
            "sum": "1.000000000000",
        }
        or policy.get("clock_policy")
        != {
            "new_publication_tolerance_seconds": (NEW_PUBLICATION_TOLERANCE_SECONDS),
            "published_after_observed_shanghai_close": True,
            "replay_uses_wall_clock": False,
        }
        or policy.get("closure_limits")
        != {
            "max_depth": MAX_CLOSURE_DEPTH,
            "max_json_bytes_per_node": MAX_JSON_BYTES_PER_NODE,
            "max_nodes": MAX_CLOSURE_NODES,
            "max_parquet_row_groups": 256,
            "max_parquet_rows": 10000,
            "max_raw_bytes_per_file": MAX_RAW_BYTES_PER_FILE,
            "max_raw_total_bytes": MAX_RAW_TOTAL_BYTES,
            "max_references": MAX_REFERENCES,
            "max_total_json_bytes": MAX_TOTAL_JSON_BYTES,
            "validation_time_seconds": MAX_VALIDATION_SECONDS,
        }
        or policy.get("bootstrap")
        != {
            "first_eligible_decision_session": (BOOTSTRAP_DECISION_SESSION),
            "first_observed_through_session": (BOOTSTRAP_OBSERVED_SESSION),
            "prior": {
                "趋势上涨": "0.250000000000",
                "震荡低波": "0.250000000000",
                "震荡高波": "0.250000000000",
                "趋势下跌": "0.200000000000",
                "未知": "0.050000000000",
            },
        }
        or policy.get("no_governance_action") is not True
    ):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "inference policy semantics mismatch",
        )
    current_implementation_sha = implementation_sha256()
    if policy.get("model_implementation_sha256") != current_implementation_sha:
        raise RegimeEvidenceV2Error(
            IMPLEMENTATION_BLOCKER,
            "policy is not bound to the exact runtime source",
        )
    transition = policy.get("transition_matrix")
    _timestamp(policy.get("not_before"), label="policy.not_before")
    _posterior(
        previous=policy["bootstrap"]["prior"],
        transition_matrix=transition,
        likelihoods={state: "0.200000000000" for state in STATE_ORDER},
    )
    reference = _policy_ref(policy, raw)
    if set(reference) != _POLICY_REF_FIELDS:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "inference policy reference shape mismatch",
        )
    return dict(policy), raw, reference


class _RegimeEvidenceStore(SourceStore):
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        output_path: str,
    ) -> None:
        super().__init__(workspace_root)
        self._output_path = PurePosixPath(output_path)

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = PurePosixPath(value)
        if path not in {self._output_path, _PRODUCER_LOCK}:
            raise SourceStorageSecurityError(
                "regime evidence producer path is not an exact permitted slot"
            )
        return super()._canonical_path(path)

    def write_exact_once(
        self,
        relative_path: str | PurePosixPath,
        raw: bytes,
    ) -> WriteResult:
        path = self._canonical_path(relative_path)
        if path == _PRODUCER_LOCK and raw != b"":
            raise SourceStorageSecurityError("regime evidence producer lock bytes are invalid")
        if path == self._output_path and path.name != _OUTPUT_NAME:
            raise SourceStorageSecurityError("regime evidence output filename mismatch")
        return super().write_exact_once(path, raw)

    @contextmanager
    def producer_locked(self) -> Iterator[None]:
        with super().locked(_PRODUCER_LOCK):
            yield


class _ClosureLoader:
    """Bounded exact-reference loader with cycle and alias detection."""

    def __init__(self, store: SourceStore) -> None:
        self.store = store
        self.started_at = monotonic()
        self.node_count = 0
        self.reference_count = 0
        self.total_json_bytes = 0
        self.total_raw_bytes = 0
        self.active: list[tuple[str, str, str]] = []
        self.identity_bindings: dict[
            tuple[str, str, str, str],
            tuple[str, str, str],
        ] = {}
        self.reference_bindings: set[tuple[str, str, str, str]] = set()
        self.raw_by_key: dict[tuple[str, str], bytes] = {}
        self.document_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def _budget(self) -> None:
        if monotonic() - self.started_at > MAX_VALIDATION_SECONDS:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure validation time budget exceeded",
            )
        if (
            self.node_count > MAX_CLOSURE_NODES
            or self.reference_count > MAX_REFERENCES
            or self.total_json_bytes > MAX_TOTAL_JSON_BYTES
            or self.total_raw_bytes > MAX_RAW_TOTAL_BYTES
        ):
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure validation resource budget exceeded",
            )

    def _bind_identity(self, reference: Mapping[str, str]) -> None:
        identity = (
            reference["artifact_version"],
            reference["artifact_id"],
            reference["strategy_id"],
            reference["cutoff"],
        )
        location = (
            reference["relative_path"],
            reference["byte_sha256"],
            reference["semantic_sha256"],
        )
        observed = self.identity_bindings.setdefault(identity, location)
        if observed != location:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "one artifact identity resolves to conflicting exact bytes",
            )

    def _read(self, reference: Mapping[str, str]) -> bytes:
        key = (
            reference["relative_path"],
            reference["byte_sha256"],
        )
        cached = self.raw_by_key.get(key)
        if cached is not None:
            return cached
        try:
            raw = self.store.read(
                reference["relative_path"],
                reference["byte_sha256"],
            )
        except SourceNotFoundError as exc:
            raise RegimeEvidenceV2InputGap(f"missing input {reference['relative_path']}") from exc
        except (
            SourceCASMismatch,
            SourceStorageSecurityError,
        ) as exc:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                f"exact input readback failed for " f"{reference['relative_path']}",
            ) from exc
        if len(raw) > MAX_RAW_BYTES_PER_FILE:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure node exceeds the per-file raw byte limit",
            )
        self.node_count += 1
        self.total_raw_bytes += len(raw)
        self.raw_by_key[key] = raw
        self._budget()
        return raw

    def _scan_hidden_refs(self, value: Any, *, path: str) -> None:
        if type(value) is dict:
            reference_markers = {
                "artifact_id",
                "artifact_version",
                "byte_sha256",
                "relative_path",
            }
            if set(value) & reference_markers:
                if set(value) != _REF_FIELDS:
                    raise RegimeEvidenceV2Error(
                        INPUT_TAMPER_BLOCKER,
                        f"partial or hidden artifact reference at {path}",
                    )
                self(value)
                return
            for key, child in value.items():
                self._scan_hidden_refs(child, path=f"{path}.{key}")
        elif type(value) is list:
            for index, child in enumerate(value):
                self._scan_hidden_refs(
                    child,
                    path=f"{path}[{index}]",
                )

    def _validate_binding(
        self,
        document: Mapping[str, Any],
        reference: Mapping[str, str],
    ) -> None:
        if (
            document.get("version") != reference["artifact_version"]
            or document.get("strategy_id") != reference["strategy_id"]
            or document.get("cutoff") != reference["cutoff"]
            or document.get("semantic_sha256") != reference["semantic_sha256"]
        ):
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure node document binding mismatch",
            )
        try:
            identity_field = artifact_identity_field(reference["artifact_version"])
        except SchemaValidationError:
            candidates = [
                key
                for key, value in document.items()
                if key.endswith("_id") and value == reference["artifact_id"]
            ]
            identity_field = candidates[0] if len(candidates) == 1 else ""
        if not identity_field or document.get(identity_field) != reference["artifact_id"]:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure node identity binding mismatch",
            )

    def __call__(self, reference: Mapping[str, str]) -> bytes:
        normalized = _normalize_ref(
            dict(reference),
            label="artifact_loader",
        )
        reference_key = (
            normalized["artifact_version"],
            normalized["artifact_id"],
            normalized["relative_path"],
            normalized["byte_sha256"],
        )
        if reference_key not in self.reference_bindings:
            self.reference_bindings.add(reference_key)
            self.reference_count += 1
        self._budget()
        self._bind_identity(normalized)
        active_key = (
            normalized["artifact_version"],
            normalized["artifact_id"],
            normalized["byte_sha256"],
        )
        if active_key in self.active:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure reference cycle detected",
            )
        if len(self.active) >= MAX_CLOSURE_DEPTH:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "closure reference depth exceeded",
            )
        raw = self._read(normalized)
        cache_key = (
            normalized["relative_path"],
            normalized["byte_sha256"],
        )
        if cache_key in self.document_by_key:
            return raw
        if len(raw) > MAX_JSON_BYTES_PER_NODE:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "JSON closure node exceeds the per-node limit",
            )
        self.total_json_bytes += len(raw)
        self._budget()
        self.active.append(active_key)
        try:
            try:
                parsed = load_canonical_resource(
                    raw,
                    label=normalized["artifact_version"],
                    max_bytes=MAX_JSON_BYTES_PER_NODE,
                )
                if type(parsed) is not dict:
                    raise CanonicalContractError("closure node root is not an object")
                document = dict(parsed)
                validate_semantic_sha(document)
            except CanonicalContractError as exc:
                raise RegimeEvidenceV2Error(
                    INPUT_TAMPER_BLOCKER,
                    "closure node is not canonical sealed JSON",
                ) from exc
            self._validate_binding(document, normalized)
            try:
                schema_path_for_version(normalized["artifact_version"])
            except SchemaValidationError:
                self._scan_hidden_refs(document, path="$")
            else:
                try:
                    load_canonical_artifact(
                        raw,
                        expected_version=normalized["artifact_version"],
                        label=normalized["artifact_version"],
                        artifact_loader=self,
                    )
                except ArtifactContractError as exc:
                    raise _contract_validation_error(
                        exc,
                        context="registered closure node validation failed",
                    ) from exc
                except (
                    SchemaValidationError,
                    CanonicalContractError,
                ) as exc:
                    raise RegimeEvidenceV2Error(
                        INPUT_TAMPER_BLOCKER,
                        "registered closure node validation failed",
                    ) from exc
            self.document_by_key[cache_key] = document
            return raw
        finally:
            self.active.pop()

    def document(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        normalized = _normalize_ref(
            reference,
            label="terminal_ref",
        )
        self(normalized)
        return dict(
            self.document_by_key[
                (
                    normalized["relative_path"],
                    normalized["byte_sha256"],
                )
            ]
        )


def _load_exact_snapshot(
    *,
    store: SourceStore,
    loader: _ClosureLoader,
    relative_path: str,
    expected_sha256: str,
    expected_version: str,
    expected_strategy_id: str,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    try:
        expected_sha = require_sha256(
            expected_sha256,
            label=f"{expected_version} SHA-256",
        )
    except IdentityContractError as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            f"{expected_version} SHA is noncanonical",
        ) from exc
    try:
        raw = store.read(relative_path, expected_sha)
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV2InputGap(f"missing input {relative_path}") from exc
    except (
        SourceCASMismatch,
        SourceStorageSecurityError,
    ) as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            f"exact input readback failed for {relative_path}",
        ) from exc
    try:
        loaded = load_canonical_artifact(
            raw,
            expected_version=expected_version,
            label=expected_version,
            artifact_loader=loader,
        )
        document = dict(loaded.payload)
        identity_field = artifact_identity_field(expected_version)
    except ArtifactContractError as exc:
        raise _contract_validation_error(
            exc,
            context=f"{expected_version} contract validation failed",
        ) from exc
    except (
        SchemaValidationError,
        CanonicalContractError,
    ) as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            f"{expected_version} canonical validation failed",
        ) from exc
    if (
        document.get("strategy_id") != expected_strategy_id
        or document.get("protocol_version") != PROTOCOL_VERSION
        or document.get("authority") != _NO_AUTHORITY
        or document.get(identity_field) != _snapshot_id(document, identity_field=identity_field)
    ):
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            f"{expected_version} identity binding mismatch",
        )
    reference = _artifact_ref(
        document,
        raw,
        relative_path=relative_path,
    )
    if loader(reference) != raw:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            f"{expected_version} closure readback mismatch",
        )
    return document, raw, reference


def _read_referenced_bytes(
    *,
    loader: _ClosureLoader,
    references: Sequence[Mapping[str, Any]],
) -> None:
    if len(references) > 128:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "source reference count exceeds policy",
        )
    seen: set[tuple[str, str]] = set()
    for index, reference in enumerate(references):
        normalized = _normalize_ref(
            reference,
            label=f"source_refs[{index}]",
        )
        key = (
            normalized["relative_path"],
            normalized["byte_sha256"],
        )
        if key in seen:
            raise RegimeEvidenceV2Error(
                SEMANTIC_BLOCKER,
                "duplicate source reference",
            )
        seen.add(key)
        raw = loader(normalized)
        try:
            schema_path_for_version(normalized["artifact_version"])
        except SchemaValidationError:
            continue
        try:
            loaded = load_canonical_artifact(
                raw,
                expected_version=normalized["artifact_version"],
                label=normalized["artifact_version"],
                artifact_loader=loader,
            )
            document = dict(loaded.payload)
            identity_field = artifact_identity_field(normalized["artifact_version"])
        except ArtifactContractError as exc:
            raise _contract_validation_error(
                exc,
                context="registered source reference validation failed",
            ) from exc
        except (
            SchemaValidationError,
            CanonicalContractError,
        ) as exc:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "registered source reference validation failed",
            ) from exc
        if (
            document.get(identity_field) != normalized["artifact_id"]
            or document.get("strategy_id") != normalized["strategy_id"]
            or document.get("cutoff") != normalized["cutoff"]
            or document.get("semantic_sha256") != normalized["semantic_sha256"]
        ):
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "registered source reference binding mismatch",
            )


def _terminal_document(
    *,
    loader: _ClosureLoader,
    reference: Mapping[str, Any],
    expected_version: str,
    identity_field: str,
    terminal_role: str,
    role_fields: frozenset[str],
) -> dict[str, Any]:
    normalized = _normalize_ref(reference, label=f"{terminal_role}_ref")
    if normalized["artifact_version"] != expected_version:
        raise RegimeEvidenceV2Error(
            SCOPE_BLOCKER,
            f"{terminal_role} terminal version mismatch",
        )
    document = loader.document(normalized)
    expected_fields = {
        "authority",
        "available_at",
        "created_at",
        "cutoff",
        identity_field,
        "protocol_version",
        "semantic_sha256",
        "strategy_id",
        "terminal_role",
        "version",
        *role_fields,
    }
    if set(document) != expected_fields:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            f"{terminal_role} terminal payload shape mismatch",
        )
    if (
        document.get(identity_field) != normalized["artifact_id"]
        or document.get("terminal_role") != terminal_role
        or document.get("protocol_version") != PROTOCOL_VERSION
        or document.get("authority") != _NO_AUTHORITY
        or document.get("created_at") != document.get("available_at")
    ):
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            f"{terminal_role} terminal semantic binding mismatch",
        )
    _, available = _timestamp(
        document["available_at"],
        label=f"{terminal_role}.available_at",
    )
    _, terminal_cutoff = _timestamp(
        document["cutoff"],
        label=f"{terminal_role}.cutoff",
    )
    if available > terminal_cutoff:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            f"{terminal_role} terminal is available after its cutoff",
        )
    return document


def _validate_terminal_scope(
    *,
    feature: Mapping[str, Any],
    loader: _ClosureLoader,
    observed_session: str,
    decision_cutoff: str,
) -> None:
    calendar_ref = feature["calendar_ref"]
    pit_ref = feature["pit_membership_ref"]
    market_refs = feature["market_source_refs"]
    locator_ref = feature["source_locator_ref"]
    if type(market_refs) is not list or not market_refs:
        raise RegimeEvidenceV2Error(
            SCOPE_BLOCKER,
            "full-market feature requires market terminals",
        )
    calendar = _terminal_document(
        loader=loader,
        reference=calendar_ref,
        expected_version=CALENDAR_TERMINAL_VERSION,
        identity_field="calendar_terminal_id",
        terminal_role="SHANGHAI_OPEN_CALENDAR",
        role_fields=frozenset({"open_sessions"}),
    )
    pit = _terminal_document(
        loader=loader,
        reference=pit_ref,
        expected_version=PIT_TERMINAL_VERSION,
        identity_field="pit_membership_terminal_id",
        terminal_role="ACTIVE_PIT_MEMBERSHIP",
        role_fields=frozenset({"active_symbols", "observed_through_session"}),
    )
    markets = [
        _terminal_document(
            loader=loader,
            reference=reference,
            expected_version=MARKET_TERMINAL_VERSION,
            identity_field="market_terminal_id",
            terminal_role="SEALED_MARKET_SYMBOL_INVENTORY",
            role_fields=frozenset({"observed_through_session", "symbols"}),
        )
        for reference in market_refs
    ]
    locator = _terminal_document(
        loader=loader,
        reference=locator_ref,
        expected_version=LOCATOR_TERMINAL_VERSION,
        identity_field="source_locator_terminal_id",
        terminal_role="REGIME_SOURCE_LOCATOR",
        role_fields=frozenset(
            {
                "calendar_ref",
                "market_source_refs",
                "pit_membership_ref",
            }
        ),
    )
    if (
        calendar.get("open_sessions") != feature["open_sessions"]
        or pit.get("observed_through_session") != observed_session
        or pit.get("active_symbols") != feature["full_market_symbols"]
        or locator.get("calendar_ref") != calendar_ref
        or locator.get("pit_membership_ref") != pit_ref
        or locator.get("market_source_refs") != market_refs
    ):
        raise RegimeEvidenceV2Error(
            SCOPE_BLOCKER,
            "terminal calendar/PIT/locator closure mismatch",
        )
    market_symbols: list[str] = []
    for market in markets:
        if (
            market.get("observed_through_session") != observed_session
            or type(market.get("symbols")) is not list
            or market["symbols"] != sorted(set(market["symbols"]))
        ):
            raise RegimeEvidenceV2Error(
                SCOPE_BLOCKER,
                "market terminal session or symbol ordering mismatch",
            )
        market_symbols.extend(market["symbols"])
    if (
        market_symbols != sorted(set(market_symbols))
        or market_symbols != feature["full_market_symbols"]
    ):
        raise RegimeEvidenceV2Error(
            SCOPE_BLOCKER,
            "market terminal inventory is a subset, duplicate, or superset",
        )
    _, factor_cutoff = _timestamp(
        decision_cutoff,
        label="factor_cutoff",
    )
    for role, terminal in [
        ("calendar", calendar),
        ("pit", pit),
        *[("market", terminal) for terminal in markets],
        ("locator", locator),
    ]:
        _, available = _timestamp(
            terminal["available_at"],
            label=f"{role}.available_at",
        )
        _, terminal_cutoff = _timestamp(
            terminal["cutoff"],
            label=f"{role}.cutoff",
        )
        if available > factor_cutoff or terminal_cutoff > factor_cutoff:
            raise RegimeEvidenceV2Error(
                TEMPORAL_BLOCKER,
                f"{role} terminal exceeds Factor cutoff",
            )


def _validate_snapshot_closure(
    *,
    feature: Mapping[str, Any],
    transition: Mapping[str, Any],
    model: Mapping[str, Any],
    feature_ref: Mapping[str, str],
    transition_ref: Mapping[str, str],
    policy: Mapping[str, Any],
    policy_ref: Mapping[str, str],
    prior_ref: Mapping[str, str] | None,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    loader: _ClosureLoader,
) -> tuple[str, list[dict[str, str]]]:
    observed = _session(
        feature.get("observed_through_session"),
        label="feature.observed_through_session",
    )
    effective = _session(
        feature.get("effective_session"),
        label="feature.effective_session",
    )
    if effective != decision_session:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "feature effective session is not the decision session",
        )
    if observed >= effective:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "observed session must precede effective session",
        )
    open_sessions = feature.get("open_sessions")
    if (
        type(open_sessions) is not list
        or any(type(item) is not str for item in open_sessions)
        or open_sessions != sorted(set(open_sessions))
        or observed not in open_sessions
        or decision_session not in open_sessions
        or open_sessions.index(decision_session) != open_sessions.index(observed) + 1
    ):
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "decision session is not the next sealed Shanghai open session",
        )
    for label, snapshot in (
        ("feature", feature),
        ("transition", transition),
        ("model", model),
    ):
        if (
            snapshot.get("strategy_id") != strategy_id
            or snapshot.get("observed_through_session") != observed
            or snapshot.get("effective_session") != decision_session
        ):
            raise RegimeEvidenceV2Error(
                TEMPORAL_BLOCKER,
                f"{label} snapshot session binding mismatch",
            )
        _, snapshot_available = _timestamp(
            snapshot.get("available_at"),
            label=f"{label}.available_at",
        )
        _, decision_cutoff = _timestamp(cutoff, label="cutoff")
        if snapshot_available > decision_cutoff:
            raise RegimeEvidenceV2Error(
                TEMPORAL_BLOCKER,
                f"{label} snapshot is after decision cutoff",
            )
        _, snapshot_cutoff = _timestamp(
            snapshot.get("cutoff"),
            label=f"{label}.cutoff",
        )
        if snapshot_cutoff > decision_cutoff:
            raise RegimeEvidenceV2Error(
                TEMPORAL_BLOCKER,
                f"{label} snapshot cutoff exceeds Factor cutoff",
            )
    if (
        feature.get("source_scope") != "FULL_PIT_MARKET"
        or feature.get("scope_kind") != "FULL_MARKET"
        or feature.get("sampled") is not False
        or type(feature.get("full_market_symbols")) is not list
        or feature["full_market_symbols"] != sorted(set(feature["full_market_symbols"]))
        or any(type(symbol) is not str or not symbol for symbol in feature["full_market_symbols"])
    ):
        raise RegimeEvidenceV2Error(
            SCOPE_BLOCKER,
            "feature snapshot is not exact full-market scope",
        )
    market_count = feature.get("market_sample_count")
    pit_count = feature.get("pit_active_symbol_count")
    symbols = feature["full_market_symbols"]
    if (
        type(market_count) is not int
        or type(pit_count) is not int
        or market_count != pit_count
        or market_count != len(symbols)
        or market_count < MINIMUM_MARKET_SAMPLE
        or feature.get("minimum_market_sample") != MINIMUM_MARKET_SAMPLE
        or feature.get("coverage_ratio") != "1.000000000000"
    ):
        raise RegimeEvidenceV2Error(
            SCOPE_BLOCKER,
            "feature snapshot count or coverage binding mismatch",
        )
    if (
        feature.get("state_order") != list(STATE_ORDER)
        or transition.get("state_order") != list(STATE_ORDER)
        or model.get("state_order") != list(STATE_ORDER)
    ):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "native state order mismatch",
        )
    risk_on, volatility, pressure, likelihoods = _derive_scores_and_likelihoods(feature)
    if (
        feature.get("risk_on_score") != risk_on
        or feature.get("volatility_score") != volatility
        or feature.get("pressure_score") != pressure
        or feature.get("state_likelihoods") != likelihoods
    ):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "feature snapshot fixed-rule replay mismatch",
        )
    expected_matrix = policy.get("transition_matrix")
    if (
        transition.get("transition_source") != TRANSITION_SOURCE
        or transition.get("transition_matrix") != expected_matrix
        or transition.get("inference_policy_ref") != policy_ref
        or transition.get("source_evidence_refs") != []
    ):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "transition snapshot is not the pinned native default",
        )
    if (
        model.get("model_kind") != MODEL_KIND
        or model.get("model_id") != MODEL_ID
        or model.get("model_version") != MODEL_VERSION
        or model.get("model_implementation_sha256") != implementation_sha256()
        or model.get("model_implementation_sha256") != policy.get("model_implementation_sha256")
        or model.get("formula_version") != FORMULA_VERSION
        or model.get("model_training_end_session") is not None
        or model.get("training_source_refs") != []
        or model.get("inference_policy_ref") != policy_ref
        or model.get("transition_matrix_ref") != transition_ref
        or model.get("predecessor_evidence_ref") != prior_ref
    ):
        raise RegimeEvidenceV2Error(
            IMPLEMENTATION_BLOCKER,
            "model snapshot is not the exact fixed no-training model",
        )
    source_refs_raw = [
        feature.get("calendar_ref"),
        feature.get("pit_membership_ref"),
        *(feature.get("market_source_refs") or []),
        feature.get("source_locator_ref"),
    ]
    if any(type(reference) is not dict for reference in source_refs_raw):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "feature source closure is incomplete",
        )
    source_refs = [
        _normalize_ref(reference, label="feature.source_ref") for reference in source_refs_raw
    ]
    source_refs.sort(
        key=lambda item: (
            item["relative_path"],
            item["byte_sha256"],
            item["artifact_id"],
        )
    )
    if len(
        {
            (
                item["relative_path"],
                item["byte_sha256"],
                item["artifact_id"],
            )
            for item in source_refs
        }
    ) != len(source_refs):
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "feature source closure contains duplicate refs",
        )
    _read_referenced_bytes(loader=loader, references=source_refs)
    _validate_terminal_scope(
        feature=feature,
        loader=loader,
        observed_session=observed,
        decision_cutoff=cutoff,
    )
    if feature_ref["strategy_id"] != strategy_id:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "feature reference strategy mismatch",
        )
    return observed, source_refs


def _load_prior(
    *,
    store: SourceStore,
    loader: _ClosureLoader,
    prior_evidence_path: str | None,
    prior_evidence_sha256: str | None,
    strategy_id: str,
    observed_session: str | None,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if prior_evidence_path is None and prior_evidence_sha256 is None:
        return None, None
    if not prior_evidence_path or not prior_evidence_sha256:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "prior evidence path and SHA must be supplied together",
        )
    try:
        raw = store.read(prior_evidence_path, prior_evidence_sha256)
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV2InputGap(f"missing prior evidence {prior_evidence_path}") from exc
    except (
        SourceCASMismatch,
        SourceStorageSecurityError,
    ) as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "prior evidence exact readback failed",
        ) from exc
    try:
        loaded = load_canonical_artifact(
            raw,
            expected_version=REGIME_EVIDENCE_VERSION,
            label="prior regime evidence v2",
            artifact_loader=loader,
        )
        prior = dict(loaded.payload)
    except ArtifactContractError as exc:
        raise _contract_validation_error(
            exc,
            context="prior evidence contract validation failed",
        ) from exc
    except (
        SchemaValidationError,
        CanonicalContractError,
    ) as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "prior evidence canonical validation failed",
        ) from exc
    expected_path = regime_evidence_v2_path(
        strategy_id=strategy_id,
        effective_session=str(prior.get("effective_session") or ""),
    )
    if (
        prior_evidence_path != expected_path
        or prior.get("strategy_id") != strategy_id
        or (observed_session is not None and prior.get("effective_session") != observed_session)
        or prior.get("status") != "AVAILABLE"
        or prior.get("blocker_codes") != []
        or prior.get("evidence_id") != regime_evidence_v2_id(prior)
        or prior.get("authority") != _NO_AUTHORITY
        or prior.get("same_session_execution_eligible") is not False
        or prior.get("no_retroactive_causal_backfill") is not True
        or prior.get("shadow_only") is not True
        or prior.get("formal_activation_eligible") is not False
        or prior.get("performance_evidence_eligible") is not False
        or prior.get("promotion_eligible") is not False
        or prior.get("model_implementation_sha256") != implementation_sha256()
    ):
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "prior evidence is not the contiguous sealed v2 predecessor",
        )
    _serialized_probabilities(
        prior.get("state_probabilities"),
        label="prior.state_probabilities",
    )
    reference = _artifact_ref(
        prior,
        raw,
        relative_path=prior_evidence_path,
    )
    if loader(reference) != raw:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "prior evidence closure readback mismatch",
        )
    return prior, reference


def _validate_inference_phase(
    *,
    observed_session: str,
    decision_session: str,
    prior: Mapping[str, Any] | None,
) -> str:
    bootstrap = (
        observed_session == BOOTSTRAP_OBSERVED_SESSION
        and decision_session == BOOTSTRAP_DECISION_SESSION
    )
    if bootstrap:
        if prior is not None:
            raise RegimeEvidenceV2Error(
                TEMPORAL_BLOCKER,
                "bootstrap publication cannot have predecessor evidence",
            )
        return "BOOTSTRAP"
    if prior is None:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "NORMAL publication requires the contiguous prior v2",
        )
    return "NORMAL"


def _new_publication_clock(
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
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "publisher clock must be timezone-aware",
        )
    normalized_now = now.astimezone(_UTC).replace(microsecond=0)
    observed_close = datetime.combine(
        date.fromisoformat(observed_session),
        time(hour=15),
        tzinfo=_SHANGHAI,
    ).astimezone(_UTC)
    cutoff_session = decision_cutoff.astimezone(_SHANGHAI).date().isoformat()
    publication_session = created.astimezone(_SHANGHAI).date().isoformat()
    if (
        abs((normalized_now - created).total_seconds()) > NEW_PUBLICATION_TOLERANCE_SECONDS
        or created < observed_close
        or created > decision_cutoff
        or cutoff_session != decision_session
        or publication_session > decision_session
    ):
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            (
                "new publication clock is stale, future, pre-close, post-cutoff, "
                "or outside the effective decision session"
            ),
        )


def _enforce_policy_not_before(
    policy: Mapping[str, Any],
    *,
    created_at: str,
) -> None:
    _, created = _timestamp(created_at, label="created_at")
    _, not_before = _timestamp(
        policy.get("not_before"),
        label="policy.not_before",
    )
    if created < not_before:
        raise RegimeEvidenceV2Error(
            TEMPORAL_BLOCKER,
            "publication predates the packaged inference policy",
        )


def _build_document(
    *,
    evidence_id: str,
    strategy_id: str,
    cutoff: str,
    created_at: str,
    decision_session: str,
    observed_session: str,
    lineage_phase: str,
    policy: Mapping[str, Any],
    policy_ref: Mapping[str, str],
    feature: Mapping[str, Any],
    feature_ref: Mapping[str, str],
    transition: Mapping[str, Any],
    transition_ref: Mapping[str, str],
    model: Mapping[str, Any],
    model_ref: Mapping[str, str],
    source_refs: Sequence[Mapping[str, str]],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    _enforce_policy_not_before(policy, created_at=created_at)
    probabilities = _posterior(
        previous=previous,
        transition_matrix=transition["transition_matrix"],
        likelihoods=feature["state_likelihoods"],
    )
    authority = regime_evidence_v2_authority_attestation()
    body: dict[str, Any] = {
        "authority": authority["authority"],
        "available_at": created_at,
        "blocker_codes": [],
        "computed_at": created_at,
        "coverage_ratio": feature["coverage_ratio"],
        "created_at": created_at,
        "cutoff": cutoff,
        "decision_session": decision_session,
        "effective_session": decision_session,
        "evidence_id": evidence_id,
        "feature_cutoff": feature["cutoff"],
        "feature_snapshot_ref": dict(feature_ref),
        "formal_activation_eligible": (authority["formal_activation_eligible"]),
        "hard_state": _hard_state(probabilities),
        "hard_state_derivation": HARD_STATE_DERIVATION,
        "inference_kind": INFERENCE_KIND,
        "inference_policy_ref": dict(policy_ref),
        "lineage_phase": lineage_phase,
        "market_sample_count": feature["market_sample_count"],
        "minimum_market_sample": MINIMUM_MARKET_SAMPLE,
        "model_id": model["model_id"],
        "model_implementation_sha256": (model["model_implementation_sha256"]),
        "model_snapshot_ref": dict(model_ref),
        "model_training_end_session": None,
        "model_version": model["model_version"],
        "no_retroactive_causal_backfill": (authority["no_retroactive_causal_backfill"]),
        "observed_through_session": observed_session,
        "performance_evidence_eligible": (authority["performance_evidence_eligible"]),
        "promotion_eligible": authority["promotion_eligible"],
        "protocol_version": PROTOCOL_VERSION,
        "publication_phase": TEMPORAL_MODE,
        "published_at": created_at,
        "same_session_execution_eligible": (authority["same_session_execution_eligible"]),
        "scope_kind": "FULL_MARKET",
        "scope_ref": dict(feature["pit_membership_ref"]),
        "shadow_only": authority["shadow_only"],
        "smoothing_used": False,
        "source_refs": [dict(reference) for reference in source_refs],
        "state_order": list(STATE_ORDER),
        "state_probabilities": probabilities,
        "status": "AVAILABLE",
        "strategy_id": strategy_id,
        "transition_matrix_ref": dict(transition_ref),
        "version": REGIME_EVIDENCE_VERSION,
    }
    expected_id = regime_evidence_v2_id(body)
    if evidence_id != expected_id:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            f"evidence_id mismatch; expected {expected_id}",
        )
    try:
        return seal_semantic(body)
    except CanonicalContractError as exc:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "evidence sealing failed",
        ) from exc


def _validate_evidence_document(
    *,
    document: Mapping[str, Any],
    evidence_path: str,
    evidence_sha256: str | None,
    raw: bytes | None,
    store: SourceStore,
    enforce_build_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if raw is None:
        try:
            raw = store.read(evidence_path, evidence_sha256)
        except SourceNotFoundError as exc:
            raise RegimeEvidenceV2InputGap(f"missing regime evidence {evidence_path}") from exc
        except (
            SourceCASMismatch,
            SourceStorageSecurityError,
        ) as exc:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "regime evidence exact readback failed",
            ) from exc
    elif evidence_sha256 is not None:
        try:
            expected = require_sha256(
                evidence_sha256,
                label="evidence_sha256",
            )
        except IdentityContractError as exc:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "evidence SHA is noncanonical",
            ) from exc
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "evidence SHA mismatch",
            )
    loader = _ClosureLoader(store)
    try:
        loaded = load_canonical_artifact(
            raw,
            expected_version=REGIME_EVIDENCE_VERSION,
            label=REGIME_EVIDENCE_VERSION,
            artifact_loader=loader,
        )
        validated = dict(loaded.payload)
    except ArtifactContractError as exc:
        raise _contract_validation_error(
            exc,
            context="regime evidence contract validation failed",
        ) from exc
    except (
        SchemaValidationError,
        CanonicalContractError,
    ) as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "regime evidence canonical validation failed",
        ) from exc
    if validated != dict(document):
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "regime evidence parsed bytes mismatch",
        )
    strategy_id = _strategy_id(validated.get("strategy_id"))
    effective_session = _session(
        validated.get("effective_session"),
        label="effective_session",
    )
    expected_path = regime_evidence_v2_path(
        strategy_id=strategy_id,
        effective_session=effective_session,
    )
    if evidence_path != expected_path:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "regime evidence path is not its exact strategy/session slot",
        )
    policy_ref = validated.get("inference_policy_ref")
    if type(policy_ref) is not dict or set(policy_ref) != _POLICY_REF_FIELDS:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "evidence policy reference shape mismatch",
        )
    policy, _, expected_policy_ref = _load_policy(
        inference_policy_path=str(policy_ref.get("relative_path") or ""),
        inference_policy_sha256=str(policy_ref.get("byte_sha256") or ""),
    )
    if policy_ref != expected_policy_ref:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "evidence policy reference binding mismatch",
        )
    feature_ref = _normalize_ref(
        validated.get("feature_snapshot_ref"),
        label="feature_snapshot_ref",
    )
    transition_ref = _normalize_ref(
        validated.get("transition_matrix_ref"),
        label="transition_matrix_ref",
    )
    model_ref = _normalize_ref(
        validated.get("model_snapshot_ref"),
        label="model_snapshot_ref",
    )
    feature, _, observed_feature_ref = _load_exact_snapshot(
        store=store,
        loader=loader,
        relative_path=feature_ref["relative_path"],
        expected_sha256=feature_ref["byte_sha256"],
        expected_version=FEATURE_SNAPSHOT_VERSION,
        expected_strategy_id=strategy_id,
    )
    transition, _, observed_transition_ref = _load_exact_snapshot(
        store=store,
        loader=loader,
        relative_path=transition_ref["relative_path"],
        expected_sha256=transition_ref["byte_sha256"],
        expected_version=TRANSITION_SNAPSHOT_VERSION,
        expected_strategy_id=strategy_id,
    )
    model, _, observed_model_ref = _load_exact_snapshot(
        store=store,
        loader=loader,
        relative_path=model_ref["relative_path"],
        expected_sha256=model_ref["byte_sha256"],
        expected_version=MODEL_SNAPSHOT_VERSION,
        expected_strategy_id=strategy_id,
    )
    if (
        feature_ref != observed_feature_ref
        or transition_ref != observed_transition_ref
        or model_ref != observed_model_ref
    ):
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "evidence direct snapshot reference mismatch",
        )
    predecessor_ref = model.get("predecessor_evidence_ref")
    prior: dict[str, Any] | None
    prior_ref: dict[str, str] | None
    if predecessor_ref is None:
        prior, prior_ref = None, None
    else:
        normalized_prior = _normalize_ref(
            predecessor_ref,
            label="predecessor_evidence_ref",
        )
        prior, prior_ref = _load_prior(
            store=store,
            loader=loader,
            prior_evidence_path=normalized_prior["relative_path"],
            prior_evidence_sha256=normalized_prior["byte_sha256"],
            strategy_id=strategy_id,
            observed_session=str(validated.get("observed_through_session") or ""),
        )
        if prior_ref != normalized_prior:
            raise RegimeEvidenceV2Error(
                INPUT_TAMPER_BLOCKER,
                "predecessor evidence reference mismatch",
            )
    observed, source_refs = _validate_snapshot_closure(
        feature=feature,
        transition=transition,
        model=model,
        feature_ref=feature_ref,
        transition_ref=transition_ref,
        policy=policy,
        policy_ref=expected_policy_ref,
        prior_ref=prior_ref,
        strategy_id=strategy_id,
        decision_session=str(validated.get("decision_session") or ""),
        cutoff=str(validated.get("cutoff") or ""),
        loader=loader,
    )
    lineage_phase = _validate_inference_phase(
        observed_session=observed,
        decision_session=str(validated.get("decision_session") or ""),
        prior=prior,
    )
    previous = policy["bootstrap"]["prior"] if prior is None else prior["state_probabilities"]
    rebuilt = _build_document(
        evidence_id=str(validated.get("evidence_id") or ""),
        strategy_id=strategy_id,
        cutoff=str(validated.get("cutoff") or ""),
        created_at=str(validated.get("created_at") or ""),
        decision_session=str(validated.get("decision_session") or ""),
        observed_session=observed,
        lineage_phase=lineage_phase,
        policy=policy,
        policy_ref=expected_policy_ref,
        feature=feature,
        feature_ref=feature_ref,
        transition=transition,
        transition_ref=transition_ref,
        model=model,
        model_ref=model_ref,
        source_refs=source_refs,
        previous=previous,
    )
    if rebuilt != validated:
        raise RegimeEvidenceV2Error(
            SEMANTIC_BLOCKER,
            "regime evidence deterministic replay mismatch",
        )
    if enforce_build_arguments is not None:
        expected_pairs = {
            "created_at": validated["created_at"],
            "cutoff": validated["cutoff"],
            "decision_session": validated["decision_session"],
            "evidence_id": validated["evidence_id"],
            "feature_snapshot_path": feature_ref["relative_path"],
            "feature_snapshot_sha256": feature_ref["byte_sha256"],
            "inference_policy_path": expected_policy_ref["relative_path"],
            "inference_policy_sha256": expected_policy_ref["byte_sha256"],
            "model_snapshot_path": model_ref["relative_path"],
            "model_snapshot_sha256": model_ref["byte_sha256"],
            "prior_evidence_path": (None if prior_ref is None else prior_ref["relative_path"]),
            "prior_evidence_sha256": (None if prior_ref is None else prior_ref["byte_sha256"]),
            "strategy_id": validated["strategy_id"],
            "transition_matrix_path": transition_ref["relative_path"],
            "transition_matrix_sha256": transition_ref["byte_sha256"],
        }
        if dict(enforce_build_arguments) != expected_pairs:
            raise RegimeEvidenceV2Conflict("fixed slot retry arguments differ from sealed evidence")
    return validated


def build_regime_evidence_v2(
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
    _now_fn: Callable[[], datetime] | None = None,
) -> RegimeEvidenceV2BuildResult:
    """Build or exactly reuse one causal regime-evidence v2 slot."""

    normalized_strategy = _strategy_id(strategy_id)
    normalized_decision = _session(
        decision_session,
        label="decision_session",
    )
    try:
        normalized_evidence_id = require_opaque_id(
            evidence_id,
            label="evidence_id",
        )
    except IdentityContractError as exc:
        raise RegimeEvidenceV2Error(
            IDENTITY_BLOCKER,
            "evidence_id is not canonical",
        ) from exc
    _timestamp(cutoff, label="cutoff")
    _timestamp(created_at, label="created_at")
    output_path = regime_evidence_v2_path(
        strategy_id=normalized_strategy,
        effective_session=normalized_decision,
    )
    writer = _RegimeEvidenceStore(
        workspace_root,
        output_path=output_path,
    )
    reader = SourceStore(workspace_root)
    build_arguments = {
        "created_at": created_at,
        "cutoff": cutoff,
        "decision_session": normalized_decision,
        "evidence_id": normalized_evidence_id,
        "feature_snapshot_path": feature_snapshot_path,
        "feature_snapshot_sha256": feature_snapshot_sha256,
        "inference_policy_path": inference_policy_path,
        "inference_policy_sha256": inference_policy_sha256,
        "model_snapshot_path": model_snapshot_path,
        "model_snapshot_sha256": model_snapshot_sha256,
        "prior_evidence_path": prior_evidence_path,
        "prior_evidence_sha256": prior_evidence_sha256,
        "strategy_id": normalized_strategy,
        "transition_matrix_path": transition_matrix_path,
        "transition_matrix_sha256": transition_matrix_sha256,
    }
    with writer.producer_locked():
        existing = writer.read_optional(output_path)
        if existing is not None:
            try:
                document = load_canonical_resource(
                    existing.data,
                    label=REGIME_EVIDENCE_VERSION,
                )
                if type(document) is not dict:
                    raise CanonicalContractError("regime evidence root is not an object")
            except CanonicalContractError as exc:
                raise RegimeEvidenceV2Error(
                    INPUT_TAMPER_BLOCKER,
                    "existing exact slot is noncanonical",
                ) from exc
            replayed = _validate_evidence_document(
                document=document,
                evidence_path=output_path,
                evidence_sha256=existing.byte_sha256,
                raw=existing.data,
                store=reader,
                enforce_build_arguments=build_arguments,
            )
            return RegimeEvidenceV2BuildResult(
                status="AVAILABLE",
                evidence_id=replayed["evidence_id"],
                evidence_path=output_path,
                evidence_sha256=existing.byte_sha256,
                created=False,
                reused=True,
                document=replayed,
            )

        now_fn = _now_fn or (lambda: datetime.now(_UTC))
        now = now_fn()
        loader = _ClosureLoader(reader)
        policy, _, policy_ref = _load_policy(
            inference_policy_path=inference_policy_path,
            inference_policy_sha256=inference_policy_sha256,
        )
        _enforce_policy_not_before(policy, created_at=created_at)
        feature, _, feature_ref = _load_exact_snapshot(
            store=reader,
            loader=loader,
            relative_path=feature_snapshot_path,
            expected_sha256=feature_snapshot_sha256,
            expected_version=FEATURE_SNAPSHOT_VERSION,
            expected_strategy_id=normalized_strategy,
        )
        transition, _, transition_ref = _load_exact_snapshot(
            store=reader,
            loader=loader,
            relative_path=transition_matrix_path,
            expected_sha256=transition_matrix_sha256,
            expected_version=TRANSITION_SNAPSHOT_VERSION,
            expected_strategy_id=normalized_strategy,
        )
        prior, prior_ref = _load_prior(
            store=reader,
            loader=loader,
            prior_evidence_path=prior_evidence_path,
            prior_evidence_sha256=prior_evidence_sha256,
            strategy_id=normalized_strategy,
            observed_session=str(feature.get("observed_through_session") or ""),
        )
        model, _, model_ref = _load_exact_snapshot(
            store=reader,
            loader=loader,
            relative_path=model_snapshot_path,
            expected_sha256=model_snapshot_sha256,
            expected_version=MODEL_SNAPSHOT_VERSION,
            expected_strategy_id=normalized_strategy,
        )
        observed, source_refs = _validate_snapshot_closure(
            feature=feature,
            transition=transition,
            model=model,
            feature_ref=feature_ref,
            transition_ref=transition_ref,
            policy=policy,
            policy_ref=policy_ref,
            prior_ref=prior_ref,
            strategy_id=normalized_strategy,
            decision_session=normalized_decision,
            cutoff=cutoff,
            loader=loader,
        )
        lineage_phase = _validate_inference_phase(
            observed_session=observed,
            decision_session=normalized_decision,
            prior=prior,
        )
        _new_publication_clock(
            created_at=created_at,
            cutoff=cutoff,
            observed_session=observed,
            decision_session=normalized_decision,
            now=now,
        )
        previous = policy["bootstrap"]["prior"] if prior is None else prior["state_probabilities"]
        document = _build_document(
            evidence_id=normalized_evidence_id,
            strategy_id=normalized_strategy,
            cutoff=cutoff,
            created_at=created_at,
            decision_session=normalized_decision,
            observed_session=observed,
            lineage_phase=lineage_phase,
            policy=policy,
            policy_ref=policy_ref,
            feature=feature,
            feature_ref=feature_ref,
            transition=transition,
            transition_ref=transition_ref,
            model=model,
            model_ref=model_ref,
            source_refs=source_refs,
            previous=previous,
        )
        raw = canonical_resource_bytes(document)
        _validate_evidence_document(
            document=document,
            evidence_path=output_path,
            evidence_sha256=hashlib.sha256(raw).hexdigest(),
            raw=raw,
            store=reader,
        )
        try:
            write_result = writer.write_exact_once(output_path, raw)
        except SourceExactOnceConflict as exc:
            raise RegimeEvidenceV2Conflict("fixed slot appeared with different bytes") from exc
        except SourceStorageError as exc:
            raise RegimeEvidenceV2Error(
                CONFLICT_BLOCKER,
                "immutable publication failed",
            ) from exc
        replayed = replay_regime_evidence_v2(
            workspace_root=workspace_root,
            evidence_path=output_path,
            evidence_sha256=write_result.byte_sha256,
        )
        return RegimeEvidenceV2BuildResult(
            status="AVAILABLE",
            evidence_id=replayed["evidence_id"],
            evidence_path=output_path,
            evidence_sha256=write_result.byte_sha256,
            created=write_result.created,
            reused=not write_result.created,
            document=replayed,
        )


def replay_regime_evidence_v2(
    *,
    workspace_root: str | Path,
    evidence_path: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Replay an exact v2 and its pinned closure without wall-clock freshness."""

    store = SourceStore(workspace_root)
    try:
        raw = store.read(evidence_path, evidence_sha256)
    except SourceNotFoundError as exc:
        raise RegimeEvidenceV2InputGap(f"missing regime evidence {evidence_path}") from exc
    except (
        SourceCASMismatch,
        SourceStorageSecurityError,
    ) as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "regime evidence exact readback failed",
        ) from exc
    try:
        document = load_canonical_resource(
            raw,
            label=REGIME_EVIDENCE_VERSION,
        )
    except CanonicalContractError as exc:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "regime evidence is noncanonical",
        ) from exc
    if type(document) is not dict:
        raise RegimeEvidenceV2Error(
            INPUT_TAMPER_BLOCKER,
            "regime evidence root is not an object",
        )
    return _validate_evidence_document(
        document=document,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
        raw=raw,
        store=store,
    )


def read_regime_evidence_v2(
    *,
    workspace_root: str | Path,
    evidence_path: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Compatibility name for explicit exact replay."""

    return replay_regime_evidence_v2(
        workspace_root=workspace_root,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
    )


def regime_evidence_v2_status(
    *,
    workspace_root: str | Path,
    evidence_path: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Return a report-only status after complete deterministic replay."""

    document = replay_regime_evidence_v2(
        workspace_root=workspace_root,
        evidence_path=evidence_path,
        evidence_sha256=evidence_sha256,
    )
    return {
        "authority": dict(document["authority"]),
        "blocker_codes": list(document["blocker_codes"]),
        "decision_session": document["decision_session"],
        "effective_session": document["effective_session"],
        "evidence_id": document["evidence_id"],
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "hard_state": document["hard_state"],
        "observed_through_session": (document["observed_through_session"]),
        "status": document["status"],
    }


__all__ = [
    "BOOTSTRAP_DECISION_SESSION",
    "BOOTSTRAP_OBSERVED_SESSION",
    "FEATURE_SNAPSHOT_VERSION",
    "FORMULA_VERSION",
    "INFERENCE_POLICY_PATH",
    "INFERENCE_POLICY_VERSION",
    "MODEL_SNAPSHOT_VERSION",
    "REGIME_EVIDENCE_VERSION",
    "STATE_ORDER",
    "TEMPORAL_MODE",
    "TRANSITION_SNAPSHOT_VERSION",
    "TRUE_CURRENT_CANONICAL_INPUT_GAP",
    "RegimeEvidenceV2BuildResult",
    "RegimeEvidenceV2Conflict",
    "RegimeEvidenceV2Error",
    "RegimeEvidenceV2InputGap",
    "build_regime_evidence_v2",
    "implementation_sha256",
    "read_regime_evidence_v2",
    "regime_evidence_v2_authority_attestation",
    "regime_evidence_v2_id",
    "regime_evidence_v2_path",
    "regime_evidence_v2_status",
    "replay_regime_evidence_v2",
]
