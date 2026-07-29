"""Shadow-only zero-history Fusion and forward-label accumulation.

This lane is deliberately separate from calibrated Fusion v1.  It fixes the
branch weights at 50/50, publishes no promotion or performance evidence, and
keeps every prediction and matured horizon label immutable.  Missing
Fundamental observations are excluded from the Fundamental percentile
universe and receive a zero Fundamental percentile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any, Final, NoReturn
from zoneinfo import ZoneInfo

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.canonical import validate_semantic_sha
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)

from .source_storage import (
    GovernedStore,
    SourceExactOnceConflict,
    SourceStorageError,
    SourceStorageSecurityError,
    canonical_governed_path,
)

SHADOW_FUSION_POLICY_VERSION: Final = "myquant.v17.v4.shadow-fusion-policy.v1"
FUSION_TOP24_V2_VERSION: Final = "myquant.v17.v4.fusion-top24.v2"
SHADOW_FUSION_OBSERVATION_VERSION: Final = "myquant.v17.v4.shadow-fusion-observation.v1"
SHADOW_FUSION_MATURED_LABEL_VERSION: Final = "myquant.v17.v4.shadow-fusion-matured-label.v1"
FORWARD_ACCUMULATION_STATE: Final = "UNCALIBRATED_FORWARD_ACCUMULATING"
QUANT_WEIGHT: Final = Decimal("0.5")
FUNDAMENTAL_WEIGHT: Final = Decimal("0.5")
BASE_TARGET: Final = Decimal("0.03")
TOP_N: Final = 24
LABEL_HORIZONS: Final = (1, 5, 20)
SHANGHAI: Final = ZoneInfo("Asia/Shanghai")
FORWARD_FUSION_ROOT: Final = PurePosixPath("results/v17_v4_shadow/forward_fusion")
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
_INELIGIBLE: Final = {
    "canary_evidence_eligible": False,
    "formal_activation_eligible": False,
    "formal_research_publication_eligible": False,
    "performance_evidence_eligible": False,
    "promotion_eligible": False,
    "shadow_only": True,
}
_PATH_ID_RE: Final = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    re.ASCII,
)
_SYMBOL_RE: Final = re.compile(
    r"^[0-9]{6}\.(?:BJ|SH|SZ)$",
    re.ASCII,
)


class ForwardFusionError(RuntimeError):
    """Raised when the zero-history Fusion lane cannot prove its closure."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise ForwardFusionError(f"V17_V4_FORWARD_FUSION_BLOCKED:{reason}")


def _path_id(value: Any, *, label: str) -> str:
    try:
        result = require_opaque_id(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")
    if _PATH_ID_RE.fullmatch(result) is None:
        _blocked(f"{label}_path")
    return result


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _blocked(f"{label}_invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _blocked(f"{label}_invalid")
    if parsed.isoformat() != value:
        _blocked(f"{label}_noncanonical")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    try:
        return require_utc_timestamp(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _sha256(value: Any, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _decimal(value: Any, *, label: str, positive: bool = False) -> Decimal:
    if type(value) is bool or type(value) not in {
        str,
        int,
        float,
        Decimal,
    }:
        _blocked(f"{label}_decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_decimal")
    if not result.is_finite() or (positive and result <= 0):
        _blocked(f"{label}_range")
    return result


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        _blocked("decimal_nonfinite")
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result or "0"


def _symbols(values: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not TOP_N <= len(values) <= 500
    ):
        _blocked("pool_size")
    result = tuple(values)
    if len(set(result)) != len(result) or any(
        type(symbol) is not str or _SYMBOL_RE.fullmatch(symbol) is None for symbol in result
    ):
        _blocked("pool_symbols")
    return result


def _score_mapping(
    value: Mapping[str, Any],
    *,
    allowed_symbols: tuple[str, ...],
    complete: bool,
    label: str,
) -> dict[str, Decimal]:
    if not isinstance(value, Mapping):
        _blocked(f"{label}_mapping")
    keys = set(value)
    allowed = set(allowed_symbols)
    if (complete and keys != allowed) or (not complete and not keys <= allowed):
        _blocked(f"{label}_same_pool")
    return {
        symbol: _decimal(value[symbol], label=f"{label}.{symbol}")
        for symbol in allowed_symbols
        if symbol in value
    }


def _weak_percentiles(
    scores: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    if not scores:
        return {}
    count = Decimal(len(scores))
    values = tuple(scores.values())
    with localcontext() as context:
        context.prec = 40
        return {
            symbol: +(Decimal(sum(candidate <= score for candidate in values)) / count)
            for symbol, score in scores.items()
        }


def _policy(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("policy_semantic_sha")
    required = {
        "authority": NO_AUTHORITY,
        "canary_evidence_eligible": False,
        "formal_activation_eligible": False,
        "formal_research_publication_eligible": False,
        "fundamental_weight": "0.5",
        "performance_evidence_eligible": False,
        "promotion_eligible": False,
        "protocol_version": PROTOCOL_VERSION,
        "quant_weight": "0.5",
        "shadow_only": True,
        "state": FORWARD_ACCUMULATION_STATE,
        "version": SHADOW_FUSION_POLICY_VERSION,
    }
    if (
        any(normalized.get(key) != expected for key, expected in required.items())
        or normalized.get("strategy_id") is None
        or normalized.get("policy_id") is None
    ):
        _blocked("policy_contract")
    _path_id(normalized["strategy_id"], label="strategy_id")
    _path_id(normalized["policy_id"], label="policy_id")
    _session(
        normalized.get("effective_from_session"),
        label="effective_from_session",
    )
    created_at = _timestamp(normalized.get("created_at"), label="created_at")
    if normalized.get("cutoff") != created_at:
        _blocked("policy_cutoff")
    return normalized


def _artifact_ref(
    document: Mapping[str, Any],
    *,
    identity_field: str,
    relative_path: str,
) -> dict[str, str]:
    path = str(canonical_governed_path(relative_path))
    raw = canonical_resource_bytes(document)
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _identity_sha(
    *,
    strategy_id: str,
    decision_session: str,
    source_locator_semantic_sha256: str,
    input_bundle_sha256: str,
    factor_set_byte_sha256: str,
    policy_semantic_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "decision_session": decision_session,
                "factor_set_byte_sha256": factor_set_byte_sha256,
                "input_bundle_sha256": input_bundle_sha256,
                "policy_semantic_sha256": policy_semantic_sha256,
                "source_locator_semantic_sha256": (source_locator_semantic_sha256),
                "strategy_id": strategy_id,
            }
        )
    ).hexdigest()


def _evidence_group_sha(
    *,
    factor_set_byte_sha256: str,
    policy_semantic_sha256: str,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "factor_set_byte_sha256": factor_set_byte_sha256,
                "policy_semantic_sha256": policy_semantic_sha256,
            }
        )
    ).hexdigest()


def build_shadow_fusion_policy(
    *,
    policy_id: str,
    strategy_id: str,
    effective_from_session: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the fixed, authority-free zero-history accumulation policy."""

    policy = _path_id(policy_id, label="policy_id")
    strategy = _path_id(strategy_id, label="strategy_id")
    effective = _session(
        effective_from_session,
        label="effective_from_session",
    )
    timestamp = _timestamp(created_at, label="created_at")
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": timestamp,
            "cutoff": timestamp,
            "effective_from_session": effective,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "fundamental_weight": _decimal_text(FUNDAMENTAL_WEIGHT),
            "performance_evidence_eligible": False,
            "policy_id": policy,
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "quant_weight": _decimal_text(QUANT_WEIGHT),
            "shadow_only": True,
            "state": FORWARD_ACCUMULATION_STATE,
            "strategy_id": strategy,
            "version": SHADOW_FUSION_POLICY_VERSION,
        }
    )


def build_forward_fusion_top24(
    *,
    output_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    pool_symbols: Sequence[str],
    quant_scores: Mapping[str, Any],
    fundamental_scores: Mapping[str, Any],
    source_locator_semantic_sha256: str,
    input_bundle_sha256: str,
    factor_set_byte_sha256: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse same-pool branch scores under the fixed zero-history policy."""

    output = _path_id(output_id, label="output_id")
    strategy = _path_id(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    timestamp = _timestamp(cutoff, label="cutoff")
    source_sha = _sha256(
        source_locator_semantic_sha256,
        label="source_locator_semantic_sha256",
    )
    bundle_sha = _sha256(input_bundle_sha256, label="input_bundle_sha256")
    factor_sha = _sha256(
        factor_set_byte_sha256,
        label="factor_set_byte_sha256",
    )
    normalized_policy = _policy(policy)
    if (
        normalized_policy["strategy_id"] != strategy
        or normalized_policy["effective_from_session"] > session
        or normalized_policy["cutoff"] > timestamp
    ):
        _blocked("policy_binding")
    pool = _symbols(pool_symbols)
    quant = _score_mapping(
        quant_scores,
        allowed_symbols=pool,
        complete=True,
        label="quant_scores",
    )
    fundamental = _score_mapping(
        fundamental_scores,
        allowed_symbols=pool,
        complete=False,
        label="fundamental_scores",
    )
    quant_percentiles = _weak_percentiles(quant)
    fundamental_percentiles = _weak_percentiles(fundamental)
    with localcontext() as context:
        context.prec = 40
        fused = {
            symbol: +(
                QUANT_WEIGHT * quant_percentiles[symbol]
                + FUNDAMENTAL_WEIGHT * fundamental_percentiles.get(symbol, Decimal("0"))
            )
            for symbol in pool
        }
    selected = sorted(
        pool,
        key=lambda symbol: (-fused[symbol], symbol.encode("ascii")),
    )[:TOP_N]
    rows: list[dict[str, Any]] = []
    for rank, symbol in enumerate(selected, start=1):
        available = symbol in fundamental
        rows.append(
            {
                "base_target": _decimal_text(BASE_TARGET),
                "fundamental_available": available,
                "fundamental_percentile": _decimal_text(
                    fundamental_percentiles.get(symbol, Decimal("0"))
                ),
                "fundamental_score": (_decimal_text(fundamental[symbol]) if available else None),
                "fused_score": _decimal_text(fused[symbol]),
                "quant_percentile": _decimal_text(quant_percentiles[symbol]),
                "quant_score": _decimal_text(quant[symbol]),
                "rank": rank,
                "symbol": symbol,
            }
        )
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "cutoff": timestamp,
            "decision_session": session,
            "factor_set_byte_sha256": factor_sha,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "fundamental_available_count": len(fundamental),
            "fundamental_unavailable_count": len(pool) - len(fundamental),
            "input_bundle_sha256": bundle_sha,
            "output_id": output,
            "performance_evidence_eligible": False,
            "policy_semantic_sha256": str(normalized_policy["semantic_sha256"]),
            "pool_size": len(pool),
            "pool_symbols_sha256": hashlib.sha256(canonical_bytes(list(pool))).hexdigest(),
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "rows": rows,
            "shadow_only": True,
            "source_locator_semantic_sha256": source_sha,
            "state": FORWARD_ACCUMULATION_STATE,
            "strategy_id": strategy,
            "version": FUSION_TOP24_V2_VERSION,
        }
    )


def build_shadow_fusion_observation(
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    source_locator_semantic_sha256: str,
    input_bundle_sha256: str,
    factor_set_byte_sha256: str,
    policy: Mapping[str, Any],
    fusion_top24: Mapping[str, Any],
    fusion_relative_path: str,
    observation_id: str | None = None,
) -> dict[str, Any]:
    """Bind one immutable prediction session to all source identities."""

    strategy = _path_id(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    cutoff_at = _timestamp(cutoff, label="cutoff")
    created = _timestamp(created_at, label="created_at")
    if created != cutoff_at:
        _blocked("observation_created_at")
    source_sha = _sha256(
        source_locator_semantic_sha256,
        label="source_locator_semantic_sha256",
    )
    bundle_sha = _sha256(input_bundle_sha256, label="input_bundle_sha256")
    factor_sha = _sha256(
        factor_set_byte_sha256,
        label="factor_set_byte_sha256",
    )
    normalized_policy = _policy(policy)
    if normalized_policy["strategy_id"] != strategy:
        _blocked("observation_policy_strategy")
    policy_sha = str(normalized_policy["semantic_sha256"])
    identity_sha = _identity_sha(
        strategy_id=strategy,
        decision_session=session,
        source_locator_semantic_sha256=source_sha,
        input_bundle_sha256=bundle_sha,
        factor_set_byte_sha256=factor_sha,
        policy_semantic_sha256=policy_sha,
    )
    expected_id = f"shadow-fusion-{identity_sha}"
    if observation_id is not None and observation_id != expected_id:
        _blocked("observation_id_binding")
    if (
        fusion_top24.get("version") != FUSION_TOP24_V2_VERSION
        or fusion_top24.get("strategy_id") != strategy
        or fusion_top24.get("decision_session") != session
        or fusion_top24.get("cutoff") != cutoff_at
        or fusion_top24.get("source_locator_semantic_sha256") != source_sha
        or fusion_top24.get("input_bundle_sha256") != bundle_sha
        or fusion_top24.get("factor_set_byte_sha256") != factor_sha
        or fusion_top24.get("policy_semantic_sha256") != policy_sha
        or fusion_top24.get("state") != FORWARD_ACCUMULATION_STATE
        or fusion_top24.get("rows") is None
    ):
        _blocked("observation_fusion_binding")
    try:
        validate_semantic_sha(fusion_top24)
    except Exception:
        _blocked("observation_fusion_semantic_sha")
    fusion_ref = _artifact_ref(
        fusion_top24,
        identity_field="output_id",
        relative_path=fusion_relative_path,
    )
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "created_at": created,
            "cutoff": cutoff_at,
            "decision_session": session,
            "evidence_group_sha256": _evidence_group_sha(
                factor_set_byte_sha256=factor_sha,
                policy_semantic_sha256=policy_sha,
            ),
            "factor_set_byte_sha256": factor_sha,
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "fusion_top24_ref": fusion_ref,
            "input_bundle_sha256": bundle_sha,
            "observation_id": expected_id,
            "performance_evidence_eligible": False,
            "policy_semantic_sha256": policy_sha,
            "prediction_identity_sha256": identity_sha,
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "shadow_only": True,
            "source_locator_semantic_sha256": source_sha,
            "state": FORWARD_ACCUMULATION_STATE,
            "strategy_id": strategy,
            "version": SHADOW_FUSION_OBSERVATION_VERSION,
        }
    )


def _observation(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("observation_semantic_sha")
    required = {
        "authority": NO_AUTHORITY,
        **_INELIGIBLE,
        "protocol_version": PROTOCOL_VERSION,
        "state": FORWARD_ACCUMULATION_STATE,
        "version": SHADOW_FUSION_OBSERVATION_VERSION,
    }
    if any(normalized.get(key) != value for key, value in required.items()):
        _blocked("observation_contract")
    strategy = _path_id(normalized.get("strategy_id"), label="strategy_id")
    session = _session(
        normalized.get("decision_session"),
        label="decision_session",
    )
    source_sha = _sha256(
        normalized.get("source_locator_semantic_sha256"),
        label="source_locator_semantic_sha256",
    )
    bundle_sha = _sha256(
        normalized.get("input_bundle_sha256"),
        label="input_bundle_sha256",
    )
    factor_sha = _sha256(
        normalized.get("factor_set_byte_sha256"),
        label="factor_set_byte_sha256",
    )
    policy_sha = _sha256(
        normalized.get("policy_semantic_sha256"),
        label="policy_semantic_sha256",
    )
    identity_sha = _identity_sha(
        strategy_id=strategy,
        decision_session=session,
        source_locator_semantic_sha256=source_sha,
        input_bundle_sha256=bundle_sha,
        factor_set_byte_sha256=factor_sha,
        policy_semantic_sha256=policy_sha,
    )
    if (
        normalized.get("prediction_identity_sha256") != identity_sha
        or normalized.get("observation_id") != f"shadow-fusion-{identity_sha}"
        or normalized.get("evidence_group_sha256")
        != _evidence_group_sha(
            factor_set_byte_sha256=factor_sha,
            policy_semantic_sha256=policy_sha,
        )
    ):
        _blocked("observation_identity")
    return normalized


def _fusion_for_observation(
    fusion_top24: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        fusion = validate_semantic_sha(fusion_top24)
    except Exception:
        _blocked("label_fusion_semantic_sha")
    reference = observation.get("fusion_top24_ref")
    if (
        type(reference) is not dict
        or fusion.get("version") != FUSION_TOP24_V2_VERSION
        or fusion.get("strategy_id") != observation["strategy_id"]
        or fusion.get("decision_session") != observation["decision_session"]
        or fusion.get("factor_set_byte_sha256") != observation["factor_set_byte_sha256"]
        or fusion.get("policy_semantic_sha256") != observation["policy_semantic_sha256"]
        or reference.get("artifact_id") != fusion.get("output_id")
        or reference.get("artifact_version") != fusion.get("version")
        or reference.get("semantic_sha256") != fusion.get("semantic_sha256")
        or reference.get("byte_sha256")
        != hashlib.sha256(canonical_resource_bytes(fusion)).hexdigest()
    ):
        _blocked("label_fusion_binding")
    rows = fusion.get("rows")
    if (
        type(rows) is not list
        or len(rows) != TOP_N
        or [row.get("rank") for row in rows] != list(range(1, TOP_N + 1))
    ):
        _blocked("label_fusion_rows")
    return fusion


def build_shadow_fusion_matured_label(
    *,
    observation: Mapping[str, Any],
    fusion_top24: Mapping[str, Any],
    observation_relative_path: str,
    horizon_sessions: int,
    shanghai_sessions: Sequence[str],
    origin_closes: Mapping[str, Any],
    end_closes: Mapping[str, Any],
    matured_at: str,
) -> dict[str, Any]:
    """Build one non-pooled label at its exact future Shanghai close."""

    prediction = _observation(observation)
    fusion = _fusion_for_observation(fusion_top24, prediction)
    if type(horizon_sessions) is not int or horizon_sessions not in LABEL_HORIZONS:
        _blocked("label_horizon")
    if (
        isinstance(shanghai_sessions, (str, bytes))
        or not isinstance(shanghai_sessions, Sequence)
        or len(shanghai_sessions) != horizon_sessions + 1
    ):
        _blocked("label_session_count")
    sessions = tuple(_session(value, label="label_session") for value in shanghai_sessions)
    if sessions[0] != prediction["decision_session"] or any(
        earlier >= later for earlier, later in zip(sessions, sessions[1:])
    ):
        _blocked("label_session_sequence")
    end_session = sessions[-1]
    matured = _timestamp(matured_at, label="matured_at")
    matured_in_shanghai = datetime.fromisoformat(matured.replace("Z", "+00:00")).astimezone(
        SHANGHAI
    )
    if (
        matured_in_shanghai.date().isoformat() != end_session
        or matured_in_shanghai.timetz().replace(tzinfo=None) < time(15, 0)
    ):
        _blocked("label_not_exact_future_close")
    symbols = tuple(row["symbol"] for row in fusion["rows"])
    origin = _score_mapping(
        origin_closes,
        allowed_symbols=symbols,
        complete=True,
        label="origin_closes",
    )
    end = _score_mapping(
        end_closes,
        allowed_symbols=symbols,
        complete=True,
        label="end_closes",
    )
    for symbol in symbols:
        if origin[symbol] <= 0 or end[symbol] <= 0:
            _blocked(f"label_close_nonpositive:{symbol}")
    with localcontext() as context:
        context.prec = 40
        rows = [
            {
                "end_close": _decimal_text(end[symbol]),
                "forward_return": _decimal_text(+(end[symbol] / origin[symbol] - Decimal("1"))),
                "origin_close": _decimal_text(origin[symbol]),
                "rank": rank,
                "symbol": symbol,
            }
            for rank, symbol in enumerate(symbols, start=1)
        ]
    calendar_sha = hashlib.sha256(canonical_bytes(list(sessions))).hexdigest()
    close_bundle_sha = hashlib.sha256(
        canonical_bytes(
            {
                "end_closes": [
                    {"close": _decimal_text(end[symbol]), "symbol": symbol} for symbol in symbols
                ],
                "end_session": end_session,
                "origin_closes": [
                    {
                        "close": _decimal_text(origin[symbol]),
                        "symbol": symbol,
                    }
                    for symbol in symbols
                ],
                "origin_session": sessions[0],
            }
        )
    ).hexdigest()
    label_identity_sha = hashlib.sha256(
        canonical_bytes(
            {
                "close_bundle_sha256": close_bundle_sha,
                "evidence_group_sha256": prediction["evidence_group_sha256"],
                "horizon_sessions": horizon_sessions,
                "label_end_session": end_session,
                "observation_id": prediction["observation_id"],
                "session_calendar_sha256": calendar_sha,
            }
        )
    ).hexdigest()
    observation_ref = _artifact_ref(
        prediction,
        identity_field="observation_id",
        relative_path=observation_relative_path,
    )
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "canary_evidence_eligible": False,
            "close_bundle_sha256": close_bundle_sha,
            "cutoff": matured,
            "evidence_group_sha256": prediction["evidence_group_sha256"],
            "factor_set_byte_sha256": prediction["factor_set_byte_sha256"],
            "formal_activation_eligible": False,
            "formal_research_publication_eligible": False,
            "horizon_sessions": horizon_sessions,
            "label_end_session": end_session,
            "label_id": f"shadow-fusion-label-{label_identity_sha}",
            "matured_at": matured,
            "observation_ref": observation_ref,
            "origin_session": sessions[0],
            "performance_evidence_eligible": False,
            "policy_semantic_sha256": prediction["policy_semantic_sha256"],
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "rows": rows,
            "session_calendar_sha256": calendar_sha,
            "shadow_only": True,
            "state": FORWARD_ACCUMULATION_STATE,
            "strategy_id": prediction["strategy_id"],
            "version": SHADOW_FUSION_MATURED_LABEL_VERSION,
        }
    )


class ForwardFusionWriter(GovernedStore):
    """Writer restricted to the additive forward-Fusion Shadow subtree."""

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        prefix = (*FORWARD_FUSION_ROOT.parts, "strategies")
        if path.parts[: len(prefix)] != prefix or len(path.parts) != len(prefix) + 3:
            raise SourceStorageSecurityError(
                "path is outside the forward-Fusion Shadow writer root"
            )
        strategy, category, filename = path.parts[-3:]
        if _PATH_ID_RE.fullmatch(strategy) is None:
            raise SourceStorageSecurityError("forward-Fusion strategy path is invalid")
        if category not in {
            "fusions",
            "labels",
            "observations",
            "policies",
        }:
            raise SourceStorageSecurityError("forward-Fusion writer category is invalid")
        if not filename.endswith(".json") or _PATH_ID_RE.fullmatch(filename[:-5]) is None:
            raise SourceStorageSecurityError("forward-Fusion writer filename is invalid")
        return path


def _path(
    *,
    strategy_id: str,
    category: str,
    identity: str,
) -> str:
    strategy = _path_id(strategy_id, label="strategy_id")
    value = _path_id(identity, label=f"{category}_id")
    return str(FORWARD_FUSION_ROOT / "strategies" / strategy / category / f"{value}.json")


def publish_forward_fusion_prediction(
    workspace_root: str,
    *,
    policy: Mapping[str, Any],
    output_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    pool_symbols: Sequence[str],
    quant_scores: Mapping[str, Any],
    fundamental_scores: Mapping[str, Any],
    source_locator_semantic_sha256: str,
    input_bundle_sha256: str,
    factor_set_byte_sha256: str,
) -> dict[str, Any]:
    """Publish one exact-once session prediction under the narrow writer."""

    normalized_policy = _policy(policy)
    strategy = _path_id(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    if normalized_policy["strategy_id"] != strategy:
        _blocked("publish_policy_strategy")
    fusion_path = _path(
        strategy_id=strategy,
        category="fusions",
        identity=session,
    )
    fusion = build_forward_fusion_top24(
        output_id=output_id,
        strategy_id=strategy,
        decision_session=session,
        cutoff=cutoff,
        pool_symbols=pool_symbols,
        quant_scores=quant_scores,
        fundamental_scores=fundamental_scores,
        source_locator_semantic_sha256=(source_locator_semantic_sha256),
        input_bundle_sha256=input_bundle_sha256,
        factor_set_byte_sha256=factor_set_byte_sha256,
        policy=normalized_policy,
    )
    observation = build_shadow_fusion_observation(
        strategy_id=strategy,
        decision_session=session,
        cutoff=cutoff,
        created_at=cutoff,
        source_locator_semantic_sha256=(source_locator_semantic_sha256),
        input_bundle_sha256=input_bundle_sha256,
        factor_set_byte_sha256=factor_set_byte_sha256,
        policy=normalized_policy,
        fusion_top24=fusion,
        fusion_relative_path=fusion_path,
    )
    policy_path = _path(
        strategy_id=strategy,
        category="policies",
        identity=str(normalized_policy["policy_id"]),
    )
    observation_path = _path(
        strategy_id=strategy,
        category="observations",
        identity=session,
    )
    writer = ForwardFusionWriter(workspace_root)
    writer.initialize()
    observed = writer.read_optional(observation_path)
    expected_observation_raw = canonical_resource_bytes(observation)
    if observed is not None and observed.data != expected_observation_raw:
        _blocked("conflicting_duplicate_session")
    try:
        policy_result = writer.write_exact_once(
            policy_path,
            canonical_resource_bytes(normalized_policy),
        )
        fusion_result = writer.write_exact_once(
            fusion_path,
            canonical_resource_bytes(fusion),
        )
        observation_result = writer.write_exact_once(
            observation_path,
            expected_observation_raw,
        )
    except SourceExactOnceConflict as exc:
        raise ForwardFusionError(
            "V17_V4_FORWARD_FUSION_BLOCKED:" "conflicting_duplicate_session"
        ) from exc
    except SourceStorageError as exc:
        raise ForwardFusionError("V17_V4_FORWARD_FUSION_BLOCKED:prediction_write") from exc
    return {
        "created": observation_result.created,
        "fusion_created": fusion_result.created,
        "fusion_path": fusion_path,
        "fusion_top24": fusion,
        "observation": observation,
        "observation_path": observation_path,
        "policy": normalized_policy,
        "policy_created": policy_result.created,
        "policy_path": policy_path,
    }


def publish_shadow_fusion_matured_label(
    workspace_root: str,
    *,
    observation: Mapping[str, Any],
    fusion_top24: Mapping[str, Any],
    observation_relative_path: str,
    horizon_sessions: int,
    shanghai_sessions: Sequence[str],
    origin_closes: Mapping[str, Any],
    end_closes: Mapping[str, Any],
    matured_at: str,
) -> dict[str, Any]:
    """Publish one horizon label without pooling or replacing any label."""

    label = build_shadow_fusion_matured_label(
        observation=observation,
        fusion_top24=fusion_top24,
        observation_relative_path=observation_relative_path,
        horizon_sessions=horizon_sessions,
        shanghai_sessions=shanghai_sessions,
        origin_closes=origin_closes,
        end_closes=end_closes,
        matured_at=matured_at,
    )
    prediction = _observation(observation)
    label_path = _path(
        strategy_id=str(prediction["strategy_id"]),
        category="labels",
        identity=(f"{prediction['observation_id']}-h{horizon_sessions}"),
    )
    writer = ForwardFusionWriter(workspace_root)
    writer.initialize()
    try:
        expected_observation_raw = canonical_resource_bytes(observation)
        if (
            writer.read(
                observation_relative_path,
                hashlib.sha256(expected_observation_raw).hexdigest(),
            )
            != expected_observation_raw
        ):
            _blocked("label_observation_readback")
        fusion_ref = observation["fusion_top24_ref"]
        expected_fusion_raw = canonical_resource_bytes(fusion_top24)
        if (
            type(fusion_ref) is not dict
            or writer.read(
                str(fusion_ref["relative_path"]),
                str(fusion_ref["byte_sha256"]),
            )
            != expected_fusion_raw
        ):
            _blocked("label_fusion_readback")
    except (KeyError, SourceStorageError) as exc:
        raise ForwardFusionError("V17_V4_FORWARD_FUSION_BLOCKED:label_source_readback") from exc
    try:
        result = writer.write_exact_once(
            label_path,
            canonical_resource_bytes(label),
        )
    except SourceExactOnceConflict as exc:
        raise ForwardFusionError(
            "V17_V4_FORWARD_FUSION_BLOCKED:" "conflicting_duplicate_label"
        ) from exc
    except SourceStorageError as exc:
        raise ForwardFusionError("V17_V4_FORWARD_FUSION_BLOCKED:label_write") from exc
    return {
        "created": result.created,
        "label": label,
        "label_path": label_path,
    }


__all__ = [
    "BASE_TARGET",
    "FORWARD_ACCUMULATION_STATE",
    "FUNDAMENTAL_WEIGHT",
    "FUSION_TOP24_V2_VERSION",
    "ForwardFusionError",
    "ForwardFusionWriter",
    "LABEL_HORIZONS",
    "QUANT_WEIGHT",
    "SHADOW_FUSION_MATURED_LABEL_VERSION",
    "SHADOW_FUSION_OBSERVATION_VERSION",
    "SHADOW_FUSION_POLICY_VERSION",
    "build_forward_fusion_top24",
    "build_shadow_fusion_matured_label",
    "build_shadow_fusion_observation",
    "build_shadow_fusion_policy",
    "publish_forward_fusion_prediction",
    "publish_shadow_fusion_matured_label",
]
