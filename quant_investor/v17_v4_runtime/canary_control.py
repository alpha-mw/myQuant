"""Historical replay policy and crash-safe operational canary control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, NoReturn, Sequence

from quant_investor.research_runtime_control import (
    ResearchRuntimeControl,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
)
from quant_investor.v17_v4_contract.identities import (
    require_opaque_id,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.eligibility_control import (
    EligibilityService,
)
from quant_investor.v17_v4_runtime.formal_activation import artifact_ref
from quant_investor.v17_v4_runtime.source_storage import (
    CANARY_ROOT,
    EMPTY_SHA256,
    ExactReferenceReader,
    GovernedStore,
    SourceStorageError,
    SourceStorageSecurityError,
    canonical_governed_path,
)

COMPARISON_VERSION: Final = "myquant.v17.v4.dual-run-comparison.v1"
POLICY_VERSION: Final = "myquant.v17.v4.historical-canary-policy.v1"
INTENT_VERSION: Final = "myquant.v17.v4.canary-transition-intent.v1"
POINTER_VERSION: Final = "myquant.v17.v4.canary-pointer.v1"
COMPLETION_VERSION: Final = "myquant.v17.v4.canary-receipt.v1"
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
PUBLICATION_AUTHORITY: Final = {
    **NO_AUTHORITY,
    "formal_research_publication": True,
}
_EXACT_INPUTS: Final = (
    "benchmark",
    "canonical_calendar",
    "holdings_snapshot",
    "market_bars",
)
_MINIMUM_METRICS: Final = (
    "rank_overlap",
    "v15_top12_recall_in_v4_top24",
)
_MAXIMUM_METRICS: Final = (
    "cash_exposure_difference",
    "cluster_exposure_difference",
    "gross_exposure_difference",
    "industry_exposure_difference",
    "l1_portfolio_distance",
    "max_common_name_target_difference",
    "turnover_difference",
)
_STATUS_BY_TRANSITION: Final = {
    "START": "CANARY_STARTED",
    "COMPLETE": "CANARY_COMPLETED",
    "FAIL": "CANARY_FAILED",
}
_CANARY_COUNTERS: Final = (
    "active_run_cas_mismatch_count",
    "analysis_time_provider_call_count",
    "broker_call_count",
    "canary_pointer_cas_mismatch_count",
    "data_pointer_cas_mismatch_count",
    "eligibility_pointer_cas_mismatch_count",
    "execution_call_count",
    "factor_pointer_cas_mismatch_count",
    "formal_pointer_cas_mismatch_count",
    "llm_control_call_count",
    "order_call_count",
    "protocol_target_cas_mismatch_count",
    "selector_cas_mismatch_count",
    "trade_call_count",
)


class CanaryError(RuntimeError):
    """Raised when replay or canary evidence fails closed."""


class CanaryCrash(RuntimeError):
    """Deterministic fault injection for canary recovery tests."""


@dataclass(frozen=True)
class CanaryResult:
    status: str
    intent_ref: Mapping[str, str]
    pointer_ref: Mapping[str, str]
    completion_ref: Mapping[str, str]
    recovered: bool


@dataclass(frozen=True)
class CanaryState:
    status: str
    intent: Mapping[str, Any] | None
    pointer: Mapping[str, Any] | None
    completion: Mapping[str, Any] | None


def _blocked(reason: str) -> NoReturn:
    raise CanaryError(f"V17_V4_CANARY_BLOCKED:{reason}")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ordered_refs(
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (dict(reference) for reference in references),
        key=lambda row: (
            row["relative_path"],
            row["byte_sha256"],
            row["artifact_id"],
        ),
    )


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool):
        _blocked(f"{label.upper()}_DECIMAL")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CanaryError(
            f"V17_V4_CANARY_BLOCKED:{label.upper()}_DECIMAL"
        ) from exc
    if not result.is_finite():
        _blocked(f"{label.upper()}_DECIMAL")
    return result


def _decimal_text(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return "0" if rendered in {"-0", ""} else rendered


def build_dual_run_comparison(
    *,
    comparison_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    stage: str,
    v15_run_ref: Mapping[str, Any],
    v4_run_ref: Mapping[str, Any],
    comparison_inputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    latency_seconds: Mapping[str, Any],
    metrics: Mapping[str, Any],
    risk_invariants: Mapping[str, bool],
    side_effect_counters: Mapping[str, int],
) -> dict[str, Any]:
    """Seal a comparison after deriving comparability from exact input bytes."""

    expected_inputs = {*_EXACT_INPUTS, "source_closure"}
    if set(comparison_inputs) != expected_inputs:
        _blocked("COMPARISON_INPUT_INVENTORY")
    mismatched: list[dict[str, Any]] = []
    normalized_inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for name in sorted(expected_inputs):
        pair = comparison_inputs[name]
        if set(pair) != {"v15_ref", "v4_ref"}:
            _blocked("COMPARISON_INPUT_PAIR")
        left = dict(pair["v15_ref"])
        right = dict(pair["v4_ref"])
        normalized_inputs[name] = {"v15_ref": left, "v4_ref": right}
        if (
            name in _EXACT_INPUTS
            and left.get("byte_sha256") != right.get("byte_sha256")
        ):
            mismatched.extend((left, right))
    classification = "NON_COMPARABLE" if mismatched else "COMPARABLE"
    document = seal_semantic(
        {
            "authority": dict(PUBLICATION_AUTHORITY),
            "classification": classification,
            "comparison_id": require_opaque_id(
                comparison_id,
                label="comparison_id",
            ),
            "comparison_inputs": normalized_inputs,
            "cutoff": require_utc_timestamp(cutoff, label="cutoff"),
            "decision_session": decision_session,
            "differing_refs": _ordered_refs(mismatched),
            "latency_seconds": {
                "v15": _decimal_text(
                    _decimal(latency_seconds["v15"], label="v15_latency")
                ),
                "v4": _decimal_text(
                    _decimal(latency_seconds["v4"], label="v4_latency")
                ),
            },
            "metrics": dict(metrics),
            "protocol_version": PROTOCOL_VERSION,
            "risk_invariants": dict(risk_invariants),
            "side_effect_counters": dict(side_effect_counters),
            "stage": stage,
            "strategy_id": require_opaque_id(
                strategy_id,
                label="strategy_id",
            ),
            "v15_protocol_id": "myquant.v15",
            "v15_run_ref": dict(v15_run_ref),
            "v4_protocol_id": PROTOCOL_VERSION,
            "v4_run_ref": dict(v4_run_ref),
            "version": COMPARISON_VERSION,
        }
    )
    validate_artifact(document)
    return document


def _nearest_rank(
    values: Sequence[Decimal],
    quantile: Decimal,
) -> Decimal:
    if not values:
        _blocked("EMPTY_QUANTILE")
    ordered = sorted(values)
    rank = max(
        1,
        int(
            (quantile * len(ordered)).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
    )
    return ordered[rank - 1]


def build_historical_canary_policy(
    *,
    policy_id: str,
    strategy_id: str,
    created_at: str,
    comparison_pairs: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Compute sealed 5th/95th percentile bands from 60 comparable origins."""

    if len(comparison_pairs) != 60:
        _blocked("HISTORICAL_ORIGIN_COUNT")
    ordered = sorted(
        comparison_pairs,
        key=lambda pair: str(pair[0].get("decision_session", "")),
    )
    try:
        sessions = [
            date.fromisoformat(str(pair[0]["decision_session"]))
            for pair in ordered
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise CanaryError(
            "V17_V4_CANARY_BLOCKED:HISTORICAL_SESSION"
        ) from exc
    if len(set(sessions)) != 60:
        _blocked("HISTORICAL_SESSION_DUPLICATE")
    for previous, current in zip(sessions, sessions[1:]):
        month_delta = (
            (current.year - previous.year) * 12
            + current.month
            - previous.month
        )
        if month_delta != 1:
            _blocked("HISTORICAL_MONTH_GAP")
    refs: list[dict[str, Any]] = []
    documents: list[Mapping[str, Any]] = []
    strategy = require_opaque_id(strategy_id, label="strategy_id")
    created = require_utc_timestamp(created_at, label="created_at")
    for document, reference in ordered:
        validated = validate_artifact(document)
        expected = artifact_ref(
            document,
            relative_path=str(reference["relative_path"]),
        )
        if (
            validated.version != COMPARISON_VERSION
            or document.get("stage") != "HISTORICAL_REPLAY"
            or document.get("classification") != "COMPARABLE"
            or document.get("strategy_id") != strategy
            or expected != dict(reference)
            or str(reference["cutoff"]) > created
        ):
            _blocked("HISTORICAL_COMPARISON")
        documents.append(document)
        refs.append(dict(reference))
    minimum = {
        name: _decimal_text(
            _nearest_rank(
                [_decimal(row["metrics"][name], label=name) for row in documents],
                Decimal("0.05"),
            )
        )
        for name in _MINIMUM_METRICS
    }
    maximum = {
        name: _decimal_text(
            _nearest_rank(
                [
                    abs(_decimal(row["metrics"][name], label=name))
                    for row in documents
                ],
                Decimal("0.95"),
            )
        )
        for name in _MAXIMUM_METRICS
    }
    policy = seal_semantic(
        {
            "absolute_risk_limits": {
                "held_name_positive_increase_count": 0,
                "macro_increase_count": 0,
                "markov_increase_count": 0,
                "v4_gross_excess_max": "0.05",
                "veto_positive_delta_count": 0,
            },
            "authority": dict(PUBLICATION_AUTHORITY),
            "created_at": created,
            "maximum_bands": maximum,
            "minimum_bands": minimum,
            "origin_count": 60,
            "pair_refs": refs,
            "policy_id": require_opaque_id(policy_id, label="policy_id"),
            "protocol_version": PROTOCOL_VERSION,
            "quantile_method": "empirical_nearest_rank",
            "strategy_id": strategy,
            "version": POLICY_VERSION,
        }
    )
    validate_artifact(policy)
    return policy


def evaluate_operational_canary(
    policy: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Evaluate five forward sessions against one sealed historical policy."""

    validate_artifact(policy)
    if len(comparisons) != 5:
        _blocked("OPERATIONAL_SESSION_COUNT")
    sessions = [str(row.get("decision_session", "")) for row in comparisons]
    if sessions != sorted(sessions) or len(set(sessions)) != 5:
        _blocked("OPERATIONAL_SESSION_ORDER")
    total_counters: dict[str, int] = {
        name: 0 for name in _CANARY_COUNTERS
    }
    checks: dict[str, bool] = {
        "five-of-five": True,
        "historical-bands": True,
        "latency": True,
        "risk-invariants": True,
        "side-effects": True,
    }
    for comparison in comparisons:
        validate_artifact(comparison)
        if (
            comparison.get("stage") != "OPERATIONAL_CANARY"
            or comparison.get("classification") != "COMPARABLE"
            or comparison.get("strategy_id") != policy.get("strategy_id")
        ):
            checks["five-of-five"] = False
        metrics = comparison["metrics"]
        for name, floor in policy["minimum_bands"].items():
            if _decimal(metrics[name], label=name) < _decimal(
                floor,
                label=f"{name}_floor",
            ):
                checks["historical-bands"] = False
        for name, ceiling in policy["maximum_bands"].items():
            if abs(_decimal(metrics[name], label=name)) > _decimal(
                ceiling,
                label=f"{name}_ceiling",
            ):
                checks["historical-bands"] = False
        latency = comparison["latency_seconds"]
        v15_latency = _decimal(latency["v15"], label="v15_latency")
        v4_latency = _decimal(latency["v4"], label="v4_latency")
        if v4_latency > Decimal("1800") or v4_latency > v15_latency * 2:
            checks["latency"] = False
        if (
            not all(comparison["risk_invariants"].values())
            or int(metrics["held_name_positive_increase_count"]) != 0
        ):
            checks["risk-invariants"] = False
        for name, value in comparison["side_effect_counters"].items():
            total_counters[name] = total_counters.get(name, 0) + int(value)
            if value != 0:
                checks["side-effects"] = False
    threshold_results = [
        {
            "observed": "1" if passed else "0",
            "status": "PASS" if passed else "FAIL",
            "threshold_id": name,
        }
        for name, passed in sorted(checks.items())
    ]
    return threshold_results, total_counters


def build_canary_transition_intent(
    *,
    intent_id: str,
    strategy_id: str,
    cutoff: str,
    created_at: str,
    transition: str,
    expected_pointer_sha256: str,
    eligibility_pointer_ref: Mapping[str, Any],
    historical_canary_policy_ref: Mapping[str, Any],
    v15_protocol_target_ref: Mapping[str, Any],
    v15_active_run_pointer_ref: Mapping[str, Any],
    session_window: Mapping[str, Any],
    paired_run_ids: Sequence[str],
    comparison_refs: Sequence[Mapping[str, Any]] = (),
    completed_sessions: Sequence[str] = (),
    threshold_results: Sequence[Mapping[str, Any]] = (),
    side_effect_counters: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    state = {
        "START": ("DEFAULT_ELIGIBLE", "CANARY"),
        "COMPLETE": ("CANARY", "CANARY"),
        "FAIL": ("CANARY", "DEFAULT_ELIGIBLE"),
    }
    if transition not in state:
        _blocked("TRANSITION")
    explicit = [
        dict(eligibility_pointer_ref),
        dict(historical_canary_policy_ref),
        dict(v15_protocol_target_ref),
        dict(v15_active_run_pointer_ref),
        *(dict(reference) for reference in comparison_refs),
    ]
    document: dict[str, Any] = {
        "authority": dict(NO_AUTHORITY),
        "created_at": created_at,
        "cutoff": cutoff,
        "eligibility_pointer_ref": dict(eligibility_pointer_ref),
        "evidence_refs": _ordered_refs(explicit),
        "expected_pointer_sha256": expected_pointer_sha256,
        "from_state": state[transition][0],
        "historical_canary_policy_ref": dict(
            historical_canary_policy_ref
        ),
        "intent_id": require_opaque_id(intent_id, label="intent_id"),
        "paired_run_ids": list(paired_run_ids),
        "protocol_version": PROTOCOL_VERSION,
        "session_window": dict(session_window),
        "strategy_id": require_opaque_id(
            strategy_id,
            label="strategy_id",
        ),
        "to_state": state[transition][1],
        "transition": transition,
        "v15_active_run_pointer_ref": dict(v15_active_run_pointer_ref),
        "v15_protocol_target_ref": dict(v15_protocol_target_ref),
        "version": INTENT_VERSION,
    }
    if transition != "START":
        document.update(
            {
                "comparison_refs": [
                    dict(reference) for reference in comparison_refs
                ],
                "completed_sessions": list(completed_sessions),
                "side_effect_counters": dict(
                    side_effect_counters or {}
                ),
                "threshold_results": [
                    dict(result) for result in threshold_results
                ],
            }
        )
    sealed = seal_semantic(document)
    validate_artifact(sealed)
    return sealed


class _CanaryWriter(GovernedStore):
    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        if (
            parts[:3] != ("results", "v17_v4_canary", "strategies")
            or len(parts) < 5
        ):
            raise SourceStorageSecurityError(
                "path is outside canary strategy roots"
            )
        try:
            require_opaque_id(parts[3], label="strategy_id")
        except ValueError as exc:
            raise SourceStorageSecurityError(
                "canary strategy identity is invalid"
            ) from exc
        suffix = parts[4:]
        if suffix in {(".current.lock",), ("_current.json",)}:
            return path
        if (
            len(suffix) == 3
            and suffix[:2]
            in {
                ("transitions", "completion_receipts"),
                ("transitions", "intents"),
            }
            and suffix[2].endswith(".json")
        ):
            try:
                require_opaque_id(
                    suffix[2][:-5],
                    label="transition id",
                )
            except ValueError as exc:
                raise SourceStorageSecurityError(
                    "canary transition identity is invalid"
                ) from exc
            return path
        if (
            len(suffix) == 3
            and suffix[:2] == ("historical", "comparisons")
            and suffix[2].endswith(".json")
        ):
            try:
                require_opaque_id(
                    suffix[2][:-5],
                    label="comparison id",
                )
            except ValueError as exc:
                raise SourceStorageSecurityError(
                    "historical comparison identity is invalid"
                ) from exc
            return path
        if (
            len(suffix) == 2
            and suffix[0] == "policies"
            and suffix[1].endswith(".json")
        ):
            try:
                require_opaque_id(
                    suffix[1][:-5],
                    label="policy id",
                )
            except ValueError as exc:
                raise SourceStorageSecurityError(
                    "historical policy identity is invalid"
                ) from exc
            return path
        if (
            len(suffix) == 3
            and suffix[0] == "operational"
            and suffix[2].endswith(".json")
        ):
            try:
                date.fromisoformat(suffix[1])
                require_opaque_id(
                    suffix[2][:-5],
                    label="comparison id",
                )
            except ValueError as exc:
                raise SourceStorageSecurityError(
                    "canary operational path identity is invalid"
                ) from exc
            return path
        raise SourceStorageSecurityError(
            "path is outside the canary writer whitelist"
        )


class CanaryService:
    """CAS only the isolated canary pointer; never the shared selector."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        repo_root: str | Path,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._repo_root = Path(repo_root)
        self._writer = _CanaryWriter(workspace_root)
        self._reader = ExactReferenceReader(workspace_root)

    @staticmethod
    def _root(strategy_id: str) -> PurePosixPath:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        return CANARY_ROOT / "strategies" / strategy

    def _paths(
        self,
        strategy_id: str,
        intent_id: str,
    ) -> dict[str, PurePosixPath]:
        root = self._root(strategy_id)
        identity = require_opaque_id(intent_id, label="intent_id")
        return {
            "completion": (
                root
                / "transitions"
                / "completion_receipts"
                / f"{identity}.json"
            ),
            "intent": (
                root / "transitions" / "intents" / f"{identity}.json"
            ),
            "lock": root / ".current.lock",
            "pointer": root / "_current.json",
        }

    def _load_v4(
        self,
        reference: Mapping[str, Any],
        *,
        expected_version: str,
    ) -> dict[str, Any]:
        try:
            raw = self._reader.read(
                str(reference["relative_path"]),
                str(reference["byte_sha256"]),
            )
            document = load_canonical_resource(raw, label=expected_version)
            if type(document) is not dict:
                _blocked("ARTIFACT_ROOT")
            validated = validate_artifact(document)
            identity = artifact_identity_field(expected_version)
        except (SourceStorageError, TypeError, ValueError) as exc:
            raise CanaryError(
                "V17_V4_CANARY_BLOCKED:ARTIFACT_VALIDATION"
            ) from exc
        if (
            validated.version != expected_version
            or document.get(identity) != reference.get("artifact_id")
            or document.get("semantic_sha256")
            != reference.get("semantic_sha256")
            or document.get("strategy_id") != reference.get("strategy_id")
            or _sha(raw) != reference.get("byte_sha256")
        ):
            _blocked("ARTIFACT_REFERENCE")
        return dict(document)

    def publish_historical_policy(
        self,
        *,
        policy_id: str,
        strategy_id: str,
        cutoff: str,
        created_at: str,
        comparisons: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Publish exactly 60 replay comparisons and their derived policy."""

        strategy = require_opaque_id(strategy_id, label="strategy_id")
        reference_cutoff = require_utc_timestamp(
            cutoff,
            label="cutoff",
        )
        if created_at < reference_cutoff:
            _blocked("POLICY_CREATED_AT")
        root = self._root(strategy)
        pairs: list[
            tuple[Mapping[str, Any], Mapping[str, Any]]
        ] = []
        paths: list[PurePosixPath] = []
        for comparison in comparisons:
            validate_artifact(comparison)
            if (
                comparison.get("version") != COMPARISON_VERSION
                or comparison.get("stage") != "HISTORICAL_REPLAY"
                or comparison.get("strategy_id") != strategy
            ):
                _blocked("HISTORICAL_PUBLISH_INPUT")
            identity = str(comparison["comparison_id"])
            path = (
                root
                / "historical"
                / "comparisons"
                / f"{identity}.json"
            )
            reference = artifact_ref(
                comparison,
                relative_path=str(path),
            )
            pairs.append((comparison, reference))
            paths.append(path)
        policy = build_historical_canary_policy(
            policy_id=policy_id,
            strategy_id=strategy,
            created_at=created_at,
            comparison_pairs=pairs,
        )
        policy_path = root / "policies" / f"{policy_id}.json"
        for path, comparison in zip(paths, comparisons):
            self._writer.write_exact_once(
                path,
                canonical_resource_bytes(comparison),
            )
        self._writer.write_exact_once(
            policy_path,
            canonical_resource_bytes(policy),
        )
        return policy, artifact_ref(
            policy,
            relative_path=str(policy_path),
            cutoff=reference_cutoff,
        )

    def publish_operational_comparison(
        self,
        comparison: Mapping[str, Any],
    ) -> dict[str, str]:
        """Publish one immutable forward-session comparison."""

        validate_artifact(comparison)
        if (
            comparison.get("version") != COMPARISON_VERSION
            or comparison.get("stage") != "OPERATIONAL_CANARY"
        ):
            _blocked("OPERATIONAL_PUBLISH_INPUT")
        strategy = require_opaque_id(
            str(comparison["strategy_id"]),
            label="strategy_id",
        )
        session = date.fromisoformat(
            str(comparison["decision_session"])
        ).isoformat()
        identity = require_opaque_id(
            str(comparison["comparison_id"]),
            label="comparison_id",
        )
        path = (
            self._root(strategy)
            / "operational"
            / session
            / f"{identity}.json"
        )
        self._writer.write_exact_once(
            path,
            canonical_resource_bytes(comparison),
        )
        return artifact_ref(comparison, relative_path=str(path))

    def _revalidate_intent(
        self,
        intent: Mapping[str, Any],
    ) -> None:
        validate_artifact(intent)
        strategy = str(intent["strategy_id"])
        eligibility = EligibilityService(
            self._workspace_root,
            repo_root=self._repo_root,
        ).resolve(strategy)
        if eligibility.status != "DEFAULT_ELIGIBLE" or eligibility.pointer is None:
            _blocked("ELIGIBILITY")
        eligibility_path = (
            "results/v17_v4_formal_research/strategies/"
            f"{strategy}/eligibility/_active.json"
        )
        if artifact_ref(
            eligibility.pointer,
            relative_path=eligibility_path,
        ) != intent["eligibility_pointer_ref"]:
            _blocked("ELIGIBILITY_POINTER")
        policy = self._load_v4(
            intent["historical_canary_policy_ref"],
            expected_version=POLICY_VERSION,
        )
        historical_pairs = [
            (
                self._load_v4(
                    reference,
                    expected_version=COMPARISON_VERSION,
                ),
                reference,
            )
            for reference in policy["pair_refs"]
        ]
        recomputed_policy = build_historical_canary_policy(
            policy_id=str(policy["policy_id"]),
            strategy_id=str(policy["strategy_id"]),
            created_at=str(policy["created_at"]),
            comparison_pairs=historical_pairs,
        )
        if recomputed_policy != policy:
            _blocked("HISTORICAL_POLICY_READBACK")
        selector, target, pointer, _run = ResearchRuntimeControl(
            str(self._workspace_root)
        ).resolve_current(strategy_id=strategy)
        if (
            selector.document.get("status") != "V15_DEFAULT"
            or target.reference != intent["v15_protocol_target_ref"]
            or pointer.reference != intent["v15_active_run_pointer_ref"]
        ):
            _blocked("V15_STANDBY")
        if intent["transition"] != "START":
            comparisons = [
                self._load_v4(
                    reference,
                    expected_version=COMPARISON_VERSION,
                )
                for reference in intent["comparison_refs"]
            ]
            thresholds, counters = evaluate_operational_canary(
                policy,
                comparisons,
            )
            sessions = [
                str(comparison["decision_session"])
                for comparison in comparisons
            ]
            should_complete = (
                all(row["status"] == "PASS" for row in thresholds)
                and all(value == 0 for value in counters.values())
            )
            if (
                thresholds != intent["threshold_results"]
                or counters != intent["side_effect_counters"]
                or sessions != intent["completed_sessions"]
                or (
                    intent["transition"] == "COMPLETE"
                    and not should_complete
                )
                or (
                    intent["transition"] == "FAIL"
                    and should_complete
                )
            ):
                _blocked("OPERATIONAL_EVALUATION")

    def transition(
        self,
        intent: Mapping[str, Any],
        *,
        crash_after: str | None = None,
    ) -> CanaryResult:
        validate_artifact(intent)
        strategy = str(intent["strategy_id"])
        identity = str(intent["intent_id"])
        paths = self._paths(strategy, identity)
        with self._writer.locked(paths["lock"]):
            current = self._writer.read_optional(paths["pointer"])
            observed = (
                EMPTY_SHA256 if current is None else current.byte_sha256
            )
            self._revalidate_intent(intent)
            intent_raw = canonical_resource_bytes(intent)
            self._writer.write_exact_once(paths["intent"], intent_raw)
            intent_ref = artifact_ref(
                intent,
                relative_path=str(paths["intent"]),
            )
            pointer = seal_semantic(
                {
                    "authority": dict(NO_AUTHORITY),
                    "cutoff": intent["cutoff"],
                    "intent_ref": intent_ref,
                    "pointer_id": f"canary-pointer-{identity}",
                    "protocol_version": PROTOCOL_VERSION,
                    "state": "PENDING_COMPLETION",
                    "strategy_id": strategy,
                    "updated_at": intent["created_at"],
                    "version": POINTER_VERSION,
                }
            )
            validate_artifact(pointer)
            pointer_raw = canonical_resource_bytes(pointer)
            proposed = _sha(pointer_raw)
            expected = str(intent["expected_pointer_sha256"])
            if crash_after == "intent":
                raise CanaryCrash("crash after canary intent")
            recovered = False
            if observed == expected:
                if intent["transition"] == "START" and current is not None:
                    _blocked("START_PRESTATE")
                if intent["transition"] != "START":
                    previous = self.resolve(strategy)
                    if previous.status != "CANARY_STARTED":
                        _blocked("COMPLETE_PRESTATE")
                self._writer.replace_cas(
                    paths["pointer"],
                    expected,
                    pointer_raw,
                )
            elif observed == proposed and current is not None:
                if current.data != pointer_raw:
                    _blocked("POINTER_HASH_COLLISION")
                recovered = True
            else:
                _blocked("POINTER_THIRD_STATE")
            if crash_after == "cas":
                raise CanaryCrash("crash after canary CAS")
            if self._writer.read(paths["pointer"], proposed) != pointer_raw:
                _blocked("POINTER_READBACK")
            if crash_after == "readback":
                raise CanaryCrash("crash after canary readback")
            pointer_ref = artifact_ref(
                pointer,
                relative_path=str(paths["pointer"]),
            )
            status = _STATUS_BY_TRANSITION[str(intent["transition"])]
            completion = seal_semantic(
                {
                    "authority": dict(PUBLICATION_AUTHORITY),
                    "cutoff": intent["cutoff"],
                    "evidence_refs": _ordered_refs(
                        [intent_ref, pointer_ref]
                    ),
                    "expected_pointer_sha256": expected,
                    "from_state": intent["from_state"],
                    "intent_ref": intent_ref,
                    "observed_pointer_sha256": expected,
                    "pointer_ref": pointer_ref,
                    "post_readback_sha256": proposed,
                    "proposed_pointer_sha256": proposed,
                    "protocol_version": PROTOCOL_VERSION,
                    "receipt_id": identity,
                    "recorded_at": intent["created_at"],
                    "status": status,
                    "strategy_id": strategy,
                    "to_state": intent["to_state"],
                    "version": COMPLETION_VERSION,
                }
            )
            validate_artifact(completion)
            self._writer.write_exact_once(
                paths["completion"],
                canonical_resource_bytes(completion),
            )
            if crash_after == "completion":
                raise CanaryCrash("crash after canary completion")
            state = self.resolve(strategy)
            if state.status != status:
                _blocked("COMPLETION_READBACK")
            return CanaryResult(
                status,
                intent_ref,
                pointer_ref,
                artifact_ref(
                    completion,
                    relative_path=str(paths["completion"]),
                ),
                recovered,
            )

    def resolve(self, strategy_id: str) -> CanaryState:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        root = self._root(strategy)
        stored = self._writer.read_optional(root / "_current.json")
        if stored is None:
            return CanaryState("DEFAULT_ELIGIBLE", None, None, None)
        pointer = load_canonical_resource(
            stored.data,
            label="canary pointer",
        )
        if type(pointer) is not dict:
            _blocked("POINTER_ROOT")
        validate_artifact(pointer)
        if pointer.get("strategy_id") != strategy:
            _blocked("POINTER_STRATEGY")
        intent = self._load_v4(
            pointer["intent_ref"],
            expected_version=INTENT_VERSION,
        )
        paths = self._paths(strategy, str(intent["intent_id"]))
        completion_stored = self._writer.read_optional(
            paths["completion"]
        )
        if completion_stored is None:
            return CanaryState(
                "PENDING_COMPLETION",
                intent,
                pointer,
                None,
            )
        completion = load_canonical_resource(
            completion_stored.data,
            label="canary completion",
        )
        if type(completion) is not dict:
            _blocked("COMPLETION_ROOT")
        validate_artifact(completion)
        intent_ref = artifact_ref(
            intent,
            relative_path=str(paths["intent"]),
        )
        pointer_ref = artifact_ref(
            pointer,
            relative_path=str(paths["pointer"]),
        )
        if (
            pointer["intent_ref"] != intent_ref
            or completion["intent_ref"] != intent_ref
            or completion["pointer_ref"] != pointer_ref
            or completion["post_readback_sha256"] != stored.byte_sha256
            or completion["proposed_pointer_sha256"]
            != stored.byte_sha256
            or completion["expected_pointer_sha256"]
            != intent["expected_pointer_sha256"]
            or completion["observed_pointer_sha256"]
            != intent["expected_pointer_sha256"]
            or completion["from_state"] != intent["from_state"]
            or completion["to_state"] != intent["to_state"]
            or completion["status"]
            != _STATUS_BY_TRANSITION[str(intent["transition"])]
        ):
            _blocked("COMPLETION_POINTER_BINDING")
        self._revalidate_intent(intent)
        return CanaryState(
            str(completion["status"]),
            intent,
            pointer,
            completion,
        )


__all__ = [
    "COMPARISON_VERSION",
    "COMPLETION_VERSION",
    "CanaryCrash",
    "CanaryError",
    "CanaryResult",
    "CanaryService",
    "CanaryState",
    "INTENT_VERSION",
    "POINTER_VERSION",
    "POLICY_VERSION",
    "build_canary_transition_intent",
    "build_dual_run_comparison",
    "build_historical_canary_policy",
    "evaluate_operational_canary",
]
