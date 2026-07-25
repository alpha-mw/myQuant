"""Offline, locally recomputed v17 shadow lifecycle orchestration.

The public lifecycle accepts only hash-bound source identities, raw Codex
research responses, and shadow target-weight proposals.  Fundamental ranking,
research adjustment, Quant timing, regime overlay, permissions, pretrade
costs, optimizer objectives, and terminal outcomes are all recomputed locally.
No provider, LLM, broker, order, or execution API is called here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    V17ContractError,
    parse_utc_timestamp,
    require_authority_false,
    require_bool,
    require_exact_keys,
    require_identifier,
    require_ratio,
    require_symbol,
)
from .latest import (
    V17LatestError,
    V17LatestPostCommitReadbackError,
    publish_terminal_latest,
    read_latest_pointer,
)
from .resources import (
    CANONICAL_POLICY_RESOURCE_NAMES,
    FROZEN_POLICY_RESOURCE_SHA256S,
    FROZEN_SCHEMA_SHA256S,
    assert_frozen_package_contracts,
)
from .runtime_pipeline import (
    DEEP_EVALUATION_VERSION,
    DEEP_REQUEST_VERSION,
    DETERMINISTIC_RESULT_VERSION,
    compute_finalization,
    compute_prepare_artifacts,
    evaluate_deep_response,
)
from .semantic import require_sha256, seal_semantic, validate_semantic_seal
from .source_bindings import (
    PORTFOLIO_REQUIRED_ROLES,
    SourceBindingBundle,
    load_source_manifest_binding,
)
from .state_machine import (
    EMPTY_SHA,
    TERMINAL_OUTPUT_VERSION,
    V17LedgerCASMismatch,
    V17PostCommitReadbackError,
    advance_run_state,
    advance_snapshot_drift_hard_stop,
    initialize_run,
    is_terminal_state,
    load_run_ledger,
)
from .storage import file_sha256, read_json

PREPARE_REQUEST_VERSION = "myquant.v17.shadow-prepare-request.v1"
DEEP_RESPONSE_VERSION = "myquant.v17.deep-research-response.v1"
FINALIZATION_VERSION = "myquant.v17.shadow-finalization.v1"
IMPORT_BINDING_VERSION = "myquant.v17.local-import-binding.v1"
STATUS_VERSION = "myquant.v17.shadow-status.v1"

IMPLEMENTATION_BINDING_RELATIVE_PATHS = (
    "quant_investor/factors/price_volume.py",
    "quant_investor/v17/__init__.py",
    "quant_investor/v17/cli.py",
    "quant_investor/v17/contracts.py",
    "quant_investor/v17/deep_research.py",
    "quant_investor/v17/forward_calibration.py",
    "quant_investor/v17/fundamental_scoring.py",
    "quant_investor/v17/holdings.py",
    "quant_investor/v17/latest.py",
    "quant_investor/v17/optimizer.py",
    "quant_investor/v17/permissions.py",
    "quant_investor/v17/pretrade.py",
    "quant_investor/v17/quant_timing.py",
    "quant_investor/v17/regime_overlay.py",
    "quant_investor/v17/resources.py",
    "quant_investor/v17/risk_policy.py",
    "quant_investor/v17/runtime.py",
    "quant_investor/v17/runtime_pipeline.py",
    "quant_investor/v17/semantic.py",
    "quant_investor/v17/source_bindings.py",
    "quant_investor/v17/source_maintain.py",
    "quant_investor/v17/state_machine.py",
    "quant_investor/v17/storage.py",
)

PREPARE_REQUEST_KEYS = frozenset(
    {
        "version",
        "run_id",
        "strategy_id",
        "market",
        "cutoff",
        "source_manifest_path",
        "source_manifest_sha256",
        "resource_sha256s",
        "schema_sha256s",
        "transition_times",
        "authority",
        "semantic_sha256",
    }
)
DETERMINISTIC_RESULT_KEYS = frozenset(
    {
        "version",
        "run_id",
        "cutoff",
        "ranked_symbols",
        "sealed_symbols",
        "appended_holdings",
        "portfolio_required_roles",
        "rows",
        "authority",
        "semantic_sha256",
    }
)
DETERMINISTIC_ROW_KEYS = frozenset(
    {
        "symbol",
        "industry",
        "fundamental_status",
        "fundamental_score",
        "score_decile",
        "base_q25_by_horizon",
        "base_eligible",
        "base_blockers",
        "fundamental_blockers",
        "selected_top24",
        "appended_holding",
    }
)
DEEP_REQUEST_KEYS = frozenset(
    {
        "version",
        "run_id",
        "cutoff",
        "symbols",
        "evidence_ids_by_symbol",
        "evidence_claims_by_symbol",
        "evidence_readiness_by_symbol",
        "authority",
        "semantic_sha256",
    }
)
DEEP_RESPONSE_KEYS = frozenset(
    {
        "version",
        "run_id",
        "cutoff",
        "review_results",
        "generated_at",
        "received_at",
        "authority",
        "semantic_sha256",
    }
)
DEEP_EVALUATION_KEYS = frozenset(
    {
        "version",
        "run_id",
        "cutoff",
        "evaluations",
        "received_at",
        "authority",
        "semantic_sha256",
    }
)
FINALIZATION_KEYS = frozenset(
    {
        "version",
        "run_id",
        "cutoff",
        "candidate_proposals",
        "generated_at",
        "finalized_at",
        "authority",
        "semantic_sha256",
    }
)

_PROHIBITED_SIDE_EFFECT_KEYS = frozenset(
    {
        "broker_order_id",
        "execution_id",
        "live_order",
        "orders_submitted",
        "trades_executed",
        "execution_authority",
        "trade_authority",
    }
)


class V17RuntimeError(V17ContractError):
    """A local shadow lifecycle request is invalid or cannot be completed."""


class V17RuntimeSnapshotDrift(V17RuntimeError):
    """A previously sealed source or lifecycle artifact no longer matches."""


class V17RuntimeArtifactDrift(V17RuntimeSnapshotDrift):
    """An immutable artifact drifted, so a new transition cannot be trusted."""

    def __init__(
        self,
        message: str,
        *,
        role: str,
        expected_hash: str,
        observed_hash: str,
    ) -> None:
        super().__init__(message)
        self.role = role
        self.expected_hash = expected_hash
        self.observed_hash = observed_hash

    @property
    def blocker(self) -> str:
        return (
            f"artifact_drift:role={self.role}:expected={self.expected_hash}"
            f":observed={self.observed_hash}"
        )


class V17RuntimeInvalidEvidence(V17RuntimeError):
    """Caller-supplied response or finalization evidence is invalid."""


def _validate_string_list(
    value: Any,
    *,
    label: str,
    symbols: bool,
    require_sorted: bool = False,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V17RuntimeError(f"{label} must be an array")
    if not value and not allow_empty:
        raise V17RuntimeError(f"{label} must be a nonempty array")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(
            require_symbol(item, label=f"{label}[{index}]")
            if symbols
            else require_identifier(item, label=f"{label}[{index}]")
        )
    if len(normalized) != len(set(normalized)):
        raise V17RuntimeError(f"{label} must be unique")
    if require_sorted and normalized != sorted(normalized):
        raise V17RuntimeError(f"{label} must be canonically sorted")
    return tuple(normalized)


def _validate_blockers(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise V17RuntimeError(f"{label} must be an array")
    if len(value) != len(set(value)) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise V17RuntimeError(f"{label} must contain unique canonical strings")
    return list(value)


def _validate_deterministic_result(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    cutoff: str,
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, DETERMINISTIC_RESULT_KEYS, label="deterministic result")
    if sealed.get("version") != DETERMINISTIC_RESULT_VERSION:
        raise V17RuntimeError("deterministic result version mismatch")
    if sealed.get("run_id") != run_id or sealed.get("cutoff") != cutoff:
        raise V17RuntimeError("deterministic result run/cutoff binding mismatch")
    require_authority_false(sealed.get("authority"))
    ranked = _validate_string_list(
        sealed.get("ranked_symbols"), label="ranked_symbols", symbols=True
    )
    universe = _validate_string_list(
        sealed.get("sealed_symbols"), label="sealed_symbols", symbols=True
    )
    appended = _validate_string_list(
        sealed.get("appended_holdings"),
        label="appended_holdings",
        symbols=True,
        allow_empty=True,
    )
    if universe != ranked + appended or set(ranked).intersection(appended):
        raise V17RuntimeError("sealed universe order/binding mismatch")
    roles = _validate_string_list(
        sealed.get("portfolio_required_roles"),
        label="portfolio_required_roles",
        symbols=False,
        require_sorted=True,
    )
    if frozenset(roles) != PORTFOLIO_REQUIRED_ROLES:
        raise V17RuntimeError("portfolio required-role set mismatch")
    rows = sealed.get("rows")
    if not isinstance(rows, list) or len(rows) != len(universe):
        raise V17RuntimeError("deterministic rows must cover sealed universe exactly")
    seen: list[str] = []
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise V17RuntimeError(f"deterministic rows[{index}] must be an object")
        require_exact_keys(item, DETERMINISTIC_ROW_KEYS, label=f"deterministic rows[{index}]")
        symbol = require_symbol(item.get("symbol"), label=f"rows[{index}].symbol")
        seen.append(symbol)
        if item.get("fundamental_status") not in {"AVAILABLE", "UNAVAILABLE"}:
            raise V17RuntimeError("deterministic Fundamental status invalid")
        if item.get("fundamental_score") is not None:
            require_ratio(item.get("fundamental_score"), label="fundamental_score")
        decile = item.get("score_decile")
        if decile is not None and (
            isinstance(decile, bool) or not isinstance(decile, int) or decile not in range(1, 11)
        ):
            raise V17RuntimeError("Fundamental score decile invalid")
        q25 = item.get("base_q25_by_horizon")
        if not isinstance(q25, Mapping) or any(key not in {"120", "252", "378"} for key in q25):
            raise V17RuntimeError("base q25 horizon map invalid")
        for horizon, value in q25.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise V17RuntimeError(f"base q25 invalid: {horizon}")
        if not isinstance(item.get("base_eligible"), bool):
            raise V17RuntimeError("base_eligible must be strict bool")
        _validate_blockers(item.get("base_blockers"), label="base_blockers")
        _validate_blockers(item.get("fundamental_blockers"), label="fundamental_blockers")
        if not isinstance(item.get("selected_top24"), bool) or not isinstance(
            item.get("appended_holding"), bool
        ):
            raise V17RuntimeError("deterministic membership flags must be bool")
    if tuple(seen) != universe:
        raise V17RuntimeError("deterministic rows reordered the sealed universe")
    return sealed


def _validate_deep_request(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    cutoff: str,
    sealed_symbols: Sequence[str],
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, DEEP_REQUEST_KEYS, label="deep-research request")
    if sealed.get("version") != DEEP_REQUEST_VERSION:
        raise V17RuntimeError("deep-research request version mismatch")
    if sealed.get("run_id") != run_id or sealed.get("cutoff") != cutoff:
        raise V17RuntimeError("deep-research request run/cutoff binding mismatch")
    require_authority_false(sealed.get("authority"))
    symbols = _validate_string_list(
        sealed.get("symbols"), label="deep request symbols", symbols=True
    )
    if tuple(symbols) != tuple(sealed_symbols):
        raise V17RuntimeError("deep request must cover the sealed universe exactly")
    evidence = sealed.get("evidence_ids_by_symbol")
    if not isinstance(evidence, Mapping) or set(evidence) != set(symbols):
        raise V17RuntimeError("deep request evidence map must exactly cover symbols")
    claims_by_symbol = sealed.get("evidence_claims_by_symbol")
    if not isinstance(claims_by_symbol, Mapping) or set(claims_by_symbol) != set(symbols):
        raise V17RuntimeError("deep request claims map must exactly cover symbols")
    readiness = sealed.get("evidence_readiness_by_symbol")
    if not isinstance(readiness, Mapping) or set(readiness) != set(symbols):
        raise V17RuntimeError("deep request readiness map must exactly cover symbols")
    for symbol in symbols:
        evidence_ids = _validate_string_list(
            evidence[symbol],
            label=f"evidence_ids_by_symbol.{symbol}",
            symbols=False,
            require_sorted=True,
            allow_empty=True,
        )
        claims = claims_by_symbol[symbol]
        if not isinstance(claims, Mapping) or set(claims) != set(evidence_ids):
            raise V17RuntimeError(
                f"deep request claims must exactly cover evidence IDs for {symbol}"
            )
        for evidence_id in evidence_ids:
            claim = claims[evidence_id]
            if not isinstance(claim, Mapping):
                raise V17RuntimeError(f"deep request claim must be an object: {evidence_id}")
            require_exact_keys(
                claim,
                frozenset({"kind", "layers", "coverage", "signals", "red_flags"}),
                label=f"evidence_claims_by_symbol.{symbol}.{evidence_id}",
            )
            require_identifier(claim.get("kind"), label=f"evidence claim kind {evidence_id}")
            for category in ("layers", "coverage", "signals", "red_flags"):
                _validate_string_list(
                    claim.get(category),
                    label=f"evidence claim {evidence_id}.{category}",
                    symbols=False,
                    require_sorted=True,
                    allow_empty=True,
                )
        require_bool(
            readiness[symbol],
            label=f"evidence_readiness_by_symbol.{symbol}",
        )
    return sealed


def validate_prepare_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, PREPARE_REQUEST_KEYS, label="shadow prepare request")
    if sealed.get("version") != PREPARE_REQUEST_VERSION:
        raise V17RuntimeError("shadow prepare request version mismatch")
    require_identifier(sealed.get("run_id"), label="run_id")
    require_identifier(sealed.get("strategy_id"), label="strategy_id")
    if sealed.get("market") != "CN":
        raise V17RuntimeError("shadow prepare market must be CN")
    cutoff = str(sealed.get("cutoff") or "")
    parse_utc_timestamp(cutoff, label="cutoff")
    require_authority_false(sealed.get("authority"))
    path = Path(str(sealed.get("source_manifest_path") or ""))
    if path.is_absolute() or ".." in path.parts:
        raise V17RuntimeError("source manifest path must be repository-relative")
    require_sha256(sealed.get("source_manifest_sha256"), label="source manifest SHA-256")
    resource_shas = sealed.get("resource_sha256s")
    if not isinstance(resource_shas, Mapping) or set(resource_shas) != set(
        CANONICAL_POLICY_RESOURCE_NAMES
    ):
        raise V17RuntimeError("canonical resource SHA set mismatch")
    for name in CANONICAL_POLICY_RESOURCE_NAMES:
        require_sha256(resource_shas.get(name), label=f"resource SHA for {name}")
    schema_shas = sealed.get("schema_sha256s")
    if not isinstance(schema_shas, Mapping) or set(schema_shas) != set(FROZEN_SCHEMA_SHA256S):
        raise V17RuntimeError("canonical schema SHA set mismatch")
    for name in FROZEN_SCHEMA_SHA256S:
        require_sha256(schema_shas.get(name), label=f"schema SHA for {name}")
    times = sealed.get("transition_times")
    if not isinstance(times, Mapping):
        raise V17RuntimeError("transition_times must be an object")
    require_exact_keys(
        times,
        frozenset({"prepared_at", "deterministic_at", "deep_request_at"}),
        label="transition_times",
    )
    parsed_times = [
        parse_utc_timestamp(times[key], label=f"transition_times.{key}")
        for key in ("prepared_at", "deterministic_at", "deep_request_at")
    ]
    if parsed_times != sorted(parsed_times):
        raise V17RuntimeError("prepare transition times must not regress")
    return sealed


def _validate_package_bindings(
    resource_shas: Mapping[str, Any],
    schema_shas: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if dict(resource_shas) != FROZEN_POLICY_RESOURCE_SHA256S:
        raise V17RuntimeError("request policy resource manifest is not the frozen package")
    if dict(schema_shas) != FROZEN_SCHEMA_SHA256S:
        raise V17RuntimeError("request schema manifest is not the frozen package")
    return assert_frozen_package_contracts()


def _revalidate_ledger_package_bindings(ledger: Mapping[str, Any]) -> None:
    try:
        bindings = ledger.get("input_bindings")
        if not isinstance(bindings, Mapping):
            raise V17RuntimeSnapshotDrift("ledger input bindings unavailable")
        resources = bindings.get("resource_sha256s")
        schemas = bindings.get("schema_sha256s")
        if not isinstance(resources, Mapping) or not isinstance(schemas, Mapping):
            raise V17RuntimeSnapshotDrift("ledger package bindings unavailable")
        package = _validate_package_bindings(resources, schemas)
        if dict(resources) != dict(package["resources"]) or dict(schemas) != dict(
            package["schemas"]
        ):
            raise V17RuntimeSnapshotDrift("ledger package binding readback drift")
    except V17RuntimeSnapshotDrift:
        raise
    except Exception as exc:
        raise V17RuntimeSnapshotDrift("frozen package resource/schema drift") from exc


def _compute_implementation_sha256s(repo_root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    try:
        for relative in IMPLEMENTATION_BINDING_RELATIVE_PATHS:
            path = repo_root / relative
            before = file_sha256(path)
            if file_sha256(path) != before:
                raise V17RuntimeSnapshotDrift(f"implementation changed during read: {relative}")
            manifest[relative] = before
    except V17RuntimeSnapshotDrift:
        raise
    except Exception as exc:
        raise V17RuntimeSnapshotDrift("implementation source unavailable") from exc
    return manifest


def _revalidate_ledger_implementation_bindings(
    repo_root: Path,
    ledger: Mapping[str, Any],
) -> None:
    bindings = ledger.get("input_bindings")
    expected = bindings.get("implementation_sha256s") if isinstance(bindings, Mapping) else None
    if not isinstance(expected, Mapping) or set(expected) != set(
        IMPLEMENTATION_BINDING_RELATIVE_PATHS
    ):
        raise V17RuntimeSnapshotDrift("ledger implementation binding set drift")
    for relative, digest in expected.items():
        require_sha256(digest, label=f"implementation SHA-256 for {relative}")
    observed = _compute_implementation_sha256s(repo_root)
    if dict(expected) != observed:
        raise V17RuntimeSnapshotDrift("implementation source SHA drift")


def _prepare_input_bindings(
    *,
    repo_root: Path,
    request_sha256: str,
    bundle: SourceBindingBundle,
    package: Mapping[str, Mapping[str, str]],
    implementation_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "prepare_request_byte_sha256": request_sha256,
        "source_manifest_path": bundle.manifest_path.relative_to(repo_root).as_posix(),
        "source_manifest_sha256": bundle.manifest_byte_sha256,
        "source_availability": dict(sorted(bundle.effective_availability_by_role.items())),
        "resource_sha256s": dict(package["resources"]),
        "schema_sha256s": dict(package["schemas"]),
        "implementation_sha256s": dict(implementation_sha256s),
    }


def _validate_prepare_resume_binding(
    ledger: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    input_bindings: Mapping[str, Any],
) -> None:
    if (
        ledger.get("run_id") != request.get("run_id")
        or ledger.get("strategy_id") != request.get("strategy_id")
        or ledger.get("market") != request.get("market")
        or ledger.get("cutoff") != request.get("cutoff")
        or ledger.get("created_at") != request["transition_times"]["prepared_at"]
    ):
        raise V17RuntimeInvalidEvidence("prepare retry run identity binding mismatch")
    if ledger.get("input_bindings") != input_bindings:
        raise V17RuntimeInvalidEvidence("prepare retry input binding mismatch")

    expected_history = {
        "PREPARED": (0, (request["transition_times"]["prepared_at"],)),
        "DETERMINISTIC_COMPLETE": (
            1,
            (
                request["transition_times"]["prepared_at"],
                request["transition_times"]["deterministic_at"],
            ),
        ),
        "DEEP_REQUEST_READY": (
            2,
            (
                request["transition_times"]["prepared_at"],
                request["transition_times"]["deterministic_at"],
                request["transition_times"]["deep_request_at"],
            ),
        ),
    }
    state = str(ledger.get("state"))
    if state not in expected_history:
        raise V17RuntimeInvalidEvidence(
            "prepare retry requires PREPARED, DETERMINISTIC_COMPLETE, or DEEP_REQUEST_READY"
        )
    sequence, timestamps = expected_history[state]
    history = ledger.get("history")
    if (
        ledger.get("sequence") != sequence
        or not isinstance(history, list)
        or tuple(item.get("at") for item in history) != timestamps
    ):
        raise V17RuntimeInvalidEvidence("prepare retry transition binding mismatch")
    expected_roles = {
        "PREPARED": frozenset(),
        "DETERMINISTIC_COMPLETE": frozenset({"deterministic_result"}),
        "DEEP_REQUEST_READY": frozenset({"deterministic_result", "deep_request"}),
    }[state]
    artifacts = ledger.get("artifacts")
    if not isinstance(artifacts, Mapping) or frozenset(artifacts) != expected_roles:
        raise V17RuntimeSnapshotDrift("prepare retry artifact-role binding drift")


def _require_recomputed_artifact(
    repo_root: Path,
    ledger: Mapping[str, Any],
    *,
    role: str,
    recomputed: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _artifact_payload(repo_root, ledger, role)
    if observed != dict(recomputed):
        raise V17RuntimeSnapshotDrift(f"recomputed artifact binding drift: {role}")
    return observed


def prepare_shadow_run(
    repo_root: str | Path,
    request: Mapping[str, Any],
    *,
    request_byte_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    """Prepare a run by locally recomputing Fundamental artifacts."""

    validated = validate_prepare_request(request)
    request_sha = require_sha256(request_byte_sha256, label="prepare request byte SHA-256")
    root = Path(repo_root).absolute()
    resumed_ledger: dict[str, Any] | None = None
    resumed_sha: str | None = None
    if expected_ledger_sha256 != EMPTY_SHA:
        expected = require_sha256(
            expected_ledger_sha256,
            label="expected ledger SHA-256",
        )
        try:
            resumed_ledger, resumed_sha = load_run_ledger(
                root,
                validated["run_id"],
                verify_artifacts=False,
            )
        except Exception as exc:
            raise V17RuntimeSnapshotDrift("prepare retry ledger readback drift") from exc
        if resumed_sha != expected:
            raise V17LedgerCASMismatch("prepare retry ledger CAS mismatch; zero writes")
    package = _validate_package_bindings(
        validated["resource_sha256s"],
        validated["schema_sha256s"],
    )
    try:
        bundle = load_source_manifest_binding(
            root,
            validated["source_manifest_path"],
            expected_manifest_sha256=validated["source_manifest_sha256"],
            validate_authority_payloads=True,
        )
    except Exception as exc:
        if resumed_ledger is not None:
            raise V17RuntimeSnapshotDrift("prepare retry source snapshot drift") from exc
        raise
    if bundle.rank_unavailable_roles:
        raise V17RuntimeError(
            "rank authority unavailable: " + ",".join(bundle.rank_unavailable_roles)
        )
    implementation_sha256s = _compute_implementation_sha256s(root)
    deterministic, deep_request = compute_prepare_artifacts(
        bundle,
        run_id=validated["run_id"],
        strategy_id=validated["strategy_id"],
        cutoff=validated["cutoff"],
    )
    deterministic = _validate_deterministic_result(
        deterministic,
        run_id=validated["run_id"],
        cutoff=validated["cutoff"],
    )
    deep_request = _validate_deep_request(
        deep_request,
        run_id=validated["run_id"],
        cutoff=validated["cutoff"],
        sealed_symbols=deterministic["sealed_symbols"],
    )
    if _compute_implementation_sha256s(root) != implementation_sha256s:
        raise V17RuntimeSnapshotDrift("implementation changed during prepare computation")
    input_bindings = _prepare_input_bindings(
        repo_root=root,
        request_sha256=request_sha,
        bundle=bundle,
        package=package,
        implementation_sha256s=implementation_sha256s,
    )
    times = validated["transition_times"]
    if resumed_ledger is None:
        ledger, ledger_sha = initialize_run(
            root,
            run_id=validated["run_id"],
            strategy_id=validated["strategy_id"],
            cutoff=validated["cutoff"],
            prepared_at=times["prepared_at"],
            input_bindings=input_bindings,
            expected_ledger_sha256=expected_ledger_sha256,
        )
    else:
        ledger = resumed_ledger
        ledger_sha = str(resumed_sha)
        _validate_prepare_resume_binding(
            ledger,
            request=validated,
            input_bindings=input_bindings,
        )

    if ledger["state"] == "PREPARED":
        ledger, ledger_sha = advance_run_state(
            root,
            run_id=validated["run_id"],
            expected_ledger_sha256=ledger_sha,
            next_state="DETERMINISTIC_COMPLETE",
            transitioned_at=times["deterministic_at"],
            artifacts={"deterministic_result": deterministic},
        )
    else:
        _require_recomputed_artifact(
            root,
            ledger,
            role="deterministic_result",
            recomputed=deterministic,
        )

    if ledger["state"] == "DETERMINISTIC_COMPLETE":
        ledger, ledger_sha = advance_run_state(
            root,
            run_id=validated["run_id"],
            expected_ledger_sha256=ledger_sha,
            next_state="DEEP_REQUEST_READY",
            transitioned_at=times["deep_request_at"],
            artifacts={"deep_request": deep_request},
        )
    else:
        _require_recomputed_artifact(
            root,
            ledger,
            role="deep_request",
            recomputed=deep_request,
        )
    return {
        "run_id": validated["run_id"],
        "state": ledger["state"],
        "ledger_sha256": ledger_sha,
        "portfolio_unavailable_roles": list(bundle.portfolio_unavailable_roles),
        "authority": False,
    }


def prepare_shadow_run_from_file(
    repo_root: str | Path,
    *,
    request_path: str | Path,
    expected_request_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    expected = require_sha256(expected_request_sha256, label="expected request SHA-256")
    before = file_sha256(request_path)
    if before != expected:
        raise V17RuntimeError("prepare request byte SHA mismatch")
    payload = read_json(request_path)
    if file_sha256(request_path) != before:
        raise V17RuntimeError("prepare request changed during import")
    return prepare_shadow_run(
        repo_root,
        payload,
        request_byte_sha256=before,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def _artifact_payload(
    repo_root: Path,
    ledger: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    artifacts = ledger.get("artifacts")
    binding = artifacts.get(role) if isinstance(artifacts, Mapping) else None
    if not isinstance(binding, Mapping):
        raise V17RuntimeSnapshotDrift(f"ledger artifact binding missing: {role}")
    expected_byte = str(binding.get("byte_sha256") or "INVALID")
    expected_semantic = str(binding.get("semantic_sha256") or "INVALID")
    path = repo_root / Path(str(binding.get("relative_path") or ""))
    try:
        observed_byte = file_sha256(path)
    except Exception as exc:
        raise V17RuntimeArtifactDrift(
            f"ledger artifact unreadable: {role}",
            role=role,
            expected_hash=expected_byte,
            observed_hash="UNREADABLE",
        ) from exc
    if observed_byte != expected_byte:
        raise V17RuntimeArtifactDrift(
            f"ledger artifact byte drift: {role}",
            role=role,
            expected_hash=expected_byte,
            observed_hash=observed_byte,
        )
    try:
        payload = validate_semantic_seal(read_json(path))
    except Exception as exc:
        raise V17RuntimeArtifactDrift(
            f"ledger artifact semantic payload unreadable: {role}",
            role=role,
            expected_hash=expected_semantic,
            observed_hash="UNREADABLE",
        ) from exc
    observed_semantic = str(payload.get("semantic_sha256") or "INVALID")
    if observed_semantic != expected_semantic:
        raise V17RuntimeArtifactDrift(
            f"ledger artifact semantic drift: {role}",
            role=role,
            expected_hash=expected_semantic,
            observed_hash=observed_semantic,
        )
    return payload


def _recompute_prepare_artifacts_from_sources(
    repo_root: Path,
    ledger: Mapping[str, Any],
    bundle: SourceBindingBundle,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        deterministic, deep_request = compute_prepare_artifacts(
            bundle,
            run_id=str(ledger["run_id"]),
            strategy_id=str(ledger["strategy_id"]),
            cutoff=str(ledger["cutoff"]),
        )
        deterministic = _validate_deterministic_result(
            deterministic,
            run_id=str(ledger["run_id"]),
            cutoff=str(ledger["cutoff"]),
        )
        deep_request = _validate_deep_request(
            deep_request,
            run_id=str(ledger["run_id"]),
            cutoff=str(ledger["cutoff"]),
            sealed_symbols=deterministic["sealed_symbols"],
        )
        _require_recomputed_artifact(
            repo_root,
            ledger,
            role="deterministic_result",
            recomputed=deterministic,
        )
        _require_recomputed_artifact(
            repo_root,
            ledger,
            role="deep_request",
            recomputed=deep_request,
        )
        return deterministic, deep_request
    except V17RuntimeSnapshotDrift:
        raise
    except Exception as exc:
        raise V17RuntimeSnapshotDrift(
            "bound deterministic/deep request recomputation failed"
        ) from exc


def _validate_stored_import_binding(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    role: str,
    imported_payload: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(
        sealed,
        frozenset(
            {
                "version",
                "run_id",
                "role",
                "source_byte_sha256",
                "payload_semantic_sha256",
                "authority",
                "semantic_sha256",
            }
        ),
        label=f"stored {role} import binding",
    )
    if (
        sealed.get("version") != IMPORT_BINDING_VERSION
        or sealed.get("run_id") != run_id
        or sealed.get("role") != role
        or sealed.get("payload_semantic_sha256") != imported_payload.get("semantic_sha256")
    ):
        raise V17RuntimeSnapshotDrift(f"stored {role} import binding drift")
    require_sha256(
        sealed.get("source_byte_sha256"),
        label=f"stored {role} source byte SHA-256",
    )
    require_authority_false(sealed.get("authority"))
    return sealed


def _recompute_stored_deep_evaluation(
    repo_root: Path,
    ledger: Mapping[str, Any],
    *,
    deterministic: Mapping[str, Any],
    deep_request: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        response = _validate_deep_response(
            _artifact_payload(repo_root, ledger, "deep_response"),
            ledger=ledger,
            deep_request=deep_request,
        )
        _validate_stored_import_binding(
            _artifact_payload(repo_root, ledger, "deep_response_import"),
            run_id=str(ledger["run_id"]),
            role="deep_response",
            imported_payload=response,
        )
        evaluation = evaluate_deep_response(
            response,
            deterministic=deterministic,
            deep_request=deep_request,
        )
        evaluation = _validate_deep_evaluation(
            evaluation,
            ledger=ledger,
            symbols=deterministic["sealed_symbols"],
        )
        _require_recomputed_artifact(
            repo_root,
            ledger,
            role="deep_evaluation",
            recomputed=evaluation,
        )
        return evaluation
    except (V17RuntimeInvalidEvidence, V17RuntimeSnapshotDrift):
        raise
    except Exception as exc:
        raise V17RuntimeSnapshotDrift(
            "stored deep response/evaluation recomputation failed"
        ) from exc


def _validate_deep_response(
    payload: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
    deep_request: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, DEEP_RESPONSE_KEYS, label="deep-research response")
    if sealed.get("version") != DEEP_RESPONSE_VERSION:
        raise V17RuntimeError("deep-research response version mismatch")
    if sealed.get("run_id") != ledger.get("run_id") or sealed.get("cutoff") != ledger.get("cutoff"):
        raise V17RuntimeError("deep-research response run/cutoff mismatch")
    require_authority_false(sealed.get("authority"))
    generated = parse_utc_timestamp(sealed.get("generated_at"), label="generated_at")
    received = parse_utc_timestamp(sealed.get("received_at"), label="received_at")
    request_ready_entries = [
        item
        for item in ledger.get("history", [])
        if isinstance(item, Mapping) and item.get("to_state") == "DEEP_REQUEST_READY"
    ]
    if len(request_ready_entries) != 1:
        raise V17RuntimeError("ledger must contain one DEEP_REQUEST_READY transition")
    request_ready_at = parse_utc_timestamp(
        request_ready_entries[0].get("at"),
        label="DEEP_REQUEST_READY history.at",
    )
    if generated < request_ready_at:
        raise V17RuntimeError("deep response generated_at precedes DEEP_REQUEST_READY")
    if received < generated:
        raise V17RuntimeError("deep response received_at precedes generated_at")
    results = sealed.get("review_results")
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise V17RuntimeError("review_results must be an array")
    requested = tuple(deep_request["symbols"])
    seen: list[str] = []
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise V17RuntimeError(f"review_results[{index}] must be an object")
        status = item.get("status")
        if status == "COMPLETE":
            require_exact_keys(
                item,
                frozenset({"symbol", "status", "research"}),
                label=f"review_results[{index}]",
            )
            if not isinstance(item.get("research"), Mapping):
                raise V17RuntimeError("completed review requires research object")
        elif status == "UNAVAILABLE":
            require_exact_keys(
                item,
                frozenset({"symbol", "status", "reason"}),
                label=f"review_results[{index}]",
            )
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
                raise V17RuntimeError("unavailable review requires canonical reason")
        else:
            raise V17RuntimeError("deep review status invalid")
        seen.append(require_symbol(item.get("symbol"), label="review symbol"))
    if tuple(seen) != requested:
        raise V17RuntimeError(
            "deep response must explicitly cover the sealed request without reordering"
        )
    return sealed


def _validate_deep_evaluation(
    payload: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
    symbols: Sequence[str],
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, DEEP_EVALUATION_KEYS, label="deep evaluation")
    if sealed.get("version") != DEEP_EVALUATION_VERSION:
        raise V17RuntimeError("deep evaluation version mismatch")
    if sealed.get("run_id") != ledger["run_id"] or sealed.get("cutoff") != ledger["cutoff"]:
        raise V17RuntimeError("deep evaluation identity mismatch")
    require_authority_false(sealed.get("authority"))
    evaluations = sealed.get("evaluations")
    if not isinstance(evaluations, list) or [item.get("symbol") for item in evaluations] != list(
        symbols
    ):
        raise V17RuntimeError("deep evaluation coverage/order mismatch")
    invalid_symbols = [
        str(item.get("symbol"))
        for item in evaluations
        if item.get("status") == "DEEP_RESEARCH_INVALID"
    ]
    if invalid_symbols:
        raise V17RuntimeInvalidEvidence(
            "deep research evaluation invalid for sealed symbols: " + ",".join(invalid_symbols)
        )
    return sealed


def _import_binding(
    *,
    run_id: str,
    role: str,
    source_byte_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return seal_semantic(
        {
            "version": IMPORT_BINDING_VERSION,
            "run_id": run_id,
            "role": role,
            "source_byte_sha256": require_sha256(
                source_byte_sha256, label=f"{role} source byte SHA-256"
            ),
            "payload_semantic_sha256": require_sha256(
                payload.get("semantic_sha256"), label=f"{role} semantic SHA-256"
            ),
            "authority": False,
        }
    )


def receive_shadow_response(
    repo_root: str | Path,
    *,
    run_id: str,
    response: Mapping[str, Any],
    response_byte_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    root = Path(repo_root).absolute()
    try:
        ledger, observed = load_run_ledger(root, run_id, verify_artifacts=False)
    except Exception as exc:
        raise V17RuntimeSnapshotDrift("response import ledger snapshot drift") from exc
    if observed != require_sha256(expected_ledger_sha256, label="expected ledger SHA-256"):
        raise V17LedgerCASMismatch("ledger CAS mismatch; response import performed zero writes")
    if ledger["state"] not in {"DEEP_REQUEST_READY", "DEEP_RESPONSE_RECEIVED"}:
        raise V17RuntimeInvalidEvidence(
            "deep response requires DEEP_REQUEST_READY or resumable DEEP_RESPONSE_RECEIVED"
        )
    _revalidate_ledger_package_bindings(ledger)
    _revalidate_ledger_implementation_bindings(root, ledger)
    bundle = _revalidate_sources(root, ledger)
    deterministic, deep_request = _recompute_prepare_artifacts_from_sources(
        root,
        ledger,
        bundle,
    )
    try:
        validated = _validate_deep_response(response, ledger=ledger, deep_request=deep_request)
        evaluation = evaluate_deep_response(
            validated,
            deterministic=deterministic,
            deep_request=deep_request,
        )
        evaluation = _validate_deep_evaluation(
            evaluation,
            ledger=ledger,
            symbols=deterministic["sealed_symbols"],
        )
        import_binding = _import_binding(
            run_id=run_id,
            role="deep_response",
            source_byte_sha256=response_byte_sha256,
            payload=validated,
        )
    except V17RuntimeSnapshotDrift:
        raise
    except Exception as exc:
        raise V17RuntimeInvalidEvidence("deep response evidence is invalid") from exc
    artifacts = {
        "deep_response": validated,
        "deep_evaluation": evaluation,
        "deep_response_import": import_binding,
    }
    if ledger["state"] == "DEEP_RESPONSE_RECEIVED":
        expected_roles = frozenset(
            {
                "deterministic_result",
                "deep_request",
                "deep_response",
                "deep_evaluation",
                "deep_response_import",
            }
        )
        if frozenset(ledger.get("artifacts", {})) != expected_roles:
            raise V17RuntimeSnapshotDrift("response retry artifact-role binding drift")
        for role, payload in artifacts.items():
            _require_recomputed_artifact(
                root,
                ledger,
                role=role,
                recomputed=payload,
            )
        return {
            "run_id": run_id,
            "state": ledger["state"],
            "ledger_sha256": observed,
            "authority": False,
        }
    ledger, ledger_sha = advance_run_state(
        root,
        run_id=run_id,
        expected_ledger_sha256=observed,
        next_state="DEEP_RESPONSE_RECEIVED",
        transitioned_at=validated["received_at"],
        artifacts=artifacts,
    )
    return {
        "run_id": run_id,
        "state": ledger["state"],
        "ledger_sha256": ledger_sha,
        "authority": False,
    }


def _validate_no_side_effect_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _PROHIBITED_SIDE_EFFECT_KEYS:
                raise V17RuntimeError(f"prohibited side-effect field at {path}.{key}")
            if key == "authority" and item is not False:
                raise V17RuntimeError(f"non-shadow authority at {path}.{key}")
            _validate_no_side_effect_fields(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_no_side_effect_fields(item, path=f"{path}[{index}]")


def _validate_finalization(
    payload: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
    sealed_symbols: Sequence[str],
) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    require_exact_keys(sealed, FINALIZATION_KEYS, label="shadow finalization")
    if sealed.get("version") != FINALIZATION_VERSION:
        raise V17RuntimeError("shadow finalization version mismatch")
    if sealed.get("run_id") != ledger.get("run_id") or sealed.get("cutoff") != ledger.get("cutoff"):
        raise V17RuntimeError("shadow finalization run/cutoff mismatch")
    require_authority_false(sealed.get("authority"))
    generated = parse_utc_timestamp(sealed.get("generated_at"), label="generated_at")
    finalized = parse_utc_timestamp(sealed.get("finalized_at"), label="finalized_at")
    if finalized < generated:
        raise V17RuntimeError("finalized_at precedes generated_at")
    proposals = sealed.get("candidate_proposals")
    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        raise V17RuntimeError("candidate_proposals must be an array")
    ids: list[str] = []
    sealed_set = set(sealed_symbols)
    for index, item in enumerate(proposals):
        if not isinstance(item, Mapping):
            raise V17RuntimeError(f"candidate_proposals[{index}] must be an object")
        require_exact_keys(
            item,
            frozenset({"candidate_id", "target_weights"}),
            label=f"candidate_proposals[{index}]",
        )
        ids.append(require_identifier(item.get("candidate_id"), label="candidate_id"))
        weights = item.get("target_weights")
        if not isinstance(weights, Mapping):
            raise V17RuntimeError("target_weights must be an object")
        for symbol, weight in weights.items():
            canonical = require_symbol(symbol, label="target weight symbol")
            if canonical not in sealed_set:
                raise V17RuntimeError(f"candidate expanded sealed universe: {canonical}")
            require_ratio(weight, label=f"target_weights.{canonical}")
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise V17RuntimeError("candidate proposals must be unique and sorted by candidate_id")
    _validate_no_side_effect_fields(sealed)
    return sealed


def _terminal_output(
    *,
    ledger: Mapping[str, Any],
    predecessor_sha256: str,
    state: str,
    generated_at: str,
    rank_output: Mapping[str, Any] | None,
    portfolio_output: Mapping[str, Any] | None,
    blockers: Sequence[str],
) -> dict[str, Any]:
    source_sha = require_sha256(
        ledger["input_bindings"]["source_manifest_sha256"],
        label="source manifest SHA-256",
    )
    output = seal_semantic(
        {
            "version": TERMINAL_OUTPUT_VERSION,
            "run_id": ledger["run_id"],
            "strategy_id": ledger["strategy_id"],
            "market": "CN",
            "cutoff": ledger["cutoff"],
            "terminal_state": state,
            "rank_output": dict(rank_output) if rank_output is not None else None,
            "portfolio_output": dict(portfolio_output) if portfolio_output is not None else None,
            "blockers": list(dict.fromkeys(blockers)),
            "source_manifest_sha256": source_sha,
            "ledger_predecessor_sha256": predecessor_sha256,
            "generated_at": generated_at,
            "authority": False,
        }
    )
    _validate_terminal_source_binding(output, ledger=ledger)
    return output


def _validate_terminal_source_binding(
    payload: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
) -> None:
    try:
        bindings = ledger.get("input_bindings")
        if not isinstance(bindings, Mapping):
            raise V17RuntimeSnapshotDrift("ledger input bindings unavailable")
        expected = require_sha256(
            bindings.get("source_manifest_sha256"),
            label="ledger source manifest SHA-256",
        )
        observed = require_sha256(
            payload.get("source_manifest_sha256"),
            label="terminal source manifest SHA-256",
        )
        if observed != expected:
            raise V17RuntimeSnapshotDrift(
                "terminal source manifest SHA does not match ledger input binding"
            )
    except V17RuntimeSnapshotDrift:
        raise
    except Exception as exc:
        raise V17RuntimeSnapshotDrift("terminal source manifest binding invalid") from exc


def _publish_after_terminal(
    repo_root: Path,
    *,
    run_id: str,
    ledger_sha256: str,
    expected_latest_sha256: str,
    published_at: str,
) -> dict[str, Any]:
    pointer, pointer_sha = publish_terminal_latest(
        repo_root,
        run_id=run_id,
        expected_ledger_sha256=ledger_sha256,
        expected_latest_sha256=expected_latest_sha256,
        published_at=published_at,
    )
    return {
        "run_id": run_id,
        "state": pointer["terminal_state"],
        "ledger_sha256": ledger_sha256,
        "latest_sha256": pointer_sha,
        "authority": False,
    }


def commit_hard_stop(
    repo_root: str | Path,
    *,
    run_id: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    state: str,
    blocker: str,
    stopped_at: str,
    skip_existing_artifact_payloads: bool = False,
) -> dict[str, Any]:
    if state not in {"HARD_STOP_SNAPSHOT_DRIFT", "HARD_STOP_INVALID_EVIDENCE"}:
        raise V17RuntimeError("invalid hard-stop state")
    root = Path(repo_root).absolute()
    ledger, observed = load_run_ledger(
        root,
        run_id,
        verify_artifacts=not skip_existing_artifact_payloads,
    )
    if observed != require_sha256(expected_ledger_sha256, label="expected ledger SHA-256"):
        raise V17LedgerCASMismatch("hard-stop CAS mismatch; zero writes")
    output = _terminal_output(
        ledger=ledger,
        predecessor_sha256=observed,
        state=state,
        generated_at=stopped_at,
        rank_output=None,
        portfolio_output=None,
        blockers=[blocker],
    )
    if skip_existing_artifact_payloads:
        if state != "HARD_STOP_SNAPSHOT_DRIFT":
            raise V17RuntimeError(
                "artifact-payload bypass is restricted to snapshot-drift hard stop"
            )
        _, terminal_sha = advance_snapshot_drift_hard_stop(
            root,
            run_id=run_id,
            expected_ledger_sha256=observed,
            transitioned_at=stopped_at,
            terminal_output=output,
        )
    else:
        _, terminal_sha = advance_run_state(
            root,
            run_id=run_id,
            expected_ledger_sha256=observed,
            next_state=state,
            transitioned_at=stopped_at,
            terminal_output=output,
        )
    return _publish_after_terminal(
        root,
        run_id=run_id,
        ledger_sha256=terminal_sha,
        expected_latest_sha256=expected_latest_sha256,
        published_at=stopped_at,
    )


def receive_shadow_response_from_file(
    repo_root: str | Path,
    *,
    run_id: str,
    response_path: str | Path,
    expected_response_sha256: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    failed_at: str,
) -> dict[str, Any]:
    expected = require_sha256(expected_response_sha256, label="expected response SHA-256")
    try:
        before = file_sha256(response_path)
        if before != expected:
            raise V17RuntimeInvalidEvidence("deep response byte SHA mismatch")
        payload = read_json(response_path)
        if file_sha256(response_path) != before:
            raise V17RuntimeInvalidEvidence("deep response changed during import")
        return receive_shadow_response(
            repo_root,
            run_id=run_id,
            response=payload,
            response_byte_sha256=before,
            expected_ledger_sha256=expected_ledger_sha256,
        )
    except V17LedgerCASMismatch:
        raise
    except (V17LatestError, V17PostCommitReadbackError):
        raise
    except V17RuntimeArtifactDrift as exc:
        return commit_hard_stop(
            repo_root,
            run_id=run_id,
            expected_ledger_sha256=expected_ledger_sha256,
            expected_latest_sha256=expected_latest_sha256,
            state="HARD_STOP_SNAPSHOT_DRIFT",
            blocker=exc.blocker,
            stopped_at=failed_at,
            skip_existing_artifact_payloads=True,
        )
    except V17RuntimeSnapshotDrift as exc:
        return commit_hard_stop(
            repo_root,
            run_id=run_id,
            expected_ledger_sha256=expected_ledger_sha256,
            expected_latest_sha256=expected_latest_sha256,
            state="HARD_STOP_SNAPSHOT_DRIFT",
            blocker=f"deep_response_snapshot_drift:{type(exc).__name__}",
            stopped_at=failed_at,
        )
    except Exception as exc:
        return commit_hard_stop(
            repo_root,
            run_id=run_id,
            expected_ledger_sha256=expected_ledger_sha256,
            expected_latest_sha256=expected_latest_sha256,
            state="HARD_STOP_INVALID_EVIDENCE",
            blocker=f"deep_response_invalid:{type(exc).__name__}",
            stopped_at=failed_at,
        )


def _revalidate_sources(
    repo_root: Path,
    ledger: Mapping[str, Any],
) -> SourceBindingBundle:
    try:
        bindings = ledger["input_bindings"]
        return load_source_manifest_binding(
            repo_root,
            bindings["source_manifest_path"],
            expected_manifest_sha256=bindings["source_manifest_sha256"],
            validate_authority_payloads=True,
        )
    except Exception as exc:
        raise V17RuntimeSnapshotDrift("sealed source manifest revalidation drift") from exc


def finalize_shadow_run(
    repo_root: str | Path,
    *,
    run_id: str,
    finalization: Mapping[str, Any],
    finalization_byte_sha256: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
) -> dict[str, Any]:
    root = Path(repo_root).absolute()
    try:
        ledger, observed = load_run_ledger(root, run_id, verify_artifacts=False)
    except Exception as exc:
        raise V17RuntimeSnapshotDrift("finalize ledger snapshot drift") from exc
    if observed != require_sha256(expected_ledger_sha256, label="expected ledger SHA-256"):
        raise V17LedgerCASMismatch("finalize CAS mismatch; zero writes")
    if ledger["state"] not in {"DEEP_RESPONSE_RECEIVED", "PORTFOLIO_COMPLETE"}:
        raise V17RuntimeInvalidEvidence(
            "finalize requires DEEP_RESPONSE_RECEIVED or resumable PORTFOLIO_COMPLETE"
        )
    _revalidate_ledger_package_bindings(ledger)
    _revalidate_ledger_implementation_bindings(root, ledger)
    bundle = _revalidate_sources(root, ledger)
    deterministic, deep_request = _recompute_prepare_artifacts_from_sources(
        root,
        ledger=ledger,
        bundle=bundle,
    )
    deep_evaluation = _recompute_stored_deep_evaluation(
        root,
        ledger,
        deterministic=deterministic,
        deep_request=deep_request,
    )
    try:
        validated = _validate_finalization(
            finalization,
            ledger=ledger,
            sealed_symbols=deterministic["sealed_symbols"],
        )
        import_binding = _import_binding(
            run_id=run_id,
            role="finalization",
            source_byte_sha256=finalization_byte_sha256,
            payload=validated,
        )
    except V17RuntimeSnapshotDrift:
        raise
    except Exception as exc:
        raise V17RuntimeInvalidEvidence("finalization evidence is invalid") from exc
    status, rank_output, portfolio_output, blockers, computation = compute_finalization(
        bundle,
        deterministic=deterministic,
        deep_evaluation=deep_evaluation,
        candidate_proposals=validated["candidate_proposals"],
        strategy_id=ledger["strategy_id"],
        cutoff=ledger["cutoff"],
    )
    artifacts = {
        "finalization_input": validated,
        "finalization_import": import_binding,
        "portfolio_computation": computation,
    }
    if ledger["state"] == "PORTFOLIO_COMPLETE":
        if status != "COMPLETE" or portfolio_output is None:
            raise V17RuntimeSnapshotDrift(
                "resumed portfolio computation no longer reaches COMPLETE"
            )
        for role, payload in artifacts.items():
            _require_recomputed_artifact(
                root,
                ledger,
                role=role,
                recomputed=payload,
            )
        state = "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION"
        output = _terminal_output(
            ledger=ledger,
            predecessor_sha256=observed,
            state=state,
            generated_at=validated["finalized_at"],
            rank_output=rank_output,
            portfolio_output=portfolio_output,
            blockers=[],
        )
        _validate_terminal_source_binding(output, ledger=ledger)
        _, terminal_sha = advance_run_state(
            root,
            run_id=run_id,
            expected_ledger_sha256=observed,
            next_state=state,
            transitioned_at=validated["finalized_at"],
            terminal_output=output,
        )
    elif status == "COMPLETE":
        ledger, observed = advance_run_state(
            root,
            run_id=run_id,
            expected_ledger_sha256=observed,
            next_state="PORTFOLIO_COMPLETE",
            transitioned_at=validated["generated_at"],
            artifacts=artifacts,
        )
        state = "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION"
        output = _terminal_output(
            ledger=ledger,
            predecessor_sha256=observed,
            state=state,
            generated_at=validated["finalized_at"],
            rank_output=rank_output,
            portfolio_output=portfolio_output,
            blockers=[],
        )
        _validate_terminal_source_binding(output, ledger=ledger)
        _, terminal_sha = advance_run_state(
            root,
            run_id=run_id,
            expected_ledger_sha256=observed,
            next_state=state,
            transitioned_at=validated["finalized_at"],
            terminal_output=output,
        )
    else:
        state = (
            "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
            if status == "NO_PORTFOLIO"
            else "SHADOW_PORTFOLIO_INFEASIBLE"
        )
        output = _terminal_output(
            ledger=ledger,
            predecessor_sha256=observed,
            state=state,
            generated_at=validated["finalized_at"],
            rank_output=rank_output,
            portfolio_output=None,
            blockers=blockers,
        )
        _validate_terminal_source_binding(output, ledger=ledger)
        _, terminal_sha = advance_run_state(
            root,
            run_id=run_id,
            expected_ledger_sha256=observed,
            next_state=state,
            transitioned_at=validated["finalized_at"],
            artifacts=artifacts,
            terminal_output=output,
        )
    return _publish_after_terminal(
        root,
        run_id=run_id,
        ledger_sha256=terminal_sha,
        expected_latest_sha256=expected_latest_sha256,
        published_at=validated["finalized_at"],
    )


def finalize_shadow_run_from_file(
    repo_root: str | Path,
    *,
    run_id: str,
    finalization_path: str | Path,
    expected_finalization_sha256: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    failed_at: str,
) -> dict[str, Any]:
    expected = require_sha256(expected_finalization_sha256, label="expected finalization SHA-256")
    ledger_expected = require_sha256(
        expected_ledger_sha256,
        label="expected ledger SHA-256",
    )
    before = ""
    payload: Mapping[str, Any] | None = None
    try:
        before = file_sha256(finalization_path)
        if before != expected:
            raise V17RuntimeInvalidEvidence("finalization byte SHA mismatch")
        payload = read_json(finalization_path)
        if file_sha256(finalization_path) != before:
            raise V17RuntimeInvalidEvidence("finalization changed during import")
        return finalize_shadow_run(
            repo_root,
            run_id=run_id,
            finalization=payload,
            finalization_byte_sha256=before,
            expected_ledger_sha256=ledger_expected,
            expected_latest_sha256=expected_latest_sha256,
        )
    except V17LedgerCASMismatch:
        raise
    except (V17LatestError, V17LatestPostCommitReadbackError, V17PostCommitReadbackError):
        raise
    except V17RuntimeArtifactDrift as exc:
        return commit_hard_stop(
            repo_root,
            run_id=run_id,
            expected_ledger_sha256=ledger_expected,
            expected_latest_sha256=expected_latest_sha256,
            state="HARD_STOP_SNAPSHOT_DRIFT",
            blocker=exc.blocker,
            stopped_at=failed_at,
            skip_existing_artifact_payloads=True,
        )
    except Exception as exc:
        failure = exc
        failure_expected_sha = ledger_expected
        try:
            current_ledger, current_sha = load_run_ledger(
                Path(repo_root).absolute(),
                run_id,
                verify_artifacts=False,
            )
        except Exception as read_exc:
            raise V17RuntimeSnapshotDrift(
                "cannot resolve finalize failure against current ledger"
            ) from read_exc

        if current_sha != ledger_expected:
            if (
                current_ledger.get("state") == "PORTFOLIO_COMPLETE"
                and current_ledger.get("previous_ledger_sha256") == ledger_expected
                and payload is not None
                and before == expected
            ):
                try:
                    return finalize_shadow_run(
                        repo_root,
                        run_id=run_id,
                        finalization=payload,
                        finalization_byte_sha256=before,
                        expected_ledger_sha256=current_sha,
                        expected_latest_sha256=expected_latest_sha256,
                    )
                except V17LedgerCASMismatch:
                    raise
                except (
                    V17LatestError,
                    V17LatestPostCommitReadbackError,
                    V17PostCommitReadbackError,
                ):
                    raise
                except V17RuntimeArtifactDrift as artifact_exc:
                    return commit_hard_stop(
                        repo_root,
                        run_id=run_id,
                        expected_ledger_sha256=current_sha,
                        expected_latest_sha256=expected_latest_sha256,
                        state="HARD_STOP_SNAPSHOT_DRIFT",
                        blocker=artifact_exc.blocker,
                        stopped_at=failed_at,
                        skip_existing_artifact_payloads=True,
                    )
                except Exception as retry_exc:
                    failure = retry_exc
                    failure_expected_sha = current_sha
            else:
                raise V17LedgerCASMismatch(
                    "finalize failure ledger changed outside resumable transition; zero writes"
                )

        state = (
            "HARD_STOP_SNAPSHOT_DRIFT"
            if isinstance(failure, V17RuntimeSnapshotDrift)
            else "HARD_STOP_INVALID_EVIDENCE"
        )
        return commit_hard_stop(
            repo_root,
            run_id=run_id,
            expected_ledger_sha256=failure_expected_sha,
            expected_latest_sha256=expected_latest_sha256,
            state=state,
            blocker=f"finalization_failed:{type(failure).__name__}",
            stopped_at=failed_at,
        )


def shadow_status(repo_root: str | Path, run_id: str) -> dict[str, Any]:
    root = Path(repo_root).absolute()
    ledger, ledger_sha = load_run_ledger(root, run_id, verify_artifacts=False)
    if ledger.get("state") != "HARD_STOP_SNAPSHOT_DRIFT":
        ledger, ledger_sha = load_run_ledger(root, run_id, verify_artifacts=True)
        _revalidate_ledger_package_bindings(ledger)
        _revalidate_ledger_implementation_bindings(root, ledger)
    latest = read_latest_pointer(root, verify_targets=True)
    latest_payload = latest[0] if latest is not None else None
    return {
        "version": STATUS_VERSION,
        "run_id": run_id,
        "state": ledger["state"],
        "terminal": is_terminal_state(str(ledger["state"])),
        "ledger_sha256": ledger_sha,
        "is_latest": bool(
            latest_payload
            and latest_payload.get("run_id") == run_id
            and latest_payload.get("ledger_sha256") == ledger_sha
        ),
        "latest": latest_payload,
        "authority": False,
    }


__all__ = [
    "DEEP_REQUEST_VERSION",
    "DEEP_RESPONSE_VERSION",
    "DETERMINISTIC_RESULT_VERSION",
    "FINALIZATION_VERSION",
    "IMPORT_BINDING_VERSION",
    "PREPARE_REQUEST_VERSION",
    "STATUS_VERSION",
    "V17RuntimeArtifactDrift",
    "V17RuntimeError",
    "V17RuntimeInvalidEvidence",
    "V17RuntimeSnapshotDrift",
    "commit_hard_stop",
    "finalize_shadow_run",
    "finalize_shadow_run_from_file",
    "prepare_shadow_run",
    "prepare_shadow_run_from_file",
    "receive_shadow_response",
    "receive_shadow_response_from_file",
    "shadow_status",
    "validate_prepare_request",
]
