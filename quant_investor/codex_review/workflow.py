"""Offline two-stage Codex review state machine for v16 candidate decisions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from quant_investor.v16.candidate_pipeline import (
    LLMBranchVerdict,
    PosteriorMenuItem,
    RetrievalEvidence as V16RetrievalEvidence,
    Stage2Decision,
    build_candidate_union,
    build_posterior_menu,
    validate_stage1_review,
    validate_stage2_portfolio,
)
from quant_investor.v16.stage1_contract import Stage1FactPackage

from .models import (
    AuthorizationDecision,
    CapitalMap,
    HumanAuthorization,
    MenuSeal,
    ReviewState,
    RunState,
    Stage1FactPackageModel,
    Stage1Request,
    Stage1Response,
    Stage2Request,
    Stage2Response,
)
from .storage import (
    CONTROL_MAX_BYTES,
    REQUEST_MAX_BYTES,
    RESPONSE_MAX_BYTES,
    DifferentBytesError,
    ProtocolError,
    StateConflictError,
    StrictJSONError,
    assert_cas,
    atomic_write_bytes,
    canonical_json_bytes,
    parse_strict_json_bytes,
    read_private_bytes,
    read_strict_json,
    run_lock,
    sha256_bytes,
    sha256_file,
    write_exact_once,
)

ZERO_SHA256 = "0" * 64
DEFAULT_REVIEW_ROOT = "results/v16/codex_review"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ReviewValidationError(ProtocolError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def symbol_set_sha256(symbols: list[str]) -> str:
    normalized = sorted(set(str(item).strip() for item in symbols))
    encoded = canonical_json_bytes(normalized)
    return sha256_bytes(encoded[:-1] if encoded.endswith(b"\n") else encoded)


def _v16_payload_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validate_fact_package(package: Stage1FactPackageModel) -> None:
    payload = package.model_dump(mode="json")
    supplied = payload.pop("payload_sha256")
    if _v16_payload_sha256(payload) != supplied:
        raise ReviewValidationError("Stage1 fact package payload_sha256 mismatch")
    row_symbols = [item.symbol for item in package.rows]
    if len(row_symbols) != len(set(row_symbols)):
        raise ReviewValidationError("Stage1 fact package row symbols must be unique")
    if row_symbols != sorted(row_symbols):
        raise ReviewValidationError("Stage1 fact package rows must be symbol-sorted")
    if package.universe_symbol_set_sha256 != symbol_set_sha256(row_symbols):
        raise ReviewValidationError("Stage1 universe_symbol_set_sha256 mismatch")
    if package.funnel_symbol_set_sha256 != symbol_set_sha256(package.funnel_symbols):
        raise ReviewValidationError("Stage1 funnel_symbol_set_sha256 mismatch")
    if not set(package.funnel_symbols).issubset(set(row_symbols)):
        raise ReviewValidationError("Stage1 Funnel contains symbols outside fact package")
    expected_strata: dict[str, int] = {}
    for row in package.rows:
        expected_strata[row.stratum] = expected_strata.get(row.stratum, 0) + 1
    if package.stratum_counts != dict(sorted(expected_strata.items())):
        raise ReviewValidationError("Stage1 stratum_counts mismatch")


def seal_json_payload(payload: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    """Return a copy with a canonical SHA over every field except the SHA field."""

    sealed = dict(payload)
    sealed.pop(digest_field, None)
    sealed[digest_field] = sha256_bytes(canonical_json_bytes(sealed))
    return sealed


def _model_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _seal_model(model_type: type[_ModelT], payload: Mapping[str, Any], field: str) -> _ModelT:
    try:
        unsealed = dict(payload)
        unsealed[field] = ZERO_SHA256
        normalized = model_type.model_validate(unsealed).model_dump(mode="json")
        return model_type.model_validate(seal_json_payload(normalized, digest_field=field))
    except ValidationError as exc:
        raise ReviewValidationError(str(exc)) from exc


def _verify_model_seal(model: BaseModel, field: str) -> str:
    payload = _model_payload(model)
    supplied = str(payload.pop(field, "")).lower()
    expected = sha256_bytes(canonical_json_bytes(payload))
    if supplied != expected:
        raise ReviewValidationError(f"{field} mismatch")
    return supplied


def _validate_model(model_type: type[_ModelT], value: Any) -> _ModelT:
    try:
        return model_type.model_validate(value)
    except ValidationError as exc:
        raise ReviewValidationError(str(exc)) from exc


def _git_sha(repo_path: str | Path) -> str:
    repo = Path(repo_path).resolve(strict=True)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"unable to resolve bound git SHA: {exc}") from exc
    value = completed.stdout.strip().lower()
    if not value or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError("bound git SHA is invalid")
    return value


def _absolute_file(path: str | Path) -> Path:
    target = Path(path).absolute()
    if target.is_symlink() or not target.is_file():
        raise ProtocolError(f"bound file is missing or unsafe: {target}")
    return target.resolve(strict=True)


def _relative(run_dir: Path, path: Path) -> str:
    return str(path.relative_to(run_dir))


def _artifact_path(run_dir: Path, relative: str) -> Path:
    candidate = run_dir / relative
    if candidate.is_symlink():
        raise ProtocolError("symlinks are forbidden in run artifacts")
    resolved_parent = candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise ProtocolError("state artifact path escapes run directory") from exc
    return candidate


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def _load_state(run_dir: Path) -> tuple[RunState, str]:
    path = _state_path(run_dir)
    payload = read_private_bytes(path, max_bytes=CONTROL_MAX_BYTES)
    value = parse_strict_json_bytes(payload, max_bytes=CONTROL_MAX_BYTES)
    return _validate_model(RunState, value), sha256_bytes(payload)


def _write_state(root: Path, run_dir: Path, state: RunState) -> str:
    return atomic_write_bytes(
        _state_path(run_dir),
        canonical_json_bytes(_model_payload(state)),
        root=root,
    )


def _state_result(state: RunState, state_sha256: str) -> dict[str, Any]:
    payload = _model_payload(state)
    payload["state_file_sha256"] = state_sha256
    payload["accepted"] = state.state not in {
        ReviewState.BLOCKED,
        ReviewState.EXPIRED,
    }
    return payload


def _environment_blockers(state: RunState) -> list[str]:
    blockers: list[str] = []
    try:
        if _git_sha(state.repo_path) != state.git_sha:
            blockers.append("git_sha_drift")
    except ProtocolError:
        blockers.append("git_sha_unreadable")
    for label, path, expected in (
        ("config", state.config_path, state.config_sha256),
        ("prompt", state.prompt_path, state.prompt_sha256),
        ("pit_pointer", state.pit_pointer_path, state.pit_pointer_sha256),
    ):
        try:
            actual = sha256_file(path)
        except ProtocolError:
            blockers.append(f"{label}_unreadable")
            continue
        if actual != expected:
            blockers.append(f"{label}_sha256_drift")
    return blockers


def _guard_live_state(
    *,
    root: Path,
    run_dir: Path,
    state: RunState,
    now: datetime,
) -> tuple[RunState, str] | None:
    if state.state in {ReviewState.AUTHORIZED, ReviewState.BLOCKED, ReviewState.EXPIRED}:
        return None
    if now >= state.expires_at:
        expired = state.model_copy(
            update={
                "state": ReviewState.EXPIRED,
                "revision": state.revision + 1,
                "updated_at": now,
                "blockers": ["review_expired"],
            }
        )
        return expired, _write_state(root, run_dir, expired)
    blockers = _environment_blockers(state)
    if blockers:
        blocked = state.model_copy(
            update={
                "state": ReviewState.BLOCKED,
                "revision": state.revision + 1,
                "updated_at": now,
                "blockers": blockers,
            }
        )
        return blocked, _write_state(root, run_dir, blocked)
    return None


def prepare_stage1_run(
    *,
    root: str | Path = DEFAULT_REVIEW_ROOT,
    run_id: str,
    payload: Stage1FactPackage | Mapping[str, Any],
    config_path: str | Path,
    prompt_path: str | Path,
    model_id: str,
    pit_pointer_path: str | Path,
    decision_cutoff_at: datetime | None = None,
    expires_at: datetime | None = None,
    repo_path: str | Path = ".",
    git_sha: str = "",
    expected_state_sha256: str = "EMPTY",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create one immutable Stage1 request and leave the run at S1_PREPARED."""

    prepared_at = now or utc_now()
    if isinstance(payload, Stage1FactPackage):
        payload.verify()
        fact_package = _validate_model(Stage1FactPackageModel, payload.to_dict())
    else:
        fact_package = _validate_model(Stage1FactPackageModel, payload)
    _validate_fact_package(fact_package)
    try:
        package_cutoff = datetime.fromisoformat(fact_package.cutoff_at.replace("Z", "+00:00"))
        package_expiry = datetime.fromisoformat(fact_package.expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewValidationError("fact package timestamps are invalid") from exc
    decision_cutoff_at = decision_cutoff_at or package_cutoff
    expires_at = expires_at or package_expiry
    if decision_cutoff_at != package_cutoff or expires_at != package_expiry:
        raise ReviewValidationError("fact package cutoff/expiry binding mismatch")
    repo = Path(repo_path).resolve(strict=True)
    current_git_sha = _git_sha(repo)
    declared_git_sha = str(git_sha).strip().lower() or current_git_sha
    if current_git_sha != declared_git_sha:
        raise ReviewValidationError("declared git_sha does not match current checkout")
    config = _absolute_file(config_path)
    prompt = _absolute_file(prompt_path)
    pit_pointer = _absolute_file(pit_pointer_path)
    if sha256_file(pit_pointer) != fact_package.pit_pointer_sha256:
        raise ReviewValidationError("fact package PIT pointer SHA mismatch")
    symbols = list(fact_package.funnel_symbols)
    model_sha = sha256_bytes(str(model_id).encode("utf-8"))
    request_payload: dict[str, Any] = {
        "schema_version": "codex-review-stage1-request.v1",
        "run_id": run_id,
        "stage": 1,
        "git_sha": declared_git_sha,
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "prompt_path": str(prompt),
        "prompt_sha256": sha256_file(prompt),
        "model_id": model_id,
        "model_sha256": model_sha,
        "pit_pointer_path": str(pit_pointer),
        "pit_pointer_sha256": sha256_file(pit_pointer),
        "symbol_set": symbols,
        "symbol_set_sha256": symbol_set_sha256(symbols),
        "predecessor_sha256": ZERO_SHA256,
        "decision_cutoff_at": decision_cutoff_at,
        "expires_at": expires_at,
        "fact_package": fact_package.model_dump(mode="json"),
    }
    request = _seal_model(Stage1Request, request_payload, "request_sha256")
    request_bytes = canonical_json_bytes(_model_payload(request))
    if len(request_bytes) > REQUEST_MAX_BYTES:
        raise StrictJSONError(f"Stage1 request exceeds {REQUEST_MAX_BYTES} bytes")

    with run_lock(root, run_id) as (root_path, run_dir):
        state_path = _state_path(run_dir)
        request_path = run_dir / "stage1" / "request.prepared.json"
        if state_path.exists():
            state, state_sha = _load_state(run_dir)
            existing = read_private_bytes(request_path, max_bytes=REQUEST_MAX_BYTES)
            if existing != request_bytes:
                raise DifferentBytesError("different Stage1 request bytes require a new run_id")
            return _state_result(state, state_sha)
        assert_cas(state_path, expected_state_sha256)
        write_exact_once(
            request_path,
            request_bytes,
            root=root_path,
        )
        state = RunState(
            run_id=run_id,
            state=ReviewState.S1_PREPARED,
            revision=1,
            updated_at=prepared_at,
            repo_path=str(repo),
            git_sha=declared_git_sha,
            config_path=str(config),
            config_sha256=sha256_file(config),
            prompt_path=str(prompt),
            prompt_sha256=sha256_file(prompt),
            model_id=model_id,
            model_sha256=model_sha,
            pit_pointer_path=str(pit_pointer),
            pit_pointer_sha256=sha256_file(pit_pointer),
            decision_cutoff_at=decision_cutoff_at,
            expires_at=expires_at,
            stage1_request_path=_relative(run_dir, request_path),
            stage1_request_sha256=request.request_sha256,
        )
        return _state_result(state, _write_state(root_path, run_dir, state))


def export_review_request(
    *,
    root: str | Path = DEFAULT_REVIEW_ROOT,
    run_id: str,
    expected_state_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Seal the currently prepared request as the local export artifact."""

    occurred_at = now or utc_now()
    with run_lock(root, run_id) as (root_path, run_dir):
        state, state_sha = _load_state(run_dir)
        if state.state in {ReviewState.BLOCKED, ReviewState.EXPIRED}:
            return _state_result(state, state_sha)
        if (
            state.state
            in {
                ReviewState.S1_EXPORTED,
                ReviewState.S1_RECEIVED,
                ReviewState.S1_VALIDATED,
                ReviewState.MENU_SEALED,
                ReviewState.S2_EXPORTED,
                ReviewState.S2_RECEIVED,
                ReviewState.S2_VALIDATED,
                ReviewState.CAPITAL_MAPPED,
                ReviewState.AWAITING_HUMAN_AUTH,
                ReviewState.AUTHORIZED,
            }
            and not state.stage2_request_path
        ):
            return _state_result(state, state_sha)
        if state.state in {
            ReviewState.S2_EXPORTED,
            ReviewState.S2_RECEIVED,
            ReviewState.S2_VALIDATED,
            ReviewState.CAPITAL_MAPPED,
            ReviewState.AWAITING_HUMAN_AUTH,
            ReviewState.AUTHORIZED,
        }:
            return _state_result(state, state_sha)
        expected_state = (
            ReviewState.S1_PREPARED
            if state.state == ReviewState.S1_PREPARED
            else ReviewState.S2_PREPARED
        )
        if state.state != expected_state:
            raise StateConflictError(f"cannot export request from {state.state.value}")
        assert_cas(_state_path(run_dir), expected_state_sha256)
        terminal = _guard_live_state(root=root_path, run_dir=run_dir, state=state, now=occurred_at)
        if terminal:
            return _state_result(*terminal)
        stage = 1 if state.state == ReviewState.S1_PREPARED else 2
        prepared_relative = state.stage1_request_path if stage == 1 else state.stage2_request_path
        prepared_path = _artifact_path(run_dir, prepared_relative)
        prepared_bytes = read_private_bytes(prepared_path, max_bytes=REQUEST_MAX_BYTES)
        export_path = run_dir / f"stage{stage}" / "request.exported.json"
        request_file_sha, _ = write_exact_once(export_path, prepared_bytes, root=root_path)
        request_type = Stage1Request if stage == 1 else Stage2Request
        prepared_request = _validate_model(
            request_type,
            parse_strict_json_bytes(prepared_bytes, max_bytes=REQUEST_MAX_BYTES),
        )
        request_sha = _verify_model_seal(prepared_request, "request_sha256")
        expected_request_sha = (
            state.stage1_request_sha256 if stage == 1 else state.stage2_request_sha256
        )
        if request_sha != expected_request_sha:
            raise ReviewValidationError("export request SHA does not match prepared request")
        updated = state.model_copy(
            update={
                "state": ReviewState.S1_EXPORTED if stage == 1 else ReviewState.S2_EXPORTED,
                "revision": state.revision + 1,
                "updated_at": occurred_at,
            }
        )
        result = _state_result(updated, _write_state(root_path, run_dir, updated))
        result["export_path"] = str(export_path)
        result["request_sha256"] = request_sha
        result["request_file_sha256"] = request_file_sha
        return result


def _response_model_for_state(state: ReviewState) -> tuple[int, type[BaseModel]]:
    if state == ReviewState.S1_EXPORTED:
        return 1, Stage1Response
    if state == ReviewState.S2_EXPORTED:
        return 2, Stage2Response
    raise StateConflictError(f"cannot receive response from {state.value}")


def receive_review_response(
    *,
    root: str | Path = DEFAULT_REVIEW_ROOT,
    run_id: str,
    response_path: str | Path,
    expected_state_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Import exact response bytes once; different bytes require a new run."""

    occurred_at = now or utc_now()
    response_bytes, response_value = read_strict_json(
        response_path,
        max_bytes=RESPONSE_MAX_BYTES,
    )
    stage_value = response_value.get("stage") if isinstance(response_value, dict) else None
    model_type = (
        Stage1Response if stage_value == 1 else Stage2Response if stage_value == 2 else None
    )
    if model_type is None:
        raise ReviewValidationError("response stage must be 1 or 2")
    response = _validate_model(model_type, response_value)
    response_sha = _verify_model_seal(response, "response_sha256")

    with run_lock(root, run_id) as (root_path, run_dir):
        state, state_sha = _load_state(run_dir)
        canonical_path = run_dir / f"stage{stage_value}" / "response.received.json"
        if canonical_path.exists():
            existing = read_private_bytes(canonical_path, max_bytes=RESPONSE_MAX_BYTES)
            if existing != response_bytes:
                raise DifferentBytesError("different response bytes require a new run_id")
            return _state_result(state, state_sha)
        expected_stage, _ = _response_model_for_state(state.state)
        if stage_value != expected_stage:
            raise StateConflictError("response stage does not match exported stage")
        if response.run_id != run_id:
            raise ReviewValidationError("response run_id mismatch")
        assert_cas(_state_path(run_dir), expected_state_sha256)
        terminal = _guard_live_state(root=root_path, run_dir=run_dir, state=state, now=occurred_at)
        if terminal:
            return _state_result(*terminal)
        stored_sha, _ = write_exact_once(
            canonical_path,
            response_bytes,
            root=root_path,
        )
        if stored_sha != sha256_bytes(response_bytes):
            raise ReviewValidationError("received response byte SHA mismatch")
        update = {
            "state": ReviewState.S1_RECEIVED if stage_value == 1 else ReviewState.S2_RECEIVED,
            "revision": state.revision + 1,
            "updated_at": occurred_at,
        }
        if stage_value == 1:
            update.update(
                stage1_response_path=_relative(run_dir, canonical_path),
                stage1_response_sha256=response_sha,
            )
        else:
            update.update(
                stage2_response_path=_relative(run_dir, canonical_path),
                stage2_response_sha256=response_sha,
            )
        updated = state.model_copy(update=update)
        return _state_result(updated, _write_state(root_path, run_dir, updated))


_BINDING_FIELDS = (
    "run_id",
    "stage",
    "git_sha",
    "config_path",
    "config_sha256",
    "prompt_path",
    "prompt_sha256",
    "model_id",
    "model_sha256",
    "pit_pointer_path",
    "pit_pointer_sha256",
    "predecessor_sha256",
    "decision_cutoff_at",
    "expires_at",
    "request_sha256",
)


def _validate_response_bindings(request: BaseModel, response: BaseModel) -> None:
    for field in _BINDING_FIELDS:
        request_value = getattr(request, field)
        response_value = getattr(response, field)
        if request_value != response_value:
            raise ReviewValidationError(f"response binding mismatch: {field}")
    if response.model_sha256 != sha256_bytes(response.model_id.encode("utf-8")):
        raise ReviewValidationError("response model_sha256 mismatch")
    if symbol_set_sha256(list(response.symbol_set)) != response.symbol_set_sha256:
        raise ReviewValidationError("response symbol_set_sha256 mismatch")
    if response.stage == 2 and (
        request.symbol_set != response.symbol_set
        or request.symbol_set_sha256 != response.symbol_set_sha256
    ):
        raise ReviewValidationError("Stage2 response symbol-set binding mismatch")


def _block_validation_failure(
    *, root: Path, run_dir: Path, state: RunState, now: datetime, reason: str
) -> dict[str, Any]:
    blocked = state.model_copy(
        update={
            "state": ReviewState.BLOCKED,
            "revision": state.revision + 1,
            "updated_at": now,
            "blockers": [reason[:512]],
        }
    )
    return _state_result(blocked, _write_state(root, run_dir, blocked))


def validate_review_response(
    *,
    root: str | Path = DEFAULT_REVIEW_ROOT,
    run_id: str,
    expected_state_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate exact lineage and stage semantics, then advance once."""

    occurred_at = now or utc_now()
    with run_lock(root, run_id) as (root_path, run_dir):
        state, state_sha = _load_state(run_dir)
        if state.state in {
            ReviewState.S1_VALIDATED,
            ReviewState.MENU_SEALED,
            ReviewState.S2_PREPARED,
            ReviewState.S2_EXPORTED,
            ReviewState.S2_VALIDATED,
            ReviewState.CAPITAL_MAPPED,
            ReviewState.AWAITING_HUMAN_AUTH,
            ReviewState.AUTHORIZED,
        }:
            return _state_result(state, state_sha)
        if state.state not in {ReviewState.S1_RECEIVED, ReviewState.S2_RECEIVED}:
            raise StateConflictError(f"cannot validate response from {state.state.value}")
        assert_cas(_state_path(run_dir), expected_state_sha256)
        terminal = _guard_live_state(root=root_path, run_dir=run_dir, state=state, now=occurred_at)
        if terminal:
            return _state_result(*terminal)
        stage = 1 if state.state == ReviewState.S1_RECEIVED else 2
        request_relative = state.stage1_request_path if stage == 1 else state.stage2_request_path
        response_relative = state.stage1_response_path if stage == 1 else state.stage2_response_path
        request_type = Stage1Request if stage == 1 else Stage2Request
        response_type = Stage1Response if stage == 1 else Stage2Response
        try:
            request_bytes = read_private_bytes(
                _artifact_path(run_dir, request_relative), max_bytes=REQUEST_MAX_BYTES
            )
            response_bytes = read_private_bytes(
                _artifact_path(run_dir, response_relative), max_bytes=RESPONSE_MAX_BYTES
            )
            request = _validate_model(
                request_type,
                parse_strict_json_bytes(request_bytes, max_bytes=REQUEST_MAX_BYTES),
            )
            response = _validate_model(
                response_type,
                parse_strict_json_bytes(response_bytes, max_bytes=RESPONSE_MAX_BYTES),
            )
            request_sha = _verify_model_seal(request, "request_sha256")
            response_sha = _verify_model_seal(response, "response_sha256")
            if request_sha != (
                state.stage1_request_sha256 if stage == 1 else state.stage2_request_sha256
            ):
                raise ReviewValidationError("state request SHA mismatch")
            if response_sha != (
                state.stage1_response_sha256 if stage == 1 else state.stage2_response_sha256
            ):
                raise ReviewValidationError("state response SHA mismatch")
            _validate_response_bindings(request, response)
            update: dict[str, Any]
            if stage == 1:
                candidate_union = build_candidate_union(
                    request.symbol_set,
                    [item.symbol for item in response.supplemental_candidates],
                )
                supplemental = [item.symbol for item in response.supplemental_candidates]
                if set(supplemental) & set(request.symbol_set):
                    raise ReviewValidationError(
                        "Stage1 supplemental candidates must be outside the Funnel"
                    )
                universe_symbols = {item.symbol for item in request.fact_package.rows}
                if not set(supplemental).issubset(universe_symbols):
                    raise ReviewValidationError(
                        "Stage1 supplemental candidates must come from the sealed fact package"
                    )
                final_symbols = list(candidate_union.symbols)
                if response.symbol_set != final_symbols:
                    raise ReviewValidationError(
                        "Stage1 response symbol_set must be the sealed Funnel+supplemental union"
                    )
                llm_verdicts = [
                    LLMBranchVerdict(
                        symbol=item.symbol,
                        raw_score=item.raw_score,
                        confidence=item.confidence,
                        supporting_fact_ids=tuple(item.supporting_fact_ids),
                        contradicting_fact_ids=tuple(item.contradicting_fact_ids),
                        rationale=item.rationale,
                    )
                    for item in response.llm_verdicts
                ]
                retrieval = [
                    V16RetrievalEvidence(
                        symbol=item.symbol,
                        branch=item.branch,
                        supporting_fact_ids=tuple(item.supporting_fact_ids),
                        contradicting_fact_ids=tuple(item.contradicting_fact_ids),
                        conflict_note=item.conflict_note or None,
                    )
                    for item in response.retrieval_evidence
                ]
                evidence_keys = {(item.symbol, item.branch) for item in response.retrieval_evidence}
                required_evidence_keys = {
                    (symbol, branch)
                    for symbol in final_symbols
                    for branch in ("quant", "fundamental", "macro")
                }
                if evidence_keys != required_evidence_keys:
                    missing = sorted(required_evidence_keys - evidence_keys)
                    extra = sorted(evidence_keys - required_evidence_keys)
                    raise ReviewValidationError(
                        "retrieval_evidence must cover exact Q/F/M annotations "
                        f"for the final union: missing={missing}, extra={extra}"
                    )
                validate_stage1_review(
                    candidate_union,
                    llm_verdicts=llm_verdicts,
                    retrieval_evidence=retrieval,
                )
                update = {
                    "state": ReviewState.S1_VALIDATED,
                    "final_symbol_set": final_symbols,
                    "final_symbol_set_sha256": symbol_set_sha256(final_symbols),
                }
            else:
                menu = [
                    PosteriorMenuItem(
                        symbol=item.symbol,
                        posterior_win_rate=item.posterior_win_rate,
                        posterior_expected_alpha=item.posterior_expected_alpha,
                        posterior_edge_after_costs=item.posterior_edge_after_costs,
                    )
                    for item in request.menu
                ]
                decisions = [
                    Stage2Decision(
                        symbol=item.symbol,
                        action=item.action.value,
                        selected_for_portfolio=item.selected_for_portfolio,
                        target_weight=item.target_weight,
                        rationale=item.rationale,
                        risk_acceptance_rationale=(item.risk_acceptance_rationale or None),
                    )
                    for item in response.verdicts
                ]
                validate_stage2_portfolio(
                    menu,
                    decisions,
                    cash_ratio=response.cash_ratio,
                    existing_weights=request.existing_weights,
                    severe_risk_symbols={
                        item.symbol
                        for item in request.menu
                        if item.risk_advisory.severity in {"high", "extreme"}
                    },
                )
                if response.menu_sha256 != request.menu_sha256:
                    raise ReviewValidationError("Stage2 menu_sha256 mismatch")
                update = {"state": ReviewState.S2_VALIDATED}
        except (ProtocolError, ValueError, TypeError) as exc:
            return _block_validation_failure(
                root=root_path,
                run_dir=run_dir,
                state=state,
                now=occurred_at,
                reason=str(exc),
            )
        updated = state.model_copy(
            update={
                **update,
                "revision": state.revision + 1,
                "updated_at": occurred_at,
                "blockers": [],
            }
        )
        return _state_result(updated, _write_state(root_path, run_dir, updated))


def _load_control_model(
    path: str | Path, model_type: type[_ModelT], digest_field: str
) -> tuple[bytes, _ModelT, str]:
    payload, value = read_strict_json(path, max_bytes=CONTROL_MAX_BYTES)
    model = _validate_model(model_type, value)
    return payload, model, _verify_model_seal(model, digest_field)


def _prepare_stage2(*, root: Path, run_dir: Path, state: RunState, now: datetime) -> RunState:
    stage1_request_value = parse_strict_json_bytes(
        read_private_bytes(
            _artifact_path(run_dir, state.stage1_request_path),
            max_bytes=REQUEST_MAX_BYTES,
        ),
        max_bytes=REQUEST_MAX_BYTES,
    )
    stage1_request = _validate_model(Stage1Request, stage1_request_value)
    menu_value = parse_strict_json_bytes(
        read_private_bytes(_artifact_path(run_dir, state.menu_path), max_bytes=CONTROL_MAX_BYTES),
        max_bytes=CONTROL_MAX_BYTES,
    )
    menu = _validate_model(MenuSeal, menu_value)
    request_payload = {
        "schema_version": "codex-review-stage2-request.v1",
        "run_id": state.run_id,
        "stage": 2,
        "git_sha": state.git_sha,
        "config_path": state.config_path,
        "config_sha256": state.config_sha256,
        "prompt_path": state.prompt_path,
        "prompt_sha256": state.prompt_sha256,
        "model_id": state.model_id,
        "model_sha256": state.model_sha256,
        "pit_pointer_path": state.pit_pointer_path,
        "pit_pointer_sha256": state.pit_pointer_sha256,
        "symbol_set": menu.symbols,
        "symbol_set_sha256": symbol_set_sha256(menu.symbols),
        "predecessor_sha256": state.stage1_response_sha256,
        "decision_cutoff_at": stage1_request.decision_cutoff_at,
        "expires_at": stage1_request.expires_at,
        "menu_sha256": state.menu_sha256,
        "existing_weights": menu.existing_weights,
        "menu": [item.model_dump(mode="json") for item in menu.items],
    }
    request = _seal_model(Stage2Request, request_payload, "request_sha256")
    request_bytes = canonical_json_bytes(_model_payload(request))
    if len(request_bytes) > REQUEST_MAX_BYTES:
        raise StrictJSONError(f"Stage2 request exceeds {REQUEST_MAX_BYTES} bytes")
    request_path = run_dir / "stage2" / "request.prepared.json"
    write_exact_once(request_path, request_bytes, root=root)
    return state.model_copy(
        update={
            "state": ReviewState.S2_PREPARED,
            "revision": state.revision + 1,
            "updated_at": now,
            "stage2_request_path": _relative(run_dir, request_path),
            "stage2_request_sha256": request.request_sha256,
        }
    )


def resume_review(
    *,
    root: str | Path = DEFAULT_REVIEW_ROOT,
    run_id: str,
    expected_state_sha256: str,
    menu_path: str | Path | None = None,
    total_capital: float | None = None,
    authorization_path: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Advance exactly one deterministic resume boundary."""

    occurred_at = now or utc_now()
    if total_capital is not None and (
        not math.isfinite(float(total_capital)) or float(total_capital) <= 0.0
    ):
        raise ReviewValidationError("total_capital must be finite and positive")
    with run_lock(root, run_id) as (root_path, run_dir):
        state, state_sha = _load_state(run_dir)
        if menu_path is not None and state.menu_path:
            supplied_menu, _ = read_strict_json(menu_path, max_bytes=CONTROL_MAX_BYTES)
            stored_menu = read_private_bytes(
                _artifact_path(run_dir, state.menu_path), max_bytes=CONTROL_MAX_BYTES
            )
            if supplied_menu != stored_menu:
                raise DifferentBytesError("different sealed menu bytes require a new run_id")
            return _state_result(state, state_sha)
        if authorization_path is not None and state.authorization_path:
            supplied_auth, _ = read_strict_json(authorization_path, max_bytes=CONTROL_MAX_BYTES)
            stored_auth = read_private_bytes(
                _artifact_path(run_dir, state.authorization_path),
                max_bytes=CONTROL_MAX_BYTES,
            )
            if supplied_auth != stored_auth:
                raise DifferentBytesError("different authorization bytes require a new run_id")
            return _state_result(state, state_sha)
        if total_capital is not None and state.capital_map_path:
            capital_value = parse_strict_json_bytes(
                read_private_bytes(
                    _artifact_path(run_dir, state.capital_map_path),
                    max_bytes=CONTROL_MAX_BYTES,
                ),
                max_bytes=CONTROL_MAX_BYTES,
            )
            capital_map = _validate_model(CapitalMap, capital_value)
            if abs(capital_map.total_capital - float(total_capital)) > 1e-9:
                raise DifferentBytesError("different total_capital requires a new run_id")
            return _state_result(state, state_sha)
        if state.state in {ReviewState.AUTHORIZED, ReviewState.BLOCKED, ReviewState.EXPIRED}:
            return _state_result(state, state_sha)
        assert_cas(_state_path(run_dir), expected_state_sha256)
        terminal = _guard_live_state(root=root_path, run_dir=run_dir, state=state, now=occurred_at)
        if terminal:
            return _state_result(*terminal)

        if state.state == ReviewState.S1_VALIDATED:
            if menu_path is None or total_capital is not None or authorization_path is not None:
                raise StateConflictError("S1_VALIDATED resume requires only --menu")
            menu_bytes, menu, menu_sha = _load_control_model(menu_path, MenuSeal, "menu_sha256")
            if menu.run_id != state.run_id:
                raise ReviewValidationError("menu run_id mismatch")
            if menu.stage1_response_sha256 != state.stage1_response_sha256:
                raise ReviewValidationError("menu Stage1 predecessor SHA mismatch")
            if not set(menu.symbols).issubset(set(state.final_symbol_set)):
                raise ReviewValidationError("menu contains symbols outside Stage1 final union")
            expected_menu = build_posterior_menu(
                [
                    PosteriorMenuItem(
                        symbol=item.symbol,
                        posterior_win_rate=item.posterior_win_rate,
                        posterior_expected_alpha=item.posterior_expected_alpha,
                        posterior_edge_after_costs=item.posterior_edge_after_costs,
                    )
                    for item in menu.items
                ]
            )
            if [item.symbol for item in expected_menu] != menu.symbols:
                raise ReviewValidationError("menu order must be edge DESC, win DESC, symbol ASC")
            canonical_path = run_dir / "menu" / "sealed.json"
            stored_sha, _ = write_exact_once(canonical_path, menu_bytes, root=root_path)
            if stored_sha != sha256_bytes(menu_bytes):
                raise ReviewValidationError("menu byte SHA mismatch")
            updated = state.model_copy(
                update={
                    "state": ReviewState.MENU_SEALED,
                    "revision": state.revision + 1,
                    "updated_at": occurred_at,
                    "menu_path": _relative(run_dir, canonical_path),
                    "menu_sha256": menu_sha,
                }
            )
        elif state.state == ReviewState.MENU_SEALED:
            if menu_path is not None or total_capital is not None or authorization_path is not None:
                raise StateConflictError("MENU_SEALED resume accepts no extra input")
            updated = _prepare_stage2(root=root_path, run_dir=run_dir, state=state, now=occurred_at)
        elif state.state == ReviewState.S2_VALIDATED:
            if total_capital is None or menu_path is not None or authorization_path is not None:
                raise StateConflictError("S2_VALIDATED resume requires only --total-capital")
            response_value = parse_strict_json_bytes(
                read_private_bytes(
                    _artifact_path(run_dir, state.stage2_response_path),
                    max_bytes=RESPONSE_MAX_BYTES,
                ),
                max_bytes=RESPONSE_MAX_BYTES,
            )
            response = _validate_model(Stage2Response, response_value)
            request_value = parse_strict_json_bytes(
                read_private_bytes(
                    _artifact_path(run_dir, state.stage2_request_path),
                    max_bytes=REQUEST_MAX_BYTES,
                ),
                max_bytes=REQUEST_MAX_BYTES,
            )
            request = _validate_model(Stage2Request, request_value)
            menu_by_symbol = {item.symbol: item for item in request.menu}
            capital_payload = {
                "schema_version": "codex-review-capital-map.v1",
                "run_id": state.run_id,
                "stage2_response_sha256": state.stage2_response_sha256,
                "mapped_at": occurred_at,
                "total_capital": float(total_capital),
                "positions": [
                    {
                        "symbol": item.symbol,
                        "target_weight": item.target_weight,
                        "capital_amount": float(total_capital) * item.target_weight,
                        "reference_price": menu_by_symbol[item.symbol].reference_price,
                        "raw_target_shares": (
                            menu_by_symbol[item.symbol].existing_shares
                            if item.action.value == "HOLD"
                            else (
                                float(total_capital)
                                * item.target_weight
                                / menu_by_symbol[item.symbol].reference_price
                            )
                        ),
                        "target_shares": (
                            menu_by_symbol[item.symbol].existing_shares
                            if item.action.value == "HOLD"
                            else (
                                float(total_capital)
                                * item.target_weight
                                / menu_by_symbol[item.symbol].reference_price
                            )
                        ),
                    }
                    for item in response.verdicts
                    if item.selected_for_portfolio
                ],
                "cash_ratio": response.cash_ratio,
                "cash_amount": float(total_capital) * response.cash_ratio,
            }
            capital_map = _seal_model(CapitalMap, capital_payload, "capital_map_sha256")
            capital_path = run_dir / "capital" / "mapped.json"
            capital_bytes = canonical_json_bytes(_model_payload(capital_map))
            capital_file_sha, _ = write_exact_once(capital_path, capital_bytes, root=root_path)
            updated = state.model_copy(
                update={
                    "state": ReviewState.CAPITAL_MAPPED,
                    "revision": state.revision + 1,
                    "updated_at": occurred_at,
                    "capital_map_path": _relative(run_dir, capital_path),
                    "capital_map_sha256": capital_map.capital_map_sha256,
                }
            )
            if capital_file_sha != sha256_bytes(capital_bytes):
                raise ReviewValidationError("capital map byte SHA mismatch")
        elif state.state == ReviewState.CAPITAL_MAPPED:
            if menu_path is not None or total_capital is not None or authorization_path is not None:
                raise StateConflictError("CAPITAL_MAPPED resume accepts no extra input")
            updated = state.model_copy(
                update={
                    "state": ReviewState.AWAITING_HUMAN_AUTH,
                    "revision": state.revision + 1,
                    "updated_at": occurred_at,
                }
            )
        elif state.state == ReviewState.AWAITING_HUMAN_AUTH:
            if authorization_path is None or menu_path is not None or total_capital is not None:
                raise StateConflictError("AWAITING_HUMAN_AUTH resume requires only --authorization")
            auth_bytes, receipt, receipt_sha = _load_control_model(
                authorization_path,
                HumanAuthorization,
                "receipt_sha256",
            )
            if receipt.run_id != state.run_id:
                raise ReviewValidationError("authorization run_id mismatch")
            if receipt.stage2_response_sha256 != state.stage2_response_sha256:
                raise ReviewValidationError("authorization response SHA mismatch")
            if receipt.capital_map_sha256 != state.capital_map_sha256:
                raise ReviewValidationError("authorization capital map SHA mismatch")
            if receipt.authorized_at < state.updated_at:
                raise ReviewValidationError("authorization predates AWAITING_HUMAN_AUTH")
            if receipt.authorized_at > occurred_at:
                raise ReviewValidationError("authorization is future-dated")
            if occurred_at >= receipt.expires_at:
                raise ReviewValidationError("authorization receipt is expired")
            if receipt.expires_at > state.expires_at:
                raise ReviewValidationError("authorization expiry exceeds the bound review expiry")
            auth_path = run_dir / "authorization" / "receipt.json"
            write_exact_once(auth_path, auth_bytes, root=root_path)
            updated = state.model_copy(
                update={
                    "state": (
                        ReviewState.AUTHORIZED
                        if receipt.decision == AuthorizationDecision.AUTHORIZED
                        else ReviewState.BLOCKED
                    ),
                    "revision": state.revision + 1,
                    "updated_at": occurred_at,
                    "authorization_path": _relative(run_dir, auth_path),
                    "authorization_sha256": receipt_sha,
                    "blockers": (
                        []
                        if receipt.decision == AuthorizationDecision.AUTHORIZED
                        else ["human_authorization_blocked"]
                    ),
                }
            )
        else:
            return _state_result(state, state_sha)
        return _state_result(updated, _write_state(root_path, run_dir, updated))


def review_status(
    *, root: str | Path = DEFAULT_REVIEW_ROOT, run_id: str, now: datetime | None = None
) -> dict[str, Any]:
    """Read status without mutating state; disclose effective expiry and drift."""

    observed_at = now or utc_now()
    with run_lock(root, run_id) as (_, run_dir):
        state, state_sha = _load_state(run_dir)
        result = _state_result(state, state_sha)
        drift = _environment_blockers(state)
        expired = observed_at >= state.expires_at and state.state not in {
            ReviewState.AUTHORIZED,
            ReviewState.BLOCKED,
            ReviewState.EXPIRED,
        }
        result["observed_at"] = observed_at.isoformat()
        result["environment_drift"] = drift
        result["effective_state"] = (
            ReviewState.EXPIRED.value
            if expired
            else ReviewState.BLOCKED.value if drift else state.state.value
        )
        result["accepted"] = result["effective_state"] not in {
            ReviewState.BLOCKED.value,
            ReviewState.EXPIRED.value,
        }
        return result
