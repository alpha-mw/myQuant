"""Isolated v16 operator-advisory state machine."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import subprocess
from contextlib import contextmanager
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from quant_investor.v16.operator_advisory.contracts import (
    BRANCHES,
    BRANCH_SHARES,
    DECISIONS,
    DECISION_SCHEMA,
    LLM_RESPONSE_SCHEMA,
    MAX_ARTIFACT_BYTES,
    MAX_JSON_BYTES,
    REPORT_SCHEMA,
    REPO_ROOT,
    STATE_ADVISORY_COMPLETE,
    STATE_DECISION_RECORDED,
    STATE_LLM_REQUEST_READY,
    STATE_LLM_RESPONSE_RECEIVED,
    AdvisoryError,
    AdvisorySideEffectError,
    AdvisoryStateError,
    advisory_root,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_state,
    make_run_id,
    read_json,
    run_directory,
    save_state,
    unit_average_rank,
    utc_now,
    validate_llm_response,
    validate_publishable,
    validate_run_id,
    write_json_atomic,
    write_json_exclusive,
)
from quant_investor.v16.operator_advisory.factor_scoring import (
    ORDERED_FACTOR_NAMES,
    build_deterministic_inputs,
    load_input_manifest,
)
from quant_investor.v16.operator_advisory.provider import (
    CODEX_DELEGATED_MODEL,
    OPENAI_MODEL,
    PROMPT_SHA256,
    RESPONSE_SCHEMA_SHA256,
    build_llm_request,
    call_openai_responses,
)

_GUARD_HASH_LIMIT = 4 * 1024 * 1024
_SHARE_QUANTUM = Decimal("0.00000001")
_MAX_RESEARCH_SHARE = Decimal("0.20")
_LLM_BACKEND_MODELS = {
    "openai": OPENAI_MODEL,
    "codex": CODEX_DELEGATED_MODEL,
}
_RESPONSE_PROVIDER_MODES = {"external_file", "codex_delegated", "openai"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_snapshot() -> dict[str, str]:
    commands = {
        "status": ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        "worktree_diff": ["git", "diff", "--no-ext-diff", "--binary"],
        "index_diff": ["git", "diff", "--cached", "--no-ext-diff", "--binary"],
    }
    output: dict[str, str] = {}
    for key, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AdvisorySideEffectError(f"git side-effect snapshot failed: {key}")
        output[key] = hashlib.sha256(completed.stdout).hexdigest()
    return output


def _inventory_control_path(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    candidates: list[Path]
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        candidates = []
        for current_raw, directories, filenames in os.walk(path, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name not in {"__pycache__", ".pytest_cache", ".mypy_cache"}
            )
            current = Path(current_raw)
            candidates.extend(current / name for name in sorted(filenames))
    else:
        candidates = [path]
    for candidate in candidates:
        metadata = os.lstat(candidate)
        relative = str(candidate.relative_to(REPO_ROOT))
        entry: dict[str, Any] = {
            "mode": stat.S_IFMT(metadata.st_mode),
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }
        if stat.S_ISLNK(metadata.st_mode):
            entry["identity"] = f"symlink:{os.readlink(candidate)}"
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_size <= _GUARD_HASH_LIMIT:
            entry["identity"] = f"sha256:{file_sha256(candidate)}"
        elif stat.S_ISREG(metadata.st_mode):
            entry["identity"] = (
                f"stable-file:{metadata.st_dev}:{metadata.st_ino}:"
                f"{metadata.st_size}:{metadata.st_mtime_ns}:{metadata.st_ctime_ns}"
            )
        else:
            entry["identity"] = "non-regular"
        entries[relative] = entry


def _guard_inventory() -> dict[str, Any]:
    """Bind Git content plus every production-control and execution output root."""

    entries: dict[str, dict[str, Any]] = {}
    guarded_paths = [
        REPO_ROOT / "data" / "parquet" / "cn" / "_latest.json",
        REPO_ROOT / "data" / "parquet" / "cn" / "_catalog.json",
        REPO_ROOT / "data" / "parquet" / "cn" / "_fundamental_latest.json",
        REPO_ROOT / "data" / "parquet" / "cn" / "macro_daily" / "_latest.json",
        REPO_ROOT / "data" / "factor_library",
        REPO_ROOT / "results" / "v15",
        REPO_ROOT / "results" / "v16",
        REPO_ROOT / "results" / "strategy_records",
        REPO_ROOT / "portfolio_dashboard" / "generated",
    ]
    for path in guarded_paths:
        _inventory_control_path(path, entries)
    git_snapshot = _git_snapshot()
    digest_payload = {"git": git_snapshot, "control_entries": entries}
    return {
        "policy": "repo-wide-git-and-production-control-identity.v1",
        "entry_count": len(entries),
        "inventory_sha256": canonical_sha256(digest_payload),
        "git": git_snapshot,
        "entries": entries,
    }


def _assert_guard_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_entries = before.get("entries")
    after_entries = after.get("entries")
    if not isinstance(before_entries, Mapping) or not isinstance(after_entries, Mapping):
        raise AdvisorySideEffectError("repository side-effect inventory invalid")
    changed = sorted(
        path
        for path in set(before_entries) | set(after_entries)
        if before_entries.get(path) != after_entries.get(path)
    )
    if before.get("git") != after.get("git"):
        changed.append("<git-content-snapshot>")
    if changed:
        raise AdvisorySideEffectError(
            "unexpected repository side effects: " + ",".join(changed[:20])
        )
    return {
        "schema_version": "v16.operator-advisory-side-effect-attestation.v1",
        "status": "passed",
        "guard_policy": before.get("policy"),
        "entry_count": before.get("entry_count"),
        "before_inventory_sha256": before.get("inventory_sha256"),
        "after_inventory_sha256": after.get("inventory_sha256"),
        "changed_paths": [],
        "allowed_output_root": str(advisory_root().relative_to(REPO_ROOT)),
        "canonical_data_mutated": False,
        "formal_v16_mutated": False,
        "production_pointer_mutated": False,
        "broker_or_trade_surface_invoked": False,
    }


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".run.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _create_run_directory(run_id: str) -> Path:
    root = advisory_root()
    run_dir = root / validate_run_id(run_id)
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise AdvisoryStateError(f"advisory run already exists: {run_id}") from exc
    if run_dir.is_symlink() or run_dir.parent.resolve() != root:
        raise AdvisoryStateError("advisory run directory invalid")
    os.chmod(run_dir, 0o700)
    return run_dir


def _write_text_exclusive(path: Path, value: str) -> str:
    raw = value.encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    observed = file_sha256(path)
    if observed != _sha256_bytes(raw):
        raise AdvisoryError(f"advisory text readback mismatch: {path.name}")
    return observed


def _artifact_record(path: Path, sha256: str) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256,
        "size": path.stat().st_size,
    }


def _read_bound_artifact(
    run_dir: Path,
    state: Mapping[str, Any],
    artifact_id: str,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AdvisoryStateError("advisory state artifact map missing")
    binding = artifacts.get(artifact_id)
    if not isinstance(binding, Mapping):
        raise AdvisoryStateError(f"advisory artifact binding missing: {artifact_id}")
    path = run_dir / str(binding.get("path") or "")
    if path.parent != run_dir or path.name != binding.get("path"):
        raise AdvisoryStateError(f"advisory artifact path invalid: {artifact_id}")
    if not path.is_file() or path.is_symlink():
        raise AdvisoryStateError(f"advisory artifact unavailable: {artifact_id}")
    try:
        observed_sha256 = file_sha256(path)
    except OSError as exc:
        raise AdvisoryStateError(f"advisory artifact unreadable: {artifact_id}") from exc
    if observed_sha256 != binding.get("sha256"):
        raise AdvisoryStateError(f"advisory artifact hash mismatch: {artifact_id}")
    try:
        return read_json(path, max_bytes=max_bytes, require_single_link=True)
    except AdvisoryError as exc:
        raise AdvisoryStateError(f"advisory artifact invalid: {artifact_id}") from exc


def _status_view(state: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    report = state.get("artifacts", {}).get("advisory_report", {})
    if report:
        report_path = run_dir / str(report.get("path") or "")
        if (
            report_path.parent != run_dir
            or not report_path.is_file()
            or report_path.is_symlink()
            or file_sha256(report_path) != report.get("sha256")
        ):
            raise AdvisoryStateError("advisory report binding mismatch")
    return {
        "schema_version": "v16.operator-advisory-status.v1",
        "run_id": state.get("run_id"),
        "state": state.get("state"),
        "state_sha256": state.get("state_sha256"),
        "run_directory": str(run_dir),
        "report_path": str(run_dir / str(report.get("path") or "")) if report else "",
        "report_sha256": str(report.get("sha256") or "") if report else "",
        "requested_provider_mode": str(state.get("requested_provider_mode") or "unknown"),
        "provider_mode": str(state.get("provider_mode") or "unknown"),
        "provider_receipt_present": "provider_receipt" in state.get("artifacts", {}),
        "provider_receipt_sha256": str(
            state.get("artifacts", {}).get("provider_receipt", {}).get("sha256") or ""
        ),
        "production_authority": False,
        "new_risk_authorized": False,
        "broker_enabled": False,
        "human_decision_required": state.get("state") == STATE_ADVISORY_COMPLETE,
    }


def prepare_advisory(
    *,
    run_id: str = "",
    max_candidates: int = 30,
    top_k: int = 12,
    llm_backend: str = "openai",
) -> dict[str, Any]:
    resolved_max_candidates = int(max_candidates)
    resolved_top_k = int(top_k)
    if not 1 <= resolved_max_candidates <= 50:
        raise AdvisoryError("max_candidates must be within 1..50")
    if not 1 <= resolved_top_k <= 12:
        raise AdvisoryError("top_k must be within 1..12")
    resolved_llm_backend = str(llm_backend or "").strip().lower()
    if resolved_llm_backend not in _LLM_BACKEND_MODELS:
        raise AdvisoryError("llm_backend must be openai or codex")
    requested_provider_mode = "codex_delegated" if resolved_llm_backend == "codex" else "openai"
    request_model_id = _LLM_BACKEND_MODELS[resolved_llm_backend]
    resolved_run_id = validate_run_id(run_id) if run_id else make_run_id()
    before = _guard_inventory()
    run_dir = _create_run_directory(resolved_run_id)
    with _run_lock(run_dir):
        factor_bundle, evidence = build_deterministic_inputs(max_candidates=resolved_max_candidates)
        manifest = load_input_manifest()
        manifest_path = run_dir / "input_manifest.json"
        bundle_path = run_dir / "factor_bundle.json"
        evidence_path = run_dir / "branch_evidence.json"
        request_path = run_dir / "llm_request.json"
        artifacts: dict[str, Any] = {}
        artifacts["input_manifest"] = _artifact_record(
            manifest_path, write_json_exclusive(manifest_path, manifest)
        )
        artifacts["factor_bundle"] = _artifact_record(
            bundle_path, write_json_exclusive(bundle_path, factor_bundle)
        )
        evidence_sha = write_json_exclusive(evidence_path, evidence)
        artifacts["branch_evidence"] = _artifact_record(evidence_path, evidence_sha)
        llm_request = build_llm_request(
            evidence=evidence,
            evidence_file_sha256=evidence_sha,
            model_id=request_model_id,
        )
        request_sha = write_json_exclusive(request_path, llm_request)
        artifacts["llm_request"] = _artifact_record(request_path, request_sha)
        after = _guard_inventory()
        attestation = _assert_guard_unchanged(before, after)
        attestation_path = run_dir / "side_effect_prepare.json"
        artifacts["side_effect_prepare"] = _artifact_record(
            attestation_path,
            write_json_exclusive(attestation_path, attestation),
        )
        state = save_state(
            run_dir,
            {
                "state": STATE_LLM_REQUEST_READY,
                "created_at": utc_now(),
                "market": "CN",
                "max_candidates": resolved_max_candidates,
                "top_k": resolved_top_k,
                "llm_backend": resolved_llm_backend,
                "llm_model_id": request_model_id,
                "requested_provider_mode": requested_provider_mode,
                "provider_mode": "none",
                "artifacts": artifacts,
                "production_authority": False,
                "new_risk_authorized": False,
                "broker_enabled": False,
            },
        )
    return _status_view(state, run_dir)


def _receive_response_payload(
    *,
    run_dir: Path,
    response: Mapping[str, Any],
    expected_state_sha256: str,
    provider_mode: str,
    provider_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = load_state(run_dir.name)
    if state.get("state_sha256") != expected_state_sha256:
        raise AdvisoryStateError("advisory state CAS mismatch")
    if state.get("state") != STATE_LLM_REQUEST_READY:
        raise AdvisoryStateError("advisory response is not expected in current state")
    request = _read_bound_artifact(run_dir, state, "llm_request")
    request_sha = state["artifacts"]["llm_request"]["sha256"]
    resolved_provider_mode = str(provider_mode or "").strip().lower()
    if resolved_provider_mode not in _RESPONSE_PROVIDER_MODES:
        raise AdvisoryError("invalid advisory response provider mode")
    request_model_id = str(request.get("model_id") or "")
    if request_model_id not in set(_LLM_BACKEND_MODELS.values()):
        raise AdvisoryStateError("advisory request model is unsupported")
    requested_provider_mode = str(
        state.get("requested_provider_mode")
        or ("openai" if request_model_id == OPENAI_MODEL else "unknown")
    )
    if resolved_provider_mode == "openai":
        if requested_provider_mode != "openai" or request_model_id != OPENAI_MODEL:
            raise AdvisoryStateError("OpenAI response does not match prepared request")
        if provider_receipt is None:
            raise AdvisoryStateError("OpenAI response requires a provider receipt")
    elif resolved_provider_mode == "codex_delegated":
        if (
            requested_provider_mode != "codex_delegated"
            or request_model_id != CODEX_DELEGATED_MODEL
        ):
            raise AdvisoryStateError("Codex response does not match prepared request")
        if provider_receipt is not None:
            raise AdvisoryStateError("Codex receipt must be generated locally")
    else:
        if (
            requested_provider_mode == "codex_delegated"
            or request_model_id == CODEX_DELEGATED_MODEL
        ):
            raise AdvisoryStateError("Codex-bound request requires codex_delegated response source")
        if provider_receipt is not None:
            raise AdvisoryStateError("external file response cannot carry a provider receipt")
    validated = validate_llm_response(
        response,
        request=request,
        request_file_sha256=request_sha,
        model_id=request_model_id,
        prompt_sha256=PROMPT_SHA256,
        response_schema_sha256=RESPONSE_SCHEMA_SHA256,
    )
    normalized = {
        "schema_version": LLM_RESPONSE_SCHEMA,
        "request_sha256": request_sha,
        "model_id": request_model_id,
        "prompt_sha256": PROMPT_SHA256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "reviews": list(validated.values()),
    }
    before = _guard_inventory()
    response_path = run_dir / "llm_response.json"
    response_sha = write_json_exclusive(response_path, normalized)
    artifacts = dict(state["artifacts"])
    artifacts["llm_response"] = _artifact_record(response_path, response_sha)
    if resolved_provider_mode == "codex_delegated":
        provider_receipt = {
            "schema_version": "v16.operator-advisory-codex-receipt.v1",
            "reviewer": "codex_delegated_reviewer",
            "generation_surface": "current_codex_task",
            "model_id": request_model_id,
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "prompt_sha256": PROMPT_SHA256,
            "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
            "generated_at": utc_now(),
            "external_provider_api_called": False,
            "tools": [],
        }
    if provider_receipt is not None:
        receipt_path = run_dir / "provider_receipt.json"
        receipt_sha = write_json_exclusive(receipt_path, provider_receipt)
        artifacts["provider_receipt"] = _artifact_record(receipt_path, receipt_sha)
    after = _guard_inventory()
    attestation = _assert_guard_unchanged(before, after)
    attestation_path = run_dir / "side_effect_receive.json"
    artifacts["side_effect_receive"] = _artifact_record(
        attestation_path,
        write_json_exclusive(attestation_path, attestation),
    )
    return save_state(
        run_dir,
        {
            **{
                key: value
                for key, value in state.items()
                if key not in {"state_sha256", "updated_at"}
            },
            "state": STATE_LLM_RESPONSE_RECEIVED,
            "provider_mode": resolved_provider_mode,
            "artifacts": artifacts,
        },
        expected_state_sha256=expected_state_sha256,
    )


def _call_openai_guarded(
    *,
    request: Mapping[str, Any],
    request_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _guard_inventory()
    try:
        response, receipt = call_openai_responses(
            request=request,
            request_file_sha256=request_file_sha256,
        )
    except Exception:
        after = _guard_inventory()
        _assert_guard_unchanged(before, after)
        raise
    after = _guard_inventory()
    attestation = _assert_guard_unchanged(before, after)
    return response, {
        **receipt,
        "provider_interval_side_effect_attestation": attestation,
    }


def receive_advisory_response(
    *,
    run_id: str,
    response_path: str | Path,
    expected_state_sha256: str,
    response_source: str = "external_file",
) -> dict[str, Any]:
    run_dir = run_directory(validate_run_id(run_id))
    supplied = read_json(
        response_path,
        max_bytes=MAX_JSON_BYTES,
        require_single_link=True,
    )
    with _run_lock(run_dir):
        state = _receive_response_payload(
            run_dir=run_dir,
            response=supplied,
            expected_state_sha256=expected_state_sha256,
            provider_mode=response_source,
        )
    return _status_view(state, run_dir)


def _allocate_research_shares(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[dict[str, Decimal], Decimal]:
    selected = [row for row in rows[:top_k] if float(row["overall_score"]) > 0.5]
    convictions = {
        row["symbol"]: Decimal(str(float(row["overall_score"]) - 0.5)) for row in selected
    }
    allocations = {symbol: Decimal("0") for symbol in convictions}
    active = dict(convictions)
    remaining = Decimal("1")
    while active and remaining > 0:
        denominator = sum(active.values(), Decimal("0"))
        if denominator <= 0:
            break
        capped: list[str] = []
        for symbol, conviction in active.items():
            proposed = remaining * conviction / denominator
            if proposed > _MAX_RESEARCH_SHARE:
                allocations[symbol] = _MAX_RESEARCH_SHARE
                capped.append(symbol)
        if not capped:
            for symbol, conviction in active.items():
                allocations[symbol] = remaining * conviction / denominator
            remaining = Decimal("0")
            break
        for symbol in capped:
            remaining -= allocations[symbol]
            active.pop(symbol)
    floored = {
        symbol: value.quantize(_SHARE_QUANTUM, rounding=ROUND_DOWN)
        for symbol, value in allocations.items()
    }
    cash = Decimal("1") - sum(floored.values(), Decimal("0"))
    if cash < 0 or any(value > _MAX_RESEARCH_SHARE for value in floored.values()):
        raise AdvisoryError("research share allocation invariant failed")
    return floored, cash


def _build_report(
    *,
    run_id: str,
    state: Mapping[str, Any],
    factor_bundle: Mapping[str, Any],
    evidence: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    request = _read_bound_artifact(run_directory(run_id), state, "llm_request")
    reviews = validate_llm_response(
        response,
        request=request,
        request_file_sha256=state["artifacts"]["llm_request"]["sha256"],
        model_id=str(request.get("model_id") or ""),
        prompt_sha256=PROMPT_SHA256,
        response_schema_sha256=RESPONSE_SCHEMA_SHA256,
    )
    items = evidence.get("items")
    if not isinstance(items, list) or not items:
        raise AdvisoryError("sealed advisory evidence is empty")
    symbols = [str(item["symbol"]) for item in items]
    source_rows = {str(row["symbol"]): row for row in factor_bundle.get("rows", [])}
    quant = pd.Series({symbol: float(source_rows[symbol]["quant_raw"]) for symbol in symbols})
    fundamental = pd.Series(
        {symbol: float(source_rows[symbol]["fundamental_raw"]) for symbol in symbols}
    )
    macro = pd.Series({symbol: float(source_rows[symbol]["macro_raw"]) for symbol in symbols})
    llm = pd.Series({symbol: float(reviews[symbol]["raw_score"]) for symbol in symbols})
    branch_units = {
        "quant": unit_average_rank(quant),
        "fundamental": unit_average_rank(fundamental),
        "macro": unit_average_rank(macro),
        "llm": unit_average_rank(llm),
    }
    ranked_rows: list[dict[str, Any]] = []
    evidence_by_symbol = {str(item["symbol"]): item for item in items}
    for symbol in symbols:
        branch_scores = {branch: float(branch_units[branch].loc[symbol]) for branch in BRANCHES}
        overall = sum(BRANCH_SHARES[branch] * branch_scores[branch] for branch in BRANCHES)
        review = reviews[symbol]
        item = evidence_by_symbol[symbol]
        ranked_rows.append(
            {
                "symbol": symbol,
                "name": str(item.get("name") or "UNKNOWN_NAME"),
                "industry": str(item.get("industry") or "UNKNOWN_INDUSTRY"),
                "overall_score": float(overall),
                "branch_scores": branch_scores,
                "llm_raw_score": float(review["raw_score"]),
                "llm_confidence": float(review["confidence"]),
                "llm_rationale": review["rationale"],
                "llm_evidence_ids": review["evidence_ids"],
                "llm_risks": review["risks"],
                "research_share": 0.0,
            }
        )
    ranked_rows.sort(key=lambda row: (-float(row["overall_score"]), row["symbol"]))
    shares, cash = _allocate_research_shares(ranked_rows, top_k=int(state["top_k"]))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
        row["research_share"] = float(shares.get(row["symbol"], Decimal("0")))

    manifest = load_input_manifest()
    report = {
        "schema_version": REPORT_SCHEMA,
        "run_id": run_id,
        "market": "CN",
        "status": STATE_ADVISORY_COMPLETE,
        "generated_at": utc_now(),
        "source_bindings": factor_bundle["source_bindings"],
        "factor_policy": {
            "ordered_factors": list(ORDERED_FACTOR_NAMES),
            "family_count": factor_bundle["factor_family_count"],
            "blend_shares": {name: 0.2 for name in ORDERED_FACTOR_NAMES},
            "common_domain_count": factor_bundle["common_domain_count"],
            "source_state": "diagnostic_nonproduction",
        },
        "branch_policy": {
            "ordered_branches": list(BRANCHES),
            "branch_shares": BRANCH_SHARES,
            "macro_constant_rank_policy": 0.5,
            "risk_authority": "advisory_only",
        },
        "llm_response_provenance": {
            "requested_provider_mode": str(state.get("requested_provider_mode") or "unknown"),
            "provider_mode": str(state.get("provider_mode") or "unknown"),
            "model_id": str(request.get("model_id") or ""),
            "provider_receipt_present": "provider_receipt" in state.get("artifacts", {}),
            "provider_receipt_sha256": str(
                state.get("artifacts", {}).get("provider_receipt", {}).get("sha256") or ""
            ),
        },
        "research_waivers": manifest["research_waivers"],
        "non_waivable_gates": manifest["non_waivable_gates"],
        "ranked_candidates": ranked_rows,
        "allocation_summary": {
            "selected_name_count": sum(1 for value in shares.values() if value > 0),
            "research_share_sum": float(Decimal("1") - cash),
            "unallocated_cash_share": float(cash),
            "maximum_name_count": int(state["top_k"]),
            "maximum_single_share": float(_MAX_RESEARCH_SHARE),
            "rounding_decimals": 8,
        },
        "risk_notes": [
            "宏观环境读数偏弱，研究排序不替代人工判断。",
            "市场波动分位较高，实际操作前需重新核对当日价格与可交易状态。",
            "五个因子仍处于非生产研究状态，正式 v16 权限保持关闭。",
        ],
        "authority": {
            "production_authority": False,
            "new_risk_authorized": False,
            "formal_v16_activation_changed": False,
            "production_pointer_changed": False,
            "dashboard_activation_changed": False,
            "factor_registry_changed": False,
            "broker_enabled": False,
        },
        "human_decision_required": True,
        "decision_boundary": "由用户另行决定是否采取任何操作；本报告不触发券商、订单或交易接口。",
    }
    validate_publishable(report)
    return report


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CN v16 操作员研究排序",
        "",
        f"- 运行：`{report['run_id']}`",
        f"- 数据日：`{report['source_bindings']['trade_date']}`",
        "- 状态：等待用户人工决定",
        "- 权限：研究建议；生产权限与券商接口关闭",
        "",
        "## 排序",
        "",
        "| 序 | 代码 公司名 | 行业 | 综合分 | 研究份额 |",
        "|---:|---|---|---:|---:|",
    ]
    for row in report["ranked_candidates"]:
        lines.append(
            f"| {row['rank']} | {row['symbol']} {row['name']} | {row['industry']} | "
            f"{float(row['overall_score']):.6f} | {float(row['research_share']):.2%} |"
        )
    summary = report["allocation_summary"]
    lines.extend(
        [
            "",
            "## 研究份额",
            "",
            f"- 标的数量：{summary['selected_name_count']}",
            f"- 分配合计：{float(summary['research_share_sum']):.2%}",
            f"- 未分配现金：{float(summary['unallocated_cash_share']):.2%}",
            "",
            "## 风险说明",
            "",
            *[f"- {note}" for note in report["risk_notes"]],
            "",
            "最终是否采取操作由用户人工决定。本文件不会触发券商、订单或交易接口。",
            "",
        ]
    )
    markdown = "\n".join(lines)
    validate_publishable(markdown)
    return markdown


def finalize_advisory(
    *,
    run_id: str,
    expected_state_sha256: str,
) -> dict[str, Any]:
    run_dir = run_directory(validate_run_id(run_id))
    with _run_lock(run_dir):
        state = load_state(run_id)
        if state.get("state_sha256") != expected_state_sha256:
            raise AdvisoryStateError("advisory state CAS mismatch")
        if state.get("state") != STATE_LLM_RESPONSE_RECEIVED:
            raise AdvisoryStateError("advisory finalize is not allowed in current state")
        stored_bundle = _read_bound_artifact(run_dir, state, "factor_bundle")
        stored_evidence = _read_bound_artifact(run_dir, state, "branch_evidence")
        response = _read_bound_artifact(run_dir, state, "llm_response")
        before = _guard_inventory()
        recomputed_bundle, recomputed_evidence = build_deterministic_inputs(
            max_candidates=int(state["max_candidates"])
        )
        if canonical_json_bytes(recomputed_bundle) != canonical_json_bytes(stored_bundle):
            raise AdvisoryStateError("factor bundle recomputation mismatch")
        if canonical_json_bytes(recomputed_evidence) != canonical_json_bytes(stored_evidence):
            raise AdvisoryStateError("branch evidence recomputation mismatch")
        report = _build_report(
            run_id=run_id,
            state=state,
            factor_bundle=recomputed_bundle,
            evidence=recomputed_evidence,
            response=response,
        )
        markdown = _render_markdown(report)
        artifacts = dict(state["artifacts"])
        report_path = run_dir / "advisory_report.json"
        markdown_path = run_dir / "advisory_report.md"
        artifacts["advisory_report"] = _artifact_record(
            report_path, write_json_exclusive(report_path, report)
        )
        artifacts["advisory_markdown"] = _artifact_record(
            markdown_path, _write_text_exclusive(markdown_path, markdown)
        )
        after = _guard_inventory()
        attestation = _assert_guard_unchanged(before, after)
        attestation_path = run_dir / "side_effect_finalize.json"
        artifacts["side_effect_finalize"] = _artifact_record(
            attestation_path,
            write_json_exclusive(attestation_path, attestation),
        )
        completed = save_state(
            run_dir,
            {
                **{
                    key: value
                    for key, value in state.items()
                    if key not in {"state_sha256", "updated_at"}
                },
                "state": STATE_ADVISORY_COMPLETE,
                "completed_at": utc_now(),
                "artifacts": artifacts,
            },
            expected_state_sha256=expected_state_sha256,
        )
        latest = {
            "schema_version": "v16.operator-advisory-latest.v1",
            "run_id": run_id,
            "state": completed["state"],
            "state_sha256": completed["state_sha256"],
            "advisory_report_path": str(report_path.relative_to(REPO_ROOT)),
            "advisory_report_sha256": artifacts["advisory_report"]["sha256"],
            "updated_at": utc_now(),
            "production_authority": False,
            "new_risk_authorized": False,
        }
        write_json_atomic(advisory_root() / "_latest.json", latest)
    return _status_view(completed, run_dir)


def run_advisory(
    *,
    run_id: str = "",
    max_candidates: int = 30,
    top_k: int = 12,
    provider: str = "none",
) -> dict[str, Any]:
    provider_mode = str(provider or "none").strip().lower()
    if provider_mode not in {"none", "openai"}:
        raise AdvisoryError("provider must be none or openai")
    prepared = prepare_advisory(
        run_id=run_id,
        max_candidates=max_candidates,
        top_k=top_k,
        llm_backend="openai",
    )
    if provider_mode == "none":
        return prepared
    return resume_advisory_provider(
        run_id=str(prepared["run_id"]),
        expected_state_sha256=str(prepared["state_sha256"]),
    )


def resume_advisory_provider(
    *,
    run_id: str,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Call the pinned provider for an existing prepared run and finalize it."""

    run_dir = run_directory(validate_run_id(run_id))
    with _run_lock(run_dir):
        state = load_state(run_dir.name)
        if state.get("state_sha256") != expected_state_sha256:
            raise AdvisoryStateError("advisory state CAS mismatch")
        if state.get("state") != STATE_LLM_REQUEST_READY:
            raise AdvisoryStateError("advisory provider is not expected in current state")
        request = _read_bound_artifact(run_dir, state, "llm_request")
        requested_provider_mode = str(
            state.get("requested_provider_mode")
            or ("openai" if request.get("model_id") == OPENAI_MODEL else "unknown")
        )
        if requested_provider_mode != "openai" or request.get("model_id") != OPENAI_MODEL:
            raise AdvisoryStateError("prepared request is not eligible for OpenAI resume")
        response, receipt = _call_openai_guarded(
            request=request,
            request_file_sha256=state["artifacts"]["llm_request"]["sha256"],
        )
        received = _receive_response_payload(
            run_dir=run_dir,
            response=response,
            expected_state_sha256=state["state_sha256"],
            provider_mode="openai",
            provider_receipt=receipt,
        )
    return finalize_advisory(
        run_id=run_dir.name,
        expected_state_sha256=received["state_sha256"],
    )


def advisory_status(*, run_id: str = "", latest: bool = False) -> dict[str, Any]:
    if bool(run_id) == bool(latest):
        raise AdvisoryError("specify exactly one of run_id or latest")
    resolved_run_id = validate_run_id(run_id) if run_id else ""
    if latest:
        pointer = read_json(
            advisory_root() / "_latest.json",
            max_bytes=MAX_JSON_BYTES,
            require_single_link=True,
        )
        resolved_run_id = validate_run_id(str(pointer.get("run_id") or ""))
    run_dir = run_directory(resolved_run_id)
    state = load_state(resolved_run_id)
    if latest:
        report = state.get("artifacts", {}).get("advisory_report", {})
        if pointer.get("state_sha256") != state.get("state_sha256") or pointer.get(
            "advisory_report_sha256"
        ) != report.get("sha256"):
            raise AdvisoryStateError("latest advisory pointer binding mismatch")
    return _status_view(state, run_dir)


def record_advisory_decision(
    *,
    run_id: str,
    decision: str,
    expected_state_sha256: str,
) -> dict[str, Any]:
    normalized_decision = str(decision or "").strip().upper()
    if normalized_decision not in DECISIONS:
        raise AdvisoryError("invalid advisory decision")
    run_dir = run_directory(validate_run_id(run_id))
    with _run_lock(run_dir):
        state = load_state(run_id)
        if state.get("state_sha256") != expected_state_sha256:
            raise AdvisoryStateError("advisory state CAS mismatch")
        if state.get("state") != STATE_ADVISORY_COMPLETE:
            raise AdvisoryStateError("advisory decision is not expected in current state")
        record = {
            "schema_version": DECISION_SCHEMA,
            "run_id": run_id,
            "decision": normalized_decision,
            "recorded_at": utc_now(),
            "advisory_report_sha256": state["artifacts"]["advisory_report"]["sha256"],
            "production_authority": False,
            "new_risk_authorized": False,
            "broker_enabled": False,
        }
        path = run_dir / "decision_record.json"
        artifacts = dict(state["artifacts"])
        artifacts["decision_record"] = _artifact_record(path, write_json_exclusive(path, record))
        decided = save_state(
            run_dir,
            {
                **{
                    key: value
                    for key, value in state.items()
                    if key not in {"state_sha256", "updated_at"}
                },
                "state": STATE_DECISION_RECORDED,
                "decision": normalized_decision,
                "artifacts": artifacts,
            },
            expected_state_sha256=expected_state_sha256,
        )
        latest_path = advisory_root() / "_latest.json"
        if latest_path.exists():
            latest = read_json(
                latest_path,
                max_bytes=MAX_JSON_BYTES,
                require_single_link=True,
            )
            if latest.get("run_id") == run_id:
                latest.update(
                    {
                        "state": decided["state"],
                        "state_sha256": decided["state_sha256"],
                        "updated_at": utc_now(),
                    }
                )
                write_json_atomic(latest_path, latest)
    return _status_view(decided, run_dir)


__all__ = [
    "advisory_status",
    "finalize_advisory",
    "prepare_advisory",
    "receive_advisory_response",
    "record_advisory_decision",
    "resume_advisory_provider",
    "run_advisory",
]
