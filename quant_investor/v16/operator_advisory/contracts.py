"""Contracts and secure local I/O for the v16 advisory-only lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

INPUT_MANIFEST_SCHEMA = "v16.operator-advisory-input-manifest.v1"
FACTOR_BUNDLE_SCHEMA = "v16.operator-advisory-factor-bundle.v1"
PREPARED_EVIDENCE_SCHEMA = "v16.operator-advisory-branch-evidence.v1"
LLM_REQUEST_SCHEMA = "v16.operator-advisory-llm-request.v1"
LLM_RESPONSE_SCHEMA = "v16.operator-advisory-llm-response.v1"
REPORT_SCHEMA = "v16.operator-advisory-report.v1"
STATE_SCHEMA = "v16.operator-advisory-state.v1"
DECISION_SCHEMA = "v16.operator-advisory-decision-record.v1"

STATE_LLM_REQUEST_READY = "LLM_REQUEST_READY"
STATE_LLM_RESPONSE_RECEIVED = "LLM_RESPONSE_RECEIVED"
STATE_ADVISORY_COMPLETE = "ADVISORY_COMPLETE_AWAITING_HUMAN_DECISION"
STATE_DECISION_RECORDED = "ADVISORY_DECISION_RECORDED"

BRANCHES = ("quant", "fundamental", "macro", "llm")
BRANCH_SHARES = {name: 0.25 for name in BRANCHES}
DECISIONS = {"ACKNOWLEDGED", "DECLINED", "DEFERRED"}
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{7,79}$")

MAX_JSON_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_RATIONALE_CHARS = 500
MAX_RISK_CHARS = 200
MAX_EVIDENCE_IDS = 8

REPO_ROOT = Path(__file__).resolve().parents[3]
ADVISORY_ROOT = REPO_ROOT / "results" / "v16_operator_advisory"
INPUT_MANIFEST_PATH = Path(__file__).with_name("operator_input_manifest.v1.json")


class AdvisoryError(RuntimeError):
    exit_code = 2


class AdvisoryProviderError(AdvisoryError):
    exit_code = 3


class AdvisoryStateError(AdvisoryError):
    exit_code = 4


class AdvisorySideEffectError(AdvisoryError):
    exit_code = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"cn-v16-advisory-{stamp}-{uuid.uuid4().hex[:8]}"


def validate_run_id(value: str) -> str:
    run_id = str(value or "").strip().lower()
    if not RUN_ID_RE.fullmatch(run_id):
        raise AdvisoryError("invalid advisory run id")
    return run_id


def advisory_root() -> Path:
    root = Path(ADVISORY_ROOT)
    expected = REPO_ROOT / "results" / "v16_operator_advisory"
    if root != expected or not root.is_absolute():
        raise AdvisoryError("advisory output root is not the fixed repository path")
    for component in (REPO_ROOT, REPO_ROOT / "results", root):
        if os.path.lexists(component) and stat.S_ISLNK(os.lstat(component).st_mode):
            raise AdvisoryError("advisory output root symlink rejected")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    formal_root = REPO_ROOT / "results" / "v16"
    resolved = root.resolve(strict=True)
    if (
        not root.is_dir()
        or root.is_symlink()
        or resolved != root
        or resolved == formal_root
        or formal_root in resolved.parents
    ):
        raise AdvisoryError("advisory output root invalid")
    forbidden_roots = (
        formal_root,
        REPO_ROOT / "results" / "v15",
        REPO_ROOT / "results" / "strategy_records",
        REPO_ROOT / "data",
        REPO_ROOT / "portfolio_dashboard" / "generated",
    )
    for forbidden_root in forbidden_roots:
        if not os.path.lexists(forbidden_root):
            continue
        try:
            forbidden_resolved = forbidden_root.resolve(strict=True)
        except OSError as exc:
            raise AdvisoryError("forbidden control root cannot be resolved") from exc
        if (
            forbidden_resolved == resolved
            or forbidden_resolved in resolved.parents
            or resolved in forbidden_resolved.parents
        ):
            raise AdvisoryError("advisory output root aliases forbidden control tree")
    os.chmod(root, 0o700)
    return root


def run_directory(run_id: str, *, must_exist: bool = True) -> Path:
    root = advisory_root()
    path = (root / validate_run_id(run_id)).resolve()
    if path.parent != root:
        raise AdvisoryError("advisory run path escaped fixed root")
    if must_exist and (not path.exists() or not path.is_dir() or path.is_symlink()):
        raise AdvisoryError(f"advisory run not found: {run_id}")
    return path


def canonical_json_bytes(value: Any) -> bytes:
    _require_finite_json(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_json(value: Any, *, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise AdvisoryError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AdvisoryError(f"non-string JSON key at {path}")
            _require_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")
        return
    raise AdvisoryError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _pairs_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AdvisoryError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def read_json(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    require_single_link: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise AdvisoryError(f"symlink JSON input rejected: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise AdvisoryError(f"JSON input unavailable: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdvisoryError(f"JSON input is not regular: {source}")
        if require_single_link and before.st_nlink != 1:
            raise AdvisoryError(f"hard-linked JSON input rejected: {source}")
        limit = max_bytes if max_bytes is not None else max(before.st_size + 1, 1)
        if before.st_size > limit:
            raise AdvisoryError(f"JSON input exceeds size limit: {source}")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise AdvisoryError(f"JSON input exceeds size limit: {source}")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AdvisoryError(f"JSON input changed during read: {source}")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AdvisoryError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdvisoryError(f"invalid JSON input: {source}") from exc
    if not isinstance(payload, dict):
        raise AdvisoryError(f"JSON input must be one object: {source}")
    _require_finite_json(payload)
    return payload


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_symlink():
        raise AdvisoryError(f"symlink output rejected: {target}")
    raw = canonical_json_bytes(dict(payload))
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise AdvisoryStateError(f"immutable advisory file already exists: {target}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(target, 0o600)
    observed = file_sha256(target)
    expected = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise AdvisoryError(f"advisory file readback mismatch: {target}")
    return observed


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = canonical_json_bytes(dict(payload))
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    observed = file_sha256(target)
    if observed != hashlib.sha256(raw).hexdigest():
        raise AdvisoryError(f"advisory atomic readback mismatch: {target}")
    return observed


def state_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "state_sha256"}


def state_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(state_payload(state))


def load_state(run_id: str) -> dict[str, Any]:
    state = read_json(
        run_directory(run_id) / "state.json",
        max_bytes=MAX_JSON_BYTES,
        require_single_link=True,
    )
    if state.get("schema_version") != STATE_SCHEMA:
        raise AdvisoryStateError("advisory state schema mismatch")
    if state.get("run_id") != run_id or state.get("state_sha256") != state_sha256(state):
        raise AdvisoryStateError("advisory state hash mismatch")
    return state


def save_state(
    run_dir: Path,
    payload: Mapping[str, Any],
    *,
    expected_state_sha256: str | None = None,
) -> dict[str, Any]:
    target = run_dir / "state.json"
    if target.exists():
        current = read_json(target, max_bytes=MAX_JSON_BYTES, require_single_link=True)
        current_sha = state_sha256(current)
        if not expected_state_sha256 or current_sha != expected_state_sha256:
            raise AdvisoryStateError("advisory state CAS mismatch")
    elif expected_state_sha256:
        raise AdvisoryStateError("advisory state missing for CAS transition")
    state = {
        **dict(payload),
        "schema_version": STATE_SCHEMA,
        "run_id": run_dir.name,
        "updated_at": utc_now(),
    }
    state["state_sha256"] = state_sha256(state)
    write_json_atomic(target, state)
    readback = load_state(run_dir.name)
    if readback != state:
        raise AdvisoryStateError("advisory state readback mismatch")
    return state


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise AdvisoryError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def centered_average_rank(values: Any) -> Any:
    """Average-tie cross-sectional rank mapped to [-1, 1]."""

    import pandas as pd

    numeric = pd.to_numeric(values, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    valid = numeric.dropna()
    result = pd.Series(float("nan"), index=numeric.index, dtype=float)
    if valid.empty:
        return result
    if len(valid) == 1 or valid.nunique(dropna=False) == 1:
        result.loc[valid.index] = 0.0
        return result
    ranks = valid.rank(method="average")
    result.loc[valid.index] = 2.0 * (ranks - 1.0) / (len(valid) - 1.0) - 1.0
    return result


def unit_average_rank(values: Any) -> Any:
    """Average-tie cross-sectional rank mapped to [0, 1]."""

    return (centered_average_rank(values) + 1.0) / 2.0


_PROHIBITED_VALUE_PATTERNS = (
    re.compile(r"\bposterior\b", re.IGNORECASE),
    re.compile(r"\bprobabilit(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\bexpected[_ -]?alpha\b", re.IGNORECASE),
    re.compile(r"\bedge[_ -]?after[_ -]?costs\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|hold|avoid|sell)\b", re.IGNORECASE),
    re.compile(r"(?:买入|持有|回避|卖出)"),
)
_PROHIBITED_KEY_FRAGMENTS = (
    "posterior",
    "probability",
    "expected_alpha",
    "edge_after_costs",
    "target_weight",
    "target_shares",
    "capital_amount",
    "order_instruction",
    "execution_plan",
)


def validate_publishable(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in _PROHIBITED_KEY_FRAGMENTS):
                raise AdvisoryError(f"prohibited publishable key at {path}.{key}")
            validate_publishable(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_publishable(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _PROHIBITED_VALUE_PATTERNS
    ):
        raise AdvisoryError(f"prohibited publishable text at {path}")


def validate_llm_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    request_file_sha256: str,
    model_id: str,
    prompt_sha256: str,
    response_schema_sha256: str,
) -> dict[str, Any]:
    require_exact_keys(
        response,
        {
            "schema_version",
            "request_sha256",
            "model_id",
            "prompt_sha256",
            "response_schema_sha256",
            "reviews",
        },
        label="LLM response",
    )
    if response.get("schema_version") != LLM_RESPONSE_SCHEMA:
        raise AdvisoryError("LLM response schema mismatch")
    bindings = {
        "request_sha256": request_file_sha256,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "response_schema_sha256": response_schema_sha256,
    }
    for key, expected in bindings.items():
        if response.get(key) != expected:
            raise AdvisoryError(f"LLM response binding mismatch: {key}")
    request_items = request.get("items")
    reviews = response.get("reviews")
    if not isinstance(request_items, list) or not isinstance(reviews, list):
        raise AdvisoryError("LLM request/response items must be arrays")
    fact_ids_by_symbol: dict[str, set[str]] = {}
    for item in request_items:
        if not isinstance(item, Mapping):
            raise AdvisoryError("LLM request item invalid")
        symbol = str(item.get("symbol") or "")
        ids = item.get("fact_ids")
        if not symbol or not isinstance(ids, list) or symbol in fact_ids_by_symbol:
            raise AdvisoryError("LLM request symbol/fact identity invalid")
        fact_ids_by_symbol[symbol] = {str(value) for value in ids}
    validated: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, Mapping):
            raise AdvisoryError("LLM review item invalid")
        require_exact_keys(
            review,
            {"symbol", "raw_score", "confidence", "rationale", "evidence_ids", "risks"},
            label="LLM review item",
        )
        symbol = str(review.get("symbol") or "")
        if symbol not in fact_ids_by_symbol or symbol in validated:
            raise AdvisoryError(f"LLM review symbol invalid or duplicated: {symbol}")
        raw_score = review.get("raw_score")
        confidence = review.get("confidence")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise AdvisoryError(f"LLM raw score invalid: {symbol}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise AdvisoryError(f"LLM confidence invalid: {symbol}")
        if not math.isfinite(float(raw_score)) or not -1.0 <= float(raw_score) <= 1.0:
            raise AdvisoryError(f"LLM raw score out of range: {symbol}")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise AdvisoryError(f"LLM confidence out of range: {symbol}")
        rationale = str(review.get("rationale") or "").strip()
        if not rationale or len(rationale) > MAX_RATIONALE_CHARS:
            raise AdvisoryError(f"LLM rationale length invalid: {symbol}")
        evidence_ids = review.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) > MAX_EVIDENCE_IDS
            or len({str(value) for value in evidence_ids}) != len(evidence_ids)
            or not {str(value) for value in evidence_ids}.issubset(fact_ids_by_symbol[symbol])
        ):
            raise AdvisoryError(f"LLM evidence binding invalid: {symbol}")
        risks = review.get("risks")
        if (
            not isinstance(risks, list)
            or len(risks) > 5
            or any(not str(value).strip() or len(str(value)) > MAX_RISK_CHARS for value in risks)
        ):
            raise AdvisoryError(f"LLM risks invalid: {symbol}")
        validate_publishable(rationale, path=f"$.reviews[{symbol}].rationale")
        validate_publishable(risks, path=f"$.reviews[{symbol}].risks")
        validated[symbol] = {
            "symbol": symbol,
            "raw_score": float(raw_score),
            "confidence": float(confidence),
            "rationale": rationale,
            "evidence_ids": [str(value) for value in evidence_ids],
            "risks": [str(value).strip() for value in risks],
        }
    if set(validated) != set(fact_ids_by_symbol):
        raise AdvisoryError("LLM response symbol set mismatch")
    return {symbol: validated[symbol] for symbol in fact_ids_by_symbol}


__all__ = [name for name in globals() if name.startswith("STATE_")] + [
    "ADVISORY_ROOT",
    "BRANCHES",
    "BRANCH_SHARES",
    "DECISIONS",
    "DECISION_SCHEMA",
    "FACTOR_BUNDLE_SCHEMA",
    "INPUT_MANIFEST_PATH",
    "INPUT_MANIFEST_SCHEMA",
    "LLM_REQUEST_SCHEMA",
    "LLM_RESPONSE_SCHEMA",
    "MAX_JSON_BYTES",
    "MAX_ARTIFACT_BYTES",
    "PREPARED_EVIDENCE_SCHEMA",
    "REPORT_SCHEMA",
    "REPO_ROOT",
    "STATE_SCHEMA",
    "AdvisoryError",
    "AdvisoryProviderError",
    "AdvisorySideEffectError",
    "AdvisoryStateError",
    "advisory_root",
    "canonical_json_bytes",
    "canonical_sha256",
    "centered_average_rank",
    "file_sha256",
    "load_state",
    "make_run_id",
    "read_json",
    "run_directory",
    "save_state",
    "state_sha256",
    "unit_average_rank",
    "utc_now",
    "validate_llm_response",
    "validate_publishable",
    "validate_run_id",
    "write_json_atomic",
    "write_json_exclusive",
]
