"""Explicit-ref V15/V17 v4 daily gray comparison and close labels."""

from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Final, Mapping, Sequence

from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
)
from quant_investor.v17_v4_runtime.deep_v2 import DEEP_BUNDLE_V2
from quant_investor.v17_v4_runtime.shadow_runtime import (
    RESEARCH_FACTOR_EVIDENCE_MODE,
    SESSION_REF_VERSION,
    SHADOW_RUN_RESEARCH_VERSION,
    SHADOW_RUN_VERSION,
    read_shadow_session,
    require_strategy_path_id,
)
from quant_investor.v17_v4_runtime.source_storage import (
    SHADOW_ROOT,
    ExactReferenceReader,
    GovernedStore,
    SourceStorageSecurityError,
    canonical_governed_path,
)

SCHEMA_VERSION: Final = "cn_aggressive_v15_v17_v4_gray_comparison.v1"
LABEL_VERSION: Final = "myquant.v17.v4.gray-close-return-label.v1"
OUTPUT_JSON: Final = "v15_v17_v4_gray_comparison.json"
OUTPUT_MARKDOWN: Final = "v15_v17_v4_gray_comparison.md"
MARKET_POINTER_PATH: Final = Path("data/parquet/cn/_latest.json")
MINIMUM_FORWARD_SESSIONS: Final = 20
NO_AUTHORITY: Final = {
    "broker_authority": False,
    "execution_authority": False,
    "formal_research_publication_authority": False,
    "order_authority": False,
    "production_default": False,
    "trade_authority": False,
}


class V4GrayComparisonError(RuntimeError):
    """Raised internally when v4 gray evidence is not comparable."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _normalized_session(value: Any) -> str:
    rendered = str(value or "").replace("-", "").strip()
    return rendered if len(rendered) == 8 and rendered.isdigit() else ""


def _date_session(value: str) -> str:
    normalized = _normalized_session(value)
    if not normalized:
        raise V4GrayComparisonError("decision_session_invalid")
    return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:]}"


def _read_regular(path: Path, *, private: bool = False) -> bytes:
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or (private and stat.S_IMODE(before.st_mode) & 0o077)
        ):
            raise V4GrayComparisonError(f"unsafe_file:{path}")
        raw = path.read_bytes()
        after = os.lstat(path)
    except OSError as exc:
        raise V4GrayComparisonError(f"unreadable_file:{path}") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != after.st_size:
        raise V4GrayComparisonError(f"changed_while_reading:{path}")
    return raw


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise V4GrayComparisonError(f"invalid_json:{label}") from exc
    if type(value) is not dict:
        raise V4GrayComparisonError(f"invalid_json_root:{label}")
    return value


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_regular(path)
    return _json_object(raw, label=str(path)), _sha(raw)


def _csv_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = _read_regular(path)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise V4GrayComparisonError(f"invalid_csv_encoding:{path}") from exc
    return (
        [dict(row) for row in csv.DictReader(text.splitlines())],
        _sha(raw),
    )


def _symbols(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol and symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    return result


def _load_v15(run_dir: Path) -> dict[str, Any]:
    manifest, _ = _read_json(run_dir / "manifest.json")
    snapshot = manifest.get("data_snapshot")
    if not isinstance(snapshot, Mapping):
        raise V4GrayComparisonError("v15_data_snapshot_missing")
    decision_session = _normalized_session(snapshot.get("analysis_trade_date"))
    if not decision_session:
        raise V4GrayComparisonError("v15_decision_session_missing")
    candidate_rows, candidate_sha = _csv_rows(run_dir / "candidate_pool.csv")
    holdings_rows, holdings_sha = _csv_rows(run_dir / "holdings_review.csv")
    bindings = manifest.get("v17_v4_comparison_inputs")
    return {
        "artifact_refs": {
            "candidate_pool": {
                "path": "candidate_pool.csv",
                "sha256": candidate_sha,
            },
            "holdings_review": {
                "path": "holdings_review.csv",
                "sha256": holdings_sha,
            },
        },
        "candidate_symbols": _symbols(candidate_rows)[:12],
        "comparison_inputs": (dict(bindings) if isinstance(bindings, Mapping) else None),
        "decision_session": decision_session,
        "holding_symbols": _symbols(holdings_rows),
        "manifest": manifest,
    }


def _load_v4_ref(
    reader: ExactReferenceReader,
    reference: Mapping[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    try:
        raw = reader.read(
            str(reference["relative_path"]),
            str(reference["byte_sha256"]),
        )
        document = load_canonical_resource(
            raw,
            label=expected_version,
        )
        validated = load_canonical_artifact(
            raw,
            expected_version=expected_version,
        )
    except Exception as exc:
        raise V4GrayComparisonError(f"v4_artifact_readback:{expected_version}") from exc
    if (
        type(document) is not dict
        or validated.payload != document
        or document.get("semantic_sha256") != reference.get("semantic_sha256")
        or document.get("strategy_id") != reference.get("strategy_id")
        or _sha(raw) != reference.get("byte_sha256")
    ):
        raise V4GrayComparisonError(f"v4_artifact_binding:{expected_version}")
    return document


def _load_explicit_shadow(
    *,
    workspace_root: Path,
    session_ref_path: str,
    expected_session_ref_sha256: str,
) -> dict[str, Any]:
    reader = ExactReferenceReader(workspace_root)
    try:
        raw = reader.read(
            session_ref_path,
            expected_session_ref_sha256,
        )
        session = load_canonical_resource(
            raw,
            label=SESSION_REF_VERSION,
        )
        validate_artifact(session)
    except Exception as exc:
        raise V4GrayComparisonError("v4_session_ref_readback") from exc
    if type(session) is not dict:
        raise V4GrayComparisonError("v4_session_ref_root")
    status = read_shadow_session(
        str(workspace_root),
        strategy_id=str(session["strategy_id"]),
        decision_session=str(session["decision_session"]),
        expected_sha256=expected_session_ref_sha256,
    )
    if status["session_path"] != session_ref_path:
        raise V4GrayComparisonError("v4_session_ref_path_mismatch")
    run = status["shadow_run"]
    if (
        run["version"] not in {SHADOW_RUN_VERSION, SHADOW_RUN_RESEARCH_VERSION}
        or run["shadow_only"] is not True
        or run["formal_activation_eligible"] is not False
        or run["canary_evidence_eligible"] is not False
    ):
        raise V4GrayComparisonError("v4_shadow_authority")
    if run["version"] == SHADOW_RUN_RESEARCH_VERSION and (
        run.get("factor_evidence_mode") != RESEARCH_FACTOR_EVIDENCE_MODE
        or "research_factor_shadow_assertion_ref" not in run
    ):
        raise V4GrayComparisonError("v4_shadow_authority")
    fusion = _load_v4_ref(
        reader,
        run["fusion_top24_ref"],
        expected_version="myquant.v17.v4.fusion-top24.v1",
    )
    deep = _load_v4_ref(
        reader,
        run["deep_bundle_ref"],
        expected_version=DEEP_BUNDLE_V2,
    )
    quant = _load_v4_ref(
        reader,
        run["quant_branch_ref"],
        expected_version=(
            "myquant.v17.v4.research-quant-branch-output.v1"
        ),
    )
    fundamental = _load_v4_ref(
        reader,
        run["fundamental_branch_ref"],
        expected_version="myquant.v17.v4.branch-output.v1",
    )
    return {
        "deep": deep,
        "fundamental": fundamental,
        "fusion": fusion,
        "run": run,
        "session": session,
        "session_sha256": _sha(raw),
        "quant": quant,
    }


def _same_comparison_inputs(
    v15: Mapping[str, Any],
    run: Mapping[str, Any],
) -> tuple[bool, str | None]:
    bindings = v15.get("comparison_inputs")
    if not isinstance(bindings, Mapping):
        return False, "v15_v4_comparison_inputs_missing"
    if _normalized_session(bindings.get("decision_session")) != _normalized_session(
        run["decision_session"]
    ):
        return False, "decision_session_binding_mismatch"
    expected = dict(run["comparison_inputs"])
    observed = {
        key: bindings.get(key)
        for key in (
            "calendar_ref",
            "holdings_ref",
            "market_bars_ref",
            "source_closure_ref",
        )
    }
    if observed != expected:
        return False, "v15_v4_input_binding_mismatch"
    return True, None


def _holding_diagnostics(
    *,
    holding_symbols: Sequence[str],
    quant: Mapping[str, Any],
    fundamental: Mapping[str, Any],
    fusion: Mapping[str, Any],
    deep: Mapping[str, Any],
) -> list[dict[str, Any]]:
    quant_scores = {row["symbol"]: row["score"] for row in quant["score_rows"]}
    fundamental_scores = {row["symbol"]: row["score"] for row in fundamental["score_rows"]}
    top24 = {row["symbol"]: row["rank"] for row in fusion["rows"]}
    deep_rows = {row["symbol"]: row for row in deep["rows"]}
    return [
        {
            "deep_buy_veto": (deep_rows.get(symbol, {}).get("buy_veto")),
            "fundamental_score": fundamental_scores.get(symbol),
            "quant_score": quant_scores.get(symbol),
            "symbol": symbol,
            "top24_rank": top24.get(symbol),
        }
        for symbol in holding_symbols
    ]


def _render_markdown(document: Mapping[str, Any]) -> str:
    effect = document["effect_evaluation"]
    metrics = document["metrics"]
    blockers = document["blockers"]
    lines = [
        "# V15 / V17 v4 日度灰度比较",
        "",
        f"- 状态：`{document['status']}`",
        f"- 可比性：`{document['classification']}`",
        f"- 决策日：`{document['decision_session'] or 'UNAVAILABLE'}`",
        f"- 效果结论：`{effect['verdict']}`",
        (
            "- 候选交集："
            f"{metrics['candidate_overlap_count']}；"
            f"V17 v4 Deep BUY_VETO："
            f"{metrics['v17_deep_buy_veto_count']}"
        ),
        ("- 1/5/20 日诊断标签数：" f"{effect['mature_label_count']}"),
    ]
    if blockers:
        lines.extend(
            [
                "",
                "## 阻断项",
                "",
                *[f"- `{blocker}`" for blocker in blockers],
            ]
        )
    lines.extend(
        [
            "",
            ("V17 v4 仅为 Shadow；本比较和 close-return 标签" "均不构成正式研究发布或交易权限。"),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _attach_v15(
    run_dir: Path,
    *,
    document: Mapping[str, Any],
    json_sha256: str,
    markdown_sha256: str,
) -> None:
    reference = {
        "classification": document["classification"],
        "json_path": OUTPUT_JSON,
        "json_sha256": json_sha256,
        "markdown_path": OUTPUT_MARKDOWN,
        "markdown_sha256": markdown_sha256,
        "production_authority": False,
        "schema_version": SCHEMA_VERSION,
        "status": document["status"],
    }
    for name in ("manifest.json", "market_snapshot.json"):
        path = run_dir / name
        payload, _ = _read_json(path)
        payload["v17_v4_gray_comparison"] = reference
        _atomic_write(
            path,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )
    report_path = run_dir / "analysis_report.md"
    report = _read_regular(report_path).decode("utf-8")
    heading = "## V15 / V17 v4 日度灰度比较"
    section = _render_markdown(document).replace(
        "# V15 / V17 v4 日度灰度比较",
        heading,
        1,
    )
    if heading not in report:
        report = report.rstrip() + "\n\n" + section
    _atomic_write(report_path, report.encode("utf-8"))


class _GrayLabelWriter(GovernedStore):
    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        if (
            parts[:3]
            != (
                "results",
                "v17_v4_shadow",
                "gray_labels",
            )
            or len(parts) != 6
        ):
            raise SourceStorageSecurityError("path is outside v4 gray label root")
        require_strategy_path_id(parts[3])
        _date_session(parts[4])
        if parts[5] not in {
            "horizon-1.json",
            "horizon-5.json",
            "horizon-20.json",
        }:
            raise SourceStorageSecurityError("v4 gray label horizon is invalid")
        return path


def _label_path(
    strategy_id: str,
    origin_session: str,
    horizon: int,
) -> str:
    return str(
        SHADOW_ROOT
        / "gray_labels"
        / require_strategy_path_id(strategy_id)
        / _date_session(origin_session)
        / f"horizon-{horizon}.json"
    )


def _frame_close_map(result: Any) -> dict[str, float]:
    frame = getattr(result, "frame", None)
    if frame is None or getattr(frame, "empty", True):
        return {}
    values: dict[str, float] = {}
    for row in frame[["trade_date", "close"]].itertuples(index=False):
        session = _normalized_session(row.trade_date)
        try:
            close = float(row.close)
        except (TypeError, ValueError):
            continue
        if session and close > 0:
            values[session] = close
    return values


def _equal_weight_return(
    close_maps: Mapping[str, Mapping[str, float]],
    symbols: Sequence[str],
    *,
    origin: str,
    target: str,
) -> float | None:
    values: list[float] = []
    for symbol in symbols:
        origin_close = close_maps.get(symbol, {}).get(origin)
        target_close = close_maps.get(symbol, {}).get(target)
        if not origin_close or not target_close:
            return None
        values.append(target_close / origin_close - 1)
    return sum(values) / len(values) if values else None


def _month_parts(
    *,
    workspace_root: Path,
    pointer: Mapping[str, Any],
    origin: str,
    target: str,
) -> list[dict[str, str]]:
    root_value = Path(str(pointer.get("table_root") or ""))
    table_root = root_value if root_value.is_absolute() else workspace_root / root_value
    cursor = date(
        int(origin[:4]),
        int(origin[4:6]),
        1,
    )
    end = date(int(target[:4]), int(target[4:6]), 1)
    refs: list[dict[str, str]] = []
    while cursor <= end:
        path = table_root / f"year={cursor.year}" / f"month={cursor.month:02d}" / "part.parquet"
        raw = _read_regular(path)
        refs.append({"path": str(path), "sha256": _sha(raw)})
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return refs


def _write_mature_labels(
    *,
    workspace_root: Path,
    base_dir: Path,
    current_session: str,
    market_pointer_path: Path,
    strategy_id: str,
) -> list[dict[str, Any]]:
    pointer, pointer_sha = _read_json(market_pointer_path)
    manifest_value = Path(str(pointer.get("manifest_path") or ""))
    manifest_path = (
        manifest_value if manifest_value.is_absolute() else workspace_root / manifest_value
    )
    manifest_raw = _read_regular(manifest_path)
    reader = MarketDataReader(
        market="CN",
        data_root=market_pointer_path.parent.parent.parent,
    )
    writer = _GrayLabelWriter(workspace_root)
    writer.initialize()
    results: list[dict[str, Any]] = []
    for run_dir in sorted(base_dir.iterdir()):
        comparison_path = run_dir / OUTPUT_JSON
        if not comparison_path.is_file():
            continue
        comparison, comparison_sha = _read_json(comparison_path)
        origin = _normalized_session(comparison.get("decision_session"))
        sets = comparison.get("selection_sets")
        if (
            comparison.get("schema_version") != SCHEMA_VERSION
            or comparison.get("classification") != "COMPARABLE"
            or comparison.get("strategy_id") != strategy_id
            or not origin
            or origin >= current_session
            or not isinstance(sets, Mapping)
        ):
            continue
        v15_symbols = [str(value) for value in sets.get("v15_candidates", [])]
        v17_symbols = [str(value) for value in sets.get("v17_top24", [])]
        if not v15_symbols or len(v17_symbols) != 24:
            continue
        all_symbols = sorted(set(v15_symbols) | set(v17_symbols))
        try:
            reads = reader.read_symbol_frames(
                all_symbols,
                start_date=origin,
                end_date=current_session,
                columns=["symbol", "trade_date", "close"],
            )
        except Exception:
            continue
        close_maps = {symbol: _frame_close_map(reads.get(symbol)) for symbol in all_symbols}
        sessions = sorted(
            {
                session
                for values in close_maps.values()
                for session in values
                if origin <= session <= current_session
            }
        )
        if origin not in sessions:
            continue
        origin_index = sessions.index(origin)
        for horizon in (1, 5, 20):
            label_path = _label_path(
                strategy_id,
                origin,
                horizon,
            )
            existing = writer.read_optional(label_path)
            if existing is not None:
                results.append(
                    {
                        "byte_sha256": existing.byte_sha256,
                        "created": False,
                        "relative_path": label_path,
                    }
                )
                continue
            target_index = origin_index + horizon
            if target_index >= len(sessions):
                continue
            target = sessions[target_index]
            v15_return = _equal_weight_return(
                close_maps,
                v15_symbols,
                origin=origin,
                target=target,
            )
            v17_return = _equal_weight_return(
                close_maps,
                v17_symbols,
                origin=origin,
                target=target,
            )
            if v15_return is None or v17_return is None:
                continue
            try:
                parts = _month_parts(
                    workspace_root=workspace_root,
                    pointer=pointer,
                    origin=origin,
                    target=target,
                )
            except V4GrayComparisonError:
                continue
            label = seal_semantic(
                {
                    "authority": dict(NO_AUTHORITY),
                    "corporate_actions": "UNAVAILABLE",
                    "created_at_session": current_session,
                    "delisting_treatment": "UNAVAILABLE",
                    "horizon_sessions": horizon,
                    "label_id": (f"{strategy_id}-{origin}-h{horizon}"),
                    "label_kind": "CLOSE_RETURN_DIAGNOSTIC_ONLY",
                    "market_evidence": {
                        "market_pointer": {
                            "path": str(market_pointer_path),
                            "sha256": pointer_sha,
                        },
                        "part_parquet_refs": parts,
                        "snapshot_manifest": {
                            "path": str(manifest_path),
                            "sha256": _sha(manifest_raw),
                        },
                    },
                    "origin_comparison_ref": {
                        "path": str(comparison_path),
                        "sha256": comparison_sha,
                    },
                    "origin_session": origin,
                    "performance_conclusion_eligible": False,
                    "protocol_version": "myquant.v17.v4",
                    "strategy_id": strategy_id,
                    "target_session": target,
                    "total_return": False,
                    "v15_equal_weight_close_return": (format(v15_return, ".12g")),
                    "v17_equal_weight_close_return": (format(v17_return, ".12g")),
                    "v17_minus_v15_close_return": (format(v17_return - v15_return, ".12g")),
                    "version": LABEL_VERSION,
                }
            )
            raw = canonical_resource_bytes(label)
            write = writer.write_exact_once(label_path, raw)
            results.append(
                {
                    "byte_sha256": write.byte_sha256,
                    "created": write.created,
                    "relative_path": label_path,
                }
            )
    return results


def _historical_strategies(base_dir: Path) -> set[str]:
    result: set[str] = set()
    if not base_dir.is_dir():
        return result
    for run_dir in sorted(base_dir.iterdir()):
        path = run_dir / OUTPUT_JSON
        if not path.is_file():
            continue
        try:
            document, _ = _read_json(path)
            strategy = require_strategy_path_id(document.get("strategy_id"))
        except Exception:
            continue
        if (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("classification") == "COMPARABLE"
        ):
            result.add(strategy)
    return result


def _existing_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document, digest = _read_json(path)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise V4GrayComparisonError("existing_v4_gray_schema_mismatch")
    return {
        "authority": dict(NO_AUTHORITY),
        "blockers": list(document.get("blockers") or []),
        "classification": document["classification"],
        "decision_session": document["decision_session"],
        "effect_verdict": document["effect_evaluation"]["verdict"],
        "metrics": dict(document["metrics"]),
        "path": str(path),
        "sha256": digest,
        "status": document["status"],
    }


def run_daily_v4_gray_comparison(
    *,
    run_dir: str | Path,
    workspace_root: str | Path,
    shadow_session_ref_path: str | None,
    expected_shadow_session_ref_sha256: str | None,
    market_pointer_path: str | Path,
    pointer_sha256_before_v15: str,
    pointer_sha256_after_v15: str,
    minimum_forward_sessions: int = MINIMUM_FORWARD_SESSIONS,
) -> dict[str, Any]:
    """Write one immutable V15/v4 gray record without v3 discovery."""

    target = Path(run_dir).resolve()
    root = Path(workspace_root).resolve()
    output_path = target / OUTPUT_JSON
    existing = _existing_result(output_path)
    if existing is not None:
        return existing
    pointer_path = Path(market_pointer_path)
    blockers: list[str] = []
    classification = "NON_COMPARABLE"
    status = "GRAY_UNAVAILABLE"
    v15: dict[str, Any] | None = None
    shadow: dict[str, Any] | None = None
    pointer_sha = ""
    try:
        if minimum_forward_sessions < 5:
            raise V4GrayComparisonError("minimum_forward_sessions_below_5")
        pointer, pointer_sha = _read_json(pointer_path)
        if pointer_sha != pointer_sha256_before_v15 or pointer_sha != pointer_sha256_after_v15:
            raise V4GrayComparisonError("market_pointer_drift_during_v15_run")
        v15 = _load_v15(target)
        if (
            _normalized_session(pointer.get("latest_complete_trade_date"))
            != v15["decision_session"]
        ):
            raise V4GrayComparisonError("v15_market_pointer_session_mismatch")
        if not shadow_session_ref_path or not expected_shadow_session_ref_sha256:
            raise V4GrayComparisonError("explicit_v4_session_ref_pair_missing")
        shadow = _load_explicit_shadow(
            workspace_root=root,
            session_ref_path=shadow_session_ref_path,
            expected_session_ref_sha256=(expected_shadow_session_ref_sha256),
        )
        if _normalized_session(shadow["run"]["decision_session"]) != v15["decision_session"]:
            raise V4GrayComparisonError("v15_v4_decision_session_mismatch")
        inputs_match, blocker = _same_comparison_inputs(
            v15,
            shadow["run"],
        )
        if not inputs_match:
            raise V4GrayComparisonError(str(blocker or "v15_v4_input_binding_mismatch"))
        classification = "COMPARABLE"
        status = "GRAY_COMPARISON_COMPLETE"
    except V4GrayComparisonError as exc:
        blockers.append(str(exc))

    decision_session = (
        v15["decision_session"]
        if v15
        else _normalized_session(shadow["run"]["decision_session"] if shadow else "")
    )
    if classification == "COMPARABLE" and v15 and shadow:
        top24 = [row["symbol"] for row in shadow["fusion"]["rows"]]
        diagnostics = _holding_diagnostics(
            holding_symbols=v15["holding_symbols"],
            quant=shadow["quant"],
            fundamental=shadow["fundamental"],
            fusion=shadow["fusion"],
            deep=shadow["deep"],
        )
        overlap = sorted(set(v15["candidate_symbols"]) & set(top24))
        metrics = {
            "candidate_overlap_count": len(overlap),
            "v15_candidate_count": len(v15["candidate_symbols"]),
            "v15_holding_count": len(v15["holding_symbols"]),
            "v17_deep_buy_veto_count": sum(
                row["buy_veto"] is True for row in shadow["deep"]["rows"]
            ),
            "v17_top24_count": len(top24),
        }
    else:
        top24 = []
        diagnostics = []
        metrics = {
            "candidate_overlap_count": 0,
            "v15_candidate_count": (len(v15["candidate_symbols"]) if v15 else 0),
            "v15_holding_count": (len(v15["holding_symbols"]) if v15 else 0),
            "v17_deep_buy_veto_count": 0,
            "v17_top24_count": 0,
        }
    labels: list[dict[str, Any]] = []
    if (
        v15
        and decision_session
        and pointer_sha
        and pointer_sha == pointer_sha256_before_v15
        and pointer_sha == pointer_sha256_after_v15
    ):
        strategies = _historical_strategies(target.parent)
        try:
            for strategy in sorted(strategies):
                labels.extend(
                    _write_mature_labels(
                        workspace_root=root,
                        base_dir=target.parent,
                        current_session=decision_session,
                        market_pointer_path=pointer_path,
                        strategy_id=strategy,
                    )
                )
        except Exception:
            blockers.append("v4_gray_label_refresh_failed")
    verdict = (
        "V15_V17_V4_CLOSE_RETURN_DIAGNOSTIC_AVAILABLE"
        if len(labels) >= minimum_forward_sessions
        else "NO_V15_V17_V4_PERFORMANCE_CONCLUSION"
    )
    document: dict[str, Any] = {
        "authority": dict(NO_AUTHORITY),
        "blockers": blockers,
        "canary_evidence_eligible": False,
        "classification": classification,
        "comparison_contract": {
            "explicit_v4_session_ref_required": True,
            "same_calendar_bytes_required": True,
            "same_decision_session_required": True,
            "same_holdings_bytes_required": True,
            "same_market_bars_bytes_required": True,
            "same_source_closure_bytes_required": True,
            "v15_is_authoritative": True,
            "v17_v4_is_shadow_only": True,
        },
        "decision_session": decision_session,
        "effect_evaluation": {
            "mature_label_count": len(labels),
            "minimum_forward_sessions": minimum_forward_sessions,
            "performance_conclusion_eligible": False,
            "verdict": verdict,
        },
        "holding_branch_diagnostics": diagnostics,
        "inputs": {
            "market_pointer": {
                "path": str(pointer_path),
                "sha256": pointer_sha,
                "sha256_after_v15": pointer_sha256_after_v15,
                "sha256_before_v15": pointer_sha256_before_v15,
            },
            "v15_record": (
                {
                    "comparison_inputs": v15["comparison_inputs"],
                    "stable_artifact_refs": v15["artifact_refs"],
                }
                if v15
                else None
            ),
            "v17_v4_session_ref": (
                {
                    "path": shadow_session_ref_path,
                    "sha256": shadow["session_sha256"],
                    "shadow_run_ref": shadow["session"]["shadow_run_ref"],
                }
                if shadow
                else None
            ),
        },
        "label_refs": labels,
        "metrics": metrics,
        "historical_policy_eligible": False,
        "observation_only": True,
        "production_default": "v15",
        "schema_version": SCHEMA_VERSION,
        "selection_sets": {
            "v15_candidates": (list(v15["candidate_symbols"]) if v15 else []),
            "v15_holdings": (list(v15["holding_symbols"]) if v15 else []),
            "v17_top24": top24,
        },
        "side_effect_attestation": {
            "broker_calls": 0,
            "execution_calls": 0,
            "llm_control_calls": 0,
            "order_calls": 0,
            "provider_calls": 0,
            "selector_writes": 0,
            "trade_calls": 0,
        },
        "status": status,
        "strategy_id": (shadow["run"]["strategy_id"] if shadow else "UNCONFIRMED"),
    }
    json_raw = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    markdown_raw = _render_markdown(document).encode("utf-8")
    markdown_path = target / OUTPUT_MARKDOWN
    _atomic_write(output_path, json_raw)
    _atomic_write(markdown_path, markdown_raw)
    raw_dir = target / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, raw_dir / OUTPUT_JSON)
    shutil.copy2(markdown_path, raw_dir / OUTPUT_MARKDOWN)
    json_sha = _sha(json_raw)
    markdown_sha = _sha(markdown_raw)
    _attach_v15(
        target,
        document=document,
        json_sha256=json_sha,
        markdown_sha256=markdown_sha,
    )
    return {
        "authority": dict(NO_AUTHORITY),
        "blockers": blockers,
        "classification": classification,
        "decision_session": decision_session,
        "effect_verdict": verdict,
        "metrics": metrics,
        "path": str(output_path),
        "report_path": str(markdown_path),
        "report_sha256": markdown_sha,
        "sha256": json_sha,
        "status": status,
    }


__all__ = [
    "LABEL_VERSION",
    "MARKET_POINTER_PATH",
    "MINIMUM_FORWARD_SESSIONS",
    "NO_AUTHORITY",
    "OUTPUT_JSON",
    "OUTPUT_MARKDOWN",
    "SCHEMA_VERSION",
    "V4GrayComparisonError",
    "run_daily_v4_gray_comparison",
]
