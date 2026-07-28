"""Fail-closed V15/V17 daily gray comparison.

The daily review remains a V15 production surface.  This module only discovers
an already completed V17 v3 model-only shadow for the same decision session and
the same canonical market-pointer bytes.  It never runs a provider, publishes a
V17 formal result, or grants execution authority.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Final, Mapping, Sequence

from quant_investor.market.market_data_reader import MarketDataReader

SCHEMA_VERSION: Final = "cn_aggressive_v15_v17_gray_comparison.v1"
V17_SUMMARY_VERSION: Final = "myquant.v17.v3.current-shadow-run-summary.v1"
V17_FUSION_VERSION: Final = "myquant.v17.v3.fusion-output.v1"
DEFAULT_V17_WORKSPACE_ROOT: Final = Path("data/private/v17_v3_workspaces")
MARKET_POINTER_PATH: Final = Path("data/parquet/cn/_latest.json")
OUTPUT_JSON: Final = "v15_v17_gray_comparison.json"
OUTPUT_MARKDOWN: Final = "v15_v17_gray_comparison.md"
MINIMUM_FORWARD_SESSIONS: Final = 20
NO_AUTHORITY: Final = {
    "broker_authority": False,
    "execution_authority": False,
    "formal_research_publication_authority": False,
    "order_authority": False,
    "production_default": False,
    "trade_authority": False,
}


class GrayComparisonError(RuntimeError):
    """Raised internally when gray evidence cannot be admitted."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _normalized_session(value: Any) -> str:
    rendered = str(value or "").replace("-", "").strip()
    return rendered if len(rendered) == 8 and rendered.isdigit() else ""


def _read_regular(path: Path, *, private: bool = False) -> bytes:
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
            raise GrayComparisonError(f"unsafe_file:{path}")
        if private and stat.S_IMODE(before.st_mode) & 0o077:
            raise GrayComparisonError(f"private_file_mode:{path}")
        raw = path.read_bytes()
        after = os.lstat(path)
    except OSError as exc:
        raise GrayComparisonError(f"unreadable_file:{path}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(raw) != after.st_size:
        raise GrayComparisonError(f"changed_while_reading:{path}")
    return raw


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise GrayComparisonError(f"invalid_json:{label}") from exc
    if type(value) is not dict:
        raise GrayComparisonError(f"invalid_json_root:{label}")
    return value


def _read_json(path: Path, *, private: bool = False) -> tuple[dict[str, Any], str]:
    raw = _read_regular(path, private=private)
    return _json_object(raw, label=str(path)), _sha(raw)


def _csv_document(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = _read_regular(path)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GrayComparisonError(f"invalid_csv_encoding:{path}") from exc
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


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise GrayComparisonError(f"path_outside_root:{path}") from exc


def _authority_is_zero(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(NO_AUTHORITY).issubset(value)
        and all(value.get(name) is expected for name, expected in NO_AUTHORITY.items())
    )


def _discover_v17_summary(
    root: Path,
    *,
    decision_session: str,
    market_pointer_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    if not root.is_absolute():
        root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise GrayComparisonError("v17_workspace_root_unavailable")
    candidates: list[tuple[str, Path, dict[str, Any], str]] = []
    for workspace in sorted(root.iterdir()):
        if not workspace.is_dir() or workspace.is_symlink():
            continue
        runs_root = workspace / "data/private/v17_v3_runs"
        if not runs_root.is_dir() or runs_root.is_symlink():
            continue
        for run_dir in sorted(runs_root.iterdir()):
            path = run_dir / "run_summary.json"
            if not run_dir.is_dir() or run_dir.is_symlink() or not path.is_file():
                continue
            try:
                summary, byte_sha = _read_json(path, private=True)
            except GrayComparisonError:
                continue
            if (
                summary.get("version") != V17_SUMMARY_VERSION
                or summary.get("status") != "SHADOW_COMPLETE"
                or _normalized_session(summary.get("decision_session")) != decision_session
                or not _authority_is_zero(summary.get("authority"))
            ):
                continue
            bindings = summary.get("source_bindings")
            if not isinstance(bindings, Mapping):
                continue
            if bindings.get("market_pointer_sha256") != market_pointer_sha256:
                continue
            candidates.append((str(summary.get("cutoff") or ""), path, summary, byte_sha))
    if not candidates:
        raise GrayComparisonError("same_session_v17_shadow_not_found")
    _, path, summary, byte_sha = sorted(candidates, key=lambda row: row[0])[-1]
    return path, summary, byte_sha


def _load_v17_fusion(
    summary_path: Path,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    run_id = str(summary.get("run_id") or "")
    if not run_id or summary_path.parent.name != run_id:
        raise GrayComparisonError("v17_run_identity_mismatch")
    fusion_path = summary_path.parent / "fusion_output.json"
    fusion, fusion_sha = _read_json(fusion_path, private=True)
    if (
        fusion.get("version") != V17_FUSION_VERSION
        or fusion.get("run_id") != run_id
        or fusion.get("status") != "READY"
        or not _authority_is_zero(fusion.get("authority"))
    ):
        raise GrayComparisonError("v17_fusion_contract_mismatch")
    return fusion, fusion_sha, fusion_path


def _load_v15_inputs(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest, manifest_sha = _read_json(manifest_path)
    data_snapshot = manifest.get("data_snapshot")
    if not isinstance(data_snapshot, Mapping):
        raise GrayComparisonError("v15_data_snapshot_missing")
    decision_session = _normalized_session(data_snapshot.get("analysis_trade_date"))
    if not decision_session:
        raise GrayComparisonError("v15_decision_session_missing")
    candidate_path = run_dir / "candidate_pool.csv"
    holdings_path = run_dir / "holdings_review.csv"
    pnl_path = run_dir / "pnl_summary.csv"
    candidate_rows, candidate_sha = _csv_document(candidate_path)
    holdings_rows, holdings_sha = _csv_document(holdings_path)
    pnl_rows, pnl_sha = _csv_document(pnl_path)
    if len(pnl_rows) != 1:
        raise GrayComparisonError("v15_pnl_summary_shape")
    readiness_reference = manifest.get("v15_run_readiness")
    readiness_input: dict[str, Any] | None = None
    if isinstance(readiness_reference, Mapping):
        readiness_relative = str(readiness_reference.get("path") or "")
        readiness_path = run_dir / readiness_relative
        if (
            readiness_relative
            and ".." not in Path(readiness_relative).parts
            and readiness_path.is_file()
        ):
            readiness_raw = _read_regular(readiness_path)
            readiness_sha = _sha(readiness_raw)
            if readiness_sha == readiness_reference.get("sha256"):
                readiness_input = {
                    "path": readiness_relative,
                    "sha256": readiness_sha,
                }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_pre_gray_sha256": manifest_sha,
        "decision_session": decision_session,
        "snapshot_id": str(data_snapshot.get("completeness", {}).get("snapshot_id") or ""),
        "candidate_symbols": _symbols(candidate_rows)[:12],
        "holding_symbols": _symbols(holdings_rows),
        "holdings_rows": holdings_rows,
        "pnl_row": pnl_rows[0],
        "artifact_refs": {
            "candidate_pool": {
                "path": "candidate_pool.csv",
                "sha256": candidate_sha,
            },
            "holdings_review": {
                "path": "holdings_review.csv",
                "sha256": holdings_sha,
            },
            "pnl_summary": {
                "path": "pnl_summary.csv",
                "sha256": pnl_sha,
            },
            "v15_run_readiness": readiness_input,
        },
    }


def _comparison_history(base_dir: Path, current_session: str) -> list[str]:
    sessions: set[str] = set()
    if not base_dir.is_dir():
        return []
    for run_dir in sorted(base_dir.iterdir()):
        path = run_dir / OUTPUT_JSON
        if not run_dir.is_dir() or run_dir.is_symlink() or not path.is_file():
            continue
        try:
            document, _ = _read_json(path)
        except GrayComparisonError:
            continue
        session = _normalized_session(document.get("decision_session"))
        if (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("classification") == "COMPARABLE"
            and session
            and session <= current_session
        ):
            sessions.add(session)
    sessions.add(current_session)
    return sorted(sessions)


def _previous_comparisons(
    base_dir: Path,
    current_session: str,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    if not base_dir.is_dir():
        return documents
    for run_dir in sorted(base_dir.iterdir()):
        path = run_dir / OUTPUT_JSON
        if not run_dir.is_dir() or run_dir.is_symlink() or not path.is_file():
            continue
        try:
            document, _ = _read_json(path)
        except GrayComparisonError:
            continue
        session = _normalized_session(document.get("decision_session"))
        sets = document.get("selection_sets")
        if (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("classification") == "COMPARABLE"
            and isinstance(sets, Mapping)
            and session
            and session < current_session
        ):
            documents.append(document)
    return documents


def _frame_close_map(result: Any) -> dict[str, float]:
    frame = getattr(result, "frame", None)
    if frame is None or getattr(frame, "empty", True):
        return {}
    if "trade_date" not in frame.columns or "close" not in frame.columns:
        return {}
    values: dict[str, float] = {}
    for row in frame[["trade_date", "close"]].itertuples(index=False):
        session = _normalized_session(row.trade_date)
        close = _float(row.close, default=-1.0)
        if session and close > 0:
            values[session] = close
    return values


def _set_forward_return(
    close_maps: Mapping[str, Mapping[str, float]],
    symbols: Sequence[str],
    *,
    origin_session: str,
    target_session: str,
) -> float | None:
    returns: list[float] = []
    for symbol in symbols:
        values = close_maps.get(symbol, {})
        origin = _float(values.get(origin_session), default=-1.0)
        eligible_target_sessions = [
            session for session in values if origin_session < session <= target_session
        ]
        if origin <= 0 or not eligible_target_sessions:
            return None
        target = _float(
            values[max(eligible_target_sessions)],
            default=-1.0,
        )
        if target <= 0:
            return None
        returns.append(target / origin - 1.0)
    return sum(returns) / len(returns) if returns else None


def _mature_forward_outcomes(
    *,
    base_dir: Path,
    current_session: str,
    market_pointer_path: Path,
) -> list[dict[str, Any]]:
    previous = _previous_comparisons(base_dir, current_session)
    if not previous:
        return []
    data_root = market_pointer_path.parent.parent.parent
    reader = MarketDataReader(market="CN", data_root=data_root)
    outcomes: list[dict[str, Any]] = []
    for document in previous:
        origin = _normalized_session(document.get("decision_session"))
        selection_sets = document["selection_sets"]
        v15_symbols = [
            str(symbol) for symbol in selection_sets.get("v15_candidates", []) if str(symbol)
        ]
        v17_symbols = [str(symbol) for symbol in selection_sets.get("v17_top24", []) if str(symbol)]
        if (
            not origin
            or not (1 <= len(v15_symbols) <= 12)
            or len(set(v15_symbols)) != len(v15_symbols)
            or len(v17_symbols) != 24
            or len(set(v17_symbols)) != 24
        ):
            continue
        all_symbols = sorted(set(v15_symbols) | set(v17_symbols))
        try:
            reads = reader.read_symbol_frames(
                all_symbols,
                start_date=origin,
                end_date=current_session,
                columns=["symbol", "trade_date", "close"],
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        close_maps = {symbol: _frame_close_map(reads.get(symbol)) for symbol in all_symbols}
        sessions = sorted(
            {session for values in close_maps.values() for session in values if session >= origin}
        )
        if origin not in sessions:
            continue
        origin_index = sessions.index(origin)
        for horizon in (1, 5, 20):
            target_index = origin_index + horizon
            if target_index >= len(sessions):
                continue
            target_session = sessions[target_index]
            v15_return = _set_forward_return(
                close_maps,
                v15_symbols,
                origin_session=origin,
                target_session=target_session,
            )
            v17_return = _set_forward_return(
                close_maps,
                v17_symbols,
                origin_session=origin,
                target_session=target_session,
            )
            if v15_return is None or v17_return is None:
                continue
            outcomes.append(
                {
                    "origin_session": origin,
                    "target_session": target_session,
                    "horizon_sessions": horizon,
                    "v15_equal_weight_return": _round(v15_return),
                    "v17_equal_weight_return": _round(v17_return),
                    "v17_minus_v15_return": _round(v17_return - v15_return),
                    "v15_symbol_count": len(v15_symbols),
                    "v17_symbol_count": len(v17_symbols),
                    "interpretation_scope": "RANK_SET_EQUAL_WEIGHT_DIAGNOSTIC_ONLY",
                }
            )
    return sorted(
        outcomes,
        key=lambda row: (
            row["origin_session"],
            row["horizon_sessions"],
        ),
    )


def _aggregate_forward_outcomes(
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for horizon in (1, 5, 20):
        rows = [row for row in outcomes if int(row.get("horizon_sessions") or 0) == horizon]
        differences = [_float(row.get("v17_minus_v15_return")) for row in rows]
        v15_returns = [_float(row.get("v15_equal_weight_return")) for row in rows]
        v17_returns = [_float(row.get("v17_equal_weight_return")) for row in rows]
        ordered = sorted(differences)
        median = (
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (
                (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
                if ordered
                else 0.0
            )
        )
        count = len(rows)
        aggregates[str(horizon)] = {
            "paired_origin_count": count,
            "v15_mean_equal_weight_return": (_round(sum(v15_returns) / count) if count else None),
            "v17_mean_equal_weight_return": (_round(sum(v17_returns) / count) if count else None),
            "v17_minus_v15_mean_return": (_round(sum(differences) / count) if count else None),
            "v17_minus_v15_median_return": _round(median) if count else None,
            "v17_outperformance_rate": (
                _round(sum(value > 0 for value in differences) / count) if count else None
            ),
        }
    return aggregates


def _round(value: float) -> float:
    return round(value, 8)


def _build_metrics(
    v15: Mapping[str, Any],
    summary: Mapping[str, Any],
    fusion: Mapping[str, Any],
) -> dict[str, Any]:
    v15_candidates = list(v15["candidate_symbols"])
    v15_holdings = list(v15["holding_symbols"])
    v17_rows = summary.get("top24")
    if not isinstance(v17_rows, list):
        raise GrayComparisonError("v17_top24_missing")
    v17_symbols = _symbols([row for row in v17_rows if isinstance(row, Mapping)])
    if len(v17_symbols) != 24:
        raise GrayComparisonError("v17_top24_shape")
    selected_symbols = [
        str(symbol).strip().upper()
        for symbol in fusion.get("selected_symbols", [])
        if str(symbol).strip()
    ]
    if selected_symbols != v17_symbols:
        raise GrayComparisonError("v17_top24_order_mismatch")
    common_ready = {
        str(symbol).strip().upper()
        for symbol in fusion.get("common_ready_domain", [])
        if str(symbol).strip()
    }
    v15_set = set(v15_candidates)
    v17_set = set(v17_symbols)
    overlap = v15_set & v17_set
    union = v15_set | v17_set
    v15_gross = sum(max(0.0, _float(row.get("nav_weight"))) for row in v15["holdings_rows"])
    v17_gross = max(0.0, _float(summary.get("gross_weight")))
    v15_cash = max(0.0, 1.0 - v15_gross)
    v17_cash = max(0.0, _float(summary.get("cash_weight"), default=1.0))
    deep_veto_count = sum(
        1 for row in v17_rows if isinstance(row, Mapping) and row.get("deep_status") == "BUY_VETO"
    )
    return {
        "v15_candidate_count": len(v15_candidates),
        "v17_top24_count": len(v17_symbols),
        "candidate_overlap_count": len(overlap),
        "candidate_jaccard": (
            _round(len(overlap) / len(union)) if v15_candidates and union else None
        ),
        "v15_candidate_recall_in_v17_top24": (
            _round(len(overlap) / len(v15_candidates)) if v15_candidates else None
        ),
        "v15_holding_count": len(v15_holdings),
        "v15_holdings_in_v17_common_ready_count": sum(
            symbol in common_ready for symbol in v15_holdings
        ),
        "v15_holdings_in_v17_top24_count": sum(symbol in v17_set for symbol in v15_holdings),
        "v15_actual_gross_weight": _round(v15_gross),
        "v17_model_gross_weight": _round(v17_gross),
        "gross_weight_difference_v17_minus_v15": _round(v17_gross - v15_gross),
        "v15_actual_cash_weight": _round(v15_cash),
        "v17_model_cash_weight": _round(v17_cash),
        "cash_weight_difference_v17_minus_v15": _round(v17_cash - v15_cash),
        "v17_deep_buy_veto_count": deep_veto_count,
        "v17_zero_target_count": sum(
            abs(_float(row.get("final_target"))) < 1e-15
            for row in v17_rows
            if isinstance(row, Mapping)
        ),
    }


def _effect_evaluation(
    *,
    classification: str,
    history_sessions: Sequence[str],
    metrics: Mapping[str, Any],
    minimum_forward_sessions: int,
    forward_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    if classification != "COMPARABLE":
        blockers.append("current_run_not_comparable")
    if metrics.get("v15_candidate_count") == 0:
        blockers.append("v15_candidate_set_empty")
    if int(metrics.get("v17_deep_buy_veto_count") or 0) > 0:
        blockers.append("v17_deep_evidence_unavailable_all_or_partial_buy_veto")
    aggregates = _aggregate_forward_outcomes(forward_outcomes)
    paired_20 = int(aggregates["20"]["paired_origin_count"])
    if len(history_sessions) < minimum_forward_sessions:
        blockers.append("minimum_forward_session_sample_not_met")
    if paired_20 < minimum_forward_sessions:
        blockers.append("minimum_mature_20_session_outcomes_not_met")
    if not forward_outcomes:
        blockers.append("forward_return_labels_not_yet_matured")
    rank_verdict = "PENDING_MINIMUM_MATURE_SAMPLE"
    if paired_20 >= minimum_forward_sessions:
        difference = _float(aggregates["20"]["v17_minus_v15_mean_return"])
        if difference > 0:
            rank_verdict = "V17_RANK_SET_OUTPERFORMED_V15"
        elif difference < 0:
            rank_verdict = "V17_RANK_SET_UNDERPERFORMED_V15"
        else:
            rank_verdict = "V15_V17_RANK_SET_TIED"
    return {
        "status": ("RANK_DIAGNOSTIC_AVAILABLE" if forward_outcomes else "INSUFFICIENT_EVIDENCE"),
        "verdict": "NO_V15_V17_PERFORMANCE_CONCLUSION",
        "rank_set_verdict": rank_verdict,
        "observed_comparable_sessions": len(history_sessions),
        "minimum_forward_sessions": minimum_forward_sessions,
        "forward_horizons_sessions": [1, 5, 20],
        "paired_forward_return_observation_count": len(forward_outcomes),
        "matured_forward_outcomes": list(forward_outcomes),
        "rank_set_aggregates": aggregates,
        "blockers": blockers,
        "interpretation": (
            "Current-session rank and exposure diagnostics are not realized "
            "performance. V17 model cash caused by unavailable Deep evidence "
            "must not be interpreted as superior risk control or alpha."
        ),
    }


def _render_markdown(document: Mapping[str, Any]) -> str:
    metrics = document["metrics"]
    effect = document["effect_evaluation"]
    blockers = ", ".join(effect["blockers"]) or "none"
    return "\n".join(
        [
            "# V15 / V17 日度灰度比较",
            "",
            f"- 比较分类：`{document['classification']}`",
            f"- 决策交易日：`{document['decision_session']}`",
            "- 正式权威：V15；V17 仅为 model-only shadow，所有交易权限均为 false。",
            (
                "- 候选集合："
                f"V15 `{metrics['v15_candidate_count']}`，"
                f"V17 Top24 `{metrics['v17_top24_count']}`，"
                f"重合 `{metrics['candidate_overlap_count']}`。"
            ),
            (
                "- 现有持仓覆盖："
                f"V17 common-ready `{metrics['v15_holdings_in_v17_common_ready_count']}`"
                f"/`{metrics['v15_holding_count']}`；"
                f"进入 V17 Top24 `{metrics['v15_holdings_in_v17_top24_count']}`。"
            ),
            (
                "- 仓位诊断："
                f"V15 实际总仓位 `{metrics['v15_actual_gross_weight']:.2%}`；"
                f"V17 model 总仓位 `{metrics['v17_model_gross_weight']:.2%}`；"
                f"V17 Deep BUY_VETO `{metrics['v17_deep_buy_veto_count']}`。"
            ),
            (
                "- 效果结论："
                f"`{effect['verdict']}`；可比日样本 "
                f"`{effect['observed_comparable_sessions']}`/"
                f"`{effect['minimum_forward_sessions']}`；blockers=`{blockers}`。"
            ),
            "",
            "本节只做灰度诊断，不改变 V15 建议、持仓、订单或交易。",
            "",
        ]
    )


def _atomic_write(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise GrayComparisonError(f"unsafe_output:{path}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(handle, mode)
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _attach_to_v15_record(
    run_dir: Path,
    *,
    document: Mapping[str, Any],
    json_sha256: str,
    markdown_sha256: str,
) -> None:
    reference = {
        "schema_version": SCHEMA_VERSION,
        "classification": document["classification"],
        "status": document["status"],
        "json_path": OUTPUT_JSON,
        "json_sha256": json_sha256,
        "markdown_path": OUTPUT_MARKDOWN,
        "markdown_sha256": markdown_sha256,
        "production_authority": False,
    }
    for name in ("manifest.json", "market_snapshot.json"):
        path = run_dir / name
        payload, _ = _read_json(path)
        payload["v17_gray_comparison"] = reference
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write(path, raw)
    report_path = run_dir / "analysis_report.md"
    report = _read_regular(report_path).decode("utf-8")
    marker = "\n## 7. V15 / V17 日度灰度比较\n"
    gray_section = _render_markdown(document).replace(
        "# V15 / V17 日度灰度比较",
        "## 7. V15 / V17 日度灰度比较",
        1,
    )
    if marker.strip() not in report:
        _atomic_write(
            report_path,
            (report.rstrip() + "\n\n" + gray_section).encode("utf-8"),
        )


def run_daily_gray_comparison(
    *,
    run_dir: str | Path,
    v17_workspace_root: str | Path,
    market_pointer_path: str | Path,
    pointer_sha256_before_v15: str,
    pointer_sha256_after_v15: str,
    minimum_forward_sessions: int = MINIMUM_FORWARD_SESSIONS,
) -> dict[str, Any]:
    """Write one authority-free comparison without blocking the V15 review."""

    target = Path(run_dir).resolve()
    workspace_root = Path(v17_workspace_root)
    pointer_path = Path(market_pointer_path)
    base_dir = target.parent
    status = "GRAY_UNAVAILABLE"
    classification = "NON_COMPARABLE"
    blockers: list[str] = []
    v15: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    summary_path: Path | None = None
    summary_sha = ""
    fusion: dict[str, Any] | None = None
    fusion_path: Path | None = None
    fusion_sha = ""
    pointer_sha = ""
    try:
        if minimum_forward_sessions < 5:
            raise GrayComparisonError("minimum_forward_sessions_below_5")
        pointer, pointer_sha = _read_json(pointer_path)
        if pointer_sha != pointer_sha256_before_v15 or pointer_sha != pointer_sha256_after_v15:
            raise GrayComparisonError("market_pointer_drift_during_dual_run")
        v15 = _load_v15_inputs(target)
        if (
            _normalized_session(pointer.get("latest_complete_trade_date"))
            != v15["decision_session"]
        ):
            raise GrayComparisonError("v15_market_pointer_session_mismatch")
        if v15["snapshot_id"] and pointer.get("snapshot_id") != v15["snapshot_id"]:
            raise GrayComparisonError("v15_market_snapshot_identity_mismatch")
        summary_path, summary, summary_sha = _discover_v17_summary(
            workspace_root,
            decision_session=v15["decision_session"],
            market_pointer_sha256=pointer_sha,
        )
        fusion, fusion_sha, fusion_path = _load_v17_fusion(summary_path, summary)
        classification = "COMPARABLE"
        status = "GRAY_COMPARISON_COMPLETE"
    except GrayComparisonError as exc:
        blockers.append(str(exc))

    if v15 is not None:
        decision_session = str(v15["decision_session"])
    else:
        try:
            fallback_pointer, _ = _read_json(pointer_path)
            decision_session = _normalized_session(
                fallback_pointer.get("latest_complete_trade_date")
            )
        except GrayComparisonError:
            decision_session = ""
    if classification == "COMPARABLE" and v15 and summary and fusion:
        try:
            metrics = _build_metrics(v15, summary, fusion)
            history_sessions = _comparison_history(base_dir, decision_session)
            forward_outcomes = _mature_forward_outcomes(
                base_dir=base_dir,
                current_session=decision_session,
                market_pointer_path=pointer_path,
            )
            if _sha(_read_regular(pointer_path)) != pointer_sha:
                raise GrayComparisonError("market_pointer_drift_during_forward_labels")
        except GrayComparisonError as exc:
            blockers.append(str(exc))
            classification = "NON_COMPARABLE"
            status = "GRAY_UNAVAILABLE"
    if classification != "COMPARABLE":
        metrics = {
            "v15_candidate_count": len(v15["candidate_symbols"]) if v15 else 0,
            "v17_top24_count": 0,
            "candidate_overlap_count": 0,
            "candidate_jaccard": None,
            "v15_candidate_recall_in_v17_top24": None,
            "v15_holding_count": len(v15["holding_symbols"]) if v15 else 0,
            "v15_holdings_in_v17_common_ready_count": 0,
            "v15_holdings_in_v17_top24_count": 0,
            "v15_actual_gross_weight": 0.0,
            "v17_model_gross_weight": 0.0,
            "gross_weight_difference_v17_minus_v15": 0.0,
            "v15_actual_cash_weight": 0.0,
            "v17_model_cash_weight": 1.0,
            "cash_weight_difference_v17_minus_v15": 0.0,
            "v17_deep_buy_veto_count": 0,
            "v17_zero_target_count": 0,
        }
        history_sessions = []
        forward_outcomes = []
    effect = _effect_evaluation(
        classification=classification,
        history_sessions=history_sessions,
        metrics=metrics,
        minimum_forward_sessions=minimum_forward_sessions,
        forward_outcomes=forward_outcomes,
    )
    blockers.extend(blocker for blocker in effect["blockers"] if blocker not in blockers)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "decision_session": decision_session,
        "strategy_id": "CN/aggressive_tech_manufacturing",
        "production_default": "v15",
        "authority": dict(NO_AUTHORITY),
        "comparison_contract": {
            "same_decision_session_required": True,
            "same_market_pointer_bytes_required": True,
            "v15_is_authoritative": True,
            "v17_is_model_only_shadow": True,
        },
        "inputs": {
            "market_pointer": {
                "path": str(pointer_path),
                "sha256": pointer_sha,
                "sha256_before_v15": pointer_sha256_before_v15,
                "sha256_after_v15": pointer_sha256_after_v15,
            },
            "v15_record": (
                {
                    "run_dir": str(target),
                    "manifest_pre_gray_sha256": v15["manifest_pre_gray_sha256"],
                    "stable_artifact_refs": v15["artifact_refs"],
                }
                if v15
                else None
            ),
            "v17_run_summary": (
                {
                    "path": _safe_relative(summary_path, workspace_root),
                    "sha256": summary_sha,
                    "run_id": summary.get("run_id"),
                    "factor_baseline_mode": summary.get("factor_baseline_mode"),
                    "portfolio_basis": summary.get("portfolio_basis"),
                    "calibration": summary.get("calibration"),
                }
                if summary_path and summary
                else None
            ),
            "v17_fusion": (
                {
                    "path": _safe_relative(fusion_path, workspace_root),
                    "sha256": fusion_sha,
                }
                if fusion_path
                else None
            ),
        },
        "metrics": metrics,
        "selection_sets": {
            "v15_candidates": (list(v15["candidate_symbols"]) if v15 else []),
            "v15_holdings": (list(v15["holding_symbols"]) if v15 else []),
            "v17_top24": (
                [
                    str(row.get("symbol"))
                    for row in summary.get("top24", [])
                    if isinstance(row, Mapping) and str(row.get("symbol") or "")
                ]
                if summary
                else []
            ),
        },
        "effect_evaluation": effect,
        "blockers": blockers,
        "side_effect_attestation": {
            "provider_calls": 0,
            "llm_control_calls": 0,
            "broker_calls": 0,
            "execution_calls": 0,
            "order_calls": 0,
            "trade_calls": 0,
            "selector_writes": 0,
        },
    }
    json_raw = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    markdown_raw = _render_markdown(document).encode("utf-8")
    json_path = target / OUTPUT_JSON
    markdown_path = target / OUTPUT_MARKDOWN
    _atomic_write(json_path, json_raw)
    _atomic_write(markdown_path, markdown_raw)
    raw_dir = target / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_path, raw_dir / OUTPUT_JSON)
    shutil.copy2(markdown_path, raw_dir / OUTPUT_MARKDOWN)
    json_sha = _sha(json_raw)
    markdown_sha = _sha(markdown_raw)
    _attach_to_v15_record(
        target,
        document=document,
        json_sha256=json_sha,
        markdown_sha256=markdown_sha,
    )
    return {
        "status": status,
        "classification": classification,
        "decision_session": decision_session,
        "path": str(json_path),
        "sha256": json_sha,
        "report_path": str(markdown_path),
        "report_sha256": markdown_sha,
        "effect_verdict": effect["verdict"],
        "metrics": metrics,
        "blockers": blockers,
        "authority": dict(NO_AUTHORITY),
    }


__all__ = [
    "DEFAULT_V17_WORKSPACE_ROOT",
    "MARKET_POINTER_PATH",
    "MINIMUM_FORWARD_SESSIONS",
    "NO_AUTHORITY",
    "OUTPUT_JSON",
    "OUTPUT_MARKDOWN",
    "SCHEMA_VERSION",
    "run_daily_gray_comparison",
]
