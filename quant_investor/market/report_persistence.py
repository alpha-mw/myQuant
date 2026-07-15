"""Report persistence helpers for market analysis outputs."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from quant_investor.fundamental_research.replay import validate_control_chain_replay
from quant_investor.market.config import resolve_market_analysis_output_dir
from quant_investor.market.runtime_profile import profile_stage


GenerateFullReport = Callable[..., dict[str, str]]
ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION = "analysis-run-manifest.v1"


@dataclass(frozen=True)
class MarketAnalysisPersistenceResult:
    """Persisted report paths plus the captured runtime profile payload."""

    report_paths: dict[str, Any]
    runtime_profile: dict[str, Any]


def _runtime_profile_dir(
    report_paths: dict[str, Any],
    *,
    analysis_output_dir: str | Path,
) -> Path:
    profile_anchor = report_paths.get("trade_report") or report_paths.get(
        "summary_report"
    )
    if profile_anchor:
        return Path(str(profile_anchor)).parent
    return Path(analysis_output_dir)


def write_runtime_profile_artifacts(
    *,
    market: str,
    analysis_output_dir: str | Path,
    report_paths: dict[str, Any],
    runtime_profiler: Any,
    runtime_profile_payload: dict[str, Any],
) -> dict[str, str]:
    """Write runtime profile JSON/Markdown next to the market reports."""

    resolved_output_dir = resolve_market_analysis_output_dir(
        market,
        analysis_output_dir,
    )
    profile_dir = _runtime_profile_dir(
        report_paths,
        analysis_output_dir=resolved_output_dir,
    )
    profile_dir = resolve_market_analysis_output_dir(market, profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_profile_json = (
        profile_dir / f"{market}_Runtime_Profile_{profile_timestamp}.json"
    )
    runtime_profile_md = (
        profile_dir / f"{market}_Runtime_Profile_{profile_timestamp}.md"
    )
    runtime_profile_json.write_text(
        json.dumps(
            runtime_profile_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    runtime_profile_md.write_text(
        runtime_profiler.to_markdown(),
        encoding="utf-8",
    )
    return {
        "runtime_profile_json": str(runtime_profile_json),
        "runtime_profile_md": str(runtime_profile_md),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise RuntimeError("private manifest path must not contain symlinks")
    if os.path.lexists(absolute):
        raise FileExistsError(f"private manifest is immutable and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _current_git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _fundamental_counterfactual_manifest_meta(
    analysis_meta: dict[str, Any],
) -> tuple[dict[str, Any], str, str] | None:
    portfolio = analysis_meta.get("portfolio_decision")
    if not isinstance(portfolio, dict):
        return None
    metadata = portfolio.get("metadata")
    if not isinstance(metadata, dict):
        return None
    replay = metadata.get("fundamental_research_counterfactual_replay")
    if not isinstance(replay, dict):
        return None
    validate_control_chain_replay(replay)
    alternative_variant = str(replay.get("variant") or "")
    actual_variant = (
        "without_dossier" if alternative_variant == "with_dossier" else "with_dossier"
    )
    alternative_meta = json.loads(json.dumps(analysis_meta, ensure_ascii=False))
    alternative_meta.update(
        {
            "branch_summaries": replay["branch_summaries"],
            "branch_verdicts_by_symbol": replay["branch_verdicts_by_symbol"],
            "bayesian_records": replay["bayesian_records"],
            "bayesian_record_count": len(replay["bayesian_records"]),
            "shortlist": replay["shortlist"],
            "bayesian_shortlist_symbols": [
                str(item.get("symbol") or "")
                for item in replay["shortlist"]
                if isinstance(item, dict) and str(item.get("symbol") or "")
            ],
            "ic_hints_by_symbol": replay["ic_hints_by_symbol"],
            "risk_decision": replay["risk_decision"],
            "ic_decisions": replay["ic_decisions"],
            "portfolio_plan": replay["portfolio_plan"],
            "portfolio_decision": replay["portfolio_decision"],
            "fundamental_research_control_chain": replay,
        }
    )
    alternative_meta["fundamental_research_variant"] = alternative_variant
    analysis_meta["fundamental_research_variant"] = actual_variant
    return alternative_meta, actual_variant, alternative_variant


def write_analysis_run_manifest(
    *,
    market: str,
    analysis_output_dir: str | Path,
    report_paths: Mapping[str, Any],
    analysis_meta: Mapping[str, Any],
) -> str:
    """Persist a private, immutable analysis manifest and validated replay companion."""

    resolved_output_dir = resolve_market_analysis_output_dir(market, analysis_output_dir)
    profile_dir = _runtime_profile_dir(
        dict(report_paths), analysis_output_dir=resolved_output_dir
    )
    profile_dir = resolve_market_analysis_output_dir(market, profile_dir)
    output_root = Path(resolved_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for component in (profile_dir, *profile_dir.parents):
        if component.is_symlink():
            raise RuntimeError("analysis output path must not contain symlinks")
    resolved_root = output_root.resolve(strict=True)
    resolved_profile_dir = profile_dir.resolve(strict=True)
    if not resolved_profile_dir.is_relative_to(resolved_root):
        raise RuntimeError("analysis manifest directory escapes analysis output root")
    profile_dir = resolved_profile_dir
    normalized_meta = json.loads(
        json.dumps(analysis_meta, ensure_ascii=False, allow_nan=False, default=str)
    )
    normalized_market = str(market).upper()
    companion = (
        _fundamental_counterfactual_manifest_meta(normalized_meta)
        if normalized_market == "CN"
        else None
    )
    now_utc = datetime.now(timezone.utc)
    generated_at = now_utc.isoformat()
    run_id = f"{normalized_market}_{now_utc.strftime('%Y%m%dT%H%M%S%fZ')}"
    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "market": normalized_market,
        "git_sha": _current_git_sha(),
        "analysis_meta_sha256": hashlib.sha256(_json_bytes(normalized_meta)).hexdigest(),
        "analysis_meta": normalized_meta,
    }
    payload["manifest_sha256"] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    run_dir = profile_dir / "fundamental_research_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    target = run_dir / "analysis_run_manifest.v1.json"
    _atomic_write_private_json(target, payload)
    if json.loads(target.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("analysis run manifest readback mismatch")
    if companion is not None:
        alternative_meta, actual_variant, alternative_variant = companion
        alternative_meta["fundamental_research_source_run_id"] = run_id
        alternative_meta["fundamental_research_source_manifest_sha256"] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
        alternative_payload: dict[str, Any] = {
            "schema_version": ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": f"{run_id}:{alternative_variant}",
            "generated_at": generated_at,
            "market": normalized_market,
            "git_sha": payload["git_sha"],
            "analysis_meta_sha256": hashlib.sha256(
                _json_bytes(alternative_meta)
            ).hexdigest(),
            "analysis_meta": alternative_meta,
        }
        alternative_payload["manifest_sha256"] = hashlib.sha256(
            _json_bytes(alternative_payload)
        ).hexdigest()
        companion_target = (
            run_dir / f"analysis_run_manifest.{alternative_variant}.v1.json"
        )
        _atomic_write_private_json(companion_target, alternative_payload)
        if json.loads(companion_target.read_text(encoding="utf-8")) != alternative_payload:
            raise RuntimeError("counterfactual analysis manifest readback mismatch")
        if actual_variant not in {"with_dossier", "without_dossier"}:
            raise RuntimeError("actual fundamental research variant is invalid")
    return str(target)


def persist_market_analysis_outputs(
    *,
    all_results: dict[str, list[dict[str, Any]]],
    market: str,
    total_capital: float,
    top_k: int,
    analysis_output_dir: str | Path,
    category_count: int,
    runtime_profiler: Any,
    report_bundle: Any,
    generate_full_report: GenerateFullReport,
) -> MarketAnalysisPersistenceResult:
    """Persist full-market reports and runtime profile artifacts."""

    resolved_output_dir = resolve_market_analysis_output_dir(
        market,
        analysis_output_dir,
    )
    with profile_stage(
        runtime_profiler,
        "analysis_report_persistence",
        {
            "category_count": int(category_count),
            "result_count": sum(len(items) for items in all_results.values()),
        },
    ) as stage_metadata:
        report_paths: dict[str, Any] = dict(
            generate_full_report(
                all_results,
                market=market,
                output_dir=str(resolved_output_dir),
                total_capital=total_capital,
                top_k=top_k,
            )
        )
        stage_metadata["report_path_count"] = len(report_paths)

    runtime_profile_payload = runtime_profiler.to_dict()
    runtime_paths = write_runtime_profile_artifacts(
        market=market,
        analysis_output_dir=resolved_output_dir,
        report_paths=report_paths,
        runtime_profiler=runtime_profiler,
        runtime_profile_payload=runtime_profile_payload,
    )
    report_paths["report_bundle"] = report_bundle
    report_paths.update(runtime_paths)
    return MarketAnalysisPersistenceResult(
        report_paths=report_paths,
        runtime_profile=runtime_profile_payload,
    )


__all__ = [
    "ANALYSIS_RUN_MANIFEST_SCHEMA_VERSION",
    "MarketAnalysisPersistenceResult",
    "persist_market_analysis_outputs",
    "write_analysis_run_manifest",
    "write_runtime_profile_artifacts",
]
