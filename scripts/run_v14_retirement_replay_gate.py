"""Replay frozen four-branch evidence through the v14 two-likelihood gate.

This is an offline cutover verifier, not a runtime compatibility adapter.  It
reads the preserved audit files as plain historical evidence, rebuilds only the
surviving Quant and Fundamental likelihood inputs, and proves that the v14 BUY
set is a subset of the frozen baseline BUY set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_investor.agent_protocol import GlobalContext
from quant_investor.bayesian.likelihood import SignalLikelihoodMapper
from quant_investor.bayesian.posterior import BayesianPosteriorEngine
from quant_investor.bayesian.prior import HierarchicalPriorBuilder
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.branch_contracts import BranchResult
from quant_investor.market.dag.common import _score_to_action
from quant_investor.market.dag.shortlist import (
    _build_shortlist_from_bayesian_records,
)
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
)


SCHEMA_VERSION = "myquant.v14-retirement-replay-gate.v1"
CANONICAL_LIKELIHOOD_ORDER = ("quant", "fundamental")
EXPECTED_SUMMARY_SHA256 = (
    "79dcba806559ec4078ee680a5b4018f6ade62467e4cc1d99151318eca93d6d27"
)
EXPECTED_ACTIONS_SHA256 = (
    "394207a9a3ac3f0fd11d8e3ddfc86bbf705da2913b68d8007ea9953e848b14af"
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RetirementReplayError(RuntimeError):
    """Raised when frozen replay evidence is incomplete or has drifted."""


class _NoCalibrationHistory:
    """Reproduce the frozen run's zero-sample calibration state."""

    @staticmethod
    def calibration_stats(_branch_name: str, _score: float) -> dict[str, float]:
        return {
            "probability": 0.5,
            "sample_size": 0.0,
            "recent_failure_rate": 0.0,
        }


def _verify_candidate_state(candidate_commit: str) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RetirementReplayError(
                "candidate_repository_state_unverifiable"
            ) from exc
        return completed.stdout.strip()

    resolved_commit = run_git(
        "rev-parse",
        "--verify",
        f"{candidate_commit}^{{commit}}",
    ).lower()
    if resolved_commit != candidate_commit:
        raise RetirementReplayError("candidate_commit_not_found")
    head_commit = run_git("rev-parse", "HEAD").lower()
    if head_commit != candidate_commit:
        raise RetirementReplayError("candidate_commit_not_head")
    status = run_git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RetirementReplayError("candidate_worktree_not_clean")
    return {
        "repository_root": str(REPOSITORY_ROOT),
        "head_commit": head_commit,
        "worktree_clean": True,
    }


def _read_verified_bytes(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RetirementReplayError(f"replay_evidence_path_unsafe:{path}")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RetirementReplayError(
            f"replay_evidence_sha256_mismatch:{path}:{actual}"
        )
    return payload


def _finite_float(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RetirementReplayError(f"{field_name}_invalid") from exc
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise RetirementReplayError(f"{field_name}_out_of_range")
    return numeric


def _load_baseline(
    *,
    summary_path: Path,
    actions_path: Path,
    expected_summary_sha256: str,
    expected_actions_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    summary_bytes = _read_verified_bytes(summary_path, expected_summary_sha256)
    actions_bytes = _read_verified_bytes(actions_path, expected_actions_sha256)
    try:
        summary = json.loads(summary_bytes.decode("utf-8"))
        reader = csv.DictReader(
            io.StringIO(actions_bytes.decode("utf-8-sig"), newline="")
        )
        rows = {
            str(row.get("symbol") or "").strip(): dict(row)
            for row in reader
            if str(row.get("symbol") or "").strip()
        }
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        raise RetirementReplayError("replay_evidence_parse_failed") from exc
    if not isinstance(summary, Mapping):
        raise RetirementReplayError("replay_summary_not_object")
    return dict(summary), rows


def _frozen_context(summary: Mapping[str, Any]) -> tuple[GlobalContext, list[str]]:
    try:
        dag = dict(summary["dag"])
        candidates = [str(item) for item in dag["candidate_symbols"]]
        context_payload = dict(
            dag["portfolio_decision"]["execution_trace"]["steps"][1][
                "metadata"
            ]["global_context"]
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RetirementReplayError("replay_global_context_missing") from exc
    if not candidates or len(candidates) != len(set(candidates)):
        raise RetirementReplayError("replay_candidate_symbols_invalid")

    source_metadata = dict(context_payload.get("metadata", {}) or {})
    source_profile = dict(source_metadata.get("selection_profile", {}) or {})
    selection_profile = {
        key: source_profile[key]
        for key in (
            "funnel_profile",
            "trend_windows",
            "volume_spike_threshold",
            "breakout_distance_pct",
            "max_candidates",
            "sector_bucket_limit",
        )
        if key in source_profile
    }
    metadata = {
        "selection_profile": selection_profile,
        "symbol_market_state": dict(
            source_metadata.get("symbol_market_state", {}) or {}
        ),
        "candidate_sector_counts": dict(
            source_metadata.get("candidate_sector_counts", {}) or {}
        ),
    }
    context = GlobalContext(
        market="CN",
        universe_key="full_a",
        rebalance_date=str(context_payload.get("rebalance_date") or "20260605"),
        latest_trade_date=str(
            context_payload.get("latest_trade_date") or "20260605"
        ),
        universe_symbols=list(candidates),
        industry_map=dict(context_payload.get("industry_map", {}) or {}),
        liquidity_filter=dict(
            context_payload.get("liquidity_filter", {}) or {}
        ),
        macro_regime=str(context_payload.get("macro_regime") or "未知"),
        cross_section_quant=dict(
            context_payload.get("cross_section_quant", {}) or {}
        ),
        style_exposures=dict(
            context_payload.get("style_exposures", {}) or {}
        ),
        risk_budget=dict(context_payload.get("risk_budget", {}) or {}),
        metadata=metadata,
    )
    return context, candidates


def _baseline_buy_symbols(summary: Mapping[str, Any]) -> set[str]:
    try:
        shortlist = summary["dag"]["portfolio_decision"]["shortlist"]
    except (KeyError, TypeError) as exc:
        raise RetirementReplayError("replay_baseline_shortlist_missing") from exc
    if not isinstance(shortlist, list):
        raise RetirementReplayError("replay_baseline_shortlist_invalid")
    return {
        str(item.get("symbol") or "").strip()
        for item in shortlist
        if isinstance(item, Mapping)
        and str(item.get("action") or "").strip().lower() == "buy"
        and str(item.get("symbol") or "").strip()
    }


def evaluate_no_new_buy(
    *,
    baseline_buy_symbols: set[str],
    target_buy_symbols: set[str],
) -> dict[str, Any]:
    new_buy = sorted(target_buy_symbols - baseline_buy_symbols)
    return {
        "passed": not new_buy,
        "baseline_buy_symbols": sorted(baseline_buy_symbols),
        "target_buy_symbols": sorted(target_buy_symbols),
        "new_buy_symbols": new_buy,
        "removed_baseline_buy_symbols": sorted(
            baseline_buy_symbols - target_buy_symbols
        ),
    }


def _replay_records(
    *,
    context: GlobalContext,
    candidates: list[str],
    rows: Mapping[str, Mapping[str, str]],
    frozen_evidence_sha256: str,
) -> list[dict[str, Any]]:
    mapper = SignalLikelihoodMapper(
        calibration_store=_NoCalibrationHistory(),
        global_context=context,
    )
    prior_builder = HierarchicalPriorBuilder()
    posterior_engine = BayesianPosteriorEngine()
    candidate_set = set(candidates)
    records: list[dict[str, Any]] = []
    for symbol in candidates:
        row = rows.get(symbol)
        if row is None:
            raise RetirementReplayError(f"replay_action_row_missing:{symbol}")
        branch_results: dict[str, BranchResult] = {}
        for branch_name in CANONICAL_LIKELIHOOD_ORDER:
            score = _finite_float(
                row.get(f"{branch_name}_score"),
                field_name=f"{symbol}_{branch_name}_score",
                minimum=-1.0,
                maximum=1.0,
            )
            confidence = _finite_float(
                row.get(f"{branch_name}_confidence"),
                field_name=f"{symbol}_{branch_name}_confidence",
                minimum=0.0,
                maximum=1.0,
            )
            branch_metadata: dict[str, Any] = {}
            if branch_name == "fundamental":
                frozen_generation = (
                    "frozen-evidence-" + frozen_evidence_sha256
                )
                branch_metadata = {
                    "fundamental_data_generation_by_symbol": {
                        symbol: frozen_generation
                    },
                    "fundamental_data_generation_status_by_symbol": {
                        symbol: "confirmed"
                    },
                    "replay_evidence_sha256": frozen_evidence_sha256,
                }
            branch_results[branch_name] = BranchResult(
                branch_name=branch_name,
                final_score=score,
                final_confidence=confidence,
                symbol_scores={symbol: score},
                metadata=branch_metadata,
            )
        prior = prior_builder.build_prior(symbol, context)
        likelihoods = mapper.compute_likelihoods(
            branch_results=branch_results,
            symbol=symbol,
            candidate_symbols=candidate_set,
        )
        posterior = posterior_engine.compute_posterior(
            prior,
            likelihoods,
            symbol=symbol,
            company_name=str(row.get("name") or ""),
            regime=context.macro_regime or "未知",
            is_degraded={"quant": False, "fundamental": False},
        )
        posterior_metadata = dict(posterior.metadata or {})
        posterior_metadata.update(
            {
                "category": "",
                "profile": str(
                    context.metadata.get("selection_profile", {}).get(
                        "funnel_profile", "classic"
                    )
                ),
                "posterior_edge_after_costs": (
                    posterior.posterior_edge_after_costs
                ),
                "posterior_capacity_penalty": (
                    posterior.posterior_capacity_penalty
                ),
                "kill_switch": bool(
                    posterior_metadata.get("kill_switch", False)
                ),
            }
        )
        records.append(
            {
                "symbol": symbol,
                "company_name": str(row.get("name") or ""),
                "prior": posterior.prior.to_dict(),
                "likelihoods": posterior.likelihoods.to_dict(),
                "posterior_win_rate": posterior.posterior_win_rate,
                "posterior_expected_alpha": posterior.posterior_expected_alpha,
                "posterior_confidence": posterior.posterior_confidence,
                "posterior_action_score": posterior.posterior_action_score,
                "posterior_edge_after_costs": (
                    posterior.posterior_edge_after_costs
                ),
                "posterior_capacity_penalty": (
                    posterior.posterior_capacity_penalty
                ),
                "metadata": posterior_metadata,
                "rank": 0,
            }
        )
    return records


def _admission_reason(record: Mapping[str, Any], admitted: bool) -> str:
    if admitted:
        return "admitted_current_v14_buy"
    metadata = dict(record.get("metadata", {}) or {})
    if bool(metadata.get("kill_switch", False)):
        return "rejected_kill_switch"
    if _score_to_action(float(record["posterior_action_score"])).value != "buy":
        return "rejected_action_score_below_buy_threshold"
    if float(record["posterior_expected_alpha"]) <= 0.0:
        return "rejected_nonpositive_expected_alpha"
    if float(record["posterior_edge_after_costs"]) <= 0.0:
        return "rejected_nonpositive_edge_after_costs"
    return "rejected_current_shortlist_contract"


def build_replay_report(
    *,
    summary_path: Path,
    actions_path: Path,
    candidate_commit: str,
    expected_summary_sha256: str = EXPECTED_SUMMARY_SHA256,
    expected_actions_sha256: str = EXPECTED_ACTIONS_SHA256,
) -> dict[str, Any]:
    commit = str(candidate_commit or "").strip().lower()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise RetirementReplayError("candidate_commit_invalid")
    candidate_state = _verify_candidate_state(commit)
    summary, rows = _load_baseline(
        summary_path=summary_path,
        actions_path=actions_path,
        expected_summary_sha256=expected_summary_sha256,
        expected_actions_sha256=expected_actions_sha256,
    )
    context, candidates = _frozen_context(summary)
    records = _replay_records(
        context=context,
        candidates=candidates,
        rows=rows,
        frozen_evidence_sha256=expected_actions_sha256,
    )
    shortlist = _build_shortlist_from_bayesian_records(
        posterior_results=records,
        company_name_map={
            symbol: str(rows[symbol].get("name") or "")
            for symbol in candidates
        },
        top_k=len(candidates),
    )
    target_buy = {item.symbol for item in shortlist}
    baseline_buy = _baseline_buy_symbols(summary)
    comparison = evaluate_no_new_buy(
        baseline_buy_symbols=baseline_buy,
        target_buy_symbols=target_buy,
    )
    admitted = set(target_buy)
    replay_rows = [
        {
            "symbol": record["symbol"],
            "likelihoods": record["likelihoods"],
            "posterior_action_score": record["posterior_action_score"],
            "posterior_expected_alpha": record[
                "posterior_expected_alpha"
            ],
            "posterior_edge_after_costs": record[
                "posterior_edge_after_costs"
            ],
            "admission_reason": _admission_reason(
                record,
                str(record["symbol"]) in admitted,
            ),
        }
        for record in records
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if comparison["passed"] else "failed",
        "candidate_commit": commit,
        "candidate_state": candidate_state,
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "canonical_branch_order": list(CANONICAL_BRANCH_ORDER),
        "canonical_likelihood_order": list(CANONICAL_LIKELIHOOD_ORDER),
        "replay_scope": {
            "name": "frozen_candidate_set_pre_control",
            "full_universe_funnel_replayed": False,
            "control_chain_replayed": False,
            "interpretation": (
                "no new BUY inside the hash-bound frozen candidate set"
            ),
        },
        "baseline": {
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": expected_summary_sha256,
            "actions_path": str(actions_path.resolve()),
            "actions_sha256": expected_actions_sha256,
            "candidate_symbols": list(candidates),
        },
        "comparison": comparison,
        "replay_rows": replay_rows,
        "gates": {
            "candidate_commit_bound": True,
            "replay_gate_passed": bool(comparison["passed"]),
            "no_new_buy": bool(comparison["passed"]),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline v14 retirement replay/no-new-BUY gate."
    )
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-actions", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build_replay_report(
            summary_path=args.baseline_summary,
            actions_path=args.baseline_actions,
            candidate_commit=args.candidate_commit,
        )
    except (OSError, RetirementReplayError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        return 2
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
