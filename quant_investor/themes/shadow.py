"""Fail-safe theme production/shadow diagnostics for theme rotation artifacts."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _default_theme_overlay_modules() -> dict[str, bool]:
    return {
        "funnel_boost": False,
        "risk_guard": False,
        "portfolio_cap": False,
    }


def build_theme_production_overlay_diagnostics(
    *,
    funnel_boost_enabled: bool = False,
    risk_guard_enabled: bool = False,
    portfolio_cap_enabled: bool = False,
) -> dict[str, Any]:
    """Return the authoritative production/control source contract."""

    modules = {
        "funnel_boost": bool(funnel_boost_enabled),
        "risk_guard": bool(risk_guard_enabled),
        "portfolio_cap": bool(portfolio_cap_enabled),
    }
    overlay_applied = any(modules.values())
    notes: list[str] = []
    if overlay_applied:
        disabled = [name for name, enabled in modules.items() if not enabled]
        if disabled:
            notes.append("theme_overlay_partial_modules")
        else:
            notes.append("theme_overlay_all_modules_enabled")
    else:
        notes.append("theme_overlay_disabled_default")
    return {
        "production_decision_source": (
            "theme_overlay_baseline" if overlay_applied else "no_theme_baseline"
        ),
        "control_decision_source": "no_theme_baseline",
        "theme_overlay_applied_to_baseline": overlay_applied,
        "theme_overlay_modules": modules,
        "canonical_branch_unchanged": True,
        "theme_likelihood_added": False,
        "posterior_formula_changed": False,
        "diagnostic_notes": notes,
    }


@dataclass
class ThemeShadowDelta:
    symbol: str
    baseline_weight: float = 0.0
    shadow_weight: float = 0.0
    weight_delta: float = 0.0
    baseline_selected: bool = False
    shadow_selected: bool = False
    primary_theme_id: str = ""
    primary_theme_name: str = ""
    phase: str = ""
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": str(self.symbol),
            "baseline_weight": _finite_float(self.baseline_weight),
            "shadow_weight": _finite_float(self.shadow_weight),
            "weight_delta": _finite_float(self.weight_delta),
            "baseline_selected": bool(self.baseline_selected),
            "shadow_selected": bool(self.shadow_selected),
            "primary_theme_id": str(self.primary_theme_id or ""),
            "primary_theme_name": str(self.primary_theme_name or ""),
            "phase": str(self.phase or ""),
            "risk_flags": [str(flag) for flag in list(self.risk_flags or []) if str(flag)],
        }


@dataclass
class ThemeShadowMonitor:
    status: str = "disabled"
    execution_target: str = "baseline"
    final_decision_source: str = "baseline"
    production_decision_source: str = "no_theme_baseline"
    control_decision_source: str = "no_theme_baseline"
    theme_overlay_applied_to_baseline: bool = False
    theme_overlay_modules: dict[str, bool] = field(default_factory=_default_theme_overlay_modules)
    canonical_branch_unchanged: bool = True
    theme_likelihood_added: bool = False
    posterior_formula_changed: bool = False
    baseline_candidate_count: int = 0
    shadow_candidate_count: int = 0
    candidate_overlap_ratio: float = 0.0
    entered_candidates: list[str] = field(default_factory=list)
    dropped_candidates: list[str] = field(default_factory=list)
    baseline_selected_count: int = 0
    shadow_selected_count: int = 0
    selected_overlap_ratio: float = 0.0
    portfolio_weight_deltas: list[ThemeShadowDelta] = field(default_factory=list)
    theme_exposure_baseline: dict[str, float] = field(default_factory=dict)
    theme_exposure_shadow: dict[str, float] = field(default_factory=dict)
    risk_delta: dict[str, Any] = field(default_factory=dict)
    funnel_diagnostics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""
    diagnostic_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status or "disabled"),
            "execution_target": "baseline",
            "final_decision_source": "baseline",
            "production_decision_source": str(
                self.production_decision_source or "no_theme_baseline"
            ),
            "control_decision_source": str(
                self.control_decision_source or "no_theme_baseline"
            ),
            "theme_overlay_applied_to_baseline": bool(
                self.theme_overlay_applied_to_baseline
            ),
            "theme_overlay_modules": {
                str(name): bool(enabled)
                for name, enabled in dict(
                    self.theme_overlay_modules or _default_theme_overlay_modules()
                ).items()
            },
            "canonical_branch_unchanged": bool(self.canonical_branch_unchanged),
            "theme_likelihood_added": bool(self.theme_likelihood_added),
            "posterior_formula_changed": bool(self.posterior_formula_changed),
            "baseline_candidate_count": int(self.baseline_candidate_count),
            "shadow_candidate_count": int(self.shadow_candidate_count),
            "candidate_overlap_ratio": _finite_float(self.candidate_overlap_ratio),
            "entered_candidates": [str(symbol) for symbol in list(self.entered_candidates or [])],
            "dropped_candidates": [str(symbol) for symbol in list(self.dropped_candidates or [])],
            "baseline_selected_count": int(self.baseline_selected_count),
            "shadow_selected_count": int(self.shadow_selected_count),
            "selected_overlap_ratio": _finite_float(self.selected_overlap_ratio),
            "portfolio_weight_deltas": [
                delta.to_dict()
                for delta in list(self.portfolio_weight_deltas or [])
            ],
            "theme_exposure_baseline": {
                str(theme_id): _finite_float(weight)
                for theme_id, weight in dict(self.theme_exposure_baseline or {}).items()
            },
            "theme_exposure_shadow": {
                str(theme_id): _finite_float(weight)
                for theme_id, weight in dict(self.theme_exposure_shadow or {}).items()
            },
            "risk_delta": _json_safe(self.risk_delta),
            "funnel_diagnostics": _json_safe(self.funnel_diagnostics),
            "artifact_path": str(self.artifact_path or ""),
            "diagnostic_notes": [str(note) for note in list(self.diagnostic_notes or [])],
            "metadata": _json_safe(self.metadata),
        }

    def to_markdown(self, *, max_rows: int = 20) -> str:
        if str(self.status).lower() == "disabled":
            return ""
        row_limit = max(int(max_rows or 0), 0)
        if str(self.status).lower() == "error":
            notes = ", ".join(self.diagnostic_notes[:row_limit])
            return (
                "## Theme Shadow Monitor\n\n"
                "- status: error\n"
                "- final_decision_source: baseline\n"
                f"- production_decision_source: {self.production_decision_source}\n"
                f"- control_decision_source: {self.control_decision_source}\n"
                f"- diagnostic_notes: {notes or '-'}\n"
                f"- {_source_note(self.theme_overlay_applied_to_baseline)}\n"
            )
        lines = [
            "## Theme Shadow Monitor",
            "",
            f"- status: {self.status}",
            "- final_decision_source: baseline",
            f"- production_decision_source: {self.production_decision_source}",
            f"- control_decision_source: {self.control_decision_source}",
            f"- theme_overlay_applied_to_baseline: {str(bool(self.theme_overlay_applied_to_baseline)).lower()}",
            f"- theme_overlay_modules: {_module_summary(self.theme_overlay_modules)}",
            f"- {_source_note(self.theme_overlay_applied_to_baseline)}",
            f"- candidate_overlap_ratio: {self.candidate_overlap_ratio:.6f}",
            f"- entered_candidates: {_join_symbols(self.entered_candidates, row_limit)}",
            f"- dropped_candidates: {_join_symbols(self.dropped_candidates, row_limit)}",
            f"- selected_overlap_ratio: {self.selected_overlap_ratio:.6f}",
            "",
            "### Largest Weight Deltas",
            _delta_table(self.portfolio_weight_deltas, row_limit),
            "",
            "### Theme Exposure Baseline vs Shadow",
            _exposure_table(self.theme_exposure_baseline, self.theme_exposure_shadow, row_limit),
            "",
            "### Risk Delta",
            _mapping_markdown(self.risk_delta, row_limit),
        ]
        if self.artifact_path:
            lines.extend(["", f"- artifact_path: {self.artifact_path}"])
        return "\n".join(lines).rstrip() + "\n"


def build_theme_shadow_monitor(
    *,
    dag_artifacts: Mapping[str, Any],
    enabled: bool,
    execution_target: str = "baseline",
    funnel_boost_enabled: bool = True,
    risk_guard_enabled: bool = True,
    portfolio_cap_enabled: bool = True,
    production_funnel_boost_enabled: bool = False,
    production_risk_guard_enabled: bool = False,
    production_portfolio_cap_enabled: bool = False,
    artifact_enabled: bool = True,
    artifact_dir: str | Path = "results/theme_shadow",
    max_rows: int = 50,
) -> ThemeShadowMonitor:
    row_limit = max(int(max_rows or 0), 0)
    production_overlay = build_theme_production_overlay_diagnostics(
        funnel_boost_enabled=production_funnel_boost_enabled,
        risk_guard_enabled=production_risk_guard_enabled,
        portfolio_cap_enabled=production_portfolio_cap_enabled,
    )
    metadata = {
        "deterministic": True,
        "no_llm": True,
        "no_network": True,
        "shadow_only": True,
        "default_final_decision_source": "baseline",
        "production_overlay": dict(production_overlay),
        "funnel_boost_enabled": bool(funnel_boost_enabled),
        "risk_guard_enabled": bool(risk_guard_enabled),
        "portfolio_cap_enabled": bool(portfolio_cap_enabled),
        "artifact_enabled": bool(artifact_enabled),
        "max_rows": row_limit,
    }
    if not enabled:
        return ThemeShadowMonitor(
            status="disabled",
            execution_target="baseline",
            final_decision_source="baseline",
            **_monitor_overlay_kwargs(production_overlay),
            metadata=metadata,
        )

    try:
        monitor = ThemeShadowMonitor(
            status="success",
            execution_target="baseline",
            final_decision_source="baseline",
            **_monitor_overlay_kwargs(production_overlay),
            metadata=dict(metadata),
        )
        requested_target = str(execution_target or "baseline").strip().lower()
        if requested_target != "baseline":
            monitor.diagnostic_notes.append("unsupported_execution_target_ignored")

        artifacts = dag_artifacts if isinstance(dag_artifacts, Mapping) else {}
        global_context = artifacts.get("global_context")
        baseline_symbols = _candidate_symbols(artifacts)
        if funnel_boost_enabled:
            _populate_candidate_shadow(
                monitor=monitor,
                artifacts=artifacts,
                global_context=global_context,
                max_rows=row_limit,
            )
        else:
            monitor.diagnostic_notes.append("funnel_shadow_disabled")

        if risk_guard_enabled:
            _populate_risk_shadow(
                monitor=monitor,
                artifacts=artifacts,
                global_context=global_context,
                symbols=baseline_symbols,
            )
        else:
            monitor.diagnostic_notes.append("risk_shadow_disabled")

        if portfolio_cap_enabled:
            _populate_portfolio_shadow(
                monitor=monitor,
                artifacts=artifacts,
                global_context=global_context,
                max_rows=row_limit,
            )
        else:
            monitor.diagnostic_notes.append("portfolio_shadow_disabled")

        if artifact_enabled:
            _safe_write_artifact(
                monitor=monitor,
                dag_artifacts=artifacts,
                artifact_dir=artifact_dir,
            )
        return monitor
    except Exception as exc:
        monitor = ThemeShadowMonitor(
            status="error",
            execution_target="baseline",
            final_decision_source="baseline",
            **_monitor_overlay_kwargs(production_overlay),
            diagnostic_notes=[f"theme_shadow_error: {exc}"],
            metadata=dict(metadata),
        )
        if artifact_enabled:
            try:
                _safe_write_artifact(
                    monitor=monitor,
                    dag_artifacts=dag_artifacts if isinstance(dag_artifacts, Mapping) else {},
                    artifact_dir=artifact_dir,
                )
            except Exception:
                monitor.diagnostic_notes.append("theme_shadow_artifact_write_failed")
        return monitor


def _populate_candidate_shadow(
    *,
    monitor: ThemeShadowMonitor,
    artifacts: Mapping[str, Any],
    global_context: Any,
    max_rows: int,
) -> None:
    quant_result = _quant_result(artifacts.get("branch_results"))
    if quant_result is None or global_context is None:
        monitor.diagnostic_notes.append("funnel_shadow_unavailable")
        return
    diagnostics_module = _theme_boost_diagnostics_module()
    base_config = _funnel_config_from_artifacts(artifacts, diagnostics_module)
    diagnostics = diagnostics_module.compare_theme_boost_candidates(
        quant_result=quant_result,
        global_context=global_context,
        base_config=base_config,
        top_n=max_rows,
    )
    diagnostics_dict = (
        diagnostics.to_dict()
        if hasattr(diagnostics, "to_dict")
        else dict(diagnostics or {})
    )
    diagnostics_dict = _limit_funnel_diagnostics(diagnostics_dict, max_rows)
    monitor.funnel_diagnostics = diagnostics_dict
    monitor.baseline_candidate_count = int(diagnostics_dict.get("baseline_count", 0) or 0)
    monitor.shadow_candidate_count = int(diagnostics_dict.get("boosted_count", 0) or 0)
    monitor.candidate_overlap_ratio = _finite_float(diagnostics_dict.get("overlap_ratio", 0.0))
    monitor.entered_candidates = _limited_text_list(diagnostics_dict.get("entered_symbols"), max_rows)
    monitor.dropped_candidates = _limited_text_list(diagnostics_dict.get("dropped_symbols"), max_rows)


def _populate_risk_shadow(
    *,
    monitor: ThemeShadowMonitor,
    artifacts: Mapping[str, Any],
    global_context: Any,
    symbols: list[str],
) -> None:
    if global_context is None or not symbols:
        monitor.diagnostic_notes.append("risk_shadow_constraints_only")
    constraints = _build_theme_risk_constraints(
        global_context=global_context,
        symbols=symbols,
        enabled=True,
    )
    baseline_risk = artifacts.get("risk_decision")
    baseline_action_cap = _action_value(getattr(baseline_risk, "action_cap", ""))
    baseline_gross_cap = _optional_float(getattr(baseline_risk, "gross_exposure_cap", None))
    baseline_max_weight = _optional_float(getattr(baseline_risk, "max_weight", None))
    theme_position_limits = dict(constraints.get("theme_position_limits", {}) or {})
    theme_risk_flags = _limited_text_list(constraints.get("theme_risk_flags"), 50)
    theme_gross_cap = _optional_float(constraints.get("theme_gross_exposure_cap"))
    theme_action_cap = str(constraints.get("theme_action_cap") or "").strip()
    theme_risk_effect = bool(
        theme_position_limits
        or theme_risk_flags
        or theme_gross_cap is not None
        or theme_action_cap
    )
    shadow_action_cap = theme_action_cap or baseline_action_cap or ""
    shadow_gross_cap = theme_gross_cap
    if shadow_gross_cap is None:
        shadow_gross_cap = baseline_gross_cap
    elif baseline_gross_cap is not None:
        shadow_gross_cap = min(shadow_gross_cap, baseline_gross_cap)

    risk_delta = {
        "baseline_action_cap": baseline_action_cap,
        "shadow_action_cap": shadow_action_cap,
        "baseline_gross_cap": baseline_gross_cap,
        "shadow_gross_cap": shadow_gross_cap,
        "baseline_max_weight": baseline_max_weight,
        "theme_risk_flags": theme_risk_flags,
        "capped_symbols": sorted(str(symbol) for symbol in theme_position_limits),
        "theme_constraints": _json_safe(constraints),
        "theme_effect": theme_risk_effect,
        "mode": "risk_shadow_constraints_only" if theme_risk_effect else "risk_shadow_no_theme_effect",
    }
    if not theme_risk_effect:
        monitor.risk_delta = risk_delta
        return

    shadow_decision = _try_rerun_risk_guard(
        artifacts=artifacts,
        constraints=constraints,
        symbols=symbols,
        baseline_gross_cap=baseline_gross_cap,
        baseline_max_weight=baseline_max_weight,
    )
    if shadow_decision is not None:
        risk_delta.update(
            {
                "shadow_action_cap": _action_value(getattr(shadow_decision, "action_cap", "")),
                "shadow_gross_cap": _optional_float(getattr(shadow_decision, "gross_exposure_cap", None)),
                "shadow_max_weight": _optional_float(getattr(shadow_decision, "max_weight", None)),
                "mode": "risk_guard_shadow_rerun",
            }
        )
    else:
        monitor.diagnostic_notes.append("risk_shadow_constraints_only")
    monitor.risk_delta = risk_delta


def _populate_portfolio_shadow(
    *,
    monitor: ThemeShadowMonitor,
    artifacts: Mapping[str, Any],
    global_context: Any,
    max_rows: int,
) -> None:
    portfolio_plan = artifacts.get("portfolio_plan")
    baseline_weights = _weights_from_plan(portfolio_plan)
    symbols = sorted(baseline_weights)
    if global_context is None or not baseline_weights:
        monitor.diagnostic_notes.append("portfolio_shadow_unavailable")
        return

    constraints = _build_theme_portfolio_constraints(
        global_context=global_context,
        symbols=symbols,
        enabled=True,
    )
    exposure_map = _selected_theme_exposure_map(
        exposure_map=constraints.get("theme_exposure_map"),
        baseline_weights=baseline_weights,
    )
    risk_constraints = _theme_risk_constraints_from_monitor(monitor)
    selected_risk_constraints = _selected_theme_risk_constraints(
        constraints=risk_constraints,
        baseline_weights=baseline_weights,
    )
    has_theme_exposure = bool(exposure_map)
    has_selected_theme_risk = _has_selected_theme_risk(
        selected_risk_constraints,
        baseline_weights=baseline_weights,
    )

    baseline_selected = {symbol for symbol, weight in baseline_weights.items() if weight > 0.0}
    monitor.baseline_selected_count = len(baseline_selected)
    monitor.shadow_selected_count = len(baseline_selected)
    monitor.selected_overlap_ratio = 1.0 if baseline_selected else 0.0

    monitor.metadata["theme_portfolio_constraints"] = _json_safe(constraints)
    if selected_risk_constraints:
        monitor.metadata["theme_portfolio_risk_constraints"] = _json_safe(selected_risk_constraints)

    if not has_theme_exposure and not has_selected_theme_risk:
        monitor.theme_exposure_baseline = {}
        monitor.theme_exposure_shadow = {}
        monitor.portfolio_weight_deltas = []
        monitor.diagnostic_notes.append("portfolio_shadow_no_theme_exposure")
        return

    shadow_weights, bound_themes = _apply_lightweight_theme_caps(
        baseline_weights=baseline_weights,
        constraints={**dict(constraints), "theme_exposure_map": exposure_map},
    )
    shadow_weights, risk_bound_symbols, risk_gross_bound = _apply_lightweight_theme_risk_limits(
        baseline_weights=shadow_weights,
        constraints=selected_risk_constraints,
    )

    monitor.theme_exposure_baseline = _theme_exposures(
        weights=baseline_weights,
        exposure_map=exposure_map,
    )
    monitor.theme_exposure_shadow = _theme_exposures(
        weights=shadow_weights,
        exposure_map=exposure_map,
    )
    shadow_selected = {symbol for symbol, weight in shadow_weights.items() if weight > 0.0}
    monitor.shadow_selected_count = len(shadow_selected)
    monitor.selected_overlap_ratio = _overlap_ratio(baseline_selected, shadow_selected)
    deltas = _weight_deltas(
        baseline_weights=baseline_weights,
        shadow_weights=shadow_weights,
        exposure_map=exposure_map,
        max_rows=max_rows,
    )
    monitor.portfolio_weight_deltas = deltas
    if deltas:
        if bound_themes:
            monitor.diagnostic_notes.append("portfolio_shadow_lightweight_cap")
        if risk_bound_symbols or risk_gross_bound:
            monitor.diagnostic_notes.append("portfolio_shadow_theme_risk_cap")
    else:
        monitor.diagnostic_notes.append("portfolio_shadow_no_theme_delta")


def _try_rerun_risk_guard(
    *,
    artifacts: Mapping[str, Any],
    constraints: Mapping[str, Any],
    symbols: list[str],
    baseline_gross_cap: float | None,
    baseline_max_weight: float | None,
) -> Any | None:
    branch_summaries = artifacts.get("branch_summaries")
    if not isinstance(branch_summaries, Mapping) or not branch_summaries:
        return None
    try:
        risk_constraints = {
            "gross_exposure_cap": 1.0 if baseline_gross_cap is None else baseline_gross_cap,
            "max_weight": 1.0 if baseline_max_weight is None else baseline_max_weight,
        }
        risk_constraints.update(copy.deepcopy(dict(constraints)))
        return _risk_guard_cls()().run(
            {
                "branch_verdicts": copy.deepcopy(dict(branch_summaries)),
                "macro_verdict": copy.deepcopy(artifacts.get("macro_verdict")),
                "portfolio_state": {
                    "candidate_symbols": list(symbols),
                    "current_weights": {},
                },
                "constraints": risk_constraints,
            }
        )
    except Exception:
        return None


def _try_rerun_portfolio_constructor(
    *,
    artifacts: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> dict[str, float] | None:
    ic_decisions = artifacts.get("ic_decisions")
    macro_verdict = artifacts.get("macro_verdict")
    risk_decision = artifacts.get("risk_decision")
    tradability_snapshot = artifacts.get("tradability_snapshot")
    if not ic_decisions or macro_verdict is None or risk_decision is None:
        return None
    if not isinstance(tradability_snapshot, Mapping):
        return None
    try:
        risk_limits = {
            "gross_exposure_cap": float(getattr(risk_decision, "gross_exposure_cap", 1.0)),
            "max_weight": float(getattr(risk_decision, "max_weight", 1.0)),
            "position_limits": dict(getattr(risk_decision, "position_limits", {}) or {}),
            "blocked_symbols": list(getattr(risk_decision, "blocked_symbols", []) or []),
            "sector_caps": {},
        }
        risk_limits.update(copy.deepcopy(dict(constraints)))
        plan = _portfolio_constructor_cls()().run(
            {
                "ic_decisions": copy.deepcopy(list(ic_decisions)),
                "macro_verdict": copy.deepcopy(macro_verdict),
                "risk_limits": risk_limits,
                "existing_portfolio": {"current_weights": {}},
                "tradability_snapshot": copy.deepcopy(dict(tradability_snapshot)),
            }
        )
        return _weights_from_plan(plan)
    except Exception:
        return None


def _safe_write_artifact(
    *,
    monitor: ThemeShadowMonitor,
    dag_artifacts: Mapping[str, Any],
    artifact_dir: str | Path,
) -> None:
    try:
        market, universe_key, as_of = _artifact_identity(dag_artifacts)
        root = Path(artifact_dir) / _safe_path_part(market)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{_safe_path_part(as_of)}_{_safe_path_part(universe_key)}_theme_shadow.json"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        previous_path = monitor.artifact_path
        monitor.artifact_path = str(path)
        tmp_path.write_text(
            json.dumps(monitor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception as exc:
        monitor.artifact_path = previous_path if "previous_path" in locals() else monitor.artifact_path
        monitor.diagnostic_notes.append(f"theme_shadow_artifact_write_failed: {exc}")


def _funnel_config_from_artifacts(
    artifacts: Mapping[str, Any],
    diagnostics_module: Any,
) -> Any:
    metadata: Mapping[str, Any] = {}
    funnel_output = artifacts.get("funnel_output")
    if isinstance(getattr(funnel_output, "funnel_metadata", None), Mapping):
        metadata = getattr(funnel_output, "funnel_metadata")
    global_context = artifacts.get("global_context")
    context_metadata = getattr(global_context, "metadata", {}) or {}
    selection_profile = {}
    if isinstance(context_metadata, Mapping):
        selection_profile = context_metadata.get("selection_profile", {}) or {}
    if not isinstance(selection_profile, Mapping):
        selection_profile = {}
    profile = str(metadata.get("profile") or selection_profile.get("funnel_profile") or "classic")
    return diagnostics_module.FunnelConfig(
        max_candidates=int(_optional_float(metadata.get("max_candidates")) or 500),
        profile=profile,
        trend_windows=tuple(
            int(item)
            for item in list(metadata.get("trend_windows") or selection_profile.get("trend_windows") or (20, 60, 120))
        ),
        volume_spike_threshold=float(metadata.get("volume_spike_threshold", 1.35) or 1.35),
        breakout_distance_pct=float(metadata.get("breakout_distance_pct", 0.06) or 0.06),
        sector_bucket_limit=int(_optional_float(metadata.get("sector_bucket_limit")) or 0),
        theme_boost_enabled=False,
        theme_boost_cap=float(metadata.get("theme_boost_cap", 0.10) or 0.10),
    )


def _quant_result(branch_results: Any) -> Any | None:
    if isinstance(branch_results, Mapping):
        return branch_results.get("quant")
    return None


def _candidate_symbols(artifacts: Mapping[str, Any]) -> list[str]:
    symbols: list[str] = []
    funnel_output = artifacts.get("funnel_output")
    symbols.extend(str(symbol) for symbol in list(getattr(funnel_output, "candidates", []) or []))
    if not symbols:
        funnel_summary = artifacts.get("funnel_summary")
        if isinstance(funnel_summary, Mapping):
            symbols.extend(_limited_text_list(funnel_summary.get("candidate_symbols"), 5000))
    if not symbols:
        shortlist = artifacts.get("shortlist")
        if not isinstance(shortlist, (str, bytes)):
            try:
                symbols.extend(str(getattr(item, "symbol", "")) for item in list(shortlist or []))
            except TypeError:
                pass
    if not symbols:
        symbols.extend(_weights_from_plan(artifacts.get("portfolio_plan")))
    return _dedupe(symbols)


def _weights_from_plan(plan: Any) -> dict[str, float]:
    source = getattr(plan, "target_weights", None)
    if not isinstance(source, Mapping):
        source = getattr(plan, "target_positions", None)
    if not isinstance(source, Mapping) and isinstance(plan, Mapping):
        source = plan.get("target_weights") or plan.get("target_positions")
    if not isinstance(source, Mapping):
        return {}
    weights: dict[str, float] = {}
    for symbol, weight in source.items():
        numeric = _optional_float(weight)
        if numeric is None:
            continue
        weights[str(symbol)] = max(0.0, numeric)
    return weights


def _selected_theme_exposure_map(
    *,
    exposure_map: Any,
    baseline_weights: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    if not isinstance(exposure_map, Mapping):
        return {}
    selected = {
        str(symbol)
        for symbol, weight in dict(baseline_weights or {}).items()
        if _finite_float(weight) > 0.0
    }
    result: dict[str, dict[str, Any]] = {}
    for symbol in sorted(selected):
        metadata = exposure_map.get(symbol)
        if not isinstance(metadata, Mapping):
            continue
        theme_id = str(metadata.get("primary_theme_id") or "").strip()
        if not theme_id:
            continue
        result[symbol] = {
            "primary_theme_id": theme_id,
            "primary_theme_name": str(metadata.get("primary_theme_name") or ""),
            "phase": str(metadata.get("phase") or ""),
            "symbol_score": _finite_float(metadata.get("symbol_score", 0.0)),
            "risk_flags": _limited_text_list(metadata.get("risk_flags"), 20),
        }
    return result


def _theme_risk_constraints_from_monitor(monitor: ThemeShadowMonitor) -> dict[str, Any]:
    risk_delta = monitor.risk_delta if isinstance(monitor.risk_delta, Mapping) else {}
    constraints = risk_delta.get("theme_constraints") if isinstance(risk_delta, Mapping) else {}
    return dict(constraints) if isinstance(constraints, Mapping) else {}


def _selected_theme_risk_constraints(
    *,
    constraints: Mapping[str, Any],
    baseline_weights: Mapping[str, float],
) -> dict[str, Any]:
    if not isinstance(constraints, Mapping):
        return {}
    selected = {
        str(symbol)
        for symbol, weight in dict(baseline_weights or {}).items()
        if _finite_float(weight) > 0.0
    }
    if not selected:
        return {}
    risk_by_symbol = {
        str(symbol): dict(metadata)
        for symbol, metadata in dict(constraints.get("theme_risk_by_symbol", {}) or {}).items()
        if str(symbol) in selected and isinstance(metadata, Mapping)
    }
    position_limits = {
        str(symbol): cap
        for symbol, cap in dict(constraints.get("theme_position_limits", {}) or {}).items()
        if str(symbol) in selected
    }
    risk_flags = _limited_text_list(constraints.get("theme_risk_flags"), 50)
    action_cap = str(constraints.get("theme_action_cap") or "").strip()
    gross_cap = _optional_float(constraints.get("theme_gross_exposure_cap"))
    if not risk_by_symbol and not position_limits and not risk_flags and not action_cap and gross_cap is None:
        return {}
    return {
        "theme_risk_by_symbol": risk_by_symbol,
        "theme_position_limits": position_limits,
        "theme_risk_flags": risk_flags,
        "theme_action_cap": action_cap,
        "theme_gross_exposure_cap": gross_cap,
    }


def _has_selected_theme_risk(
    constraints: Mapping[str, Any],
    *,
    baseline_weights: Mapping[str, float],
) -> bool:
    if not isinstance(constraints, Mapping) or not constraints:
        return False
    selected = {
        str(symbol)
        for symbol, weight in dict(baseline_weights or {}).items()
        if _finite_float(weight) > 0.0
    }
    if not selected:
        return False
    risk_by_symbol = dict(constraints.get("theme_risk_by_symbol", {}) or {})
    position_limits = dict(constraints.get("theme_position_limits", {}) or {})
    if any(str(symbol) in selected for symbol in set(risk_by_symbol) | set(position_limits)):
        return True
    return bool(
        _limited_text_list(constraints.get("theme_risk_flags"), 50)
        or str(constraints.get("theme_action_cap") or "").strip()
        or _optional_float(constraints.get("theme_gross_exposure_cap")) is not None
    )


def _apply_lightweight_theme_caps(
    *,
    baseline_weights: Mapping[str, float],
    constraints: Mapping[str, Any],
) -> tuple[dict[str, float], list[str]]:
    adjusted = {
        str(symbol): max(0.0, _finite_float(weight))
        for symbol, weight in dict(baseline_weights or {}).items()
    }
    exposure_map = constraints.get("theme_exposure_map")
    theme_caps = constraints.get("theme_caps")
    if not isinstance(exposure_map, Mapping) or not isinstance(theme_caps, Mapping):
        return adjusted, []
    grouped: dict[str, list[str]] = {}
    for symbol in sorted(adjusted):
        metadata = exposure_map.get(symbol)
        if not isinstance(metadata, Mapping):
            continue
        theme_id = str(metadata.get("primary_theme_id") or "").strip()
        if theme_id and theme_id in theme_caps:
            grouped.setdefault(theme_id, []).append(symbol)
    bound_themes: list[str] = []
    for theme_id in sorted(grouped):
        cap = _optional_float(theme_caps.get(theme_id))
        if cap is None:
            continue
        cap = _clamp(cap, 0.0, 1.0)
        total = sum(adjusted[symbol] for symbol in grouped[theme_id])
        if total <= cap + 1e-8 or total <= 0.0:
            continue
        scale = cap / total
        for symbol in grouped[theme_id]:
            adjusted[symbol] = round(adjusted[symbol] * scale, 6)
        bound_themes.append(theme_id)
    return {symbol: weight for symbol, weight in adjusted.items() if weight > 1e-8}, bound_themes


def _apply_lightweight_theme_risk_limits(
    *,
    baseline_weights: Mapping[str, float],
    constraints: Mapping[str, Any],
) -> tuple[dict[str, float], list[str], bool]:
    adjusted = {
        str(symbol): max(0.0, _finite_float(weight))
        for symbol, weight in dict(baseline_weights or {}).items()
    }
    if not isinstance(constraints, Mapping) or not constraints:
        return adjusted, [], False
    bound_symbols: list[str] = []
    position_limits = constraints.get("theme_position_limits")
    if isinstance(position_limits, Mapping):
        for symbol in sorted(adjusted):
            cap = _optional_float(position_limits.get(symbol))
            if cap is None:
                continue
            capped = min(adjusted[symbol], _clamp(cap, 0.0, 1.0))
            if capped < adjusted[symbol] - 1e-8:
                adjusted[symbol] = round(capped, 6)
                bound_symbols.append(symbol)
    gross_bound = False
    gross_cap = _optional_float(constraints.get("theme_gross_exposure_cap"))
    if gross_cap is not None:
        gross_cap = _clamp(gross_cap, 0.0, 1.0)
        total = sum(adjusted.values())
        if total > gross_cap + 1e-8 and total > 0.0:
            scale = gross_cap / total
            adjusted = {
                symbol: round(weight * scale, 6)
                for symbol, weight in adjusted.items()
                if weight > 0.0
            }
            gross_bound = True
    return {symbol: weight for symbol, weight in adjusted.items() if weight > 1e-8}, bound_symbols, gross_bound



def _theme_exposures(weights: Mapping[str, float], exposure_map: Any) -> dict[str, float]:
    if not isinstance(exposure_map, Mapping):
        return {}
    exposures: dict[str, float] = {}
    for symbol, weight in dict(weights or {}).items():
        metadata = exposure_map.get(symbol)
        if not isinstance(metadata, Mapping):
            continue
        theme_id = str(metadata.get("primary_theme_id") or "").strip()
        if not theme_id:
            continue
        exposures[theme_id] = exposures.get(theme_id, 0.0) + _finite_float(weight)
    return {
        theme_id: round(total, 6)
        for theme_id, total in sorted(exposures.items())
        if total > 1e-8
    }


def _weight_deltas(
    *,
    baseline_weights: Mapping[str, float],
    shadow_weights: Mapping[str, float],
    exposure_map: Any,
    max_rows: int,
) -> list[ThemeShadowDelta]:
    rows: list[ThemeShadowDelta] = []
    symbols = sorted(set(baseline_weights) | set(shadow_weights))
    for symbol in symbols:
        baseline = _finite_float(baseline_weights.get(symbol, 0.0))
        shadow = _finite_float(shadow_weights.get(symbol, 0.0))
        delta = round(shadow - baseline, 12)
        if abs(delta) <= 1e-8:
            continue
        metadata = exposure_map.get(symbol, {}) if isinstance(exposure_map, Mapping) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        rows.append(
            ThemeShadowDelta(
                symbol=symbol,
                baseline_weight=baseline,
                shadow_weight=shadow,
                weight_delta=delta,
                baseline_selected=baseline > 0.0,
                shadow_selected=shadow > 0.0,
                primary_theme_id=str(metadata.get("primary_theme_id") or ""),
                primary_theme_name=str(metadata.get("primary_theme_name") or ""),
                phase=str(metadata.get("phase") or ""),
                risk_flags=_limited_text_list(metadata.get("risk_flags"), 20),
            )
        )
    return sorted(rows, key=lambda item: (-abs(item.weight_delta), item.symbol))[:max_rows]


def _artifact_identity(dag_artifacts: Mapping[str, Any]) -> tuple[str, str, str]:
    global_context = dag_artifacts.get("global_context")
    theme_rotation = dag_artifacts.get("theme_rotation")
    if not isinstance(theme_rotation, Mapping):
        metadata = getattr(global_context, "metadata", {}) if global_context is not None else {}
        theme_rotation = metadata.get("theme_rotation", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(theme_rotation, Mapping):
        theme_rotation = {}
    market = str(theme_rotation.get("market") or getattr(global_context, "market", "") or "unknown")
    universe_key = str(theme_rotation.get("universe_key") or getattr(global_context, "universe_key", "") or "unknown")
    as_of = str(
        theme_rotation.get("as_of")
        or getattr(global_context, "latest_trade_date", "")
        or getattr(global_context, "rebalance_date", "")
        or "unknown"
    )
    return market, universe_key, as_of


def _limit_funnel_diagnostics(payload: Mapping[str, Any], max_rows: int) -> dict[str, Any]:
    result = _json_safe(dict(payload))
    if not isinstance(result, dict):
        return {}
    for key in (
        "entered_symbols",
        "dropped_symbols",
        "improved_symbols",
        "deteriorated_symbols",
        "largest_score_increases",
        "largest_score_decreases",
    ):
        if isinstance(result.get(key), list):
            result[key] = result[key][:max_rows]
    deltas = result.get("deltas_by_symbol")
    if isinstance(deltas, Mapping):
        result["deltas_by_symbol"] = {
            str(symbol): deltas[symbol]
            for symbol in sorted(deltas)[:max_rows]
        }
    return result


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / max(len(union), 1)


def _mapping_markdown(value: Mapping[str, Any], max_rows: int) -> str:
    if not isinstance(value, Mapping) or not value:
        return "_none_"
    rows: list[str] = []
    for key in sorted(value)[:max_rows]:
        item = value.get(key)
        if isinstance(item, Mapping):
            rendered = ", ".join(f"{inner_key}={inner_value}" for inner_key, inner_value in list(item.items())[:max_rows])
        elif isinstance(item, list):
            rendered = ", ".join(str(element) for element in item[:max_rows])
        else:
            rendered = str(item)
        rows.append(f"- {key}: {rendered}")
    return "\n".join(rows) if rows else "_none_"


def _monitor_overlay_kwargs(overlay: Mapping[str, Any]) -> dict[str, Any]:
    modules = overlay.get("theme_overlay_modules")
    if not isinstance(modules, Mapping):
        modules = _default_theme_overlay_modules()
    return {
        "production_decision_source": str(
            overlay.get("production_decision_source") or "no_theme_baseline"
        ),
        "control_decision_source": str(
            overlay.get("control_decision_source") or "no_theme_baseline"
        ),
        "theme_overlay_applied_to_baseline": bool(
            overlay.get("theme_overlay_applied_to_baseline", False)
        ),
        "theme_overlay_modules": {
            str(name): bool(modules.get(name, False))
            for name in _default_theme_overlay_modules()
        },
        "canonical_branch_unchanged": bool(
            overlay.get("canonical_branch_unchanged", True)
        ),
        "theme_likelihood_added": bool(overlay.get("theme_likelihood_added", False)),
        "posterior_formula_changed": bool(
            overlay.get("posterior_formula_changed", False)
        ),
    }


def _module_summary(modules: Mapping[str, Any]) -> str:
    normalized = {
        name: bool(dict(modules or {}).get(name, False))
        for name in _default_theme_overlay_modules()
    }
    return ", ".join(
        f"{name}={str(enabled).lower()}"
        for name, enabled in normalized.items()
    )


def _source_note(overlay_applied: bool) -> str:
    if overlay_applied:
        return (
            "Theme impact toggles are part of the production baseline; "
            "no-theme baseline is retained as control."
        )
    return "Shadow monitor only; final executable decision remains baseline."


def _delta_table(deltas: list[ThemeShadowDelta], max_rows: int) -> str:
    rows = list(deltas[:max_rows])
    if not rows:
        return "_none_"
    lines = [
        "| symbol | baseline_weight | shadow_weight | delta | theme | phase |",
        "|---|---:|---:|---:|---|---|",
    ]
    for delta in rows:
        theme = delta.primary_theme_name or delta.primary_theme_id or "-"
        lines.append(
            f"| {delta.symbol} | {delta.baseline_weight:.6f} | {delta.shadow_weight:.6f} | "
            f"{delta.weight_delta:.6f} | {theme} | {delta.phase or '-'} |"
        )
    return "\n".join(lines)


def _exposure_table(
    baseline: Mapping[str, float],
    shadow: Mapping[str, float],
    max_rows: int,
) -> str:
    theme_ids = sorted(set(baseline) | set(shadow))[:max_rows]
    if not theme_ids:
        return "_none_"
    lines = [
        "| theme | baseline | shadow | delta |",
        "|---|---:|---:|---:|",
    ]
    for theme_id in theme_ids:
        baseline_value = _finite_float(baseline.get(theme_id, 0.0))
        shadow_value = _finite_float(shadow.get(theme_id, 0.0))
        lines.append(
            f"| {theme_id} | {baseline_value:.6f} | {shadow_value:.6f} | {shadow_value - baseline_value:.6f} |"
        )
    return "\n".join(lines)


def _join_symbols(symbols: list[str], max_rows: int) -> str:
    if not symbols:
        return "-"
    suffix = f" (+{len(symbols) - max_rows} more)" if len(symbols) > max_rows else ""
    return ", ".join(str(symbol) for symbol in symbols[:max_rows]) + suffix


def _dedupe(symbols: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        text = str(symbol or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _limited_text_list(value: Any, max_rows: int) -> list[str]:
    if isinstance(value, (str, bytes)):
        values = [str(value)]
    else:
        try:
            values = [str(item) for item in list(value or [])]
        except TypeError:
            values = []
    return _dedupe(values)[:max_rows]


def _action_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _safe_path_part(value: str) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_float(value: Any) -> float:
    numeric = _optional_float(value)
    return 0.0 if numeric is None else numeric


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return str(value.value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return str(value)


def _theme_boost_diagnostics_module() -> Any:
    from quant_investor.funnel import theme_boost_diagnostics

    return theme_boost_diagnostics


def _build_theme_risk_constraints(**kwargs: Any) -> dict[str, Any]:
    from quant_investor.market.dag.theme_context import build_theme_risk_constraints

    return build_theme_risk_constraints(**kwargs)


def _build_theme_portfolio_constraints(**kwargs: Any) -> dict[str, Any]:
    from quant_investor.market.dag.theme_context import build_theme_portfolio_constraints

    return build_theme_portfolio_constraints(**kwargs)


def _risk_guard_cls() -> Any:
    from quant_investor.agents.risk_guard import RiskGuard

    return RiskGuard


def _portfolio_constructor_cls() -> Any:
    from quant_investor.agents.portfolio_constructor import PortfolioConstructor

    return PortfolioConstructor
