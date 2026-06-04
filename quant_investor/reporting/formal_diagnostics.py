from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


SUPPORTED_WARNING_CODES = {
    "provider_missing",
    "snapshot_missing",
    "stale_snapshot",
    "llm_confidence_unavailable",
    "retired_signal_suppressed",
}
SUPPORTED_WARNING_SCOPES = {"global", "universe", "holding", "branch", "review_layer"}
SUPPORTED_WARNING_SEVERITIES = {"info", "warning", "material"}
SUPPORTED_BRANCH_VS_FINAL = {
    "aligned",
    "conflict_requires_arbitration",
    "conflict_downgraded",
    "insufficient_evidence",
    "unknown",
}
SUPPORTED_DECISION_IMPACTS = {
    "none",
    "disclosure_only",
    "downgraded_final_label",
    "requires_arbitration",
}
SUPPORTED_DISPLAY_LABELS = {
    "hold",
    "no_action",
    "rebalance",
    "no_action_evidence_impaired",
    "hold_arbitrated",
    "watch",
    "reduce_watch",
    "unknown",
}

_WARNING_SEVERITY_RANK = {"material": 0, "warning": 1, "info": 2}
_WARNING_SCOPE_RANK = {"global": 0, "universe": 1, "holding": 2, "branch": 3, "review_layer": 4}
_FUNDAMENTAL_MODULE_LABELS = {
    "financial_quality": "财务质量",
    "forecast_revision": "盈利预测",
    "valuation": "估值",
    "management_governance": "管理层治理",
    "ownership": "股东结构",
    "document_semantics": "文档语义",
}


def _require_choice(name: str, value: str, allowed: set[str]) -> str:
    text = str(value or "").strip()
    if text not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return text


def _require_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _coerce_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _dedupe_texts(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_date_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return text
    if len(text) >= 10:
        candidate = text[:10]
        try:
            return datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
        except ValueError:
            return text
    return text


def _parse_date(value: Any) -> datetime | None:
    text = _normalize_date_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _is_date_before(left: Any, right: Any) -> bool:
    left_dt = _parse_date(left)
    right_dt = _parse_date(right)
    if left_dt is None or right_dt is None:
        return False
    return left_dt.date() < right_dt.date()


def _escape_markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _stable_sorted_codes(codes: Iterable[str]) -> list[str]:
    return sorted(_dedupe_texts(str(code or "").strip() for code in codes if str(code or "").strip()))


def _review_layer_uses_codex_handoff(payload: Mapping[str, Any]) -> bool:
    data = _coerce_mapping(payload)
    if bool(data.get("codex_handoff")):
        return True
    if bool(data.get("local_llm_disabled")):
        return True
    fallback_text = " ".join(str(item or "").strip().lower() for item in _coerce_sequence(data.get("fallback_reasons")))
    return "codex_handoff" in fallback_text or "local_llm_disabled" in fallback_text


def _has_quote_snapshot(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and text.upper() not in {"N/A", "NONE", "NULL"})


def _dominant_full_a_coverage_ratio(completeness: Mapping[str, Any], dominant_date: str | None) -> float:
    normalized_dominant = (_normalize_date_text(dominant_date) or "").replace("-", "")
    categories = _coerce_mapping(completeness.get("categories"))
    full_a = _coerce_mapping(categories.get("full_a"))
    expected = int(full_a.get("expected", 0) or 0)
    if expected <= 0:
        return 0.0
    date_counts = _coerce_mapping(full_a.get("date_counts"))
    count = int(date_counts.get(normalized_dominant, 0) or 0)
    return count / expected


def is_previous_day_realtime_decision_sufficient(
    *,
    target_date: str | None,
    dominant_local_snapshot_date: str | None,
    completeness_state: Mapping[str, Any] | None = None,
    quote_snapshot: str | None = None,
) -> bool:
    """Return whether previous-day daily bars plus realtime quotes are enough for intraday decisions."""

    completeness = _coerce_mapping(completeness_state)
    normalized_target = _normalize_date_text(target_date)
    normalized_dominant = _normalize_date_text(dominant_local_snapshot_date)
    normalized_stable = _normalize_date_text(completeness.get("stable_trade_date"))
    normalized_strict = _normalize_date_text(completeness.get("strict_trade_date"))
    if not normalized_target or not normalized_dominant:
        return False
    if not _is_date_before(normalized_dominant, normalized_target):
        return False
    if normalized_stable and normalized_dominant != normalized_stable:
        return False
    if normalized_strict and normalized_strict != normalized_target:
        return False
    if not _has_quote_snapshot(quote_snapshot or completeness.get("quote_snapshot")):
        return False

    coverage_threshold = float(completeness.get("coverage_threshold", 0.95) or 0.95)
    if _dominant_full_a_coverage_ratio(completeness, normalized_dominant) < coverage_threshold:
        return False
    return True


def _final_label_from_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    mapping = {
        "hold": "hold",
        "continue_hold": "hold",
        "继续持有": "hold",
        "持有": "hold",
        "core_hold": "hold",
        "stable_hold": "hold",
        "watch": "watch",
        "observe": "watch",
        "继续观察": "watch",
        "观察": "watch",
        "no_action": "no_action",
        "rebalance": "rebalance",
        "减仓待确认": "reduce_watch",
        "reduce_watch": "reduce_watch",
        "sell": "reduce_watch",
        "avoid": "reduce_watch",
        "no_action_evidence_impaired": "no_action_evidence_impaired",
        "hold_arbitrated": "hold_arbitrated",
    }
    return mapping.get(text, "unknown")


def _warning_sort_key(warning: "ReportWarning") -> tuple[Any, ...]:
    return (
        _WARNING_SEVERITY_RANK[warning.severity],
        _WARNING_SCOPE_RANK[warning.scope],
        warning.affected_symbol or "",
        warning.code,
        warning.source,
        warning.data_date or "",
        warning.human_message,
    )


@dataclass(frozen=True)
class ReportWarning:
    code: str
    scope: str
    source: str
    severity: str
    data_date: str | None
    affected_symbol: str | None
    decision_impact: str
    human_message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_choice("code", self.code, SUPPORTED_WARNING_CODES))
        object.__setattr__(self, "scope", _require_choice("scope", self.scope, SUPPORTED_WARNING_SCOPES))
        object.__setattr__(self, "severity", _require_choice("severity", self.severity, SUPPORTED_WARNING_SEVERITIES))
        object.__setattr__(self, "source", _require_text("source", self.source))
        object.__setattr__(
            self,
            "decision_impact",
            _require_choice("decision_impact", self.decision_impact, SUPPORTED_DECISION_IMPACTS),
        )
        object.__setattr__(self, "human_message", _require_text("human_message", self.human_message))
        object.__setattr__(self, "data_date", _normalize_date_text(self.data_date))
        symbol = str(self.affected_symbol or "").strip().upper()
        object.__setattr__(self, "affected_symbol", symbol or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "scope": self.scope,
            "source": self.source,
            "severity": self.severity,
            "data_date": self.data_date,
            "affected_symbol": self.affected_symbol,
            "decision_impact": self.decision_impact,
            "human_message": self.human_message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReportWarning":
        data = _coerce_mapping(payload)
        return cls(
            code=str(data.get("code", "")),
            scope=str(data.get("scope", "")),
            source=str(data.get("source", "")),
            severity=str(data.get("severity", "")),
            data_date=data.get("data_date"),
            affected_symbol=data.get("affected_symbol"),
            decision_impact=str(data.get("decision_impact", "")),
            human_message=str(data.get("human_message", "")),
        )


@dataclass(frozen=True)
class HoldingDecisionDiagnostic:
    symbol: str
    name: str
    data_date: str | None
    final_label: str
    branch_vs_final: str
    llm_confidence: float | None
    warning_codes: list[str] = field(default_factory=list)
    decision_impact: str = "none"
    arbitration_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_text("symbol", self.symbol).upper())
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "data_date", _normalize_date_text(self.data_date))
        object.__setattr__(self, "final_label", _final_label_from_value(self.final_label))
        object.__setattr__(
            self,
            "branch_vs_final",
            _require_choice("branch_vs_final", self.branch_vs_final, SUPPORTED_BRANCH_VS_FINAL),
        )
        if self.llm_confidence is None:
            llm_confidence = None
        else:
            llm_confidence = float(self.llm_confidence)
        object.__setattr__(self, "llm_confidence", llm_confidence)
        object.__setattr__(self, "warning_codes", _stable_sorted_codes(self.warning_codes))
        object.__setattr__(
            self,
            "decision_impact",
            _require_choice("decision_impact", self.decision_impact, SUPPORTED_DECISION_IMPACTS),
        )
        object.__setattr__(self, "arbitration_note", str(self.arbitration_note or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "data_date": self.data_date,
            "final_label": self.final_label,
            "branch_vs_final": self.branch_vs_final,
            "llm_confidence": self.llm_confidence,
            "warning_codes": list(self.warning_codes),
            "decision_impact": self.decision_impact,
            "arbitration_note": self.arbitration_note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HoldingDecisionDiagnostic":
        data = _coerce_mapping(payload)
        return cls(
            symbol=str(data.get("symbol", "")),
            name=str(data.get("name", "")),
            data_date=data.get("data_date"),
            final_label=str(data.get("final_label", "")),
            branch_vs_final=str(data.get("branch_vs_final", "")),
            llm_confidence=_safe_float(data.get("llm_confidence")),
            warning_codes=list(data.get("warning_codes", []) or []),
            decision_impact=str(data.get("decision_impact", "none")),
            arbitration_note=str(data.get("arbitration_note", "")),
        )


@dataclass(frozen=True)
class ReportDecisionGuardrailResult:
    provisional_label: str
    display_label: str
    material_warning_count: int
    all_zero_llm_confidence: bool
    triggered_warning_codes: list[str] = field(default_factory=list)
    arbitration_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provisional_label", _final_label_from_value(self.provisional_label))
        object.__setattr__(self, "display_label", _final_label_from_value(self.display_label))
        if self.display_label not in SUPPORTED_DISPLAY_LABELS:
            raise ValueError(f"display_label must be one of {sorted(SUPPORTED_DISPLAY_LABELS)}")
        object.__setattr__(self, "material_warning_count", int(self.material_warning_count))
        object.__setattr__(self, "all_zero_llm_confidence", bool(self.all_zero_llm_confidence))
        object.__setattr__(self, "triggered_warning_codes", _stable_sorted_codes(self.triggered_warning_codes))
        object.__setattr__(self, "arbitration_note", str(self.arbitration_note or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provisional_label": self.provisional_label,
            "display_label": self.display_label,
            "material_warning_count": self.material_warning_count,
            "all_zero_llm_confidence": self.all_zero_llm_confidence,
            "triggered_warning_codes": list(self.triggered_warning_codes),
            "arbitration_note": self.arbitration_note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReportDecisionGuardrailResult":
        data = _coerce_mapping(payload)
        return cls(
            provisional_label=str(data.get("provisional_label", "")),
            display_label=str(data.get("display_label", "")),
            material_warning_count=int(data.get("material_warning_count", 0) or 0),
            all_zero_llm_confidence=bool(data.get("all_zero_llm_confidence", False)),
            triggered_warning_codes=list(data.get("triggered_warning_codes", []) or []),
            arbitration_note=str(data.get("arbitration_note", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _warning_dedupe_key(warning: ReportWarning) -> tuple[Any, ...]:
    return (
        warning.code,
        warning.scope,
        warning.source,
        warning.data_date,
        warning.affected_symbol,
        warning.decision_impact,
        warning.human_message,
    )


def _dedupe_and_sort_warnings(warnings: Iterable[ReportWarning]) -> list[ReportWarning]:
    unique: dict[tuple[Any, ...], ReportWarning] = {}
    for warning in warnings:
        unique.setdefault(_warning_dedupe_key(warning), warning)
    return sorted(unique.values(), key=_warning_sort_key)


def _fundamental_payload_for_symbol(
    symbol: str,
    branch_diagnostics: Mapping[str, Any],
    fundamental_coverage_by_symbol: Mapping[str, Any],
) -> dict[str, Any]:
    direct = _coerce_mapping(fundamental_coverage_by_symbol.get(symbol))
    if direct:
        return direct
    symbol_payload = _coerce_mapping(branch_diagnostics.get(symbol))
    branch_verdicts = _coerce_mapping(symbol_payload.get("reviewed_branch_verdicts") or symbol_payload.get("branch_verdicts"))
    fundamental = _coerce_mapping(branch_verdicts.get("fundamental"))
    metadata = _coerce_mapping(fundamental.get("metadata"))
    data_quality = _coerce_mapping(metadata.get("data_quality"))
    return {
        "coverage_ratio": data_quality.get("coverage_ratio"),
        "missing_modules": (
            _coerce_mapping(data_quality.get("missing_modules")).get(symbol)
            if isinstance(data_quality.get("missing_modules"), Mapping)
            else data_quality.get("missing_modules")
        ),
        "snapshot_quality": (
            _coerce_mapping(data_quality.get("snapshot_quality_by_symbol")).get(symbol)
            if isinstance(data_quality.get("snapshot_quality_by_symbol"), Mapping)
            else data_quality.get("snapshot_quality")
        ),
        "module_coverage": metadata.get("module_coverage"),
        "conclusion": symbol_payload.get("llm_conclusion") or fundamental.get("thesis"),
    }


def _extract_intelligence_payloads(
    branch_diagnostics: Mapping[str, Any],
    intelligence_diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for payload in intelligence_diagnostics.values():
        mapping = _coerce_mapping(payload)
        if mapping:
            collected.append(mapping)
    if collected:
        return collected

    for symbol_payload in branch_diagnostics.values():
        mapping = _coerce_mapping(symbol_payload)
        branch_verdicts = _coerce_mapping(mapping.get("reviewed_branch_verdicts") or mapping.get("branch_verdicts"))
        intelligence = _coerce_mapping(branch_verdicts.get("intelligence"))
        if intelligence:
            collected.append(intelligence)
    return collected


def _extract_enhanced_data_flags_for_symbol(
    symbol: str,
    branch_diagnostics: Mapping[str, Any],
    enhanced_data_flags_by_symbol: Mapping[str, Any],
) -> dict[str, Any]:
    direct = _coerce_mapping(enhanced_data_flags_by_symbol.get(symbol))
    if direct:
        return direct
    payload = _fundamental_payload_for_symbol(symbol, branch_diagnostics, {})
    return _coerce_mapping(payload.get("snapshot_quality"))


def _module_label(name: str) -> str:
    return _FUNDAMENTAL_MODULE_LABELS.get(name, name)


def _classify_missing_modules(
    *,
    missing_modules: Sequence[str],
    snapshot_quality: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    provider_missing_modules: list[str] = []
    snapshot_missing_modules: list[str] = []
    unresolved_modules: list[str] = []
    normalized_quality = _coerce_mapping(snapshot_quality)

    for module_name in missing_modules:
        quality = _coerce_mapping(normalized_quality.get(module_name))
        label = _module_label(str(module_name))
        missing_scope = str(quality.get("missing_scope", "")).strip().lower()
        provider_missing = bool(quality.get("provider_missing", False))
        snapshot_missing = bool(quality.get("snapshot_missing", False))
        if provider_missing or missing_scope == "global":
            provider_missing_modules.append(label)
        elif snapshot_missing or missing_scope == "symbol" or quality.get("status") == "missing_symbol":
            snapshot_missing_modules.append(label)
        else:
            unresolved_modules.append(label)
    return (
        _dedupe_texts(provider_missing_modules),
        _dedupe_texts(snapshot_missing_modules),
        _dedupe_texts(unresolved_modules),
    )


def collect_formal_report_warnings(
    *,
    target_date: str | None,
    dominant_local_snapshot_date: str | None,
    completeness_state: Mapping[str, Any] | None = None,
    holdings_review: Sequence[Mapping[str, Any]] | None = None,
    branch_diagnostics: Mapping[str, Any] | None = None,
    fundamental_coverage_by_symbol: Mapping[str, Any] | None = None,
    enhanced_data_flags_by_symbol: Mapping[str, Any] | None = None,
    intelligence_diagnostics: Mapping[str, Any] | None = None,
    review_layer_diagnostics: Mapping[str, Any] | None = None,
) -> list[ReportWarning]:
    completeness = _coerce_mapping(completeness_state)
    branch_diagnostics = _coerce_mapping(branch_diagnostics)
    fundamental_coverage_by_symbol = _coerce_mapping(fundamental_coverage_by_symbol)
    enhanced_data_flags_by_symbol = _coerce_mapping(enhanced_data_flags_by_symbol)
    intelligence_diagnostics = _coerce_mapping(intelligence_diagnostics)
    review_layer_diagnostics = _coerce_mapping(review_layer_diagnostics)
    holdings = list(holdings_review or [])

    warnings: list[ReportWarning] = []
    blocking_gap = int(completeness.get("blocking_incomplete_count", 0) or 0)
    complete = bool(completeness.get("complete", True))
    normalized_target_date = _normalize_date_text(target_date)
    normalized_dominant_date = _normalize_date_text(dominant_local_snapshot_date)
    if normalized_target_date and normalized_dominant_date and _is_date_before(normalized_dominant_date, normalized_target_date):
        previous_day_realtime_ok = is_previous_day_realtime_decision_sufficient(
            target_date=normalized_target_date,
            dominant_local_snapshot_date=normalized_dominant_date,
            completeness_state=completeness,
        )
        severity = "info" if previous_day_realtime_ok else ("material" if blocking_gap > 0 or not complete else "warning")
        warnings.append(
            ReportWarning(
                code="stale_snapshot",
                scope="global",
                source="completeness/local_snapshot",
                severity=severity,
                data_date=normalized_dominant_date,
                affected_symbol=None,
                decision_impact="downgraded_final_label" if severity == "material" else "disclosure_only",
                human_message=(
                    (
                        f"盘中复盘口径采用 {normalized_dominant_date} 稳定日线并结合实时行情，"
                        f"{normalized_target_date} 当日日线未广泛可用不视为决策阻断。"
                    )
                    if previous_day_realtime_ok
                    else (
                        f"本地主导快照日期 {normalized_dominant_date} 早于报告目标日期 {normalized_target_date}"
                        + (f"，阻塞缺口 {blocking_gap} 个。" if blocking_gap > 0 else "。")
                    )
                ),
            )
        )

    symbols: list[str] = sorted(
        {
            *(str(item.get("symbol", "")).strip().upper() for item in holdings if str(item.get("symbol", "")).strip()),
            *branch_diagnostics.keys(),
            *fundamental_coverage_by_symbol.keys(),
            *enhanced_data_flags_by_symbol.keys(),
        }
    )

    for symbol in symbols:
        coverage_payload = _fundamental_payload_for_symbol(symbol, branch_diagnostics, fundamental_coverage_by_symbol)
        missing_modules = [
            str(item).strip()
            for item in _coerce_sequence(coverage_payload.get("missing_modules"))
            if str(item).strip()
        ]
        snapshot_quality = _extract_enhanced_data_flags_for_symbol(symbol, branch_diagnostics, enhanced_data_flags_by_symbol)
        provider_modules, snapshot_modules, unresolved_modules = _classify_missing_modules(
            missing_modules=missing_modules,
            snapshot_quality=snapshot_quality,
        )
        if provider_modules:
            warnings.append(
                ReportWarning(
                    code="provider_missing",
                    scope="holding",
                    source="enhanced_data_layer/fundamental",
                    severity="warning",
                    data_date=normalized_dominant_date,
                    affected_symbol=symbol,
                    decision_impact="disclosure_only",
                    human_message=f"{symbol} 缺少 provider 级基本面模块：{'、'.join(provider_modules)}。",
                )
            )
        if snapshot_modules or unresolved_modules:
            named_modules = snapshot_modules + [item for item in unresolved_modules if item not in snapshot_modules]
            warnings.append(
                ReportWarning(
                    code="snapshot_missing",
                    scope="holding",
                    source="fundamental/module_coverage",
                    severity="warning",
                    data_date=normalized_dominant_date,
                    affected_symbol=symbol,
                    decision_impact="disclosure_only",
                    human_message=f"{symbol} 缺少 symbol 级基本面模块：{'、'.join(named_modules)}。",
                )
            )

    retired_detected = False
    for payload in _extract_intelligence_payloads(branch_diagnostics, intelligence_diagnostics):
        coverage_notes = [str(item).strip().lower() for item in _coerce_sequence(payload.get("coverage_notes"))]
        investment_risks = [str(item).strip().lower() for item in _coerce_sequence(payload.get("investment_risks"))]
        metadata = _coerce_mapping(payload.get("metadata"))
        branch_mode = str(metadata.get("branch_mode", "")).strip().lower()
        if (
            "legacy batch retired" in " ".join(coverage_notes)
            or "旧 batch pipeline" in " ".join(investment_risks)
            or branch_mode == "structured_intelligence_fusion"
        ):
            retired_detected = True
            break
    if retired_detected:
        warnings.append(
            ReportWarning(
                code="retired_signal_suppressed",
                scope="branch",
                source="intelligence/structured_fusion",
                severity="info",
                data_date=normalized_dominant_date,
                affected_symbol=None,
                decision_impact="disclosure_only",
                human_message="旧 intelligence batch 路径已退役，当前分支使用 lightweight structured fusion，这属于设计路径，不是数据缺失。",
            )
        )

    per_symbol_zero_confidence: list[str] = []
    codex_handoff_active = _review_layer_uses_codex_handoff(review_layer_diagnostics)
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        llm_confidence = _safe_float(holding.get("llm_confidence"))
        llm_effective_calls = int(holding.get("llm_effective_calls", 0) or 0)
        llm_degraded = bool(holding.get("llm_degraded", False))
        confidence_source = str(holding.get("llm_confidence_source", "")).strip()
        if (
            not codex_handoff_active
            and (llm_confidence is None or llm_effective_calls <= 0 or llm_degraded or not confidence_source)
        ):
            warnings.append(
                ReportWarning(
                    code="llm_confidence_unavailable",
                    scope="holding",
                    source="review_layer/confidence",
                    severity="warning",
                    data_date=normalized_dominant_date,
                    affected_symbol=symbol,
                    decision_impact="disclosure_only",
                    human_message=f"{symbol} 缺少可用的 structured LLM confidence 语义，当前置信度不能视为完整有效。",
                )
            )
        if llm_confidence is not None and abs(llm_confidence) <= 1e-12:
            per_symbol_zero_confidence.append(symbol)

    effective_call_count = int(review_layer_diagnostics.get("effective_call_count", 0) or 0)
    if effective_call_count <= 0 and not codex_handoff_active:
        warnings.append(
            ReportWarning(
                code="llm_confidence_unavailable",
                scope="review_layer",
                source="review_layer/effective_calls",
                severity="material",
                data_date=normalized_dominant_date,
                affected_symbol=None,
                decision_impact="downgraded_final_label",
                human_message="review-layer 没有可用的有效置信度调用结果，正式结论需要按证据受损处理。",
            )
        )
    elif holdings and len(per_symbol_zero_confidence) == len([item for item in holdings if str(item.get("symbol", "")).strip()]):
        warnings.append(
            ReportWarning(
                code="llm_confidence_unavailable",
                scope="review_layer",
                source="review_layer/all_zero_confidence",
                severity="material",
                data_date=normalized_dominant_date,
                affected_symbol=None,
                decision_impact="downgraded_final_label",
                human_message="所有持仓的 structured LLM confidence 均为 0.00，正式结论需要按证据受损处理。",
            )
        )

    return _dedupe_and_sort_warnings(warnings)


def reconcile_branch_vs_final(
    *,
    symbol: str,
    provisional_final_label: str,
    holding_review: Mapping[str, Any] | None = None,
    branch_signals: Mapping[str, Any] | None = None,
    warnings: Sequence[ReportWarning] | None = None,
) -> tuple[str, str]:
    normalized_symbol = _require_text("symbol", symbol).upper()
    holding_review = _coerce_mapping(holding_review)
    branch_signals = _coerce_mapping(branch_signals)
    final_label = _final_label_from_value(provisional_final_label)
    clean_hold_like = final_label in {"hold", "watch", "no_action"}

    structured_actions: list[tuple[str, str]] = []
    llm_action = _final_label_from_value(holding_review.get("llm_action"))
    if llm_action != "unknown":
        structured_actions.append(("holding.llm_action", llm_action))

    recommended_action = _final_label_from_value(holding_review.get("recommended_action"))
    if recommended_action != "unknown":
        structured_actions.append(("holding.recommended_action", recommended_action))

    recommendation = _coerce_mapping(branch_signals.get("recommendation"))
    recommendation_action = _final_label_from_value(recommendation.get("action"))
    if recommendation_action != "unknown":
        structured_actions.append(("review.recommendation", recommendation_action))

    ic_hint = _coerce_mapping(branch_signals.get("ic_hint"))
    ic_action = _final_label_from_value(ic_hint.get("action"))
    if ic_action != "unknown":
        structured_actions.append(("review.ic_hint", ic_action))

    branch_overlays = _coerce_mapping(branch_signals.get("branch_overlays"))
    for branch_name, payload in branch_overlays.items():
        action = _final_label_from_value(_coerce_mapping(payload).get("action"))
        if action != "unknown":
            structured_actions.append((f"overlay.{branch_name}", action))

    reviewed_branchs = _coerce_mapping(branch_signals.get("reviewed_branch_verdicts") or branch_signals.get("branch_verdicts"))
    for branch_name, payload in reviewed_branchs.items():
        action = _final_label_from_value(_coerce_mapping(payload).get("action"))
        if action != "unknown":
            structured_actions.append((f"branch.{branch_name}", action))

    conflict_sources = [source for source, action in structured_actions if action in {"reduce_watch"}]
    if clean_hold_like and conflict_sources:
        note = (
            f"{normalized_symbol} 存在结构化分支减仓/回避信号（{', '.join(conflict_sources[:4])}），"
            f"但当前显示标签仍维持 {final_label}，需要显式仲裁说明。"
        )
        return "conflict_requires_arbitration", note

    material_warning_codes = {
        item.code
        for item in warnings or []
        if item.severity == "material" or item.decision_impact == "downgraded_final_label"
    }
    if clean_hold_like and material_warning_codes.intersection(
        {"stale_snapshot", "provider_missing", "snapshot_missing", "llm_confidence_unavailable"}
    ):
        return (
            "insufficient_evidence",
            f"{normalized_symbol} 当前没有足够完整的结构化证据支持 clean {final_label} 展示标签。",
        )

    if structured_actions:
        return "aligned", ""

    prose = " ".join(
        [
            str(holding_review.get("llm_conclusion", "")).strip().lower(),
            str(branch_signals.get("report_excerpt", "")).strip().lower(),
        ]
    )
    bearish_tokens = ("卖出", "bearish", "downside", "减仓", "回避")
    if clean_hold_like and any(token in prose for token in bearish_tokens):
        return (
            "conflict_requires_arbitration",
            f"{normalized_symbol} 的自由文本结论包含 bearish/sell 语义，但缺少足够的结构化仲裁信息。",
        )

    return ("unknown", "") if not structured_actions else ("aligned", "")


def build_holding_decision_diagnostics(
    *,
    holdings_review: Sequence[Mapping[str, Any]],
    warnings: Sequence[ReportWarning],
    provisional_label_by_symbol: Mapping[str, str] | None = None,
    data_date_by_symbol: Mapping[str, str | None] | None = None,
    branch_signals_by_symbol: Mapping[str, Any] | None = None,
) -> list[HoldingDecisionDiagnostic]:
    provisional_label_by_symbol = _coerce_mapping(provisional_label_by_symbol)
    data_date_by_symbol = _coerce_mapping(data_date_by_symbol)
    branch_signals_by_symbol = _coerce_mapping(branch_signals_by_symbol)

    diagnostics: list[HoldingDecisionDiagnostic] = []
    for holding in holdings_review:
        symbol = str(holding.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        related_warnings = [
            warning
            for warning in warnings
            if warning.affected_symbol in (None, symbol)
        ]
        provisional_label = provisional_label_by_symbol.get(symbol) or holding.get("recommended_action") or holding.get("llm_action") or "unknown"
        branch_vs_final, arbitration_note = reconcile_branch_vs_final(
            symbol=symbol,
            provisional_final_label=str(provisional_label),
            holding_review=holding,
            branch_signals=_coerce_mapping(branch_signals_by_symbol.get(symbol)),
            warnings=related_warnings,
        )
        if branch_vs_final == "conflict_requires_arbitration":
            decision_impact = "requires_arbitration"
        elif any(warning.severity == "material" for warning in related_warnings):
            decision_impact = "downgraded_final_label"
        elif related_warnings:
            decision_impact = "disclosure_only"
        else:
            decision_impact = "none"
        diagnostics.append(
            HoldingDecisionDiagnostic(
                symbol=symbol,
                name=str(holding.get("name", symbol)),
                data_date=data_date_by_symbol.get(symbol),
                final_label=str(provisional_label),
                branch_vs_final=branch_vs_final if branch_vs_final in SUPPORTED_BRANCH_VS_FINAL else "unknown",
                llm_confidence=_safe_float(holding.get("llm_confidence")),
                warning_codes=[warning.code for warning in related_warnings],
                decision_impact=decision_impact,
                arbitration_note=arbitration_note,
            )
        )
    return diagnostics


def apply_report_decision_guardrail(
    *,
    provisional_label: str,
    warnings: Sequence[ReportWarning],
    holding_diagnostics: Sequence[HoldingDecisionDiagnostic],
    llm_confidences: Sequence[float | None] | None = None,
) -> ReportDecisionGuardrailResult:
    normalized_provisional = _final_label_from_value(provisional_label)
    confidences = [value for value in (llm_confidences or [])]
    non_none_confidences = [float(value) for value in confidences if value is not None]
    all_zero_llm_confidence = bool(non_none_confidences) and all(abs(value) <= 1e-12 for value in non_none_confidences)

    material_warnings = [warning for warning in warnings if warning.severity == "material"]
    triggered_codes = [warning.code for warning in material_warnings]
    display_label = normalized_provisional
    arbitration_note = ""

    clean_hold_like = normalized_provisional in {"hold", "no_action", "watch"}
    conflicts = [item for item in holding_diagnostics if item.branch_vs_final == "conflict_requires_arbitration"]
    has_material_blockers = bool(material_warnings) or all_zero_llm_confidence
    if clean_hold_like and has_material_blockers:
        safe_arbitration = (
            conflicts
            and all(item.arbitration_note for item in conflicts)
            and not any(
                warning.code in {"stale_snapshot", "provider_missing", "snapshot_missing", "llm_confidence_unavailable"}
                and warning.severity == "material"
                for warning in material_warnings
            )
            and not all_zero_llm_confidence
        )
        if safe_arbitration:
            display_label = "hold_arbitrated"
            arbitration_note = "存在冲突分支信号，但结构化仲裁信息完整，报告显示标签已按仲裁方式降级。"
        else:
            display_label = "no_action_evidence_impaired"
            if material_warnings and all_zero_llm_confidence:
                arbitration_note = "当前 formal review 同时存在 material 级证据缺口和全零 LLM confidence，报告显示标签已降级。"
            elif material_warnings:
                arbitration_note = "当前 formal review 存在 material 级证据缺口，报告显示标签已降级。"
            else:
                arbitration_note = "当前 formal review 存在全零 LLM confidence，报告显示标签已降级。"
    elif normalized_provisional in {"hold_arbitrated", "no_action_evidence_impaired"}:
        display_label = normalized_provisional

    if not arbitration_note and conflicts:
        arbitration_note = "；".join(item.arbitration_note for item in conflicts if item.arbitration_note)

    return ReportDecisionGuardrailResult(
        provisional_label=normalized_provisional,
        display_label=display_label,
        material_warning_count=len(material_warnings),
        all_zero_llm_confidence=all_zero_llm_confidence,
        triggered_warning_codes=triggered_codes,
        arbitration_note=arbitration_note,
        metadata={
            "conflict_count": len(conflicts),
            "clean_hold_like": clean_hold_like,
            "triggered_warning_count": len(triggered_codes),
        },
    )


def render_holding_diagnostic_markdown_table(
    diagnostics: Sequence[HoldingDecisionDiagnostic],
) -> str:
    header = "| symbol | name | data_date | final_label | branch_vs_final | llm_confidence | warning_codes | decision_impact | arbitration_note |"
    if not diagnostics:
        return header + "\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| | | | | | | | | no holding diagnostics available |"

    lines = [
        header,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in diagnostics:
        llm_confidence = "" if item.llm_confidence is None else f"{item.llm_confidence:.2f}"
        warning_codes = ",".join(_stable_sorted_codes(item.warning_codes))
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown_cell(item.symbol),
                    _escape_markdown_cell(item.name),
                    _escape_markdown_cell(item.data_date or ""),
                    _escape_markdown_cell(item.final_label),
                    _escape_markdown_cell(item.branch_vs_final),
                    _escape_markdown_cell(llm_confidence),
                    _escape_markdown_cell(warning_codes),
                    _escape_markdown_cell(item.decision_impact),
                    _escape_markdown_cell(item.arbitration_note),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


__all__ = [
    "HoldingDecisionDiagnostic",
    "ReportDecisionGuardrailResult",
    "ReportWarning",
    "apply_report_decision_guardrail",
    "build_holding_decision_diagnostics",
    "collect_formal_report_warnings",
    "is_previous_day_realtime_decision_sufficient",
    "render_holding_diagnostic_markdown_table",
    "reconcile_branch_vs_final",
]
