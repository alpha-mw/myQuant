from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

from quant_investor.themes.replay import (
    ThemeCalibrationDataset,
    ThemeReplayRecord,
)


_BASE_METADATA = {
    "deterministic": True,
    "no_llm": True,
    "no_network": True,
    "offline_only": True,
}
_PHASE_THRESHOLDS = (
    "accumulation",
    "early_acceleration",
    "confirmed_rotation",
    "overextended",
    "distribution",
)
_RISK_FLAG_THRESHOLDS = (
    "theme_overextended",
    "theme_fake_breakout_risk",
    "theme_low_breadth",
    "theme_distribution_risk",
)


@dataclass
class CalibrationMetricSummary:
    group_key: str
    group_name: str = ""
    record_count: int = 0
    available_count: int = 0

    avg_symbol_theme_score: float | None = None
    avg_theme_score: float | None = None

    avg_forward_alpha_5d: float | None = None
    avg_forward_alpha_10d: float | None = None
    avg_forward_alpha_20d: float | None = None

    median_forward_alpha_5d: float | None = None
    median_forward_alpha_10d: float | None = None

    hit_rate_5d: float | None = None
    hit_rate_10d: float | None = None

    avg_max_drawdown_10d: float | None = None
    avg_max_runup_10d: float | None = None

    evidence_quality: str = "insufficient"
    diagnostic_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdDiagnostic:
    threshold_name: str
    threshold_value: float | str
    selected_count: int = 0
    available_count: int = 0
    coverage_ratio: float = 0.0

    avg_forward_alpha_5d: float | None = None
    avg_forward_alpha_10d: float | None = None
    hit_rate_5d: float | None = None
    hit_rate_10d: float | None = None
    avg_max_drawdown_10d: float | None = None

    pass_min_sample: bool = False
    pass_alpha: bool = False
    pass_hit_rate: bool = False
    pass_drawdown: bool = False
    recommended_action: str = "watch_only"
    diagnostic_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThemeCalibrationReport:
    record_count: int = 0
    available_count: int = 0
    phase_summaries: dict[str, CalibrationMetricSummary] = field(default_factory=dict)
    score_bucket_summaries: dict[str, CalibrationMetricSummary] = field(
        default_factory=dict
    )
    theme_score_bucket_summaries: dict[str, CalibrationMetricSummary] = field(
        default_factory=dict
    )
    risk_flag_summaries: dict[str, CalibrationMetricSummary] = field(
        default_factory=dict
    )
    theme_summaries: dict[str, CalibrationMetricSummary] = field(default_factory=dict)
    threshold_diagnostics: list[ThresholdDiagnostic] = field(default_factory=list)
    recommended_thresholds: list[ThresholdDiagnostic] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "available_count": self.available_count,
            "phase_summaries": _summary_dict(self.phase_summaries),
            "score_bucket_summaries": _summary_dict(self.score_bucket_summaries),
            "theme_score_bucket_summaries": _summary_dict(
                self.theme_score_bucket_summaries
            ),
            "risk_flag_summaries": _summary_dict(self.risk_flag_summaries),
            "theme_summaries": _summary_dict(self.theme_summaries),
            "threshold_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.threshold_diagnostics
            ],
            "recommended_thresholds": [
                diagnostic.to_dict() for diagnostic in self.recommended_thresholds
            ],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        rows.extend(_summary_rows("phase", self.phase_summaries))
        rows.extend(_summary_rows("symbol_theme_score_bucket", self.score_bucket_summaries))
        rows.extend(
            _summary_rows("theme_score_bucket", self.theme_score_bucket_summaries)
        )
        rows.extend(_summary_rows("risk_flag", self.risk_flag_summaries))
        rows.extend(_summary_rows("theme", self.theme_summaries))
        return pd.DataFrame(rows)

    def thresholds_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [diagnostic.to_dict() for diagnostic in self.threshold_diagnostics]
        )

    def to_markdown(self, *, max_rows: int = 20) -> str:
        lines = [
            "## Theme Calibration Diagnostics",
            "",
            f"Record count: {self.record_count}",
            f"Available forward data count: {self.available_count}",
            "",
            "This report is offline calibration only and does not change trading behavior.",
            "",
            "### Phase Summaries",
        ]
        lines.extend(_metric_lines(self.phase_summaries, max_rows=max_rows))
        lines.extend(["", "### Symbol Theme Score Buckets"])
        lines.extend(_metric_lines(self.score_bucket_summaries, max_rows=max_rows))
        lines.extend(["", "### Theme Score Buckets"])
        lines.extend(_metric_lines(self.theme_score_bucket_summaries, max_rows=max_rows))
        lines.extend(["", "### Risk Flag Diagnostics"])
        lines.extend(_metric_lines(self.risk_flag_summaries, max_rows=max_rows))
        lines.extend(["", "### Recommended Thresholds"])
        lines.extend(_threshold_lines(self.recommended_thresholds, max_rows=max_rows))
        lines.extend(["", "### Warnings"])
        if self.warnings:
            lines.extend(f"- {warning}" for warning in self.warnings)
        else:
            lines.append("- none")
        return "\n".join(lines)


def safe_mean(values: Iterable[Any]) -> float | None:
    clean = _clean_numbers(values)
    if not clean:
        return None
    return sum(clean) / len(clean)


def safe_median(values: Iterable[Any]) -> float | None:
    clean = sorted(_clean_numbers(values))
    if not clean:
        return None
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) / 2.0


def safe_hit_rate(values: Iterable[Any]) -> float | None:
    labels = [value for value in values if isinstance(value, bool)]
    if not labels:
        return None
    return sum(1 for value in labels if value) / len(labels)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def evidence_quality(available_count: int) -> str:
    count = max(int(available_count or 0), 0)
    if count < 10:
        return "insufficient"
    if count < 30:
        return "weak"
    if count < 100:
        return "moderate"
    return "strong"


def summarize_records(
    records: Iterable[ThemeReplayRecord],
    *,
    group_key: str,
    group_name: str = "",
    min_sample: int = 10,
) -> CalibrationMetricSummary:
    record_list = list(records or [])
    available_records = [record for record in record_list if record.data_available]
    summary = CalibrationMetricSummary(
        group_key=str(group_key),
        group_name=str(group_name or ""),
        record_count=len(record_list),
        available_count=len(available_records),
        avg_symbol_theme_score=safe_mean(
            record.symbol_theme_score for record in record_list
        ),
        avg_theme_score=safe_mean(record.theme_score for record in record_list),
        avg_forward_alpha_5d=safe_mean(
            record.forward_alpha_5d for record in available_records
        ),
        avg_forward_alpha_10d=safe_mean(
            record.forward_alpha_10d for record in available_records
        ),
        avg_forward_alpha_20d=safe_mean(
            record.forward_alpha_20d for record in available_records
        ),
        median_forward_alpha_5d=safe_median(
            record.forward_alpha_5d for record in available_records
        ),
        median_forward_alpha_10d=safe_median(
            record.forward_alpha_10d for record in available_records
        ),
        hit_rate_5d=safe_hit_rate(record.hit_5d for record in available_records),
        hit_rate_10d=safe_hit_rate(record.hit_10d for record in available_records),
        avg_max_drawdown_10d=safe_mean(
            record.max_drawdown_10d for record in available_records
        ),
        avg_max_runup_10d=safe_mean(
            record.max_runup_10d for record in available_records
        ),
        evidence_quality=evidence_quality(len(available_records)),
        metadata={**_BASE_METADATA, "min_sample": int(min_sample)},
    )
    flags: list[str] = []
    if summary.available_count < int(min_sample):
        flags.append("low_sample")
    if (
        summary.avg_forward_alpha_5d is not None
        and summary.avg_forward_alpha_5d < 0.0
    ):
        flags.append("negative_alpha_5d")
    if summary.hit_rate_5d is not None and summary.hit_rate_5d < 0.50:
        flags.append("weak_hit_rate_5d")
    if (
        summary.avg_max_drawdown_10d is not None
        and summary.avg_max_drawdown_10d < -0.08
    ):
        flags.append("drawdown_high")
    summary.diagnostic_flags = flags
    return summary


def score_bucket(value: float | None) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "missing"
    if numeric < 0.40:
        return "0.00-0.40"
    if numeric < 0.55:
        return "0.40-0.55"
    if numeric < 0.70:
        return "0.55-0.70"
    if numeric < 0.85:
        return "0.70-0.85"
    return "0.85-1.00"


def theme_score_bucket(value: float | None) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "missing"
    if numeric < 40.0:
        return "0-40"
    if numeric < 55.0:
        return "40-55"
    if numeric < 70.0:
        return "55-70"
    if numeric < 85.0:
        return "70-85"
    return "85-100"


def evaluate_threshold(
    records: list[ThemeReplayRecord],
    *,
    threshold_name: str,
    threshold_value: float | str,
    predicate: Callable[[ThemeReplayRecord], bool],
    total_record_count: int,
    min_sample: int = 10,
    min_avg_alpha_5d: float = 0.0,
    min_hit_rate_5d: float = 0.52,
    max_avg_drawdown_10d: float = -0.10,
) -> ThresholdDiagnostic:
    selected = [record for record in list(records or []) if _predicate_passes(predicate, record)]
    summary = summarize_records(
        selected,
        group_key=str(threshold_name),
        group_name=str(threshold_value),
        min_sample=min_sample,
    )
    selected_count = len(selected)
    coverage_ratio = selected_count / max(int(total_record_count or 0), 1)
    pass_min_sample = summary.available_count >= int(min_sample)
    pass_alpha = (
        summary.avg_forward_alpha_5d is not None
        and summary.avg_forward_alpha_5d >= float(min_avg_alpha_5d)
    )
    pass_hit_rate = (
        summary.hit_rate_5d is not None
        and summary.hit_rate_5d >= float(min_hit_rate_5d)
    )
    pass_drawdown = (
        summary.avg_max_drawdown_10d is None
        or summary.avg_max_drawdown_10d >= float(max_avg_drawdown_10d)
    )
    flags: list[str] = []
    if selected_count == 0:
        flags.append("empty_selection")
    if not pass_min_sample:
        flags.append("low_sample")
    if (
        summary.avg_forward_alpha_5d is not None
        and summary.avg_forward_alpha_5d < 0.0
    ):
        flags.append("negative_alpha")
    if summary.hit_rate_5d is not None and summary.hit_rate_5d < float(min_hit_rate_5d):
        flags.append("weak_hit_rate")
    if (
        summary.avg_max_drawdown_10d is not None
        and summary.avg_max_drawdown_10d < float(max_avg_drawdown_10d)
    ):
        flags.append("drawdown_high")

    recommended_action = "watch_only"
    if selected_count == 0:
        recommended_action = "ignore"
    elif pass_min_sample and pass_alpha and pass_hit_rate and pass_drawdown:
        recommended_action = "candidate_boost"
    elif (
        (
            summary.avg_forward_alpha_5d is not None
            and summary.avg_forward_alpha_5d < 0.0
        )
        or (summary.hit_rate_5d is not None and summary.hit_rate_5d < 0.45)
        or not pass_drawdown
    ):
        recommended_action = "risk_reduce"

    return ThresholdDiagnostic(
        threshold_name=str(threshold_name),
        threshold_value=threshold_value,
        selected_count=selected_count,
        available_count=summary.available_count,
        coverage_ratio=coverage_ratio,
        avg_forward_alpha_5d=summary.avg_forward_alpha_5d,
        avg_forward_alpha_10d=summary.avg_forward_alpha_10d,
        hit_rate_5d=summary.hit_rate_5d,
        hit_rate_10d=summary.hit_rate_10d,
        avg_max_drawdown_10d=summary.avg_max_drawdown_10d,
        pass_min_sample=pass_min_sample,
        pass_alpha=pass_alpha,
        pass_hit_rate=pass_hit_rate,
        pass_drawdown=pass_drawdown,
        recommended_action=recommended_action,
        diagnostic_flags=flags,
        metadata={
            **_BASE_METADATA,
            "min_sample": int(min_sample),
            "min_avg_alpha_5d": float(min_avg_alpha_5d),
            "min_hit_rate_5d": float(min_hit_rate_5d),
            "max_avg_drawdown_10d": float(max_avg_drawdown_10d),
        },
    )


def build_threshold_diagnostics(
    dataset: ThemeCalibrationDataset,
    *,
    min_sample: int = 10,
) -> list[ThresholdDiagnostic]:
    records = list(getattr(dataset, "records", []) or [])
    total_record_count = len(records)
    diagnostics: list[ThresholdDiagnostic] = []

    for value in (0.55, 0.70, 0.85):
        diagnostics.append(
            evaluate_threshold(
                records,
                threshold_name=f"symbol_theme_score >= {value:.2f}",
                threshold_value=value,
                predicate=lambda record, threshold=value: (
                    safe_float(record.symbol_theme_score, -math.inf) >= threshold
                ),
                total_record_count=total_record_count,
                min_sample=min_sample,
            )
        )
    for value in (55.0, 70.0, 85.0):
        diagnostics.append(
            evaluate_threshold(
                records,
                threshold_name=f"theme_score >= {int(value)}",
                threshold_value=value,
                predicate=lambda record, threshold=value: (
                    safe_float(record.theme_score, -math.inf) >= threshold
                ),
                total_record_count=total_record_count,
                min_sample=min_sample,
            )
        )
    for phase in _PHASE_THRESHOLDS:
        diagnostics.append(
            evaluate_threshold(
                records,
                threshold_name=f"phase == {phase}",
                threshold_value=phase,
                predicate=lambda record, phase_value=phase: record.phase == phase_value,
                total_record_count=total_record_count,
                min_sample=min_sample,
            )
        )
    diagnostics.append(
        evaluate_threshold(
            records,
            threshold_name="no risk flags",
            threshold_value="no_risk_flags",
            predicate=lambda record: not list(record.risk_flags or []),
            total_record_count=total_record_count,
            min_sample=min_sample,
        )
    )
    for flag in _RISK_FLAG_THRESHOLDS:
        diagnostics.append(
            evaluate_threshold(
                records,
                threshold_name=f"has {flag}",
                threshold_value=flag,
                predicate=lambda record, risk_flag=flag: risk_flag
                in set(record.risk_flags or []),
                total_record_count=total_record_count,
                min_sample=min_sample,
            )
        )
    return diagnostics


def build_theme_calibration_report(
    dataset: ThemeCalibrationDataset,
    *,
    min_sample: int = 10,
    top_themes: int = 20,
) -> ThemeCalibrationReport:
    records = list(getattr(dataset, "records", []) or [])
    record_count = len(records)
    available_count = sum(1 for record in records if record.data_available)

    phase_summaries = _group_summaries(
        records,
        key_func=lambda record: record.phase or "unknown_phase",
        min_sample=min_sample,
    )
    score_bucket_summaries = _group_summaries(
        records,
        key_func=lambda record: score_bucket(record.symbol_theme_score),
        min_sample=min_sample,
    )
    theme_score_bucket_summaries = _group_summaries(
        records,
        key_func=lambda record: theme_score_bucket(record.theme_score),
        min_sample=min_sample,
    )
    risk_flag_summaries = _risk_flag_summaries(records, min_sample=min_sample)
    theme_summaries = _theme_summaries(
        records,
        min_sample=min_sample,
        top_themes=top_themes,
    )
    threshold_diagnostics = build_threshold_diagnostics(dataset, min_sample=min_sample)
    recommended_thresholds = sorted(
        [
            diagnostic
            for diagnostic in threshold_diagnostics
            if diagnostic.recommended_action == "candidate_boost"
        ],
        key=lambda diagnostic: (
            -(
                diagnostic.avg_forward_alpha_5d
                if diagnostic.avg_forward_alpha_5d is not None
                else -math.inf
            ),
            -(diagnostic.hit_rate_5d if diagnostic.hit_rate_5d is not None else -math.inf),
            -diagnostic.available_count,
            diagnostic.threshold_name,
        ),
    )
    warnings: list[str] = []
    if available_count < int(min_sample):
        warnings.append("insufficient_calibration_sample")
    if not recommended_thresholds:
        warnings.append("no_threshold_passed_conservative_filters")
    if record_count > 0 and available_count / max(record_count, 1) < 0.60:
        warnings.append("forward_data_coverage_low")

    return ThemeCalibrationReport(
        record_count=record_count,
        available_count=available_count,
        phase_summaries=phase_summaries,
        score_bucket_summaries=score_bucket_summaries,
        theme_score_bucket_summaries=theme_score_bucket_summaries,
        risk_flag_summaries=risk_flag_summaries,
        theme_summaries=theme_summaries,
        threshold_diagnostics=threshold_diagnostics,
        recommended_thresholds=recommended_thresholds,
        warnings=warnings,
        metadata={
            **_BASE_METADATA,
            "min_sample": int(min_sample),
            "top_themes": int(top_themes),
            "record_count": record_count,
            "available_count": available_count,
        },
    )


def _clean_numbers(values: Iterable[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        numeric = safe_float(value)
        if numeric is not None:
            clean.append(numeric)
    return clean


def _predicate_passes(
    predicate: Callable[[ThemeReplayRecord], bool],
    record: ThemeReplayRecord,
) -> bool:
    try:
        return bool(predicate(record))
    except Exception:
        return False


def _group_summaries(
    records: list[ThemeReplayRecord],
    *,
    key_func: Callable[[ThemeReplayRecord], str],
    min_sample: int,
) -> dict[str, CalibrationMetricSummary]:
    grouped: dict[str, list[ThemeReplayRecord]] = {}
    for record in records:
        key = str(key_func(record) or "unknown")
        grouped.setdefault(key, []).append(record)
    return {
        key: summarize_records(group, group_key=key, min_sample=min_sample)
        for key, group in sorted(grouped.items())
    }


def _risk_flag_summaries(
    records: list[ThemeReplayRecord],
    *,
    min_sample: int,
) -> dict[str, CalibrationMetricSummary]:
    grouped: dict[str, list[ThemeReplayRecord]] = {}
    for record in records:
        for flag in record.risk_flags:
            grouped.setdefault(str(flag), []).append(record)
    return {
        key: summarize_records(group, group_key=key, min_sample=min_sample)
        for key, group in sorted(grouped.items())
    }


def _theme_summaries(
    records: list[ThemeReplayRecord],
    *,
    min_sample: int,
    top_themes: int,
) -> dict[str, CalibrationMetricSummary]:
    grouped: dict[str, list[ThemeReplayRecord]] = {}
    for record in records:
        key = record.primary_theme_id or "unknown_theme"
        grouped.setdefault(key, []).append(record)
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -sum(1 for record in item[1] if record.data_available),
            -len(item[1]),
            item[0],
        ),
    )
    summaries: dict[str, CalibrationMetricSummary] = {}
    for key, group in ordered[: max(int(top_themes), 0)]:
        name = next(
            (record.primary_theme_name for record in group if record.primary_theme_name),
            key,
        )
        summaries[key] = summarize_records(
            group,
            group_key=key,
            group_name=name,
            min_sample=min_sample,
        )
    return summaries


def _summary_dict(
    summaries: dict[str, CalibrationMetricSummary],
) -> dict[str, dict[str, Any]]:
    return {key: summary.to_dict() for key, summary in summaries.items()}


def _summary_rows(
    summary_type: str,
    summaries: dict[str, CalibrationMetricSummary],
) -> list[dict[str, Any]]:
    return [
        {"summary_type": summary_type, **summary.to_dict()}
        for key, summary in sorted(summaries.items())
    ]


def _metric_lines(
    summaries: dict[str, CalibrationMetricSummary],
    *,
    max_rows: int,
) -> list[str]:
    if not summaries:
        return ["- none"]
    ordered = sorted(
        summaries.items(),
        key=lambda item: (
            -item[1].available_count,
            -item[1].record_count,
            item[0],
        ),
    )
    lines: list[str] = []
    for key, summary in ordered[: max(int(max_rows), 0)]:
        label = summary.group_name or key
        lines.append(
            "- "
            f"{label}: records={summary.record_count}, "
            f"available={summary.available_count}, "
            f"quality={summary.evidence_quality}, "
            f"alpha5={_format_optional(summary.avg_forward_alpha_5d)}, "
            f"hit5={_format_optional(summary.hit_rate_5d)}, "
            f"drawdown10={_format_optional(summary.avg_max_drawdown_10d)}"
        )
    return lines or ["- none"]


def _threshold_lines(
    diagnostics: list[ThresholdDiagnostic],
    *,
    max_rows: int,
) -> list[str]:
    if not diagnostics:
        return ["- none"]
    lines: list[str] = []
    for diagnostic in diagnostics[: max(int(max_rows), 0)]:
        lines.append(
            "- "
            f"{diagnostic.threshold_name}: action={diagnostic.recommended_action}, "
            f"available={diagnostic.available_count}, "
            f"alpha5={_format_optional(diagnostic.avg_forward_alpha_5d)}, "
            f"hit5={_format_optional(diagnostic.hit_rate_5d)}"
        )
    return lines or ["- none"]


def _format_optional(value: Any) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.4f}"


__all__ = [
    "CalibrationMetricSummary",
    "ThemeCalibrationReport",
    "ThresholdDiagnostic",
    "build_theme_calibration_report",
    "build_threshold_diagnostics",
    "evaluate_threshold",
    "evidence_quality",
    "safe_float",
    "safe_hit_rate",
    "safe_mean",
    "safe_median",
    "score_bucket",
    "summarize_records",
    "theme_score_bucket",
]
