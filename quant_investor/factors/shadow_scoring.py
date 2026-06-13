"""Read-only production factor shadow scoring comparison helpers.

This module compares local production factor matrix signals against already
computed official candidate rankings. It does not fetch data, call providers,
alter candidates, or connect factor scores to stock selection, posterior
scoring, RiskGuard, PortfolioConstructor, orders, or execution.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_STATUS_PRODUCTION,
    FactorDefinition,
    FactorLibraryEntry,
    ProductionFactorLibrary,
)
from quant_investor.factors.shadow_scoring_types import (
    DEFAULT_FACTOR_SHADOW_SCORING_DIR,
    DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME,
    DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME,
    DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME,
    DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME,
    DEFAULT_SHADOW_FACTOR_SCORES_FILENAME,
    SHADOW_COMPARISON_STATUS_FAIL,
    SHADOW_COMPARISON_STATUS_OK,
    SHADOW_COMPARISON_STATUS_WARN,
    SHADOW_SCORE_STATUS_AUDIT_BLOCKED,
    SHADOW_SCORE_STATUS_INSUFFICIENT_DATA,
    SHADOW_SCORE_STATUS_LIBRARY_MISSING,
    SHADOW_SCORE_STATUS_MISSING_DATE,
    SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX,
    SHADOW_SCORE_STATUS_MISSING_SYMBOL,
    SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR,
    SHADOW_SCORE_STATUS_OK,
    SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
    ShadowCandidateScore,
    ShadowFactorScore,
    ShadowScoringComparisonReport,
    ShadowScoringConfig,
    _blocked_factor_ids,
    _coerce_json_dict,
    _entry_key,
    _first_present,
    _is_finite_number,
    _json_safe,
    _optional_str,
    _positive_int,
    _short_hash,
    _slug,
    _to_optional_float,
)
from quant_investor.versioning import (
    FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION,
    FACTOR_SHADOW_SCORING_SCHEMA_VERSION,
)


def make_shadow_scoring_config_id(config: ShadowScoringConfig) -> str:
    payload = config.to_dict()
    payload["config_id"] = ""
    parts = [
        payload.get("as_of", ""),
        payload.get("top_n", ""),
        payload.get("min_factor_coverage_ratio", ""),
        payload,
    ]
    return f"factor-shadow-scoring-config-{_slug(config.as_of)}-{_short_hash(parts)}"


def make_shadow_comparison_report_id(
    *,
    as_of: str,
    generated_at: str,
    candidate_symbols: Sequence[str],
) -> str:
    ordered_symbols = sorted({str(symbol).strip() for symbol in candidate_symbols if str(symbol).strip()})
    parts = [str(as_of), str(generated_at), ordered_symbols]
    return f"factor-shadow-comparison-{_slug(as_of)}-{_short_hash(parts)}"


def extract_factor_value_for_symbol(
    matrix: FactorMatrix,
    *,
    symbol: str,
    as_of: str,
) -> tuple[float | None, str]:
    resolved_symbol = str(symbol).strip()
    if resolved_symbol not in matrix.symbols:
        return None, SHADOW_SCORE_STATUS_MISSING_SYMBOL

    as_of_date = str(as_of).strip()[:10]
    eligible_dates = [
        (date_value, index)
        for index, date_value in enumerate(matrix.dates)
        if str(date_value) <= as_of_date
    ]
    if not eligible_dates:
        return None, SHADOW_SCORE_STATUS_MISSING_DATE

    _date_value, date_index = max(eligible_dates, key=lambda item: (item[0], item[1]))
    symbol_index = matrix.symbols.index(resolved_symbol)
    try:
        raw_value = matrix.values[symbol_index][date_index]
    except IndexError:
        return None, SHADOW_SCORE_STATUS_INSUFFICIENT_DATA
    if not _is_finite_number(raw_value):
        return None, SHADOW_SCORE_STATUS_INSUFFICIENT_DATA
    return float(raw_value), SHADOW_SCORE_STATUS_OK


def rank_normalize_factor_values(
    values_by_symbol: Mapping[str, float | None],
    *,
    expected_direction: float = 1.0,
) -> dict[str, tuple[float | None, int | None]]:
    direction = -1.0 if float(expected_direction) < 0 else 1.0
    valid_rows = [
        (str(symbol), float(value) * direction)
        for symbol, value in values_by_symbol.items()
        if _is_finite_number(value)
    ]
    valid_rows = sorted(valid_rows, key=lambda item: (-item[1], item[0]))

    output: dict[str, tuple[float | None, int | None]] = {
        str(symbol): (None, None)
        for symbol in values_by_symbol.keys()
    }
    count = len(valid_rows)
    for index, (symbol, _adjusted_value) in enumerate(valid_rows, start=1):
        normalized = 1.0 if count == 1 else 1.0 - ((index - 1) / (count - 1))
        output[symbol] = (float(normalized), index)
    return output


def resolve_factor_expected_direction(
    *,
    factor_id: str,
    factor_version: str,
    definitions: Sequence[FactorDefinition] | None = None,
    matrix: FactorMatrix | None = None,
) -> float:
    for definition in definitions or []:
        if definition.factor_id == factor_id and definition.version == factor_version:
            return -1.0 if float(definition.expected_direction) < 0 else 1.0

    if matrix is not None:
        value = matrix.metadata.get("expected_direction")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 1.0
        return -1.0 if number < 0 else 1.0
    return 1.0


def build_factor_matrix_lookup(
    matrices: Sequence[FactorMatrix],
) -> dict[tuple[str, str], FactorMatrix]:
    lookup: dict[tuple[str, str], FactorMatrix] = {}
    for matrix in matrices:
        if not matrix.factor_id or not matrix.factor_version:
            continue
        key = (matrix.factor_id, matrix.factor_version)
        max_date = max(matrix.dates) if matrix.dates else ""
        current = lookup.get(key)
        if current is None:
            lookup[key] = matrix
            continue
        current_max_date = max(current.dates) if current.dates else ""
        if (max_date, _slug(current.matrix_id)) > (current_max_date, _slug(current.matrix_id)):
            lookup[key] = matrix
        elif max_date == current_max_date and matrix.matrix_id < current.matrix_id:
            lookup[key] = matrix
    return dict(sorted(lookup.items(), key=lambda item: item[0]))


def select_usable_production_factors(
    *,
    library: ProductionFactorLibrary | None,
    audit_report: Any | None = None,
    include_blocked_factors: bool = False,
) -> list[FactorLibraryEntry]:
    if library is None:
        return []
    blocked_ids = set() if include_blocked_factors else _blocked_factor_ids(audit_report)
    entries = [
        entry
        for entry in library.entries
        if entry.status == FACTOR_STATUS_PRODUCTION and entry.factor_id not in blocked_ids
    ]
    return sorted(entries, key=lambda entry: (entry.factor_id, entry.factor_version))


def _extract_candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        payload = dict(candidate)
        symbol = _optional_str(_first_present(payload, ["symbol", "ts_code", "code"]))
        if symbol is None:
            continue
        name = _optional_str(_first_present(payload, ["name", "company_name", "stock_name"]))
        official_score = _to_optional_float(
            _first_present(
                payload,
                ["official_score", "final_score", "posterior_action_score", "score"],
            )
        )
        official_rank_raw = _first_present(payload, ["official_rank", "rank", "final_rank"])
        official_rank = None
        if official_rank_raw is not None:
            try:
                official_rank = _positive_int(official_rank_raw, "official_rank")
            except (TypeError, ValueError):
                official_rank = None
        rows.append(
            {
                "index": index,
                "symbol": symbol,
                "name": name,
                "official_score": official_score,
                "official_rank": official_rank,
            }
        )

    derived_order = sorted(
        rows,
        key=lambda row: (
            row["official_score"] is None,
            -float(row["official_score"] or 0.0),
            row["symbol"],
        ),
    )
    derived_ranks = {row["symbol"]: rank for rank, row in enumerate(derived_order, start=1)}
    for row in rows:
        if row["official_rank"] is None:
            row["official_rank"] = derived_ranks.get(row["symbol"])
    return rows


def _rank_shadow_scores(scores_by_symbol: Mapping[str, float | None]) -> dict[str, int | None]:
    valid_rows = [
        (str(symbol), float(score))
        for symbol, score in scores_by_symbol.items()
        if _is_finite_number(score)
    ]
    valid_rows = sorted(valid_rows, key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int | None] = {str(symbol): None for symbol in scores_by_symbol.keys()}
    for index, (symbol, _score) in enumerate(valid_rows, start=1):
        ranks[symbol] = index
    return ranks


def build_shadow_candidate_scores(
    *,
    candidates: Sequence[Mapping[str, Any]],
    library: ProductionFactorLibrary | None,
    factor_matrices: Sequence[FactorMatrix],
    definitions: Sequence[FactorDefinition] | None = None,
    audit_report: Any | None = None,
    config: ShadowScoringConfig,
    metadata: Mapping[str, Any] | None = None,
) -> list[ShadowCandidateScore]:
    candidate_rows = _extract_candidate_rows(candidates)
    candidate_symbols = [row["symbol"] for row in candidate_rows]
    matrix_lookup = build_factor_matrix_lookup(factor_matrices)
    usable_factors = select_usable_production_factors(
        library=library,
        audit_report=audit_report,
        include_blocked_factors=config.include_blocked_factors,
    )

    blocked_ids = _blocked_factor_ids(audit_report)
    excluded_blocked_ids = sorted(
        blocked_ids
        - {entry.factor_id for entry in usable_factors}
    )
    warnings_by_symbol: dict[str, set[str]] = {symbol: set() for symbol in candidate_symbols}
    factor_scores_by_symbol: dict[str, list[ShadowFactorScore]] = {
        symbol: []
        for symbol in candidate_symbols
    }
    normalized_by_symbol_factor: dict[tuple[str, str, str], float | None] = {}

    if library is None:
        for symbol in candidate_symbols:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_LIBRARY_MISSING)
    if excluded_blocked_ids and not config.include_blocked_factors:
        for symbol in candidate_symbols:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_AUDIT_BLOCKED)

    non_production_entries = [
        entry.factor_id
        for entry in (library.entries if library is not None else [])
        if entry.status != FACTOR_STATUS_PRODUCTION
    ]
    if non_production_entries:
        for symbol in candidate_symbols:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR)

    for entry in usable_factors:
        key = _entry_key(entry)
        matrix = matrix_lookup.get(key)
        raw_values: dict[str, float | None] = {}
        statuses: dict[str, str] = {}
        if matrix is None:
            for symbol in candidate_symbols:
                raw_values[symbol] = None
                statuses[symbol] = SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX
                warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX)
        else:
            for symbol in candidate_symbols:
                value, status = extract_factor_value_for_symbol(
                    matrix,
                    symbol=symbol,
                    as_of=config.as_of,
                )
                raw_values[symbol] = value
                statuses[symbol] = status
                if status != SHADOW_SCORE_STATUS_OK:
                    warnings_by_symbol[symbol].add(status)

        direction = resolve_factor_expected_direction(
            factor_id=entry.factor_id,
            factor_version=entry.factor_version,
            definitions=definitions,
            matrix=matrix,
        )
        normalized = (
            rank_normalize_factor_values(raw_values, expected_direction=direction)
            if config.normalize_factor_scores
            else {
                symbol: (raw_values[symbol], None)
                for symbol in raw_values
            }
        )
        for symbol in candidate_symbols:
            normalized_score, rank = normalized.get(symbol, (None, None))
            if normalized_score is not None:
                normalized_by_symbol_factor[(symbol, entry.factor_id, entry.factor_version)] = normalized_score
            warning_codes = [] if statuses[symbol] == SHADOW_SCORE_STATUS_OK else [statuses[symbol]]
            factor_scores_by_symbol[symbol].append(
                ShadowFactorScore(
                    factor_id=entry.factor_id,
                    factor_version=entry.factor_version,
                    symbol=symbol,
                    as_of=config.as_of,
                    raw_value=raw_values[symbol],
                    normalized_score=normalized_score,
                    rank=rank,
                    coverage_status=statuses[symbol],
                    warning_codes=warning_codes,
                    metadata={
                        **_coerce_json_dict(metadata, "metadata"),
                        "expected_direction": direction,
                        "matrix_id": matrix.matrix_id if matrix is not None else None,
                    },
                )
            )

    shadow_scores_by_symbol: dict[str, float | None] = {}
    for symbol in candidate_symbols:
        covered_scores = [
            score.normalized_score
            for score in factor_scores_by_symbol[symbol]
            if score.normalized_score is not None
        ]
        shadow_scores_by_symbol[symbol] = (
            float(sum(covered_scores) / len(covered_scores))
            if covered_scores
            else None
        )
        if not covered_scores:
            warnings_by_symbol[symbol].add(SHADOW_SCORE_STATUS_INSUFFICIENT_DATA)

    shadow_ranks = _rank_shadow_scores(shadow_scores_by_symbol)
    official_scores_outside_unit = any(
        row["official_score"] is not None
        and not 0.0 <= float(row["official_score"]) <= 1.0
        for row in candidate_rows
    )

    output: list[ShadowCandidateScore] = []
    for row in candidate_rows:
        symbol = row["symbol"]
        factor_scores = factor_scores_by_symbol[symbol]
        covered_factor_count = sum(
            1
            for score in factor_scores
            if score.normalized_score is not None
        )
        factor_count = len(usable_factors)
        coverage_ratio = covered_factor_count / factor_count if factor_count else 0.0
        shadow_rank = shadow_ranks.get(symbol)
        official_rank = row["official_rank"]
        shadow_score = shadow_scores_by_symbol.get(symbol)
        rank_delta = (
            int(official_rank) - int(shadow_rank)
            if official_rank is not None and shadow_rank is not None
            else None
        )
        score_delta = (
            float(shadow_score) - float(row["official_score"])
            if shadow_score is not None and row["official_score"] is not None
            else None
        )
        candidate_metadata = {
            **_coerce_json_dict(metadata, "metadata"),
            "factor_weight_policy": config.factor_weight_policy,
        }
        if official_scores_outside_unit:
            candidate_metadata["official_score_scale_note"] = (
                "official_score is outside [0, 1]; score_delta is a raw difference"
            )
        output.append(
            ShadowCandidateScore(
                symbol=symbol,
                name=row["name"],
                as_of=config.as_of,
                official_score=row["official_score"],
                official_rank=official_rank,
                shadow_factor_score=shadow_score,
                shadow_factor_rank=shadow_rank,
                rank_delta=rank_delta,
                score_delta=score_delta,
                factor_count=factor_count,
                covered_factor_count=covered_factor_count,
                factor_coverage_ratio=coverage_ratio,
                warning_codes=list(warnings_by_symbol[symbol]),
                factor_scores=factor_scores,
                metadata=candidate_metadata,
            )
        )

    return sorted(
        output,
        key=lambda score: (
            score.official_rank is None,
            score.official_rank if score.official_rank is not None else 10**9,
            score.symbol,
        ),
    )


def _delta_row(candidate: ShadowCandidateScore) -> dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "name": candidate.name,
        "official_rank": candidate.official_rank,
        "shadow_factor_rank": candidate.shadow_factor_rank,
        "rank_delta": candidate.rank_delta,
        "official_score": candidate.official_score,
        "shadow_factor_score": candidate.shadow_factor_score,
        "factor_coverage_ratio": candidate.factor_coverage_ratio,
        "warning_codes": list(candidate.warning_codes),
    }


def build_shadow_scoring_comparison_report(
    *,
    candidates: Sequence[Mapping[str, Any]],
    library: ProductionFactorLibrary | None,
    factor_matrices: Sequence[FactorMatrix],
    definitions: Sequence[FactorDefinition] | None = None,
    audit_report: Any | None = None,
    config: ShadowScoringConfig,
    generated_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> ShadowScoringComparisonReport:
    candidate_scores = build_shadow_candidate_scores(
        candidates=candidates,
        library=library,
        factor_matrices=factor_matrices,
        definitions=definitions,
        audit_report=audit_report,
        config=config,
        metadata=metadata,
    )
    candidate_count = len(candidate_scores)
    selected_factors = select_usable_production_factors(
        library=library,
        audit_report=audit_report,
        include_blocked_factors=config.include_blocked_factors,
    )
    matrix_lookup = build_factor_matrix_lookup(factor_matrices)
    production_factor_count = (
        len([entry for entry in library.entries if entry.status == FACTOR_STATUS_PRODUCTION])
        if library is not None
        else 0
    )
    used_factor_count = sum(1 for entry in selected_factors if _entry_key(entry) in matrix_lookup)
    scored_candidate_count = sum(
        1
        for candidate in candidate_scores
        if candidate.shadow_factor_score is not None
    )
    average_factor_coverage_ratio = (
        sum(candidate.factor_coverage_ratio for candidate in candidate_scores) / candidate_count
        if candidate_count
        else None
    )

    official_sorted = sorted(
        candidate_scores,
        key=lambda score: (
            score.official_rank is None,
            score.official_rank if score.official_rank is not None else 10**9,
            score.symbol,
        ),
    )
    shadow_sorted = sorted(
        [score for score in candidate_scores if score.shadow_factor_rank is not None],
        key=lambda score: (score.shadow_factor_rank or 10**9, score.symbol),
    )
    official_top_symbols = [score.symbol for score in official_sorted[: config.top_n]]
    shadow_top_symbols = [score.symbol for score in shadow_sorted[: config.top_n]]
    shadow_top_set = set(shadow_top_symbols)
    overlap_top_symbols = [
        score.symbol
        for score in official_sorted[: config.top_n]
        if score.symbol in shadow_top_set
    ]
    denominator = min(config.top_n, candidate_count)
    overlap_ratio = (
        len(overlap_top_symbols) / denominator
        if denominator
        else None
    )

    positive_deltas = sorted(
        [
            candidate
            for candidate in candidate_scores
            if candidate.rank_delta is not None and candidate.rank_delta > 0
        ],
        key=lambda score: (-(score.rank_delta or 0), score.symbol),
    )
    negative_deltas = sorted(
        [
            candidate
            for candidate in candidate_scores
            if candidate.rank_delta is not None and candidate.rank_delta < 0
        ],
        key=lambda score: ((score.rank_delta or 0), score.symbol),
    )
    warning_codes = set()
    for candidate in candidate_scores:
        warning_codes.update(candidate.warning_codes)
        if (
            candidate.rank_delta is not None
            and abs(candidate.rank_delta) > config.max_rank_delta_warning
        ):
            warning_codes.add("large_rank_delta")
    if library is None:
        warning_codes.add(SHADOW_SCORE_STATUS_LIBRARY_MISSING)
    if selected_factors and used_factor_count < len(selected_factors):
        warning_codes.add(SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX)
    if production_factor_count == 0 or not selected_factors:
        warning_codes.add("no_usable_production_factors")
    if (
        average_factor_coverage_ratio is not None
        and average_factor_coverage_ratio < config.min_factor_coverage_ratio
    ):
        warning_codes.add("low_factor_coverage")

    status = SHADOW_COMPARISON_STATUS_OK
    if warning_codes:
        status = SHADOW_COMPARISON_STATUS_WARN

    base_metadata = _coerce_json_dict(metadata, "metadata")
    report_metadata = {
        **base_metadata,
        "factor_shadow_scoring_schema_version": FACTOR_SHADOW_SCORING_SCHEMA_VERSION,
        "factor_shadow_comparison_schema_version": FACTOR_SHADOW_COMPARISON_SCHEMA_VERSION,
        "non_runtime_impact": True,
        "non_runtime_impact_note": SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
    }
    report_id = make_shadow_comparison_report_id(
        as_of=config.as_of,
        generated_at=generated_at,
        candidate_symbols=[candidate.symbol for candidate in candidate_scores],
    )
    return ShadowScoringComparisonReport(
        report_id=report_id,
        generated_at=generated_at,
        as_of=config.as_of,
        config=config,
        production_factor_count=production_factor_count,
        used_factor_count=used_factor_count,
        candidate_count=candidate_count,
        scored_candidate_count=scored_candidate_count,
        average_factor_coverage_ratio=average_factor_coverage_ratio,
        official_top_symbols=official_top_symbols,
        shadow_top_symbols=shadow_top_symbols,
        overlap_top_symbols=overlap_top_symbols,
        overlap_ratio=overlap_ratio,
        largest_positive_rank_deltas=[_delta_row(candidate) for candidate in positive_deltas[: config.top_n]],
        largest_negative_rank_deltas=[_delta_row(candidate) for candidate in negative_deltas[: config.top_n]],
        warning_codes=list(warning_codes),
        status=status,
        candidate_scores=candidate_scores,
        metadata=report_metadata,
    )


def _escape_pipe(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_optional_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _render_delta_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- None"]
    output = [
        "| Symbol | Name | Official Rank | Shadow Rank | Rank Delta | Shadow Score | Coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output.append(
            "| "
            f"`{_escape_pipe(row.get('symbol', ''))}` | "
            f"{_escape_pipe(row.get('name') or '')} | "
            f"{row.get('official_rank') or ''} | "
            f"{row.get('shadow_factor_rank') or ''} | "
            f"{row.get('rank_delta') or ''} | "
            f"{_format_optional_float(row.get('shadow_factor_score'))} | "
            f"{_format_optional_float(row.get('factor_coverage_ratio'))} |"
        )
    return output


def render_shadow_scoring_comparison_markdown(
    report: ShadowScoringComparisonReport,
) -> str:
    lines = [
        f"# Factor Shadow Scoring Comparison: {report.report_id}",
        "",
        f"Generated at: `{_escape_pipe(report.generated_at)}`",
        f"As of: `{_escape_pipe(report.as_of)}`",
        "",
        "## Status",
        "",
        f"`{_escape_pipe(report.status)}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Production factor count | {report.production_factor_count} |",
        f"| Used factor count | {report.used_factor_count} |",
        f"| Candidate count | {report.candidate_count} |",
        f"| Scored candidate count | {report.scored_candidate_count} |",
        (
            "| Average factor coverage ratio | "
            f"{_format_optional_float(report.average_factor_coverage_ratio)} |"
        ),
        "",
        "## Top-N Overlap Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Top N | {report.config.top_n} |",
        f"| Overlap ratio | `{_format_optional_float(report.overlap_ratio)}` |",
        f"| Official top symbols | `{_escape_pipe(', '.join(report.official_top_symbols))}` |",
        f"| Shadow top symbols | `{_escape_pipe(', '.join(report.shadow_top_symbols))}` |",
        f"| Overlap symbols | `{_escape_pipe(', '.join(report.overlap_top_symbols))}` |",
        "",
        "## Largest Positive Rank Deltas",
        "",
    ]
    lines.extend(_render_delta_rows(report.largest_positive_rank_deltas))
    lines.extend(["", "## Largest Negative Rank Deltas", ""])
    lines.extend(_render_delta_rows(report.largest_negative_rank_deltas))

    lines.extend(
        [
            "",
            "## Candidate Score Table",
            "",
            "| Official Rank | Symbol | Name | Official Score | Shadow Rank | Shadow Score | Rank Delta | Coverage | Warnings |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if report.candidate_scores:
        for candidate in report.candidate_scores:
            lines.append(
                "| "
                f"{candidate.official_rank or ''} | "
                f"`{_escape_pipe(candidate.symbol)}` | "
                f"{_escape_pipe(candidate.name or '')} | "
                f"{_format_optional_float(candidate.official_score)} | "
                f"{candidate.shadow_factor_rank or ''} | "
                f"{_format_optional_float(candidate.shadow_factor_score)} | "
                f"{candidate.rank_delta if candidate.rank_delta is not None else ''} | "
                f"{_format_optional_float(candidate.factor_coverage_ratio)} | "
                f"`{_escape_pipe(', '.join(candidate.warning_codes))}` |"
            )
    else:
        lines.append("|  |  |  |  |  |  |  |  | No candidates. |")

    lines.extend(["", "## Warnings", ""])
    if report.warning_codes:
        lines.extend([f"- `{_escape_pipe(code)}`" for code in report.warning_codes])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Runtime Impact",
            "",
            SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def build_shadow_scoring_dashboard_payload(
    report: ShadowScoringComparisonReport,
) -> dict[str, Any]:
    payload = {
        "status": report.status,
        "as_of": report.as_of,
        "production_factor_count": report.production_factor_count,
        "used_factor_count": report.used_factor_count,
        "candidate_count": report.candidate_count,
        "scored_candidate_count": report.scored_candidate_count,
        "overlap_ratio": report.overlap_ratio,
        "official_top_symbols": list(report.official_top_symbols),
        "shadow_top_symbols": list(report.shadow_top_symbols),
        "overlap_top_symbols": list(report.overlap_top_symbols),
        "warning_codes": list(report.warning_codes),
        "largest_positive_rank_deltas": _json_safe(report.largest_positive_rank_deltas),
        "largest_negative_rank_deltas": _json_safe(report.largest_negative_rank_deltas),
        "metadata": dict(_json_safe(report.metadata)),
    }
    json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return dict(_json_safe(payload))


__all__ = [
    "SHADOW_SCORE_STATUS_OK",
    "SHADOW_SCORE_STATUS_MISSING_FACTOR_MATRIX",
    "SHADOW_SCORE_STATUS_MISSING_SYMBOL",
    "SHADOW_SCORE_STATUS_MISSING_DATE",
    "SHADOW_SCORE_STATUS_NON_PRODUCTION_FACTOR",
    "SHADOW_SCORE_STATUS_LIBRARY_MISSING",
    "SHADOW_SCORE_STATUS_AUDIT_BLOCKED",
    "SHADOW_SCORE_STATUS_INSUFFICIENT_DATA",
    "SHADOW_COMPARISON_STATUS_OK",
    "SHADOW_COMPARISON_STATUS_WARN",
    "SHADOW_COMPARISON_STATUS_FAIL",
    "DEFAULT_FACTOR_SHADOW_SCORING_DIR",
    "DEFAULT_SHADOW_FACTOR_SCORES_FILENAME",
    "DEFAULT_SHADOW_CANDIDATE_SCORES_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_REPORTS_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_MARKDOWN_FILENAME",
    "DEFAULT_SHADOW_COMPARISON_DASHBOARD_FILENAME",
    "SHADOW_SCORING_NON_RUNTIME_IMPACT_NOTE",
    "ShadowScoringConfig",
    "ShadowFactorScore",
    "ShadowCandidateScore",
    "ShadowScoringComparisonReport",
    "make_shadow_scoring_config_id",
    "make_shadow_comparison_report_id",
    "extract_factor_value_for_symbol",
    "rank_normalize_factor_values",
    "resolve_factor_expected_direction",
    "build_factor_matrix_lookup",
    "select_usable_production_factors",
    "build_shadow_candidate_scores",
    "build_shadow_scoring_comparison_report",
    "render_shadow_scoring_comparison_markdown",
    "build_shadow_scoring_dashboard_payload",
]
