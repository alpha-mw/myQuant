"""Read-only and explicit-fill data governance entrypoint for CN market data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.market.branch_readiness import (
    DEFAULT_READINESS_ROOT,
    DEFAULT_FUNDAMENTAL_ROOT,
    DEFAULT_MACRO_ROOT,
    assess_branch_data_readiness,
    make_run_id,
    write_branch_readiness_report,
)
from quant_investor.market.market_data_reader import MarketDataReader

_FULL_A_KEYS = {"full_a", "full_market", "all_a", "all", "full"}


def _compact_date(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _normalized_symbols(values: Any, *, field: str, blockers: list[str]) -> list[str]:
    if not isinstance(values, list):
        blockers.append(f"{field}_invalid")
        return []
    symbols = [str(value or "").strip().upper() for value in values]
    if any(not symbol for symbol in symbols) or len(symbols) != len(set(symbols)):
        blockers.append(f"{field}_contains_duplicates_or_empty")
    return [symbol for symbol in symbols if symbol]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bound_path(reader: MarketDataReader, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, reader.data_root / path]
    if reader.data_root.name == "data":
        candidates.append(reader.data_root.parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _resolve_current_full_a_scope(
    *,
    reader: MarketDataReader,
    snapshot: Mapping[str, Any],
    effective_as_of: str,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve the current full-A readiness scope from the bound coverage contract."""

    blockers: list[str] = []
    if snapshot.get("healthy") is not True:
        blockers.append("snapshot_unhealthy")
        blockers.extend(str(item) for item in list(snapshot.get("blockers", []) or []))
    provenance_blockers = [
        str(item)
        for item in list(snapshot.get("coverage_provenance_blockers", []) or [])
        if str(item).strip()
    ]
    if provenance_blockers:
        blockers.append("coverage_provenance_invalid")
        blockers.extend(provenance_blockers)

    coverage = snapshot.get("coverage")
    if not isinstance(coverage, Mapping):
        coverage = {}
        blockers.append("coverage_missing_or_invalid")
    if coverage.get("complete") is not True:
        blockers.append("coverage_not_complete")
    coverage_schema_version = str(coverage.get("coverage_schema_version") or "")
    if coverage_schema_version not in {
        "cn-full-a-coverage.v2",
        "cn-full-a-coverage.v3",
    }:
        blockers.append(
            f"coverage_schema_version_unsupported:{coverage_schema_version or 'missing'}"
        )

    coverage_date = _compact_date(coverage.get("coverage_trade_date"))
    if not coverage_date:
        blockers.append("coverage_trade_date_missing")
    elif coverage_date != _compact_date(effective_as_of):
        blockers.append(
            f"coverage_trade_date_mismatch:{coverage_date}!={_compact_date(effective_as_of)}"
        )

    categories = coverage.get("categories_checked")
    if not isinstance(categories, list):
        blockers.append("coverage_categories_checked_invalid")
        normalized_categories: set[str] = set()
    else:
        normalized_categories = {
            str(category or "").strip().lower()
            for category in categories
            if str(category or "").strip()
        }
    if "full_a" not in normalized_categories:
        blockers.append("coverage_full_a_category_missing")

    components_path = reader.data_root / "cn_universe" / "cn_index_components.json"
    components: Mapping[str, Any] = {}
    if not components_path.exists():
        blockers.append(f"full_a_components_missing:{components_path}")
    else:
        try:
            payload = json.loads(components_path.read_text(encoding="utf-8"))
        except Exception as exc:
            blockers.append(f"full_a_components_unreadable:{type(exc).__name__}")
        else:
            if isinstance(payload, Mapping):
                components = payload
            else:
                blockers.append("full_a_components_invalid")

    expected_symbols = _normalized_symbols(
        components.get("full_a", []),
        field="full_a_components",
        blockers=blockers,
    )
    expected_symbol_set = set(expected_symbols)
    if not expected_symbol_set:
        blockers.append("full_a_components_empty")
    canonical_expected_symbols = sorted(expected_symbol_set)
    expected_scope_sha256 = hashlib.sha256(
        "\n".join(canonical_expected_symbols).encode("utf-8")
    ).hexdigest()
    declared_scope_sha256 = str(coverage.get("expected_scope_sha256") or "").strip().lower()
    if expected_scope_sha256 != declared_scope_sha256:
        blockers.append(
            f"coverage_expected_scope_sha256_mismatch:{expected_scope_sha256}!={declared_scope_sha256}"
        )
    declared_scope_count = coverage.get("expected_scope_count")
    if isinstance(declared_scope_count, bool) or not isinstance(declared_scope_count, int):
        blockers.append("coverage_expected_scope_count_invalid")
    elif declared_scope_count != len(canonical_expected_symbols):
        blockers.append(
            f"coverage_expected_scope_count_mismatch:{declared_scope_count}!={len(canonical_expected_symbols)}"
        )

    classification_keys = (
        "suspended_symbols",
        "inactive_symbols",
        "verified_nontrading_bak_daily_zero_symbols",
        "allowed_stale_symbols",
        "true_missing_symbols",
    )
    classifications = {
        key: set(_normalized_symbols(coverage.get(key, []), field=key, blockers=blockers))
        for key in classification_keys
    }
    if coverage.get("classification_sets_disjoint") is not True:
        blockers.append("coverage_classification_sets_not_declared_disjoint")
    for index, left_key in enumerate(classification_keys):
        for right_key in classification_keys[index + 1 :]:
            if classifications[left_key] & classifications[right_key]:
                blockers.append(f"coverage_classification_overlap:{left_key}:{right_key}")

    non_blocking_symbols = set(
        _normalized_symbols(
            coverage.get("non_blocking_absent_symbols", []),
            field="non_blocking_absent_symbols",
            blockers=blockers,
        )
    )
    declared_non_blocking = (
        classifications["suspended_symbols"]
        | classifications["inactive_symbols"]
        | classifications["verified_nontrading_bak_daily_zero_symbols"]
        | classifications["allowed_stale_symbols"]
    )
    if non_blocking_symbols != declared_non_blocking:
        blockers.append("coverage_non_blocking_absent_union_mismatch")
    if classifications["true_missing_symbols"]:
        blockers.append("coverage_true_missing_symbols_nonempty")
    classified_symbols = non_blocking_symbols | classifications["true_missing_symbols"]
    if not classified_symbols.issubset(expected_symbol_set):
        blockers.append("coverage_classification_outside_expected_scope")

    suspended_evidence = set(
        _normalized_symbols(
            coverage.get("suspended_evidence_symbols", []),
            field="suspended_evidence_symbols",
            blockers=blockers,
        )
    )
    inactive_evidence = set(
        _normalized_symbols(
            coverage.get("inactive_evidence_symbols", []),
            field="inactive_evidence_symbols",
            blockers=blockers,
        )
    )
    if not classifications["suspended_symbols"].issubset(suspended_evidence):
        blockers.append("coverage_suspended_evidence_mismatch")
    if not classifications["inactive_symbols"].issubset(inactive_evidence):
        blockers.append("coverage_inactive_evidence_mismatch")
    if not (suspended_evidence | inactive_evidence).issubset(expected_symbol_set):
        blockers.append("coverage_status_evidence_outside_expected_scope")

    pit_path: Path | None = None
    pit_sha256 = ""
    if non_blocking_symbols:
        pit_path = _resolve_bound_path(reader, coverage.get("pit_membership_path"))
        expected_pit_sha256 = str(coverage.get("pit_membership_sha256") or "").strip().lower()
        if pit_path is None:
            blockers.append("coverage_pit_membership_path_missing")
        elif not pit_path.exists():
            blockers.append(f"coverage_pit_membership_missing:{pit_path}")
        else:
            pit_sha256 = _file_sha256(pit_path)
            if pit_sha256 != expected_pit_sha256:
                blockers.append(
                    "coverage_pit_membership_sha256_mismatch:"
                    f"{pit_sha256}!={expected_pit_sha256}"
                )

    readiness_symbols = sorted(expected_symbol_set - non_blocking_symbols)
    observed_bar_count = coverage.get("observed_bar_count")
    if isinstance(observed_bar_count, bool) or not isinstance(observed_bar_count, int):
        blockers.append("coverage_observed_bar_count_invalid")
    elif observed_bar_count != len(readiness_symbols):
        blockers.append(
            f"coverage_observed_bar_count_mismatch:{observed_bar_count}!={len(readiness_symbols)}"
        )

    serving_root_text = str(snapshot.get("serving_root") or "").strip()
    serving_root = Path(serving_root_text) if serving_root_text else Path(".")
    if not serving_root_text or not serving_root.exists():
        blockers.append(f"serving_root_missing:{serving_root}")
        serving_symbols: set[str] = set()
    else:
        serving_symbols = {
            path.parent.name.split("symbol=", 1)[-1].strip().upper()
            for path in serving_root.glob("symbol=*/bars.parquet")
            if path.parent.name.startswith("symbol=")
        }
    missing_serving_symbols = sorted(set(readiness_symbols) - serving_symbols)
    if missing_serving_symbols:
        blockers.append(f"readiness_symbols_missing_serving:{len(missing_serving_symbols)}")

    unique_blockers = list(dict.fromkeys(blockers))
    metadata = {
        "status": "blocked" if unique_blockers else "passed",
        "blockers": unique_blockers,
        "policy": "snapshot_bound_current_full_a",
        "snapshot_id": str(snapshot.get("snapshot_id") or ""),
        "coverage_schema_version": coverage_schema_version,
        "coverage_trade_date": coverage_date,
        "components_path": str(components_path),
        "expected_scope_count": len(canonical_expected_symbols),
        "expected_scope_sha256": expected_scope_sha256,
        "observed_bar_count": (
            observed_bar_count
            if isinstance(observed_bar_count, int) and not isinstance(observed_bar_count, bool)
            else None
        ),
        "readiness_symbol_count": len(readiness_symbols),
        "non_blocking_absent_count": len(non_blocking_symbols),
        "non_blocking_absent_symbols": sorted(non_blocking_symbols),
        "classification_counts": {key: len(value) for key, value in classifications.items()},
        "suspended_evidence_count": len(suspended_evidence),
        "inactive_evidence_count": len(inactive_evidence),
        "pit_membership_path": str(pit_path or ""),
        "pit_membership_sha256": pit_sha256,
        "serving_inventory_count": len(serving_symbols),
    }
    return ([] if unique_blockers else readiness_symbols), metadata


def _normalize_categories(categories: Sequence[str] | str | None, category: str = "") -> list[str]:
    values: list[str] = []
    if isinstance(categories, str):
        values.extend(item.strip() for item in categories.split(",") if item.strip())
    elif categories:
        values.extend(str(item).strip() for item in categories if str(item).strip())
    if category and not values:
        values.extend(item.strip() for item in str(category).split(",") if item.strip())
    return list(dict.fromkeys(values or ["full_a"]))


def _read_local_frames(
    *,
    market: str,
    category: str,
    as_of: str,
    data_dir: str | Path | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, Any],
    MarketDataReader,
    dict[str, Any],
    str,
]:
    reader = MarketDataReader(market=market, data_root=data_dir or "data", mode_policy="strict")
    try:
        snapshot = reader.snapshot()
    except Exception as exc:
        effective_as_of = _compact_date(as_of)
        return (
            {},
            {},
            reader,
            {
                "status": "blocked",
                "blockers": ["market_snapshot_unavailable:" f"{type(exc).__name__}:{exc}"],
                "policy": "reader_category_scope",
                "snapshot_id": "",
                "coverage_trade_date": effective_as_of,
                "readiness_symbol_count": 0,
            },
            effective_as_of,
        )
    effective_as_of = as_of or str(snapshot.get("latest_complete_trade_date") or "")
    normalized_category = str(category or "full_a").strip().lower()
    if str(market or "").strip().upper() == "CN" and normalized_category in _FULL_A_KEYS:
        symbols, scope_metadata = _resolve_current_full_a_scope(
            reader=reader,
            snapshot=snapshot,
            effective_as_of=effective_as_of,
        )
    else:
        category_blockers: list[str] = []
        try:
            symbols = reader.list_symbols(
                universe_key=category,
                category=None if normalized_category == "full_a" else category,
                as_of=effective_as_of,
            )
        except Exception as exc:
            symbols = []
            category_blockers.append(
                "market_category_scope_unavailable:" f"{type(exc).__name__}:{exc}"
            )
        scope_metadata = {
            "status": "blocked" if category_blockers else "passed",
            "blockers": category_blockers,
            "policy": "reader_category_scope",
            "snapshot_id": str(snapshot.get("snapshot_id") or ""),
            "coverage_trade_date": _compact_date(effective_as_of),
            "readiness_symbol_count": len(symbols),
        }
    frames: dict[str, pd.DataFrame] = {}
    read_results: dict[str, Any] = {}
    for symbol in symbols:
        result = reader.read_symbol_frame(
            symbol,
            universe_key=category,
            category=None if category == "full_a" else category,
            end_date=effective_as_of,
        )
        frames[str(symbol)] = result.frame
        read_results[str(symbol)] = result
    return frames, read_results, reader, scope_metadata, _compact_date(effective_as_of)


def run_data_governance(
    *,
    market: str = "CN",
    category: str = "full_a",
    categories: Sequence[str] | str | None = None,
    as_of: str = "",
    allow_live: bool = False,
    allow_public_fallback: bool = False,
    output_dir: str | Path = DEFAULT_READINESS_ROOT,
    data_dir: str | Path | None = None,
    fundamental_root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
    macro_root: str | Path = DEFAULT_MACRO_ROOT,
) -> dict[str, Any]:
    """Audit branch data readiness and optionally refresh local marts.

    Default behavior is local/read-only. Provider calls happen only when
    ``allow_live`` or ``allow_public_fallback`` is explicitly true.
    """

    if str(market).upper() != "CN":
        raise ValueError("data-governance currently supports CN only")
    selected_categories = _normalize_categories(categories, category)
    run_id = make_run_id(as_of)

    if allow_live or allow_public_fallback:
        from quant_investor.market.fundamental_mart import run_cn_fundamental_maintenance
        from quant_investor.market.macro_mart import run_cn_macro_maintenance

        run_cn_fundamental_maintenance(
            market="CN",
            universes=(
                "full_a" if "full_a" in selected_categories else ",".join(selected_categories)
            ),
            as_of=as_of,
            data_root=fundamental_root,
            allow_live=bool(allow_live),
        )
        run_cn_macro_maintenance(
            as_of=as_of,
            data_root=macro_root,
            allow_live=bool(allow_live),
            allow_public_fallback=bool(allow_public_fallback),
            run_id=run_id,
        )

    reports: list[dict[str, Any]] = []
    artifacts_by_category: dict[str, dict[str, str]] = {}
    for selected_category in selected_categories:
        (
            frames,
            read_results,
            _reader,
            scope_metadata,
            effective_as_of,
        ) = _read_local_frames(
            market="CN",
            category=selected_category,
            as_of=as_of,
            data_dir=data_dir,
        )
        report = assess_branch_data_readiness(
            frames=frames,
            read_results=read_results,
            candidate_symbols=list(frames.keys()),
            market="CN",
            category=selected_category,
            as_of=effective_as_of,
            fundamental_root=fundamental_root,
            macro_root=macro_root,
            run_id=run_id if len(selected_categories) == 1 else f"{run_id}_{selected_category}",
        )
        report.metadata.update(
            {
                "allow_live": bool(allow_live),
                "allow_public_fallback": bool(allow_public_fallback),
                "local_read_only": not bool(allow_live or allow_public_fallback),
                "quant_scope": scope_metadata,
            }
        )
        if scope_metadata.get("status") == "blocked":
            quant_readiness = report.readiness["quant"]
            scope_blocker = (
                "full_a_governance_scope_invalid"
                if scope_metadata.get("policy") == "snapshot_bound_current_full_a"
                else "category_governance_scope_invalid"
            )
            quant_readiness.status = "block"
            quant_readiness.blockers = list(
                dict.fromkeys(
                    [
                        *quant_readiness.blockers,
                        scope_blocker,
                        *[
                            str(blocker)
                            for blocker in scope_metadata.get("blockers", [])
                            if str(blocker).strip()
                        ],
                    ]
                )
            )
            quant_readiness.metadata["scope"] = scope_metadata
        artifacts = write_branch_readiness_report(report, output_dir=output_dir)
        artifacts_by_category[selected_category] = artifacts
        payload = report.to_dict(include_branch_data=False)
        payload["artifacts"] = artifacts
        reports.append(payload)

    overall_status = (
        "blocked"
        if any(
            readiness.get("status") == "block"
            for report in reports
            for readiness in dict(report.get("readiness", {})).values()
            if isinstance(readiness, dict)
        )
        else "passed"
    )
    return {
        "run_id": run_id,
        "status": overall_status,
        "market": "CN",
        "categories": selected_categories,
        "as_of": as_of,
        "allow_live": bool(allow_live),
        "allow_public_fallback": bool(allow_public_fallback),
        "local_read_only": not bool(allow_live or allow_public_fallback),
        "reports": reports,
        "artifacts": artifacts_by_category,
    }


__all__ = ["run_data_governance"]
