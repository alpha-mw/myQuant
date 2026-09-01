#!/usr/bin/env python3
"""Read back and verify a generated CN aggressive Dashboard bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cn_dashboard_common import validate_bundle_shape, verify_source_refs
from cn_dashboard_v2 import validate_v2_shape, verify_v2_source_refs
from cn_dashboard_v2_selector import read_selector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "portfolio_dashboard"
    / "private"
    / "generated"
    / "cn_aggressive_dashboard.v1.json"
)
DEFAULT_JS_BUNDLE = DEFAULT_BUNDLE.with_suffix(".js")
DEFAULT_V2_BUNDLE = DEFAULT_BUNDLE.with_name("cn_aggressive_dashboard.v2.json")
DEFAULT_V2_JS_BUNDLE = DEFAULT_V2_BUNDLE.with_suffix(".js")
DEFAULT_SELECTOR = DEFAULT_BUNDLE.with_name("cn_aggressive_dashboard_selector.v2.json")
DEFAULT_SELECTOR_JS = DEFAULT_SELECTOR.with_suffix(".js")
OFFICIAL_VALUATION_PUBLICATION_REQUIRED = "OFFICIAL_VALUATION_PUBLICATION_REQUIRED"
OFFICIAL_CLOSE_NOT_CAUGHT_UP = "OFFICIAL_CLOSE_NOT_CAUGHT_UP"


def _read_js_wrapper(path: Path, variable: str) -> dict:
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise ValueError(f"{variable}_js_unstable_double_read")
    prefix = f"window.{variable} = ".encode("utf-8")
    suffix = b";\n"
    if not first.startswith(prefix) or not first.endswith(suffix):
        raise ValueError(f"{variable}_js_wrapper_invalid")
    value = json.loads(first[len(prefix) : -len(suffix)].decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{variable}_js_payload_not_object")
    return value


def official_valuation_publication_requirement(bundle: dict, v2_bundle: dict | None) -> str | None:
    """Return the typed upstream-writer requirement without granting write authority."""

    if not isinstance(v2_bundle, dict):
        return None
    performance_end = bundle.get("portfolio", {}).get("performance_end_date")
    mark_as_of = v2_bundle.get("freshness", {}).get("mark_as_of")
    continuity = v2_bundle.get("continuity_authority", {})
    current = bundle.get("current_evidence", {})
    if (
        isinstance(performance_end, str)
        and isinstance(mark_as_of, str)
        and mark_as_of > performance_end
        and v2_bundle.get("freshness", {}).get("status") == "UPDATED"
        and continuity.get("status") == "NO_ACTION_BOUND"
        and current.get("official_valuation") is not True
    ):
        return OFFICIAL_VALUATION_PUBLICATION_REQUIRED
    return None


def latest_required_close_date(project_root: Path) -> str | None:
    """Return the Market/benchmark required close, independent of event readiness."""

    try:
        market = json.loads(
            (project_root / "data/parquet/cn/_latest.json").read_text(encoding="utf-8")
        )
        from quant_investor.market.cn_benchmark_store import (
            CNBenchmarkStoreError,
            load_generation,
        )

        benchmark = load_generation(project_root / "data/parquet/cn/benchmarks")
    except (OSError, ValueError, json.JSONDecodeError, CNBenchmarkStoreError):
        return None
    market_date = str(market.get("latest_complete_trade_date") or "")
    if len(market_date) == 8 and market_date.isdigit():
        market_date = f"{market_date[:4]}-{market_date[4:6]}-{market_date[6:]}"
    benchmark_date = str(benchmark.get("pointer", {}).get("end_date") or "")
    if not market_date or not benchmark_date:
        return None
    return min(market_date, benchmark_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--js-bundle", type=Path)
    parser.add_argument("--v2-bundle", type=Path)
    parser.add_argument("--v2-js-bundle", type=Path)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--selector-js", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        raw = args.bundle.read_bytes()
        if raw != args.bundle.read_bytes():
            raise ValueError("bundle_unstable_double_read")
        bundle = json.loads(raw.decode("utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, ensure_ascii=False))
        return 2
    errors = validate_bundle_shape(bundle) + verify_source_refs(bundle, args.project_root.resolve())
    v2_bundle = None
    selector = None
    use_default_bundle_set = (
        args.bundle.resolve() == DEFAULT_BUNDLE.resolve()
        and args.js_bundle is None
        and args.v2_bundle is None
        and args.v2_js_bundle is None
        and args.selector is None
        and args.selector_js is None
    )
    custom_bundle_set = not use_default_bundle_set
    if custom_bundle_set and any(
        value is None
        for value in (
            args.js_bundle,
            args.v2_bundle,
            args.v2_js_bundle,
            args.selector,
            args.selector_js,
        )
    ):
        errors.append("custom_bundle_requires_complete_v1_v2_selector_set")
    js_path = (
        DEFAULT_JS_BUNDLE.resolve()
        if use_default_bundle_set
        else args.js_bundle.resolve() if args.js_bundle is not None else None
    )
    if js_path is not None:
        try:
            if _read_js_wrapper(js_path, "MyQuantCNAggressiveDashboard") != bundle:
                errors.append("v1_json_js_payload_mismatch")
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            errors.append(str(exc))
    v2_path = (
        DEFAULT_V2_BUNDLE.resolve()
        if use_default_bundle_set
        else args.v2_bundle.resolve() if args.v2_bundle is not None else None
    )
    selector_path = (
        DEFAULT_SELECTOR.resolve()
        if use_default_bundle_set
        else args.selector.resolve() if args.selector is not None else None
    )
    v2_js_path = (
        DEFAULT_V2_JS_BUNDLE.resolve()
        if use_default_bundle_set
        else args.v2_js_bundle.resolve() if args.v2_js_bundle is not None else None
    )
    selector_js_path = (
        DEFAULT_SELECTOR_JS.resolve()
        if use_default_bundle_set
        else args.selector_js.resolve() if args.selector_js is not None else None
    )
    if (v2_path is None) != (selector_path is None):
        errors.append("v2_bundle_and_selector_must_be_paired")
    if v2_path is not None and (v2_js_path is None or selector_js_path is None):
        errors.append("v2_json_js_and_selector_js_must_be_paired")
    if v2_path is not None and selector_path is not None:
        try:
            v2_raw = v2_path.read_bytes()
            if v2_raw != v2_path.read_bytes():
                raise ValueError("v2_bundle_unstable_double_read")
            v2_bundle = json.loads(v2_raw.decode("utf-8"))
            selector = read_selector(selector_path)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            errors.append(str(exc))
        if isinstance(v2_bundle, dict) and isinstance(selector, dict):
            if v2_js_path is not None and selector_js_path is not None:
                try:
                    if _read_js_wrapper(v2_js_path, "MyQuantCNAggressiveDashboardV2") != v2_bundle:
                        errors.append("v2_json_js_payload_mismatch")
                    if (
                        _read_js_wrapper(
                            selector_js_path,
                            "MyQuantCNAggressiveDashboardSelectorV2",
                        )
                        != selector
                    ):
                        errors.append("selector_json_js_payload_mismatch")
                except (
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    errors.append(str(exc))
            errors.extend(validate_v2_shape(v2_bundle))
            errors.extend(verify_v2_source_refs(v2_bundle, args.project_root.resolve()))
            if v2_bundle.get("canonical_v1") != bundle:
                errors.append("v2_canonical_v1_mismatch")
            if selector.get("status") != "UPDATED":
                errors.append("v2_selector_not_updated")
            if selector.get("attempt_id") != v2_bundle.get("publication_attempt_id"):
                errors.append("v2_selector_attempt_mismatch")
            if selector.get("v2_content_sha256") != v2_bundle.get("content_sha256"):
                errors.append("v2_selector_content_mismatch")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False))
        return 2
    required_close = latest_required_close_date(args.project_root.resolve())
    publication_requirement = official_valuation_publication_requirement(bundle, v2_bundle)
    performance_end = bundle.get("portfolio", {}).get("performance_end_date")
    if (
        isinstance(required_close, str)
        and isinstance(performance_end, str)
        and performance_end < required_close
    ):
        publication_requirement = OFFICIAL_CLOSE_NOT_CAUGHT_UP
    print(
        json.dumps(
            {
                "ok": True,
                "status": bundle["status"],
                "latest_valid_record": bundle["latest_valid_record"],
                "content_sha256": bundle["content_sha256"],
                "v2_content_sha256": (
                    v2_bundle.get("content_sha256") if isinstance(v2_bundle, dict) else None
                ),
                "freshness_status": (
                    v2_bundle.get("freshness", {}).get("status")
                    if isinstance(v2_bundle, dict)
                    else None
                ),
                "mark_as_of": (
                    v2_bundle.get("freshness", {}).get("mark_as_of")
                    if isinstance(v2_bundle, dict)
                    else None
                ),
                "latest_required_close_date": required_close,
                "publication_requirement": publication_requirement,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
