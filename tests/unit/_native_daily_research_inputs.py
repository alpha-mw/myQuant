"""Synthetic research facts for the exact native Top100; no rank substitution."""

import hashlib
import json
from pathlib import Path
import pandas as pd
from quant_investor.market.tushare import (
    build_industry_membership_partition_capture,
    build_industry_membership_capture,
)


def augment(root: Path, day: str = "20260827", *, native_fundamental: bool = False):
    from test_tushare_industry_capture_stable import _membership_plan, _member_row
    from test_daily_evidence_research_sources import put

    w = root / "factor-workspace"
    bound = json.loads((root / ("research-inputs-" + day + ".json")).read_text())
    request = json.loads((w / bound["request_ref"]["path"]).read_text())
    companies = bound["companies"]
    prefix = "synthetic-research/" + day + "/"

    def emit(name, value):
        return put(w, prefix + name, value)

    taxonomy, tax_capture, plan = _membership_plan()
    parts = []
    keys = plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    current_flag = keys[0].rsplit("=", 1)[1]
    populated_keys = [key for key in keys if key.rsplit("=", 1)[1] == current_flag]
    if len(companies) > len(populated_keys) * 1999:
        raise ValueError("synthetic industry universe exceeds native partition capacity")
    chunks = {}
    for index, key in enumerate(populated_keys):
        start = index * 1999
        stop = start + 1999
        chunks[key] = companies[start:stop]
    for index, key in enumerate(keys):
        code = key.split("|", 1)[0].split("=", 1)[1]
        flag = key.rsplit("=", 1)[1]
        rows = [{**_member_row(l3_code=code, flag=flag), "ts_code": c} for c in chunks.get(key, [])]
        parts.append(
            build_industry_membership_partition_capture(
                membership_plan=plan,
                taxonomy_plan=taxonomy,
                taxonomy_capture=tax_capture,
                partition_key=key,
                partition_ordinal=index,
                provider_request_id="synthetic-industry-" + str(index),
                reported_count=len(rows),
                rows=rows,
                captured_at="2026-08-11T07:31:00Z",
            )
        )
    capture = build_industry_membership_capture(
        membership_plan=plan,
        taxonomy_plan=taxonomy,
        taxonomy_capture=tax_capture,
        partition_documents=parts,
        completed_at="2026-08-11T07:32:00Z",
    )
    request["industry_source"] = {
        "taxonomy_plan": emit("industry/tax-plan.json", taxonomy),
        "taxonomy_capture": emit("industry/tax-capture.json", tax_capture),
        "membership_plan": emit("industry/plan.json", plan),
        "membership_capture": emit("industry/capture.json", capture),
        "membership_partitions": [emit(f"industry/part-{i}.json", p) for i, p in enumerate(parts)],
    }
    theme = request["policy"]["payload"]["technology_theme_ids"][0]
    exposures = [
        {
            "company_code": c,
            "available_at": "2026-08-20T08:00:00Z",
            "primary_theme_id": theme,
            "source": emit("companies/" + c + ".json", {"synthetic": True, "ratio": "0.5"}),
            "source_page": 1,
            "source_type": "ANNUAL_REPORT",
            "theme_revenue_share": "0.5",
        }
        for c in companies
    ]
    if native_fundamental:
        from test_fundamental_generation_promotion import _publish_verified_primary
        from quant_investor.market.fundamental_generation import (
            load_fundamental_pointer,
            FUNDAMENTAL_POINTER_FILENAME,
        )

        canonical = w / "data/parquet/cn"
        prior_path = canonical / FUNDAMENTAL_POINTER_FILENAME
        prior_sha = (
            hashlib.sha256(prior_path.read_bytes()).hexdigest() if prior_path.exists() else None
        )
        _publish_verified_primary(
            canonical,
            run_id="native-dag-" + day,
            symbols_override=companies,
            evidence_namespace="native-dag-" + day,
            expected_pointer_sha256=prior_sha,
        )
        pointer = load_fundamental_pointer(canonical)
        if (
            not pointer
            or pointer.get("derivation_binding", {}).get("binding_aware_research_ready") is not True
        ):
            raise ValueError("native synthetic Fundamental binding is not ready")
        daily = canonical / pointer["tables"]["fundamental_daily"]
        pointer_path = canonical / FUNDAMENTAL_POINTER_FILENAME

        def source_ref(path):
            return {
                "path": str(path.relative_to(w)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        # Provider time is simulated; actual DAG custody uses real terminal clocks.
        manifest = json.loads((canonical / pointer["manifest_path"]).read_bytes())
        available = manifest["metadata"]["provider_manifest"]["derivation"]["derivation_timestamp"]
        source = {
            "available_at": available,
            "daily_parquet": source_ref(daily),
            "pointer": source_ref(pointer_path),
        }
        bound["fundamental_fixture_mode"] = "NATIVE_SYNTHETIC_PRIMARY"
        bound["fundamental_provider_time_simulated"] = True
    else:
        columns = [
            "fin_roe",
            "fin_roa",
            "fin_debt_to_assets",
            "fin_net_profit_yoy",
            "fin_ocf_to_profit",
            "fin_fcf_to_profit",
            "fcf_to_price",
            "forecast_revision",
        ]
        path = w / prefix / "fundamental/fund-g1/daily.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {"ts_code": c, "trade_date": "20260814", **{k: 0.1 + i / 1000 for k in columns}}
                for i, c in enumerate(companies)
            ]
        ).to_parquet(path, index=False)
        path.chmod(0o600)
        source = {
            "available_at": "2026-08-14T08:00:00Z",
            "daily_parquet": {
                "path": str(path.relative_to(w)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "pointer": emit(
                "fundamental/pointer.json",
                {
                    "generation_id": "fund-g1",
                    "status": "OK",
                    "metadata": {"binding_aware_research_ready": True, "gate2_passed": True},
                },
            ),
        }
    request["company_evidence"] = {
        "exposure_rows": exposures,
        "fundamental_source": source,
        "macro_risk": None,
    }
    bound["request_ref"] = emit("request-with-sources.json", request)
    (root / ("research-inputs-expanded-" + day + ".json")).write_text(
        json.dumps(bound, indent=2) + "\n"
    )
    return bound
