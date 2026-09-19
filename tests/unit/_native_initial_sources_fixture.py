"""Broad pre-core synthetic sources; never invent a Top100 or observation binding."""

import hashlib
import json
from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence.storage import approved_theme_policy_v2


def prepare_initial_sources(root, companies):
    from _native_daily_research_inputs import augment
    from _native_shared_macro_fixture import build_shared_macro

    workspace = root / "factor-workspace"
    day = "20260827"
    assert not (workspace / "results/operations/daily_production/CN" / day).exists()
    companies = sorted(set(companies))
    assert len(companies) >= 100

    def put(name, value):
        path = workspace / "initial-source-inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        raw = canonical_json_bytes(value)
        path.write_bytes(raw)
        path.chmod(0o600)
        return {"path": str(path.relative_to(workspace)), "sha256": hashlib.sha256(raw).hexdigest()}

    seed = put("source-seed.json", {"policy": approved_theme_policy_v2()})
    # augment consumes only companies and a source template. No pool or Factor refs
    # are asserted here; broad source coverage is deliberately independent of rank.
    (root / f"research-inputs-{day}.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "source_scope": "PRE_CORE_BROAD_UNIVERSE",
                "companies": companies,
                "request_ref": seed,
            }
        )
        + "\n"
    )
    expanded = augment(root, native_fundamental=True)
    sources = json.loads((workspace / expanded["request_ref"]["path"]).read_bytes())
    macro = build_shared_macro(workspace, "2026-08-27")
    result = {
        "industry_source_ref": put("industry.json", sources["industry_source"]),
        "exposure_rows_ref": put("exposure.json", sources["company_evidence"]["exposure_rows"]),
        "fundamental": {
            "mode": "PINNED",
            "source_ref": put(
                "fundamental.json", sources["company_evidence"]["fundamental_source"]
            ),
        },
        "macro": {
            "mode": "PINNED",
            "source_ref": put(
                "macro.json",
                {"classification": "CANONICAL_MACRO_READY", "source": macro["closure_ref"]},
            ),
        },
    }
    assert not (workspace / "results/operations/daily_production/CN" / day).exists()
    (root / "initial-pre-core-sources.json").write_text(
        json.dumps(
            {"synthetic": True, "company_count": len(companies), "sources": result}, indent=2
        )
        + "\n"
    )
    return result
