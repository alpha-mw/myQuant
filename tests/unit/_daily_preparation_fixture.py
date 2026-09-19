"""Native Calendar/Store/Event/benchmark fixtures, with explicit install/Factor seams."""

from datetime import datetime
import hashlib
from types import SimpleNamespace

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import GRAPH_SHA256
from quant_investor.operations.research_timing import approved_research_timing_policy, CURRENT
from quant_investor.operations.theme_acquisition import POLICY_SCHEMA_V3
from quant_investor.factors.production_pit import FOCUS_COMPANIES
from quant_investor.intelligence.storage import approved_theme_policy_v2
from test_daily_evidence_requested_session import capture

NOW = datetime.fromisoformat("2026-08-24T20:20:01+08:00")


def put(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent
    while parent != root:
        parent.chmod(0o700)
        parent = parent.parent
    raw = value if type(value) is bytes else canonical_json_bytes(value)
    target.write_bytes(raw)
    target.chmod(0o600)
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def calendar(root, now=NOW, **kwargs):
    value = capture(now.isoformat(), **kwargs)
    return (
        put(root, "prepare-fixture/calendar.json", value.receipt),
        put(root, "prepare-fixture/calendar.raw.json", value.raw_response_bytes),
    )


def config(root, *, deadline=None, seed=None):
    other = put(root, "prepare-fixture/static.json", {"synthetic": True})
    value = {
        "schema_version": "cn-daily-preparation-config.v1",
        "market": "CN",
        "strategy_id": "aggressive_tech_manufacturing",
        "graph_sha256": GRAPH_SHA256,
        **{
            field: other
            for field in (
                "release_ref",
                "release_install_ref",
                "factor_loop_context_ref",
                "store_policy_ref",
                "industry_source_ref",
                "risk_free_ref",
            )
        },
        "research_policy_ref": put(
            root, "prepare-fixture/research.json", approved_theme_policy_v2()
        ),
        "timing_policy_ref": put(
            root, "prepare-fixture/timing.json", approved_research_timing_policy(CURRENT)
        ),
        "theme_acquisition_ref": put(
            root,
            "prepare-fixture/theme.json",
            {
                "schema_version": POLICY_SCHEMA_V3,
                "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
                "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
                "maximum_companies": 100,
                "special_company_keyset": list(FOCUS_COMPANIES),
            },
        ),
        "corporate_action_template_ref": put(
            root,
            "prepare-fixture/corporate.json",
            {
                "schema_version": "cn-corporate-action-template.v1",
                "strategy_id": "aggressive_tech_manufacturing",
                "tracking_policy_ref": other,
                "named_event_refs": None,
                "anchor_reviews_ref": None,
            },
        ),
        "exposure_rows_ref": None,
        "seed_completion_ref": seed,
        "prediction_deadline_local_time": deadline,
        "publish_current_dashboard": True,
    }
    return value, put(root, "prepare-fixture/config.json", value)


def native_inputs(root, monkeypatch):
    from _native_daily_store_fixture import NativeStoreFixture
    from quant_investor.operations import daily_preparation as module

    book = NativeStoreFixture(root)
    book.advance("2026-08-24")  # Produce sources only; do not close/advance official Store.
    value, _ = config(root)
    value["store_policy_ref"] = {"path": book.policy_path, "sha256": book.policy_sha}
    value["risk_free_ref"] = put(
        root,
        "prepare-fixture/risk-free.csv",
        (root / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv").read_bytes(),
    )
    reference = put(root, "prepare-fixture/config.json", value)
    monkeypatch.setattr(module, "verify_recipe_static_controls", lambda **kwargs: {})
    parent = put(root, "prepare-fixture/factor-parent.json", {"synthetic": "Factor parent"})
    pointer = SimpleNamespace(byte_sha256=parent["sha256"])
    marker = object()
    monkeypatch.setattr(
        module,
        "FactorProductionStore",
        lambda _: SimpleNamespace(
            read=lambda path: pointer if path == module.FACTOR_ACTIVE_POINTER_PATH else marker,
            verify_active=lambda: {
                "factor_authority": module.FACTOR_AUTHORITY_ACTIVE,
                "as_of": "20260821",
                "factor_pointer_byte_sha256": parent["sha256"],
            },
        ),
    )
    return book, value, reference


def snapshot(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
