"""Native local source fixtures; install/Factor admission remain explicit seams."""

from pathlib import Path
from types import SimpleNamespace

from _daily_preparation_fixture import config, put
from _native_source_producer_fixture import event_arguments, NOW


def build(root, monkeypatch):
    from quant_investor.operations import daily_preparation as prepare

    book, event_args = event_arguments(root)
    value, _ = config(root)
    value["store_policy_ref"] = {
        "path": event_args.policy_path,
        "sha256": event_args.policy_sha256,
    }
    risk = Path("portfolio_dashboard/inputs/cn_govt_bond_yield.csv")
    value["risk_free_ref"] = put(root, "fixtures/source-risk-free.csv", (root / risk).read_bytes())
    cfg = put(root, "fixtures/source-config.json", value)
    parent_ref = put(
        root, "fixtures/source-factor-parent.json", {"synthetic": True, "date": "20260824"}
    )
    pointer = SimpleNamespace(byte_sha256=parent_ref["sha256"])
    marker = object()
    monkeypatch.setattr(prepare, "verify_recipe_static_controls", lambda **kwargs: {})
    monkeypatch.setattr(
        prepare,
        "FactorProductionStore",
        lambda _: SimpleNamespace(
            read=lambda path: pointer if path == prepare.FACTOR_ACTIVE_POINTER_PATH else marker,
            verify_active=lambda: {
                "factor_authority": prepare.FACTOR_AUTHORITY_ACTIVE,
                "as_of": "20260824",
                "factor_pointer_byte_sha256": parent_ref["sha256"],
            },
        ),
    )
    return {
        "book": book,
        "config": value,
        "config_ref": cfg,
        "now": NOW,
        "event_args": event_args,
        "release_install_ref": value["release_install_ref"],
    }
