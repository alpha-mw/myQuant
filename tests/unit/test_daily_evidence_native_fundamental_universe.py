"""Native primary fixture supports the actual pool universe, not a fake ready pointer."""

import json
import pandas as pd
from quant_investor.market.fundamental_generation import load_fundamental_pointer
from test_fundamental_generation_promotion import _publish_verified_primary


def test_native_primary_derives_requested_company_universe(tmp_path, monkeypatch):
    import socket

    def forbidden(*a, **kw):
        raise AssertionError("native fixture attempted live network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    symbols = ["000001.SZ", "000002.SZ"]
    root = tmp_path / "data/parquet/cn"
    _publish_verified_primary(root, symbols_override=symbols)
    pointer = load_fundamental_pointer(root)
    assert pointer["derivation_binding"]["binding_aware_research_ready"] is True
    assert "binding_aware_research_ready" not in pointer["metadata"]
    daily = pd.read_parquet(root / pointer["tables"]["fundamental_daily"])
    assert set(daily["ts_code"]) == set(symbols)
    manifest = json.loads((root / pointer["manifest_path"]).read_bytes())
    assert manifest["metadata"]["provider_manifest"]["derivation"]["derivation_timestamp"]


def test_native_research_fixture_projects_real_primary_generation(tmp_path, monkeypatch):
    from functools import partial
    from test_daily_evidence_research_sources import context
    from _native_daily_research_inputs import augment
    from quant_investor.cli import unified
    from quant_investor.operations.research_projection import project_research_source

    workspace = tmp_path / "factor-workspace"
    workspace.mkdir()
    ctx, _ = context(workspace)
    day = ctx.request["as_of"][:10].replace("-", "")
    (tmp_path / f"research-inputs-{day}.json").write_text(
        json.dumps({"request_ref": ctx.request_ref, "companies": ctx.companies})
    )
    bound = augment(tmp_path, day, native_fundamental=True)
    request = json.loads((workspace / bound["request_ref"]["path"]).read_bytes())
    source = request["company_evidence"]["fundamental_source"]
    pointer = json.loads((workspace / source["pointer"]["path"]).read_bytes())
    assert pointer["schema_version"] == "cn-fundamental-pointer.v1"
    assert "binding_aware_research_ready" not in pointer["metadata"]
    projected = project_research_source(
        node="fundamental",
        recipe={"as_of": request["as_of"], "fundamental_source": source},
        companies=ctx.companies,
        source_document=partial(unified._daily_source_document, str(workspace)),
        source_file=partial(unified._daily_source_file, workspace.resolve()),
        workspace=workspace,
    )
    assert projected.state.value == "SUCCEEDED"
    assert bound["fundamental_fixture_mode"] == "NATIVE_SYNTHETIC_PRIMARY"


def test_successor_primary_keeps_old_source_evidence_replayable(tmp_path):
    import hashlib
    from quant_investor.market.fundamental_generation import (
        inspect_fundamental_pointer_bytes,
        FUNDAMENTAL_POINTER_FILENAME,
    )

    root = tmp_path / "data/parquet/cn"
    _publish_verified_primary(
        root, run_id="first", symbols_override=["000001.SZ"], evidence_namespace="first"
    )
    old = (root / FUNDAMENTAL_POINTER_FILENAME).read_bytes()
    sha = hashlib.sha256(old).hexdigest()
    _publish_verified_primary(
        root,
        run_id="second",
        symbols_override=["000001.SZ", "000002.SZ"],
        evidence_namespace="second",
        expected_pointer_sha256=sha,
    )
    prior = inspect_fundamental_pointer_bytes(root, pointer_bytes=old, expected_pointer_sha256=sha)[
        "pointer"
    ]
    assert prior["derivation_binding"]["binding_aware_research_ready"] is True
    assert set(pd.read_parquet(root / prior["tables"]["fundamental_daily"])["ts_code"]) == {
        "000001.SZ"
    }
    current = load_fundamental_pointer(root)
    assert set(pd.read_parquet(root / current["tables"]["fundamental_daily"])["ts_code"]) == {
        "000001.SZ",
        "000002.SZ",
    }
