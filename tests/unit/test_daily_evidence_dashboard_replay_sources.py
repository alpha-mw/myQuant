"""Historical byte resolution is closed-set, isolated, and cannot publish."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from quant_investor.operations.dashboard_replay_sources import (
    RetainedDashboardSources,
    retained_dashboard_sources,
)
from cn_dashboard_common import stable_read
from cn_dashboard_v2 import _stable_artifact
from export_cn_aggressive_dashboard_data import publish_bundle


def test_retained_native_reads_preserve_paths_without_using_current_alias(tmp_path):
    path = tmp_path / "alias.json"
    path.write_bytes(b"current")
    before = path.stat().st_mtime_ns
    sources = {"alias.json": b"retained"}
    frozen = RetainedDashboardSources(tmp_path, sources)
    sources["alias.json"] = b"tampered"
    with retained_dashboard_sources(frozen):
        assert stable_read(path, tmp_path).data == b"retained"
        artifact = _stable_artifact(path, tmp_path)
        assert artifact.raw == b"retained"
        assert artifact.relative_path == "alias.json"
        with pytest.raises(ValueError, match="NOT_RETAINED"):
            stable_read(tmp_path / "missing.json", tmp_path)
        with pytest.raises(ValueError, match="CANNOT_PUBLISH"):
            publish_bundle({}, tmp_path / "out.json", tmp_path / "out.js", tmp_path)
    assert stable_read(path, tmp_path).data == b"current"
    assert path.stat().st_mtime_ns == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("path", [".", "../escape", "/absolute", "a/../b", "a//b", "a\\b"])
def test_retained_source_paths_fail_closed(tmp_path, path):
    with pytest.raises(ValueError):
        RetainedDashboardSources(tmp_path, {path: b"data"})


def test_context_releases_after_exception_and_rejects_nested_context(tmp_path):
    selected = RetainedDashboardSources(tmp_path, {})
    with pytest.raises(RuntimeError):
        with retained_dashboard_sources(selected):
            with pytest.raises(ValueError, match="CONTEXT_INVALID"):
                with retained_dashboard_sources(selected):
                    pass
            raise RuntimeError("stop replay")
    path = tmp_path / "normal"
    path.write_bytes(b"normal")
    assert stable_read(path, tmp_path).data == b"normal"


def test_native_inventory_detects_changed_transitive_price_file(tmp_path):
    import hashlib
    from test_daily_evidence_native_dashboard import fixture
    from quant_investor.operations.dashboard_replay_sources import retained_store_inventory

    f, _, _, _, _ = fixture(tmp_path)
    pointer = f.root / "_record_store/current.v1.json"
    ref = {
        "path": str(pointer.relative_to(tmp_path)),
        "sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
    }
    import pandas as pd

    sources = retained_store_inventory(project_root=tmp_path, record_root=f.root, pointer_ref=ref)
    price_paths = [
        p for p in sources if p.startswith("data/parquet/cn/_snapshots/") and p.endswith(".parquet")
    ]
    assert price_paths
    assert all("/table/bars/" in p and "/serving/" not in p for p in price_paths)
    path = tmp_path / price_paths[-1]
    frame = pd.read_parquet(path)
    frame["close"] = frame["close"] + 1.0
    frame.to_parquet(path, index=False)
    # v2 evidence pins the table partition bytes, so any change fails on SHA.
    with pytest.raises(ValueError, match="CLOSE_SOURCE_SHA_MISMATCH"):
        retained_store_inventory(project_root=tmp_path, record_root=f.root, pointer_ref=ref)


def test_native_inventory_does_not_require_serving_projection(tmp_path):
    import hashlib
    import shutil
    from test_daily_evidence_native_dashboard import fixture
    from quant_investor.operations.dashboard_replay_sources import retained_store_inventory

    f, _, _, _, _ = fixture(tmp_path)
    for serving in (tmp_path / "data/parquet/cn/_snapshots").glob("*/serving"):
        shutil.rmtree(serving)
    pointer = f.root / "_record_store/current.v1.json"
    ref = {
        "path": str(pointer.relative_to(tmp_path)),
        "sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
    }
    sources = retained_store_inventory(project_root=tmp_path, record_root=f.root, pointer_ref=ref)
    assert not any("/serving/" in p for p in sources)
