"""Native retained catalog replay does not select a historical pointer."""

import hashlib
import json
import pytest

from quant_investor.strategy_records.store import load_catalog_snapshot, publish_catalog
from test_strategy_record_store import _bootstrap
from quant_investor.strategy_records import store


def test_retained_pointer_reads_original_catalog_after_current_advances(tmp_path):
    first = _bootstrap(tmp_path)
    current = tmp_path / "_record_store/current.v1.json"
    raw = current.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    retained = tmp_path / "_record_store/retained/committed-pointer.v1.json"
    retained.parent.mkdir(mode=0o700)
    retained.write_bytes(raw)
    retained.chmod(0o600)
    publish_catalog(
        tmp_path,
        expected_pointer_sha256=sha,
        records=first["catalog"]["records"],
        active_record_id="r2",
        previous_record_id="r1",
        generation_id="g2",
        published_at="2026-08-11T00:00:00Z",
    )
    after = current.read_bytes()
    pointer, catalog = load_catalog_snapshot(
        tmp_path,
        pointer_relative_path=str(retained.relative_to(tmp_path)),
        expected_pointer_sha256=sha,
    )
    assert pointer["generation_id"] == catalog["generation_id"] == "g1"
    assert current.read_bytes() == after and after != raw


def test_retained_pointer_sha_and_parent_symlink_fail_closed(tmp_path):
    _bootstrap(tmp_path)
    raw = (tmp_path / "_record_store/current.v1.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    retained = tmp_path / "_record_store/retained/pointer.json"
    retained.parent.mkdir(mode=0o700)
    retained.write_bytes(raw)
    retained.chmod(0o600)
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        load_catalog_snapshot(
            tmp_path,
            pointer_relative_path="_record_store/retained/pointer.json",
            expected_pointer_sha256="0" * 64,
        )
    alias = tmp_path / "_record_store/alias"
    alias.symlink_to(retained.parent, target_is_directory=True)
    with pytest.raises(Exception, match="symlink|non-directory"):
        load_catalog_snapshot(
            tmp_path,
            pointer_relative_path="_record_store/alias/pointer.json",
            expected_pointer_sha256=sha,
        )


@pytest.mark.parametrize("current", ["absent", "corrupt"])
def test_catalog_bytes_reader_uses_only_exact_retained_source(tmp_path, monkeypatch, current):
    first = _bootstrap(tmp_path)
    path = tmp_path / "_record_store/current.v1.json"
    raw = path.read_bytes()
    if current == "absent":
        path.unlink()
    else:
        path.write_bytes(b"unrelated current state")
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("retained bytes reader selected current or wrote state")

    monkeypatch.setattr(store, "load_registered_catalog", forbidden)
    monkeypatch.setattr(store, "publish_catalog", forbidden)
    pointer, catalog = store.load_catalog_snapshot_bytes(
        tmp_path,
        pointer_bytes=raw,
        expected_pointer_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert (pointer, catalog) == (first["pointer"], first["catalog"])
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_id", "unknown"),
        ("active_record_id", "r1"),
        ("broker_order_trade_authority", True),
        ("catalog_sha256", "0" * 64),
    ],
)
def test_catalog_bytes_cannot_waive_native_pointer_closure(tmp_path, field, value):
    _bootstrap(tmp_path)
    pointer = json.loads((tmp_path / "_record_store/current.v1.json").read_bytes())
    pointer[field] = value
    pointer["content_sha256"] = store.content_sha256(pointer)
    raw = store.canonical_json_bytes(pointer)
    with pytest.raises(store.StrategyRecordStoreError):
        store.load_catalog_snapshot_bytes(
            tmp_path,
            pointer_bytes=raw,
            expected_pointer_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_catalog_bytes_bad_digest_rejects_before_root_access(tmp_path):
    with pytest.raises(store.StrategyRecordStoreError, match="SHA-256 mismatch"):
        store.load_catalog_snapshot_bytes(
            tmp_path / "absent",
            pointer_bytes=b"{}",
            expected_pointer_sha256="0" * 64,
        )


def test_catalog_bytes_rejects_catalog_drift(tmp_path, monkeypatch):
    _bootstrap(tmp_path)
    raw = (tmp_path / "_record_store/current.v1.json").read_bytes()
    original = store._validate_external_catalog_bindings

    def mutate(root, catalog):
        original(root, catalog)
        pointer = json.loads(raw)
        path = root / pointer["catalog_path"]
        path.write_bytes(path.read_bytes() + b"changed")

    monkeypatch.setattr(store, "_validate_external_catalog_bindings", mutate)
    with pytest.raises(store.StrategyRecordStoreError, match="catalog was unstable"):
        store.load_catalog_snapshot_bytes(
            tmp_path,
            pointer_bytes=raw,
            expected_pointer_sha256=hashlib.sha256(raw).hexdigest(),
        )
