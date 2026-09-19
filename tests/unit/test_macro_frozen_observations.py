"""Frozen observation replay uses only retained bytes and native generation checks."""

import hashlib
import json

import pytest

from quant_investor.macro import store
from test_macro_observation_evidence_store import _publish_with_evidence, _row


def load(root, raw, generation="g1"):
    return store.load_frozen_observations(
        root,
        pointer_raw=raw,
        expected_pointer_sha256=hashlib.sha256(raw).hexdigest(),
        expected_generation_id=generation,
    )


@pytest.mark.parametrize("current", ["missing", "corrupt", "newer"])
def test_frozen_reader_never_reads_current_or_writes(tmp_path, monkeypatch, current):
    root = tmp_path / "observations"
    _publish_with_evidence(root, _row(), run_id="g1")
    pointer = root / "_latest.json"
    raw = pointer.read_bytes()
    expected = store.load_observations(root)
    if current == "newer":
        _publish_with_evidence(
            root,
            _row(period="2026-05-31", available="2026-06-01T01:00:00Z"),
            run_id="g2",
        )
    elif current == "corrupt":
        pointer.write_bytes(b"broken-current-pointer")
    else:
        pointer.unlink()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("frozen reader touched current pointer or publication")

    monkeypatch.setattr(store, "_strict_pointer", forbidden)
    monkeypatch.setattr(store, "_optional_pointer_bytes", forbidden)
    monkeypatch.setattr(store, "publish_observations", forbidden)
    assert load(root, raw) == expected
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("generation_id", "g2", "generation_id_mismatch"),
        ("row_count", 5, "row_count_mismatch"),
        ("content_set_hash", "0" * 64, "content_set_hash_mismatch"),
        ("schema_version", "unknown", "shape_invalid"),
        ("production_eligible", True, "observer_flags_invalid"),
    ],
)
def test_frozen_pointer_cannot_waive_native_checks(tmp_path, field, value, error):
    _publish_with_evidence(tmp_path, _row(), run_id="g1")
    pointer = json.loads((tmp_path / "_latest.json").read_bytes())
    pointer[field] = value
    with pytest.raises(store.MacroObservationStoreError, match=error):
        load(tmp_path, json.dumps(pointer).encode())


@pytest.mark.parametrize("leaf", ["table", "manifest", "evidence"])
def test_frozen_generation_tampering_rejects(tmp_path, leaf):
    _publish_with_evidence(tmp_path, _row(), run_id="g1")
    raw = (tmp_path / "_latest.json").read_bytes()
    pointer = json.loads(raw)
    path = tmp_path / pointer["table_path" if leaf == "table" else "manifest_path"]
    if leaf == "evidence":
        manifest = json.loads(path.read_bytes())
        path = path.parent / manifest["evidence_files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(store.MacroObservationStoreError, match="hash_mismatch|size_mismatch"):
        load(tmp_path, raw)


def test_bad_pointer_digest_rejects_before_root_lookup(tmp_path):
    with pytest.raises(store.MacroObservationStoreError, match="pointer_sha_mismatch"):
        store.load_frozen_observations(
            tmp_path / "absent",
            pointer_raw=b"{}",
            expected_pointer_sha256="0" * 64,
            expected_generation_id="g1",
        )
