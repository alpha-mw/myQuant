"""Actual native primary generation and frozen-pointer research readback."""

import hashlib
from pathlib import Path
import pytest
from quant_investor.market import fundamental_generation as generation
from quant_investor.cli.unified import _daily_fundamental_source
from quant_investor.cli.output import CommandError
from test_fundamental_generation_promotion import _publish_verified_primary


def test_native_binding_replays_after_current_pointer_removed(tmp_path, monkeypatch):
    root = tmp_path / "data/parquet/cn"
    _publish_verified_primary(root)
    pointer_path = root / generation.FUNDAMENTAL_POINTER_FILENAME
    pointer_raw = pointer_path.read_bytes()
    pointer_sha = hashlib.sha256(pointer_raw).hexdigest()
    loaded = generation.load_fundamental_pointer(root)
    assert loaded["derivation_binding"]["binding_aware_research_ready"] is True
    assert "binding_aware_research_ready" not in loaded["metadata"]
    daily_path = root / loaded["tables"]["fundamental_daily"]
    saved = tmp_path / "retained-pointer.json"
    saved.write_bytes(pointer_raw)
    saved.chmod(0o600)
    pointer_path.unlink()
    generation._validate_fundamental_pointer_cached.cache_clear()
    original = generation._stable_file_bytes

    def no_current(path):
        if Path(path).name == generation.FUNDAMENTAL_POINTER_FILENAME:
            pytest.fail("read current Fundamental pointer")
        return original(path)

    monkeypatch.setattr(generation, "_stable_file_bytes", no_current)
    frozen = generation.inspect_fundamental_pointer_bytes(
        root, pointer_bytes=pointer_raw, expected_pointer_sha256=pointer_sha
    )
    assert frozen["current_pointer_replayed"] is False
    assert frozen["pointer"]["derivation_binding"] == loaded["derivation_binding"]
    assert not pointer_path.exists()

    def ref(path):
        return {
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    evidence = {
        "fundamental_source": {
            "available_at": loaded["manifest"]["metadata"]["provider_manifest"]["derivation"][
                "derivation_timestamp"
            ],
            "pointer": ref(saved),
            "daily_parquet": ref(daily_path),
        }
    }

    def source_file(reference, *, code):
        path = tmp_path / reference["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
        return path, raw, reference

    from copy import deepcopy

    backdated = deepcopy(evidence)
    backdated["fundamental_source"]["available_at"] = "2024-05-10T08:00:00Z"
    with pytest.raises(CommandError) as rejected:
        _daily_fundamental_source(backdated, source_file, workspace=tmp_path)
    assert "NATIVE_AVAILABILITY_BACKDATED" in str(rejected.value.__cause__)
    frame, source = _daily_fundamental_source(evidence, source_file, workspace=tmp_path)
    assert not frame.empty
    assert set(frame["ts_code"]) == {"000002.SZ"}
    assert source["sha256"] == evidence["fundamental_source"]["daily_parquet"]["sha256"]
    assert not pointer_path.exists()
    with pytest.raises(CommandError):
        _daily_fundamental_source(evidence, source_file)
    # A copied/supplied table cannot replace the generation's exact native table.
    extra = tmp_path / "different.parquet"
    extra.write_bytes(daily_path.read_bytes())
    evidence["fundamental_source"]["daily_parquet"] = ref(extra)
    with pytest.raises(CommandError):
        _daily_fundamental_source(evidence, source_file, workspace=tmp_path)
    daily_path.write_bytes(b"changed")
    with pytest.raises(generation.FundamentalGenerationError):
        generation.inspect_fundamental_pointer_bytes(
            root, pointer_bytes=pointer_raw, expected_pointer_sha256=pointer_sha
        )


def test_bad_frozen_pointer_sha_fails_before_root_access(tmp_path):
    root = tmp_path / "does-not-exist"
    with pytest.raises(generation.FundamentalGenerationError, match="SHA differs"):
        generation.inspect_fundamental_pointer_bytes(
            root, pointer_bytes=b"{}", expected_pointer_sha256="a" * 64
        )
    assert not root.exists()
