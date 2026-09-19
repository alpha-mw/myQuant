"""Native Parquet permissions survive readback; byte changes still fail closed."""

import hashlib
import pytest
import pandas as pd
from quant_investor.operations.research_file_readback import ResearchFileReadback
from quant_investor.cli.output import CommandError


def test_native_0644_parquet_uses_producer_reader_and_detects_changes(tmp_path):
    path = tmp_path / "daily.parquet"
    pd.DataFrame({"value": [1.0, 2.0]}).to_parquet(path, index=False)
    path.chmod(0o644)
    raw = path.read_bytes()
    ref = {"path": "daily.parquet", "sha256": hashlib.sha256(raw).hexdigest()}
    reader = ResearchFileReadback(str(tmp_path))
    resolved, observed, checked = reader.source_file(ref, code="TEST_SOURCE_INVALID")
    assert resolved == path and observed == raw and checked == ref
    reader.recheck()
    assert path.stat().st_mode & 0o777 == 0o644
    path.write_bytes(b"changed")
    with pytest.raises(CommandError):
        reader.recheck()


def test_symlink_escape_is_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    (workspace / "source").symlink_to(outside)
    ref = {"path": "source", "sha256": hashlib.sha256(b"private").hexdigest()}
    with pytest.raises(CommandError):
        ResearchFileReadback(str(workspace)).source_file(ref, code="TEST_ESCAPE")
