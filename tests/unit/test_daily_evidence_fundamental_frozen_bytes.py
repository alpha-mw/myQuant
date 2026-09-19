"""Native Fundamental parsing must consume the bytes whose SHA was verified."""

from io import BytesIO
import hashlib
from pathlib import Path
import pandas as pd
from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.unified import _daily_fundamental_source


def test_fundamental_parser_does_not_reopen_verified_path(tmp_path):
    frame = pd.DataFrame([{"symbol": "000001.SZ", "value": 7}])
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    source = buffer.getvalue()
    pointer = canonical_json_bytes(
        {
            "status": "OK",
            "generation_id": "generation-a",
            "metadata": {"binding_aware_research_ready": True, "gate2_passed": True},
        }
    )
    refs = {
        "pointer": {"path": "pointer.json", "sha256": hashlib.sha256(pointer).hexdigest()},
        "daily_parquet": {
            "path": "generation-a/data.parquet",
            "sha256": hashlib.sha256(source).hexdigest(),
        },
    }

    def reader(ref, *, code):
        # The path deliberately has no file. Only returned verified bytes are usable.
        return tmp_path / ref["path"], pointer if ref == refs["pointer"] else source, ref

    parsed, metadata = _daily_fundamental_source(
        {"fundamental_source": {**refs, "available_at": "2026-09-04T07:00:00Z"}}, reader
    )
    pd.testing.assert_frame_equal(parsed, frame)
    assert metadata["sha256"] == hashlib.sha256(source).hexdigest()
    assert not list(tmp_path.iterdir())
