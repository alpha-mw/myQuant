"""Source-time extraction verifies native file refs, without inventing publication."""

import hashlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.prospective_sources import read_research_source_times
from quant_investor.operations.daily_contract import ContractError
from quant_investor.cli.unified import CommandError


def fixture(root):
    def put(name, value):
        raw = canonical_json_bytes(value)
        p = root / name
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    source = put("source.json", {"created_at": "2020-01-01T00:00:00Z"})
    pointer = put("pointer.json", {"fixture": "pointer"})
    req = {
        "industry_source": None,
        "theme_source": None,
        "company_evidence": {
            "exposure_rows": [
                {"source": source, "available_at": t}
                for t in ["2026-09-08T06:00:00Z", "2026-09-08T07:00:00Z"]
            ],
            "fundamental_source": {
                "pointer": pointer,
                "daily_parquet": source,
                "available_at": "2026-09-08T06:30:00Z",
            },
            "macro_risk": {"classification": "fixture", "source": source},
        },
    }
    return put("request.json", req), source


def test_declared_times_and_duplicate_source_conservative_bound(tmp_path):
    request, source = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.iterdir()}
    result = read_research_source_times(workspace=str(tmp_path), request_ref=request)
    assert result["exposure"] == [
        {"source_ref": source, "declared_available_at": "2026-09-08T07:00:00Z"}
    ]
    assert result["macro"] == [{"source_ref": source, "declared_available_at": None}]
    assert result["industry"] == result["theme"] == []
    assert result["fundamental"][0]["declared_available_at"] is None
    assert before == {str(p): p.read_bytes() for p in tmp_path.iterdir()}


def test_native_fractional_source_time_is_retained_exactly(tmp_path):
    request, _ = fixture(tmp_path)
    path = tmp_path / request["path"]
    value = json.loads(path.read_bytes())
    stamp = "2026-09-08T06:30:00.000001Z"
    value["company_evidence"]["fundamental_source"]["available_at"] = stamp
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    request["sha256"] = hashlib.sha256(raw).hexdigest()
    result = read_research_source_times(workspace=str(tmp_path), request_ref=request)
    assert result["fundamental"][1]["declared_available_at"] == stamp


@pytest.mark.parametrize("target", ["request.json", "source.json"])
def test_changed_ref_rejected(tmp_path, target):
    request, _ = fixture(tmp_path)
    (tmp_path / target).write_bytes(b"changed")
    with pytest.raises((ContractError, CommandError)):
        read_research_source_times(workspace=str(tmp_path), request_ref=request)


def test_both_theme_scopes_are_retained_without_invented_availability(tmp_path):
    request_ref, source = fixture(tmp_path)
    value = json.loads((tmp_path / request_ref["path"]).read_bytes())
    focus_raw = canonical_json_bytes({"fixture": "independent focus source"})
    path = tmp_path / "focus.json"
    path.write_bytes(focus_raw)
    path.chmod(0o600)
    focus = {"path": path.name, "sha256": hashlib.sha256(focus_raw).hexdigest()}

    def descriptor(ref):
        return {
            "dc_plan": ref,
            "dc_capture": ref,
            "dc_partitions": [ref],
            "tdx_plan": None,
            "tdx_capture": None,
            "tdx_partitions": [],
        }

    value["theme_source"] = {
        "schema_version": "cn-daily-theme-evidence-source.v2",
        "pool": descriptor(source),
        "pcb_ai_hardware": descriptor(focus),
    }
    raw = canonical_json_bytes(value)
    (tmp_path / request_ref["path"]).write_bytes(raw)
    request_ref["sha256"] = hashlib.sha256(raw).hexdigest()
    result = read_research_source_times(workspace=str(tmp_path), request_ref=request_ref)
    assert result["theme"] == [
        {"source_ref": ref, "declared_available_at": None} for ref in (focus, source)
    ]
    path.write_bytes(b"changed")
    with pytest.raises(CommandError):
        read_research_source_times(workspace=str(tmp_path), request_ref=request_ref)
