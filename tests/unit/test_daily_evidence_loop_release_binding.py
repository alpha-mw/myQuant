"""Native release-artifact validation at the existing Factor-to-pool seam."""

from types import SimpleNamespace
import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.system.store import object_ref_for_artifact
from quant_investor.market.daily_factor_loop import DailyFactorLoop
from test_unified_factor_manifest import _release


def context():
    release = _release()
    raw = canonical_json_bytes(release)
    expected = object_ref_for_artifact(release)
    loop = object.__new__(DailyFactorLoop)
    loop.installation = {"state": "PASS", "release_ref": expected}
    loop.context = {"release_install_input_ref": {"path": "release-input.json", "sha256": "d" * 64}}
    calls = []

    def read(path):
        calls.append(path)
        return SimpleNamespace(data=raw, byte_sha256=hashlib.sha256(raw).hexdigest())

    loop.store = SimpleNamespace(read=read)
    return loop, expected, calls


def test_core_uses_native_release_artifact_instead_of_installation_envelope():
    loop, expected, calls = context()
    result = loop._core_release_ref()
    assert result == {
        "path": f"results/factors/objects/system.release/{expected['byte_sha256']}.json",
        "sha256": expected["byte_sha256"],
    }
    assert result != loop.context["release_install_input_ref"]
    assert calls == [result["path"]]


@pytest.mark.parametrize("fault", ["missing", "sha", "object"])
def test_missing_or_wrong_release_cannot_fall_back_to_envelope(fault):
    loop, expected, calls = context()
    if fault == "missing":
        loop.store.read = lambda path: None
    elif fault == "sha":
        loop.store.read = lambda path: SimpleNamespace(data=b"{}", byte_sha256="0" * 64)
    else:
        loop.installation["release_ref"] = {**expected, "artifact_id": "different-release"}
    with pytest.raises(ValueError, match="DAILY_CORE_RELEASE"):
        loop._core_release_ref()
