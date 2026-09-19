"""Actual registered observation validation at the core source boundary."""

from copy import deepcopy
import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_observation import build_factor_production_observation
from quant_investor.operations.core_pool import CoreContext
from quant_investor.operations.daily_contract import ContractError
from test_unified_factor_production_observation import _inputs


def write_observation(root, inputs, *, alias="LOW"):
    row = next(r for r in inputs["factor_rows"] if r["factor_alias"] == alias)
    artifact = build_factor_production_observation(
        inputs=inputs, factor_row=row, registered_at="2026-08-20T13:00:00Z"
    )
    path = root / "results/factors/observations/2026/08/20/LOW.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(artifact))
    path.chmod(0o600)
    return artifact


def test_registered_child_replays_exact_factor_and_sources(tmp_path):
    inputs = _inputs()
    artifact = write_observation(tmp_path, inputs)
    context = CoreContext(
        str(tmp_path),
        "20260820",
        inputs["factor_pointer_sha256"],
        {"path": "release.json", "sha256": "e" * 64},
    )
    result, ref = context.observation("LOW", inputs)
    assert result == artifact
    assert ref["path"].endswith("/LOW.json")


@pytest.mark.parametrize(
    "field",
    [
        "factor_pointer_sha256",
        "market_pointer_sha256",
        "pit_pointer_sha256",
        "pit_membership_sha256",
    ],
)
def test_resealed_observation_cannot_switch_source_binding(tmp_path, field):
    original = _inputs()
    mutated = deepcopy(original)
    mutated[field] = "f" * 64
    write_observation(tmp_path, mutated)
    context = CoreContext(
        str(tmp_path),
        "20260820",
        original["factor_pointer_sha256"],
        {"path": "release.json", "sha256": "e" * 64},
    )
    with pytest.raises(ContractError, match="BINDING_MISMATCH"):
        context.observation("LOW", original)


def test_w80_cannot_be_used_as_low_child(tmp_path):
    inputs = _inputs()
    write_observation(tmp_path, inputs, alias="W80")
    context = CoreContext(
        str(tmp_path),
        "20260820",
        inputs["factor_pointer_sha256"],
        {"path": "release.json", "sha256": "e" * 64},
    )
    with pytest.raises(ContractError, match="SIGNAL_MISMATCH"):
        context.observation("LOW", inputs)


def test_native_recovery_consumes_exact_children_without_factor_or_maintenance_writer(
    tmp_path, monkeypatch
):
    import hashlib
    from quant_investor.market import daily_factor_loop as module
    from quant_investor.operations import core_pool

    inputs = _inputs()
    refs = {}
    for alias in ("LOW", "W80"):
        row = next(r for r in inputs["factor_rows"] if r["factor_alias"] == alias)
        value = build_factor_production_observation(
            inputs=inputs, factor_row=row, registered_at="2026-08-20T13:00:00Z"
        )
        path = tmp_path / f"results/factors/observations/2026/08/20/{alias}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json_bytes(value)
        path.write_bytes(raw)
        path.chmod(0o600)
        refs[alias] = {
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    for path in tmp_path.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
    loop = object.__new__(module.DailyFactorLoop)
    loop.workspace = tmp_path
    loop.context = {"release_install_input_ref": {"path": "release.json", "sha256": "f" * 64}}
    release_ref = {"path": "release-artifact.json", "sha256": "e" * 64}
    # This fixture bypasses construction to isolate exact-child recovery. Native
    # installation/artifact binding is exercised by the release-binding tests.
    loop._core_release_ref = lambda: release_ref
    called = []
    monkeypatch.setattr(
        core_pool,
        "publish_core_pool",
        lambda **kwargs: called.append(kwargs) or {"status": "SUCCEEDED"},
    )
    result = loop._recover_core({"core_observation_refs": refs})
    assert result["status"] == "SUCCEEDED"
    assert called[0]["trade_date"] == "20260820"
    assert called[0]["factor_pointer_sha256"] == inputs["factor_pointer_sha256"]
    assert called[0]["release_ref"] == release_ref
    refs["W80"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA_MISMATCH"):
        loop._recover_core({"core_observation_refs": refs})
    assert len(called) == 1
