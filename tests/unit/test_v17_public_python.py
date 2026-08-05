from __future__ import annotations

import inspect

import pytest


def test_public_package_exports_only_v17_surface() -> None:
    import quant_investor

    assert quant_investor.__all__ == [
        "MainlineStore",
        "QuantInvestor",
        "V17MainlineError",
    ]
    assert not hasattr(quant_investor, "QuantInvestorPipelineResult")
    signature = inspect.signature(quant_investor.QuantInvestor)
    assert list(signature.parameters) == ["workspace_root", "strategy_id"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_quant_investor_reads_v17_public_run(monkeypatch, tmp_path) -> None:
    from quant_investor.pipeline import mainline
    from quant_investor import v17_mainline

    expected = {"schema_id": "myquant.v17.v4.mainline-public-run.v1"}
    captured = {}

    def fake_read_public_run(workspace_root, *, strategy_id, **kwargs):
        captured.update(
            workspace_root=workspace_root,
            strategy_id=strategy_id,
            kwargs=kwargs,
        )
        return expected

    monkeypatch.setattr(v17_mainline, "read_public_run", fake_read_public_run)
    investor = mainline.QuantInvestor(
        workspace_root=tmp_path,
        strategy_id="cn-mainline",
    )

    assert investor.run() is expected
    assert captured == {
        "workspace_root": tmp_path,
        "strategy_id": "cn-mainline",
        "kwargs": {},
    }


def test_quant_investor_rejects_legacy_constructor_shape() -> None:
    from quant_investor import QuantInvestor

    with pytest.raises(TypeError):
        QuantInvestor(stock_pool=["000001.SZ"])  # type: ignore[call-arg]


def test_uninitialized_public_read_is_deterministic_and_no_write(tmp_path) -> None:
    from quant_investor import QuantInvestor, V17MainlineError

    assert list(tmp_path.iterdir()) == []
    with pytest.raises(V17MainlineError) as first:
        QuantInvestor(workspace_root=tmp_path, strategy_id="cn-mainline").run()
    with pytest.raises(V17MainlineError) as second:
        QuantInvestor(workspace_root=tmp_path, strategy_id="cn-mainline").run()

    assert first.value.code == "V17_MAINLINE_UNINITIALIZED"
    assert second.value.code == "V17_MAINLINE_UNINITIALIZED"
    assert list(tmp_path.iterdir()) == []


def test_public_facade_reads_core_synthetic_fixture(tmp_path) -> None:
    from quant_investor import QuantInvestor
    from quant_investor.v17_mainline.testing import write_synthetic_fixture_for_tests

    write_synthetic_fixture_for_tests(tmp_path, synthetic_only=True)
    payload = QuantInvestor(
        workspace_root=tmp_path,
        strategy_id="cn-mainline",
    ).run()

    assert payload["schema_id"] == "myquant.v17.v4.mainline-public-run.v1"
    assert payload["protocol"] == "myquant.v17.v4"
    assert payload["market"] == "CN_A_SHARE"
    assert payload["selector_used"] is False
    assert payload["fallback_used"] is False
