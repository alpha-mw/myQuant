"""EOD control gates only; fixture adapters are never full native DAG proof."""

from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import pytest
from scripts.daily_completion import _replay, seal_native_completion
from scripts.daily_native_registry import NativeDailyRegistry
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from quant_investor.operations.daily_runner import DayRunner, NativeOutcome
from test_daily_evidence_runner import FixtureAdapter
from test_daily_evidence_dag_journal import request


def fixture(root):
    calls = []
    adapters = {node: FixtureAdapter(root, node, calls) for node in EOD_NODE_IDS}
    runner = DayRunner(str(root), "20260904", adapters)
    templates = {node: {**request(), "node_id": node} for node in EOD_NODE_IDS}
    runner.run(templates)
    registry = object.__new__(NativeDailyRegistry)
    registry.workspace = str(root)
    registry.trade_date = "20260904"
    registry.runner = runner
    registry.adapters = adapters
    registry.templates = templates
    registry.inputs = SimpleNamespace(
        release_ref=request()["release_ref"], publish_current_dashboard=False
    )
    return registry


def test_replay_requires_exact_all_node_results(tmp_path, monkeypatch):
    registry = fixture(tmp_path)
    j = registry.runner.journal
    with j.locked():
        refs = _replay(registry)
        assert set(refs) == EOD_NODE_IDS and "morning" not in refs
        node = registry.adapters["dashboard"]
        original = node.probe

        def changed(request):
            result = original(request)
            from quant_investor.operations.daily_runner import Probe

            return Probe(NativeOutcome(result.outcome.state, {}))

        monkeypatch.setattr(node, "probe", changed)
        with pytest.raises(ContractError, match="EOD_NATIVE_REPLAY_FAILED"):
            _replay(registry)
    assert not (tmp_path / j.root / "completion.v1.json").exists()


def test_seal_rejects_non_native_adapters_and_missing_nodes(tmp_path):
    registry = fixture(tmp_path)
    ref = {"path": "input.json", "sha256": "a" * 64}
    with registry.runner.journal.locked():
        with pytest.raises(ContractError, match="EOD_NATIVE_ADAPTER_TYPE_INVALID"):
            seal_native_completion(registry, native_inputs_ref=ref, synthetic=True)
        registry.adapters.pop("macro")
        with pytest.raises(ContractError, match="EOD_NATIVE_NODE_SET_INCOMPLETE"):
            seal_native_completion(registry, native_inputs_ref=ref, synthetic=True)
    assert not (tmp_path / registry.runner.journal.root / "completion.v1.json").exists()


@pytest.mark.parametrize("entry", ["run_and_seal_native_input", "run_materialized_native_input"])
def test_new_unmaterialized_work_cannot_create_v1(tmp_path, entry):
    from scripts import daily_completion as completion
    from test_daily_evidence_native_inputs import context, put

    value, args = context(tmp_path)
    selected = put(tmp_path, value)
    pointer = args["record_root"] / "_record_store/current.v1.json"
    before = pointer.read_bytes()
    with pytest.raises(ContractError, match="EOD_MATERIALIZATION_REQUIRED"):
        getattr(completion, entry)(
            workspace=str(tmp_path), input_ref=selected, resume=True, synthetic=True
        )
    assert pointer.read_bytes() == before
    assert not (
        tmp_path / "results/operations/daily_production/CN/20260824/completion.v1.json"
    ).exists()


@pytest.mark.parametrize("entry", ["_seal_loaded_completion", "run_and_seal_loaded_native_input"])
def test_legacy_loaded_entries_refuse_new_writes(tmp_path, monkeypatch, entry):
    from scripts import daily_completion as completion
    from scripts import daily_native_inputs as inputs

    registry = fixture(tmp_path)

    def forbidden(*a, **kw):
        pytest.fail("legacy entry invoked writer")

    monkeypatch.setattr(inputs, "run_loaded_native_input", forbidden)
    monkeypatch.setattr(registry.runner.journal.storage, "write", forbidden)
    kwargs = {"synthetic": True}
    ref = {"path": "input.json", "sha256": "a" * 64}
    if entry == "_seal_loaded_completion":
        kwargs["native_inputs_ref"] = ref
    else:
        kwargs.update(input_ref=ref, resume=True)
    with (
        registry.runner.journal.locked(),
        pytest.raises(ContractError, match="EOD_MATERIALIZATION_REQUIRED"),
    ):
        getattr(completion, entry)(registry, **kwargs)
