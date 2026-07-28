from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v2_contract.canonical import canonical_resource_bytes
from quant_investor.v17_v2_contract.resources import (
    expected_ledger_contract_bindings,
    expected_ledger_implementation_bindings,
)
from quant_investor.v17_v2_contract.validators import (
    SHADOW_LEDGER_VERSION,
    SOURCE_LOCATOR_VERSION,
    seal_semantic,
)
from quant_investor.v17_v2_runtime import CASMismatchError, LedgerStore, SecureStore


def build_ledger_chain(
    run_id: str = "run-1",
    *,
    terminal: bool = False,
) -> list[bytes]:
    states = ["PREPARED", "DETERMINISTIC_COMPLETE"]
    if terminal:
        states.extend(
            [
                "DEEP_REQUEST_READY",
                "DEEP_RESPONSE_RECEIVED",
                "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            ]
        )
    actions = [
        "SHADOW_PREPARE",
        "SHADOW_PREPARE",
        "SHADOW_PREPARE",
        "SHADOW_RECEIVE",
        "SHADOW_FINALIZE",
    ][: len(states)]
    times = [f"2026-07-22T00:0{index + 1}:00Z" for index in range(len(states))]
    source_sha = "1" * 64
    locator_ref = {
        "artifact_id": "locator-1",
        "artifact_version": SOURCE_LOCATOR_VERSION,
        "relative_path": "data/private/v17_sources/protocol-v2/locators/locator-1.json",
        "byte_sha256": source_sha,
        "semantic_sha256": "2" * 64,
    }
    input_bindings = [{"role": "market_bars_dataset", "artifact_ref": locator_ref}]
    history: list[dict[str, Any]] = []
    result: list[bytes] = []
    for sequence, (state, action, at) in enumerate(zip(states, actions, times, strict=True)):
        previous = "EMPTY" if sequence == 0 else hashlib.sha256(result[-1]).hexdigest()
        history.append(
            {
                "sequence": sequence,
                "attempt_id": f"attempt-{sequence}",
                "action": action,
                "acceptance_checkpoint": "INITIALIZED",
                "from_state": None if sequence == 0 else states[sequence - 1],
                "to_state": state,
                "at": at,
                "expected_ledger_sha256": previous,
                "input_binding_sha256s": [source_sha],
                "artifact_roles": [],
            }
        )
        ledger = seal_semantic(
            {
                "protocol_version": "myquant.v17.v2",
                "version": SHADOW_LEDGER_VERSION,
                "run_id": run_id,
                "strategy_id": "cn-shadow",
                "market": "CN",
                "cutoff": "2026-07-22T00:00:00Z",
                "state": state,
                "sequence": sequence,
                "action": action,
                "checkpoint": "INITIALIZED",
                "created_at": times[0],
                "updated_at": at,
                "previous_ledger_sha256": previous,
                "locator_binding": {
                    "locator_id": "locator-1",
                    "locator_ref": locator_ref,
                },
                "contract_bindings": expected_ledger_contract_bindings(),
                "implementation_bindings": expected_ledger_implementation_bindings(),
                "input_bindings": input_bindings,
                "artifacts": [],
                "history": copy.deepcopy(history),
                "authority": False,
            }
        )
        result.append(canonical_resource_bytes(ledger))
    return result


def test_ledger_store_retains_sequence_zero_and_appends_by_cas(tmp_path: Path) -> None:
    chain = build_ledger_chain()
    store = SecureStore(tmp_path)
    store.initialize()
    ledgers = LedgerStore(store)
    ledgers.initialize("run-1", chain[0])
    first_sha = hashlib.sha256(chain[0]).hexdigest()
    ledgers.append("run-1", first_sha, chain[1])
    assert ledgers.read_chain("run-1") == tuple(chain)
    assert store.read(
        "results/v17_shadow/protocol-v2/runs/run-1/events/ledger-000000.json"
    ) == chain[0]


def test_ledger_cas_mismatch_has_no_payload_write(tmp_path: Path) -> None:
    chain = build_ledger_chain()
    store = SecureStore(tmp_path)
    store.initialize()
    ledgers = LedgerStore(store)
    ledgers.initialize("run-1", chain[0])
    before = sorted(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    with pytest.raises(CASMismatchError):
        ledgers.append("run-1", "0" * 64, chain[1])
    after = sorted(
        (path.relative_to(tmp_path).as_posix(), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert after == before
