from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from quant_investor.v17_v2_contract.canonical import (
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v2_contract.validators import (
    SHADOW_LATEST_POINTER_VERSION,
    SHADOW_OUTPUT_VERSION,
    seal_semantic,
)
from quant_investor.v17_v2_runtime import (
    LedgerStore,
    SecureStore,
    TerminalPublisher,
)

from test_v17_v2_ledger_store import build_ledger_chain


def _ref(document: dict[str, Any], path: str) -> dict[str, Any]:
    return {
        "artifact_id": document["run_id"],
        "artifact_version": document["version"],
        "relative_path": path,
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
        "semantic_sha256": document["semantic_sha256"],
    }


def terminal_documents(
    ledger_bytes: bytes,
    *,
    previous_sha: str = "EMPTY",
    mode: str = "NORMAL",
) -> tuple[str, bytes, bytes]:
    ledger = load_canonical_resource(ledger_bytes)
    run_id = ledger["run_id"]
    ledger_path = f"results/v17_shadow/protocol-v2/runs/{run_id}/ledger.json"
    outcome_path = f"results/v17_shadow/protocol-v2/outcomes/{run_id}.json"
    rank_output = seal_semantic(
        {
            "protocol_version": "myquant.v17.v2",
            "version": "myquant.v17.v2.rank-output.v1",
            "output_id": f"{run_id}-rank",
            "run_id": run_id,
            "strategy_id": "cn-shadow",
            "market": "CN",
            "cutoff": "2026-07-22T00:00:00Z",
            "status": "COMPLETE",
            "candidate_ordering": "rank-ascending-then-security_code-ascending",
            "candidates": [],
            "generated_at": "2026-07-22T00:06:00Z",
            "authority": False,
        }
    )
    output = seal_semantic(
        {
            "protocol_version": "myquant.v17.v2",
            "version": SHADOW_OUTPUT_VERSION,
            "run_id": run_id,
            "strategy_id": "cn-shadow",
            "market": "CN",
            "cutoff": "2026-07-22T00:00:00Z",
            "terminal_state": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            "ledger_ref": _ref(ledger, ledger_path),
            "source_locator_ref": ledger["locator_binding"]["locator_ref"],
            "rank_output": rank_output,
            "portfolio_output": None,
            "blockers": [],
            "generated_at": "2026-07-22T00:06:00Z",
            "authority": False,
        }
    )
    latest = seal_semantic(
        {
            "protocol_version": "myquant.v17.v2",
            "version": SHADOW_LATEST_POINTER_VERSION,
            "pointer_path": "results/v17_shadow/protocol-v2/_latest/shadow.json",
            "run_id": run_id,
            "terminal_state": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            "ledger_ref": _ref(ledger, ledger_path),
            "terminal_output_ref": _ref(output, outcome_path),
            "previous_pointer_byte_sha256": previous_sha,
            "publication_mode": mode,
            "published_at": "2026-07-22T00:07:00Z",
            "authority": False,
        }
    )
    return outcome_path, canonical_resource_bytes(output), canonical_resource_bytes(latest)


def test_terminal_publication_is_exact_once(tmp_path: Path) -> None:
    chain = build_ledger_chain(terminal=True)
    store = SecureStore(tmp_path)
    store.initialize()
    ledgers = LedgerStore(store)
    ledgers.initialize("run-1", chain[0])
    for predecessor, successor in zip(chain, chain[1:]):
        ledgers.append("run-1", hashlib.sha256(predecessor).hexdigest(), successor)
    outcome_path, outcome_bytes, latest_bytes = terminal_documents(chain[-1])
    publisher = TerminalPublisher(store)
    result = publisher.publish(
        "run-1",
        hashlib.sha256(chain[-1]).hexdigest(),
        outcome_path,
        outcome_bytes,
        "EMPTY",
        latest_bytes,
    )
    retry = publisher.publish(
        "run-1",
        hashlib.sha256(chain[-1]).hexdigest(),
        outcome_path,
        outcome_bytes,
        "EMPTY",
        latest_bytes,
    )
    assert result.outcome_created and result.latest_replaced
    assert not retry.outcome_created and not retry.latest_replaced
    assert store.read(outcome_path) == outcome_bytes
    assert store.read("results/v17_shadow/protocol-v2/_latest/shadow.json") == latest_bytes

    previous_sha = hashlib.sha256(latest_bytes).hexdigest()
    _, _, repair_bytes = terminal_documents(
        chain[-1],
        previous_sha=previous_sha,
        mode="REPAIR",
    )
    repair = publisher.repair_latest(
        "run-1",
        hashlib.sha256(chain[-1]).hexdigest(),
        outcome_path,
        outcome_bytes,
        previous_sha,
        repair_bytes,
    )
    assert repair.repaired and repair.latest_replaced
    assert not repair.outcome_created
    assert store.read(outcome_path) == outcome_bytes
    assert store.read("results/v17_shadow/protocol-v2/_latest/shadow.json") == repair_bytes
