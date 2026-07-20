from __future__ import annotations

import copy
import hashlib

import pytest

from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    GENESIS_SHA256,
    HOLDOUT_READY,
    HOLDOUT_UNSEALED_FINALIZING,
    PRECOMMITTED,
    PROTOCOL_VERSION,
    STATE_SCHEMA_VERSION,
    STATE_SEQUENCE,
    TERMINAL,
    FactorGovernanceCycleStateV4_1Error,
    build_genesis_cycle_state_v4_1,
    build_next_cycle_state_v4_1,
    byte_sha256,
    canonical_file_bytes,
    canonical_json_bytes,
    semantic_sha256,
    validate_cycle_state_v4_1,
    validate_genesis_cycle_state_v4_1,
    validate_next_cycle_state_v4_1,
)


CYCLE_ID = "factor-cycle-v4.1-test"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _genesis(
    *, cycle_id: str = CYCLE_ID, cycle_root_sha256: str | None = None
) -> dict[str, object]:
    return build_genesis_cycle_state_v4_1(
        cycle_id=cycle_id,
        cycle_root_sha256=cycle_root_sha256 or _digest("cycle-root"),
        source_chain_node_sha256=_digest("precommit-source"),
    )


def _advance(
    predecessor: dict[str, object],
    next_state: str,
    *,
    cycle_id: str | None = None,
    cycle_root_sha256: str | None = None,
    predecessor_byte_sha256: str | None = None,
    expected_predecessor_byte_sha256: str | None = None,
    expected_predecessor_semantic_sha256: str | None = None,
    terminal_reason: str | None = None,
) -> dict[str, object]:
    actual_byte_sha = byte_sha256(predecessor)
    return build_next_cycle_state_v4_1(
        predecessor=predecessor,
        predecessor_byte_sha256=(
            predecessor_byte_sha256 or actual_byte_sha
        ),
        expected_predecessor_byte_sha256=(
            expected_predecessor_byte_sha256 or actual_byte_sha
        ),
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
            or str(predecessor["state_semantic_sha256"])
        ),
        cycle_id=cycle_id or str(predecessor["cycle_id"]),
        cycle_root_sha256=(
            cycle_root_sha256 or str(predecessor["cycle_root_sha256"])
        ),
        next_state=next_state,
        source_chain_node_sha256=_digest(f"source-{next_state}"),
        terminal_reason=terminal_reason,
    )


def _through(target_state: str) -> dict[str, object]:
    artifact = _genesis()
    if target_state == PRECOMMITTED:
        return artifact
    for state in STATE_SEQUENCE[1:]:
        artifact = _advance(
            artifact,
            state,
            terminal_reason="cycle_completed" if state == TERMINAL else None,
        )
        if state == target_state:
            break
    return artifact


def _reseal(artifact: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(artifact)
    result.pop("state_semantic_sha256")
    result["state_semantic_sha256"] = semantic_sha256(result)
    return result


def test_happy_full_sequence_has_exact_states_and_chain_bindings() -> None:
    artifacts = [_genesis()]
    for state in STATE_SEQUENCE[1:]:
        artifacts.append(
            _advance(
                artifacts[-1],
                state,
                terminal_reason="cycle_completed" if state == TERMINAL else None,
            )
        )

    assert [artifact["state"] for artifact in artifacts] == list(STATE_SEQUENCE)
    assert [artifact["holdout_unsealed"] for artifact in artifacts] == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert [artifact["allowed_next_state"] for artifact in artifacts] == [
        DISCOVERY,
        HOLDOUT_READY,
        HOLDOUT_UNSEALED_FINALIZING,
        TERMINAL,
        None,
    ]
    assert artifacts[0]["predecessor"] == {
        "kind": "genesis",
        "byte_sha256": GENESIS_SHA256,
        "semantic_sha256": GENESIS_SHA256,
    }
    for predecessor, artifact in zip(artifacts, artifacts[1:]):
        assert artifact["expected_predecessor_state"] == predecessor["state"]
        assert artifact["predecessor"] == {
            "kind": "cycle_state",
            "byte_sha256": byte_sha256(predecessor),
            "semantic_sha256": predecessor["state_semantic_sha256"],
        }
        assert validate_cycle_state_v4_1(artifact) == artifact
    assert artifacts[-1]["terminal_reason"] == "cycle_completed"


@pytest.mark.parametrize(
    ("predecessor_state", "next_state"),
    [
        (PRECOMMITTED, DISCOVERY),
        (DISCOVERY, HOLDOUT_READY),
        (HOLDOUT_READY, HOLDOUT_UNSEALED_FINALIZING),
        (HOLDOUT_UNSEALED_FINALIZING, TERMINAL),
    ],
)
def test_every_transition_validates_against_explicit_predecessor(
    predecessor_state: str, next_state: str
) -> None:
    predecessor = _through(predecessor_state)
    artifact = _advance(
        predecessor,
        next_state,
        terminal_reason="done" if next_state == TERMINAL else None,
    )
    predecessor_byte_sha = byte_sha256(predecessor)

    assert validate_next_cycle_state_v4_1(
        artifact,
        predecessor=predecessor,
        predecessor_byte_sha256=predecessor_byte_sha,
        expected_predecessor_byte_sha256=predecessor_byte_sha,
        expected_predecessor_semantic_sha256=str(
            predecessor["state_semantic_sha256"]
        ),
        cycle_id=CYCLE_ID,
        cycle_root_sha256=_digest("cycle-root"),
    ) == artifact


def test_transition_rejects_skipped_state() -> None:
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="transition must advance exactly PRECOMMITTED -> DISCOVERY",
    ):
        _advance(_genesis(), HOLDOUT_READY)


def test_transition_rejects_stale_byte_cas() -> None:
    genesis = _genesis()
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error, match="stale predecessor byte SHA CAS"
    ):
        _advance(
            genesis,
            DISCOVERY,
            expected_predecessor_byte_sha256=_digest("stale-byte"),
        )


def test_transition_rejects_stale_semantic_cas() -> None:
    genesis = _genesis()
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="stale predecessor semantic SHA CAS",
    ):
        _advance(
            genesis,
            DISCOVERY,
            expected_predecessor_semantic_sha256=_digest("stale-semantic"),
        )


def test_transition_rejects_byte_descriptor_not_matching_artifact() -> None:
    genesis = _genesis()
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="predecessor byte SHA does not match the normalized artifact",
    ):
        _advance(
            genesis,
            DISCOVERY,
            predecessor_byte_sha256=_digest("wrong-explicit-byte"),
            expected_predecessor_byte_sha256=_digest("wrong-explicit-byte"),
        )


@pytest.mark.parametrize("substitution", ["cycle", "root"])
def test_transition_rejects_cross_cycle_or_root_substitution(
    substitution: str,
) -> None:
    kwargs: dict[str, str] = {}
    if substitution == "cycle":
        kwargs["cycle_id"] = "other-cycle"
    else:
        kwargs["cycle_root_sha256"] = _digest("other-root")
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match=f"cross-{substitution}",
    ):
        _advance(_genesis(), DISCOVERY, **kwargs)


def test_second_or_replayed_unseal_is_rejected() -> None:
    unsealed = _through(HOLDOUT_UNSEALED_FINALIZING)
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="second or replayed holdout unseal",
    ):
        _advance(unsealed, HOLDOUT_UNSEALED_FINALIZING)


def test_terminal_state_cannot_restart_or_reopen() -> None:
    terminal = _through(TERMINAL)
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="cannot be reopened or restarted",
    ):
        _advance(terminal, PRECOMMITTED)


@pytest.mark.parametrize("bad_value", [True, 0, 1, None, "true"])
def test_validator_rejects_wrong_monotonic_holdout_flag(bad_value: object) -> None:
    artifact = _through(HOLDOUT_READY)
    artifact["holdout_unsealed"] = bad_value
    artifact = _reseal(artifact)

    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error, match="holdout_unsealed"
    ):
        validate_cycle_state_v4_1(artifact)


@pytest.mark.parametrize("bad_value", [None, PRECOMMITTED, TERMINAL, "terminal"])
def test_validator_rejects_wrong_allowed_next_state(bad_value: object) -> None:
    artifact = _through(DISCOVERY)
    artifact["allowed_next_state"] = bad_value
    artifact = _reseal(artifact)

    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error, match="allowed_next_state"
    ):
        validate_cycle_state_v4_1(artifact)


def test_validator_rejects_tampered_self_hash() -> None:
    artifact = _through(DISCOVERY)
    artifact["source_chain_node_sha256"] = _digest("tampered-source")

    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error, match="state_semantic_sha256 mismatch"
    ):
        validate_cycle_state_v4_1(artifact)


@pytest.mark.parametrize(
    "unsafe_cycle_id",
    ["", " cycle", "cycle ", ".cycle", "../cycle", "cycle/next", "a\\b", "a..b", "交易周期"],
)
def test_builder_rejects_unsafe_cycle_id(unsafe_cycle_id: str) -> None:
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error, match="exact safe non-empty"
    ):
        _genesis(cycle_id=unsafe_cycle_id)


@pytest.mark.parametrize("terminal_reason", [None, "", " ", " done", "done "])
def test_terminal_requires_exact_nonempty_reason(
    terminal_reason: str | None,
) -> None:
    predecessor = _through(HOLDOUT_UNSEALED_FINALIZING)
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="exact non-empty string at TERMINAL",
    ):
        _advance(predecessor, TERMINAL, terminal_reason=terminal_reason)


def test_nonterminal_forbids_terminal_reason() -> None:
    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="terminal_reason must be null before TERMINAL",
    ):
        _advance(_genesis(), DISCOVERY, terminal_reason="not-terminal")


@pytest.mark.parametrize(
    ("state", "bad_expected_predecessor_state"),
    [
        (PRECOMMITTED, DISCOVERY),
        (DISCOVERY, None),
        (HOLDOUT_READY, PRECOMMITTED),
        (HOLDOUT_UNSEALED_FINALIZING, DISCOVERY),
        (TERMINAL, HOLDOUT_READY),
    ],
)
def test_validator_enforces_predecessor_state_rules(
    state: str, bad_expected_predecessor_state: object
) -> None:
    artifact = _through(state)
    artifact["expected_predecessor_state"] = bad_expected_predecessor_state
    artifact = _reseal(artifact)

    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error, match="expected_predecessor_state"
    ):
        validate_cycle_state_v4_1(artifact)


def test_precommitted_requires_exact_zero_hash_genesis_descriptor() -> None:
    artifact = _genesis()
    artifact["predecessor"] = {
        "kind": "cycle_state",
        "byte_sha256": _digest("not-genesis-byte"),
        "semantic_sha256": _digest("not-genesis-semantic"),
    }
    artifact = _reseal(artifact)

    with pytest.raises(
        FactorGovernanceCycleStateV4_1Error,
        match="PRECOMMITTED predecessor must be exact genesis",
    ):
        validate_cycle_state_v4_1(artifact)


def test_non_genesis_requires_nonzero_lowercase_predecessor_hashes() -> None:
    artifact = _through(DISCOVERY)
    artifact["predecessor"] = {
        "kind": "cycle_state",
        "byte_sha256": GENESIS_SHA256,
        "semantic_sha256": "A" * 64,
    }
    artifact = _reseal(artifact)

    with pytest.raises(FactorGovernanceCycleStateV4_1Error, match="predecessor"):
        validate_cycle_state_v4_1(artifact)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "wrong_type"])
def test_artifact_schema_has_exact_fields_and_types(mutation: str) -> None:
    artifact = _genesis()
    if mutation == "missing":
        artifact.pop("source_chain_node_sha256")
    elif mutation == "unknown":
        artifact["registry_sha256"] = _digest("forbidden-registry")
    else:
        artifact["holdout_unsealed"] = 0

    with pytest.raises(FactorGovernanceCycleStateV4_1Error):
        validate_cycle_state_v4_1(artifact)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("cycle_root_sha256", GENESIS_SHA256),
        ("cycle_root_sha256", "A" * 64),
        ("source_chain_node_sha256", GENESIS_SHA256),
        ("source_chain_node_sha256", "short"),
    ],
)
def test_non_genesis_context_hashes_are_lowercase_nonzero(
    field: str, bad_value: str
) -> None:
    artifact = _genesis()
    artifact[field] = bad_value
    artifact = _reseal(artifact)

    with pytest.raises(FactorGovernanceCycleStateV4_1Error, match=field):
        validate_cycle_state_v4_1(artifact)


def test_canonical_semantic_sha_has_no_trailing_newline_ambiguity() -> None:
    payload = {"z": [1, 2], "a": "synthetic"}
    semantic_bytes = canonical_json_bytes(payload)
    file_bytes = canonical_file_bytes(payload)

    assert not semantic_bytes.endswith(b"\n")
    assert file_bytes == semantic_bytes + b"\n"
    assert semantic_sha256(payload) == hashlib.sha256(semantic_bytes).hexdigest()
    assert byte_sha256(payload) == hashlib.sha256(file_bytes).hexdigest()
    assert semantic_sha256(payload) != byte_sha256(payload)


def test_genesis_exact_schema_and_version() -> None:
    artifact = _genesis()

    assert artifact["schema_version"] == "factor-governance-cycle-state.v4.1"
    assert artifact["schema_version"] == STATE_SCHEMA_VERSION
    assert artifact["protocol_version"] == PROTOCOL_VERSION == "v4"
    assert set(artifact) == {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "cycle_root_sha256",
        "state",
        "expected_predecessor_state",
        "predecessor",
        "source_chain_node_sha256",
        "holdout_unsealed",
        "terminal_reason",
        "allowed_next_state",
        "state_semantic_sha256",
    }
    assert validate_genesis_cycle_state_v4_1(
        artifact,
        expected_cycle_id=CYCLE_ID,
        expected_cycle_root_sha256=_digest("cycle-root"),
    ) == artifact
