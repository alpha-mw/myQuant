from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from quant_investor.v17_mainline import (
    ACTIVE_POINTER_SCHEMA_ID,
    EMPTY_SHA256,
    MAINLINE_RUN_SCHEMA_ID,
    MainlineBlocker,
    MainlineCASMismatch,
    MainlineStore,
    MainlineUnavailable,
    PROTOCOL,
    PUBLIC_RUN_SCHEMA_ID,
    derive_mainline_state,
    read_public_run,
)
from quant_investor.v17_mainline.contracts import (
    MainlineContractError,
    build_active_pointer,
    canonical_bytes,
    parse_canonical,
    seal_document,
)
from quant_investor.v17_mainline.runtime import active_pointer_path
from quant_investor.v17_mainline.testing import write_synthetic_fixture_for_tests

STRATEGY = "cn-mainline"


def _fixture(tmp_path: Path, **kwargs: object):
    return write_synthetic_fixture_for_tests(
        tmp_path,
        strategy_id=STRATEGY,
        synthetic_only=True,
        **kwargs,
    )


def _assert_blocked(tmp_path: Path, blocker: MainlineBlocker) -> None:
    result = derive_mainline_state(tmp_path, strategy_id=STRATEGY)
    assert result.derived_state == f"V17_MAINLINE_BLOCKED:{blocker.value}"
    assert result.blocker is blocker
    assert result.public_run is None


def test_missing_pointer_is_uninitialized_and_writes_nothing(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    result = derive_mainline_state(tmp_path, strategy_id=STRATEGY)
    assert result.derived_state == "V17_MAINLINE_UNINITIALIZED"
    assert result.blocker is MainlineBlocker.ACTIVE_POINTER_ABSENT
    assert result.public_run is None
    assert tuple(tmp_path.iterdir()) == before
    with pytest.raises(MainlineUnavailable) as exc_info:
        read_public_run(tmp_path, strategy_id=STRATEGY)
    assert exc_info.value.code == "V17_MAINLINE_UNINITIALIZED"
    assert exc_info.value.blocker is MainlineBlocker.ACTIVE_POINTER_ABSENT


def test_relative_empty_workspace_is_uninitialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)
    result = derive_mainline_state("workspace", strategy_id=STRATEGY)
    assert result.derived_state == "V17_MAINLINE_UNINITIALIZED"
    assert tuple(workspace.iterdir()) == ()


def test_valid_exact_closure_derives_active_public_dto(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = derive_mainline_state(
        tmp_path,
        strategy_id=STRATEGY,
        expected_pointer_sha256=fixture.pointer_sha256,
    )
    assert result.derived_state == "ACTIVE"
    assert result.blocker is None
    dto = read_public_run(tmp_path, canonical_strategy_id=STRATEGY)
    assert dto["schema_id"] == PUBLIC_RUN_SCHEMA_ID
    assert dto["protocol"] == PROTOCOL
    assert dto["authority_source"] == "FORMAL_V17_V4"
    assert dto["market"] == "CN_A_SHARE"
    assert dto["capability"] == "RESEARCH_PORTFOLIO"
    assert dto["selector_used"] is False
    assert dto["fallback_used"] is False
    assert dto["read_only"] is True
    assert set(dto["authority_flags"].values()) == {False}
    assert dto["active_pointer_ref"]["schema_id"] == ACTIVE_POINTER_SCHEMA_ID
    assert dto["mainline_run_ref"]["schema_id"] == MAINLINE_RUN_SCHEMA_ID
    assert canonical_bytes(dto) == canonical_bytes(parse_canonical(canonical_bytes(dto)))

    for path in (
        fixture.pointer_path,
        fixture.run_path,
        fixture.formal_path,
        fixture.portfolio_path,
        fixture.source_closure_path,
    ):
        assert stat.S_IMODE((tmp_path / path).stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "results/v17_mainline").stat().st_mode) == 0o700


def test_forged_pointer_semantic_sha_is_blocked(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = tmp_path / fixture.pointer_path
    pointer = json.loads(path.read_text(encoding="ascii"))
    pointer["run_id"] = "forged-run"
    path.write_bytes(canonical_bytes(pointer))
    _assert_blocked(tmp_path, MainlineBlocker.ACTIVE_POINTER_INVALID)


def test_stale_expected_pointer_sha_is_blocked(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = derive_mainline_state(
        tmp_path,
        strategy_id=STRATEGY,
        expected_pointer_sha256="f" * 64,
    )
    assert result.blocker is MainlineBlocker.ACTIVE_POINTER_SHA_MISMATCH


def test_pointer_symlink_and_hardlink_are_storage_violations(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    pointer = tmp_path / fixture.pointer_path
    original = tmp_path / "pointer-original.json"
    original.write_bytes(pointer.read_bytes())
    original.chmod(0o600)
    pointer.unlink()
    pointer.symlink_to(original)
    _assert_blocked(tmp_path, MainlineBlocker.STORAGE_SECURITY_VIOLATION)

    pointer.unlink()
    os.link(original, pointer)
    _assert_blocked(tmp_path, MainlineBlocker.STORAGE_SECURITY_VIOLATION)


def test_pointer_cas_conflict_preserves_exact_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = MainlineStore(tmp_path)
    before = store.read(fixture.pointer_path)
    with pytest.raises(MainlineCASMismatch):
        store.compare_and_swap(
            fixture.pointer_path,
            b"forged\n",
            expected_sha256=EMPTY_SHA256,
        )
    after = store.read(fixture.pointer_path)
    assert after.data == before.data
    assert after.byte_sha256 == before.byte_sha256


def test_torn_pointer_is_invalid_not_active(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = tmp_path / fixture.pointer_path
    path.write_bytes(path.read_bytes()[:31])
    _assert_blocked(tmp_path, MainlineBlocker.ACTIVE_POINTER_INVALID)


def test_incomplete_pointer_never_scans_for_an_old_run(tmp_path: Path) -> None:
    store = MainlineStore(tmp_path)
    missing_ref = {
        "schema_id": MAINLINE_RUN_SCHEMA_ID,
        "relative_path": (
            "results/v17_mainline/strategies/cn-mainline/" "runs/missing-run/run.json"
        ),
        "byte_sha256": "a" * 64,
    }
    pointer = build_active_pointer(
        canonical_strategy_id=STRATEGY,
        run_id="missing-run",
        updated_at="2026-08-04T00:00:00Z",
        run_ref=missing_ref,
    )
    store.compare_and_swap(
        active_pointer_path(STRATEGY),
        canonical_bytes(pointer),
        expected_sha256=EMPTY_SHA256,
    )
    _assert_blocked(tmp_path, MainlineBlocker.ACTIVE_RUN_MISSING)


def test_run_sha_mismatch_is_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    store = MainlineStore(tmp_path)
    pointer_stored = store.read(fixture.pointer_path)
    pointer = parse_canonical(pointer_stored.data)
    pointer.pop("semantic_sha256")
    pointer["run_ref"]["byte_sha256"] = "f" * 64
    forged = canonical_bytes(seal_document(pointer))
    store.compare_and_swap(
        fixture.pointer_path,
        forged,
        expected_sha256=pointer_stored.byte_sha256,
    )
    _assert_blocked(tmp_path, MainlineBlocker.ACTIVE_RUN_SHA_MISMATCH)


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"market": "US_EQUITY"}, MainlineBlocker.UNSUPPORTED_MARKET),
        ({"capabilities": ["BACKTEST"]}, MainlineBlocker.UNSUPPORTED_CAPABILITY),
    ],
)
def test_unsupported_market_and_capability_fail_closed(
    tmp_path: Path,
    overrides: dict[str, object],
    blocker: MainlineBlocker,
) -> None:
    _fixture(
        tmp_path,
        run_overrides=overrides,
        allow_invalid_run_for_tests=True,
    )
    _assert_blocked(tmp_path, blocker)


def test_shadow_or_run_forward_cannot_be_authority(tmp_path: Path) -> None:
    _fixture(
        tmp_path,
        formal_overrides={
            "shadow_only": True,
            "authority_kind": "RUN_FORWARD_SHADOW",
        },
    )
    _assert_blocked(tmp_path, MainlineBlocker.SHADOW_AUTHORITY_FORBIDDEN)


@pytest.mark.parametrize(
    ("kwargs", "blocker"),
    [
        (
            {"formal_overrides": {"terminal_state": "DRAFT"}},
            MainlineBlocker.FORMAL_OUTPUT_INVALID,
        ),
        (
            {"portfolio_overrides": {"status": "PARTIAL"}},
            MainlineBlocker.PORTFOLIO_OUTPUT_INVALID,
        ),
        (
            {"source_overrides": {"source_closure_sha256": "bad"}},
            MainlineBlocker.SOURCE_CLOSURE_INVALID,
        ),
    ],
)
def test_invalid_transitive_closure_is_closed(
    tmp_path: Path,
    kwargs: dict[str, object],
    blocker: MainlineBlocker,
) -> None:
    _fixture(tmp_path, **kwargs)
    _assert_blocked(tmp_path, blocker)


def test_contract_failure_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(MainlineContractError):
        _fixture(tmp_path, run_overrides={"market": "UNSUPPORTED"})
    assert tuple(tmp_path.iterdir()) == ()


def test_synthetic_writer_is_not_an_implicit_publisher(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="synthetic_only"):
        write_synthetic_fixture_for_tests(tmp_path, strategy_id=STRATEGY)
    assert tuple(tmp_path.iterdir()) == ()
