"""The release pointer: both readers accept one file and fail closed on the same faults."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quant_investor.operations.release_pointer import (
    ReleasePointerError,
    parse_release_pointer,
    read_release_pointer,
)

from _release_pointer_fixture import COMMIT, write_release_fixture

ROOT = Path(__file__).resolve().parents[2]
ZSH_READER = ROOT / "scripts/operations/release_pointer.sh"
SCHEDULED_SCRIPTS = (
    "scripts/operations/run_daily_factor_loop.sh",
    "scripts/operations/run_cn_evening_close.py",
    "scripts/operations/prune_market_serving.sh",
)


def _zsh(pointer: Path, workspace: Path) -> subprocess.CompletedProcess[str]:
    script = (
        f"source {ZSH_READER}; read_release_pointer {pointer} {workspace} || exit $?; "
        'print -- "$RELEASE_COMMIT|$INSTALLED_PYTHON|$EXPECTED_IMPORT_ROOT|'
        '$FACTOR_LOOP_CONTEXT|$PRUNE_RELEASE_INSTALL_DIR"'
    )
    return subprocess.run(["/bin/zsh", "-c", script], capture_output=True, text=True)


def test_both_readers_resolve_the_same_release(tmp_path: Path) -> None:
    pointer, values = write_release_fixture(tmp_path)
    workspace = tmp_path / "workspace"

    release = read_release_pointer(pointer, workspace_root=workspace)
    assert release.commit == COMMIT
    assert release.python == Path(values["RELEASE_INSTALL_DIR"]) / "bin/python"
    assert release.prune_python == release.python
    assert release.factor_loop_context == workspace / values["FACTOR_LOOP_CONTEXT"]

    completed = _zsh(pointer, workspace)
    assert completed.returncode == 0, completed.stderr
    commit, python, import_root, context, prune = completed.stdout.strip().split("|")
    assert commit == COMMIT and Path(python) == release.python
    assert import_root == f"{values['RELEASE_INSTALL_DIR']}/lib/python3.13/site-packages"
    assert Path(context) == release.factor_loop_context and prune == ""


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"RELEASE_COMMIT": "660f066"}, "RELEASE_POINTER_COMMIT_INVALID"),
        (
            {"RELEASE_INSTALL_DIR": "/nonexistent/" + COMMIT + "-x"},
            "RELEASE_POINTER_INSTALL_INVALID",
        ),
        ({"RELEASE_CHECKOUT_DIR": "/nonexistent"}, "RELEASE_POINTER_CHECKOUT_INVALID"),
        ({"RELEASE_INSTALL_INPUT_SHA256": "zz"}, "RELEASE_POINTER_SHA_INVALID"),
        ({"FACTOR_LOOP_CONTEXT_SHA256": "0" * 64}, "RELEASE_POINTER_CONTEXT_SHA_MISMATCH"),
        ({"PRUNE_RELEASE_INSTALL_DIR": "/nonexistent"}, "RELEASE_POINTER_PRUNE_INSTALL_INVALID"),
    ],
)
def test_both_readers_fail_closed_with_the_same_code(
    tmp_path: Path, override: dict[str, str], code: str
) -> None:
    pointer, _ = write_release_fixture(tmp_path, **override)
    workspace = tmp_path / "workspace"

    with pytest.raises(ReleasePointerError, match=code):
        read_release_pointer(pointer, workspace_root=workspace)
    completed = _zsh(pointer, workspace)
    assert completed.returncode == 2 and completed.stderr.strip() == code


def test_unknown_or_malformed_lines_are_rejected_not_evaluated(tmp_path: Path) -> None:
    pointer, _ = write_release_fixture(tmp_path)
    workspace = tmp_path / "workspace"
    marker = tmp_path / "evaluated"

    pointer.write_text(pointer.read_text() + f"EXTRA=$(touch {marker})\n")
    with pytest.raises(ReleasePointerError, match="RELEASE_POINTER_KEY_UNKNOWN:EXTRA"):
        read_release_pointer(pointer, workspace_root=workspace)
    assert _zsh(pointer, workspace).stderr.strip() == "RELEASE_POINTER_KEY_UNKNOWN:EXTRA"
    assert not marker.exists()

    with pytest.raises(ReleasePointerError, match="RELEASE_POINTER_LINE_INVALID"):
        parse_release_pointer("RELEASE_COMMIT " + COMMIT)
    with pytest.raises(ReleasePointerError, match="RELEASE_POINTER_KEY_MISSING:RELEASE_COMMIT"):
        parse_release_pointer("RELEASE_INSTALL_DIR=/x\n")
    with pytest.raises(ReleasePointerError, match="RELEASE_POINTER_MISSING"):
        read_release_pointer(tmp_path / "absent.env", workspace_root=workspace)


def test_scheduled_scripts_carry_no_release_path_of_their_own() -> None:
    for relative in SCHEDULED_SCRIPTS:
        text = (ROOT / relative).read_text()
        assert "release-authority" not in text and "release-checkouts" not in text, relative
        assert "operations/releases/active.env" in text, relative
