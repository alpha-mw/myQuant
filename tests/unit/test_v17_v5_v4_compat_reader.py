from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes, seal_semantic
from quant_investor.v17_v4_runtime.portfolio_control import build_regime_evidence
from quant_investor.v17_v5_runtime import v4_compat_reader as subject
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4CompatibilityError,
    read_v4_artifact,
)

CUTOFF = "2026-07-29T08:00:00Z"
RELATIVE_PATH = "data/private/v17_v4_runs/run-1/regime.json"


def _artifact() -> dict[str, object]:
    return build_regime_evidence(
        run_id="run-1",
        strategy_id="quant-first",
        cutoff=CUTOFF,
        role="markov_evidence",
        available_at="2026-07-29T07:59:00Z",
        gross_multiplier="0.8",
    )


def _write_artifact(root: Path, artifact: dict[str, object] | None = None) -> str:
    path = root / RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_resource_bytes(_artifact() if artifact is None else artifact)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _tree(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )


def _read(root: Path, sha256: str):
    return read_v4_artifact(
        root,
        relative_path=RELATIVE_PATH,
        expected_byte_sha256=sha256,
        expected_strategy_id="quant-first",
        decision_cutoff=CUTOFF,
    )


def test_reader_validates_exact_v4_artifact_and_writes_nothing(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path)
    before = _tree(tmp_path)

    result = _read(tmp_path, sha256)

    assert result.document["version"] == "myquant.v17.v4.regime-evidence.v1"
    assert result.document["authority"] == {
        "broker": False,
        "execution": False,
        "formal_research_publication": False,
        "order": False,
        "research_runtime_default": False,
        "trade": False,
    }
    assert result.predecessor_git_commit == "ec1370553fdf7ca0951ec4b03ea9fc426a872b4e"
    assert result.closure[0].relative_path == RELATIVE_PATH
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_byte_sha256", "0" * 64, "byte SHA-256 mismatch"),
        ("expected_strategy_id", "other-strategy", "strategy binding mismatch"),
        ("decision_cutoff", "2026-07-29T07:58:59Z", "cutoff is in the future"),
    ],
)
def test_reader_fails_closed_on_binding_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    sha256 = _write_artifact(tmp_path)
    kwargs = {
        "relative_path": RELATIVE_PATH,
        "expected_byte_sha256": sha256,
        "expected_strategy_id": "quant-first",
        "decision_cutoff": CUTOFF,
    }
    kwargs[field] = value

    with pytest.raises(V4CompatibilityError, match=message):
        read_v4_artifact(tmp_path, **kwargs)


def test_reader_rejects_unknown_version_before_v4_dispatch(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact.pop("semantic_sha256")
    artifact["version"] = "myquant.v17.v4.unknown-artifact.v1"
    sha256 = _write_artifact(tmp_path, seal_semantic(artifact))

    with pytest.raises(V4CompatibilityError, match="not allowed"):
        _read(tmp_path, sha256)


def test_reader_rejects_unallowlisted_path_symlink_and_hardlink(tmp_path: Path) -> None:
    sha256 = _write_artifact(tmp_path)
    raw = (tmp_path / RELATIVE_PATH).read_bytes()
    outside = tmp_path / "data/private/other/regime.json"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(raw)
    with pytest.raises(V4CompatibilityError, match="outside"):
        read_v4_artifact(
            tmp_path,
            relative_path="data/private/other/regime.json",
            expected_byte_sha256=sha256,
            expected_strategy_id="quant-first",
            decision_cutoff=CUTOFF,
        )

    target = tmp_path / RELATIVE_PATH
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(V4CompatibilityError, match="secure read failed|owner file"):
        _read(tmp_path, sha256)

    target.unlink()
    target.write_bytes(raw)
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(target)
    with pytest.raises(V4CompatibilityError, match="owner file"):
        _read(tmp_path, sha256)


def test_reader_detects_toctou_fingerprint_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha256 = _write_artifact(tmp_path)
    original = subject._file_fingerprint
    calls = 0

    def drifting(value):
        nonlocal calls
        calls += 1
        fingerprint = original(value)
        if calls == 11:
            return (*fingerprint[:-1], fingerprint[-1] + 1)
        return fingerprint

    monkeypatch.setattr(subject, "_file_fingerprint", drifting)
    with pytest.raises(V4CompatibilityError, match="changed during read"):
        _read(tmp_path, sha256)


def test_reader_enforces_resource_limit_and_rejects_hidden_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha256 = _write_artifact(tmp_path)
    policy = subject.load_compatibility_policy()
    policy["closure_limits"]["max_artifact_bytes"] = 1
    monkeypatch.setattr(subject, "load_compatibility_policy", lambda: policy)
    with pytest.raises(V4CompatibilityError, match="bounded owner file"):
        _read(tmp_path, sha256)
    monkeypatch.undo()

    artifact = _artifact()
    artifact.pop("semantic_sha256")
    artifact["unexpected_ref"] = {
        "artifact_id": "child",
        "artifact_version": "myquant.v17.v4.regime-evidence.v1",
        "byte_sha256": "0" * 64,
        "cutoff": CUTOFF,
        "relative_path": RELATIVE_PATH,
        "semantic_sha256": "0" * 64,
        "strategy_id": "quant-first",
    }
    sha256 = _write_artifact(tmp_path, seal_semantic(artifact))
    with pytest.raises(V4CompatibilityError, match="unallowlisted transitive"):
        _read(tmp_path, sha256)
