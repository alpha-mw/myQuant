from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_investor.factors import (
    governance_prior_diagnostic_nomination_v4_3 as diagnostic,
)
from scripts import build_factor_v4_3_prior_diagnostic_nomination as subject
from tests.unit.test_factor_governance_prior_diagnostic_nomination_bundle_v4_3 import (
    _bundle_artifacts,
    _portable_private_publication,
)


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(
        *subject.bundle_v4_3.ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION
    )
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _args(
    *,
    snapshot_id: str = subject.FIXED_SNAPSHOT_ID,
    analysis_start: str = subject.FIXED_ANALYSIS_START,
    cutoff: str = subject.FIXED_CUTOFF,
    identity_sha256: str = diagnostic.DEFINITION_IDENTITY_SHA256,
) -> argparse.Namespace:
    # A synthetic collector is substituted in publication tests.  These are
    # deliberately only the fields consulted before that collector boundary.
    return argparse.Namespace(
        snapshot_id=snapshot_id,
        analysis_start=analysis_start,
        cutoff=cutoff,
        expected_definition_identity_sha256=identity_sha256,
    )


def _entry() -> subject.PublicationInputs:
    return subject.PublicationInputs(
        run_id=diagnostic.RUN_ID,
        artifacts=_bundle_artifacts(),
    )


def _file_hashes(bundle_path: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle_path.iterdir()
    }


def test_git_object_reads_scrub_inherited_git_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout=b"bound\n", stderr=b"")

    monkeypatch.setenv("GIT_DIR", "/tmp/redirected-object-store")
    monkeypatch.setenv("GIT_WORK_TREE", "/tmp/redirected-work-tree")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/redirected-objects")
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", "/tmp/alternate-objects")
    monkeypatch.setenv("GIT_INDEX_FILE", "/tmp/redirected-index")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.sshCommand")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    monkeypatch.setattr(subject.subprocess, "run", fake_run)

    assert subject._run_git(tmp_path, ["rev-parse", "HEAD"]) == b"bound\n"
    environment = captured["environment"]
    assert set(key for key in environment if key.startswith("GIT_")) == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
    }
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert captured["command"] == [
        "git",
        "-C",
        str(tmp_path),
        "rev-parse",
        "HEAD",
    ]


def test_public_cli_is_fixed_and_has_no_mutation_or_identity_overrides() -> None:
    parser = subject.build_parser()
    help_text = parser.format_help()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    help_text += "".join(
        command.format_help() for command in subparsers.choices.values()
    )
    for token in subject._FORBIDDEN_ARGUMENT_TOKENS:
        assert token not in help_text
    assert str(subject.PRODUCTION_PRIVATE_ROOT) == (
        "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
        "v4_3_prior_diagnostic_nomination"
    )
    assert subject._validate_run_identity(
        snapshot_id=subject.FIXED_SNAPSHOT_ID,
        analysis_start=subject.FIXED_ANALYSIS_START,
        cutoff=subject.FIXED_CUTOFF,
    ) == diagnostic.RUN_ID


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (
            _args(snapshot_id="20260718T172132Z"),
            "stale or different diagnostic identity",
        ),
        (
            _args(
                identity_sha256=(
                    "227d307ebd56ca81418e4fb8836c6aae0e41a528ff06ec2c705b5d264eab64fa"
                )
            ),
            "superseded provisional definition identity",
        ),
    ],
)
def test_stale_or_provisional_identity_rejects_before_probe_collect_or_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: argparse.Namespace,
    match: str,
) -> None:
    root = tmp_path.joinpath(
        *subject.bundle_v4_3.ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION
    )
    calls = {"probe": 0, "collect": 0, "publish": 0, "readback": 0}

    def counted(name: str) -> Any:
        calls[name] += 1
        raise AssertionError(f"{name} must not run before identity rejection")

    monkeypatch.setattr(
        subject,
        "_collect_publication_inputs",
        lambda *_args, **_kwargs: counted("collect"),
    )
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "publish_prior_diagnostic_nomination_bundle_v4_3",
        lambda **_kwargs: counted("publish"),
    )
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "readback_prior_diagnostic_nomination_bundle_v4_3",
        lambda *_args, **_kwargs: counted("readback"),
    )

    with pytest.raises(
        subject.FactorV4_3PriorDiagnosticNominationRunnerError,
        match=match,
    ):
        subject.run_publish(
            args,
            private_root=root,
            exclusive_rename_probe=lambda: counted("probe"),
        )

    assert calls == {"probe": 0, "collect": 0, "publish": 0, "readback": 0}
    assert not root.exists()


def test_publish_uses_two_collections_real_shared_io_and_historical_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    entry = _entry()
    calls = {"collect": 0, "publish": 0, "readback": 0, "probe": 0}

    def collect(*_args: Any, **_kwargs: Any) -> subject.PublicationInputs:
        calls["collect"] += 1
        return entry

    real_publish = (
        subject.bundle_v4_3.publish_prior_diagnostic_nomination_bundle_v4_3
    )
    real_readback = (
        subject.bundle_v4_3.readback_prior_diagnostic_nomination_bundle_v4_3
    )

    def publish(**kwargs: Any) -> dict[str, Any]:
        calls["publish"] += 1
        return real_publish(**kwargs)

    def readback(path: str | os.PathLike[str]) -> dict[str, Any]:
        calls["readback"] += 1
        return real_readback(path)

    def probe() -> None:
        calls["probe"] += 1

    monkeypatch.setattr(subject, "_collect_publication_inputs", collect)
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "publish_prior_diagnostic_nomination_bundle_v4_3",
        publish,
    )
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "readback_prior_diagnostic_nomination_bundle_v4_3",
        readback,
    )

    result = subject.run_publish(
        _args(), private_root=root, exclusive_rename_probe=probe
    )
    assert calls == {"collect": 2, "publish": 1, "readback": 1, "probe": 1}
    assert result["accepted"] is True
    assert result["publisher_return_accepted"] is True
    assert result["independent_reopen_accepted"] is True
    assert result["authority"] == diagnostic.AUTHORITY_FLAGS
    assert result["side_effects"] == diagnostic.SIDE_EFFECT_FLAGS

    bundle_path = Path(result["bundle_path"])
    assert bundle_path == root / diagnostic.RUN_ID
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(bundle_path.stat().st_mode) == 0o700
    bundle_files = tuple(bundle_path.iterdir())
    assert len(bundle_files) == 3
    assert {path.name for path in bundle_files} == {
        *subject.bundle_v4_3.INPUT_FILENAMES_V4_3,
        subject.bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
    }
    for path in bundle_files:
        metadata = path.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1

    # Independently inspect the exact descriptor mapping and report hashes.
    observed = real_readback(bundle_path)
    descriptors = observed["artifact_descriptors"]
    assert isinstance(descriptors, dict)
    assert len(descriptors) == 3
    report_name = (
        subject.bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
    )
    report_descriptor = descriptors[report_name]
    assert result["readback_report_path"] == report_descriptor["absolute_path"]
    assert result["readback_report_byte_sha256"] == report_descriptor["byte_sha256"]
    assert (
        result["readback_report_semantic_sha256"]
        == observed["readback_report"]["artifact_semantic_sha256"]
    )
    assert report_descriptor["byte_sha256"] == hashlib.sha256(
        (bundle_path / report_name).read_bytes()
    ).hexdigest()

    historical = subject.run_readback(
        argparse.Namespace(
            bundle_path=str(bundle_path),
            expected_readback_report_byte_sha256=result[
                "readback_report_byte_sha256"
            ],
            expected_readback_report_semantic_sha256=result[
                "readback_report_semantic_sha256"
            ],
        )
    )
    assert calls["readback"] == 2
    assert historical["accepted"] is True
    assert historical["current_mutable_sources_read"] is False
    assert historical["authority"] == diagnostic.AUTHORITY_FLAGS
    assert historical["side_effects"] == diagnostic.SIDE_EFFECT_FLAGS

    before = _file_hashes(bundle_path)
    assert len(before) == 3
    before_root_inventory = sorted(path.name for path in root.iterdir())
    before_counts = dict(calls)
    with pytest.raises(
        subject.FactorV4_3PriorDiagnosticNominationRunnerError,
        match="already exists",
    ):
        subject.run_publish(
            _args(), private_root=root, exclusive_rename_probe=probe
        )
    assert calls["collect"] == before_counts["collect"]
    assert calls["publish"] == before_counts["publish"]
    assert calls["readback"] == before_counts["readback"]
    assert calls["probe"] == before_counts["probe"] + 1
    assert sorted(path.name for path in root.iterdir()) == before_root_inventory
    assert _file_hashes(bundle_path) == before


def test_locked_second_collection_drift_rejects_without_final_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    first = _entry()
    drifted = subject.PublicationInputs(
        run_id=first.run_id,
        artifacts={**copy.deepcopy(first.artifacts), "unexpected.v4_3.json": {}},
    )
    values = iter((first, drifted))
    collection_count = 0

    def collect(*_args: Any, **_kwargs: Any) -> subject.PublicationInputs:
        nonlocal collection_count
        collection_count += 1
        return next(values)

    monkeypatch.setattr(subject, "_collect_publication_inputs", collect)
    with pytest.raises(Exception, match="publication inputs changed before commit"):
        subject.run_publish(
            _args(),
            private_root=root,
            exclusive_rename_probe=lambda: None,
        )
    assert collection_count == 2
    assert not (root / diagnostic.RUN_ID).exists()


def test_signals_use_explicit_node_level_nonfinite_replacement_and_pit_remask() -> None:
    index = pd.date_range("2025-01-01", periods=310, freq="D")
    columns = ["000001.SZ", "600000.SH"]
    values = np.column_stack(
        (
            np.linspace(10.0, 50.0, len(index)),
            np.linspace(30.0, 5.0, len(index)),
        )
    )
    price = pd.DataFrame(values, index=index, columns=columns)
    mask = pd.DataFrame(True, index=index, columns=columns)
    mask.iloc[40:43, 0] = False
    mask.iloc[170, 1] = False
    price.iloc[80, 1] = np.inf

    observed = subject._signals(price, mask)
    clean = price.replace([np.inf, -np.inf], np.nan).where(mask)
    return_1d = clean.pct_change(periods=1, fill_method=None)
    return_1d = return_1d.replace([np.inf, -np.inf], np.nan).where(mask)
    vol5 = return_1d.rolling(
        window=5, min_periods=5, center=False
    ).std(ddof=1)
    vol5 = vol5.replace([np.inf, -np.inf], np.nan).where(mask)
    expected_vol = vol5.rolling(
        window=20, min_periods=20, center=False
    ).std(ddof=1)
    expected_vol = expected_vol.replace([np.inf, -np.inf], np.nan).where(mask)

    mom252 = clean.pct_change(periods=252, fill_method=None)
    mom252 = mom252.replace([np.inf, -np.inf], np.nan).where(mask)
    expected_mom = mom252.shift(periods=21)
    expected_mom = expected_mom.replace([np.inf, -np.inf], np.nan).where(mask)

    mom60 = clean.pct_change(periods=60, fill_method=None)
    mom60 = mom60.replace([np.inf, -np.inf], np.nan).where(mask)
    mean120 = mom60.rolling(
        window=120, min_periods=120, center=False
    ).mean()
    mean120 = mean120.replace([np.inf, -np.inf], np.nan).where(mask)
    expected_excess = mom60.subtract(mean120)
    expected_excess = expected_excess.replace(
        [np.inf, -np.inf], np.nan
    ).where(mask)

    pd.testing.assert_frame_equal(observed["VOL_OF_VOL_20D"], expected_vol)
    pd.testing.assert_frame_equal(observed["MOM_12M_SKIP1M"], expected_mom)
    pd.testing.assert_frame_equal(observed["EXCESS_MOM_60D"], expected_excess)
    assert all(frame.where(~mask).isna().all().all() for frame in observed.values())


def test_assignment_key_accepts_only_literal_subscript_identity() -> None:
    literal = ast.parse("self.factor_functions['VOL_OF_VOL_20D'] = lambda x: x")
    dynamic = ast.parse("self.factor_functions[name] = lambda x: x")
    literal_target = next(
        node.targets[0] for node in ast.walk(literal) if isinstance(node, ast.Assign)
    )
    dynamic_target = next(
        node.targets[0] for node in ast.walk(dynamic) if isinstance(node, ast.Assign)
    )
    assert subject._assignment_key(literal_target) == "VOL_OF_VOL_20D"
    assert subject._assignment_key(dynamic_target) is None
