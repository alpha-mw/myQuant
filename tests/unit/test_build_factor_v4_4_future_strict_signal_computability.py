from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
from typing import Any

import pytest

from scripts import build_factor_v4_4_future_strict_signal_computability as subject


def _manifest(*, cutoff: str = "2026-07-20") -> dict[str, Any]:
    snapshot_id = f"{cutoff.replace('-', '')}T010203Z"
    digest_chars = "123456789abcdef"
    return {
        "schema_version": subject.MANIFEST_SCHEMA_VERSION,
        "protocol_version": "v4",
        "evidence_contract_version": "v4.4",
        "cycle_id": subject.deterministic_cycle_id(
            cutoff=cutoff, snapshot_id=snapshot_id
        ),
        "cutoff": cutoff,
        "snapshot_id": snapshot_id,
        "proof_output_start": cutoff,
        "preregistration": {
            "bundle_path": "/private/tmp/future-prereg-bundle",
            "readback_byte_sha256": "1" * 64,
            "readback_semantic_sha256": "2" * 64,
            "artifact_count": 27,
            "cycle_id": f"cn_full_a_v4_4_{cutoff.replace('-', '')}_{snapshot_id}",
            "candidate_rows_semantic_sha256": "3" * 64,
        },
        "strict_source_expected": {
            "strict_source_binding_semantic_sha256": "4" * 64,
            "snapshot_manifest_byte_sha256": "5" * 64,
            "pit_generation_manifest_byte_sha256": "6" * 64,
            "pit_membership_byte_sha256": "7" * 64,
            "table_inventory_semantic_sha256": "8" * 64,
            "full_a_scope_count": 2,
            "full_a_scope_sha256": "9" * 64,
            "source_calendar_semantic_sha256": "a" * 64,
            "recorded_latest_pointer_byte_sha256": "b" * 64,
            "recorded_components_byte_sha256": "c" * 64,
        },
        "source_definition_bindings": [
            {
                "order": index,
                "name": f"candidate_{index}",
                "definition_identity_sha256": f"{index:x}" * 64,
                "direction": -1 if index % 2 else 1,
                "source_repository": "myQuant",
                "source_commit": f"{index:x}" * 40,
                "source_tree_oid": f"{index + 1:x}" * 40,
                "source_relative_path": "quant_investor/alpha158.py",
                "source_blob_oid": f"{index + 5:x}" * 40,
                "source_raw_sha256": digest_chars[index + 4] * 64,
                "source_ast_sha256": digest_chars[index + 8] * 64,
                "field_semantics_sha256": digest_chars[index + 9] * 64,
                "operator_program_sha256": "e" * 64,
                "operator_program_set_sha256": "0" * 64,
            }
            for index in range(1, 6)
        ],
        "code_binding_set": [
            {"relative_path": relative, "byte_sha256": "d" * 64}
            for relative in subject.FIXED_CODE_BINDING_PATHS
        ],
        "runtime_binding_expected_semantic_sha256": "e" * 64,
        "protected_control_expected_sha256": {
            key: "f" * 64 for key, _path in subject.PROTECTED_CONTROL_RELATIVE_PATHS
        },
        "resource_contract": dict(subject.RESOURCE_CONTRACT),
        "selection_disclosures": dict(subject.SELECTION_DISCLOSURES),
        "negative_claims": json.loads(json.dumps(subject.NEGATIVE_CLAIMS)),
    }


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_manifest(path: Path, value: object | None = None) -> tuple[Path, str]:
    raw = _canonical(_manifest() if value is None else value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return path, hashlib.sha256(raw).hexdigest()


def _publish_flags(parser: argparse.ArgumentParser) -> set[str]:
    action = next(
        item
        for item in parser._actions
        if isinstance(item, argparse._SubParsersAction)
    )
    return {
        option
        for item in action.choices["publish"]._actions
        for option in item.option_strings
    }


def test_cli_surface_is_exact_and_has_no_root_data_or_live_overrides() -> None:
    assert _publish_flags(subject.build_parser()) == {
        "-h",
        "--help",
        "--input-manifest",
        "--expected-input-manifest-byte-sha256",
    }
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint({"numpy", "pandas", "pyarrow", "quant_investor"})


def test_current_cutoff_rejects_before_project_import_or_any_later_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "_ISOLATED_CHILD_ACTIVE", True)
    value = _manifest(cutoff="2026-07-19")
    path, digest = _write_manifest(tmp_path / "manifest.json", value)
    touched: list[str] = []
    monkeypatch.setattr(
        subject,
        "_load_project_modules_after_stage0",
        lambda: touched.append("project-import"),
    )
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="strictly later"):
        subject.run_publish(
            input_manifest=path,
            expected_input_manifest_byte_sha256=digest,
        )
    assert touched == []


def test_publish_loader_is_torn_down_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(subject, "_ISOLATED_CHILD_ACTIVE", True)
    manifest = subject._stage0_manifest_validate(_manifest())
    stable = subject.StableBytes(
        path=tmp_path / "manifest.json",
        raw=b"{}\n",
        byte_sha256="a" * 64,
        signature=(1, 2, 3),
    )
    snapshot = subject.RawPrivateBundleSnapshot(
        path=tmp_path / "prereg",
        values={
            subject.PREREGISTRATION_READBACK_FILENAME: {
                "cycle_id": manifest["preregistration"]["cycle_id"]
            }
        },
        files={},
    )
    modules = SimpleNamespace()
    teardown: list[Any] = []
    monkeypatch.setattr(
        subject,
        "_read_manifest_two_fresh",
        lambda *_args, **_kwargs: (manifest, stable),
    )
    monkeypatch.setattr(
        subject, "_read_private_bundle_raw_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(
        subject, "_load_project_modules_after_stage0", lambda **_kwargs: modules
    )
    monkeypatch.setattr(
        subject, "_teardown_loaded_modules", lambda loaded: teardown.append(loaded)
    )
    monkeypatch.setattr(
        subject,
        "_run_publish_after_stage0",
        lambda **_kwargs: {"accepted": True},
    )
    assert subject.run_publish(
        input_manifest=stable.path,
        expected_input_manifest_byte_sha256=stable.byte_sha256,
    ) == {"accepted": True}
    assert teardown == [modules]

    def fail(**_kwargs: Any) -> Any:
        raise subject.FactorV4_4FutureStrictRunnerError("forced publish failure")

    monkeypatch.setattr(subject, "_run_publish_after_stage0", fail)
    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError, match="forced publish failure"
    ):
        subject.run_publish(
            input_manifest=stable.path,
            expected_input_manifest_byte_sha256=stable.byte_sha256,
        )
    assert teardown == [modules, modules]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update(unexpected=True), "fields are not exact"),
        (lambda value: value.update(cycle_id="wrong"), "cycle_id is not deterministic"),
        (
            lambda value: value["selection_disclosures"].update(
                outcome_informed_selection=False
            ),
            "outcome-informed claims",
        ),
    ],
)
def test_stage0_rejects_unknown_or_contract_drift(
    mutator: Any, message: str
) -> None:
    value = _manifest()
    mutator(value)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match=message):
        subject._stage0_manifest_validate(value)


def test_secure_manifest_rejects_duplicate_noncanonical_and_wrong_sha(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    duplicate = b'{"a":1,"a":1}\n'
    path.write_bytes(duplicate)
    path.chmod(0o600)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="duplicate"):
        subject._read_manifest_two_fresh(
            str(path), expected_byte_sha256=hashlib.sha256(duplicate).hexdigest()
        )

    noncanonical = json.dumps(_manifest(), indent=2).encode() + b"\n"
    path.write_bytes(noncanonical)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="canonical"):
        subject._read_manifest_two_fresh(
            str(path), expected_byte_sha256=hashlib.sha256(noncanonical).hexdigest()
        )

    _path, digest = _write_manifest(path)
    assert digest != "0" * 64
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="SHA mismatch"):
        subject._read_manifest_two_fresh(
            str(path), expected_byte_sha256="0" * 64
        )


def test_secure_manifest_rejects_mode_owner_symlink_and_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_manifest(tmp_path / "manifest.json")
    path.chmod(0o644)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="0600"):
        subject._read_manifest_two_fresh(str(path), expected_byte_sha256=digest)
    path.chmod(0o600)

    real_uid = os.getuid()
    monkeypatch.setattr(subject.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="owner"):
        subject._read_manifest_two_fresh(str(path), expected_byte_sha256=digest)
    monkeypatch.undo()

    link = tmp_path / "manifest-hardlink.json"
    os.link(path, link)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="hard-link"):
        subject._read_manifest_two_fresh(str(path), expected_byte_sha256=digest)
    link.unlink()

    symlink = tmp_path / "manifest-symlink.json"
    symlink.symlink_to(path)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="anchored read"):
        subject._read_manifest_two_fresh(str(symlink), expected_byte_sha256=digest)


def test_secure_manifest_two_fresh_open_rejects_inode_or_byte_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_manifest(tmp_path / "manifest.json")
    original = subject._read_owner_private_once
    calls = 0

    def swapped(*args: Any, **kwargs: Any) -> subject.StableBytes:
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == 2:
            return subject.StableBytes(
                path=value.path,
                raw=value.raw,
                byte_sha256=value.byte_sha256,
                signature=(*value.signature[:-1], value.signature[-1] + 1),
            )
        return value

    monkeypatch.setattr(subject, "_read_owner_private_once", swapped)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="two fresh"):
        subject._read_manifest_two_fresh(str(path), expected_byte_sha256=digest)


def test_manifest_path_must_be_absolute_and_normalized(tmp_path: Path) -> None:
    path, digest = _write_manifest(tmp_path / "manifest.json")
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="absolute"):
        subject._read_manifest_two_fresh(path.name, expected_byte_sha256=digest)
    aliased = str(path.parent) + "//" + path.name
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="normalized"):
        subject._read_manifest_two_fresh(aliased, expected_byte_sha256=digest)


def test_pit_projection_uses_inclusive_exclusive_historical_union() -> None:
    import numpy as np

    sessions = tuple(f"2026-07-{day:02d}" for day in range(1, 21))
    cutoff_symbols = ("000001.SZ", "600000.SH")
    cutoff_sha = hashlib.sha256("\n".join(cutoff_symbols).encode()).hexdigest()
    stack = subject.DataStack(
        np=np, pd=None, pa=None, pq=None, pc=None, evaluator=None
    )
    projected = subject._validate_pit_rows(
        [
            {
                "symbol": "000001.SZ",
                "effective_from": "2026-07-01",
                "effective_to": "",
                "list_date": "2026-07-01",
                "source_list_status": "L",
            },
            {
                "symbol": "300001.SZ",
                "effective_from": "2026-07-01",
                "effective_to": "2026-07-15",
                "list_date": "2026-07-01",
                "delist_date": "2026-07-15",
                "source_list_status": "D",
            },
            {
                "symbol": "600000.SH",
                "effective_from": "2026-07-10",
                "effective_to": "",
                "list_date": "2026-07-10",
                "source_list_status": "L",
            },
        ],
        calendar_sessions=sessions,
        cutoff_symbols=cutoff_symbols,
        expected_cutoff_sha256=cutoff_sha,
        expected_cutoff_count=2,
        stack=stack,
        membership_byte_sha256="a" * 64,
    )
    assert projected.historical_symbols == (
        "000001.SZ",
        "300001.SZ",
        "600000.SH",
    )
    delisted_column = projected.historical_symbols.index("300001.SZ")
    assert projected.eligibility_mask[13, delisted_column]
    assert not projected.eligibility_mask[14, delisted_column]
    late_ipo_column = projected.historical_symbols.index("600000.SH")
    assert not projected.eligibility_mask[8, late_ipo_column]
    assert projected.eligibility_mask[9, late_ipo_column]

    duplicate = [
        {"symbol": "000001.SZ", "effective_from": "2026-07-01", "effective_to": ""},
        {"symbol": "000001.SZ", "effective_from": "2026-07-02", "effective_to": ""},
    ]
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="one row"):
        subject._validate_pit_rows(
            duplicate,
            calendar_sessions=sessions,
            cutoff_symbols=("000001.SZ",),
            expected_cutoff_sha256=hashlib.sha256(b"000001.SZ").hexdigest(),
            expected_cutoff_count=1,
            stack=stack,
            membership_byte_sha256="a" * 64,
        )


def test_block_evaluation_streams_full_descriptors_without_full_wide_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import numpy as np
    import pandas as pd

    evaluator = importlib.import_module(
        "quant_investor.factors.governance_future_strict_exact_five_eval_v4_4"
    )
    contract = importlib.import_module(
        "quant_investor.factors.governance_future_strict_signal_computability_v4_4"
    )
    dates = tuple(
        pd.date_range("2026-01-01", periods=200, freq="D").strftime("%Y-%m-%d")
    )
    symbols = ("000001.SZ", "300001.SZ", "600000.SH")
    rows = []
    for row_index, trade_date in enumerate(dates):
        for column_index, symbol in enumerate(symbols):
            close = (
                10.0
                + column_index * 3.0
                + row_index * (0.01 + column_index * 0.005)
                + ((row_index + column_index) % 7) * 0.02
            )
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": symbol,
                    "open": close
                    * (1.0 + (((row_index + column_index) % 5) - 2) * 0.001),
                    "close": close,
                    "vol": 1000.0
                    + 20.0 * column_index
                    + (row_index % 13) * 7.0
                    + row_index * 0.3,
                    "adj_close": close * (1.0 + 0.0002 * row_index),
                }
            )
    mask = np.ones((len(dates), len(symbols)), dtype=bool)
    mask.setflags(write=False)
    pit = subject.PITProjection(
        records=tuple(
            {"symbol": symbol, "effective_from": dates[0], "effective_to": ""}
            for symbol in symbols
        ),
        historical_symbols=symbols,
        cutoff_symbols=symbols,
        eligibility_mask=mask,
        row_count=len(symbols),
        membership_byte_sha256="a" * 64,
    )
    stack = subject.DataStack(
        np=np,
        pd=pd,
        pa=None,
        pq=None,
        pc=None,
        evaluator=evaluator,
    )
    original = subject._block_input
    original_pandas_engine = evaluator.evaluate_pandas_engine_v4_4
    original_numpy_engine = evaluator.evaluate_numpy_engine_v4_4
    observed_rows: list[int] = []
    observed_programs: list[Any] = []

    def observe_block(**kwargs: Any) -> Any:
        block = original(**kwargs)
        observed_rows.append(len(block.dates))
        assert len(block.dates) <= evaluator.HALO + evaluator.OUTPUT_BLOCK
        return block

    def observe_pandas_engine(
        input_block: Any, *, operator_program_set: Any
    ) -> Any:
        observed_programs.append(operator_program_set)
        return original_pandas_engine(
            input_block, operator_program_set=operator_program_set
        )

    def observe_numpy_engine(
        input_block: Any, *, operator_program_set: Any
    ) -> Any:
        observed_programs.append(operator_program_set)
        return original_numpy_engine(
            input_block, operator_program_set=operator_program_set
        )

    monkeypatch.setattr(subject, "_block_input", observe_block)
    monkeypatch.setattr(
        evaluator, "evaluate_pandas_engine_v4_4", observe_pandas_engine
    )
    monkeypatch.setattr(
        evaluator, "evaluate_numpy_engine_v4_4", observe_numpy_engine
    )
    (
        block_manifest,
        proof_pit,
        candidate_masks,
        engine_bodies,
        full_pit,
        inputs,
        outside,
    ) = subject._evaluate_exact_five_blocks(
        pass_id="fresh_pass_1",
        table=pd.DataFrame(rows),
        pit=pit,
        calendar_sessions=dates,
        stack=stack,
        contract=contract,
        operator_program_set=contract.operator_program_set_v4_4(),
    )
    assert block_manifest["block_count"] == 2
    assert observed_rows == [188, 72]
    assert full_pit["row_count"] == len(dates)
    assert full_pit["one_count"] == len(dates) * len(symbols)
    assert proof_pit["row_count"] == len(dates) - evaluator.HALO
    assert len(candidate_masks) == 5
    assert [row["descriptor"]["row_count"] for row in inputs] == [200] * 4
    assert outside == {field: 0 for field in evaluator.INPUT_FIELDS}
    assert all(
        descriptor["finite_count"] > 0
        for engine in engine_bodies
        for descriptor in engine["raw_matrix_descriptors"].values()
    )
    assert len(observed_programs) == 4
    assert all(program is observed_programs[0] for program in observed_programs)
    assert "_full_input_matrices" not in vars(subject)


def test_streaming_binary_mask_preserves_partial_bytes_across_chunks() -> None:
    import base64
    import importlib
    import numpy as np

    contract = importlib.import_module(
        "quant_investor.factors.governance_future_strict_signal_computability_v4_4"
    )
    stack = subject.DataStack(
        np=np, pd=None, pa=None, pq=None, pc=None, evaluator=None
    )
    state = subject._StreamingPackedBinaryMask(
        symbols=("000001.SZ",), stack=stack
    )
    state.update(
        ("2026-07-20", "2026-07-21", "2026-07-22"),
        np.array([[True], [False], [True]], dtype=bool),
    )
    state.update(
        ("2026-07-23", "2026-07-24"),
        np.array([[True], [False]], dtype=bool),
    )
    state.update(
        ("2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28"),
        np.array([[False], [True], [True], [True]], dtype=bool),
    )
    descriptor = state.finalize(contract=contract)
    assert base64.b64decode(descriptor["packed_bits_base64"]) == b"\xcd\x01"
    assert descriptor["bit_count"] == 9
    assert descriptor["one_count"] == 6
    assert descriptor["padding_bit_count"] == 7


def test_parquet_projection_accepts_large_string_and_rejects_wrong_type(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    stack = subject.DataStack(
        np=None, pd=None, pa=pa, pq=pq, pc=None, evaluator=None
    )
    valid = pa.table(
        {
            "trade_date": pa.array(["2026-07-20"], type=pa.large_string()),
            "ts_code": pa.array(["000001.SZ"], type=pa.large_string()),
            "open": pa.array([1.0], type=pa.float64()),
            "close": pa.array([1.1], type=pa.float64()),
            "vol": pa.array([100.0], type=pa.float64()),
            "adj_close": pa.array([1.1], type=pa.float64()),
        }
    )
    path = tmp_path / "valid.parquet"
    pq.write_table(valid, path)
    with path.open("rb") as handle:
        projected = subject._parse_physical_table(handle, stack=stack)
    assert projected.column_names == [
        "trade_date",
        "ts_code",
        "open",
        "close",
        "vol",
        "adj_close",
    ]

    wrong = valid.set_column(1, "ts_code", pa.array([1], type=pa.int64()))
    wrong_path = tmp_path / "wrong.parquet"
    pq.write_table(wrong, wrong_path)
    with wrong_path.open("rb") as handle:
        with pytest.raises(
            subject.FactorV4_4FutureStrictRunnerError, match="prohibited Arrow type"
        ):
            subject._parse_physical_table(handle, stack=stack)


def test_stream_hash_then_parse_rejects_path_swap_after_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.json"
    raw = b'{"stable":true}\n'
    path.write_bytes(raw)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(raw)

    def swap(opened: Any) -> dict[str, bool]:
        replacement.replace(path)
        return json.loads(opened.read())

    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError, match="path identity|changed"
    ):
        subject._stream_hash_then_parse(
            path,
            label="race source",
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_size=len(raw),
            max_bytes=1024,
            expected_nlink=1,
            parser=swap,
        )


def test_git_invocation_uses_fixed_binary_and_fresh_allowlist_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    observed: dict[str, Any] = {}
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/hostile.dylib")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "objects"))
    monkeypatch.setenv("HTTPS_PROXY", "https://hostile.invalid")

    def fake_run(command: Any, **kwargs: Any) -> Any:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    output = subject._invoke_git(tmp_path, "show_toplevel")
    assert output == b"ok\n"
    assert observed["command"][0] == "/usr/bin/git"
    assert observed["command"][1:3] == ["--no-replace-objects", "-c"]
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["stdin"] == subject.subprocess.DEVNULL
    assert observed["kwargs"]["cwd"] == "/"
    environment = observed["kwargs"]["env"]
    assert environment == subject._git_environment()
    assert "DYLD_INSERT_LIBRARIES" not in environment
    assert "GIT_OBJECT_DIRECTORY" not in environment
    assert environment["HTTPS_PROXY"] == ""
    assert environment["NO_PROXY"] == "*"


def test_git_blob_binding_rejects_mode_path_and_raw_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = b"print('pinned')\n"
    blob_oid = hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw
    ).hexdigest()
    commit = "1" * 40
    root_tree = "2" * 40
    tree_path = "pkg/source.py"
    snapshot = {"stable": (1, 2, 3)}
    monkeypatch.setattr(
        subject, "_validate_git_repository", lambda _repository: snapshot
    )
    tamper: dict[str, Any] = {"tree": False, "raw": False}

    def git_read(_repository: Path, operation: str, *values: str) -> bytes:
        responses = {
            "resolve_commit": commit.encode() + b"\n",
            "resolve_tree": root_tree.encode() + b"\n",
            "ls_tree": (
                (b"100755" if tamper["tree"] else b"100644")
                + b" blob "
                + blob_oid.encode()
                + b"\t"
                + tree_path.encode()
                + b"\0"
            ),
            "object_size": str(len(raw)).encode() + b"\n",
            "cat_blob": raw + (b"tamper" if tamper["raw"] else b""),
        }
        if operation == "object_type":
            return b"tree\n" if values == (root_tree,) else (
                b"blob\n" if values == (blob_oid,) else b"commit\n"
            )
        return responses[operation]

    monkeypatch.setattr(subject, "_git_read", git_read)
    kwargs = {
        "repository": tmp_path,
        "commit": commit,
        "tree_path": tree_path,
        "blob_oid": blob_oid,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "root_tree_oid": root_tree,
        "expected_size": len(raw),
    }
    assert subject._verify_git_blob_binding(**kwargs) == raw
    tamper["tree"] = True
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="tree/blob"):
        subject._verify_git_blob_binding(**kwargs)
    tamper["tree"] = False
    tamper["raw"] = True
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="raw SHA"):
        subject._verify_git_blob_binding(**kwargs)


def test_source_ast_recomputation_is_scope_exact_and_tamper_evident() -> None:
    aquant_source = '''
def generate_default_candidates():
    def add(name, expression, family, rationale, factor_type="alpha"):
        pass
    pos = "(close - ts_min(close, {w})) / (ts_max(close, {w}) - ts_min(close, {w}))"
    for window in [20, 60, 120]:
        add(f"alpha_range_position_reversal_{window}d", f"cs_rank(-{pos.format(w=window)})", "price_momentum", "Buys stocks low in their recent range.")
        add(f"alpha_range_position_momentum_{window}d", f"cs_rank({pos.format(w=window)})", "price_momentum", "Buys stocks high in their recent range.")
'''
    aquant_tree = ast.parse(aquant_source)
    identity = subject._aquant_definition_identity_sha256(
        aquant_tree,
        pinned_commit="4424dcecc384f614b0e9fd5e36cf094e9244bad5",
    )
    assert identity == "8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f"
    duplicate = ast.parse(
        aquant_source
        + '\ndef dead():\n    add(f"alpha_range_position_momentum_{window}d", "x", "price_momentum", "x")\n'
    )
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="module-wide"):
        subject._aquant_definition_identity_sha256(
            duplicate,
            pinned_commit="4424dcecc384f614b0e9fd5e36cf094e9244bad5",
        )

    myquant_source = '''
class Alpha158:
    def _register_quality_factors(self):
        self.factor_functions['PRICE_VOL_CONSISTENCY_20D'] = lambda df: (df['close'].diff().apply(np.sign) * df['volume'].diff().apply(np.sign)).rolling(20).mean()
    def _register_volatility_regime_factors(self):
        self.factor_functions['VOL_RATIO_10_60'] = lambda df: (df['close'].pct_change().rolling(10).std() / (df['close'].pct_change().rolling(60).std() + 1e-9))
        self.factor_functions['VOL_OF_VOL_20D'] = lambda df: (df['close'].pct_change().rolling(5).std().rolling(20).std())
        self.factor_functions['OVERNIGHT_GAP_20D'] = lambda df: ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-9)).abs().rolling(20).mean()
'''
    myquant_tree = ast.parse(myquant_source)
    expected = {
        "OVERNIGHT_GAP_20D": "b34b831028f83f5aa7615d04f5dc81dd6c1b6a8d0a53899922348e68845a6196",
        "VOL_RATIO_10_60": "07327e6bfab4290088a9bbbdb1b92a80e9df23087fd255b8529b878444d32ba6",
        "PRICE_VOL_CONSISTENCY_20D": "d8b54e3b192002dba5fb4caf5adbe9a4ac26128c9cdc5750cbc71aad39398895",
        "VOL_OF_VOL_20D": "295f0b8580b0b77e749da27274b02bcb6662afeff0c6b7b22245e677ed49aa31",
    }
    for factor, method in (
        ("OVERNIGHT_GAP_20D", "_register_volatility_regime_factors"),
        ("VOL_RATIO_10_60", "_register_volatility_regime_factors"),
        ("PRICE_VOL_CONSISTENCY_20D", "_register_quality_factors"),
        ("VOL_OF_VOL_20D", "_register_volatility_regime_factors"),
    ):
        assert subject._myquant_lambda_ast_sha256(
            myquant_tree, factor, method
        ) == expected[factor]
    wrong_target = ast.parse(myquant_source.replace("self.factor_functions", "other.factor_functions", 1))
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="assignment drifted"):
        subject._myquant_lambda_ast_sha256(
            wrong_target,
            "PRICE_VOL_CONSISTENCY_20D",
            "_register_quality_factors",
        )


def test_temp_publish_readback_locked_revalidation_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy
    from types import SimpleNamespace

    from quant_investor.factors import governance_private_bundle_io as private_io
    from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_2 import (
        _portable_private_publication,
    )

    monkeypatch.setattr(subject, "_ISOLATED_CHILD_ACTIVE", True)
    _portable_private_publication(monkeypatch)
    filenames = (
        "strict_computability_input_manifest.v4_4.json",
        "strict_computability_input_receipt.v4_4.json",
        "strict_data_field_receipt.v4_4.json",
        "strict_two_pass_equivalence_receipt.v4_4.json",
        "strict_exact_five_signal_computability_proof.v4_4.json",
    )
    report_filename = "strict_signal_computability_readback.v4_4.json"
    canonical_filenames = (*filenames, report_filename)

    def validate_artifact(_filename: str, value: Any) -> dict[str, Any]:
        assert isinstance(value, dict)
        return copy.deepcopy(value)

    def validate_complete(
        values: Any,
    ) -> dict[str, dict[str, Any]]:
        assert tuple(values) == canonical_filenames
        return {
            filename: copy.deepcopy(dict(values[filename]))
            for filename in canonical_filenames
        }

    def build_report(
        *, run_id: str, artifacts: Any, artifact_bindings: Any
    ) -> dict[str, Any]:
        body = {
            "run_id": run_id,
            "cycle_id": artifacts[filenames[0]]["cycle_id"],
            "readback_scope": "SEALED_BUNDLE_GRAPH_ONLY",
            "artifact_bindings": [dict(row) for row in artifact_bindings],
            "external_predecessor_revalidated": False,
            "immutable_source_revalidated": False,
            "protected_controls_revalidated": False,
            "external_state_claimed": False,
        }
        body["artifact_semantic_sha256"] = hashlib.sha256(
            private_io.canonical_json_bytes(body)
        ).hexdigest()
        return body

    shared_contract = private_io.PrivateBundleContract(
        root_suffix=subject.ROOT_SUFFIX,
        input_filenames=filenames,
        readback_report_filename=report_filename,
        canonicalize=private_io.canonical_json_file_bytes,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_report,
        max_artifact_bytes=1024 * 1024,
        max_bundle_bytes=8 * 1024 * 1024,
    )
    collection_sha = "1" * 64
    operator_program = {"artifact_semantic_sha256": "a" * 64}
    engine_builder_programs: list[Any] = []

    def build_engine_result(**kwargs: Any) -> dict[str, Any]:
        engine_builder_programs.append(kwargs["operator_program_set"])
        return {
            "pass_id": kwargs["pass_id"],
            "engine_id": kwargs["engine_id"],
            "collection_sha256": kwargs["collection_sha256"],
        }

    fake_contract = SimpleNamespace(
        INPUT_FILENAMES=filenames,
        ROOT_SUFFIX=subject.ROOT_SUFFIX,
        INPUT_MANIFEST_FILENAME=filenames[0],
        INPUT_RECEIPT_FILENAME=filenames[1],
        DATA_FIELD_RECEIPT_FILENAME=filenames[2],
        TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME=filenames[3],
        PROOF_FILENAME=filenames[4],
        READBACK_FILENAME=report_filename,
        BUNDLE_FILENAMES=canonical_filenames,
        build_input_receipt_v4_4=lambda **_kwargs: {"kind": "input-receipt"},
        build_data_field_receipt_v4_4=lambda **_kwargs: {"kind": "data-receipt"},
        build_candidate_non_null_mask_set_v4_4=lambda **_kwargs: {
            "kind": "candidate-masks"
        },
        build_collection_descriptor_v4_4=lambda **kwargs: {
            "pass_id": kwargs["pass_id"],
            "collection_sha256": collection_sha,
        },
        build_engine_pass_result_v4_4=build_engine_result,
        build_two_pass_equivalence_receipt_v4_4=lambda **_kwargs: {
            "kind": "two-pass"
        },
        build_proof_v4_4=lambda **_kwargs: {"kind": "proof"},
        private_bundle_contract_v4_4=lambda: shared_contract,
        canonical_json_bytes_v4_4=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode(),
    )
    modules = subject.LoadedModules(
        contract=fake_contract,
        prereg_core=None,
        prereg_bundle=None,
        predecessor_bundle=None,
        private_io=private_io,
        prebound_runtime={
            "artifact_semantic_sha256": "e" * 64,
            "distributions": [],
        },
        import_guard=object(),
    )
    evaluator = SimpleNamespace(
        semantic_sha256_v4_4=lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    stack = subject.DataStack(
        np=None, pd=None, pa=None, pq=None, pc=None, evaluator=evaluator
    )
    frozen = subject.FrozenBindings(
        code=(),
        code_files=(),
        protected=(),
        runtime={"artifact_semantic_sha256": "e" * 64, "distributions": []},
    )
    root = tmp_path.joinpath(*subject.ROOT_SUFFIX)
    root.mkdir(parents=True)
    root.chmod(0o700)
    pass_calls: list[str] = []
    locked_calls: list[str] = []
    fail_locked = False

    def evidence(pass_id: str) -> subject.PassEvidence:
        body_id = "engine-a" if pass_id == "fresh_pass_1" else "engine-a"
        return subject.PassEvidence(
            pass_id=pass_id,
            accepted_preregistration_semantic_sha256="2" * 64,
            source_observation_bindings={"source": "3" * 64},
            source_identity_signatures={"source": [1, 2, 3]},
            block_manifest={
                "source_calendar": ["2026-07-20"],
                "manifest_semantic_sha256": "4" * 64,
            },
            pit_mask_descriptor={},
            proof_pit_mask_descriptor={},
            candidate_non_null_mask_descriptors=({},) * 5,
            historical_symbol_axis={},
            pit_membership_contract={},
            input_matrix_descriptors=(),
            engine_matrix_descriptors=(
                {
                    "engine_id": body_id,
                    "raw_matrix_descriptors": {
                        f"candidate_{index}": {} for index in range(1, 6)
                    },
                    "adjusted_matrix_descriptors": {
                        f"candidate_{index}": {} for index in range(1, 6)
                    },
                },
                {
                    "engine_id": "engine-b",
                    "raw_matrix_descriptors": {
                        f"candidate_{index}": {} for index in range(1, 6)
                    },
                    "adjusted_matrix_descriptors": {
                        f"candidate_{index}": {} for index in range(1, 6)
                    },
                },
            ),
            field_missing_counts={},
            outside_pit_non_null_counts={},
            bars_outside_pit_interval_count=0,
            ignored_pre_analysis_row_count=0,
            dense_projected_row_count=1,
            table_content_binding_sha256="5" * 64,
            elapsed_seconds=0.01,
            peak_rss_bytes=1024,
        )

    def execute_pass(*, pass_id: str, **_kwargs: Any) -> subject.PassEvidence:
        pass_calls.append(pass_id)
        return evidence(pass_id)

    def locked(**_kwargs: Any) -> None:
        locked_calls.append("locked")
        if fail_locked:
            raise subject.FactorV4_4FutureStrictRunnerError("locked source drift")

    monkeypatch.setattr(subject, "_accept_preregistration", lambda **_kwargs: object())
    monkeypatch.setattr(
        subject,
        "_publication_preflight",
        lambda *, manifest, **_kwargs: subject.PublicationPreflight(
            root=root,
            root_signature=subject._signature(root.stat()),
            cycle_id=manifest["cycle_id"],
        ),
    )
    monkeypatch.setattr(subject, "_collect_fixed_bindings", lambda **_kwargs: frozen)
    monkeypatch.setattr(subject, "_lazy_data_stack", lambda: stack)
    monkeypatch.setattr(subject, "_verify_loaded_stack_identity", lambda **_kwargs: None)
    monkeypatch.setattr(
        subject,
        "_verify_pinned_source_definitions",
        lambda **_kwargs: operator_program,
    )
    monkeypatch.setattr(subject, "_audit_closed_project_imports", lambda *_args: None)
    monkeypatch.setattr(subject, "_audit_closed_runtime_imports", lambda *_args: None)
    monkeypatch.setattr(
        subject,
        "_rehash_runtime_shadow",
        lambda: {"tree_semantic_sha256": "b" * 64},
    )
    monkeypatch.setattr(subject, "_execute_data_pass", execute_pass)
    monkeypatch.setattr(subject, "_locked_precommit_revalidate", locked)
    monkeypatch.setattr(
        subject, "_load_project_modules_after_stage0", lambda **_kwargs: modules
    )
    monkeypatch.setattr(
        subject, "_load_readback_modules_after_stage0", lambda **_kwargs: modules
    )

    forbidden_real_roots = (
        subject.PRODUCTION_PRIVATE_ROOT,
        subject.PROJECT_ROOT.joinpath(*subject.PREREGISTRATION_ROOT_SUFFIX),
        subject.PROJECT_ROOT / "data",
    )
    real_stat = os.stat
    real_open = os.open

    def reject_real_root_access(path: Any) -> None:
        if isinstance(path, int):
            return
        raw_path = os.fspath(path)
        candidate = Path(os.fsdecode(raw_path))
        if not candidate.is_absolute():
            return
        if any(
            candidate == forbidden or forbidden in candidate.parents
            for forbidden in forbidden_real_roots
        ):
            raise AssertionError(f"real governed root accessed: {candidate}")

    def guarded_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        reject_real_root_access(path)
        return real_stat(path, *args, **kwargs)

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> int:
        reject_real_root_access(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(os, "open", guarded_open)

    manifest_path, manifest_sha = _write_manifest(tmp_path / "manifest.json")
    manifest, stable = subject._read_manifest_two_fresh(
        str(manifest_path), expected_byte_sha256=manifest_sha
    )
    manifest = subject._stage0_manifest_validate(manifest)
    result = subject._run_publish_after_stage0(
        manifest=manifest,
        stable_manifest=stable,
        modules=modules,
        test_fault_hook=None,
        test_race_hook=None,
    )
    assert result["accepted"] is True
    assert pass_calls == ["fresh_pass_1", "fresh_pass_2"]
    assert locked_calls == ["locked"]
    assert len(engine_builder_programs) == 4
    assert all(program is operator_program for program in engine_builder_programs)
    assert not hasattr(subject, "_engine_body_from_streams")
    assert not hasattr(subject, "_finalize_engine_result")
    bundle = Path(result["bundle_path"])
    assert {path.name for path in bundle.iterdir()} == set(canonical_filenames)
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in bundle.iterdir())

    original_teardown = subject._teardown_loaded_modules
    teardown_calls: list[subject.LoadedModules] = []
    monkeypatch.setattr(
        subject,
        "_teardown_loaded_modules",
        lambda loaded: teardown_calls.append(loaded),
    )
    historical = subject.run_readback(
        bundle_path=bundle,
        expected_readback_report_byte_sha256=result[
            "readback_report_byte_sha256"
        ],
        expected_readback_report_semantic_sha256=result[
            "readback_report_semantic_sha256"
        ],
    )
    assert historical["accepted"] is True
    assert historical["external_state_claimed"] is False
    assert teardown_calls == [modules]

    original_readback = private_io.readback_private_bundle

    def fail_readback(*_args: Any, **_kwargs: Any) -> Any:
        raise subject.FactorV4_4FutureStrictRunnerError("forced sealed readback failure")

    monkeypatch.setattr(private_io, "readback_private_bundle", fail_readback)
    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError,
        match="forced sealed readback failure",
    ):
        subject.run_readback(
            bundle_path=bundle,
            expected_readback_report_byte_sha256=result[
                "readback_report_byte_sha256"
            ],
            expected_readback_report_semantic_sha256=result[
                "readback_report_semantic_sha256"
            ],
        )
    assert teardown_calls == [modules, modules]
    monkeypatch.setattr(private_io, "readback_private_bundle", original_readback)
    monkeypatch.setattr(subject, "_teardown_loaded_modules", original_teardown)

    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.iterdir()
    }
    with pytest.raises(Exception, match="already exists"):
        subject._run_publish_after_stage0(
            manifest=manifest,
            stable_manifest=stable,
            modules=modules,
            test_fault_hook=None,
            test_race_hook=None,
        )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.iterdir()
    } == before

    drift_value = _manifest(cutoff="2026-07-21")
    drift_path, drift_sha = _write_manifest(
        tmp_path / "drift-manifest.json", drift_value
    )
    drift_manifest, drift_stable = subject._read_manifest_two_fresh(
        str(drift_path), expected_byte_sha256=drift_sha
    )
    drift_manifest = subject._stage0_manifest_validate(drift_manifest)
    fail_locked = True
    with pytest.raises(Exception, match="locked source drift"):
        subject._run_publish_after_stage0(
            manifest=drift_manifest,
            stable_manifest=drift_stable,
            modules=modules,
            test_fault_hook=None,
            test_race_hook=None,
        )
    assert not (root / drift_manifest["cycle_id"]).exists()
    fail_locked = False

    race_value = _manifest(cutoff="2026-07-22")
    race_path, race_sha = _write_manifest(tmp_path / "race-manifest.json", race_value)
    race_manifest, race_stable = subject._read_manifest_two_fresh(
        str(race_path), expected_byte_sha256=race_sha
    )
    race_manifest = subject._stage0_manifest_validate(race_manifest)
    sentinel = b"competing publisher"

    def race() -> None:
        destination = root / race_manifest["cycle_id"]
        destination.mkdir(mode=0o700)
        (destination / "sentinel").write_bytes(sentinel)

    with pytest.raises(Exception):
        subject._run_publish_after_stage0(
            manifest=race_manifest,
            stable_manifest=race_stable,
            modules=modules,
            test_fault_hook=None,
            test_race_hook=race,
        )
    destination = root / race_manifest["cycle_id"]
    assert (destination / "sentinel").read_bytes() == sentinel
    assert tuple(path.name for path in destination.iterdir()) == ("sentinel",)


def test_direct_publish_and_readback_apis_reject_before_any_path_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    monkeypatch.setattr(subject, "_ISOLATED_CHILD_ACTIVE", False)
    monkeypatch.setattr(
        subject,
        "_read_manifest_two_fresh",
        lambda *_args, **_kwargs: touched.append("manifest"),
    )
    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError, match="direct nonisolated publish"
    ):
        subject.run_publish(
            input_manifest="/private/tmp/never-opened",
            expected_input_manifest_byte_sha256="0" * 64,
        )
    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError, match="direct nonisolated readback"
    ):
        subject.run_readback(
            bundle_path="/private/tmp/never-opened",
            expected_readback_report_byte_sha256="0" * 64,
            expected_readback_report_semantic_sha256="0" * 64,
        )
    assert touched == []


def test_real_pinned_sources_lower_to_the_contract_golden_program_set() -> None:
    from quant_investor.factors import (
        governance_future_strict_signal_computability_v4_4 as contract,
    )

    derived = subject._verify_pinned_source_definitions(
        manifest={
            "source_definition_bindings": list(contract.SOURCE_DEFINITION_BINDINGS)
        },
        contract=contract,
    )
    assert derived == contract.OPERATOR_PROGRAM_SET
    assert (
        derived["artifact_semantic_sha256"]
        == contract.OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    )


@pytest.mark.parametrize(
    "expression",
    [
        "lambda df: df['close'].pct_change(fill_method=None).rolling(10).std()",
        "lambda df: df['close'].pct_change(2).rolling(10).std()",
        "lambda df: df['close'].pct_change().rolling(10, min_periods=1).std()",
        "lambda df: df['close'].pct_change().rolling(10).std(ddof=0)",
    ],
)
def test_myquant_ir_lowering_rejects_nonexact_default_or_api_variants(
    expression: str,
) -> None:
    from quant_investor.factors import (
        governance_future_strict_signal_computability_v4_4 as contract,
    )

    lambda_node = ast.parse(expression, mode="eval").body
    assert isinstance(lambda_node, ast.Lambda)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError):
        subject._lower_myquant_lambda(
            lambda_node,
            adapter=contract.FIELD_SEMANTICS[2],
            source_factor="VOL_RATIO_10_60",
        )


def test_myquant_ir_lowering_expands_only_the_exact_omitted_defaults() -> None:
    from quant_investor.factors import (
        governance_future_strict_signal_computability_v4_4 as contract,
    )

    lambda_node = ast.parse(
        "lambda df: (df['close'].pct_change().rolling(10).std() / "
        "(df['close'].pct_change().rolling(60).std() + 1e-9))",
        mode="eval",
    ).body
    assert isinstance(lambda_node, ast.Lambda)
    nodes, output = subject._lower_myquant_lambda(
        lambda_node,
        adapter=contract.FIELD_SEMANTICS[2],
        source_factor="VOL_RATIO_10_60",
    )
    golden = contract.OPERATOR_PROGRAM_SET["candidates"][2]
    assert nodes == golden["nodes"]
    assert output == golden["output_node_id"]
    assert nodes[5]["parameters"] == {
        "window": 10,
        "min_periods": 10,
        "ddof": 1,
    }


def _minimal_runtime_tree(root: Path) -> None:
    root.mkdir(mode=0o700)
    for name in subject.RUNTIME_SHADOW_DIRECTORY_ROOTS:
        (root / name).mkdir(mode=0o700)
    (root / "six.py").write_bytes(b"__version__ = '1.17.0'\n")


def _scan_minimal_runtime_tree(
    root: Path, *, exact: bool = False
) -> subject.RuntimeTreeScan:
    descriptor = subject._open_runtime_root(root, "synthetic runtime root")
    try:
        return subject._scan_runtime_tree_fd(
            descriptor, exact_root_entries=exact, require_sealed=False
        )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("constant", "value", "message"),
    [
        ("RUNTIME_SHADOW_MAX_FILE_BYTES", 1, "per-file byte cap"),
        ("RUNTIME_SHADOW_MAX_FILES", 0, "fixed resource cap"),
        ("RUNTIME_SHADOW_MAX_DIRECTORIES", 1, "directory-count cap"),
        ("RUNTIME_SHADOW_MAX_TOTAL_BYTES", 1, "fixed resource cap"),
    ],
)
def test_runtime_tree_enforces_each_fixed_resource_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    value: int,
    message: str,
) -> None:
    root = tmp_path / constant
    _minimal_runtime_tree(root)
    monkeypatch.setattr(subject, constant, value)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match=message):
        _scan_minimal_runtime_tree(root)


def test_distribution_descriptor_uses_sealed_inventory_without_record_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata as importlib_metadata

    class DistributionWithoutTraversableRecord:
        version = "2.4.3"

        def locate_file(self, _path: str) -> Path:
            return tmp_path

        @property
        def files(self) -> Any:
            raise AssertionError("RECORD traversal is forbidden")

    inventory = {
        "files": [
            {
                "relative_path": "numpy/__init__.py",
                "size_bytes": 1,
                "byte_sha256": "1" * 64,
            },
            {
                "relative_path": "numpy-2.4.3.dist-info/RECORD",
                "size_bytes": 1,
                "byte_sha256": "2" * 64,
            },
        ]
    }
    monkeypatch.setattr(subject, "_ACTIVE_RUNTIME_SHADOW_INVENTORY", inventory)
    monkeypatch.setattr(subject, "_trusted_site_packages_root", lambda: tmp_path)
    monkeypatch.setattr(
        importlib_metadata,
        "distribution",
        lambda name: DistributionWithoutTraversableRecord(),
    )

    descriptor = subject._distribution_descriptor("numpy")

    assert descriptor["name"] == "numpy"
    assert descriptor["version"] == "2.4.3"
    assert descriptor["distribution_file_count"] == 2


def test_runtime_tree_rejects_missing_extra_libs_and_shadow_extra(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    _minimal_runtime_tree(missing)
    (missing / subject.RUNTIME_SHADOW_DIRECTORY_ROOTS[-1]).rmdir()
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="missing"):
        _scan_minimal_runtime_tree(missing)

    extra_libs = tmp_path / "extra-libs"
    _minimal_runtime_tree(extra_libs)
    (extra_libs / "numpy.libs").mkdir()
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match=r"\.libs"):
        _scan_minimal_runtime_tree(extra_libs)

    shadow_extra = tmp_path / "shadow-extra"
    _minimal_runtime_tree(shadow_extra)
    (shadow_extra / "seventh").mkdir()
    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError, match="missing or extra"
    ):
        _scan_minimal_runtime_tree(shadow_extra, exact=True)


def test_runtime_tree_rejects_symlink_hardlink_and_scan_copy_swap(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"must-not-be-followed")
    linked = tmp_path / "linked"
    _minimal_runtime_tree(linked)
    (linked / "numpy" / "escape.py").symlink_to(outside)
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="symlink"):
        _scan_minimal_runtime_tree(linked)
    assert outside.read_bytes() == b"must-not-be-followed"

    hardlinked = tmp_path / "hardlinked"
    _minimal_runtime_tree(hardlinked)
    first = hardlinked / "numpy" / "payload.py"
    first.write_bytes(b"x = 1\n")
    os.link(first, hardlinked / "pandas" / "payload.py")
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="one link"):
        _scan_minimal_runtime_tree(hardlinked)

    swapped = tmp_path / "swapped"
    _minimal_runtime_tree(swapped)
    shadow = tmp_path / "shadow"
    shadow.mkdir(mode=0o700)
    shadow_fd = subject._open_runtime_root(shadow, "synthetic shadow")

    def swap(stage: str) -> None:
        if stage == "after_source_scan":
            replacement = swapped / "replacement"
            replacement.write_bytes(b"__version__ = '1.17.0'\n")
            os.replace(replacement, swapped / "six.py")

    try:
        with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="swapped"):
            subject._build_runtime_shadow(
                shadow_root_fd=shadow_fd,
                shadow_root_path=shadow,
                source_root=swapped,
                _test_fault_hook=swap,
            )
    finally:
        os.close(shadow_fd)


def test_runtime_module_audit_rejects_seventh_and_no_origin(
    tmp_path: Path,
) -> None:
    import types

    finder = subject._ClosedVerifiedFinder(())
    seventh = types.ModuleType("seventh_runtime")
    seventh.__file__ = str(tmp_path / "seventh_runtime.py")
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="escaped"):
        subject._audit_closed_runtime_imports(
            finder, _module_items=(("seventh_runtime", seventh),)
        )
    no_origin = types.ModuleType("mystery_runtime")
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="no-origin"):
        subject._audit_closed_runtime_imports(
            finder, _module_items=(("mystery_runtime", no_origin),)
        )


def test_native_preflight_rejects_external_and_unresolved_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site-packages"
    package = site / "numpy"
    package.mkdir(parents=True)
    loader = package / "bad.so"
    loader.write_bytes(b"native-placeholder")
    loader.chmod(0o400)
    inventory = {"files": [{"relative_path": "numpy/bad.so"}]}
    monkeypatch.setattr(subject, "_native_rpaths", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        subject,
        "_otool_output",
        lambda option, path: (
            f"{path}:\n\t/opt/evil/libbad.dylib "
            "(compatibility version 1.0.0, current version 1.0.0)\n"
        ),
    )
    with pytest.raises(subject.FactorV4_4FutureStrictRunnerError, match="outside"):
        subject._preflight_native_shadow(shadow_site=site, inventory=inventory)

    monkeypatch.setattr(
        subject,
        "_otool_output",
        lambda option, path: (
            f"{path}:\n\t@rpath/libmissing.dylib "
            "(compatibility version 1.0.0, current version 1.0.0)\n"
        ),
    )
    with pytest.raises(
        subject.FactorV4_4FutureStrictRunnerError, match="missing or ambiguous"
    ):
        subject._preflight_native_shadow(shadow_site=site, inventory=inventory)


def test_real_parent_child_shadow_parquet_handshake_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    probe = tmp_path / "v44-isolated-probe-table.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), probe)
    probe.chmod(0o600)
    monkeypatch.setenv("PYTHONPATH", "/private/tmp/poison")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/private/tmp/poison")
    monkeypatch.setenv("HTTPS_PROXY", "http://poison.invalid")
    prefix = f"{subject.RUNTIME_SHADOW_PREFIX}{os.getpid()}-"
    assert not tuple(subject.RUNTIME_SHADOW_PARENT.glob(prefix + "*"))
    returncode = subject._run_isolated_parent((), _test_probe_path=probe)
    output = json.loads(capsys.readouterr().out.strip())
    assert returncode == 0
    assert output == {
        "accepted": True,
        "column_count": 1,
        "mode": "private_isolated_parquet_probe",
        "row_count": 3,
    }
    assert not tuple(subject.RUNTIME_SHADOW_PARENT.glob(prefix + "*"))


def test_real_parent_child_early_manifest_error_cleans_shadow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    value = _manifest(cutoff="2026-07-19")
    manifest, digest = _write_manifest(tmp_path / "early.json", value)
    prefix = f"{subject.RUNTIME_SHADOW_PREFIX}{os.getpid()}-"
    returncode = subject._run_isolated_parent(
        (
            "publish",
            "--input-manifest",
            str(manifest),
            "--expected-input-manifest-byte-sha256",
            digest,
        )
    )
    output = json.loads(capsys.readouterr().out.strip())
    assert returncode == 2
    assert output["accepted"] is False
    assert "strictly later" in output["error"]
    assert not tuple(subject.RUNTIME_SHADOW_PARENT.glob(prefix + "*"))


def test_handled_parent_signal_anchored_cleans_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SignalledChild:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            del timeout
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("signal handler must interrupt communicate")

        def send_signal(self, signum: int) -> None:
            self.returncode = -signum

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            self.returncode = -signal.SIGKILL

    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: SignalledChild())
    prefix = f"{subject.RUNTIME_SHADOW_PREFIX}{os.getpid()}-"
    assert subject._run_isolated_parent(()) == 128 + signal.SIGTERM
    assert not tuple(subject.RUNTIME_SHADOW_PARENT.glob(prefix + "*"))


def test_stale_shadow_cleanup_is_exact_dead_pid_owner_private_and_non_symlink(
    tmp_path: Path,
) -> None:
    dead_pid = 99_999_999
    assert subject._process_is_absent(dead_pid)
    seed = hashlib.sha256(os.fsencode(str(tmp_path))).hexdigest()
    stale = subject.RUNTIME_SHADOW_PARENT / (
        f"{subject.RUNTIME_SHADOW_PREFIX}{dead_pid}-{seed[:32]}"
    )
    wrong_mode = subject.RUNTIME_SHADOW_PARENT / (
        f"{subject.RUNTIME_SHADOW_PREFIX}{dead_pid}-{seed[32:]}"
    )
    linked = subject.RUNTIME_SHADOW_PARENT / (
        f"{subject.RUNTIME_SHADOW_PREFIX}{dead_pid}-{seed[1:33]}"
    )
    near_prefix = subject.RUNTIME_SHADOW_PARENT / (
        f"{subject.RUNTIME_SHADOW_PREFIX}{dead_pid}-{seed[:31]}x"
    )
    link_target = tmp_path / "link-target"

    for path in (stale, wrong_mode, linked, near_prefix):
        assert not path.exists() and not path.is_symlink()
    try:
        stale.mkdir(mode=0o700)
        (stale / "nested").mkdir(mode=0o700)
        (stale / "nested" / "payload").write_bytes(b"stale")
        wrong_mode.mkdir(mode=0o755)
        near_prefix.mkdir(mode=0o700)
        link_target.mkdir(mode=0o700)
        linked.symlink_to(link_target, target_is_directory=True)

        subject._cleanup_stale_runtime_shadows()

        assert not stale.exists()
        assert wrong_mode.is_dir()
        assert linked.is_symlink()
        assert near_prefix.is_dir()
    finally:
        if linked.is_symlink():
            linked.unlink()
        if near_prefix.is_dir():
            near_prefix.rmdir()
        if wrong_mode.is_dir():
            wrong_mode.chmod(0o700)
            wrong_mode.rmdir()
        if stale.is_dir():
            payload = stale / "nested" / "payload"
            if payload.exists():
                payload.unlink()
            nested = stale / "nested"
            if nested.is_dir():
                nested.rmdir()
            stale.rmdir()
