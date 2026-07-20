#!/usr/bin/env python3
"""Build the bounded exact-37 Factor v4.1 computability proof.

The runner is read-only except for one owner-private, no-clobber evidence
bundle.  It never imports A_quant, opens the A_quant worktree, invokes a data
provider, generates candidates, constructs a portfolio, or touches an
execution surface.
"""

from __future__ import annotations

import argparse
import ast
import base64
import builtins
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import (  # noqa: E402
    governance_aquant_no_label_eval_v4_1 as evaluator,
)
from quant_investor.factors import (  # noqa: E402
    governance_no_label_diagnostic_v4_1 as no_label,
)
from quant_investor.factors import (  # noqa: E402
    governance_operator_runtime_equivalence_v4_1 as operator_equivalence,
)
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import (  # noqa: E402
    governance_signal_computability_v4_1 as contract,
)


GIT_EXECUTABLE = Path("/usr/bin/git")
MATRIX_DATASET_PATH = "A_quant/app/factor_sandbox/matrix_dataset.py"
PROVIDER_CONTEXT_PATH = "A_quant/app/data/providers/tushare.py"
SETTINGS_CONTEXT_PATH = "A_quant/app/config/settings.py"
BARS_TREE_PATH = "A_quant/data/parquet/bars"
FINANCIAL_TREE_PATH = "A_quant/data/parquet/financials"

EXPECTED_SOURCE_BLOBS = {
    MATRIX_DATASET_PATH: {
        "oid": "ef6f6d0a408176a0e3151d619d097c5190d60ef8",
        "size": 19091,
        "sha256": "eab9ba96576d040622ae170fc36689a4ee62b64f13a91ae0efe9ff9cd8942547",
    },
    PROVIDER_CONTEXT_PATH: {
        "oid": "87f08b40d0a371445a7aa8be74ea326a46bf79b1",
        "size": 48041,
        "sha256": "c90f223da6c748e0d93bad1b101d8c63e920f34a437b4f63d9451627981a283f",
    },
    SETTINGS_CONTEXT_PATH: {
        "oid": "bbdc3f268185cb3f958c452ff52ded48d2f11908",
        "size": 7711,
        "sha256": "f0929499d90a630850082844d2f8703b096e146a5f984fcb178024a08419c22b",
    },
}

EXPECTED_TRANSFORMATION_AST_ROWS = (
    ("constant", "FINANCIAL_BASE_COLUMNS", "b4422845337ec9fe8dd8700b89258375d307c7149463252f342815c335cc29d0"),
    ("constant", "FINANCIAL_GROWTH_COLUMNS", "3353a6f0f3020e4a454e0ad7bb04ed0566259eb7b7252e277859195df58cca35"),
    ("method", "_pivot", "626e1653b17bbe2e2217040073566192872f577019397bdb2fdccb82c91ca275"),
    ("method", "_align_symbol_financials", "6c5f4acda6e6e7fef482aa99acafafcf227d0d783ddcbc6b1b8e64b33eac2550"),
    ("method", "_derive_financial_columns", "b913ec9221080dbacef8904536215afb1972495ec9f80803405efdb610c87397"),
    ("method", "_build_valuation_matrices", "f6d7855e3f750fac508bd4525cdb26387cc77f60cbd8518da4f38abb33b5a18f"),
    ("method", "_numeric_col", "81b0ebc98ebd31e3306608aee0163ccb0aae41003f45384bc2428aca336decf0"),
    ("method", "_safe_divide", "955a75610dd54a9078a4b1d826bf7ebeea756a2633a70ae3c67a474a9b2f5ba4"),
    ("adapter", "_normalize_bar_frame", "cb1d18110f3bf16e207f642fd1d1fe9cdd8d6218486c1a50f955464db1084ad9"),
)

EXISTING_CODE_HASHES = {
    "governance_aquant_no_label_eval_v4_1.py": "454af650ddfcbb05df56f4098e596f0f2a80d1d27f20b356c0cf5f6db19ffa71",
    "governance_no_label_diagnostic_v4_1.py": "a23891aaca91b7553f48be91746ec470dd1a207e434e6d8cb7805a08ab4a3900",
    "governance_operator_runtime_equivalence_v4_1.py": "dd39db3f7ab0661451238ec65e3cfaed48f612108e205b7214ad30e5374dceb8",
    "governance_private_bundle_io.py": "b61c10ea9d5970f8a0708336942492be23e6d7e82d3609f176ad5a6170d85406",
}

EXPECTED_LOCAL_IMPORTS = {
    "quant_investor.factors.governance_aquant_no_label_eval_v4_1",
    "quant_investor.factors.governance_no_label_diagnostic_v4_1",
    "quant_investor.factors.governance_operator_runtime_equivalence_v4_1",
    "quant_investor.factors.governance_private_bundle_io",
    "quant_investor.factors.governance_signal_computability_v4_1",
}
FORBIDDEN_CONTEXT_MODULES = {
    "quant_investor.factors.governance_aquant_input_resolution_v4_1",
    "quant_investor.factors.governance_same_snapshot_screening_v4_1",
}

EXPECTED_MARKET_INVENTORY_SHA256 = (
    "d3b281045dfa34af49371a2847877920a062ac077aeee8525d381fc4713a7330"
)
EXPECTED_COMPONENT_COUNT = 5502
EXPECTED_PIT_COUNT = 5866
EXPECTED_ANALYSIS_START = "2021-06-25"
EXPECTED_CUTOFF = "2026-07-17"
MARKET_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)

MAX_GIT_BLOB_BYTES = contract.EXPECTED_RESOURCE_LIMITS["max_git_blob_bytes"]
MAX_FULL_GIT_INPUT_BYTES = contract.EXPECTED_RESOURCE_LIMITS[
    "max_full_git_input_bytes"
]
MAX_SELECTED_BARS_BYTES = contract.EXPECTED_RESOURCE_LIMITS[
    "max_selected_bars_bytes"
]
MAX_FINANCIAL_BYTES = contract.EXPECTED_RESOURCE_LIMITS["max_financial_bytes"]
MAX_SELECTED_BAR_ROWS = contract.EXPECTED_RESOURCE_LIMITS["max_selected_bar_rows"]
MAX_FINANCIAL_ROWS = contract.EXPECTED_RESOURCE_LIMITS["max_financial_rows"]
MAX_AXIS_CELLS = contract.EXPECTED_RESOURCE_LIMITS["max_axis_cells"]
MAX_PRIMITIVE_MATRICES = contract.EXPECTED_RESOURCE_LIMITS[
    "max_primitive_matrices"
]
MAX_FRAME_BYTES = contract.EXPECTED_RESOURCE_LIMITS["max_frame_bytes"]
MAX_MESSAGE_BYTES = contract.EXPECTED_RESOURCE_LIMITS["max_message_bytes"]
MAX_CHILD_OUTPUT_BYTES = contract.EXPECTED_RESOURCE_LIMITS[
    "max_child_output_bytes"
]
MAX_CHILD_SECONDS = contract.EXPECTED_RESOURCE_LIMITS["max_child_seconds"]
MAX_TOTAL_WALL_SECONDS = contract.EXPECTED_RESOURCE_LIMITS[
    "max_total_wall_seconds"
]
MAX_CHILD_RSS_BYTES = contract.EXPECTED_RESOURCE_LIMITS["max_child_rss_bytes"]
MAX_PARENT_RSS_BYTES = contract.EXPECTED_RESOURCE_LIMITS["max_parent_rss_bytes"]
CHILD_ADDRESS_SPACE_BYTES = contract.EXPECTED_RESOURCE_LIMITS[
    "child_address_space_bytes"
]
CHILD_DATA_BYTES = contract.EXPECTED_RESOURCE_LIMITS["child_data_bytes"]
CHILD_NOFILE = contract.EXPECTED_RESOURCE_LIMITS["child_nofile"]

LOGICAL_FINANCIAL_COLUMNS = (
    "symbol",
    "report_period",
    "report_type",
    "announce_date",
    "availability_date",
    "revenue",
    "operating_profit",
    "net_profit",
    "eps",
    "total_assets",
    "total_equity",
    "total_debt",
    "operating_cashflow",
    "free_cashflow",
    "roe",
    "roa",
    "debt_to_equity",
)


class FactorV4_1SignalComputabilityRunnerError(ValueError):
    """An untrusted intake or runner invariant failed; no evidence is valid."""


class TrustedComputabilityBlocker(ValueError):
    """Trusted intake completed, but the bounded computability claim failed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"{context} is not a SHA-256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"{context} is not a SHA-256"
        ) from exc
    return value


def _absolute_path(value: Any, context: str) -> Path:
    if type(value) is not str or not value:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"{context} must be an absolute path"
        )
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"{context} must be absolute and normalized"
        )
    return path


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stable_bytes(
    path: Path,
    *,
    expected_sha256: str,
    private: bool,
) -> bytes:
    expected = _sha(expected_sha256, f"expected SHA for {path}")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FactorV4_1SignalComputabilityRunnerError(
            f"bound path is not a regular non-symlink file: {path}"
        )
    if private and (
        before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or int(before.st_nlink) != 1
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            f"private binding owner/mode/link check failed: {path}"
        )
    first = path.read_bytes()
    middle = os.lstat(path)
    second = path.read_bytes()
    after = os.lstat(path)
    if not _signature(before) == _signature(middle) == _signature(after) or first != second:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"bound file changed across stable readback: {path}"
        )
    if hashlib.sha256(first).hexdigest() != expected:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"bound file SHA mismatch: {path}"
        )
    return first


def _strict_json(raw: bytes, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in items:
            if key in payload:
                raise FactorV4_1SignalComputabilityRunnerError(
                    f"duplicate JSON key in {context}: {key}"
                )
            payload[key] = value
        return payload

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"bound JSON parse failed: {context}"
        ) from exc
    if not isinstance(value, dict):
        raise FactorV4_1SignalComputabilityRunnerError(
            f"bound JSON must be an object: {context}"
        )
    return value


def _ast_sha(raw: bytes, path: Path) -> str:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"code AST parse failed: {path}"
        ) from exc
    return hashlib.sha256(
        ast.dump(tree, include_attributes=True).encode("utf-8")
    ).hexdigest()


def _parse_porcelain(root: Path) -> dict[str, str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    result: dict[str, str] = {}
    for record in (item for item in raw.split(b"\0") if item):
        if record[2:3] != b" ":
            raise FactorV4_1SignalComputabilityRunnerError(
                "worktree baseline encountered unsupported rename/copy status"
            )
        status_code = record[:2].decode("ascii")
        path = record[3:].decode("utf-8")
        if status_code not in {" M", "??"}:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"unsupported dirty status {status_code!r}: {path}"
            )
        result[path] = status_code
    return result


def _dirty_row(root: Path, path_text: str, status_code: str) -> dict[str, Any]:
    path = root / path_text
    metadata = os.lstat(path)
    if stat.S_ISREG(metadata.st_mode):
        kind = "regular"
        content = path.read_bytes()
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        content = os.readlink(path).encode("utf-8")
    else:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"unsupported dirty path type: {path_text}"
        )
    if status_code == "??":
        diff = b"untracked-v1\0" + path_text.encode("utf-8") + b"\0" + content
        tracked = False
    else:
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-color",
                "HEAD",
                "--",
                path_text,
            ],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        tracked = True
    return {
        "byte_sha256": hashlib.sha256(content).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "path": path_text,
        "size": len(content),
        "status": status_code,
        "tracked": tracked,
        "type": kind,
    }


def _validate_worktree_baseline(
    *, root: Path, baseline: Mapping[str, Any]
) -> None:
    if baseline.get("schema_version") != (
        "factor-governance-v4.1-worktree-content-baseline.v1"
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "worktree content baseline schema mismatch"
        )
    rows = baseline.get("dirty_paths")
    if not isinstance(rows, list) or baseline.get("dirty_path_count") != len(rows):
        raise FactorV4_1SignalComputabilityRunnerError(
            "worktree content baseline row count mismatch"
        )
    if _semantic_sha(rows) != baseline.get("dirty_paths_semantic_sha256"):
        raise FactorV4_1SignalComputabilityRunnerError(
            "worktree content baseline semantic SHA mismatch"
        )
    permitted = set(baseline.get("permitted_delta_paths") or [])
    current = _parse_porcelain(root)
    baseline_paths = {str(row.get("path")) for row in rows}
    unexpected = sorted(set(current) - baseline_paths - permitted)
    if unexpected:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worktree gained an unpermitted dirty path: " + ",".join(unexpected)
        )
    for expected in rows:
        path_text = str(expected.get("path"))
        if path_text in permitted:
            continue
        if current.get(path_text) != expected.get("status"):
            raise FactorV4_1SignalComputabilityRunnerError(
                f"pre-existing dirty status drift: {path_text}"
            )
        actual = _dirty_row(root, path_text, current[path_text])
        if actual != expected:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"pre-existing dirty file byte/diff drift: {path_text}"
            )


def _local_imports(raw: bytes, path: Path) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"import-closure AST parse failed: {path}"
        ) from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "quant_investor.factors":
                for alias in node.names:
                    imports.add(f"{node.module}.{alias.name}")
            elif node.module.startswith("quant_investor"):
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("quant_investor")
            )
    return imports


def _read_code_bindings(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    paths = {
        "build_factor_v4_1_signal_computability.py": (
            root / "scripts/build_factor_v4_1_signal_computability.py",
            args.expected_builder_sha256,
        ),
        "governance_aquant_no_label_eval_v4_1.py": (
            root / "quant_investor/factors/governance_aquant_no_label_eval_v4_1.py",
            EXISTING_CODE_HASHES["governance_aquant_no_label_eval_v4_1.py"],
        ),
        "governance_no_label_diagnostic_v4_1.py": (
            root / "quant_investor/factors/governance_no_label_diagnostic_v4_1.py",
            EXISTING_CODE_HASHES["governance_no_label_diagnostic_v4_1.py"],
        ),
        "governance_operator_runtime_equivalence_v4_1.py": (
            root
            / "quant_investor/factors/governance_operator_runtime_equivalence_v4_1.py",
            EXISTING_CODE_HASHES[
                "governance_operator_runtime_equivalence_v4_1.py"
            ],
        ),
        "governance_private_bundle_io.py": (
            root / "quant_investor/factors/governance_private_bundle_io.py",
            EXISTING_CODE_HASHES["governance_private_bundle_io.py"],
        ),
        "governance_signal_computability_v4_1.py": (
            root / "quant_investor/factors/governance_signal_computability_v4_1.py",
            args.expected_contract_sha256,
        ),
    }
    rows = []
    import_union: set[str] = set()
    for binding_id in sorted(paths):
        path, expected = paths[binding_id]
        raw = _stable_bytes(path, expected_sha256=expected, private=False)
        import_union.update(_local_imports(raw, path))
        rows.append(
            {
                "binding_id": binding_id,
                "absolute_path": str(path),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "ast_sha256": _ast_sha(raw, path),
            }
        )
    if import_union != EXPECTED_LOCAL_IMPORTS or import_union & FORBIDDEN_CONTEXT_MODULES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "executed local import closure mismatch"
        )
    return rows


def _distribution_descriptor(distribution_name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    rows = []
    native = []
    record_sha = None
    for item in sorted(distribution.files or (), key=lambda value: str(value)):
        path = Path(distribution.locate_file(item))
        try:
            metadata = os.lstat(path)
        except FileNotFoundError as exc:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"runtime distribution file missing: {path}"
            ) from exc
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FactorV4_1SignalComputabilityRunnerError(
                f"runtime distribution member is not a regular file: {path}"
            )
        raw = path.read_bytes()
        row = {
            "relative_path": str(item),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        rows.append(row)
        if str(item).endswith((".so", ".dylib", ".pyd", ".dll")):
            native.append(row)
        if str(item).endswith(".dist-info/RECORD"):
            record_sha = row["sha256"]
    if not rows or record_sha is None:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"runtime distribution inventory incomplete: {distribution_name}"
        )
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "record_sha256": record_sha,
        "file_inventory_semantic_sha256": _semantic_sha(rows),
        "native_binary_count": len(native),
        "native_binary_inventory_semantic_sha256": _semantic_sha(native),
    }


def _runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    raw = executable.read_bytes()
    distributions = [
        _distribution_descriptor(name)
        for name in ("numpy", "pandas", "pyarrow", "scipy")
    ]
    payload = {
        "python": {
            "executable": str(executable),
            "executable_sha256": hashlib.sha256(raw).hexdigest(),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "distributions": distributions,
    }
    return {**payload, "runtime_semantic_sha256": _semantic_sha(payload)}


def _peak_rss_bytes(*, children: bool = False) -> int:
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    value = int(resource.getrusage(who).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _resource_limits() -> dict[str, Any]:
    return copy.deepcopy(contract.EXPECTED_RESOURCE_LIMITS)


def _git_environment() -> dict[str, str]:
    allowed = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "NO_PROXY": "*",
    }
    return allowed


def _git_command(root: Path, *args: str) -> list[str]:
    return [
        str(GIT_EXECUTABLE),
        "--no-replace-objects",
        "-C",
        str(root),
        "-c",
        "core.useReplaceRefs=false",
        "-c",
        "extensions.partialClone=",
        "-c",
        "remote.origin.promisor=false",
        "-c",
        "fetch.fsckObjects=true",
        "-c",
        "transfer.fsckObjects=true",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        *args,
    ]


def _run_git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        _git_command(root, *args),
        input=input_bytes,
        check=False,
        capture_output=True,
        env=_git_environment(),
    )
    if result.returncode != 0:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"sanitized Git command failed: {' '.join(args)}:{result.stderr[:500]!r}"
        )
    return result.stdout


def _git_identity(root: Path) -> dict[str, Any]:
    metadata = os.lstat(GIT_EXECUTABLE)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FactorV4_1SignalComputabilityRunnerError(
            "Git executable must be a regular non-symlink file"
        )
    raw = GIT_EXECUTABLE.read_bytes()
    version = _run_git(root, "--version").decode("utf-8").strip()
    return {
        "absolute_path": str(GIT_EXECUTABLE),
        "version": version,
        "executable_sha256": hashlib.sha256(raw).hexdigest(),
        "replacement_objects_disabled": True,
        "lazy_fetch_disabled": True,
        "network_protocols_disabled": ["ext", "file"],
        "alternates_environment_removed": True,
    }


def _parse_ls_tree(raw: bytes) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for record in (item for item in raw.split(b"\0") if item):
        try:
            head, path_raw = record.split(b"\t", 1)
            mode, object_type, oid, size = head.decode("ascii").split()
            path = path_raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise FactorV4_1SignalComputabilityRunnerError(
                "malformed NUL-delimited git ls-tree response"
            ) from exc
        if path in seen:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"duplicate Git tree path: {path}"
            )
        seen.add(path)
        if mode != "100644" or object_type != "blob":
            raise FactorV4_1SignalComputabilityRunnerError(
                f"Git tree contains unsupported mode/type: {path}"
            )
        try:
            size_int = int(size)
        except ValueError as exc:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"Git tree object size is invalid: {path}"
            ) from exc
        if size_int < 0 or size_int > MAX_GIT_BLOB_BYTES:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"Git tree object exceeds size limit: {path}"
            )
        rows.append(
            {
                "mode": mode,
                "type": object_type,
                "oid": oid,
                "size": size_int,
                "path": path,
            }
        )
    if not rows or rows != sorted(rows, key=lambda row: row["path"]):
        raise FactorV4_1SignalComputabilityRunnerError(
            "Git tree inventory must be non-empty and lexicographically ordered"
        )
    return rows


def _ls_tree(root: Path, tree_oid: str) -> list[dict[str, Any]]:
    return _parse_ls_tree(_run_git(root, "ls-tree", "-r", "-z", "-l", tree_oid))


def _read_exact(stream: Any, size: int) -> bytes:
    parts = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise FactorV4_1SignalComputabilityRunnerError(
                "truncated git cat-file batch object"
            )
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _stream_git_blobs(
    root: Path, entries: Sequence[Mapping[str, Any]]
) -> Iterator[tuple[dict[str, Any], bytes]]:
    process = subprocess.Popen(
        _git_command(root, "cat-file", "--batch"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise FactorV4_1SignalComputabilityRunnerError(
            "failed to open git cat-file batch pipes"
        )
    writer_error: list[BaseException] = []

    def write_requests() -> None:
        try:
            for row in entries:
                process.stdin.write((str(row["oid"]) + "\n").encode("ascii"))
            process.stdin.close()
        except BaseException as exc:  # pragma: no cover - defensive pipe failure
            writer_error.append(exc)

    writer = threading.Thread(target=write_requests, daemon=True)
    writer.start()
    try:
        for raw_row in entries:
            row = copy.deepcopy(dict(raw_row))
            header = process.stdout.readline(512)
            if not header.endswith(b"\n"):
                raise FactorV4_1SignalComputabilityRunnerError(
                    "malformed or oversized git cat-file batch header"
                )
            fields = header[:-1].decode("ascii").split()
            if fields != [row["oid"], "blob", str(row["size"])]:
                raise FactorV4_1SignalComputabilityRunnerError(
                    f"git cat-file object identity mismatch: {row['path']}"
                )
            data = _read_exact(process.stdout, int(row["size"]))
            if process.stdout.read(1) != b"\n":
                raise FactorV4_1SignalComputabilityRunnerError(
                    "git cat-file batch object terminator mismatch"
                )
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise FactorV4_1SignalComputabilityRunnerError(
                    f"Git-LFS pointer is not admissible data: {row['path']}"
                )
            yield row, data
    finally:
        writer.join(timeout=5)
        if writer.is_alive():
            process.kill()
            raise FactorV4_1SignalComputabilityRunnerError(
                "git cat-file request writer did not terminate"
            )
        stderr = process.stderr.read()
        return_code = process.wait(timeout=10)
        if writer_error or return_code != 0 or stderr:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"git cat-file batch failed: {writer_error!r}:{stderr[:500]!r}"
            )


def _direct_tree_oid(root: Path, commit: str, path: str) -> str:
    raw = _run_git(root, "ls-tree", "-d", commit, "--", path)
    try:
        head, found_path = raw.rstrip(b"\n").split(b"\t", 1)
        mode, object_type, oid = head.decode("ascii").split()
    except (ValueError, UnicodeDecodeError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"direct Git tree binding parse failed: {path}"
        ) from exc
    if mode != "040000" or object_type != "tree" or found_path.decode() != path:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"direct Git tree binding mismatch: {path}"
        )
    return oid


def _read_source_blob(root: Path, commit: str, path: str) -> tuple[bytes, dict[str, Any]]:
    raw = _run_git(root, "ls-tree", commit, "--", path)
    try:
        head, found_path = raw.rstrip(b"\n").split(b"\t", 1)
        mode, object_type, oid = head.decode("ascii").split()
    except (ValueError, UnicodeDecodeError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"source blob binding parse failed: {path}"
        ) from exc
    expected = EXPECTED_SOURCE_BLOBS[path]
    if (
        mode != "100644"
        or object_type != "blob"
        or found_path.decode() != path
        or oid != expected["oid"]
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            f"source blob Git identity mismatch: {path}"
        )
    data = _run_git(root, "cat-file", "blob", oid)
    if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected[
        "sha256"
    ]:
        raise FactorV4_1SignalComputabilityRunnerError(
            f"source blob byte identity mismatch: {path}"
        )
    return data, {
        "path": path,
        "mode": mode,
        "type": object_type,
        "oid": oid,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _extract_transform_code(source: bytes) -> tuple[Any, list[dict[str, Any]]]:
    if hashlib.sha256(source).hexdigest() != EXPECTED_SOURCE_BLOBS[
        MATRIX_DATASET_PATH
    ]["sha256"]:
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned MatrixDataset source SHA mismatch"
        )
    try:
        tree = ast.parse(source.decode("utf-8"), filename=MATRIX_DATASET_PATH)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned MatrixDataset AST parse failed"
        ) from exc
    constant_names = {"FINANCIAL_BASE_COLUMNS", "FINANCIAL_GROWTH_COLUMNS"}
    constants: list[ast.stmt] = []
    rows: list[dict[str, Any]] = []
    for name in sorted(constant_names):
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        if len(matches) != 1:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"pinned transformation constant inventory mismatch: {name}"
            )
        node = matches[0]
        constants.append(copy.deepcopy(node))
        rows.append(
            {
                "kind": "constant",
                "name": name,
                "ast_sha256": hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode("utf-8")
                ).hexdigest(),
            }
        )
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FactorMatrixDataset"
    ]
    if len(classes) != 1:
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned FactorMatrixDataset class inventory mismatch"
        )
    source_class = classes[0]
    method_names = (
        "_pivot",
        "_align_symbol_financials",
        "_derive_financial_columns",
        "_build_valuation_matrices",
        "_numeric_col",
        "_safe_divide",
    )
    methods: list[ast.stmt] = []
    for name in method_names:
        matches = [
            node
            for node in source_class.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(matches) != 1:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"pinned transformation method inventory mismatch: {name}"
            )
        node = matches[0]
        methods.append(copy.deepcopy(node))
        rows.append(
            {
                "kind": "method",
                "name": name,
                "ast_sha256": hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode("utf-8")
                ).hexdigest(),
            }
        )
    readers = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_read_bars"
    ]
    if len(readers) != 1 or len(readers[0].body) != 15:
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned bar-reader AST inventory mismatch"
        )
    reader = readers[0]
    adapter = ast.FunctionDef(
        name="_normalize_bar_frame",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="raw"),
                ast.arg(arg="start"),
                ast.arg(arg="end"),
                ast.arg(arg="symbols"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *copy.deepcopy(reader.body[6:14]),
            ast.Return(value=ast.Name(id="raw", ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    ast.fix_missing_locations(adapter)
    rows.append(
        {
            "kind": "adapter",
            "name": adapter.name,
            "ast_sha256": hashlib.sha256(
                ast.dump(adapter, include_attributes=False).encode("utf-8")
            ).hexdigest(),
        }
    )
    expected_rows = [
        {"kind": kind, "name": name, "ast_sha256": digest}
        for kind, name, digest in EXPECTED_TRANSFORMATION_AST_ROWS
    ]
    if rows != expected_rows or _semantic_sha(rows) != (
        contract.EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned transformation AST manifest mismatch"
        )
    restricted_class = ast.ClassDef(
        name="FactorMatrixDataset",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[*constants, restricted_class, adapter], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        code = compile(module, MATRIX_DATASET_PATH, "exec", dont_inherit=True)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise FactorV4_1SignalComputabilityRunnerError(
            "restricted pinned transformation compile failed"
        ) from exc
    return code, rows


def _arrow_stream_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _sha_lines(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _normalize_schema(table: pa.Table) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in table.schema
    ]


def _load_aquant_bars(
    root: Path, tree_oid: str
) -> tuple[pd.DataFrame, bytes, dict[str, Any], list[dict[str, Any]]]:
    entries = _ls_tree(root, tree_oid)
    if _semantic_sha(entries) != contract.EXPECTED_BARS_FULL_INVENTORY_SHA256:
        raise FactorV4_1SignalComputabilityRunnerError(
            "full A_quant bars tree inventory SHA mismatch"
        )
    if sum(int(row["size"]) for row in entries) > MAX_FULL_GIT_INPUT_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "full A_quant bars tree exceeds resource limit"
        )
    selected = []
    for row in entries:
        match = re.fullmatch(
            r"year=(\d{4})/month=(\d{2})/bars\.(parquet|lock)", row["path"]
        )
        if match is None:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"unexpected A_quant bars tree path: {row['path']}"
            )
        if match.group(3) == "parquet" and (
            "2021-06" <= f"{match.group(1)}-{match.group(2)}" <= "2026-06"
        ):
            selected.append(row)
    if _semantic_sha(selected) != contract.EXPECTED_BARS_SELECTED_INVENTORY_SHA256:
        raise FactorV4_1SignalComputabilityRunnerError(
            "selected A_quant bars inventory SHA mismatch"
        )
    content_rows = []
    selected_tables: list[pa.Table] = []
    for row, data in _stream_git_blobs(root, entries):
        digest = hashlib.sha256(data).hexdigest()
        content_rows.append(
            {
                "mode": row["mode"],
                "type": row["type"],
                "oid": row["oid"],
                "size": row["size"],
                "path": row["path"],
                "sha256": digest,
            }
        )
        if row["path"].endswith(".lock"):
            if data:
                raise FactorV4_1SignalComputabilityRunnerError(
                    f"A_quant bars lock is not empty: {row['path']}"
                )
            continue
        try:
            schema = pq.read_schema(io.BytesIO(data))
        except Exception as exc:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"malformed A_quant bars Parquet: {row['path']}"
            ) from exc
        required = {"symbol", "trade_date", "turnover_rate", "total_mv"}
        if not required.issubset(schema.names):
            raise FactorV4_1SignalComputabilityRunnerError(
                f"A_quant bars schema is missing the exact projection: {row['path']}"
            )
        if row in selected:
            selected_tables.append(
                pq.read_table(
                    io.BytesIO(data),
                    columns=["symbol", "trade_date", "turnover_rate", "total_mv"],
                )
            )
    table = pa.concat_tables(selected_tables, promote_options="default")
    dates = pc.strftime(pc.cast(table["trade_date"], pa.timestamp("ns")), format="%Y%m%d")
    keep = pc.and_(
        pc.greater_equal(dates, pa.scalar("20210625")),
        pc.less_equal(dates, pa.scalar("20260626")),
    )
    table = table.filter(keep)
    if table.nbytes != contract.EXPECTED_BARS_ACCOUNTING[
        "projected_arrow_buffer_bytes"
    ]:
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant projected bars Arrow byte count mismatch"
        )
    ipc_bytes = _arrow_stream_bytes(table)
    if (
        len(ipc_bytes) != contract.EXPECTED_BARS_ACCOUNTING["projected_ipc_bytes"]
        or hashlib.sha256(ipc_bytes).hexdigest()
        != contract.EXPECTED_BARS_IPC_SHA256
        or len(ipc_bytes) > MAX_MESSAGE_BYTES
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant projected bars IPC identity or limit mismatch"
        )
    frame = table.to_pandas()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
    duplicate_count = int(frame.duplicated(["trade_date", "symbol"], keep=False).sum())
    dates_text = sorted(frame["trade_date"].dt.strftime("%Y%m%d").unique().tolist())
    accounting = {
        "tree_oid": tree_oid,
        "full_inventory_sha256": _semantic_sha(entries),
        "full_content_manifest_sha256": _semantic_sha(content_rows),
        "selected_inventory_sha256": _semantic_sha(selected),
        "selected_content_manifest_sha256": _semantic_sha(
            [row for row in content_rows if any(row["path"] == item["path"] for item in selected)]
        ),
        "projected_ipc_sha256": hashlib.sha256(ipc_bytes).hexdigest(),
        "full_entry_count": len(entries),
        "full_parquet_count": sum(row["path"].endswith(".parquet") for row in entries),
        "full_lock_count": sum(row["path"].endswith(".lock") for row in entries),
        "full_byte_count": sum(int(row["size"]) for row in entries),
        "selected_parquet_count": len(selected),
        "selected_byte_count": sum(int(row["size"]) for row in selected),
        "selected_row_count": len(frame),
        "selected_date_count": len(dates_text),
        "selected_symbol_count": int(frame["symbol"].nunique()),
        "turnover_non_null_count": int(frame["turnover_rate"].notna().sum()),
        "market_cap_non_null_count": int(frame["total_mv"].notna().sum()),
        "duplicate_date_symbol_count": duplicate_count,
        "min_observed_bar_date": frame["trade_date"].min().strftime("%Y-%m-%d"),
        "max_observed_bar_date": frame["trade_date"].max().strftime("%Y-%m-%d"),
        "projected_arrow_buffer_bytes": table.nbytes,
        "projected_ipc_bytes": len(ipc_bytes),
    }
    for key, expected in contract.EXPECTED_BARS_ACCOUNTING.items():
        if accounting.get(key) != expected:
            raise TrustedComputabilityBlocker(
                f"bars_accounting_drift:{key}:actual={accounting.get(key)}:expected={expected}"
            )
    return frame, ipc_bytes, accounting, entries


def _load_aquant_financials(
    root: Path, tree_oid: str
) -> tuple[pd.DataFrame, bytes, dict[str, Any], list[dict[str, Any]]]:
    entries = _ls_tree(root, tree_oid)
    if _semantic_sha(entries) != contract.EXPECTED_FINANCIAL_INVENTORY_SHA256:
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant financial tree inventory SHA mismatch"
        )
    if sum(int(row["size"]) for row in entries) > MAX_FINANCIAL_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant financial tree exceeds resource limit"
        )
    content_rows = []
    tables: list[pa.Table] = []
    schema_counts: dict[str, int] = {}
    for row, data in _stream_git_blobs(root, entries):
        if re.fullmatch(r"symbol=[^/]+/financials\.parquet", row["path"]) is None:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"unexpected A_quant financial path: {row['path']}"
            )
        content_rows.append(
            {
                "mode": row["mode"],
                "type": row["type"],
                "oid": row["oid"],
                "size": row["size"],
                "path": row["path"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        try:
            table = pq.read_table(io.BytesIO(data))
        except Exception as exc:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"malformed A_quant financial Parquet: {row['path']}"
            ) from exc
        if tuple(table.column_names) != LOGICAL_FINANCIAL_COLUMNS:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"A_quant financial logical schema mismatch: {row['path']}"
            )
        schema_key = _canonical_json_bytes(_normalize_schema(table)).decode("utf-8")
        schema_counts[schema_key] = schema_counts.get(schema_key, 0) + 1
        tables.append(table)
    schema_manifest = [
        {"schema": json.loads(key), "file_count": count}
        for key, count in sorted(schema_counts.items())
    ]
    if _semantic_sha(schema_manifest) != contract.EXPECTED_FINANCIAL_SCHEMA_MANIFEST_SHA256:
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant financial physical schema manifest mismatch"
        )
    table = pa.concat_tables(tables, promote_options="default")
    if table.nbytes != contract.EXPECTED_FINANCIAL_ACCOUNTING[
        "logical_arrow_buffer_bytes"
    ]:
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant financial logical Arrow byte count mismatch"
        )
    ipc_bytes = _arrow_stream_bytes(table)
    if (
        len(ipc_bytes) != contract.EXPECTED_FINANCIAL_ACCOUNTING["logical_ipc_bytes"]
        or hashlib.sha256(ipc_bytes).hexdigest()
        != contract.EXPECTED_FINANCIAL_IPC_SHA256
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "A_quant financial IPC identity mismatch"
        )
    frame = table.to_pandas()
    path_symbols = [row["path"].split("/", 1)[0].split("=", 1)[1] for row in entries]
    mismatches = 0
    offset = 0
    for path_symbol, table_part in zip(path_symbols, tables, strict=True):
        rows = table_part.num_rows
        if not (frame.iloc[offset : offset + rows]["symbol"].astype(str) == path_symbol).all():
            mismatches += 1
        offset += rows
    exact_duplicate_excess = int(frame.duplicated(keep="first").sum())
    report_duplicate_excess = int(
        frame.duplicated(["symbol", "report_period"], keep="last").sum()
    )
    ordered = frame.copy()
    ordered["availability_date"] = pd.to_datetime(
        ordered["availability_date"], errors="raise"
    )
    ordered["report_period"] = pd.to_datetime(ordered["report_period"], errors="raise")
    selected = ordered.sort_values(
        ["symbol", "availability_date", "report_period"], kind="mergesort"
    ).drop_duplicates(["symbol", "report_period"], keep="last")
    accounting = {
        "tree_oid": tree_oid,
        "inventory_sha256": _semantic_sha(entries),
        "content_manifest_sha256": _semantic_sha(content_rows),
        "physical_schema_manifest_sha256": _semantic_sha(schema_manifest),
        "logical_ipc_sha256": hashlib.sha256(ipc_bytes).hexdigest(),
        "blob_count": len(entries),
        "byte_count": sum(int(row["size"]) for row in entries),
        "row_count": len(frame),
        "symbol_count": int(frame["symbol"].nunique()),
        "logical_column_count": len(frame.columns),
        "physical_schema_variant_count": len(schema_manifest),
        "path_symbol_mismatch_count": mismatches,
        "exact_duplicate_excess": exact_duplicate_excess,
        "duplicate_report_period_excess": report_duplicate_excess,
        "post_report_period_selection_row_count": len(selected),
        "max_availability_date": pd.to_datetime(frame["availability_date"]).max().strftime(
            "%Y-%m-%d"
        ),
        "logical_arrow_buffer_bytes": table.nbytes,
        "logical_ipc_bytes": len(ipc_bytes),
    }
    for key, expected in contract.EXPECTED_FINANCIAL_ACCOUNTING.items():
        if accounting.get(key) != expected:
            raise TrustedComputabilityBlocker(
                f"financial_accounting_drift:{key}:actual={accounting.get(key)}:expected={expected}"
            )
    return frame, ipc_bytes, accounting, entries


def _inventory_table(table_root: Path) -> tuple[list[dict[str, Any]], str]:
    if table_root.is_symlink() or not table_root.is_dir():
        raise FactorV4_1SignalComputabilityRunnerError(
            "strict-market table root must be a regular directory"
        )
    inventory = []
    for path in sorted(table_root.rglob("*")):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise FactorV4_1SignalComputabilityRunnerError(
                f"strict-market inventory contains a symlink: {path}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            continue
        relative = path.relative_to(table_root)
        raw = path.read_bytes()
        inventory.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "hard_link_count": int(metadata.st_nlink),
                "dataset_member": bool(
                    path.suffix == ".parquet"
                    and all(not part.startswith((".", "_")) for part in relative.parts)
                ),
            }
        )
    if not any(row["dataset_member"] for row in inventory):
        raise FactorV4_1SignalComputabilityRunnerError(
            "strict-market table inventory has no dataset members"
        )
    digest = hashlib.sha256(_canonical_json_bytes(inventory) + b"\n").hexdigest()
    if digest != EXPECTED_MARKET_INVENTORY_SHA256:
        raise FactorV4_1SignalComputabilityRunnerError(
            "strict-market table inventory SHA mismatch"
        )
    return inventory, digest


def _derive_vwap(amount: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    amount_values = amount.to_numpy(dtype=np.float64, copy=True)
    volume_values = volume.to_numpy(dtype=np.float64, copy=True)
    result = np.full(amount_values.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(amount_values)
        & np.isfinite(volume_values)
        & (volume_values != 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result[valid] = amount_values[valid] * 10.0 / volume_values[valid]
    result[~np.isfinite(result)] = np.nan
    return pd.DataFrame(result, index=amount.index, columns=amount.columns)


def _load_strict_market_and_mask(
    *,
    table_root: Path,
    pit_path: Path,
    expected_pit_sha256: str,
    pit_manifest_path: Path,
    expected_pit_manifest_sha256: str,
    components_path: Path,
    expected_components_sha256: str,
    no_label_diagnostic: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    inventory, inventory_sha = _inventory_table(table_root)
    paths = [
        str(table_root / row["relative_path"])
        for row in inventory
        if row["dataset_member"] is True
    ]
    dataset = ds.dataset(paths, format="parquet")
    if not set(MARKET_COLUMNS).issubset(dataset.schema.names):
        raise FactorV4_1SignalComputabilityRunnerError(
            "strict-market dataset is missing exact market columns"
        )
    table = dataset.to_table(
        columns=list(MARKET_COLUMNS),
        filter=(ds.field("trade_date") >= "20210625")
        & (ds.field("trade_date") <= "20260717"),
    )
    raw = table.to_pandas()
    if raw.empty or raw.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1SignalComputabilityRunnerError(
            "strict-market projection is empty or contains duplicates"
        )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    sessions = pd.DatetimeIndex(sorted(raw["trade_date"].unique()), name="trade_date")
    if (
        len(sessions) != contract.EXPECTED_SESSION_COUNT
        or sessions[0].strftime("%Y-%m-%d") != EXPECTED_ANALYSIS_START
        or sessions[-1].strftime("%Y-%m-%d") != EXPECTED_CUTOFF
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "strict-market session axes mismatch"
        )
    pit_raw = _stable_bytes(pit_path, expected_sha256=expected_pit_sha256, private=False)
    manifest_raw = _stable_bytes(
        pit_manifest_path,
        expected_sha256=expected_pit_manifest_sha256,
        private=False,
    )
    pit_manifest = _strict_json(manifest_raw, str(pit_manifest_path))
    if (
        pit_manifest.get("row_count") != EXPECTED_PIT_COUNT
        or pit_manifest.get("canonical_path") != str(pit_path)
        or pit_manifest.get("canonical_sha256") != hashlib.sha256(pit_raw).hexdigest()
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "PIT manifest does not bind the explicit membership file"
        )
    pit = pd.read_parquet(io.BytesIO(pit_raw))
    required_pit = {"symbol", "effective_from", "effective_to"}
    if len(pit) != EXPECTED_PIT_COUNT or not required_pit.issubset(pit.columns):
        raise FactorV4_1SignalComputabilityRunnerError("PIT membership contract mismatch")
    symbols = pit["symbol"].astype(str).tolist()
    if symbols != sorted(set(symbols)):
        raise FactorV4_1SignalComputabilityRunnerError(
            "PIT symbol axis must be sorted and distinct"
        )
    components_raw = _stable_bytes(
        components_path, expected_sha256=expected_components_sha256, private=False
    )
    components = _strict_json(components_raw, str(components_path)).get("full_a")
    if (
        not isinstance(components, list)
        or len(components) != EXPECTED_COMPONENT_COUNT
        or components != sorted(set(components))
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "component file must contain exact sorted full_a symbols"
        )
    session_text = np.asarray(sessions.strftime("%Y%m%d"), dtype="U8")
    starts = pit["effective_from"].astype(str).str.replace("-", "", regex=False).to_numpy()
    ends = pit["effective_to"].fillna("").astype(str).str.replace("-", "", regex=False).to_numpy()
    mask_values = np.zeros((len(sessions), len(symbols)), dtype=bool)
    for index in range(len(symbols)):
        mask_values[:, index] = (session_text >= starts[index]) & (
            (ends[index] == "") | (session_text < ends[index])
        )
    component_set = set(components)
    mask_values[-1, :] &= np.fromiter(
        (symbol in component_set for symbol in symbols), dtype=bool, count=len(symbols)
    )
    mask = pd.DataFrame(mask_values, index=sessions, columns=symbols, dtype=bool)
    expected_mask = no_label_diagnostic["session_scope_binding"]["eligibility_matrix"]
    actual_mask = evaluator.matrix_hash_descriptor_v4_1(mask.astype(float))
    if actual_mask != expected_mask:
        raise FactorV4_1SignalComputabilityRunnerError(
            "reproduced PIT mask differs from accepted no-label envelope"
        )
    matrices: dict[str, pd.DataFrame] = {}
    mapping = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "vol",
        "amount": "amount",
    }
    for field, source_column in mapping.items():
        matrix = raw.pivot(index="trade_date", columns="ts_code", values=source_column)
        matrices[field] = matrix.reindex(index=sessions, columns=symbols).astype(float).where(mask)
    matrices["vwap"] = _derive_vwap(matrices["amount"], matrices["volume"]).where(mask)
    expected_bindings = {
        row["binding_id"].split(":", 1)[1]: row
        for row in no_label_diagnostic["market_matrix_bindings"]
    }
    for field in ("amount", "close", "high", "low", "open", "volume", "vwap"):
        descriptor = evaluator.matrix_hash_descriptor_v4_1(matrices[field])
        expected = expected_bindings[field]
        if (
            descriptor["matrix_sha256"] != expected["byte_sha256"]
            or _semantic_sha(descriptor) != expected["semantic_sha256"]
        ):
            raise FactorV4_1SignalComputabilityRunnerError(
                f"accepted myQuant market matrix drift: {field}"
            )
    binding = {
        "table_inventory_sha256": inventory_sha,
        "pit_membership_sha256": hashlib.sha256(pit_raw).hexdigest(),
        "pit_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "components_sha256": hashlib.sha256(components_raw).hexdigest(),
        "eligibility_matrix": actual_mask,
    }
    return matrices, mask, binding, {"pit_manifest": pit_manifest}


def _matrix_counts(matrix: pd.DataFrame, mask: pd.DataFrame) -> dict[str, int]:
    values = matrix.to_numpy(dtype=np.float64, copy=False)
    eligible = values[mask.to_numpy(dtype=bool, copy=False)]
    outside = values[~mask.to_numpy(dtype=bool, copy=False)]
    return {
        "finite_count": int(np.isfinite(eligible).sum()),
        "nan_count": int(np.isnan(eligible).sum()),
        "positive_inf_count": int(np.isposinf(eligible).sum()),
        "negative_inf_count": int(np.isneginf(eligible).sum()),
        "outside_mask_non_nan_count": int((~np.isnan(outside)).sum()),
    }


def _primitive_row(
    field: str, source: str, matrix: pd.DataFrame, mask: pd.DataFrame
) -> dict[str, Any]:
    return {
        "field": field,
        "source": source,
        "matrix": evaluator.matrix_hash_descriptor_v4_1(matrix),
        **_matrix_counts(matrix, mask),
    }


def _worker_matrix_descriptor(value: pd.DataFrame) -> dict[str, Any]:
    dates = [item.isoformat() for item in value.index]
    symbols = list(value.columns)
    array = np.asarray(value.to_numpy(dtype=np.float64, copy=True), dtype="<f8", order="C")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype="<f8")
    bits = array.view("<u8")
    bits[np.isnan(array)] = np.uint64(0x7FF8000000000000)
    header = {
        "contract": evaluator.MATRIX_HASH_CONTRACT_VERSION,
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "dtype": "float64-little-endian",
        "date_axis_sha256": _semantic_sha(
            {"contract": "factor-no-label-axis.v1", "axis": "date", "values": dates}
        ),
        "symbol_axis_sha256": _semantic_sha(
            {
                "contract": "factor-no-label-axis.v1",
                "axis": "symbol",
                "values": symbols,
            }
        ),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return {**header, "matrix_sha256": digest.hexdigest()}


def _worker_primitive_row(
    field: str, matrix: pd.DataFrame, mask: pd.DataFrame
) -> dict[str, Any]:
    row = _primitive_row(field, "restricted_pinned_child", matrix, mask)
    row["matrix"] = _worker_matrix_descriptor(matrix)
    return row


def _pack_mask(mask: pd.DataFrame) -> str:
    raw = np.packbits(mask.to_numpy(dtype=np.uint8, copy=False).reshape(-1), bitorder="little")
    return base64.b64encode(raw.tobytes()).decode("ascii")


def _unpack_mask(metadata: Mapping[str, Any]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(metadata["dates"], errors="raise"), name="trade_date")
    symbols = pd.Index(metadata["symbols"], dtype=object)
    shape = (len(dates), len(symbols))
    packed = base64.b64decode(metadata["mask_base64"], validate=True)
    expected_bytes = (shape[0] * shape[1] + 7) // 8
    if len(packed) != expected_bytes:
        raise FactorV4_1SignalComputabilityRunnerError(
            "child PIT mask packed size mismatch"
        )
    values = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder="little"
    )[: shape[0] * shape[1]].reshape(shape)
    return pd.DataFrame(values.astype(bool), index=dates, columns=symbols)


def _encode_worker_message(
    metadata: Mapping[str, Any], payloads: Sequence[bytes]
) -> bytes:
    normalized = copy.deepcopy(dict(metadata))
    normalized["payload_sizes"] = [len(item) for item in payloads]
    metadata_raw = _canonical_json_bytes(normalized)
    if len(metadata_raw) > MAX_FRAME_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker metadata exceeds frame limit"
        )
    message = struct.pack(">Q", len(metadata_raw)) + metadata_raw
    for payload in payloads:
        message += struct.pack(">Q", len(payload)) + payload
    if len(message) > MAX_MESSAGE_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker aggregate message exceeds limit"
        )
    return message


def _read_worker_message(stream: Any) -> tuple[dict[str, Any], list[bytes]]:
    header = _read_exact(stream, 8)
    metadata_size = struct.unpack(">Q", header)[0]
    if metadata_size <= 0 or metadata_size > MAX_FRAME_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker metadata frame size is invalid"
        )
    metadata = _strict_json(_read_exact(stream, metadata_size), "worker metadata")
    sizes = metadata.get("payload_sizes")
    if not isinstance(sizes, list) or not sizes:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker payload inventory is invalid"
        )
    payloads = []
    total = 8 + metadata_size
    for expected in sizes:
        if type(expected) is not int or expected < 0:
            raise FactorV4_1SignalComputabilityRunnerError(
                "worker payload size is invalid"
            )
        observed = struct.unpack(">Q", _read_exact(stream, 8))[0]
        if observed != expected:
            raise FactorV4_1SignalComputabilityRunnerError(
                "worker payload frame length mismatch"
            )
        payloads.append(_read_exact(stream, expected))
        total += 8 + expected
    if total > MAX_MESSAGE_BYTES or stream.read(1) not in (b"", None):
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker message exceeds limit or has trailing bytes"
        )
    return metadata, payloads


def _execute_restricted_transform(code: Any) -> tuple[dict[str, Any], int]:
    state = {"exec_count": 0}

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "exec":
            if len(args) != 1 or args[0] is not code or state["exec_count"] != 0:
                raise RuntimeError("unverified exec audit event")
            state["exec_count"] += 1
            return
        if event == "compile" or event == "open" or event == "os.system":
            raise RuntimeError(f"forbidden child audit event: {event}")
        if event.startswith("socket.") or event.startswith("subprocess."):
            raise RuntimeError(f"forbidden child audit event: {event}")

    namespace = {
        "__builtins__": {
            "__build_class__": builtins.__build_class__,
            "bool": bool,
            "dict": dict,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "next": next,
            "object": object,
            "str": str,
            "tuple": tuple,
        },
        "__name__": "pinned_matrix_dataset_restricted",
        "np": np,
        "pd": pd,
    }
    sys.addaudithook(audit)
    exec(code, namespace)
    if state["exec_count"] != 1:
        raise RuntimeError("verified child exec audit count mismatch")
    return namespace, state["exec_count"]


def _prewarm_worker_runtime() -> None:
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="date")
    frame = pd.DataFrame(
        {
            "date": [dates[0], dates[1]],
            "symbol": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )
    pivoted = frame.pivot_table(
        index="date", columns="symbol", values="value", aggfunc="last"
    )
    pivoted.reindex(index=dates, columns=["A"]).astype(float).where(
        pd.DataFrame(True, index=dates, columns=["A"])
    )
    financial = pd.DataFrame(
        {
            "availability_date": dates,
            "report_period": dates,
            "net_profit": [1.0, 2.0],
            "total_assets": [2.0, 3.0],
        }
    )
    financial.sort_values(["availability_date", "report_period"]).drop_duplicates(
        ["report_period"], keep="last"
    )
    financial["net_profit"].pct_change(periods=1)
    financial.groupby("net_profit", sort=False)
    series = pd.Series([1.0, np.nan], index=dates)
    series.reindex(dates.union(pd.DatetimeIndex(["2020-01-03"]))).sort_index().ffill()
    series.replace(0, np.nan)


def _child_bar_descriptors(
    *, metadata: Mapping[str, Any], table: pa.Table, code: Any
) -> dict[str, Any]:
    raw = table.to_pandas()
    mask = _unpack_mask(metadata)
    namespace, exec_count = _execute_restricted_transform(code)
    normalized = namespace["_normalize_bar_frame"](
        raw, EXPECTED_ANALYSIS_START, "2026-06-26", None
    )
    dataset_class = namespace["FactorMatrixDataset"]
    instance = dataset_class.__new__(dataset_class)
    matrices = {}
    for field, source_field in (
        ("turnover_rate", "turnover_rate"),
        ("market_cap", "total_mv"),
    ):
        matrix = instance._pivot(normalized, source_field)
        matrix = matrix.reindex(index=mask.index, columns=mask.columns).astype(float).where(mask)
        matrices[field] = matrix
    return {
        "operation": "bars",
        "exec_event_count": exec_count,
        "rows": [
            _worker_primitive_row(field, matrices[field], mask)
            for field in ("market_cap", "turnover_rate")
        ],
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _child_financial_descriptors(
    *,
    metadata: Mapping[str, Any],
    table: pa.Table,
    market_cap_raw: bytes,
    code: Any,
) -> dict[str, Any]:
    financials = table.to_pandas()
    mask = _unpack_mask(metadata)
    expected_market_bytes = mask.shape[0] * mask.shape[1] * 8
    if len(market_cap_raw) != expected_market_bytes:
        raise FactorV4_1SignalComputabilityRunnerError(
            "child market-cap matrix byte length mismatch"
        )
    market_cap_values = np.frombuffer(market_cap_raw, dtype="<f8").reshape(mask.shape)
    market_cap = pd.DataFrame(
        market_cap_values.copy(), index=mask.index, columns=mask.columns
    ).where(mask)
    namespace, exec_count = _execute_restricted_transform(code)
    dataset_class = namespace["FactorMatrixDataset"]
    instance = dataset_class.__new__(dataset_class)
    fields = (
        "fin_debt_to_assets",
        "fin_free_cashflow",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_roa",
        "fin_roe",
    )
    matrices = {
        field: pd.DataFrame(np.nan, index=mask.index, columns=mask.columns, dtype=float)
        for field in fields
    }
    target_symbols = set(mask.columns)
    for symbol, frame in financials.groupby("symbol", sort=False):
        symbol_text = str(symbol)
        if symbol_text not in target_symbols:
            continue
        aligned = instance._align_symbol_financials(frame, mask.index)
        for field in fields:
            if field in aligned:
                matrices[field].loc[:, symbol_text] = aligned[field].to_numpy(dtype=float)
    for field in fields:
        matrices[field] = matrices[field].where(mask)
    valuation_inputs = {**matrices, "market_cap": market_cap}
    valuation = instance._build_valuation_matrices(valuation_inputs)
    if "fcf_to_price" not in valuation:
        raise FactorV4_1SignalComputabilityRunnerError(
            "restricted pinned valuation did not produce fcf_to_price"
        )
    matrices["fcf_to_price"] = valuation["fcf_to_price"].where(mask)
    return {
        "operation": "financials",
        "exec_event_count": exec_count,
        "rows": [
            _worker_primitive_row(field, matrices[field], mask)
            for field in sorted(matrices)
        ],
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _worker_main() -> int:
    metadata, payloads = _read_worker_message(sys.stdin.buffer)
    source = base64.b64decode(metadata.get("matrix_source_base64", ""), validate=True)
    _prewarm_worker_runtime()
    code, rows = _extract_transform_code(source)
    if metadata.get("ast_manifest_sha256") != _semantic_sha(rows):
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker transformation AST manifest binding mismatch"
        )
    operation = metadata.get("operation")
    if operation == "bars" and len(payloads) == 1:
        table = pa_ipc.open_stream(payloads[0]).read_all()
        result = _child_bar_descriptors(metadata=metadata, table=table, code=code)
    elif operation == "financials" and len(payloads) == 2:
        table = pa_ipc.open_stream(payloads[0]).read_all()
        result = _child_financial_descriptors(
            metadata=metadata,
            table=table,
            market_cap_raw=payloads[1],
            code=code,
        )
    else:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker operation or payload inventory mismatch"
        )
    encoded = _canonical_json_bytes(result) + b"\n"
    if len(encoded) > MAX_CHILD_OUTPUT_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "worker output exceeds limit"
        )
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


def _child_preexec() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (CHILD_NOFILE, CHILD_NOFILE))
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CHILD_SECONDS, MAX_CHILD_SECONDS))
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        resource.setrlimit(
            resource.RLIMIT_AS, (CHILD_ADDRESS_SPACE_BYTES, CHILD_ADDRESS_SPACE_BYTES)
        )
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_DATA"):
        resource.setrlimit(resource.RLIMIT_DATA, (CHILD_DATA_BYTES, CHILD_DATA_BYTES))
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_RSS"):
        resource.setrlimit(resource.RLIMIT_RSS, (MAX_CHILD_RSS_BYTES, MAX_CHILD_RSS_BYTES))


def _allowed_child_stderr(raw: bytes) -> bool:
    if not raw:
        return True
    lines = [line for line in raw.decode("utf-8", errors="replace").splitlines() if line]
    return bool(lines) and all(
        "arrow/cpp/src/arrow/util/cpu_info.cc" in line
        and "Operation not permitted" in line
        for line in lines
    )


def _run_restricted_child(
    *,
    operation: str,
    matrix_source: bytes,
    mask: pd.DataFrame,
    payloads: Sequence[bytes],
) -> dict[str, Any]:
    metadata = {
        "schema_version": "factor-v4.1-restricted-transform-worker.v1",
        "operation": operation,
        "matrix_source_base64": base64.b64encode(matrix_source).decode("ascii"),
        "ast_manifest_sha256": contract.EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256,
        "dates": [item.isoformat() for item in mask.index],
        "symbols": list(mask.columns),
        "mask_base64": _pack_mask(mask),
    }
    message = _encode_worker_message(metadata, payloads)
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
            "NO_PROXY": "*",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
        },
        preexec_fn=_child_preexec,
    )
    try:
        stdout, stderr = process.communicate(input=message, timeout=MAX_CHILD_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise FactorV4_1SignalComputabilityRunnerError(
            "restricted transformation child timed out"
        ) from exc
    if (
        process.returncode != 0
        or len(stdout) > MAX_CHILD_OUTPUT_BYTES
        or not _allowed_child_stderr(stderr)
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            f"restricted transformation child failed:{process.returncode}:"
            f"stdout={stdout[:5000]!r}:stderr={stderr[:5000]!r}"
        )
    result = _strict_json(stdout, f"{operation} child output")
    if (
        result.get("operation") != operation
        or result.get("exec_event_count") != 1
        or type(result.get("peak_rss_bytes")) is not int
        or result["peak_rss_bytes"] > MAX_CHILD_RSS_BYTES
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "restricted transformation child result invariant mismatch"
        )
    return result


def _parent_bar_matrices(
    raw: pd.DataFrame, mask: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    frame = raw.copy()
    date_col = next(
        (column for column in frame.columns if column.lower() in ("date", "trade_date", "datetime")),
        None,
    )
    if date_col and date_col != "date":
        frame = frame.rename(columns={date_col: "date"})
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    symbol_col = next(
        (column for column in frame.columns if column.lower() in ("symbol", "code", "ticker")),
        None,
    )
    if symbol_col and symbol_col != "symbol":
        frame = frame.rename(columns={symbol_col: "symbol"})
    frame = frame[frame["date"] >= pd.Timestamp(EXPECTED_ANALYSIS_START)]
    frame = frame[frame["date"] <= pd.Timestamp("2026-06-26")]
    result = {}
    for field, source_field in (
        ("turnover_rate", "turnover_rate"),
        ("market_cap", "total_mv"),
    ):
        matrix = frame.pivot_table(
            index="date", columns="symbol", values=source_field, aggfunc="last"
        )
        result[field] = matrix.reindex(
            index=mask.index, columns=mask.columns
        ).astype(float).where(mask)
    return result


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _parent_align_financials(
    financials: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    frame = financials.copy()
    if "availability_date" not in frame.columns:
        if "announce_date" in frame.columns:
            frame["availability_date"] = frame["announce_date"]
        elif "report_period" in frame.columns:
            frame["availability_date"] = pd.to_datetime(frame["report_period"]) + pd.Timedelta(
                days=45
            )
        else:
            return {}
    for column in ("availability_date", "announce_date", "report_period"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    sort_columns = ["availability_date"]
    if "report_period" in frame.columns:
        sort_columns.append("report_period")
    frame = frame[frame["availability_date"].notna()].sort_values(sort_columns)
    if frame.empty:
        return {}
    if "report_period" in frame.columns:
        frame = frame.drop_duplicates(subset=["report_period"], keep="last")
    else:
        frame = frame.drop_duplicates(subset=["availability_date"], keep="last")
    growth_columns = {
        "revenue": "fin_revenue_yoy",
        "operating_profit": "fin_operating_profit_yoy",
        "net_profit": "fin_net_profit_yoy",
        "eps": "fin_eps_yoy",
    }
    for source_column, target_column in growth_columns.items():
        if source_column in frame.columns and "report_period" in frame.columns:
            frame[target_column] = pd.to_numeric(
                frame[source_column], errors="coerce"
            ).pct_change(periods=4)
    revenue = _numeric_series(frame, "revenue")
    net_profit = _numeric_series(frame, "net_profit")
    total_assets = _numeric_series(frame, "total_assets")
    total_equity = _numeric_series(frame, "total_equity")
    total_debt = _numeric_series(frame, "total_debt")
    operating_cashflow = _numeric_series(frame, "operating_cashflow")
    free_cashflow = _numeric_series(frame, "free_cashflow")
    derived = {
        "fin_profit_margin": _safe_divide_series(net_profit, revenue),
        "fin_ocf_to_profit": _safe_divide_series(operating_cashflow, net_profit),
        "fin_fcf_to_profit": _safe_divide_series(free_cashflow, net_profit),
        "fin_asset_turnover": _safe_divide_series(revenue, total_assets),
        "fin_equity_ratio": _safe_divide_series(total_equity, total_assets),
        "fin_debt_to_assets": _safe_divide_series(total_debt, total_assets),
    }
    base_columns = {
        "revenue": "fin_revenue",
        "operating_profit": "fin_operating_profit",
        "net_profit": "fin_net_profit",
        "eps": "fin_eps",
        "total_assets": "fin_total_assets",
        "total_equity": "fin_total_equity",
        "total_debt": "fin_total_debt",
        "operating_cashflow": "fin_operating_cashflow",
        "free_cashflow": "fin_free_cashflow",
        "roe": "fin_roe",
        "roa": "fin_roa",
        "debt_to_equity": "fin_debt_to_equity",
    }
    for source_column, target_column in base_columns.items():
        if source_column in frame.columns:
            derived[target_column] = pd.to_numeric(frame[source_column], errors="coerce")
    for target_column in growth_columns.values():
        if target_column in frame.columns:
            derived[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    aligned = {}
    for field, values in derived.items():
        series = pd.Series(values.values, index=frame["availability_date"]).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        full_index = pd.DatetimeIndex(dates).union(pd.DatetimeIndex(series.index))
        aligned[field] = series.reindex(full_index).sort_index().ffill().reindex(dates)
    return aligned


def _parent_financial_matrices(
    financials: pd.DataFrame,
    market_cap: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    fields = (
        "fin_debt_to_assets",
        "fin_free_cashflow",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_roa",
        "fin_roe",
    )
    matrices = {
        field: pd.DataFrame(np.nan, index=mask.index, columns=mask.columns, dtype=float)
        for field in fields
    }
    target_symbols = set(mask.columns)
    for symbol, frame in financials.groupby("symbol", sort=False):
        symbol_text = str(symbol)
        if symbol_text not in target_symbols:
            continue
        aligned = _parent_align_financials(frame, mask.index)
        for field in fields:
            if field in aligned:
                matrices[field].loc[:, symbol_text] = aligned[field].to_numpy(dtype=float)
    for field in fields:
        matrices[field] = matrices[field].where(mask)
    safe_market_cap = market_cap.replace(0, np.nan)
    matrices["fcf_to_price"] = matrices["fin_free_cashflow"].reindex_like(
        market_cap
    ) / safe_market_cap
    matrices["fcf_to_price"] = matrices["fcf_to_price"].where(mask)
    return matrices


def _compare_child_parent(
    *,
    child_rows: Sequence[Mapping[str, Any]],
    parent_matrices: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> None:
    child = {str(row["field"]): row for row in child_rows}
    if set(child) != set(parent_matrices):
        raise FactorV4_1SignalComputabilityRunnerError(
            "restricted child/parent primitive inventory mismatch"
        )
    for field, matrix in parent_matrices.items():
        expected = _primitive_row(field, "parent_independent", matrix, mask)
        observed = child[field]
        if (
            observed.get("matrix") != expected["matrix"]
            or any(
                observed.get(key) != expected[key]
                for key in (
                    "finite_count",
                    "nan_count",
                    "positive_inf_count",
                    "negative_inf_count",
                    "outside_mask_non_nan_count",
                )
            )
        ):
            raise FactorV4_1SignalComputabilityRunnerError(
                f"restricted pinned child differs from independent parent: {field}"
            )


def _read_json_binding(
    *,
    path: Path,
    expected_sha256: str,
    private: bool,
    binding_id: str,
    semantic_field: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _stable_bytes(path, expected_sha256=expected_sha256, private=private)
    value = _strict_json(raw, str(path))
    binding = {
        "binding_id": binding_id,
        "absolute_path": str(path),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if semantic_field is not None:
        semantic = value.get(semantic_field)
        _sha(semantic, f"{binding_id}.{semantic_field}")
        binding["semantic_sha256"] = semantic
    return value, binding


def _read_predecessors(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source_receipt, source_binding = _read_json_binding(
        path=_absolute_path(args.source_receipt_path, "source receipt path"),
        expected_sha256=args.expected_source_receipt_sha256,
        private=True,
        binding_id="aquant_source_receipt",
        semantic_field="receipt_semantic_sha256",
    )
    source_audit, source_audit_binding = _read_json_binding(
        path=_absolute_path(args.source_audit_path, "source audit path"),
        expected_sha256=args.expected_source_audit_sha256,
        private=True,
        binding_id="source_idea_audit",
        semantic_field="audit_semantic_sha256",
    )
    mapping_proof, mapping_binding = _read_json_binding(
        path=_absolute_path(args.mapping_proof_path, "mapping proof path"),
        expected_sha256=args.expected_mapping_proof_sha256,
        private=True,
        binding_id="primitive_mapping_proof",
        semantic_field="proof_semantic_sha256",
    )
    candidate_catalog, catalog_binding = _read_json_binding(
        path=_absolute_path(args.candidate_catalog_path, "candidate catalog path"),
        expected_sha256=args.expected_candidate_catalog_sha256,
        private=True,
        binding_id="candidate_catalog",
        semantic_field="semantic_sha256",
    )
    ideas = evaluator.bind_pinned_source_ideas_v4_1(
        source_receipt=source_receipt,
        source_idea_audit=source_audit,
        primitive_mapping_proof=mapping_proof,
        formal_catalog=candidate_catalog,
    )
    no_label_root = _absolute_path(args.no_label_bundle_path, "no-label bundle path")
    profile, profile_binding = _read_json_binding(
        path=no_label_root / no_label.OPERATOR_PROFILE_FILENAME,
        expected_sha256=args.expected_no_label_profile_sha256,
        private=True,
        binding_id="no_label_operator_profile",
        semantic_field="operator_profile_semantic_sha256",
    )
    diagnostic, diagnostic_binding = _read_json_binding(
        path=no_label_root / no_label.DIAGNOSTIC_FILENAME,
        expected_sha256=args.expected_no_label_diagnostic_sha256,
        private=True,
        binding_id="no_label_signal_diagnostic",
        semantic_field="diagnostic_semantic_sha256",
    )
    _stable_bytes(
        no_label_root / no_label.READBACK_FILENAME,
        expected_sha256=args.expected_no_label_readback_sha256,
        private=True,
    )
    no_label_contract = no_label.build_private_bundle_contract_v4_1(
        expected_artifacts={
            no_label.OPERATOR_PROFILE_FILENAME: profile,
            no_label.DIAGNOSTIC_FILENAME: diagnostic,
        }
    )
    no_label_readback = private_io.readback_private_bundle(
        no_label_root, contract=no_label_contract
    )
    if no_label_readback.get("accepted") is not True:
        raise FactorV4_1SignalComputabilityRunnerError(
            "accepted no-label predecessor readback failed"
        )
    operator_root = _absolute_path(args.operator_bundle_path, "operator bundle path")
    operator_proof, operator_binding = _read_json_binding(
        path=operator_root / operator_equivalence.PROOF_FILENAME,
        expected_sha256=args.expected_operator_proof_sha256,
        private=True,
        binding_id="operator_runtime_equivalence_proof",
        semantic_field="proof_semantic_sha256",
    )
    _stable_bytes(
        operator_root / operator_equivalence.READBACK_FILENAME,
        expected_sha256=args.expected_operator_readback_sha256,
        private=True,
    )
    normalized_operator = operator_equivalence.validate_operator_runtime_equivalence_proof_v4_1(
        operator_proof
    )
    operator_contract = operator_equivalence.build_private_bundle_contract_v4_1(
        expected_proof=normalized_operator
    )
    operator_readback = private_io.readback_private_bundle(
        operator_root, contract=operator_contract
    )
    if (
        operator_readback.get("accepted") is not True
        or normalized_operator.get("operator_runtime_equivalence_verified") is not True
        or normalized_operator.get("signal_computability_proven") is not False
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "operator-equivalence predecessor authority mismatch"
        )
    idea_ids = [row["candidate_id"] for row in ideas]
    diagnostic_ids = [row["candidate_id"] for row in diagnostic["rows"]]
    operator_ids = [row["candidate_id"] for row in normalized_operator["rows"]]
    if not idea_ids == diagnostic_ids == operator_ids:
        raise FactorV4_1SignalComputabilityRunnerError(
            "exact-37 predecessor candidate order mismatch"
        )
    bindings = sorted(
        [
            source_binding,
            source_audit_binding,
            mapping_binding,
            catalog_binding,
            profile_binding,
            diagnostic_binding,
            operator_binding,
        ],
        key=lambda row: row["binding_id"],
    )
    return ideas, diagnostic, normalized_operator, bindings


def _calendar_accounting(
    aquant_bars: pd.DataFrame, myquant_dates: pd.DatetimeIndex
) -> dict[str, Any]:
    aquant_dates = sorted(aquant_bars["trade_date"].dt.strftime("%Y%m%d").unique())
    myquant = sorted(myquant_dates.strftime("%Y%m%d").tolist())
    aquant_set = set(aquant_dates)
    myquant_set = set(myquant)
    max_observed = max(aquant_dates)
    missing = sorted(date for date in myquant if date <= max_observed and date not in aquant_set)
    off_calendar = sorted(aquant_set - myquant_set)
    intersection = sorted(aquant_set & myquant_set)
    tail = sorted(date for date in myquant if date > max_observed)
    return {
        "aquant_date_count": len(aquant_dates),
        "aquant_dates_sha256": _sha_lines(aquant_dates),
        "intersection_count": len(intersection),
        "intersection_sha256": _sha_lines(intersection),
        "max_observation_age_open_sessions": len(tail),
        "missing_myquant_through_max_observed_count": len(missing),
        "missing_myquant_through_max_observed_sha256": _sha_lines(missing),
        "myquant_date_count": len(myquant),
        "myquant_dates_sha256": _sha_lines(myquant),
        "off_myquant_calendar_count": len(off_calendar),
        "off_myquant_calendar_sha256": _sha_lines(off_calendar),
        "tail_after_max_observed_count": len(tail),
        "tail_after_max_observed_sha256": _sha_lines(tail),
    }


def _assert_bounded_calendar(accounting: Mapping[str, Any]) -> None:
    if dict(accounting) != contract.EXPECTED_CALENDAR_ACCOUNTING:
        changed = sorted(
            key
            for key, expected in contract.EXPECTED_CALENDAR_ACCOUNTING.items()
            if accounting.get(key) != expected
        )
        raise TrustedComputabilityBlocker(
            "bounded_calendar_fact_drift:" + ",".join(changed)
        )


def _source_for_primitive(field: str) -> str:
    if field in contract.SOURCE_PARTITION["myquant_frozen_fields"]:
        return "myquant_strict_snapshot_20260717T172132Z"
    if field in contract.SOURCE_PARTITION["aquant_git_bar_fields"]:
        return "aquant_git_bars_4424dcec"
    if field in contract.SOURCE_PARTITION["aquant_financial_fields"]:
        return "aquant_git_financials_4424dcec"
    raise FactorV4_1SignalComputabilityRunnerError(
        f"primitive source partition is undefined: {field}"
    )


def _candidate_row(
    *,
    idea: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    signal: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict[str, Any]:
    counts = _matrix_counts(signal, mask)
    descriptor = evaluator.matrix_hash_descriptor_v4_1(signal)
    predecessor_status = predecessor["status"]
    preserved = predecessor_status == no_label.STATUS_SIGNAL_DIAGNOSTIC
    if preserved:
        expected = {
            "signal_matrix": predecessor["signal_matrix"],
            "finite_count": predecessor["finite_count"],
            "nan_count": predecessor["nan_count"],
            "positive_inf_count": predecessor["positive_inf_count"],
            "negative_inf_count": predecessor["negative_inf_count"],
            "outside_mask_non_nan_count": predecessor["outside_mask_non_nan_count"],
        }
        actual = {"signal_matrix": descriptor, **counts}
        if actual != expected:
            raise FactorV4_1SignalComputabilityRunnerError(
                f"accepted predecessor descriptor drift: {idea['name']}"
            )
    row = {
        "candidate_id": idea["candidate_id"],
        "name": idea["name"],
        "input_fields": list(idea["input_fields"]),
        "source_definition_sha256": idea["source_definition_sha256"],
        "catalog_definition_sha256": idea["catalog_definition_sha256"],
        "mapping_semantic_sha256": idea["mapping_semantic_sha256"],
        "normalized_ast_sha256": idea["full_candidate_normalized_ast_sha256"],
        "predecessor_status": predecessor_status,
        "predecessor_descriptor_preserved": preserved,
        "status": "source_semantic_computability_verified",
        "eligible_cell_count": int(mask.to_numpy(dtype=bool, copy=False).sum()),
        **counts,
        "signal_matrix": descriptor,
    }
    if row["finite_count"] <= 0 or row["outside_mask_non_nan_count"] != 0:
        raise TrustedComputabilityBlocker(
            f"candidate_not_computable:{idea['name']}"
        )
    row["row_semantic_sha256"] = _semantic_sha(row)
    return row


def _compute_once(
    *,
    args: argparse.Namespace,
    ideas: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    git_root: Path,
    matrix_source: bytes,
    source_blobs: Sequence[Mapping[str, Any]],
    pass_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    bars_tree = _direct_tree_oid(git_root, contract.PINNED_COMMIT, BARS_TREE_PATH)
    financial_tree = _direct_tree_oid(
        git_root, contract.PINNED_COMMIT, FINANCIAL_TREE_PATH
    )
    if bars_tree != contract.EXPECTED_BARS_TREE_OID or financial_tree != (
        contract.EXPECTED_FINANCIAL_TREE_OID
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned A_quant direct tree OID mismatch"
        )
    bars, bars_ipc, bars_accounting, _ = _load_aquant_bars(git_root, bars_tree)
    financials, financial_ipc, financial_accounting, _ = _load_aquant_financials(
        git_root, financial_tree
    )
    market_matrices, mask, strict_binding, _ = _load_strict_market_and_mask(
        table_root=_absolute_path(args.table_root, "strict table root"),
        pit_path=_absolute_path(args.pit_membership_path, "PIT membership path"),
        expected_pit_sha256=args.expected_pit_membership_sha256,
        pit_manifest_path=_absolute_path(
            args.pit_manifest_path, "PIT generation manifest path"
        ),
        expected_pit_manifest_sha256=args.expected_pit_manifest_sha256,
        components_path=_absolute_path(args.components_path, "components path"),
        expected_components_sha256=args.expected_components_sha256,
        no_label_diagnostic=diagnostic,
    )
    if mask.size > MAX_AXIS_CELLS:
        raise FactorV4_1SignalComputabilityRunnerError(
            "accepted PIT axes exceed resource limit"
        )
    calendar = _calendar_accounting(bars, mask.index)
    _assert_bounded_calendar(calendar)
    parent_bars = _parent_bar_matrices(bars, mask)
    child_bars = _run_restricted_child(
        operation="bars", matrix_source=matrix_source, mask=mask, payloads=[bars_ipc]
    )
    _compare_child_parent(
        child_rows=child_bars["rows"], parent_matrices=parent_bars, mask=mask
    )
    market_cap_values = np.asarray(
        parent_bars["market_cap"].to_numpy(dtype=np.float64, copy=True),
        dtype="<f8",
        order="C",
    )
    child_financials = _run_restricted_child(
        operation="financials",
        matrix_source=matrix_source,
        mask=mask,
        payloads=[financial_ipc, market_cap_values.tobytes(order="C")],
    )
    parent_financials = _parent_financial_matrices(
        financials, parent_bars["market_cap"], mask
    )
    _compare_child_parent(
        child_rows=child_financials["rows"],
        parent_matrices=parent_financials,
        mask=mask,
    )
    all_matrices = {
        **market_matrices,
        **parent_bars,
        **parent_financials,
    }
    if set(all_matrices) != set(contract.PRIMITIVE_NAMES) or len(all_matrices) > (
        MAX_PRIMITIVE_MATRICES
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "exact primitive matrix inventory mismatch"
        )
    primitives = [
        _primitive_row(
            field, _source_for_primitive(field), all_matrices[field], mask
        )
        for field in contract.PRIMITIVE_NAMES
    ]
    expression_matrices = {
        field: all_matrices[field] for field in contract.EXPRESSION_PRIMITIVE_NAMES
    }
    predecessors = {row["candidate_id"]: row for row in diagnostic["rows"]}
    rows = []
    for idea in ideas:
        signal = evaluator.evaluate_pinned_idea_v4_1(
            idea=idea,
            matrices=expression_matrices,
            eligibility_mask=mask,
        )
        rows.append(
            _candidate_row(
                idea=idea,
                predecessor=predecessors[idea["candidate_id"]],
                signal=signal,
                mask=mask,
            )
        )
        del signal
    if len(rows) != contract.EXPECTED_CANDIDATE_COUNT:
        raise FactorV4_1SignalComputabilityRunnerError(
            "exact candidate computation count mismatch"
        )
    runtime = _runtime_identity()
    elapsed = time.monotonic() - started
    parent_peak = _peak_rss_bytes()
    if elapsed > MAX_TOTAL_WALL_SECONDS or parent_peak > MAX_PARENT_RSS_BYTES:
        raise FactorV4_1SignalComputabilityRunnerError(
            "parent computation exceeded wall/RSS abort ceiling"
        )
    result_manifest = contract.semantic_sha256_v4_1(
        {"primitive_matrices": primitives, "rows": rows}
    )
    transformation = {
        "matrix_dataset_path": MATRIX_DATASET_PATH,
        "matrix_dataset_sha256": EXPECTED_SOURCE_BLOBS[MATRIX_DATASET_PATH]["sha256"],
        "ast_manifest_sha256": contract.EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256,
        "child_exec_event_count": 1,
        "child_filesystem_access_after_audit": False,
        "child_network_access_after_audit": False,
        "child_parent_all_match": True,
        "descriptor_contract": evaluator.MATRIX_HASH_CONTRACT_VERSION,
        "bars_child_descriptor_semantic_sha256": _semantic_sha(child_bars["rows"]),
        "financial_child_descriptor_semantic_sha256": _semantic_sha(
            child_financials["rows"]
        ),
    }
    return {
        "pass_id": pass_id,
        "bars_inventory": bars_accounting,
        "calendar_accounting": calendar,
        "financial_inventory": financial_accounting,
        "strict_binding": strict_binding,
        "source_blobs": [copy.deepcopy(dict(row)) for row in source_blobs],
        "transformation_contract": transformation,
        "runtime_identity": runtime,
        "primitive_matrices": primitives,
        "rows": rows,
        "result_manifest_sha256": result_manifest,
        "child_parent_all_match": True,
        "outside_mask_all_zero": all(
            row["outside_mask_non_nan_count"] == 0 for row in [*primitives, *rows]
        ),
        "candidate_count": len(rows),
        "elapsed_seconds": elapsed,
        "parent_peak_rss_bytes": parent_peak,
        "child_peak_rss_bytes": max(
            int(child_bars["peak_rss_bytes"]),
            int(child_financials["peak_rss_bytes"]),
        ),
    }


def _stable_pass_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key
        not in {
            "pass_id",
            "elapsed_seconds",
            "parent_peak_rss_bytes",
            "child_peak_rss_bytes",
        }
    }


def _baseline_bindings(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution_path = _absolute_path(args.execution_baseline_path, "execution baseline path")
    worktree_path = _absolute_path(args.worktree_baseline_path, "worktree baseline path")
    execution_raw = _stable_bytes(
        execution_path,
        expected_sha256=args.expected_execution_baseline_sha256,
        private=True,
    )
    worktree_raw = _stable_bytes(
        worktree_path,
        expected_sha256=args.expected_worktree_baseline_sha256,
        private=True,
    )
    execution = _strict_json(execution_raw, str(execution_path))
    worktree = _strict_json(worktree_raw, str(worktree_path))
    if execution.get("schema_version") != (
        "factor-governance-v4.1-signal-computability-execution-baseline.v1"
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "execution baseline schema mismatch"
        )
    return (
        [
            {
                "binding_id": "execution_baseline",
                "absolute_path": str(execution_path),
                "byte_sha256": hashlib.sha256(execution_raw).hexdigest(),
                "semantic_sha256": _semantic_sha(execution),
            },
            {
                "binding_id": "worktree_content_baseline",
                "absolute_path": str(worktree_path),
                "byte_sha256": hashlib.sha256(worktree_raw).hexdigest(),
                "semantic_sha256": _semantic_sha(worktree),
            },
        ],
        worktree,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.cycle_id != contract.EXPECTED_CYCLE_ID:
        raise FactorV4_1SignalComputabilityRunnerError("cycle identity mismatch")
    repository_root = _absolute_path(args.repository_root, "repository root")
    if repository_root != PROJECT_ROOT:
        raise FactorV4_1SignalComputabilityRunnerError(
            "repository root differs from the executing checkout"
        )
    git_root = _absolute_path(args.aquant_git_root, "A_quant Git root")
    private_root = _absolute_path(args.private_root, "private root")
    baseline_bindings, worktree_baseline = _baseline_bindings(args)
    _validate_worktree_baseline(root=repository_root, baseline=worktree_baseline)
    code_bindings = _read_code_bindings(args, repository_root)
    commit_type = _run_git(git_root, "cat-file", "-t", contract.PINNED_COMMIT)
    resolved = _run_git(
        git_root, "rev-parse", "--verify", f"{contract.PINNED_COMMIT}^{{commit}}"
    )
    if commit_type != b"commit\n" or resolved.strip().decode() != contract.PINNED_COMMIT:
        raise FactorV4_1SignalComputabilityRunnerError(
            "pinned A_quant commit object identity mismatch"
        )
    source_blobs = []
    matrix_source = b""
    for path in sorted(EXPECTED_SOURCE_BLOBS):
        raw, binding = _read_source_blob(git_root, contract.PINNED_COMMIT, path)
        source_blobs.append(binding)
        if path == MATRIX_DATASET_PATH:
            matrix_source = raw
    _extract_transform_code(matrix_source)
    ideas, diagnostic, operator_proof, predecessor_bindings = _read_predecessors(
        args
    )
    first = _compute_once(
        args=args,
        ideas=ideas,
        diagnostic=diagnostic,
        git_root=git_root,
        matrix_source=matrix_source,
        source_blobs=source_blobs,
        pass_id="first",
    )
    fresh = _compute_once(
        args=args,
        ideas=ideas,
        diagnostic=diagnostic,
        git_root=git_root,
        matrix_source=matrix_source,
        source_blobs=source_blobs,
        pass_id="fresh_readback",
    )
    if _stable_pass_identity(first) != _stable_pass_identity(fresh):
        raise FactorV4_1SignalComputabilityRunnerError(
            "fresh source recomputation differs from first pass"
        )
    if time.monotonic() - started > MAX_TOTAL_WALL_SECONDS:
        raise FactorV4_1SignalComputabilityRunnerError(
            "total computation exceeded wall abort ceiling"
        )
    git_identity = _git_identity(git_root)
    receipt = contract.build_input_semantics_receipt_v4_1(
        baseline_bindings=baseline_bindings,
        code_bindings=code_bindings,
        predecessor_bindings=predecessor_bindings,
        git_identity=git_identity,
        aquant_source_blobs=source_blobs,
        bars_inventory=first["bars_inventory"],
        calendar_accounting=first["calendar_accounting"],
        financial_inventory=first["financial_inventory"],
        transformation_contract=first["transformation_contract"],
        runtime_identity=first["runtime_identity"],
        source_partition=copy.deepcopy(contract.SOURCE_PARTITION),
        resource_limits=_resource_limits(),
        protected_contexts=copy.deepcopy(contract.EXPECTED_PROTECTED_CONTEXTS),
    )
    pass_rows = [
        {
            "pass_id": value["pass_id"],
            "result_manifest_sha256": value["result_manifest_sha256"],
            "runtime_semantic_sha256": value["runtime_identity"][
                "runtime_semantic_sha256"
            ],
            "child_parent_all_match": value["child_parent_all_match"],
            "outside_mask_all_zero": value["outside_mask_all_zero"],
            "candidate_count": value["candidate_count"],
            "elapsed_seconds": value["elapsed_seconds"],
            "parent_peak_rss_bytes": value["parent_peak_rss_bytes"],
            "child_peak_rss_bytes": value["child_peak_rss_bytes"],
        }
        for value in (first, fresh)
    ]
    proof = contract.build_signal_computability_proof_v4_1(
        semantics_receipt=receipt,
        predecessor_proof_bindings=predecessor_bindings,
        computation_passes=pass_rows,
        primitive_matrices=first["primitive_matrices"],
        rows=first["rows"],
    )
    if (
        operator_proof["operator_runtime_equivalence_verified"] is not True
        or proof["signal_computability_proven"] is not True
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "final operator/computability authority chain mismatch"
        )
    artifacts = {
        contract.SEMANTICS_FILENAME: receipt,
        contract.PROOF_FILENAME: proof,
    }
    bundle_contract = contract.build_private_bundle_contract_v4_1(
        expected_artifacts=artifacts
    )

    def revalidate_inputs() -> None:
        _validate_worktree_baseline(root=repository_root, baseline=worktree_baseline)
        if _read_code_bindings(args, repository_root) != code_bindings:
            raise FactorV4_1SignalComputabilityRunnerError(
                "code bindings drifted before publication"
            )
        current_baselines, _ = _baseline_bindings(args)
        if current_baselines != baseline_bindings:
            raise FactorV4_1SignalComputabilityRunnerError(
                "baseline bindings drifted before publication"
            )
        if _direct_tree_oid(git_root, contract.PINNED_COMMIT, BARS_TREE_PATH) != (
            contract.EXPECTED_BARS_TREE_OID
        ) or _direct_tree_oid(
            git_root, contract.PINNED_COMMIT, FINANCIAL_TREE_PATH
        ) != contract.EXPECTED_FINANCIAL_TREE_OID:
            raise FactorV4_1SignalComputabilityRunnerError(
                "A_quant tree bindings drifted before publication"
            )
        _inventory_table(_absolute_path(args.table_root, "strict table root"))

    published = private_io.publish_private_bundle(
        private_root=private_root,
        run_id=args.run_id,
        artifacts=artifacts,
        contract=bundle_contract,
        revalidate_inputs=revalidate_inputs,
    )
    independent = private_io.readback_private_bundle(
        published["bundle_path"], contract=bundle_contract
    )
    if (
        independent.get("accepted") is not True
        or independent["readback_report"].get("signal_computability_proven") is not True
        or independent["readback_report"].get("new_risk_authorized") is not False
    ):
        raise FactorV4_1SignalComputabilityRunnerError(
            "independent signal computability bundle readback failed"
        )
    return {
        "accepted": True,
        "readiness": contract.READINESS,
        "bundle_path": independent["bundle_path"],
        "receipt_semantic_sha256": receipt["receipt_semantic_sha256"],
        "proof_semantic_sha256": proof["proof_semantic_sha256"],
        "readback_report_semantic_sha256": independent["readback_report"][
            "report_semantic_sha256"
        ],
        "candidate_count": contract.EXPECTED_CANDIDATE_COUNT,
        "predecessor_preserved_count": contract.EXPECTED_PREDECESSOR_PRESERVED_COUNT,
        "newly_computed_count": contract.EXPECTED_NEWLY_COMPUTED_COUNT,
        "signal_computability_proven": True,
        "new_risk_authorized": False,
        "claim_negatives": copy.deepcopy(contract.CLAIM_NEGATIVES),
        "side_effects": copy.deepcopy(contract.SIDE_EFFECT_FIELDS),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the read-only Factor v4.1 exact-37 computability proof."
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repository-root")
    parser.add_argument("--aquant-git-root")
    parser.add_argument("--private-root")
    parser.add_argument("--run-id")
    parser.add_argument("--cycle-id")
    parser.add_argument("--execution-baseline-path")
    parser.add_argument("--expected-execution-baseline-sha256")
    parser.add_argument("--worktree-baseline-path")
    parser.add_argument("--expected-worktree-baseline-sha256")
    parser.add_argument("--source-receipt-path")
    parser.add_argument("--expected-source-receipt-sha256")
    parser.add_argument("--source-audit-path")
    parser.add_argument("--expected-source-audit-sha256")
    parser.add_argument("--mapping-proof-path")
    parser.add_argument("--expected-mapping-proof-sha256")
    parser.add_argument("--candidate-catalog-path")
    parser.add_argument("--expected-candidate-catalog-sha256")
    parser.add_argument("--no-label-bundle-path")
    parser.add_argument("--expected-no-label-profile-sha256")
    parser.add_argument("--expected-no-label-diagnostic-sha256")
    parser.add_argument("--expected-no-label-readback-sha256")
    parser.add_argument("--operator-bundle-path")
    parser.add_argument("--expected-operator-proof-sha256")
    parser.add_argument("--expected-operator-readback-sha256")
    parser.add_argument("--pit-membership-path")
    parser.add_argument("--expected-pit-membership-sha256")
    parser.add_argument("--pit-manifest-path")
    parser.add_argument("--expected-pit-manifest-sha256")
    parser.add_argument("--components-path")
    parser.add_argument("--expected-components-sha256")
    parser.add_argument("--table-root")
    parser.add_argument("--expected-builder-sha256")
    parser.add_argument("--expected-contract-sha256")
    args = parser.parse_args(argv)
    if args._worker:
        return args
    missing = [
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "_worker"} and getattr(args, action.dest) is None
    ]
    if missing:
        parser.error("missing required arguments: " + ",".join(sorted(missing)))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args._worker:
        return _worker_main()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
