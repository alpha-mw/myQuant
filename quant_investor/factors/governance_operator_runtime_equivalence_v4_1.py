"""Fail-closed operator equivalence proof for the exact A_quant v4.1 subset.

This module proves only that the pinned A_quant arithmetic, ``ts_mean``, and
``cs_rank`` operators agree with the isolated myQuant evaluator when both are
run under the same hash-bound point-in-time mask envelope.  It does not prove
input-data semantics, signal computability, screening, qualification, or any
production authority.
"""

from __future__ import annotations

import __future__ as future_module
import ast
import builtins
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats as scipy_stats

from quant_investor.factors import governance_private_bundle_io as private_io
from quant_investor.factors.governance_aquant_no_label_eval_v4_1 import (
    ALLOWED_DATA_FIELDS,
    EXPECTED_PINNED_IDEA_COUNT,
    PINNED_COMMIT,
    PINNED_SOURCE_FILES,
    evaluate_expression_v4_1,
    evaluate_pinned_idea_v4_1,
    matrix_hash_descriptor_v4_1,
    semantic_sha256_v4_1,
)


PROTOCOL_VERSION = "v4.1"
PROOF_SCHEMA_VERSION = (
    "factor-governance-aquant-operator-runtime-equivalence-proof.v4.1"
)
READBACK_SCHEMA_VERSION = (
    "factor-governance-aquant-operator-runtime-equivalence-readback.v4.1"
)
PROOF_FILENAME = "operator_runtime_equivalence_proof.v4_1.json"
READBACK_FILENAME = "operator_runtime_equivalence_readback.v4_1.json"
BUNDLE_INPUT_FILENAMES = (PROOF_FILENAME,)
BUNDLE_FILENAMES = (*BUNDLE_INPUT_FILENAMES, READBACK_FILENAME)
PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_operator_runtime_equivalence",
)
READINESS = "EXPLORATORY_OPERATOR_RUNTIME_EQUIVALENCE_ONLY"
CLAIM_SCOPE = "pinned_aquant_operators_under_hash_bound_myquant_pit_envelope"
EXPECTED_CYCLE_ID = "cn_full_a_v4_1_20260717"
EXPECTED_BOUND_IDEA_MANIFEST_SHA256 = (
    "47fb13fc38aa2709c4807c814895eba4d8ff5ff3f125ae683e564c0810028caf"
)
EXPECTED_DIFFERENTIAL_RESULT_SHA256 = (
    "370375e132094ecef2d564be08721e3356b346ea363e6101a20e146e442fad1d"
)
EXPECTED_TS_MEAN_WINDOWS = (5, 20, 40, 60, 80, 120, 160, 200)
EXPECTED_FIXTURE_CELL_COUNT = 211 * 11

EXPRESSION_SOURCE_PATH = "A_quant/app/factor_sandbox/expression.py"
OPERATORS_SOURCE_PATH = "A_quant/app/factor_sandbox/operators.py"
PINNED_RUNTIME_SOURCE_HASHES = {
    EXPRESSION_SOURCE_PATH: PINNED_SOURCE_FILES[EXPRESSION_SOURCE_PATH],
    OPERATORS_SOURCE_PATH: PINNED_SOURCE_FILES[OPERATORS_SOURCE_PATH],
}
PINNED_RUNTIME_SOURCE_SIZES = {
    EXPRESSION_SOURCE_PATH: 8484,
    OPERATORS_SOURCE_PATH: 6814,
}

REQUIRED_INPUT_BINDING_IDS = (
    "aquant_source_receipt",
    "candidate_catalog",
    "formal_catalog_readback",
    "no_label_operator_profile",
    "no_label_readback",
    "no_label_signal_diagnostic",
    "primitive_mapping_proof",
    "source_idea_audit",
)
REQUIRED_CODE_BINDING_IDS = (
    "build_factor_v4_1_operator_runtime_equivalence.py",
    "governance_aquant_no_label_eval_v4_1.py",
    "governance_operator_runtime_equivalence_v4_1.py",
    "governance_private_bundle_io.py",
)

EXPECTED_INPUT_BINDING_SPECS = {
    "aquant_source_receipt": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_cycle/"
            "factor_v4_1_discovery_20260718T170345Z/"
            "aquant_source_receipt.v4_1.json"
        ),
        "byte_sha256": (
            "80c7957ee86d474975328fdf09bf35011212a49c6b92874e5e7bfb80cb465608"
        ),
        "semantic_sha256": (
            "874b19f3bf1d94a054f30230e8c493714a35feecb71a55ca08a1a5036db44992"
        ),
    },
    "candidate_catalog": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_formal_catalog/"
            "factor_v4_1_formal_catalog_20260718T191045Z/"
            "candidate_catalog.v4.json"
        ),
        "byte_sha256": (
            "09cb6ac73590a48e826845f608e4bd733e27c183b6abaa2079436ba5bb2169ee"
        ),
        "semantic_sha256": (
            "b4f2b2b80e1bfc69ea8be9228d9021afdbeee28540fc51c2e7ead100a219f75a"
        ),
    },
    "formal_catalog_readback": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_formal_catalog/"
            "factor_v4_1_formal_catalog_20260718T191045Z/"
            "formal_catalog_materialization_readback.v4_1.json"
        ),
        "byte_sha256": (
            "edbac74d2765de32a50d32b77e3fb95e772f0eb71affdcc046bac8e18575a666"
        ),
        "semantic_sha256": (
            "e606b64707b65f40b6049299c9a2e3ccfcd887210ee64a95108b80119eca8288"
        ),
    },
    "no_label_operator_profile": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_no_label_diagnostic/"
            "factor_v4_1_no_label_diagnostic_20260718T204202Z/"
            "no_label_operator_profile.v4_1.json"
        ),
        "byte_sha256": (
            "c65c71d88946dfc153f193369204ac1620ca78314d275c4bcb5dbfa8d5b63a1e"
        ),
        "semantic_sha256": (
            "6064d1cc378ed02eb8189f92b47946eb17ff7c61253eeb1628d6a08780fe473e"
        ),
    },
    "no_label_readback": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_no_label_diagnostic/"
            "factor_v4_1_no_label_diagnostic_20260718T204202Z/"
            "no_label_diagnostic_readback.v4_1.json"
        ),
        "byte_sha256": (
            "14da22940edbf60184135afd57890a23ffd04e6675c93a84a8d3334ac766f41b"
        ),
        "semantic_sha256": (
            "cc28d326defbf525d09d1abe19b20d088778555f8e483574d307ab77cab1489b"
        ),
    },
    "no_label_signal_diagnostic": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_no_label_diagnostic/"
            "factor_v4_1_no_label_diagnostic_20260718T204202Z/"
            "no_label_signal_diagnostic.v4_1.json"
        ),
        "byte_sha256": (
            "cc3ffd2c3a767b92d5752c9df47199b380d529f00ce7dfff2578435aed3b534e"
        ),
        "semantic_sha256": (
            "928520637d4fa8294a6a1a2bea9bd6dd7504bb2223f476340047cbb9cde68aea"
        ),
    },
    "primitive_mapping_proof": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_formal_catalog/"
            "factor_v4_1_formal_catalog_20260718T191045Z/"
            "primitive_mapping_proof.v4_1.json"
        ),
        "byte_sha256": (
            "40aa4a55ba75267cb1dd66aa01c7a06ee1d77486997b0767b52fb78bd7cba1d1"
        ),
        "semantic_sha256": (
            "4c9254ab375ac88982bd35c3a88e3bb67a62402f685f5345c789f0d356d878ba"
        ),
    },
    "source_idea_audit": {
        "relative_path": (
            "reports/factor_governance/private/v4_1_cycle/"
            "factor_v4_1_discovery_20260718T170345Z/"
            "source_idea_audit.v4_1.json"
        ),
        "byte_sha256": (
            "4c76a911bd0eff4d2e0d602d6f02bdac706832b857432b6adb051395fe066f18"
        ),
        "semantic_sha256": (
            "3faee9d33a10d1b25182308e88c7cf85edd4e7b8d90901de4bdff3311287a0ba"
        ),
    },
}

CODE_BINDING_RELATIVE_PATHS = {
    "build_factor_v4_1_operator_runtime_equivalence.py": (
        "scripts/build_factor_v4_1_operator_runtime_equivalence.py"
    ),
    "governance_aquant_no_label_eval_v4_1.py": (
        "quant_investor/factors/governance_aquant_no_label_eval_v4_1.py"
    ),
    "governance_operator_runtime_equivalence_v4_1.py": (
        "quant_investor/factors/governance_operator_runtime_equivalence_v4_1.py"
    ),
    "governance_private_bundle_io.py": (
        "quant_investor/factors/governance_private_bundle_io.py"
    ),
}

AUTHORITY_FIELDS = {
    "classification_only": True,
    "operator_runtime_equivalence_verified": True,
    "signal_computability_proven": False,
    "screening_authority": False,
    "screening_eligible": False,
    "bh_authority": False,
    "family_bh_authoritative": False,
    "qualification": False,
    "qualified": False,
    "healthy": False,
    "admission_authority": False,
    "formal_admission_authority": False,
    "proposal_authority": False,
    "proposal_eligible": False,
    "registry_authority": False,
    "registry_entry_created": False,
    "production_apply_enabled": False,
    "new_risk_eligible": False,
    "new_risk_authorized": False,
}
SIDE_EFFECT_FIELDS = {
    "provider": False,
    "network": False,
    "wal": False,
    "budget": False,
    "proposal": False,
    "registry": False,
    "apply": False,
    "transaction": False,
    "production": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
}
OPERATOR_PROBE_EXPRESSIONS = (
    ("binary_add", "close + open"),
    ("binary_divide", "close / amount"),
    ("binary_multiply", "close * open"),
    ("binary_subtract", "close - open"),
    ("cs_rank_ties_and_nonfinite", "cs_rank(close)"),
    ("ts_mean_window_200", "ts_mean(close, 200)"),
    ("unary_minus", "-close"),
)

_SHA256_CHARS = frozenset("0123456789abcdef")
MAX_BOUND_FILE_BYTES = 64 * 1024 * 1024


class FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(ValueError):
    """Raised when operator-equivalence evidence fails closed."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_1(dict(value)) + b"\n"


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"{label} must be a non-empty canonical string"
        )
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"{label} fields mismatch"
        )
    return copy.deepcopy(dict(value))


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if field in payload:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"self-hash field already exists: {field}"
        )
    payload[field] = semantic_sha256_v4_1(payload)
    return payload


def _validate_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    stored = _sha(value.get(field), f"{label}.{field}")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if semantic_sha256_v4_1(payload) != stored:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"{label} self-hash mismatch"
        )


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stable_regular_bytes(path: str | Path, label: str) -> bytes:
    target = Path(path)
    if not target.is_absolute() or any(
        not hasattr(os, flag) for flag in ("O_CLOEXEC", "O_NOFOLLOW")
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"stable descriptor read is unavailable: {label}"
        )
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"bound file descriptor open failed: {label}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > MAX_BOUND_FILE_BYTES
        ):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"bound file type/size check failed: {label}"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                    f"bound file was truncated during read: {label}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"bound file grew during read: {label}"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_metadata = os.lstat(target)
    except OSError as exc:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"bound file path disappeared: {label}"
        ) from exc
    if (
        _stat_signature(before) != _stat_signature(after)
        or stat.S_ISLNK(path_metadata.st_mode)
        or _stat_signature(path_metadata) != _stat_signature(after)
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"bound file identity changed during read: {label}"
        )
    return b"".join(chunks)


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(_stable_regular_bytes(path, str(path))).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expected_input_bindings_v4_1() -> list[dict[str, str]]:
    if set(EXPECTED_INPUT_BINDING_SPECS) != set(REQUIRED_INPUT_BINDING_IDS):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "expected input binding specification inventory mismatch"
        )
    root = _repository_root()
    return [
        {
            "binding_id": binding_id,
            "absolute_path": str(
                root / EXPECTED_INPUT_BINDING_SPECS[binding_id]["relative_path"]
            ),
            "byte_sha256": EXPECTED_INPUT_BINDING_SPECS[binding_id][
                "byte_sha256"
            ],
            "semantic_sha256": EXPECTED_INPUT_BINDING_SPECS[binding_id][
                "semantic_sha256"
            ],
        }
        for binding_id in REQUIRED_INPUT_BINDING_IDS
    ]


def expected_code_bindings_v4_1() -> list[dict[str, str]]:
    if set(CODE_BINDING_RELATIVE_PATHS) != set(REQUIRED_CODE_BINDING_IDS):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "expected code binding path inventory mismatch"
        )
    root = _repository_root()
    rows: list[dict[str, str]] = []
    for binding_id in REQUIRED_CODE_BINDING_IDS:
        path = root / CODE_BINDING_RELATIVE_PATHS[binding_id]
        first = _stable_regular_bytes(path, binding_id)
        second = _stable_regular_bytes(path, binding_id)
        if first != second:
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"bound code changed across stable reads: {binding_id}"
            )
        try:
            tree = ast.parse(first.decode("utf-8"), filename=str(path))
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"bound code is not valid UTF-8 Python: {binding_id}"
            ) from exc
        rows.append(
            {
                "binding_id": binding_id,
                "absolute_path": str(path),
                "byte_sha256": hashlib.sha256(first).hexdigest(),
                "ast_sha256": hashlib.sha256(
                    ast.dump(tree, include_attributes=True).encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows


def _module_descriptor(
    name: str, module: Any, *, version: str | None = None
) -> dict[str, str]:
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    resolved_version = _text(
        version or getattr(module, "__version__", None), f"{name} version"
    )
    return {
        "module": name,
        "version": resolved_version,
        "absolute_path": str(path),
        "byte_sha256": _file_sha256(path),
    }


def runtime_environment_descriptor_v4_1() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_sha256": _file_sha256(executable),
        "modules": [
            _module_descriptor("numpy", np),
            _module_descriptor("pandas", pd),
            _module_descriptor("scipy", scipy),
            _module_descriptor(
                "scipy.stats", scipy_stats, version=str(scipy.__version__)
            ),
        ],
        "import_isolation": "preloaded_exact_modules_with_deny_by_default_importer",
    }


def _validate_runtime_environment(value: Any) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "python_implementation",
            "python_version",
            "python_executable",
            "python_executable_sha256",
            "modules",
            "import_isolation",
        },
        "runtime environment",
    )
    _text(payload["python_implementation"], "python implementation")
    _text(payload["python_version"], "python version")
    executable = _text(payload["python_executable"], "python executable")
    if not executable.startswith("/"):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "python executable must be absolute"
        )
    _sha(payload["python_executable_sha256"], "python executable SHA")
    if payload["import_isolation"] != (
        "preloaded_exact_modules_with_deny_by_default_importer"
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "runtime import isolation contract mismatch"
        )
    modules = payload["modules"]
    if not isinstance(modules, list) or len(modules) != 4:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "runtime module inventory mismatch"
        )
    expected_names = ["numpy", "pandas", "scipy", "scipy.stats"]
    for expected_name, raw in zip(expected_names, modules, strict=True):
        row = _exact(
            raw,
            {"module", "version", "absolute_path", "byte_sha256"},
            "runtime module",
        )
        if row["module"] != expected_name or not str(row["absolute_path"]).startswith("/"):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                "runtime module identity mismatch"
            )
        _text(row["version"], "runtime module version")
        _sha(row["byte_sha256"], "runtime module SHA")
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(
        runtime_environment_descriptor_v4_1()
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "runtime environment differs from exact local recomputation"
        )
    return payload


@dataclass(frozen=True)
class _PinnedRuntime:
    expression_type: type
    operators: types.ModuleType


def _isolated_importer(
    name: str,
    _globals: Any = None,
    _locals: Any = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    if level != 0:
        raise ImportError("relative imports are forbidden in the pinned runtime")
    allowed = {
        "__future__": future_module,
        "ast": ast,
        "numpy": np,
        "pandas": pd,
        "scipy": scipy,
        "scipy.stats": scipy_stats,
        "typing": typing,
    }
    module = allowed.get(name)
    if module is None:
        raise ImportError(f"pinned runtime import is not allowlisted: {name}")
    return module


def _execute_pinned_module(name: str, source: bytes) -> types.ModuleType:
    try:
        text = source.decode("utf-8")
        ast.parse(text, filename=name)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"pinned source is not valid UTF-8 Python: {name}"
        ) from exc
    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = _isolated_importer
    module = types.ModuleType(name)
    module.__dict__["__builtins__"] = safe_builtins
    module.__dict__["__package__"] = ""
    try:
        exec(compile(text, name, "exec", dont_inherit=True), module.__dict__)
    except Exception as exc:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"pinned source execution failed: {name}: {type(exc).__name__}"
        ) from exc
    return module


def load_pinned_runtime_v4_1(
    *, expression_source: bytes, operators_source: bytes
) -> _PinnedRuntime:
    sources = {
        EXPRESSION_SOURCE_PATH: expression_source,
        OPERATORS_SOURCE_PATH: operators_source,
    }
    for path, expected in PINNED_RUNTIME_SOURCE_HASHES.items():
        raw = sources[path]
        if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != expected:
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"pinned runtime source SHA mismatch: {path}"
            )
    operators = _execute_pinned_module(OPERATORS_SOURCE_PATH, operators_source)
    expression = _execute_pinned_module(EXPRESSION_SOURCE_PATH, expression_source)
    expression_type = getattr(expression, "FactorExpression", None)
    if not isinstance(expression_type, type):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "pinned FactorExpression class is missing"
        )
    for function in ("ts_mean", "cs_rank"):
        if not callable(getattr(operators, function, None)):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"pinned operator is missing: {function}"
            )
    return _PinnedRuntime(expression_type=expression_type, operators=operators)


class _PitEnvelopeOperators:
    def __init__(self, pinned: types.ModuleType, mask: pd.DataFrame) -> None:
        self._pinned = pinned
        self._mask = mask

    def ts_mean(self, frame: pd.DataFrame, window: int) -> pd.DataFrame:
        return self._pinned.ts_mean(frame, window).where(self._mask)

    def cs_rank(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._pinned.cs_rank(frame.where(self._mask)).where(self._mask)


def build_differential_fixture_v4_1() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    dates = pd.date_range("2025-01-02", periods=211, freq="B", name="trade_date")
    symbols = [f"{index:06d}.SZ" for index in range(1, 12)]
    row_grid = np.arange(len(dates), dtype=np.float64)[:, None]
    column_grid = np.arange(len(symbols), dtype=np.float64)[None, :]
    mask_values = ((row_grid * 3.0 + column_grid * 5.0) % 17.0) > 2.0
    mask_values[:3, :] = True
    mask_values[40:45, 2] = False
    mask_values[44:49, 2] = True
    mask = pd.DataFrame(mask_values, index=dates, columns=symbols, dtype=bool)

    matrices: dict[str, pd.DataFrame] = {}
    for field_index, field in enumerate(sorted(ALLOWED_DATA_FIELDS)):
        values = (
            ((row_grid + 1.0) * (column_grid + 2.0) + field_index * 11.0) % 53.0
        ) - 19.0
        values = values / float(field_index % 5 + 1)
        values[((row_grid + column_grid + field_index) % 31.0) == 0.0] = 0.0
        values[((row_grid * 7.0 + column_grid + field_index) % 47.0) == 0.0] = np.nan
        tie_row = 10 + field_index % 5
        values[tie_row, 0] = 7.0
        values[tie_row, 1] = 7.0
        values[20 + field_index % 7, (field_index * 3) % len(symbols)] = np.inf
        values[30 + field_index % 7, (field_index * 5 + 1) % len(symbols)] = -np.inf
        matrices[field] = pd.DataFrame(
            values,
            index=dates,
            columns=symbols,
            dtype=float,
        )
    return matrices, mask


def _outside_non_nan_count(frame: pd.DataFrame, mask: pd.DataFrame) -> int:
    return int(frame.where(~mask).notna().to_numpy().sum())


def _source_bindings(
    expression_source: bytes, operators_source: bytes
) -> list[dict[str, Any]]:
    values = {
        EXPRESSION_SOURCE_PATH: expression_source,
        OPERATORS_SOURCE_PATH: operators_source,
    }
    return [
        {
            "repository_path": path,
            "raw_sha256": hashlib.sha256(values[path]).hexdigest(),
            "size_bytes": len(values[path]),
        }
        for path in sorted(values)
    ]


def _normalize_binding_rows(
    value: Any,
    *,
    expected_ids: Sequence[str],
    include_ast: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "proof binding inventory mismatch"
        )
    fields = {"binding_id", "absolute_path", "byte_sha256"}
    fields.add("ast_sha256" if include_ast else "semantic_sha256")
    rows = [_exact(raw, fields, "proof binding") for raw in value]
    if [row["binding_id"] for row in rows] != list(expected_ids):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "proof binding ids or order mismatch"
        )
    for row in rows:
        if not str(row["absolute_path"]).startswith("/"):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                "proof binding path must be absolute"
            )
        _sha(row["byte_sha256"], "proof binding byte SHA")
        _sha(
            row["ast_sha256" if include_ast else "semantic_sha256"],
            "proof binding semantic/AST SHA",
        )
    expected = (
        expected_code_bindings_v4_1()
        if include_ast
        else expected_input_bindings_v4_1()
    )
    if canonical_json_bytes_v4_1(rows) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "proof binding identity mismatch"
        )
    return rows


def _validate_source_bindings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "pinned source binding inventory mismatch"
        )
    rows = [
        _exact(raw, {"repository_path", "raw_sha256", "size_bytes"}, "source binding")
        for raw in value
    ]
    if [row["repository_path"] for row in rows] != sorted(PINNED_RUNTIME_SOURCE_HASHES):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "pinned source binding paths mismatch"
        )
    for row in rows:
        if (
            row["raw_sha256"]
            != PINNED_RUNTIME_SOURCE_HASHES[row["repository_path"]]
            or type(row["size_bytes"]) is not int
            or row["size_bytes"]
            != PINNED_RUNTIME_SOURCE_SIZES[row["repository_path"]]
        ):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                "pinned source binding identity mismatch"
            )
    return rows


def _fixture_descriptor(
    matrices: Mapping[str, pd.DataFrame], mask: pd.DataFrame
) -> dict[str, Any]:
    matrix_rows = [
        {"field": field, **matrix_hash_descriptor_v4_1(matrices[field])}
        for field in sorted(matrices)
    ]
    return {
        "fixture_id": "aquant-operator-differential-adversarial.v1",
        "field_count": len(matrix_rows),
        "fields": matrix_rows,
        "eligibility_mask": matrix_hash_descriptor_v4_1(mask.astype(float)),
        "adversarial_properties": [
            "changing_pit_eligibility",
            "cross_section_ties",
            "division_by_zero",
            "negative_infinity",
            "not_a_number",
            "positive_infinity",
            "ts_mean_window_200",
            "zero_values",
        ],
    }


def _finite_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.to_numpy(dtype=np.float64, copy=False)
    return {
        "reference_finite_count": int(np.isfinite(values).sum()),
        "reference_nan_count": int(np.isnan(values).sum()),
        "reference_posinf_count": int(np.isposinf(values).sum()),
        "reference_neginf_count": int(np.isneginf(values).sum()),
    }


def _ts_mean_windows(ideas: Sequence[Mapping[str, Any]]) -> list[int]:
    windows: set[int] = set()

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        if node.get("kind") == "call" and node.get("function") == "ts_mean":
            arguments = node.get("arguments")
            if (
                not isinstance(arguments, list)
                or len(arguments) != 2
                or not isinstance(arguments[1], Mapping)
                or arguments[1].get("kind") != "constant"
                or type(arguments[1].get("value")) is not int
            ):
                raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                    "bound ts_mean window is malformed"
                )
            windows.add(arguments[1]["value"])
        for key in ("operand", "left", "right"):
            visit(node.get(key))
        arguments = node.get("arguments")
        if isinstance(arguments, list):
            for argument in arguments:
                visit(argument)

    for idea in ideas:
        visit(idea.get("normalized_expression_ast"))
    return sorted(windows)


def _build_comparison_row(
    *,
    idea: Mapping[str, Any],
    matrices: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    runtime: _PinnedRuntime,
) -> dict[str, Any]:
    expression = _text(idea.get("expression"), "idea expression")
    masked_inputs = {
        field: matrices[field].astype(float).where(mask)
        for field in idea.get("input_fields", [])
    }
    if set(masked_inputs) != set(idea.get("input_fields", [])):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "idea input field binding mismatch"
        )
    reference_expression = runtime.expression_type(expression)
    envelope = _PitEnvelopeOperators(runtime.operators, mask)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw_reference = reference_expression.evaluate(
            masked_inputs,
            operators_module=runtime.operators,
        )
        enveloped_reference = reference_expression.evaluate(
            masked_inputs,
            operators_module=envelope,
        ).where(mask)
        local = evaluate_pinned_idea_v4_1(
            idea=idea,
            matrices=masked_inputs,
            eligibility_mask=mask,
        )
    reference_descriptor = matrix_hash_descriptor_v4_1(enveloped_reference)
    local_descriptor = matrix_hash_descriptor_v4_1(local)
    raw_masked_descriptor = matrix_hash_descriptor_v4_1(raw_reference.where(mask))
    if reference_descriptor != local_descriptor:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"operator differential mismatch: {idea.get('candidate_id')}"
        )
    local_outside = _outside_non_nan_count(local, mask)
    if local_outside != 0:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "local evaluator emitted a value outside the PIT mask"
        )
    return {
        "candidate_id": _text(idea.get("candidate_id"), "candidate id"),
        "name": _text(idea.get("name"), "candidate name"),
        "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        "normalized_ast_sha256": _sha(
            idea.get("full_candidate_normalized_ast_sha256"),
            "normalized AST SHA",
        ),
        "source_definition_sha256": _sha(
            idea.get("source_definition_sha256"), "source definition SHA"
        ),
        "catalog_definition_sha256": _sha(
            idea.get("catalog_definition_sha256"), "catalog definition SHA"
        ),
        "mapping_semantic_sha256": _sha(
            idea.get("mapping_semantic_sha256"), "mapping semantic SHA"
        ),
        "reference_masked_matrix_sha256": reference_descriptor["matrix_sha256"],
        "local_matrix_sha256": local_descriptor["matrix_sha256"],
        "raw_reference_masked_matrix_sha256": raw_masked_descriptor["matrix_sha256"],
        "raw_reference_matches_enveloped": (
            raw_masked_descriptor["matrix_sha256"]
            == reference_descriptor["matrix_sha256"]
        ),
        "local_outside_mask_non_nan_count": local_outside,
        "match": True,
        **_finite_counts(enveloped_reference),
    }


def _build_operator_probe_row(
    *,
    probe_id: str,
    expression: str,
    matrices: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    runtime: _PinnedRuntime,
) -> dict[str, Any]:
    reference_expression = runtime.expression_type(expression)
    envelope = _PitEnvelopeOperators(runtime.operators, mask)
    masked_inputs = {
        field: frame.astype(float).where(mask) for field, frame in matrices.items()
    }
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reference = reference_expression.evaluate(
            masked_inputs,
            operators_module=envelope,
        ).where(mask)
        local = evaluate_expression_v4_1(
            expression=expression,
            matrices=masked_inputs,
            eligibility_mask=mask,
        )
    reference_descriptor = matrix_hash_descriptor_v4_1(reference)
    local_descriptor = matrix_hash_descriptor_v4_1(local)
    if reference_descriptor != local_descriptor:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"operator probe differential mismatch: {probe_id}"
        )
    outside = _outside_non_nan_count(local, mask)
    if outside != 0:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"operator probe emitted outside the PIT mask: {probe_id}"
        )
    return {
        "probe_id": probe_id,
        "expression": expression,
        "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        "reference_matrix_sha256": reference_descriptor["matrix_sha256"],
        "local_matrix_sha256": local_descriptor["matrix_sha256"],
        "local_outside_mask_non_nan_count": outside,
        "match": True,
        **_finite_counts(reference),
    }


def _candidate_manifest(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "candidate_id",
        "name",
        "expression_sha256",
        "normalized_ast_sha256",
        "source_definition_sha256",
        "catalog_definition_sha256",
        "mapping_semantic_sha256",
    )
    return [{field: row[field] for field in fields} for row in rows]


def _differential_result_sha256(
    *,
    rows: Sequence[Mapping[str, Any]],
    operator_probes: Sequence[Mapping[str, Any]],
    raw_reference_divergence_count: int,
    reference_inf_count: int,
) -> str:
    return semantic_sha256_v4_1(
        {
            "rows": [copy.deepcopy(dict(row)) for row in rows],
            "operator_probes": [
                copy.deepcopy(dict(row)) for row in operator_probes
            ],
            "raw_reference_divergence_count": raw_reference_divergence_count,
            "reference_inf_count": reference_inf_count,
        }
    )


def build_operator_runtime_equivalence_proof_v4_1(
    *,
    cycle_id: str,
    bound_ideas: Sequence[Mapping[str, Any]],
    expression_source: bytes,
    operators_source: bytes,
    input_bindings: Sequence[Mapping[str, Any]],
    code_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if cycle_id != EXPECTED_CYCLE_ID:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof cycle identity mismatch"
        )
    ideas = [copy.deepcopy(dict(item)) for item in bound_ideas]
    if len(ideas) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof requires the exact 37 bound ideas"
        )
    candidate_ids = [_text(item.get("candidate_id"), "candidate id") for item in ideas]
    if candidate_ids != sorted(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof candidate ids must be exact, sorted, and distinct"
        )
    normalized_inputs = _normalize_binding_rows(
        list(input_bindings),
        expected_ids=REQUIRED_INPUT_BINDING_IDS,
        include_ast=False,
    )
    normalized_code = _normalize_binding_rows(
        list(code_bindings),
        expected_ids=REQUIRED_CODE_BINDING_IDS,
        include_ast=True,
    )
    runtime = load_pinned_runtime_v4_1(
        expression_source=expression_source,
        operators_source=operators_source,
    )
    matrices, mask = build_differential_fixture_v4_1()
    rows = [
        _build_comparison_row(
            idea=idea,
            matrices=matrices,
            mask=mask,
            runtime=runtime,
        )
        for idea in ideas
    ]
    operator_probes = [
        _build_operator_probe_row(
            probe_id=probe_id,
            expression=expression,
            matrices=matrices,
            mask=mask,
            runtime=runtime,
        )
        for probe_id, expression in OPERATOR_PROBE_EXPRESSIONS
    ]
    raw_divergence_count = sum(
        row["raw_reference_matches_enveloped"] is False for row in rows
    )
    if raw_divergence_count <= 0:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "differential fixture did not distinguish the raw unmasked runtime"
        )
    windows = _ts_mean_windows(ideas)
    if windows != list(EXPECTED_TS_MEAN_WINDOWS):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "exact candidate set has unexpected ts_mean window coverage"
        )
    manifest_sha256 = semantic_sha256_v4_1(_candidate_manifest(rows))
    if manifest_sha256 != EXPECTED_BOUND_IDEA_MANIFEST_SHA256:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "exact bound-idea manifest identity mismatch"
        )
    reference_inf_count = sum(
        row["reference_posinf_count"] + row["reference_neginf_count"]
        for row in operator_probes
    )
    if reference_inf_count <= 0:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "differential fixture did not exercise infinite outputs"
        )
    differential_result_sha256 = _differential_result_sha256(
        rows=rows,
        operator_probes=operator_probes,
        raw_reference_divergence_count=raw_divergence_count,
        reference_inf_count=reference_inf_count,
    )
    if differential_result_sha256 != EXPECTED_DIFFERENTIAL_RESULT_SHA256:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator differential result identity mismatch"
        )
    payload = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "claim_scope": CLAIM_SCOPE,
        "pinned_commit": PINNED_COMMIT,
        "source_bindings": _source_bindings(expression_source, operators_source),
        "input_bindings": normalized_inputs,
        "code_bindings": normalized_code,
        "runtime_environment": runtime_environment_descriptor_v4_1(),
        "fixture": _fixture_descriptor(matrices, mask),
        "candidate_count": EXPECTED_PINNED_IDEA_COUNT,
        "candidate_order_sha256": semantic_sha256_v4_1(candidate_ids),
        "bound_idea_manifest_sha256": manifest_sha256,
        "operator_probe_count": len(OPERATOR_PROBE_EXPRESSIONS),
        "operator_probes": operator_probes,
        "reference_runtime_contract": (
            "exact_pinned_factor_expression_and_operators_with_explicit_pit_wrappers"
        ),
        "reference_outputs_remasked_before_comparison": True,
        "raw_reference_divergence_count": raw_divergence_count,
        "reference_inf_count": reference_inf_count,
        "ts_mean_windows": windows,
        "differential_all_match": True,
        "differential_result_sha256": differential_result_sha256,
        "rows": rows,
        "side_effects": dict(SIDE_EFFECT_FIELDS),
        **AUTHORITY_FIELDS,
    }
    return validate_operator_runtime_equivalence_proof_v4_1(
        _seal(payload, "proof_semantic_sha256")
    )


def _validate_authority(payload: Mapping[str, Any]) -> None:
    for field, expected in AUTHORITY_FIELDS.items():
        if payload.get(field) is not expected:
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                f"operator proof authority mismatch: {field}"
            )
    if payload.get("side_effects") != SIDE_EFFECT_FIELDS:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof side effects must all remain false"
        )


def validate_operator_runtime_equivalence_proof_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "claim_scope",
        "pinned_commit",
        "source_bindings",
        "input_bindings",
        "code_bindings",
        "runtime_environment",
        "fixture",
        "candidate_count",
        "candidate_order_sha256",
        "bound_idea_manifest_sha256",
        "operator_probe_count",
        "operator_probes",
        "reference_runtime_contract",
        "reference_outputs_remasked_before_comparison",
        "raw_reference_divergence_count",
        "reference_inf_count",
        "ts_mean_windows",
        "differential_all_match",
        "differential_result_sha256",
        "rows",
        "side_effects",
        "proof_semantic_sha256",
        *AUTHORITY_FIELDS,
    }
    payload = _exact(value, fields, "operator runtime equivalence proof")
    _validate_self_hash(payload, "proof_semantic_sha256", "operator proof")
    if (
        payload["schema_version"] != PROOF_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["cycle_id"] != EXPECTED_CYCLE_ID
        or payload["claim_scope"] != CLAIM_SCOPE
        or payload["pinned_commit"] != PINNED_COMMIT
        or payload["candidate_count"] != EXPECTED_PINNED_IDEA_COUNT
        or payload["reference_runtime_contract"]
        != "exact_pinned_factor_expression_and_operators_with_explicit_pit_wrappers"
        or payload["reference_outputs_remasked_before_comparison"] is not True
        or payload["differential_all_match"] is not True
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof identity mismatch"
        )
    _sha(payload["candidate_order_sha256"], "candidate order SHA")
    _validate_source_bindings(payload["source_bindings"])
    _normalize_binding_rows(
        payload["input_bindings"],
        expected_ids=REQUIRED_INPUT_BINDING_IDS,
        include_ast=False,
    )
    _normalize_binding_rows(
        payload["code_bindings"],
        expected_ids=REQUIRED_CODE_BINDING_IDS,
        include_ast=True,
    )
    _validate_runtime_environment(payload["runtime_environment"])
    fixture = payload["fixture"]
    expected_matrices, expected_mask = build_differential_fixture_v4_1()
    expected_fixture = _fixture_descriptor(expected_matrices, expected_mask)
    if canonical_json_bytes_v4_1(fixture) != canonical_json_bytes_v4_1(
        expected_fixture
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof fixture differs from exact recomputation"
        )
    windows = payload["ts_mean_windows"]
    if (
        not isinstance(windows, list)
        or any(type(window) is not int for window in windows)
        or windows != list(EXPECTED_TS_MEAN_WINDOWS)
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof ts_mean window coverage mismatch"
        )
    probe_fields = {
        "probe_id",
        "expression",
        "expression_sha256",
        "reference_matrix_sha256",
        "local_matrix_sha256",
        "local_outside_mask_non_nan_count",
        "match",
        "reference_finite_count",
        "reference_nan_count",
        "reference_posinf_count",
        "reference_neginf_count",
    }
    probes = payload["operator_probes"]
    if (
        type(payload["operator_probe_count"]) is not int
        or payload["operator_probe_count"] != len(OPERATOR_PROBE_EXPRESSIONS)
        or not isinstance(probes, list)
        or len(probes) != len(OPERATOR_PROBE_EXPRESSIONS)
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof probe inventory mismatch"
        )
    normalized_probes = [
        _exact(raw, probe_fields, "operator proof probe") for raw in probes
    ]
    if [
        (row["probe_id"], row["expression"]) for row in normalized_probes
    ] != list(OPERATOR_PROBE_EXPRESSIONS):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof probe definitions mismatch"
        )
    for row in normalized_probes:
        _text(row["probe_id"], "operator probe id")
        _text(row["expression"], "operator probe expression")
        if (
            row["expression_sha256"]
            != hashlib.sha256(row["expression"].encode("utf-8")).hexdigest()
            or row["reference_matrix_sha256"] != row["local_matrix_sha256"]
            or row["local_outside_mask_non_nan_count"] != 0
            or row["match"] is not True
            or any(
                type(row[field]) is not int or row[field] < 0
                for field in (
                    "reference_finite_count",
                    "reference_nan_count",
                    "reference_posinf_count",
                    "reference_neginf_count",
                )
            )
            or sum(
                row[field]
                for field in (
                    "reference_finite_count",
                    "reference_nan_count",
                    "reference_posinf_count",
                    "reference_neginf_count",
                )
            )
            != EXPECTED_FIXTURE_CELL_COUNT
        ):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                "operator proof probe differential mismatch"
            )
        for field in (
            "expression_sha256",
            "reference_matrix_sha256",
            "local_matrix_sha256",
        ):
            _sha(row[field], f"operator probe {field}")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_PINNED_IDEA_COUNT:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof row inventory mismatch"
        )
    row_fields = {
        "candidate_id",
        "name",
        "expression_sha256",
        "normalized_ast_sha256",
        "source_definition_sha256",
        "catalog_definition_sha256",
        "mapping_semantic_sha256",
        "reference_masked_matrix_sha256",
        "local_matrix_sha256",
        "raw_reference_masked_matrix_sha256",
        "raw_reference_matches_enveloped",
        "local_outside_mask_non_nan_count",
        "match",
        "reference_finite_count",
        "reference_nan_count",
        "reference_posinf_count",
        "reference_neginf_count",
    }
    normalized_rows = [_exact(row, row_fields, "operator proof row") for row in rows]
    candidate_ids = [
        _text(row["candidate_id"], "operator candidate id")
        for row in normalized_rows
    ]
    for row in normalized_rows:
        _text(row["name"], "operator candidate name")
    if (
        candidate_ids != sorted(candidate_ids)
        or len(set(candidate_ids)) != EXPECTED_PINNED_IDEA_COUNT
        or semantic_sha256_v4_1(candidate_ids) != payload["candidate_order_sha256"]
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof candidate order mismatch"
        )
    if (
        _sha(payload["bound_idea_manifest_sha256"], "bound idea manifest SHA")
        != EXPECTED_BOUND_IDEA_MANIFEST_SHA256
        or semantic_sha256_v4_1(_candidate_manifest(normalized_rows))
        != EXPECTED_BOUND_IDEA_MANIFEST_SHA256
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof bound-idea manifest mismatch"
        )
    for row in normalized_rows:
        for field in row_fields:
            if field.endswith("sha256"):
                _sha(row[field], f"operator row {field}")
        if (
            row["reference_masked_matrix_sha256"] != row["local_matrix_sha256"]
            or row["local_outside_mask_non_nan_count"] != 0
            or row["match"] is not True
            or not isinstance(row["raw_reference_matches_enveloped"], bool)
            or any(
                type(row[field]) is not int or row[field] < 0
                for field in (
                    "reference_finite_count",
                    "reference_nan_count",
                    "reference_posinf_count",
                    "reference_neginf_count",
                )
            )
            or sum(
                row[field]
                for field in (
                    "reference_finite_count",
                    "reference_nan_count",
                    "reference_posinf_count",
                    "reference_neginf_count",
                )
            )
            != EXPECTED_FIXTURE_CELL_COUNT
        ):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                "operator proof row differential mismatch"
            )
    divergence = sum(
        row["raw_reference_matches_enveloped"] is False for row in normalized_rows
    )
    if (
        type(payload["raw_reference_divergence_count"]) is not int
        or payload["raw_reference_divergence_count"] <= 0
        or payload["raw_reference_divergence_count"] != divergence
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "raw-reference PIT-envelope divergence accounting mismatch"
        )
    reference_inf_count = sum(
        row["reference_posinf_count"] + row["reference_neginf_count"]
        for row in normalized_probes
    )
    if (
        type(payload["reference_inf_count"]) is not int
        or payload["reference_inf_count"] <= 0
        or payload["reference_inf_count"] != reference_inf_count
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator proof infinite-output accounting mismatch"
        )
    if (
        _sha(
            payload["differential_result_sha256"],
            "operator differential result SHA",
        )
        != EXPECTED_DIFFERENTIAL_RESULT_SHA256
        or _differential_result_sha256(
            rows=normalized_rows,
            operator_probes=normalized_probes,
            raw_reference_divergence_count=payload[
                "raw_reference_divergence_count"
            ],
            reference_inf_count=payload["reference_inf_count"],
        )
        != EXPECTED_DIFFERENTIAL_RESULT_SHA256
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator differential result manifest mismatch"
        )
    _validate_authority(payload)
    return payload


def build_readback_report_v4_1(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(artifacts) != {PROOF_FILENAME}:
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator readback requires exactly one proof artifact"
        )
    proof = validate_operator_runtime_equivalence_proof_v4_1(
        artifacts[PROOF_FILENAME]
    )
    bindings = [copy.deepcopy(dict(item)) for item in artifact_bindings]
    proof_raw = canonical_file_bytes_v4_1(proof)
    expected_bindings = [
        {
            "filename": PROOF_FILENAME,
            "byte_sha256": hashlib.sha256(proof_raw).hexdigest(),
            "size_bytes": len(proof_raw),
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    ]
    if canonical_json_bytes_v4_1(bindings) != canonical_json_bytes_v4_1(
        expected_bindings
    ):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator readback proof binding identity mismatch"
        )
    payload = {
        "schema_version": READBACK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": proof["cycle_id"],
        "run_id": _text(run_id, "run_id"),
        "readiness": READINESS,
        "accepted": True,
        "artifact_bindings": bindings,
        "proof_semantic_sha256": proof["proof_semantic_sha256"],
        "side_effects": dict(SIDE_EFFECT_FIELDS),
        **AUTHORITY_FIELDS,
    }
    return _seal(payload, "report_semantic_sha256")


def validate_readback_report_v4_1(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "readiness",
        "accepted",
        "artifact_bindings",
        "proof_semantic_sha256",
        "side_effects",
        "report_semantic_sha256",
        *AUTHORITY_FIELDS,
    }
    payload = _exact(value, fields, "operator readback")
    _validate_self_hash(payload, "report_semantic_sha256", "operator readback")
    expected = build_readback_report_v4_1(
        run_id=payload["run_id"],
        artifacts=artifacts,
        artifact_bindings=artifact_bindings,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            "operator readback differs from exact recomputation"
        )
    _validate_authority(payload)
    return payload


def build_private_bundle_contract_v4_1(
    *, expected_proof: Mapping[str, Any]
) -> private_io.PrivateBundleContract:
    expected = validate_operator_runtime_equivalence_proof_v4_1(expected_proof)

    def validate_artifact(filename: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if filename == PROOF_FILENAME:
            normalized = validate_operator_runtime_equivalence_proof_v4_1(value)
            if canonical_json_bytes_v4_1(normalized) != canonical_json_bytes_v4_1(
                expected
            ):
                raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                    "operator proof differs from expected bytes"
                )
            return normalized
        if filename == READBACK_FILENAME:
            return copy.deepcopy(dict(value))
        raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
            f"unexpected operator proof artifact: {filename}"
        )

    def validate_complete(
        values: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        if set(values) != set(BUNDLE_FILENAMES):
            raise FactorGovernanceOperatorRuntimeEquivalenceV4_1Error(
                "operator proof bundle inventory mismatch"
            )
        proof = validate_operator_runtime_equivalence_proof_v4_1(
            values[PROOF_FILENAME]
        )
        report = copy.deepcopy(dict(values[READBACK_FILENAME]))
        bindings = report.get("artifact_bindings")
        validate_readback_report_v4_1(
            report,
            artifacts={PROOF_FILENAME: proof},
            artifact_bindings=bindings,
        )
        return {PROOF_FILENAME: proof, READBACK_FILENAME: report}

    return private_io.PrivateBundleContract(
        root_suffix=PRIVATE_ROOT_SUFFIX,
        input_filenames=BUNDLE_INPUT_FILENAMES,
        readback_report_filename=READBACK_FILENAME,
        canonicalize=canonical_file_bytes_v4_1,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_readback_report_v4_1,
    )


__all__ = [
    "AUTHORITY_FIELDS",
    "BUNDLE_FILENAMES",
    "CLAIM_SCOPE",
    "EXPRESSION_SOURCE_PATH",
    "EXPECTED_BOUND_IDEA_MANIFEST_SHA256",
    "EXPECTED_CYCLE_ID",
    "EXPECTED_DIFFERENTIAL_RESULT_SHA256",
    "EXPECTED_INPUT_BINDING_SPECS",
    "EXPECTED_TS_MEAN_WINDOWS",
    "FactorGovernanceOperatorRuntimeEquivalenceV4_1Error",
    "OPERATORS_SOURCE_PATH",
    "OPERATOR_PROBE_EXPRESSIONS",
    "PINNED_RUNTIME_SOURCE_HASHES",
    "PINNED_RUNTIME_SOURCE_SIZES",
    "PRIVATE_ROOT_SUFFIX",
    "PROOF_FILENAME",
    "PROOF_SCHEMA_VERSION",
    "READBACK_FILENAME",
    "READBACK_SCHEMA_VERSION",
    "REQUIRED_CODE_BINDING_IDS",
    "REQUIRED_INPUT_BINDING_IDS",
    "SIDE_EFFECT_FIELDS",
    "build_differential_fixture_v4_1",
    "build_operator_runtime_equivalence_proof_v4_1",
    "build_private_bundle_contract_v4_1",
    "build_readback_report_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "expected_code_bindings_v4_1",
    "expected_input_bindings_v4_1",
    "load_pinned_runtime_v4_1",
    "runtime_environment_descriptor_v4_1",
    "validate_operator_runtime_equivalence_proof_v4_1",
    "validate_readback_report_v4_1",
]
