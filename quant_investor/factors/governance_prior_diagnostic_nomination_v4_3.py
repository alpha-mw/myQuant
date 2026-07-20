"""Pure prior-diagnostic nomination contract for Factor Governance v4.3.

The helpers in this module are deliberately offline and filesystem-free.  They
validate caller-supplied runtime inventories and row-level diagnostic results,
recompute every statistic used by the nomination, and keep every authority and
side-effect flag false.  Publication and all source/Git/Parquet reads belong to
the separate runner and private-bundle layers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd


RUNTIME_BINDING_SCHEMA_VERSION = (
    "factor-governance-prior-diagnostic-runtime-binding.v4.3"
)
NOMINATION_SCHEMA_VERSION = "factor-governance-prior-diagnostic-nomination.v4.3"
DEFINITION_IDENTITY_SCHEMA_VERSION = "factor-governance-definition-identity.v4.3"
PROTOCOL_VERSION = "v4"
RUN_ID = "cn_full_a_v4_3_prior_nomination_20260717_20260717T172132Z"
PURPOSE = "DIAGNOSTIC_ONLY_PRIOR_OUTCOME_INFORMED_NOMINATION"

SNAPSHOT_ID = "20260717T172132Z"
CUTOFF_DATE = "2026-07-17"
ANALYSIS_FLOOR = "2021-06-25"
SESSION_COUNT = 1227
SCOPE_COLUMN_COUNT = 5866
COMPONENT_COUNT = 5502
WARMUP_SESSIONS = 260
HORIZON_SESSIONS = 30
MIN_COMMON_SYMBOLS = 20
MATURITY_COVERAGE_THRESHOLD = 0.60
MULTIPLE_TEST_COUNT = 3

EXPECTED_MONTHLY_DATES = (
    "2022-07-29",
    "2022-08-31",
    "2022-09-30",
    "2022-10-31",
    "2022-11-30",
    "2022-12-30",
    "2023-01-31",
    "2023-02-28",
    "2023-03-31",
    "2023-04-28",
    "2023-05-31",
    "2023-06-30",
    "2023-07-31",
    "2023-08-31",
    "2023-09-28",
    "2023-10-31",
    "2023-11-30",
    "2023-12-29",
    "2024-01-31",
    "2024-02-29",
    "2024-03-29",
    "2024-04-30",
    "2024-05-31",
    "2024-06-28",
    "2024-07-31",
    "2024-08-30",
    "2024-09-30",
    "2024-10-31",
    "2024-11-29",
    "2024-12-31",
    "2025-01-27",
    "2025-02-28",
    "2025-03-31",
    "2025-04-30",
    "2025-05-30",
    "2025-06-30",
    "2025-07-31",
    "2025-08-29",
    "2025-09-30",
    "2025-10-31",
    "2025-11-28",
    "2025-12-31",
    "2026-01-30",
    "2026-02-27",
    "2026-03-31",
    "2026-04-30",
    "2026-05-29",
)

SOURCE_BINDING_EXPECTED = {
    "cutoff_bundle_path": (
        "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
        "v4_1_cycle/factor_v4_1_cutoff_20260718T151837Z"
    ),
    "cutoff_input_binding_byte_sha256": (
        "3a86e2cc4c939ec9a8fbc3e86d32139056c8b14d47fff19cbd2a590a3c16f1f2"
    ),
    "design_source_byte_sha256": (
        "2a000bcc6065af5ea2173b6a44fcdc0640ed3157565485714fb675703843a7ad"
    ),
    "design_source_semantic_sha256": (
        "37cccbf4f5b74c500d54603454030311701a146600ec66205397360ee3afd6a2"
    ),
    "snapshot_id": SNAPSHOT_ID,
    "cutoff_date": CUTOFF_DATE,
    "analysis_floor": ANALYSIS_FLOOR,
    "calendar_semantic_sha256": (
        "214a48e4d6ef78075a27f839798d14343ed9ada6a24e5a22cfb3533931de2a62"
    ),
    "table_inventory_sha256": (
        "d3b281045dfa34af49371a2847877920a062ac077aeee8525d381fc4713a7330"
    ),
    "pit_membership_sha256": (
        "6a4d42edd9581c5cf8ba6a472e3b89bd1747eb1c1731f07cfd53efc949ebf4e1"
    ),
    "component_symbols_newline_sha256": (
        "41ad09c4c6f759714682ffce4420f6cbb9c2bc34827f443bb4f6965485e69721"
    ),
    "session_count": SESSION_COUNT,
    "scope_column_count": SCOPE_COLUMN_COUNT,
    "component_count": COMPONENT_COUNT,
    "eligibility_matrix_sha256": (
        "fa569408b393f15ca53880c3ea27bd98bd8baf5f9a1585fc1ee561b93bf5f9a2"
    ),
    "date_axis_sha256": (
        "79ba77f87207fb7c2d4ab3dca9d24848a2b6e389ef3aa7a42fb63ebfd162a645"
    ),
    "symbol_axis_sha256": (
        "26aa240a381125c4fd064b2342594153fcbdb538f9f7a0498308e7d6e145060d"
    ),
}

MATRIX_BINDINGS_EXPECTED = {
    "adj_close": "b302dd03dd08722825fa261357909f9ed425ef2c1c63d30337e910b94045a3ad",
    "forward_return": "5afdff2828f41c0e00c5c8eb52dd439b4c710135be58a18d7b32186688e632eb",
    "VOL_OF_VOL_20D": "b15fc068126c949840a444e4135e048ed67b2b8877ea6ebd7558e526c1e58b34",
    "MOM_12M_SKIP1M": "9c7ff3f9f543bf711b78793cc85d5d534a73449db5025dadd9bbd27a0cdbd586",
    "EXCESS_MOM_60D": "78ebb65f46dc331892931fadee380ac3a5ed04b7aa1b5dc808b56473ecbb27ad",
}

EXPECTED_DISTRIBUTIONS = (
    ("numpy", "2.4.3", 889, "9aef1256a157056f0a0460bd63dd35e5038d07ca575c377c8e0a1e1b4d6edaf7"),
    ("pandas", "3.0.1", 1515, "a0b3d69ca9204930ed5f1520481556b6cce6f1758b8a39a88e082a63565651ed"),
    ("scipy", "1.17.1", 1419, "88bdfabd7e11a347f236818369bf362bba1dfac6376cdb5344af1e30c7643b75"),
    ("pyarrow", "24.0.0", 749, "d7d41b57b5204254fff5ddeb093d85b0210f3a01ce9c117653e888cff3c085f6"),
)

PROJECT_BINDING_PATHS = (
    "quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_bundle_v4_3.py",
    "scripts/build_factor_v4_3_prior_diagnostic_nomination.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_source_v4_1.py",
    "quant_investor/factors/governance_aquant_no_label_eval_v4_1.py",
)
FIXED_EXISTING_PROJECT_SHA256 = {
    "quant_investor/factors/governance_private_bundle_io.py": (
        "b61c10ea9d5970f8a0708336942492be23e6d7e82d3609f176ad5a6170d85406"
    ),
    "quant_investor/factors/governance_source_readback_v4_1.py": (
        "0d684d2690ddf5c7e75fb0e4a5f24aaad3051d91ce6456ee9d78226e6260db95"
    ),
    "quant_investor/factors/governance_source_v4_1.py": (
        "03ba13ca0875ac2ea18c84fe2305b225e458b56580e3d5983ad38532e9b293c6"
    ),
    "quant_investor/factors/governance_aquant_no_label_eval_v4_1.py": (
        "454af650ddfcbb05df56f4098e596f0f2a80d1d27f20b356c0cf5f6db19ffa71"
    ),
}

ATTEMPT_SPECS = (
    {
        "source_name": "VOL_OF_VOL_20D",
        "operator_semantics": {
            "ordered_nodes": [
                {
                    "operator": "pct_change",
                    "periods": 1,
                    "fill_method": None,
                },
                {
                    "operator": "rolling_std",
                    "window": 5,
                    "min_periods": 5,
                    "center": False,
                    "ddof": 1,
                },
                {
                    "operator": "rolling_std",
                    "window": 20,
                    "min_periods": 20,
                    "center": False,
                    "ddof": 1,
                },
            ],
            "replace_nonfinite_and_pit_remask_after_each_node": True,
            "first_computable_session_1_based": 25,
        },
        "effective_start": "2023-04-28",
        "valid_period_count": 38,
        "raw_mean_ic": -0.08384510537150455,
        "direction": -1,
        "adjusted_mean_ic": 0.08384510537150455,
        "ic_std_ddof1": 0.15717866962052365,
        "icir": 0.5334381921792043,
        "t_statistic": 3.2883338615879425,
        "p_value": 0.0010078224765626695,
        "bonferroni_p": 0.0030234674296880084,
        "coverage": 0.8972804924004522,
    },
    {
        "source_name": "MOM_12M_SKIP1M",
        "operator_semantics": {
            "ordered_nodes": [
                {
                    "operator": "pct_change",
                    "periods": 252,
                    "fill_method": None,
                },
                {"operator": "shift", "periods": 21},
            ],
            "replace_nonfinite_and_pit_remask_after_each_node": True,
            "first_computable_session_1_based": 274,
        },
        "effective_start": "2024-04-30",
        "valid_period_count": 26,
        "raw_mean_ic": -0.05115153205955326,
        "direction": -1,
        "adjusted_mean_ic": 0.05115153205955326,
        "ic_std_ddof1": 0.1297075198156405,
        "icir": 0.3943605747165421,
        "t_statistic": 2.0108522658713133,
        "p_value": 0.044341063583125374,
        "bonferroni_p": 0.13302319074937613,
        "coverage": 0.8917425057043196,
    },
    {
        "source_name": "EXCESS_MOM_60D",
        "operator_semantics": {
            "ordered_nodes": [
                {
                    "operator": "pct_change",
                    "periods": 60,
                    "fill_method": None,
                },
                {
                    "operator": "rolling_mean",
                    "window": 120,
                    "min_periods": 120,
                    "center": False,
                },
                {"operator": "subtract", "left": "pct_change_60d", "right": "rolling_mean_120"},
            ],
            "replace_nonfinite_and_pit_remask_after_each_node": True,
            "first_computable_session_1_based": 180,
        },
        "effective_start": "2023-12-29",
        "valid_period_count": 30,
        "raw_mean_ic": -0.05285507895502,
        "direction": -1,
        "adjusted_mean_ic": 0.05285507895502,
        "ic_std_ddof1": 0.15671597878535928,
        "icir": 0.3372666869369534,
        "t_statistic": 1.8472857233040232,
        "p_value": 0.06470574098146233,
        "bonferroni_p": 0.194117222944387,
        "coverage": 0.8558699852255938,
    },
)

AUTHORITY_FLAGS = {
    "healthy_source_receipt": False,
    "screening_authorized": False,
    "family_bh_authorized": False,
    "maturity_authorized": False,
    "qualification_authorized": False,
    "candidate_qualified": False,
    "admission_authorized": False,
    "production_candidate_authorized": False,
    "production_new_risk_authorized": False,
    "registry_write_authorized": False,
    "production_proposal_authorized": False,
    "apply_authorized": False,
}
SIDE_EFFECT_FLAGS = {
    "registry": False,
    "wal": False,
    "budget": False,
    "production_receipt": False,
    "production_pointer": False,
    "proposal": False,
    "apply": False,
    "provider": False,
    "network": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
    "transaction": False,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")


class FactorGovernancePriorDiagnosticNominationV4_3Error(ValueError):
    """Raised when the v4.3 diagnostic contract fails closed."""


def _error(message: str) -> FactorGovernancePriorDiagnosticNominationV4_3Error:
    return FactorGovernancePriorDiagnosticNominationV4_3Error(message)


def canonical_json_bytes_v4_3(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error(f"value is not canonical finite JSON: {exc}") from exc


def canonical_file_bytes_v4_3(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_3(value) + b"\n"


def semantic_sha256_v4_3(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_3(value)).hexdigest()


def _exact(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(set(fields) - set(payload))
    unknown = sorted(set(payload) - set(fields))
    if missing or unknown:
        raise _error(
            f"{label} fields invalid: missing={','.join(missing) or '-'};"
            f"unknown={','.join(unknown) or '-'}"
        )
    canonical_json_bytes_v4_3(payload)
    return payload


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    if type(value) is not str or _OID_RE.fullmatch(value) is None:
        raise _error(f"{label} must be a lowercase Git OID")
    return value


def _finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise _error(f"{label} must be a finite float")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise _error(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _iso_date(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    return value


def _self_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "artifact_semantic_sha256"}


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["artifact_semantic_sha256"] = semantic_sha256_v4_3(payload)
    return payload


def _validate_self(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    supplied = _sha(payload.get("artifact_semantic_sha256"), f"{label} self SHA")
    if supplied != semantic_sha256_v4_3(_self_payload(payload)):
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return payload


def build_definition_identity_payload_v4_3() -> dict[str, Any]:
    """Return the only canonical identity payload for the nominated factor."""

    return {
        "schema_version": DEFINITION_IDENTITY_SCHEMA_VERSION,
        "name": "pv_low_vol_of_vol_20d",
        "source": "myQuant.alpha158:VOL_OF_VOL_20D",
        "implementation": "alpha158:VOL_OF_VOL_20D",
        "family": "volatility_of_volatility",
        "slot": "primitive:volatility_of_volatility",
        "direction": -1,
        "input_fields": ["adj_close"],
        "catalog_lookback_sessions": 20,
        "operator_semantics": {
            "return_periods": 1,
            "pct_change_fill_method": None,
            "inner_rolling_window_sessions": 5,
            "outer_rolling_window_sessions": 20,
            "rolling_min_periods_equals_window": True,
            "rolling_center": False,
            "std_ddof": 1,
            "node_level_pit_remask": True,
            "first_computable_session_1_based": 25,
        },
        "source_binding": {
            "repository": "myQuant",
            "commit": "c03d36f115c0865602433183a04139677f2f87fb",
            "path": "quant_investor/alpha158.py",
            "blob_oid": "e2ec6e5456c4bf5970de6b020651fc81e6ce1db7",
            "file_sha256": "12e6910c793f570b3699c45eb3157b594577c49f56be64d2c27c6287538a9fc8",
            "value_ast_sha256": "295f0b8580b0b77e749da27274b02bcb6662afeff0c6b7b22245e677ed49aa31",
        },
    }


DEFINITION_IDENTITY_SHA256 = "eb401bc44af71069b87eee44a3c4bb5ba73abe5337dc38a9ab1ac9e6b4bb261a"


def definition_identity_sha256_v4_3() -> str:
    payload = build_definition_identity_payload_v4_3()
    raw = canonical_json_bytes_v4_3(payload)
    if len(raw) != 971:
        raise _error("canonical definition identity length drift")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DEFINITION_IDENTITY_SHA256:
        raise _error("canonical definition identity SHA-256 drift")
    return digest


def validate_definition_identity_payload_v4_3(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_definition_identity_payload_v4_3()
    if canonical_json_bytes_v4_3(value) != canonical_json_bytes_v4_3(expected):
        raise _error("definition identity payload differs from the exact rebuilt payload")
    definition_identity_sha256_v4_3()
    return expected


def validate_definition_identity_v4_3(
    payload: Mapping[str, Any],
    identity: Any,
) -> dict[str, Any]:
    """Validate the exact rebuilt payload together with its computed identity."""

    normalized = validate_definition_identity_payload_v4_3(payload)
    supplied = _sha(identity, "definition identity SHA")
    if supplied != definition_identity_sha256_v4_3():
        raise _error("definition identity SHA differs from the exact rebuilt payload")
    return normalized


_RUNTIME_FIELDS = {
    "schema_version",
    "protocol_version",
    "run_id",
    "purpose",
    "python",
    "distributions",
    "project_bindings",
    "source_binding",
    "matrix_bindings",
    "artifact_semantic_sha256",
}


def _validate_distribution(value: Any, expected: tuple[str, str, int, str], index: int) -> dict[str, Any]:
    name, version, count, inventory_sha = expected
    payload = _exact(
        value,
        {
            "distribution",
            "version",
            "package_prefix",
            "record_path",
            "record_byte_sha256",
            "record_selected_entry_count",
            "unhashed_selected_entry_count",
            "hash_mismatch_count",
            "size_mismatch_count",
            "file_inventory",
            "file_inventory_semantic_sha256",
        },
        f"distribution[{index}]",
    )
    if (
        payload["distribution"] != name
        or payload["version"] != version
        or payload["package_prefix"] != f"{name}/"
        or type(payload["record_path"]) is not str
        or not payload["record_path"].endswith(".dist-info/RECORD")
        or payload["record_selected_entry_count"] != count
        or payload["unhashed_selected_entry_count"] != 0
        or payload["hash_mismatch_count"] != 0
        or payload["size_mismatch_count"] != 0
    ):
        raise _error(f"distribution[{index}] frozen identity/RECORD proof mismatch")
    _sha(payload["record_byte_sha256"], f"distribution[{index}] RECORD SHA")
    inventory = payload["file_inventory"]
    if not isinstance(inventory, list) or len(inventory) != count:
        raise _error(f"distribution[{index}] inventory count mismatch")
    normalized: list[dict[str, Any]] = []
    for row_index, item in enumerate(inventory):
        row = _exact(
            item,
            {"path", "sha256", "size_bytes"},
            f"distribution[{index}].file_inventory[{row_index}]",
        )
        path = row["path"]
        if (
            type(path) is not str
            or not path.startswith(f"{name}/")
            or path.startswith("/")
            or "\x00" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise _error(f"distribution[{index}] inventory path escapes package prefix")
        _sha(row["sha256"], f"distribution[{index}] file SHA")
        _nonnegative_int(row["size_bytes"], f"distribution[{index}] file size")
        normalized.append(copy.deepcopy(row))
    paths = [row["path"] for row in normalized]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise _error(f"distribution[{index}] inventory must be sorted and unique")
    supplied_inventory_sha = _sha(
        payload["file_inventory_semantic_sha256"],
        f"distribution[{index}] inventory semantic SHA",
    )
    if supplied_inventory_sha != semantic_sha256_v4_3(normalized) or supplied_inventory_sha != inventory_sha:
        raise _error(f"distribution[{index}] inventory semantic SHA mismatch")
    return copy.deepcopy(payload)


def validate_prior_diagnostic_runtime_binding_v4_3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(value, _RUNTIME_FIELDS, "prior diagnostic runtime binding")
    if (
        payload["schema_version"] != RUNTIME_BINDING_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["run_id"] != RUN_ID
        or payload["purpose"] != PURPOSE
    ):
        raise _error("runtime binding schema/protocol/run/purpose mismatch")
    python = _exact(
        payload["python"],
        {"implementation", "version", "executable", "executable_sha256"},
        "runtime python",
    )
    if (
        python["implementation"] != "CPython"
        or python["version"] != "3.13.7"
        or type(python["executable"]) is not str
        or not python["executable"].startswith("/")
        or os.path.abspath(python["executable"]) != python["executable"]
        or python["executable_sha256"]
        != "a708f6e9f4803b806b29146c4e0feecfd9bf2d9eb60f3e15b850cd7cb56f200b"
    ):
        raise _error("runtime Python identity mismatch")
    distributions = payload["distributions"]
    if not isinstance(distributions, list) or len(distributions) != len(EXPECTED_DISTRIBUTIONS):
        raise _error("runtime distribution inventory mismatch")
    normalized_distributions = [
        _validate_distribution(item, expected, index)
        for index, (item, expected) in enumerate(zip(distributions, EXPECTED_DISTRIBUTIONS, strict=True))
    ]
    project = payload["project_bindings"]
    if not isinstance(project, list) or len(project) != len(PROJECT_BINDING_PATHS):
        raise _error("project binding inventory count mismatch")
    normalized_project: list[dict[str, Any]] = []
    for index, item in enumerate(project):
        row = _exact(item, {"relative_path", "byte_sha256", "size_bytes"}, f"project binding[{index}]")
        expected_path = PROJECT_BINDING_PATHS[index]
        if row["relative_path"] != expected_path:
            raise _error("project binding paths/order mismatch")
        digest = _sha(row["byte_sha256"], f"project binding[{index}] SHA")
        _positive_int(row["size_bytes"], f"project binding[{index}] size")
        fixed = FIXED_EXISTING_PROJECT_SHA256.get(expected_path)
        if fixed is not None and digest != fixed:
            raise _error(f"fixed project source drift: {expected_path}")
        normalized_project.append(copy.deepcopy(row))
    if [row["relative_path"] for row in normalized_project] != list(PROJECT_BINDING_PATHS):
        raise _error("project binding exact order mismatch")
    source = _exact(
        payload["source_binding"],
        set(SOURCE_BINDING_EXPECTED),
        "source binding",
    )
    if source != SOURCE_BINDING_EXPECTED:
        raise _error("source binding differs from the sealed cutoff contract")
    matrix = _exact(
        payload["matrix_bindings"],
        set(MATRIX_BINDINGS_EXPECTED),
        "matrix bindings",
    )
    if matrix != MATRIX_BINDINGS_EXPECTED:
        raise _error("matrix bindings differ from the independently reproduced baseline")
    _validate_self(payload, "prior diagnostic runtime binding")
    return {
        **copy.deepcopy(payload),
        "distributions": normalized_distributions,
        "project_bindings": normalized_project,
    }


def build_prior_diagnostic_runtime_binding_v4_3(
    *,
    python: Mapping[str, Any],
    distributions: Sequence[Mapping[str, Any]],
    project_bindings: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
    matrix_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_prior_diagnostic_runtime_binding_v4_3(
        _seal(
            {
                "schema_version": RUNTIME_BINDING_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "python": copy.deepcopy(dict(python)),
                "distributions": [copy.deepcopy(dict(row)) for row in distributions],
                "project_bindings": [copy.deepcopy(dict(row)) for row in project_bindings],
                "source_binding": copy.deepcopy(dict(source_binding)),
                "matrix_bindings": copy.deepcopy(dict(matrix_bindings)),
            }
        )
    )


_MATURITY_ROW_FIELDS = {
    "date",
    "finite_signal_count",
    "eligible_signal_count",
    "scope_column_count",
    "coverage_rate",
}
_EVALUATION_ROW_FIELDS = _MATURITY_ROW_FIELDS | {
    "common_symbol_count",
    "rank_ic",
    "exclusion_reason",
}
_ATTEMPT_FIELDS = {
    "source_name",
    "operator_semantics",
    "signal_matrix_sha256",
    "maturity_coverage_rows",
    "effective_start",
    "evaluation_period_rows",
    "valid_period_count",
    "raw_mean_ic",
    "direction",
    "adjusted_mean_ic",
    "ic_std_ddof1",
    "icir",
    "t_statistic",
    "p_value",
    "bonferroni_p",
    "coverage_numerator",
    "coverage_denominator",
    "coverage",
}


def _validate_maturity_row(value: Any, expected_date: str, label: str) -> dict[str, Any]:
    row = _exact(value, _MATURITY_ROW_FIELDS, label)
    if _iso_date(row["date"], f"{label} date") != expected_date:
        raise _error(f"{label} frozen date/order mismatch")
    finite = _nonnegative_int(row["finite_signal_count"], f"{label} finite count")
    eligible = _nonnegative_int(row["eligible_signal_count"], f"{label} eligible count")
    if (
        row["scope_column_count"] != SCOPE_COLUMN_COUNT
        or finite > eligible
        or eligible > SCOPE_COLUMN_COUNT
    ):
        raise _error(f"{label} count bounds mismatch")
    coverage = _finite_float(row["coverage_rate"], f"{label} coverage")
    if coverage != finite / SCOPE_COLUMN_COUNT:
        raise _error(f"{label} coverage must be finite_signal_count/5866")
    return copy.deepcopy(row)


def _recompute_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_ics = [
        row["rank_ic"]
        for row in rows
        if row["exclusion_reason"] is None and row["rank_ic"] is not None
    ]
    if len(rank_ics) < 2:
        raise _error("attempt requires at least two valid RankIC periods")
    series = pd.Series(rank_ics, dtype="float64")
    raw_mean = float(series.mean())
    std = float(series.std(ddof=1))
    if not math.isfinite(raw_mean) or not math.isfinite(std) or std <= 0.0:
        raise _error("attempt RankIC mean/std is unavailable")
    direction = -1 if raw_mean < 0.0 else 1
    adjusted_mean = float(direction * raw_mean)
    icir = float(adjusted_mean / std)
    t_statistic = float(adjusted_mean / (std / math.sqrt(len(series))))
    p_value = float(math.erfc(abs(t_statistic) / math.sqrt(2.0)))
    return {
        "valid_period_count": len(rank_ics),
        "raw_mean_ic": raw_mean,
        "direction": direction,
        "adjusted_mean_ic": adjusted_mean,
        "ic_std_ddof1": std,
        "icir": icir,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "bonferroni_p": float(min(1.0, p_value * MULTIPLE_TEST_COUNT)),
    }


def _validate_attempt(value: Any, expected: Mapping[str, Any], index: int) -> dict[str, Any]:
    attempt = _exact(value, _ATTEMPT_FIELDS, f"attempt[{index}]")
    source_name = expected["source_name"]
    if (
        attempt["source_name"] != source_name
        or attempt["operator_semantics"] != expected["operator_semantics"]
        or attempt["signal_matrix_sha256"] != MATRIX_BINDINGS_EXPECTED[source_name]
    ):
        raise _error(f"attempt[{index}] source/operator/matrix binding mismatch")
    maturity = attempt["maturity_coverage_rows"]
    if not isinstance(maturity, list) or len(maturity) != len(EXPECTED_MONTHLY_DATES):
        raise _error(f"attempt[{index}] must contain exactly 47 maturity rows")
    normalized_maturity = [
        _validate_maturity_row(row, expected_date, f"attempt[{index}].maturity[{row_index}]")
        for row_index, (row, expected_date) in enumerate(zip(maturity, EXPECTED_MONTHLY_DATES, strict=True))
    ]
    qualifying = [row["date"] for row in normalized_maturity if row["coverage_rate"] >= MATURITY_COVERAGE_THRESHOLD]
    if not qualifying or attempt["effective_start"] != qualifying[0] or attempt["effective_start"] != expected["effective_start"]:
        raise _error(f"attempt[{index}] effective start is not the recomputed first maturity date")
    suffix_dates = EXPECTED_MONTHLY_DATES[EXPECTED_MONTHLY_DATES.index(attempt["effective_start"]):]
    evaluation = attempt["evaluation_period_rows"]
    if not isinstance(evaluation, list) or len(evaluation) != len(suffix_dates):
        raise _error(f"attempt[{index}] evaluation rows must equal the maturity suffix")
    maturity_by_date = {row["date"]: row for row in normalized_maturity}
    normalized_evaluation: list[dict[str, Any]] = []
    for row_index, (item, expected_date) in enumerate(zip(evaluation, suffix_dates, strict=True)):
        row = _exact(item, _EVALUATION_ROW_FIELDS, f"attempt[{index}].evaluation[{row_index}]")
        base = {field: row[field] for field in _MATURITY_ROW_FIELDS}
        if base != maturity_by_date[expected_date]:
            raise _error(f"attempt[{index}] evaluation row differs from its maturity row")
        common = _nonnegative_int(row["common_symbol_count"], "evaluation common count")
        if common > row["finite_signal_count"]:
            raise _error(f"attempt[{index}] common count exceeds finite signal count")
        rank_ic = row["rank_ic"]
        reason = row["exclusion_reason"]
        if reason is None:
            value_ic = _finite_float(rank_ic, "evaluation RankIC")
            if common < MIN_COMMON_SYMBOLS or not -1.0 <= value_ic <= 1.0:
                raise _error(f"attempt[{index}] valid RankIC row violates complete-case rules")
        else:
            if reason not in {
                "COMMON_SYMBOL_COUNT_LT_20",
                "SIGNAL_NOT_UNIQUE",
                "FORWARD_NOT_UNIQUE",
                "RANK_IC_NONFINITE",
            } or rank_ic is not None:
                raise _error(f"attempt[{index}] exclusion reason/RankIC mismatch")
        normalized_evaluation.append(copy.deepcopy(row))
    statistics = _recompute_statistics(normalized_evaluation)
    for field, recomputed in statistics.items():
        supplied = attempt[field]
        if type(recomputed) is int:
            if supplied != recomputed or supplied != expected[field]:
                raise _error(f"attempt[{index}] recomputed {field} mismatch")
        else:
            _finite_float(supplied, f"attempt[{index}] {field}")
            if supplied != recomputed or supplied != expected[field]:
                raise _error(f"attempt[{index}] recomputed {field} mismatch")
    numerator = sum(row["finite_signal_count"] for row in normalized_evaluation)
    denominator = len(normalized_evaluation) * SCOPE_COLUMN_COUNT
    coverage = float(numerator / denominator)
    if (
        attempt["coverage_numerator"] != numerator
        or attempt["coverage_denominator"] != denominator
        or attempt["coverage"] != coverage
        or attempt["coverage"] != expected["coverage"]
    ):
        raise _error(f"attempt[{index}] recomputed suffix coverage mismatch")
    return {
        **copy.deepcopy(attempt),
        "maturity_coverage_rows": normalized_maturity,
        "evaluation_period_rows": normalized_evaluation,
    }


def build_prior_diagnostic_attempt_v4_3(
    *,
    source_name: str,
    maturity_coverage_rows: Sequence[Mapping[str, Any]],
    evaluation_period_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_by_name = {row["source_name"]: row for row in ATTEMPT_SPECS}
    if source_name not in expected_by_name:
        raise _error("unknown prior-diagnostic attempt")
    expected = expected_by_name[source_name]
    raw_maturity = [copy.deepcopy(dict(row)) for row in maturity_coverage_rows]
    raw_evaluation = [copy.deepcopy(dict(row)) for row in evaluation_period_rows]
    statistics = _recompute_statistics(raw_evaluation)
    numerator = sum(int(row["finite_signal_count"]) for row in raw_evaluation)
    denominator = len(raw_evaluation) * SCOPE_COLUMN_COUNT
    payload = {
        "source_name": source_name,
        "operator_semantics": copy.deepcopy(expected["operator_semantics"]),
        "signal_matrix_sha256": MATRIX_BINDINGS_EXPECTED[source_name],
        "maturity_coverage_rows": raw_maturity,
        "effective_start": expected["effective_start"],
        "evaluation_period_rows": raw_evaluation,
        **statistics,
        "coverage_numerator": numerator,
        "coverage_denominator": denominator,
        "coverage": float(numerator / denominator),
    }
    return _validate_attempt(payload, expected, list(expected_by_name).index(source_name))


_NOMINATION_FIELDS = {
    "schema_version",
    "protocol_version",
    "run_id",
    "purpose",
    "selection_method",
    "definition_identity",
    "definition_identity_sha256",
    "attempts",
    "winner",
    "winner_candidate",
    "invalid_raw_close_debug",
    "authority",
    "side_effects",
    "runtime_binding_semantic_sha256",
    "artifact_semantic_sha256",
}


def _winner_row(attempt: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "source_name": attempt["source_name"],
        "adjusted_mean_ic": attempt["adjusted_mean_ic"],
        "bonferroni_p": attempt["bonferroni_p"],
        "coverage": attempt["coverage"],
        "valid_period_count": attempt["valid_period_count"],
        "effective_start": attempt["effective_start"],
    }


def _rank_attempts(attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        attempts,
        key=lambda row: (
            -row["adjusted_mean_ic"],
            row["bonferroni_p"],
            -row["coverage"],
            row["source_name"],
        ),
    )
    return [_winner_row(row, index) for index, row in enumerate(ranked, start=1)]


def _winner_candidate() -> dict[str, Any]:
    identity = build_definition_identity_payload_v4_3()
    return {
        "name": identity["name"],
        "source": identity["source"],
        "implementation": identity["implementation"],
        "family": identity["family"],
        "slot": identity["slot"],
        "direction": identity["direction"],
        "input_fields": copy.deepcopy(identity["input_fields"]),
        "catalog_lookback_sessions": identity["catalog_lookback_sessions"],
        "inner_rolling_window_sessions": 5,
        "first_computable_session_1_based": 25,
        "initial_weight": 0,
        "diagnostic_only": True,
        "outcome_informed_selection": True,
        "external_label_independence": False,
        "prior_statistics_nomination_only": True,
    }


INVALID_RAW_CLOSE_DEBUG = {
    "retained": False,
    "hash": "UNCONFIRMED",
    "excluded": True,
    "used_for_choice": False,
    "used_for_evidence": False,
    "reason": "adjusted-price/PIT mismatch",
}


def validate_prior_diagnostic_nomination_v4_3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(value, _NOMINATION_FIELDS, "prior diagnostic nomination")
    if (
        payload["schema_version"] != NOMINATION_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["run_id"] != RUN_ID
        or payload["purpose"] != PURPOSE
    ):
        raise _error("nomination schema/protocol/run/purpose mismatch")
    if payload["selection_method"] != {
        "ordered_attempt_count": 3,
        "outcome_informed": True,
        "external_label_independence": False,
        "diagnostic_only": True,
        "winner_sort": [
            "adjusted_mean_ic_desc",
            "bonferroni_p_asc",
            "coverage_desc",
            "source_name_asc",
        ],
        "multiple_testing": "bonferroni_p_times_3_capped_at_1",
        "spearman_implementation": "pandas.Series.corr(method=spearman,min_periods=20)",
    }:
        raise _error("nomination selection method mismatch")
    identity = validate_definition_identity_v4_3(
        payload["definition_identity"],
        payload["definition_identity_sha256"],
    )
    attempts = payload["attempts"]
    if not isinstance(attempts, list) or len(attempts) != len(ATTEMPT_SPECS):
        raise _error("nomination must contain exactly three ordered attempts")
    normalized_attempts = [
        _validate_attempt(item, expected, index)
        for index, (item, expected) in enumerate(zip(attempts, ATTEMPT_SPECS, strict=True))
    ]
    ranking = _rank_attempts(normalized_attempts)
    if payload["winner"] != {"source_name": "VOL_OF_VOL_20D", "ranking": ranking}:
        raise _error("nomination winner/ranking mismatch")
    if payload["winner_candidate"] != _winner_candidate():
        raise _error("nomination winner candidate contract mismatch")
    if payload["invalid_raw_close_debug"] != INVALID_RAW_CLOSE_DEBUG:
        raise _error("invalid raw-close debug exclusion contract mismatch")
    if payload["authority"] != AUTHORITY_FLAGS or payload["side_effects"] != SIDE_EFFECT_FLAGS:
        raise _error("nomination authority/side-effect flags must remain exact false")
    _sha(payload["runtime_binding_semantic_sha256"], "runtime binding semantic SHA")
    _validate_self(payload, "prior diagnostic nomination")
    return {
        **copy.deepcopy(payload),
        "definition_identity": identity,
        "attempts": normalized_attempts,
    }


def build_prior_diagnostic_nomination_v4_3(
    *,
    attempts: Sequence[Mapping[str, Any]],
    runtime_binding_semantic_sha256: str,
) -> dict[str, Any]:
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        raise _error("attempts must be the exact ordered three-attempt sequence")
    if len(attempts) != len(ATTEMPT_SPECS):
        raise _error("nomination must contain exactly three ordered attempts")
    normalized_attempts = [
        _validate_attempt(row, expected, index)
        for index, (row, expected) in enumerate(
            zip(attempts, ATTEMPT_SPECS, strict=True)
        )
    ]
    ranking = _rank_attempts(normalized_attempts)
    return validate_prior_diagnostic_nomination_v4_3(
        _seal(
            {
                "schema_version": NOMINATION_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "selection_method": {
                    "ordered_attempt_count": 3,
                    "outcome_informed": True,
                    "external_label_independence": False,
                    "diagnostic_only": True,
                    "winner_sort": [
                        "adjusted_mean_ic_desc",
                        "bonferroni_p_asc",
                        "coverage_desc",
                        "source_name_asc",
                    ],
                    "multiple_testing": "bonferroni_p_times_3_capped_at_1",
                    "spearman_implementation": "pandas.Series.corr(method=spearman,min_periods=20)",
                },
                "definition_identity": build_definition_identity_payload_v4_3(),
                "definition_identity_sha256": definition_identity_sha256_v4_3(),
                "attempts": normalized_attempts,
                "winner": {"source_name": "VOL_OF_VOL_20D", "ranking": ranking},
                "winner_candidate": _winner_candidate(),
                "invalid_raw_close_debug": copy.deepcopy(INVALID_RAW_CLOSE_DEBUG),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
                "runtime_binding_semantic_sha256": _sha(
                    runtime_binding_semantic_sha256,
                    "runtime binding semantic SHA",
                ),
            }
        )
    )


def validate_prior_diagnostic_nomination_against_runtime_v4_3(
    nomination: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    normalized_runtime = validate_prior_diagnostic_runtime_binding_v4_3(runtime_binding)
    normalized_nomination = validate_prior_diagnostic_nomination_v4_3(nomination)
    if (
        normalized_nomination["runtime_binding_semantic_sha256"]
        != normalized_runtime["artifact_semantic_sha256"]
        or normalized_nomination["run_id"] != normalized_runtime["run_id"]
    ):
        raise _error("nomination/runtime binding crosslink mismatch")
    return {
        "runtime_binding": normalized_runtime,
        "nomination": normalized_nomination,
    }


__all__ = [
    "ANALYSIS_FLOOR",
    "ATTEMPT_SPECS",
    "AUTHORITY_FLAGS",
    "CUTOFF_DATE",
    "DEFINITION_IDENTITY_SHA256",
    "EXPECTED_DISTRIBUTIONS",
    "EXPECTED_MONTHLY_DATES",
    "FactorGovernancePriorDiagnosticNominationV4_3Error",
    "INVALID_RAW_CLOSE_DEBUG",
    "MATRIX_BINDINGS_EXPECTED",
    "NOMINATION_SCHEMA_VERSION",
    "PROJECT_BINDING_PATHS",
    "PROTOCOL_VERSION",
    "PURPOSE",
    "RUN_ID",
    "RUNTIME_BINDING_SCHEMA_VERSION",
    "SCOPE_COLUMN_COUNT",
    "SIDE_EFFECT_FLAGS",
    "SNAPSHOT_ID",
    "SOURCE_BINDING_EXPECTED",
    "build_definition_identity_payload_v4_3",
    "build_prior_diagnostic_attempt_v4_3",
    "build_prior_diagnostic_nomination_v4_3",
    "build_prior_diagnostic_runtime_binding_v4_3",
    "canonical_file_bytes_v4_3",
    "canonical_json_bytes_v4_3",
    "definition_identity_sha256_v4_3",
    "semantic_sha256_v4_3",
    "validate_definition_identity_payload_v4_3",
    "validate_definition_identity_v4_3",
    "validate_prior_diagnostic_nomination_against_runtime_v4_3",
    "validate_prior_diagnostic_nomination_v4_3",
    "validate_prior_diagnostic_runtime_binding_v4_3",
]
