#!/usr/bin/env python3
"""Build the owner-private, no-label v4.1 A_quant input-resolution bundle.

All paths and hashes are explicit.  The runner reads only the frozen snapshot,
its serving projection, the frozen PIT membership, and the active fundamental
generation.  It never requests labels, returns, providers, registry state,
portfolio state, or execution surfaces.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import governance_aquant_input_resolution_v4_1 as contract  # noqa: E402
from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator  # noqa: E402
from quant_investor.factors import governance_formal_catalog_materialization_v4_1 as formal_materialization  # noqa: E402
from quant_investor.factors import governance_no_label_diagnostic_v4_1 as no_label  # noqa: E402
from quant_investor.factors import governance_operator_runtime_equivalence_v4_1 as operator_proof  # noqa: E402
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_source_v4_1 as source_v4  # noqa: E402
from quant_investor.factors.pit_fundamentals import build_fundamental_metric_matrices  # noqa: E402
from quant_investor.market import fundamental_generation, fundamental_mart  # noqa: E402


EXPECTED_SESSION_COUNT = 1227
EXPECTED_PIT_COUNT = 5866
EXPECTED_COMPONENT_COUNT = 5502
EXPECTED_ELIGIBLE_CELL_COUNT = 6_346_625
EXPECTED_START = "20210625"
EXPECTED_END = "20260717"
FUNDAMENTAL_FIELDS = (
    "fcf_to_price",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_roa",
    "fin_roe",
)
TABLE_COLUMNS = ("ts_code", "trade_date", "close", "vol", "amount")
SERVING_COLUMNS = ("ts_code", "trade_date", "turnover_rate")


class FactorV4_1AquantInputResolutionRunnerError(ValueError):
    """Raised when an explicit v4.1 input-resolution binding is rejected."""


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"{label} must be a lowercase SHA-256"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"{label} must be a lowercase SHA-256"
        ) from exc
    if value.lower() != value:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _absolute(value: Any, label: str) -> Path:
    if type(value) is not str:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"{label} must be an absolute normalized path"
        )
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"{label} must be an absolute normalized path"
        )
    return path


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
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


def _stable_bytes(
    path: Path,
    *,
    expected_sha256: str,
    private: bool = False,
) -> bytes:
    expected = _sha(expected_sha256, f"expected SHA for {path}")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"bound path is unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"bound path is not a regular non-symlink file: {path}"
        )
    if private and (
        before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_nlink != 1
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"private predecessor owner/mode/link check failed: {path}"
        )
    first = path.read_bytes()
    middle = os.lstat(path)
    second = path.read_bytes()
    after = os.lstat(path)
    if (
        _signature(before) != _signature(middle)
        or _signature(middle) != _signature(after)
        or first != second
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"bound file changed across stable readback: {path}"
        )
    if hashlib.sha256(first).hexdigest() != expected:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"bound file SHA mismatch: {path}"
        )
    return first


def _json(raw: bytes, label: str, *, canonical_private: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"bound JSON is invalid: {label}"
        ) from exc
    if not isinstance(value, Mapping):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"bound JSON must be an object: {label}"
        )
    payload = copy.deepcopy(dict(value))
    if canonical_private and raw != contract.canonical_file_bytes_v4_1(payload):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"private predecessor is not canonical JSON: {label}"
        )
    return payload


def _parse_predecessor_bindings(values: Sequence[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw in values:
        parts = raw.split("=", 3)
        if len(parts) != 4:
            raise FactorV4_1AquantInputResolutionRunnerError(
                "predecessor binding must be id=absolute_path=byte_sha256=semantic_sha256"
            )
        binding_id, path, byte_sha, semantic_sha = parts
        if binding_id in rows:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"duplicate predecessor binding: {binding_id}"
            )
        rows[binding_id] = {
            "binding_id": binding_id,
            "path": str(_absolute(path, f"predecessor {binding_id}")),
            "byte_sha256": _sha(byte_sha, f"predecessor {binding_id} byte SHA"),
            "semantic_sha256": _sha(
                semantic_sha, f"predecessor {binding_id} semantic SHA"
            ),
        }
    if set(rows) != set(contract.PREDECESSOR_BINDING_IDS):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "predecessor binding ids differ from the exact formal/no-label/operator inventory"
        )
    return rows


def _parse_file_bindings(
    values: Sequence[str],
    *,
    expected_ids: Sequence[str],
    label: str,
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw in values:
        parts = raw.split("=", 2)
        if len(parts) != 3:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"{label} must be id=absolute_path=byte_sha256"
            )
        binding_id, path, byte_sha = parts
        if binding_id in rows:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"duplicate {label}: {binding_id}"
            )
        rows[binding_id] = {
            "binding_id": binding_id,
            "path": str(_absolute(path, f"{label} {binding_id}")),
            "byte_sha256": _sha(byte_sha, f"{label} {binding_id} SHA"),
        }
    if set(rows) != set(expected_ids):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"{label} ids differ from the exact required inventory"
        )
    return rows


def _inventory_root(
    root: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    if not root.is_dir() or root.is_symlink():
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"inventory root is not a real directory: {root}"
        )
    detailed: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"inventory contains a symlink: {path}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            continue
        raw = path.read_bytes()
        if _signature(metadata) != _signature(os.lstat(path)):
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"inventory file changed during read: {path}"
            )
        relative = path.relative_to(root).as_posix()
        dataset_member = bool(
            path.suffix == ".parquet"
            and all(not part.startswith((".", "_")) for part in Path(relative).parts)
        )
        digest = hashlib.sha256(raw).hexdigest()
        detailed.append(
            {
                "relative_path": relative,
                "size_bytes": len(raw),
                "sha256": digest,
                "hard_link_count": int(metadata.st_nlink),
                "dataset_member": dataset_member,
            }
        )
        artifact_rows.append(
            {
                "relative_path": relative,
                "byte_sha256": digest,
                "size_bytes": len(raw),
                "dataset_member": dataset_member,
            }
        )
    if not artifact_rows or not any(row["dataset_member"] for row in artifact_rows):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"inventory has no Parquet dataset member: {root}"
        )
    observed = hashlib.sha256(
        contract.canonical_json_bytes_v4_1(detailed) + b"\n"
    ).hexdigest()
    if observed != _sha(expected_sha256, f"expected inventory SHA for {root}"):
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"inventory SHA mismatch: {root}"
        )
    return artifact_rows, observed


def _read_predecessors(
    bindings: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    values: dict[str, dict[str, Any]] = {}
    semantic_fields = {
        "formal_catalog": "semantic_sha256",
        "formal_catalog_readback": "report_semantic_sha256",
        "no_label_diagnostic": "diagnostic_semantic_sha256",
        "no_label_readback": "report_semantic_sha256",
        "operator_runtime_equivalence_proof": "proof_semantic_sha256",
        "operator_runtime_equivalence_readback": "report_semantic_sha256",
    }
    for binding_id in contract.PREDECESSOR_BINDING_IDS:
        binding = bindings[binding_id]
        raw = _stable_bytes(
            Path(binding["path"]),
            expected_sha256=binding["byte_sha256"],
            private=True,
        )
        value = _json(raw, binding["path"], canonical_private=True)
        if value.get(semantic_fields[binding_id]) != binding["semantic_sha256"]:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"predecessor semantic SHA mismatch: {binding_id}"
            )
        values[binding_id] = value

    catalog = values["formal_catalog"]
    if (
        catalog.get("schema_version") != "factor-candidate-catalog.v4"
        or len(catalog.get("candidates", [])) != 267
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "formal catalog predecessor identity mismatch"
        )
    no_label_diagnostic = no_label.validate_signal_diagnostic_v4_1(
        values["no_label_diagnostic"]
    )
    proof = operator_proof.validate_operator_runtime_equivalence_proof_v4_1(
        values["operator_runtime_equivalence_proof"]
    )
    operator_proof.validate_readback_report_v4_1(
        values["operator_runtime_equivalence_readback"],
        artifacts={operator_proof.PROOF_FILENAME: proof},
        artifact_bindings=values["operator_runtime_equivalence_readback"].get(
            "artifact_bindings"
        ),
    )
    if (
        no_label_diagnostic.get("cycle_id") != contract.EXPECTED_CYCLE_ID
        or proof.get("cycle_id") != contract.EXPECTED_CYCLE_ID
        or proof.get("operator_runtime_equivalence_verified") is not True
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "no-label/operator predecessor cycle or proof mismatch"
        )
    blocked_status = {
        "alpha_turnover_low_20d": "turnover_data_blocked",
        "alpha_turnover_low_60d": "turnover_data_blocked",
        **{
            name: "fundamental_semantic_blocked"
            for name in contract.RESOLVED_CANDIDATE_NAMES
            if not name.startswith("alpha_turnover_low_")
        },
    }
    diagnostic_by_name = {row["name"]: row for row in no_label_diagnostic["rows"]}
    proof_by_name = {row["name"]: row for row in proof["rows"]}
    catalog_by_name = {row["name"]: row for row in catalog["candidates"]}
    ideas: list[dict[str, Any]] = []
    for name in contract.RESOLVED_CANDIDATE_NAMES:
        expected = contract.EXPECTED_CANDIDATES[name]
        diagnostic_row = diagnostic_by_name.get(name)
        proof_row = proof_by_name.get(name)
        candidate = catalog_by_name.get(name)
        if not all(isinstance(item, Mapping) for item in (diagnostic_row, proof_row, candidate)):
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"predecessor omits exact resolution row: {name}"
            )
        if (
            diagnostic_row.get("status") != blocked_status[name]
            or proof_row.get("match") is not True
            or candidate.get("implementation") != "aquant_expression_ast.v1"
            or candidate.get("input_fields") != expected["input_fields"]
            or candidate.get("definition_sha256")
            != expected["catalog_definition_sha256"]
        ):
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"predeclared blocker predecessor drift: {name}"
            )
        for field in (
            "source_definition_sha256",
            "catalog_definition_sha256",
            "mapping_semantic_sha256",
        ):
            if proof_row.get(field) != expected[field]:
                raise FactorV4_1AquantInputResolutionRunnerError(
                    f"operator proof identity drift for {name}: {field}"
                )
        if proof_row.get("normalized_ast_sha256") != expected["normalized_ast_sha256"]:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"operator proof AST drift: {name}"
            )
        expression = candidate.get("expression")
        tree = evaluator.normalize_expression_ast_v4_1(expression)
        if evaluator.semantic_sha256_v4_1(tree) != expected["normalized_ast_sha256"]:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"formal expression AST drift: {name}"
            )
        ideas.append(
            {
                "candidate_id": f"aquant:{evaluator.PINNED_COMMIT}:{name}",
                "name": name,
                "expression": expression,
                "normalized_expression_ast": tree,
                "input_fields": list(expected["input_fields"]),
                "source_definition_sha256": expected["source_definition_sha256"],
                "full_candidate_normalized_ast_sha256": expected[
                    "normalized_ast_sha256"
                ],
                "catalog_definition_sha256": expected[
                    "catalog_definition_sha256"
                ],
                "mapping_semantic_sha256": expected["mapping_semantic_sha256"],
            }
        )
    rows = [copy.deepcopy(dict(bindings[binding_id])) for binding_id in contract.PREDECESSOR_BINDING_IDS]
    return rows, {"values": values, "bound_ideas": ideas}


def _read_code_bindings(
    bindings: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    expected_paths = {
        "aquant_no_label_evaluator": Path(evaluator.__file__).resolve(),
        "formal_catalog_materialization": Path(formal_materialization.__file__).resolve(),
        "fundamental_generation": Path(fundamental_generation.__file__).resolve(),
        "fundamental_mart": Path(fundamental_mart.__file__).resolve(),
        "no_label_diagnostic": Path(no_label.__file__).resolve(),
        "operator_runtime_equivalence": Path(operator_proof.__file__).resolve(),
        "pit_fundamentals": Path(sys.modules[build_fundamental_metric_matrices.__module__].__file__).resolve(),
        "private_bundle_io": Path(private_io.__file__).resolve(),
        "resolution_builder": Path(__file__).resolve(),
        "resolution_contract": Path(contract.__file__).resolve(),
        "source_v4_1": Path(source_v4.__file__).resolve(),
    }
    rows: list[dict[str, Any]] = []
    for binding_id in contract.CODE_BINDING_IDS:
        binding = bindings[binding_id]
        path = Path(binding["path"])
        if path != expected_paths[binding_id]:
            raise FactorV4_1AquantInputResolutionRunnerError(
                f"code binding path substitution: {binding_id}"
            )
        _stable_bytes(path, expected_sha256=binding["byte_sha256"])
        rows.append(
            {
                **copy.deepcopy(dict(binding)),
                "semantic_sha256": contract.semantic_sha256_v4_1(
                    {
                        "binding_id": binding_id,
                        "path": str(path),
                        "byte_sha256": binding["byte_sha256"],
                    }
                ),
            }
        )
    return rows


def _read_json_source(binding: Mapping[str, str]) -> tuple[dict[str, Any], str]:
    raw = _stable_bytes(
        Path(binding["path"]), expected_sha256=binding["byte_sha256"]
    )
    value = _json(raw, binding["path"])
    return value, contract.semantic_sha256_v4_1(value)


def _source_semantic(path: Path, byte_sha256: str) -> str:
    metadata = os.lstat(path)
    return contract.semantic_sha256_v4_1(
        {
            "path": str(path),
            "byte_sha256": byte_sha256,
            "size_bytes": int(metadata.st_size),
        }
    )


def _dataset_paths(
    root: Path, inventory: Sequence[Mapping[str, Any]]
) -> list[str]:
    paths = [
        str(root / row["relative_path"])
        for row in inventory
        if row["dataset_member"] is True
    ]
    if not paths:
        raise FactorV4_1AquantInputResolutionRunnerError(
            f"dataset inventory is empty: {root}"
        )
    return paths


def _table_sessions(
    root: Path, inventory: Sequence[Mapping[str, Any]]
) -> pd.DatetimeIndex:
    dataset = ds.dataset(_dataset_paths(root, inventory), format="parquet")
    if "trade_date" not in dataset.schema.names:
        raise FactorV4_1AquantInputResolutionRunnerError(
            "strict table lacks trade_date"
        )
    table = dataset.to_table(
        columns=["trade_date"],
        filter=(ds.field("trade_date") >= EXPECTED_START)
        & (ds.field("trade_date") <= EXPECTED_END),
    )
    sessions = pd.DatetimeIndex(
        pd.to_datetime(table.column("trade_date").to_pylist(), format="%Y%m%d")
    ).unique().sort_values()
    if (
        len(sessions) != EXPECTED_SESSION_COUNT
        or sessions[0].strftime("%Y%m%d") != EXPECTED_START
        or sessions[-1].strftime("%Y%m%d") != EXPECTED_END
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "strict table session axis differs from exact frozen cycle"
        )
    return sessions


def _build_eligibility_mask(
    *,
    sessions: pd.DatetimeIndex,
    membership_path: Path,
    membership_sha256: str,
    manifest: Mapping[str, Any],
    components: Mapping[str, Any],
) -> pd.DataFrame:
    if (
        manifest.get("canonical_path") != str(membership_path)
        or manifest.get("canonical_sha256") != membership_sha256
        or manifest.get("row_count") != EXPECTED_PIT_COUNT
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "PIT manifest does not bind exact membership"
        )
    records = source_v4.validate_pit_records_v4_1(
        pd.read_parquet(membership_path).to_dict(orient="records")
    )
    if len(records) != EXPECTED_PIT_COUNT:
        raise FactorV4_1AquantInputResolutionRunnerError(
            "PIT membership row count mismatch"
        )
    component_symbols = components.get("full_a")
    if (
        not isinstance(component_symbols, list)
        or len(component_symbols) != EXPECTED_COMPONENT_COUNT
        or component_symbols != sorted(set(component_symbols))
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "component file differs from exact sorted full-A scope"
        )
    symbols = [row["symbol"] for row in records]
    by_symbol = {row["symbol"]: row for row in records}
    positions = {symbol: index for index, symbol in enumerate(symbols)}
    if len(by_symbol) != EXPECTED_PIT_COUNT or not set(component_symbols).issubset(by_symbol):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "component/PIT symbol identity mismatch"
        )
    mask = np.zeros((len(sessions), len(symbols)), dtype=bool)
    for session_index, timestamp in enumerate(sessions):
        session = timestamp.strftime("%Y-%m-%d")
        domain = component_symbols if timestamp == sessions[-1] else symbols
        for symbol in domain:
            record = by_symbol[symbol]
            if record["effective_from"] <= session and (
                record["effective_to"] is None or session < record["effective_to"]
            ):
                mask[session_index, positions[symbol]] = True
    frame = pd.DataFrame(mask, index=sessions, columns=symbols, dtype=bool)
    if int(mask.sum()) != EXPECTED_ELIGIBLE_CELL_COUNT:
        raise FactorV4_1AquantInputResolutionRunnerError(
            "PIT eligibility cell count differs from exact frozen cycle"
        )
    return frame


def _load_market_matrices(
    root: Path,
    inventory: Sequence[Mapping[str, Any]],
    eligibility_mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    dataset = ds.dataset(_dataset_paths(root, inventory), format="parquet")
    if not set(TABLE_COLUMNS).issubset(dataset.schema.names):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "strict table lacks exact market input columns"
        )
    table = dataset.to_table(
        columns=list(TABLE_COLUMNS),
        filter=(ds.field("trade_date") >= EXPECTED_START)
        & (ds.field("trade_date") <= EXPECTED_END),
    )
    raw = table.to_pandas()
    if raw.empty or raw.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1AquantInputResolutionRunnerError(
            "strict table is empty or contains duplicate serving keys"
        )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")

    def pivot(column: str) -> pd.DataFrame:
        return (
            raw.pivot(index="trade_date", columns="ts_code", values=column)
            .reindex(index=eligibility_mask.index, columns=eligibility_mask.columns)
            .astype(float)
            .where(eligibility_mask)
        )

    close = pivot("close")
    volume = pivot("vol")
    amount = pivot("amount")
    vwap = (amount * 10.0).div(volume.where(volume > 0.0))
    vwap = vwap.replace([np.inf, -np.inf], np.nan).where(vwap > 0.0).where(eligibility_mask)
    return {"close": close, "vwap": vwap}


def _load_serving_turnover(
    root: Path,
    inventory: Sequence[Mapping[str, Any]],
    eligibility_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Load turnover strictly from serving and preserve its sparse shape."""

    dataset = ds.dataset(_dataset_paths(root, inventory), format="parquet")
    if not set(SERVING_COLUMNS).issubset(dataset.schema.names):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "strict serving projection lacks turnover_rate"
        )
    table = dataset.to_table(
        columns=list(SERVING_COLUMNS),
        filter=(ds.field("trade_date") >= EXPECTED_START)
        & (ds.field("trade_date") <= EXPECTED_END),
    )
    raw = table.to_pandas()
    if raw.empty or raw.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1AquantInputResolutionRunnerError(
            "strict serving turnover is empty or contains duplicate serving keys"
        )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    raw["turnover_rate"] = pd.to_numeric(
        raw["turnover_rate"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    raw.loc[raw["turnover_rate"] < 0.0, "turnover_rate"] = np.nan
    return (
        raw.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
        .reindex(index=eligibility_mask.index, columns=eligibility_mask.columns)
        .astype(float)
        .where(eligibility_mask)
    )


def _load_fundamentals(
    *,
    root: Path,
    pointer: Mapping[str, Any],
    generation_manifest: Mapping[str, Any],
    expected_manifest_path: Path,
    expected_daily_path: Path,
    eligibility_mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    validated = fundamental_generation.load_fundamental_pointer(root)
    if not isinstance(validated, Mapping):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "active fundamental pointer validation returned no generation"
        )
    raw_manifest_path = Path(str(pointer.get("manifest_path", "")))
    manifest_path = raw_manifest_path if raw_manifest_path.is_absolute() else root / raw_manifest_path
    raw_daily_path = Path(str(dict(pointer.get("tables", {})).get("fundamental_daily", "")))
    daily_path = raw_daily_path if raw_daily_path.is_absolute() else root / raw_daily_path
    if (
        pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer.get("status") != "OK"
        or validated.get("generation_id") != pointer.get("generation_id")
        or validated.get("manifest") != generation_manifest
        or Path(os.path.normpath(manifest_path)) != expected_manifest_path
        or Path(os.path.normpath(daily_path)) != expected_daily_path
        or generation_manifest.get("generation_id") != pointer.get("generation_id")
        or generation_manifest.get("status") != "OK"
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "active fundamental pointer/generation binding mismatch"
        )
    daily_dataset = ds.dataset(str(expected_daily_path), format="parquet")
    required = {"ts_code", "trade_date", *FUNDAMENTAL_FIELDS}
    if not required.issubset(daily_dataset.schema.names):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "active fundamental_daily lacks required fields"
        )
    keys = daily_dataset.to_table(columns=["ts_code", "trade_date"]).to_pandas()
    if keys.empty or keys.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1AquantInputResolutionRunnerError(
            "active fundamental_daily is empty or has duplicate keys"
        )
    matrices, diagnostics = build_fundamental_metric_matrices(
        eligibility_mask.index,
        list(eligibility_mask.columns),
        metrics=FUNDAMENTAL_FIELDS,
        mart_root=root,
        allow_legacy_fallback=False,
    )
    if (
        diagnostics.get("legacy_fallback_allowed") is not False
        or dict(diagnostics.get("daily", {})).get("blocker") != ""
        or list(diagnostics.get("metrics_requested", [])) != list(FUNDAMENTAL_FIELDS)
        or set(matrices) != set(FUNDAMENTAL_FIELDS)
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "fundamental matrix loader did not prove active-generation no-fallback mode"
        )
    return {
        field: matrices[field]
        .reindex(index=eligibility_mask.index, columns=eligibility_mask.columns)
        .astype(float)
        .where(eligibility_mask)
        for field in FUNDAMENTAL_FIELDS
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cycle_id != contract.EXPECTED_CYCLE_ID:
        raise FactorV4_1AquantInputResolutionRunnerError("cycle identity mismatch")
    predecessor_spec = _parse_predecessor_bindings(args.predecessor_binding)
    code_spec = _parse_file_bindings(
        args.code_binding,
        expected_ids=contract.CODE_BINDING_IDS,
        label="code binding",
    )
    source_spec = _parse_file_bindings(
        args.source_binding,
        expected_ids=contract.SOURCE_BINDING_IDS,
        label="source binding",
    )
    predecessor_rows, predecessor_values = _read_predecessors(predecessor_spec)
    code_rows = _read_code_bindings(code_spec)

    latest, latest_semantic = _read_json_source(source_spec["latest_pointer"])
    snapshot, snapshot_semantic = _read_json_source(source_spec["snapshot_manifest"])
    pit_manifest, pit_manifest_semantic = _read_json_source(
        source_spec["pit_generation_manifest"]
    )
    components, components_semantic = _read_json_source(source_spec["pit_components"])
    fundamental_pointer, fundamental_pointer_semantic = _read_json_source(
        source_spec["fundamental_pointer"]
    )
    generation_manifest, generation_manifest_semantic = _read_json_source(
        source_spec["fundamental_generation_manifest"]
    )
    table_root = Path(source_spec["table_inventory"]["path"])
    serving_root = Path(source_spec["serving_inventory"]["path"])
    table_inventory, table_inventory_sha = _inventory_root(
        table_root,
        expected_sha256=source_spec["table_inventory"]["byte_sha256"],
    )
    serving_inventory, serving_inventory_sha = _inventory_root(
        serving_root,
        expected_sha256=source_spec["serving_inventory"]["byte_sha256"],
    )
    snapshot_path = Path(source_spec["snapshot_manifest"]["path"])
    declared_manifest = Path(str(latest.get("manifest_path", "")))
    if not declared_manifest.is_absolute():
        declared_manifest = PROJECT_ROOT / declared_manifest
    declared_table = Path(str(snapshot.get("table_root", "")))
    declared_serving = Path(str(snapshot.get("derived_serving_root", "")))
    if not declared_table.is_absolute():
        declared_table = PROJECT_ROOT / declared_table
    if not declared_serving.is_absolute():
        declared_serving = PROJECT_ROOT / declared_serving
    if (
        latest.get("status") != "OK"
        or snapshot.get("status") != "OK"
        or latest.get("snapshot_id") != snapshot.get("snapshot_id")
        or Path(os.path.normpath(declared_manifest)) != snapshot_path
        or Path(os.path.normpath(declared_table)) != table_root
        or Path(os.path.normpath(declared_serving)) != serving_root
    ):
        raise FactorV4_1AquantInputResolutionRunnerError(
            "latest/snapshot/table/serving exact binding mismatch"
        )

    membership_binding = source_spec["pit_membership"]
    membership_path = Path(membership_binding["path"])
    _stable_bytes(
        membership_path, expected_sha256=membership_binding["byte_sha256"]
    )
    daily_binding = source_spec["fundamental_daily"]
    daily_path = Path(daily_binding["path"])
    _stable_bytes(daily_path, expected_sha256=daily_binding["byte_sha256"])
    sessions = _table_sessions(table_root, table_inventory)
    eligibility_mask = _build_eligibility_mask(
        sessions=sessions,
        membership_path=membership_path,
        membership_sha256=membership_binding["byte_sha256"],
        manifest=pit_manifest,
        components=components,
    )
    market = _load_market_matrices(table_root, table_inventory, eligibility_mask)
    turnover = _load_serving_turnover(
        serving_root, serving_inventory, eligibility_mask
    )
    fundamental_root = Path(source_spec["fundamental_pointer"]["path"]).parent
    fundamentals = _load_fundamentals(
        root=fundamental_root,
        pointer=fundamental_pointer,
        generation_manifest=generation_manifest,
        expected_manifest_path=Path(
            source_spec["fundamental_generation_manifest"]["path"]
        ),
        expected_daily_path=daily_path,
        eligibility_mask=eligibility_mask,
    )
    field_matrices = {
        "close": market["close"],
        "vwap": market["vwap"],
        "turnover_rate": turnover,
        **fundamentals,
    }

    inventory_semantics = {
        "table_inventory": contract.semantic_sha256_v4_1(table_inventory),
        "serving_inventory": contract.semantic_sha256_v4_1(serving_inventory),
    }
    json_semantics = {
        "latest_pointer": latest_semantic,
        "snapshot_manifest": snapshot_semantic,
        "pit_generation_manifest": pit_manifest_semantic,
        "pit_components": components_semantic,
        "fundamental_pointer": fundamental_pointer_semantic,
        "fundamental_generation_manifest": generation_manifest_semantic,
    }
    file_semantics = {
        "pit_membership": _source_semantic(
            membership_path, membership_binding["byte_sha256"]
        ),
        "fundamental_daily": _source_semantic(
            daily_path, daily_binding["byte_sha256"]
        ),
    }
    source_rows: list[dict[str, Any]] = []
    for binding_id in contract.SOURCE_BINDING_IDS:
        binding = source_spec[binding_id]
        source_rows.append(
            {
                **copy.deepcopy(dict(binding)),
                "semantic_sha256": (
                    inventory_semantics.get(binding_id)
                    or json_semantics.get(binding_id)
                    or file_semantics[binding_id]
                ),
            }
        )
    protected = [
        {
            "binding_id": row["binding_id"],
            "path": row["path"],
            "before_sha256": row["byte_sha256"],
            "after_sha256": row["byte_sha256"],
        }
        for row in source_rows
    ]
    artifact = contract.build_input_resolution_artifact_v4_1(
        cycle_id=args.cycle_id,
        predecessor_bindings=predecessor_rows,
        code_bindings=code_rows,
        source_bindings=source_rows,
        table_root=str(table_root),
        table_inventory=table_inventory,
        serving_root=str(serving_root),
        serving_inventory=serving_inventory,
        snapshot_id=snapshot["snapshot_id"],
        fundamental_generation_id=fundamental_pointer["generation_id"],
        eligibility_mask=eligibility_mask,
        field_matrices=field_matrices,
        bound_ideas=predecessor_values["bound_ideas"],
        protected_stability=protected,
    )

    def revalidate_inputs() -> None:
        _read_predecessors(predecessor_spec)
        _read_code_bindings(code_spec)
        _inventory_root(table_root, expected_sha256=table_inventory_sha)
        _inventory_root(serving_root, expected_sha256=serving_inventory_sha)
        for binding_id, binding in source_spec.items():
            if binding_id in {"table_inventory", "serving_inventory"}:
                continue
            _stable_bytes(
                Path(binding["path"]), expected_sha256=binding["byte_sha256"]
            )

    bundle_contract = contract.build_private_bundle_contract_v4_1(
        expected_artifact=artifact
    )
    publication = private_io.publish_private_bundle(
        private_root=_absolute(args.private_root, "private root"),
        run_id=args.run_id,
        artifacts={contract.ARTIFACT_FILENAME: artifact},
        contract=bundle_contract,
        revalidate_inputs=revalidate_inputs,
    )
    readback = private_io.readback_private_bundle(
        publication["bundle_path"], contract=bundle_contract
    )
    artifact_descriptor = readback["artifact_descriptors"][
        contract.ARTIFACT_FILENAME
    ]
    readback_descriptor = readback["artifact_descriptors"][
        contract.READBACK_FILENAME
    ]
    return {
        "accepted": True,
        "bundle_path": readback["bundle_path"],
        "resolution_profile": artifact["resolution_profile"],
        "resolution_artifact_path": artifact_descriptor["absolute_path"],
        "resolution_artifact_sha256": artifact_descriptor["byte_sha256"],
        "resolution_semantic_sha256": artifact["resolution_semantic_sha256"],
        "resolution_readback_path": readback_descriptor["absolute_path"],
        "resolution_readback_sha256": readback_descriptor["byte_sha256"],
        "protected_stability": artifact["protected_stability"],
        "artifact_mode": artifact_descriptor["mode"],
        "readback_mode": readback_descriptor["mode"],
        "input_resolution_verified": True,
        "no_label": True,
        "new_risk_authorized": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact-ten no-label A_quant input-resolution evidence."
    )
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--predecessor-binding", action="append", required=True)
    parser.add_argument("--code-binding", action="append", required=True)
    parser.add_argument("--source-binding", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (FactorV4_1AquantInputResolutionRunnerError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {"accepted": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
