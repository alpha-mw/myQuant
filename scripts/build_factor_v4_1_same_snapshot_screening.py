#!/usr/bin/env python3
"""Build one explicit-path, research-only Factor v4.1 same-snapshot screen.

The runner has no production mutation surface.  It validates every predecessor
bundle before opening governed market inputs, constructs the exact cutoff PIT
matrix, generates signals without a forward-return context, and delegates all
267-family BH and exploratory shortlist derivation to the pure v4.1 contract.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import importlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as aquant_eval  # noqa: E402
from quant_investor.factors import governance_formal_catalog_bundle_v4_1 as formal_bundle  # noqa: E402
from quant_investor.factors import governance_formal_catalog_materialization_v4_1 as formal_materialization  # noqa: E402
from quant_investor.factors import governance_no_label_diagnostic_v4_1 as no_label_contract  # noqa: E402
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_screening_v4 as screening_v4  # noqa: E402
from quant_investor.factors.aquant_expression import AquantExpressionInputs  # noqa: E402
from quant_investor.factors.pit_fundamentals import (  # noqa: E402
    FUNDAMENTAL_METRICS,
    build_fundamental_metric_matrices,
)
from quant_investor.factors.runtime import MinedFactorRegistry  # noqa: E402
from quant_investor.market.fundamental_generation import (  # noqa: E402
    load_fundamental_pointer,
)
from scripts import build_factor_v4_1_no_label_diagnostic as predecessor_reader  # noqa: E402
from scripts import build_factor_v4_pre_admission_report as base_factory  # noqa: E402
from scripts.mine_quant_branch_factors import (  # noqa: E402
    MiningCandidate,
    _formulaic_primitives,
    compute_candidate_signal,
)
from scripts.retest_aquant_alpha_mix_8gate import (  # noqa: E402
    RetestContext,
    compute_existing_composite,
)


HORIZON_SESSIONS = 30
WARMUP_SESSIONS = 260
MIN_COMMON_SYMBOLS = 20
MIN_VALID_MONTHS = 3
PAIRWISE_ABS_SPEARMAN_THRESHOLD = 0.70
EXPECTED_BASE_COUNT = 230
EXPECTED_NEW_COUNT = 37
EXPECTED_TOTAL_COUNT = 267
MARKET_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "vol",
    "amount",
)
SERVING_TURNOVER_COLUMNS = ("ts_code", "trade_date", "turnover_rate")
TURNOVER_RESOLUTION_SOURCE_SEMANTICS = (
    "inventory_bound_strict_serving_root.turnover_rate.float64.exact_pit_mask"
)
FUNDAMENTAL_RESOLUTION_SOURCE_SEMANTICS = (
    "explicit_active_pit_generation.allow_legacy_fallback_false.exact_pit_mask"
)
STATISTIC_CONTRACT = {
    "raw_p_method": screening_v4.RAW_P_METHOD,
    "fdr_method": screening_v4.FDR_METHOD,
    "q": screening_v4.FDR_Q,
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FactorV4_1SameSnapshotRunnerError(ValueError):
    """Raised when an explicit same-snapshot binding fails closed."""


def _contract_module() -> Any:
    try:
        return importlib.import_module(
            "quant_investor.factors.governance_same_snapshot_screening_v4_1"
        )
    except ImportError as exc:
        raise FactorV4_1SameSnapshotRunnerError(
            "same-snapshot screening contract module is unavailable"
        ) from exc


def _input_resolution_module() -> Any:
    try:
        return importlib.import_module(
            "quant_investor.factors.governance_aquant_input_resolution_v4_1"
        )
    except ImportError as exc:
        raise FactorV4_1SameSnapshotRunnerError(
            "A_quant input-resolution contract module is unavailable"
        ) from exc


def _read_input_resolution_bundle(args: argparse.Namespace) -> dict[str, Any] | None:
    enabled = getattr(args, "resolve_predeclared_input_fields", False)
    option_names = (
        "input_resolution_bundle_path",
        "expected_input_resolution_artifact_sha256",
        "expected_input_resolution_semantic_sha256",
    )
    supplied = {name: getattr(args, name, None) for name in option_names}
    if not enabled:
        if any(value is not None for value in supplied.values()):
            raise FactorV4_1SameSnapshotRunnerError(
                "input-resolution bundle options require "
                "--resolve-predeclared-input-fields"
            )
        return None
    if any(value is None for value in supplied.values()):
        raise FactorV4_1SameSnapshotRunnerError(
            "resolved-input mode requires an explicit input-resolution bundle "
            "path plus expected artifact and semantic SHAs"
        )
    bundle_path = _absolute_path(
        supplied["input_resolution_bundle_path"],
        "input-resolution bundle",
    )
    metadata = os.lstat(bundle_path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution bundle must be an owner-only real directory"
        )
    module = _input_resolution_module()
    artifact_path = bundle_path / module.ARTIFACT_FILENAME
    expected_artifact_sha = _sha(
        supplied["expected_input_resolution_artifact_sha256"],
        "input-resolution artifact SHA",
    )
    expected_semantic_sha = _sha(
        supplied["expected_input_resolution_semantic_sha256"],
        "input-resolution semantic SHA",
    )
    # The producer contract performs an independent descriptor-based stable read;
    # the local private-file read additionally enforces owner/mode/link policy.
    result = module.validate_input_resolution_bundle_v4_1(
        artifact_path=artifact_path,
        expected_artifact_sha256=expected_artifact_sha,
        expected_semantic_sha256=expected_semantic_sha,
    )
    _stable_bytes(
        artifact_path,
        expected_sha256=expected_artifact_sha,
        private=True,
    )
    return {
        **result,
        "bundle_path": str(bundle_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_artifact_sha,
        "semantic_sha256": expected_semantic_sha,
    }


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
        raise FactorV4_1SameSnapshotRunnerError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, context: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise FactorV4_1SameSnapshotRunnerError(
            f"{context} must be lowercase SHA-256"
        )
    return value


def _absolute_path(value: Any, context: str) -> Path:
    if type(value) is not str or not value:
        raise FactorV4_1SameSnapshotRunnerError(
            f"{context} must be an absolute normalized path"
        )
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise FactorV4_1SameSnapshotRunnerError(
            f"{context} must be an absolute normalized path"
        )
    return path


def _run_id(value: Any) -> str:
    if type(value) is not str or _RUN_ID_RE.fullmatch(value) is None or ".." in value:
        raise FactorV4_1SameSnapshotRunnerError(
            "run_id must be one safe non-empty path segment"
        )
    return value


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


def _stable_bytes(path: Path, *, expected_sha256: str, private: bool) -> bytes:
    expected = _sha(expected_sha256, f"expected SHA for {path}")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FactorV4_1SameSnapshotRunnerError(
            f"bound input must be a regular non-symlink file: {path}"
        )
    if before.st_uid != os.getuid():
        raise FactorV4_1SameSnapshotRunnerError(f"bound input owner mismatch: {path}")
    if private and (
        stat.S_IMODE(before.st_mode) != 0o600 or int(before.st_nlink) != 1
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            f"private input mode/link check failed: {path}"
        )
    first = path.read_bytes()
    middle = os.lstat(path)
    second = path.read_bytes()
    after = os.lstat(path)
    if not _signature(before) == _signature(middle) == _signature(after) or first != second:
        raise FactorV4_1SameSnapshotRunnerError(
            f"bound input changed across stable read: {path}"
        )
    if hashlib.sha256(first).hexdigest() != expected:
        raise FactorV4_1SameSnapshotRunnerError(f"bound input SHA mismatch: {path}")
    return first


def _stable_current_bytes(path: Path, *, private: bool) -> tuple[bytes, str]:
    """Stable-read a pointer-derived file and return its observed byte hash."""

    initial = path.read_bytes()
    expected_sha256 = hashlib.sha256(initial).hexdigest()
    return (
        _stable_bytes(
            path,
            expected_sha256=expected_sha256,
            private=private,
        ),
        expected_sha256,
    )


def _strict_json(raw: bytes, context: str, *, canonical_private: bool) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FactorV4_1SameSnapshotRunnerError(
                    f"duplicate JSON key in {context}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorV4_1SameSnapshotRunnerError(f"invalid JSON: {context}") from exc
    if not isinstance(value, dict):
        raise FactorV4_1SameSnapshotRunnerError(f"JSON object required: {context}")
    if canonical_private and raw != _canonical_json_bytes(value) + b"\n":
        raise FactorV4_1SameSnapshotRunnerError(
            f"private JSON is not canonical newline-terminated bytes: {context}"
        )
    return value


def _parse_binding(value: str, context: str) -> dict[str, str]:
    path_text, separator, digest = value.rpartition("=")
    if not separator:
        raise FactorV4_1SameSnapshotRunnerError(
            f"{context} must use ABSOLUTE_PATH=SHA256"
        )
    path = _absolute_path(path_text, context)
    return {"absolute_path": str(path), "byte_sha256": _sha(digest, context)}


def _parse_bindings(values: Sequence[str], context: str) -> list[dict[str, str]]:
    rows = [_parse_binding(value, context) for value in values]
    if not rows or len({row["absolute_path"] for row in rows}) != len(rows):
        raise FactorV4_1SameSnapshotRunnerError(
            f"{context} paths must be non-empty and distinct"
        )
    return sorted(rows, key=lambda row: row["absolute_path"])


def _base_runtime_bindings(
    formal_ontology: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> tuple[list[MiningCandidate], dict[str, Any], dict[str, Any]]:
    candidates, base_ontology, base_catalog = base_factory._build_frozen_catalog()
    screening_v4.validate_primitive_ontology_v4(base_ontology)
    screening_v4.validate_candidate_catalog_v4(base_catalog, ontology=base_ontology)
    screening_v4.validate_primitive_ontology_v4(formal_ontology)
    screening_v4.validate_candidate_catalog_v4(
        formal_catalog, ontology=formal_ontology
    )
    if (
        len(candidates) != EXPECTED_BASE_COUNT
        or len(base_catalog["candidates"]) != EXPECTED_BASE_COUNT
        or len(formal_catalog["candidates"]) != EXPECTED_TOTAL_COUNT
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "formal/base candidate accounting must be exactly 230+37=267"
        )
    formal_by_name = {row["name"]: row for row in formal_catalog["candidates"]}
    for row in base_catalog["candidates"]:
        if formal_by_name.get(row["name"]) != row:
            raise FactorV4_1SameSnapshotRunnerError(
                f"formal base definition differs from factory: {row['name']}"
            )
    if len(set(formal_by_name) - {row["name"] for row in base_catalog["candidates"]}) != EXPECTED_NEW_COUNT:
        raise FactorV4_1SameSnapshotRunnerError(
            "formal catalog new-candidate membership must contain exactly 37 rows"
        )
    return candidates, base_ontology, base_catalog


def _cross_validate_bound_ideas(
    ideas: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> None:
    classifications = profile.get("candidate_classifications")
    rows = diagnostic.get("rows")
    if not isinstance(classifications, list) or not isinstance(rows, list):
        raise FactorV4_1SameSnapshotRunnerError(
            "no-label profile/diagnostic candidate inventories are missing"
        )
    if not len(ideas) == len(classifications) == len(rows) == EXPECTED_NEW_COUNT:
        raise FactorV4_1SameSnapshotRunnerError(
            "no-label/bound-idea accounting must contain exactly 37 rows"
        )
    field_map = {
        "candidate_id": "candidate_id",
        "name": "name",
        "source_definition_sha256": "source_definition_sha256",
        "full_candidate_normalized_ast_sha256": "normalized_ast_sha256",
        "catalog_definition_sha256": "catalog_definition_sha256",
        "mapping_semantic_sha256": "mapping_semantic_sha256",
        "input_fields": "input_fields",
    }
    for index, (idea, classification, row) in enumerate(
        zip(ideas, classifications, rows, strict=True)
    ):
        expected = {target: copy.deepcopy(idea[source]) for source, target in field_map.items()}
        observed_profile = {field: copy.deepcopy(classification.get(field)) for field in expected}
        observed_row = {field: copy.deepcopy(row.get(field)) for field in expected}
        if observed_profile != expected or observed_row != expected:
            raise FactorV4_1SameSnapshotRunnerError(
                f"bound idea differs from no-label row at index {index}"
            )


def _preflight_bundles(args: argparse.Namespace) -> dict[str, Any]:
    """Read and validate every private predecessor before governed data."""

    discovery_path = _absolute_path(args.discovery_bundle_path, "Discovery bundle")
    formal_path = _absolute_path(args.formal_bundle_path, "formal bundle")
    no_label_path = _absolute_path(args.no_label_bundle_path, "no-label bundle")
    cutoff_path = _absolute_path(args.cutoff_bundle_path, "cutoff bundle")
    discovery = predecessor_reader._read_bundle(
        bundle_path=discovery_path,
        filenames=predecessor_reader.DISCOVERY_FILENAMES,
        report_filename=predecessor_reader.DISCOVERY_REPORT,
        expected_report_sha256=args.expected_discovery_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_discovery_readback_report_semantic_sha256
        ),
        report_semantic_field="report_semantic_sha256",
    )
    formal = predecessor_reader._read_bundle(
        bundle_path=formal_path,
        filenames=formal_bundle.FORMAL_CATALOG_BUNDLE_FILENAMES,
        report_filename=formal_bundle.FORMAL_CATALOG_READBACK_REPORT_FILENAME,
        expected_report_sha256=args.expected_formal_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_formal_readback_report_semantic_sha256
        ),
        report_semantic_field="report_semantic_sha256",
    )
    no_label = predecessor_reader._read_bundle(
        bundle_path=no_label_path,
        filenames=no_label_contract.BUNDLE_FILENAMES,
        report_filename=no_label_contract.READBACK_FILENAME,
        expected_report_sha256=args.expected_no_label_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_no_label_readback_report_semantic_sha256
        ),
        report_semantic_field="report_semantic_sha256",
    )
    cutoff = predecessor_reader._read_bundle(
        bundle_path=cutoff_path,
        filenames=predecessor_reader.CUTOFF_FILENAMES,
        report_filename=predecessor_reader.CUTOFF_REPORT,
        expected_report_sha256=args.expected_cutoff_readback_report_sha256,
        expected_report_semantic_sha256=(
            args.expected_cutoff_readback_report_semantic_sha256
        ),
        report_semantic_field=None,
        allow_lock=True,
    )

    formal_values = formal["values"]
    formal_report = formal_values[
        formal_bundle.FORMAL_CATALOG_READBACK_REPORT_FILENAME
    ]
    formal_inputs = {
        filename: formal_values[filename]
        for filename in formal_bundle.FORMAL_CATALOG_INPUT_FILENAMES
    }
    formal_bundle.validate_formal_catalog_readback_report_v4_1(
        formal_report,
        artifacts=formal_inputs,
        protected_bindings=formal_report.get("protected_bindings"),
    )
    formal_ontology = formal_values[formal_materialization.FORMAL_ONTOLOGY_FILENAME]
    formal_catalog = formal_values[formal_materialization.FORMAL_CATALOG_FILENAME]
    base_candidates, base_ontology, base_catalog = _base_runtime_bindings(
        formal_ontology, formal_catalog
    )

    no_label_values = no_label_contract.validate_bundle_values_v4_1(
        no_label["values"]
    )
    no_label_report = no_label_values[no_label_contract.READBACK_FILENAME]
    no_label_contract.validate_readback_report_v4_1(
        no_label_report,
        artifacts={
            filename: no_label_values[filename]
            for filename in no_label_contract.BUNDLE_INPUT_FILENAMES
        },
        artifact_bindings=no_label_report.get("artifact_bindings"),
    )
    profile = no_label_values[no_label_contract.OPERATOR_PROFILE_FILENAME]
    diagnostic = no_label_values[no_label_contract.DIAGNOSTIC_FILENAME]
    if profile.get("cycle_id") != args.cycle_id or diagnostic.get("cycle_id") != args.cycle_id:
        raise FactorV4_1SameSnapshotRunnerError("no-label cycle_id mismatch")

    ideas = aquant_eval.bind_pinned_source_ideas_v4_1(
        source_receipt=discovery["values"]["aquant_source_receipt.v4_1.json"],
        source_idea_audit=discovery["values"]["source_idea_audit.v4_1.json"],
        primitive_mapping_proof=formal_values[
            formal_materialization.PRIMITIVE_MAPPING_PROOF_FILENAME
        ],
        formal_catalog=formal_catalog,
    )
    _cross_validate_bound_ideas(ideas, profile, diagnostic)

    cutoff_values = cutoff["values"]
    cutoff_report = cutoff_values[predecessor_reader.CUTOFF_REPORT]
    cutoff_state = cutoff_values["cycle_state.precommitted.v4_1.json"]
    if (
        cutoff_report.get("readiness") != "EXPLORATORY_PRECOMMITTED"
        or cutoff_state.get("state") != "PRECOMMITTED"
        or cutoff_state.get("cycle_id") != args.cycle_id
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "cutoff predecessor is not the exact precommitted cycle"
        )
    if discovery["values"][predecessor_reader.DISCOVERY_REPORT].get("readiness") != "EXPLORATORY_DISCOVERY":
        raise FactorV4_1SameSnapshotRunnerError("Discovery predecessor readiness mismatch")
    if formal_report.get("readiness") != formal_bundle.READINESS:
        raise FactorV4_1SameSnapshotRunnerError("formal predecessor readiness mismatch")

    return {
        "discovery": discovery,
        "formal": formal,
        "no_label": no_label,
        "cutoff": cutoff,
        "base_candidates": base_candidates,
        "base_ontology": base_ontology,
        "base_catalog": base_catalog,
        "formal_ontology": formal_ontology,
        "formal_catalog": formal_catalog,
        "bound_ideas": ideas,
        "profile": profile,
        "diagnostic": diagnostic,
    }


def _blank_signal_context(context: RetestContext) -> RetestContext:
    blank = pd.DataFrame(
        np.nan,
        index=context.forward_return.index,
        columns=context.forward_return.columns,
        dtype=float,
    )
    return replace(context, forward_return=blank)


def _masked_forward_returns(
    adj_close: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
    *,
    horizon: int = HORIZON_SESSIONS,
) -> pd.DataFrame:
    if (
        not adj_close.index.equals(eligibility_mask.index)
        or not adj_close.columns.equals(eligibility_mask.columns)
        or type(horizon) is not int
        or horizon <= 0
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "forward-return inputs must have exact axes and a positive horizon"
        )
    prices = adj_close.where(eligibility_mask)
    result = prices.shift(-horizon).div(prices.shift(-1)).sub(1.0)
    return result.where(eligibility_mask).replace([np.inf, -np.inf], np.nan)


def _monthly_rebalance_dates(
    dates: pd.DatetimeIndex,
    *,
    warmup: int = WARMUP_SESSIONS,
    horizon: int = HORIZON_SESSIONS,
) -> list[pd.Timestamp]:
    ordered = pd.DatetimeIndex(dates)
    if not ordered.is_monotonic_increasing or ordered.has_duplicates:
        raise FactorV4_1SameSnapshotRunnerError(
            "screening calendar must be sorted and unique"
        )
    if (
        type(warmup) is not int
        or warmup < 0
        or type(horizon) is not int
        or horizon <= 0
        or len(ordered) <= warmup + horizon
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "screening calendar does not satisfy the fixed warmup"
        )
    full_month_ends = (
        pd.Series(ordered, index=ordered)
        .groupby(ordered.to_period("M"))
        .max()
    )
    first_usable = pd.Timestamp(ordered[warmup])
    last_usable = pd.Timestamp(ordered[-horizon - 1])
    closed = full_month_ends[
        (full_month_ends >= first_usable)
        & (full_month_ends <= last_usable)
    ]
    if closed.empty:
        raise FactorV4_1SameSnapshotRunnerError(
            "screening calendar has no closed month ends after fixed guards"
        )
    return [pd.Timestamp(value) for value in closed.tolist()]


def _rank_ic_series(
    signal: pd.DataFrame,
    forward_return: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    common_columns = signal.columns.intersection(forward_return.columns)
    for raw_date in dates:
        date = pd.Timestamp(raw_date)
        if date not in signal.index or date not in forward_return.index:
            continue
        frame = pd.concat(
            [
                signal.loc[date, common_columns].rename("signal"),
                forward_return.loc[date, common_columns].rename("forward"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if (
            len(frame) < MIN_COMMON_SYMBOLS
            or frame["signal"].nunique(dropna=True) <= 1
            or frame["forward"].nunique(dropna=True) <= 1
        ):
            continue
        value = frame["signal"].corr(frame["forward"], method="spearman")
        if pd.notna(value):
            values[date] = float(value)
    return pd.Series(values, dtype=float).sort_index()


def _raw_p_value(rank_ic: pd.Series) -> float:
    clean = rank_ic.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(clean) < 2:
        raise FactorV4_1SameSnapshotRunnerError(
            "raw p-value requires at least two valid monthly RankIC observations"
        )
    standard_error = float(clean.std(ddof=1)) / math.sqrt(len(clean))
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        raise FactorV4_1SameSnapshotRunnerError(
            "raw p-value standard error is unavailable"
        )
    statistic = float(clean.mean()) / standard_error
    value = math.erfc(abs(statistic) / math.sqrt(2.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FactorV4_1SameSnapshotRunnerError("raw p-value is not finite in [0,1]")
    return float(value)


def _signal_summary(
    signal: pd.DataFrame,
    eligibility_mask: pd.DataFrame,
    forward_return: pd.DataFrame,
    monthly_dates: Sequence[pd.Timestamp],
    *,
    minimum_valid_rank_ic_months: int = 2,
) -> tuple[dict[str, Any], np.ndarray]:
    if not signal.index.equals(eligibility_mask.index) or not signal.columns.equals(
        eligibility_mask.columns
    ):
        raise FactorV4_1SameSnapshotRunnerError("signal axes differ from PIT mask")
    masked = signal.where(eligibility_mask)
    values = masked.to_numpy(dtype=np.float64, copy=False)
    eligible = eligibility_mask.to_numpy(dtype=bool, copy=False)
    finite_count = int(np.isfinite(values[eligible]).sum())
    eligible_count = int(eligible.sum())
    if finite_count <= 0 or eligible_count <= 0:
        raise FactorV4_1SameSnapshotRunnerError("signal has no finite PIT observations")
    rank_ic = _rank_ic_series(masked, forward_return, monthly_dates)
    if (
        type(minimum_valid_rank_ic_months) is not int
        or minimum_valid_rank_ic_months < 2
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "minimum valid RankIC months must be an integer of at least two"
        )
    if len(rank_ic) < minimum_valid_rank_ic_months:
        raise FactorV4_1SameSnapshotRunnerError(
            "signal lacks required monthly RankIC coverage: "
            f"observed={len(rank_ic)};required={minimum_valid_rank_ic_months}"
        )
    descriptor = aquant_eval.matrix_hash_descriptor_v4_1(masked)
    monthly = (
        masked.reindex(index=pd.DatetimeIndex(monthly_dates))
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32, copy=True)
    )
    return {
        "signal_sha256": descriptor["matrix_sha256"],
        "finite_ratio": float(finite_count / eligible_count),
        "raw_p_value": _raw_p_value(rank_ic),
    }, monthly


def _run_config_semantic_sha(
    preflight: Mapping[str, Any],
    *,
    resolve_predeclared_input_fields: bool = False,
    input_resolution_artifact_path: str | None = None,
    input_resolution_artifact_sha256: str | None = None,
    input_resolution_semantic_sha256: str | None = None,
) -> str:
    contract = _contract_module()
    if type(resolve_predeclared_input_fields) is not bool:
        raise FactorV4_1SameSnapshotRunnerError(
            "resolved-input mode flag must be boolean"
        )
    if resolve_predeclared_input_fields:
        normalized_resolution_artifact_path = str(
            _absolute_path(
                input_resolution_artifact_path,
                "input-resolution artifact",
            )
        )
        normalized_resolution_artifact_sha = _sha(
            input_resolution_artifact_sha256,
            "input-resolution artifact SHA",
        )
        normalized_resolution_semantic_sha = _sha(
            input_resolution_semantic_sha256,
            "input-resolution semantic SHA",
        )
    elif (
        input_resolution_artifact_sha256 is not None
        or input_resolution_artifact_path is not None
        or input_resolution_semantic_sha256 is not None
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution hashes require resolved-input mode"
        )
    else:
        normalized_resolution_artifact_sha = None
        normalized_resolution_semantic_sha = None
        normalized_resolution_artifact_path = None
    payload = {
        "schema_version": "factor-v4.1-same-snapshot-run-config.v1",
        "horizon_sessions": HORIZON_SESSIONS,
        "warmup_sessions": WARMUP_SESSIONS,
        "rebalance_policy": (
            "closed_natural_month_last_open_session_after_warmup_and_horizon"
        ),
        "minimum_common_symbols": MIN_COMMON_SYMBOLS,
        "minimum_valid_pair_months": MIN_VALID_MONTHS,
        "pairwise_abs_spearman_threshold": PAIRWISE_ABS_SPEARMAN_THRESHOLD,
        "candidate_count": EXPECTED_TOTAL_COUNT,
        "base_candidate_count": EXPECTED_BASE_COUNT,
        "new_candidate_count": EXPECTED_NEW_COUNT,
        "signal_forward_return_context": "all_nan",
        "statistic_forward_return_source": "pit_masked_adj_close_shift_minus_30",
        "cutoff_fundamental_authority": False,
        "discovery_report_semantic_sha256": preflight["discovery"][
            "report_semantic_sha256"
        ],
        "formal_report_semantic_sha256": preflight["formal"][
            "report_semantic_sha256"
        ],
        "no_label_report_semantic_sha256": preflight["no_label"][
            "report_semantic_sha256"
        ],
        "cutoff_report_semantic_sha256": preflight["cutoff"][
            "report_semantic_sha256"
        ],
    }
    if resolve_predeclared_input_fields:
        payload.update(
            {
                "resolve_predeclared_input_fields": True,
                "input_resolution_artifact_sha256": (
                    normalized_resolution_artifact_sha
                ),
                "input_resolution_artifact_path": (
                    normalized_resolution_artifact_path
                ),
                "input_resolution_semantic_sha256": (
                    normalized_resolution_semantic_sha
                ),
                "predeclared_turnover_candidate_names": sorted(
                    contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
                ),
                "predeclared_fundamental_candidate_names": sorted(
                    contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
                ),
                "allowed_accounting_profiles": sorted(contract.ACCOUNTING_PROFILES),
                "turnover_resolution_source_semantics": (
                    TURNOVER_RESOLUTION_SOURCE_SEMANTICS
                ),
                "fundamental_resolution_source_semantics": (
                    FUNDAMENTAL_RESOLUTION_SOURCE_SEMANTICS
                ),
                "resolved_evaluator": (
                    "governance_aquant_no_label_eval_v4_1."
                    "evaluate_pinned_idea_v4_1"
                ),
                "resolved_eligibility_scope": "exact_bound_pit_mask",
                "turnover_minimum_valid_rank_ic_months": MIN_VALID_MONTHS,
                "resolved_diagnostic_reproduction_candidate_names": sorted(
                    set(
                        row["name"]
                        for row in preflight["formal_catalog"]["candidates"]
                    )
                    - set(
                        row["name"]
                        for row in preflight["base_catalog"]["candidates"]
                    )
                    - set(contract.PREDECLARED_BLOCKED_CANDIDATE_NAMES)
                ),
            }
        )
    return _semantic_sha(payload)


def _declared_path(value: Any, context: str) -> Path:
    if type(value) is not str or not value:
        raise FactorV4_1SameSnapshotRunnerError(f"{context} path is missing")
    raw = Path(value)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    normalized = Path(os.path.normpath(path))
    if normalized != path:
        raise FactorV4_1SameSnapshotRunnerError(f"{context} path is not normalized")
    return normalized


def _binding_by_suffix(
    rows: Sequence[Mapping[str, str]], suffix: str
) -> Mapping[str, str]:
    matches = [row for row in rows if row["absolute_path"].endswith(suffix)]
    if len(matches) != 1:
        raise FactorV4_1SameSnapshotRunnerError(
            f"protected binding suffix must match exactly once: {suffix}"
        )
    return matches[0]


def _read_control_inputs(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    code_rows = _parse_bindings(args.code_binding, "code binding")
    protected_rows = _parse_bindings(args.protected_binding, "protected binding")
    expected_protected = preflight["formal"]["values"][
        formal_bundle.FORMAL_CATALOG_READBACK_REPORT_FILENAME
    ]["protected_bindings"]
    if protected_rows != expected_protected:
        raise FactorV4_1SameSnapshotRunnerError(
            "explicit protected bindings differ from formal predecessor"
        )
    protected_raw: dict[str, bytes] = {}
    for row in protected_rows:
        path = Path(row["absolute_path"])
        protected_raw[str(path)] = _stable_bytes(
            path,
            expected_sha256=row["byte_sha256"],
            private=False,
        )
    code_raw: dict[str, bytes] = {}
    for row in code_rows:
        path = Path(row["absolute_path"])
        code_raw[str(path)] = _stable_bytes(
            path,
            expected_sha256=row["byte_sha256"],
            private=False,
        )
    required_paths = {
        str(Path(__file__).resolve()),
        str(Path(aquant_eval.__file__).resolve()),
        str(Path(screening_v4.__file__).resolve()),
        str(Path(no_label_contract.__file__).resolve()),
        str(Path(formal_bundle.__file__).resolve()),
        str(Path(formal_materialization.__file__).resolve()),
        str(Path(private_io.__file__).resolve()),
        str(Path(base_factory.__file__).resolve()),
        str(Path(predecessor_reader.__file__).resolve()),
        str(Path(sys.modules[compute_candidate_signal.__module__].__file__).resolve()),
        str(Path(sys.modules[RetestContext.__module__].__file__).resolve()),
        str(Path(sys.modules[AquantExpressionInputs.__module__].__file__).resolve()),
        str(Path(sys.modules[build_fundamental_metric_matrices.__module__].__file__).resolve()),
        str(Path(sys.modules[load_fundamental_pointer.__module__].__file__).resolve()),
        str(Path(sys.modules[MinedFactorRegistry.__module__].__file__).resolve()),
    }
    contract_path = Path(_contract_module().__file__).resolve()
    required_paths.add(str(contract_path))
    if getattr(args, "resolve_predeclared_input_fields", False):
        required_paths.add(str(Path(_input_resolution_module().__file__).resolve()))
    if not required_paths.issubset(code_raw):
        missing = sorted(required_paths - set(code_raw))
        raise FactorV4_1SameSnapshotRunnerError(
            f"code bindings omit direct runtime sources: {missing}"
        )
    code_sha = _semantic_sha(
        {
            path: hashlib.sha256(raw).hexdigest()
            for path, raw in sorted(code_raw.items())
        }
    )
    return {
        "code_rows": code_rows,
        "code_sha256": code_sha,
        "protected_rows": protected_rows,
        "protected_raw": protected_raw,
    }


def _inventory_root(
    root: Path, expected_sha256: str, context: str
) -> tuple[list[dict[str, Any]], str]:
    inventory, observed = predecessor_reader._inventory_table(root)
    if observed != _sha(expected_sha256, f"expected {context} inventory SHA"):
        raise FactorV4_1SameSnapshotRunnerError(f"{context} inventory SHA mismatch")
    return inventory, observed


def _bind_governed_inputs(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    protected_rows = controls["protected_rows"]
    latest_row = _binding_by_suffix(protected_rows, "/data/parquet/cn/_latest.json")
    fundamental_pointer_row = _binding_by_suffix(
        protected_rows, "/data/parquet/cn/_fundamental_latest.json"
    )
    fundamental_manifest_row = _binding_by_suffix(
        protected_rows, "/data/parquet/cn/latest_manifest.json"
    )
    registry_row = _binding_by_suffix(
        protected_rows, "/quant_investor/factor_registry/mined_factors.json"
    )
    protected_raw = controls["protected_raw"]
    latest = _strict_json(
        protected_raw[latest_row["absolute_path"]],
        latest_row["absolute_path"],
        canonical_private=False,
    )
    fundamental_pointer = _strict_json(
        protected_raw[fundamental_pointer_row["absolute_path"]],
        fundamental_pointer_row["absolute_path"],
        canonical_private=False,
    )
    fundamental_manifest = _strict_json(
        protected_raw[fundamental_manifest_row["absolute_path"]],
        fundamental_manifest_row["absolute_path"],
        canonical_private=False,
    )
    registry_payload = _strict_json(
        protected_raw[registry_row["absolute_path"]],
        registry_row["absolute_path"],
        canonical_private=False,
    )

    snapshot_path = _absolute_path(args.snapshot_manifest_path, "snapshot manifest")
    snapshot_raw = _stable_bytes(
        snapshot_path,
        expected_sha256=args.expected_snapshot_manifest_sha256,
        private=False,
    )
    snapshot = _strict_json(
        snapshot_raw, str(snapshot_path), canonical_private=False
    )
    table_root = _absolute_path(args.table_root, "table root")
    serving_root = _absolute_path(args.serving_root, "serving root")
    coverage = snapshot.get("coverage")
    if (
        latest.get("status") != "OK"
        or list(latest.get("blockers", []) or [])
        or snapshot.get("status") != "OK"
        or snapshot.get("readback_validated") is not True
        or list(snapshot.get("blockers", []) or [])
        or snapshot.get("snapshot_id") != latest.get("snapshot_id")
        or not isinstance(coverage, Mapping)
        or coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("complete") is not True
        or "full_a" not in list(coverage.get("categories_checked", []) or [])
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "strict full-A pointer/snapshot is not healthy"
        )
    if (
        _declared_path(latest.get("manifest_path"), "latest snapshot manifest")
        != snapshot_path
        or _declared_path(snapshot.get("manifest_path"), "snapshot self binding")
        != snapshot_path
        or _declared_path(snapshot.get("table_root"), "snapshot table root")
        != table_root
        or _declared_path(snapshot.get("derived_serving_root"), "snapshot serving root")
        != serving_root
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "explicit snapshot/table/serving paths differ from strict bindings"
        )

    fundamental_root = Path(fundamental_pointer_row["absolute_path"]).parent
    validated_fundamental_pointer = load_fundamental_pointer(fundamental_root)
    primary = (
        validated_fundamental_pointer.get("primary_provenance", {})
        if isinstance(validated_fundamental_pointer, Mapping)
        else {}
    )
    if (
        fundamental_pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or fundamental_pointer.get("status") != "OK"
        or not isinstance(validated_fundamental_pointer, Mapping)
        or validated_fundamental_pointer.get("generation_id")
        != fundamental_pointer.get("generation_id")
        or validated_fundamental_pointer.get("status") != "OK"
        or validated_fundamental_pointer.get("manifest_path")
        != fundamental_pointer.get("manifest_path")
        or validated_fundamental_pointer.get("tables")
        != fundamental_pointer.get("tables")
        or validated_fundamental_pointer.get("primary_provenance_verified") is not True
        or primary.get("status") != "verified_live_tushare"
        or primary.get("source_priority") != "tushare_primary"
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "Fundamental protected pointer/generation readback is invalid"
        )
    raw_manifest_path = Path(str(fundamental_pointer.get("manifest_path", "")))
    fundamental_generation_manifest_path = (
        raw_manifest_path
        if raw_manifest_path.is_absolute()
        else fundamental_root / raw_manifest_path
    )
    fundamental_generation_manifest_path = Path(
        os.path.normpath(fundamental_generation_manifest_path)
    )
    try:
        fundamental_generation_manifest_path.relative_to(fundamental_root)
    except ValueError as exc:
        raise FactorV4_1SameSnapshotRunnerError(
            "Fundamental generation manifest escapes the protected root"
        ) from exc
    (
        fundamental_generation_manifest_raw,
        fundamental_generation_manifest_sha256,
    ) = _stable_current_bytes(
        fundamental_generation_manifest_path,
        private=False,
    )
    fundamental_generation_manifest = _strict_json(
        fundamental_generation_manifest_raw,
        str(fundamental_generation_manifest_path),
        canonical_private=False,
    )
    if fundamental_generation_manifest != validated_fundamental_pointer.get(
        "manifest"
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "Fundamental generation manifest differs from validated pointer readback"
        )

    table_inventory, table_inventory_sha = _inventory_root(
        table_root, args.expected_table_inventory_sha256, "table"
    )
    serving_inventory, serving_inventory_sha = _inventory_root(
        serving_root, args.expected_serving_inventory_sha256, "serving"
    )
    pit_path = _absolute_path(args.pit_membership_path, "PIT membership")
    pit_manifest_path = _absolute_path(
        args.pit_generation_manifest_path, "PIT generation manifest"
    )
    components_path = _absolute_path(args.components_path, "components")
    cutoff_binding = preflight["cutoff"]["values"][
        "cutoff_input_binding.v4_1.json"
    ]
    table_binding = cutoff_binding.get("table", {})
    pit_binding = cutoff_binding.get("pit_generation", {})
    component_binding = cutoff_binding.get("components", {})
    exact_checks = (
        (str(table_root), table_binding.get("absolute_root")),
        (table_inventory_sha, table_binding.get("inventory_sha256")),
        (
            str(pit_path),
            pit_binding.get("membership", {}).get("absolute_path"),
        ),
        (
            args.expected_pit_membership_sha256,
            pit_binding.get("membership", {}).get("sha256"),
        ),
        (
            str(pit_manifest_path),
            pit_binding.get("manifest", {}).get("absolute_path"),
        ),
        (
            args.expected_pit_generation_manifest_sha256,
            pit_binding.get("manifest", {}).get("sha256"),
        ),
        (str(components_path), component_binding.get("absolute_path")),
        (args.expected_components_sha256, component_binding.get("sha256")),
    )
    if any(actual != expected for actual, expected in exact_checks):
        raise FactorV4_1SameSnapshotRunnerError(
            "explicit market/PIT/component input differs from cutoff binding"
        )
    if (
        table_inventory != table_binding.get("parquet_inventory")
        or len(table_inventory) != table_binding.get("regular_file_count")
        or sum(row["dataset_member"] is True for row in table_inventory)
        != table_binding.get("parquet_file_count")
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "table inventory differs from cutoff predecessor"
        )
    components = predecessor_reader._load_components(
        components_path, args.expected_components_sha256
    )
    pit_records, _pit_manifest = predecessor_reader._load_pit_records(
        membership_path=pit_path,
        expected_membership_sha256=args.expected_pit_membership_sha256,
        manifest_path=pit_manifest_path,
        expected_manifest_sha256=args.expected_pit_generation_manifest_sha256,
    )
    design_source = preflight["cutoff"]["values"]["design_source.v4_1.json"]
    eligibility_mask, scope_binding = predecessor_reader._reproduce_session_scope(
        design_source=design_source,
        pit_records=pit_records,
        component_symbols=components,
    )
    market_inventory = {
        "schema_version": "factor-v4.1-same-snapshot-market-input.v1",
        "snapshot_id": snapshot["snapshot_id"],
        "latest_pointer_sha256": latest_row["byte_sha256"],
        "snapshot_manifest_sha256": args.expected_snapshot_manifest_sha256,
        "pit_membership_sha256": args.expected_pit_membership_sha256,
        "pit_generation_manifest_sha256": (
            args.expected_pit_generation_manifest_sha256
        ),
        "components_sha256": args.expected_components_sha256,
        "table_root": str(table_root),
        "table_inventory_sha256": table_inventory_sha,
        "table_inventory": table_inventory,
        "serving_root": str(serving_root),
        "serving_inventory_sha256": serving_inventory_sha,
        "serving_inventory": serving_inventory,
        "fundamental_generation": {
            "pointer_sha256": fundamental_pointer_row["byte_sha256"],
            "generation_id": fundamental_pointer["generation_id"],
            "manifest_path": str(fundamental_generation_manifest_path),
            "manifest_sha256": fundamental_generation_manifest_sha256,
            "tables": copy.deepcopy(fundamental_pointer["tables"]),
        },
    }
    return {
        "latest_row": latest_row,
        "fundamental_pointer_row": fundamental_pointer_row,
        "fundamental_manifest_row": fundamental_manifest_row,
        "fundamental_manifest": fundamental_manifest,
        "fundamental_generation_manifest_path": (
            fundamental_generation_manifest_path
        ),
        "fundamental_generation_manifest_sha256": (
            fundamental_generation_manifest_sha256
        ),
        "fundamental_generation": copy.deepcopy(dict(validated_fundamental_pointer)),
        "fundamental_root": fundamental_root,
        "registry_row": registry_row,
        "registry_payload": registry_payload,
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "table_root": table_root,
        "serving_root": serving_root,
        "table_inventory": table_inventory,
        "serving_inventory": serving_inventory,
        "market_inventory": market_inventory,
        "market_data_input_sha256": _semantic_sha(market_inventory),
        "pit_path": pit_path,
        "pit_manifest_path": pit_manifest_path,
        "components_path": components_path,
        "eligibility_mask": eligibility_mask,
        "scope_binding": scope_binding,
    }


def _load_market_matrices(
    table_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    eligibility_mask: pd.DataFrame,
    *,
    serving_root: Path | None = None,
    serving_inventory: Sequence[Mapping[str, Any]] | None = None,
    include_turnover_rate: bool = False,
) -> dict[str, pd.DataFrame]:
    paths = [
        str(table_root / row["relative_path"])
        for row in inventory
        if row["dataset_member"] is True
    ]
    dataset = ds.dataset(paths, format="parquet")
    if not set(MARKET_COLUMNS).issubset(set(dataset.schema.names)):
        raise FactorV4_1SameSnapshotRunnerError(
            "strict table lacks one or more authorized market columns"
        )
    sessions = eligibility_mask.index.strftime("%Y%m%d")
    table = dataset.to_table(
        columns=list(MARKET_COLUMNS),
        filter=(ds.field("trade_date") >= sessions[0])
        & (ds.field("trade_date") <= sessions[-1]),
    )
    raw = table.to_pandas()
    if raw.empty or list(raw.columns) != list(MARKET_COLUMNS):
        raise FactorV4_1SameSnapshotRunnerError(
            "strict market projection is empty or reordered"
        )
    if raw.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1SameSnapshotRunnerError(
            "strict market projection contains duplicate symbol/session rows"
        )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    matrices: dict[str, pd.DataFrame] = {}
    columns = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "vol",
        "amount": "amount",
    }
    for name, source_column in columns.items():
        matrix = raw.pivot(
            index="trade_date", columns="ts_code", values=source_column
        ).reindex(
            index=eligibility_mask.index,
            columns=eligibility_mask.columns,
        )
        matrices[name] = matrix.astype(float).where(eligibility_mask)
    matrices["vwap"] = predecessor_reader._derive_vwap(
        matrices["amount"], matrices["volume"]
    ).astype(float).where(eligibility_mask)
    if include_turnover_rate:
        if serving_root is None or serving_inventory is None:
            raise FactorV4_1SameSnapshotRunnerError(
                "resolved turnover projection requires an explicit bound serving root "
                "and inventory"
            )
        matrices["turnover_rate"] = _load_serving_turnover_rate(
            serving_root,
            serving_inventory,
            eligibility_mask,
        )
    return matrices


def _load_serving_turnover_rate(
    serving_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    eligibility_mask: pd.DataFrame,
) -> pd.DataFrame:
    paths = [
        str(serving_root / row["relative_path"])
        for row in inventory
        if row["dataset_member"] is True
    ]
    if not paths:
        raise FactorV4_1SameSnapshotRunnerError(
            "strict serving inventory has no dataset members"
        )
    dataset = ds.dataset(paths, format="parquet")
    if not set(SERVING_TURNOVER_COLUMNS).issubset(set(dataset.schema.names)):
        raise FactorV4_1SameSnapshotRunnerError(
            "strict serving root lacks the authorized turnover_rate projection"
        )
    sessions = eligibility_mask.index.strftime("%Y%m%d")
    table = dataset.to_table(
        columns=list(SERVING_TURNOVER_COLUMNS),
        filter=(ds.field("trade_date") >= sessions[0])
        & (ds.field("trade_date") <= sessions[-1]),
    )
    raw = table.to_pandas()
    if raw.empty or list(raw.columns) != list(SERVING_TURNOVER_COLUMNS):
        raise FactorV4_1SameSnapshotRunnerError(
            "strict serving turnover projection is empty or reordered"
        )
    if raw.duplicated(["trade_date", "ts_code"]).any():
        raise FactorV4_1SameSnapshotRunnerError(
            "strict serving turnover projection contains duplicate symbol/session rows"
        )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    matrix = raw.pivot(
        index="trade_date", columns="ts_code", values="turnover_rate"
    ).reindex(index=eligibility_mask.index, columns=eligibility_mask.columns)
    return matrix.astype(float).where(eligibility_mask)


def _runtime_inputs(
    governed: Mapping[str, Any],
    matrices: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    eligibility_mask = governed["eligibility_mask"]
    dates = eligibility_mask.index
    symbols = list(eligibility_mask.columns)
    fundamental, diagnostics = build_fundamental_metric_matrices(
        dates,
        symbols,
        metrics=FUNDAMENTAL_METRICS,
        mart_root=governed["fundamental_root"],
        allow_legacy_fallback=False,
    )
    if diagnostics.get("legacy_fallback_allowed") is not False:
        raise FactorV4_1SameSnapshotRunnerError(
            "Fundamental loader did not prove strict no-fallback mode"
        )
    # Preserve the canonical runtime precision for every full-session input.
    # The only float32 conversion in this runner is the compact monthly signal
    # slice retained after a candidate has been evaluated.
    fundamental = {
        name: frame.reindex(index=dates, columns=symbols)
        .astype(float)
        .where(eligibility_mask)
        for name, frame in fundamental.items()
    }
    if "turnover_rate" in matrices:
        turnover_rate = (
            matrices["turnover_rate"]
            .reindex(index=dates, columns=symbols)
            .astype(float)
            .where(eligibility_mask)
        )
    else:
        turnover_rate = pd.DataFrame(
            np.nan, index=dates, columns=symbols, dtype=float
        )
    expression_inputs = AquantExpressionInputs(
        open=matrices["open"],
        high=matrices["high"],
        low=matrices["low"],
        close=matrices["close"],
        adj_close=matrices["adj_close"],
        vwap=matrices["vwap"],
        volume=matrices["volume"],
        amount=matrices["amount"],
        turnover_rate=turnover_rate,
        fin_roe=fundamental["fin_roe"],
        fin_roa=fundamental["fin_roa"],
        fin_debt_to_assets=fundamental["fin_debt_to_assets"],
        fin_net_profit_yoy=fundamental["fin_net_profit_yoy"],
        fin_ocf_to_profit=fundamental["fin_ocf_to_profit"],
        fin_fcf_to_profit=fundamental["fin_fcf_to_profit"],
        fcf_to_price=fundamental["fcf_to_price"],
        diagnostics={"pit": diagnostics, "legacy_fallback_allowed": False},
    )
    registry = MinedFactorRegistry.from_dict(governed["registry_payload"])
    existing, blocker = compute_existing_composite(
        registry,
        matrices["adj_close"],
        matrices["volume"],
        matrices["amount"],
    )
    if existing is None or blocker:
        raise FactorV4_1SameSnapshotRunnerError(
            f"bound existing composite is unavailable: {blocker}"
        )
    existing = existing.where(eligibility_mask).astype(float)
    monthly_dates = _monthly_rebalance_dates(dates)
    forward_return = _masked_forward_returns(
        matrices["adj_close"], eligibility_mask
    )
    statistical_context = RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in symbols},
        adj_close=matrices["adj_close"],
        volume=matrices["volume"],
        amount=matrices["amount"],
        forward_return=forward_return,
        rebalance_dates=monthly_dates,
        biweekly_dates=[],
        existing_composite=existing,
        existing_blocker="",
    )
    signal_context = _blank_signal_context(statistical_context)
    formulaic_primitives = _formulaic_primitives(
        signal_context, expression_inputs
    )
    aquant_matrices = {
        name: value
        for name, value in expression_inputs.context().items()
        if isinstance(value, pd.DataFrame)
    }
    return {
        "expression_inputs": expression_inputs,
        "aquant_matrices": aquant_matrices,
        "formulaic_primitives": formulaic_primitives,
        "signal_context": signal_context,
        "statistical_context": statistical_context,
        "monthly_dates": monthly_dates,
    }


def _inventory_identity_projection(
    value: Any,
    *,
    label: str,
    digest_field: str,
    require_hard_link_count: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FactorV4_1SameSnapshotRunnerError(
            f"{label} must be a non-empty inventory"
        )
    common_fields = {
        "relative_path",
        "size_bytes",
        "dataset_member",
        digest_field,
    }
    expected_fields = (
        common_fields | {"hard_link_count"}
        if require_hard_link_count
        else common_fields
    )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise FactorV4_1SameSnapshotRunnerError(
                f"{label}[{index}] inventory fields are not exact"
            )
        relative_path = raw["relative_path"]
        if type(relative_path) is not str or not relative_path:
            raise FactorV4_1SameSnapshotRunnerError(
                f"{label}[{index}].relative_path must be non-empty text"
            )
        normalized_relative = PurePosixPath(relative_path)
        if (
            normalized_relative.is_absolute()
            or normalized_relative.as_posix() != relative_path
            or any(part in {".", ".."} for part in normalized_relative.parts)
        ):
            raise FactorV4_1SameSnapshotRunnerError(
                f"{label}[{index}].relative_path is not normalized and relative"
            )
        size_bytes = raw["size_bytes"]
        if type(size_bytes) is not int or size_bytes <= 0:
            raise FactorV4_1SameSnapshotRunnerError(
                f"{label}[{index}].size_bytes must be positive"
            )
        dataset_member = raw["dataset_member"]
        if type(dataset_member) is not bool:
            raise FactorV4_1SameSnapshotRunnerError(
                f"{label}[{index}].dataset_member must be boolean"
            )
        if require_hard_link_count:
            hard_link_count = raw["hard_link_count"]
            if type(hard_link_count) is not int or hard_link_count <= 0:
                raise FactorV4_1SameSnapshotRunnerError(
                    f"{label}[{index}].hard_link_count must be positive"
                )
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "dataset_member": dataset_member,
                "digest": _sha(
                    raw[digest_field], f"{label}[{index}].{digest_field}"
                ),
            }
        )
    paths = [row["relative_path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise FactorV4_1SameSnapshotRunnerError(
            f"{label} paths must be sorted and unique"
        )
    return rows


def _assert_inventory_identity_equivalent(
    artifact_inventory: Any,
    governed_inventory: Any,
    *,
    label: str,
) -> None:
    artifact_identity = _inventory_identity_projection(
        artifact_inventory,
        label=f"input-resolution {label}",
        digest_field="byte_sha256",
        require_hard_link_count=False,
    )
    governed_identity = _inventory_identity_projection(
        governed_inventory,
        label=f"governed {label}",
        digest_field="sha256",
        require_hard_link_count=True,
    )
    if artifact_identity != governed_identity:
        raise FactorV4_1SameSnapshotRunnerError(
            f"input-resolution {label} identity differs from active governed input"
        )


def _bind_resolution_to_runtime(
    resolution: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    governed: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    module = _input_resolution_module()
    contract = _contract_module()
    artifact = resolution.get("artifact")
    if not isinstance(artifact, Mapping):
        raise FactorV4_1SameSnapshotRunnerError(
            "validated input-resolution artifact is missing"
        )
    if set(module.RESOLVED_CANDIDATE_NAMES) != set(
        contract.PREDECLARED_BLOCKED_CANDIDATE_NAMES
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution and screening candidate inventories differ"
        )
    _assert_inventory_identity_equivalent(
        artifact["table_inventory"],
        governed["table_inventory"],
        label="table inventory",
    )
    _assert_inventory_identity_equivalent(
        artifact["serving_inventory"],
        governed["serving_inventory"],
        label="serving inventory",
    )
    if (
        artifact["cycle_id"] != args.cycle_id
        or artifact["table_root"] != str(governed["table_root"])
        or artifact["serving_root"] != str(governed["serving_root"])
        or artifact["fundamental_generation_id"]
        != governed["fundamental_generation"]["generation_id"]
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution artifact differs from the active governed sources"
        )
    eligibility_mask = governed["eligibility_mask"]
    mask_descriptor = aquant_eval.matrix_hash_descriptor_v4_1(
        eligibility_mask.astype(float)
    )
    artifact_context = artifact["matrix_context"]
    if (
        artifact_context["snapshot_id"] != governed["snapshot"]["snapshot_id"]
        or artifact_context["eligibility_mask"] != mask_descriptor
        or artifact_context["date_axis_sha256"]
        != mask_descriptor["date_axis_sha256"]
        or artifact_context["symbol_axis_sha256"]
        != mask_descriptor["symbol_axis_sha256"]
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution artifact PIT axes differ from the active mask"
        )
    source_by_id = {
        row["binding_id"]: row for row in artifact["source_bindings"]
    }
    expected_source_bytes = {
        "fundamental_generation_manifest": governed[
            "fundamental_generation_manifest_sha256"
        ],
        "fundamental_pointer": governed["fundamental_pointer_row"]["byte_sha256"],
        "latest_pointer": governed["latest_row"]["byte_sha256"],
        "pit_components": _sha(
            args.expected_components_sha256, "components SHA"
        ),
        "pit_generation_manifest": _sha(
            args.expected_pit_generation_manifest_sha256,
            "PIT generation manifest SHA",
        ),
        "pit_membership": _sha(
            args.expected_pit_membership_sha256, "PIT membership SHA"
        ),
        "snapshot_manifest": _sha(
            args.expected_snapshot_manifest_sha256, "snapshot manifest SHA"
        ),
    }
    if any(
        source_by_id[binding_id]["byte_sha256"] != digest
        for binding_id, digest in expected_source_bytes.items()
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution artifact source bytes differ from active bindings"
        )
    if (
        source_by_id["table_inventory"]["semantic_sha256"]
        != artifact["table_inventory_semantic_sha256"]
        or source_by_id["serving_inventory"]["semantic_sha256"]
        != artifact["serving_inventory_semantic_sha256"]
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "input-resolution inventory bindings are inconsistent"
        )

    field_rows = {row["field_name"]: row for row in artifact["field_rows"]}
    aquant_matrices = runtime.get("aquant_matrices")
    if not isinstance(aquant_matrices, Mapping) or set(field_rows) - set(
        aquant_matrices
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "runtime omits input-resolution field matrices"
        )
    resolved_matrices: dict[str, pd.DataFrame] = {}
    for field, row in field_rows.items():
        matrix = (
            aquant_matrices[field]
            .reindex(
                index=eligibility_mask.index,
                columns=eligibility_mask.columns,
            )
            .astype(float)
        )
        if field in {"close", "vwap"}:
            matrix = matrix.where(matrix > 0.0)
        elif field == "turnover_rate":
            matrix = matrix.where(matrix >= 0.0)
        matrix = matrix.where(eligibility_mask)
        if aquant_eval.matrix_hash_descriptor_v4_1(matrix) != row["matrix"]:
            raise FactorV4_1SameSnapshotRunnerError(
                f"runtime matrix differs from input-resolution proof: {field}"
            )
        resolved_matrices[field] = matrix
    return {
        "aquant_matrices": resolved_matrices,
        "candidate_signal_descriptors": {
            row["name"]: copy.deepcopy(row["signal_matrix"])
            for row in artifact["candidate_rows"]
        },
        "semantic_sha256": resolution["semantic_sha256"],
        "artifact_sha256": resolution["artifact_sha256"],
        "artifact": copy.deepcopy(dict(artifact)),
    }


def _failure_reason(exc: Exception) -> str:
    message = " ".join(str(exc).strip().split())
    return f"{type(exc).__name__}:{message}" if message else type(exc).__name__


def _evaluate_bound_idea(
    idea: Mapping[str, Any],
    *,
    aquant_matrices: Mapping[str, pd.DataFrame],
    eligibility_mask: pd.DataFrame,
) -> pd.DataFrame:
    fields = idea.get("input_fields")
    if not isinstance(fields, list) or not fields or any(
        type(field) is not str or field not in aquant_matrices for field in fields
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            f"resolved candidate has unavailable bound inputs: {idea.get('name')}"
        )
    return aquant_eval.evaluate_pinned_idea_v4_1(
        idea=idea,
        matrices={field: aquant_matrices[field] for field in fields},
        eligibility_mask=eligibility_mask,
    )


def _precompute_resolved_new_candidates(
    *,
    contract: Any,
    ideas_by_name: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
    eligibility_mask: pd.DataFrame,
) -> tuple[dict[str, tuple[dict[str, Any], np.ndarray]], bool]:
    aquant_matrices = runtime.get("aquant_matrices")
    expected_signal_descriptors = runtime.get("candidate_signal_descriptors")
    if not isinstance(aquant_matrices, Mapping):
        raise FactorV4_1SameSnapshotRunnerError(
            "resolved-input mode requires the complete bound A_quant matrix map"
        )
    if not isinstance(expected_signal_descriptors, Mapping) or set(
        expected_signal_descriptors
    ) != set(contract.PREDECLARED_BLOCKED_CANDIDATE_NAMES):
        raise FactorV4_1SameSnapshotRunnerError(
            "resolved-input mode requires exact candidate signal proofs"
        )
    forward_return = runtime["statistical_context"].forward_return
    monthly_dates = runtime["monthly_dates"]
    resolved: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for name in sorted(contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES):
        try:
            signal = _evaluate_bound_idea(
                ideas_by_name[name],
                aquant_matrices=aquant_matrices,
                eligibility_mask=eligibility_mask,
            )
            if (
                aquant_eval.matrix_hash_descriptor_v4_1(signal)
                != expected_signal_descriptors[name]
            ):
                raise FactorV4_1SameSnapshotRunnerError(
                    f"resolved candidate signal differs from bound proof: {name}"
                )
            resolved[name] = _signal_summary(
                signal,
                eligibility_mask,
                forward_return,
                monthly_dates,
            )
        except KeyError as exc:
            raise FactorV4_1SameSnapshotRunnerError(
                f"predeclared fundamental candidate is not bound: {name}"
            ) from exc
        except Exception as exc:
            raise FactorV4_1SameSnapshotRunnerError(
                "predeclared fundamental candidates must resolve atomically: "
                f"{name}:{_failure_reason(exc)}"
            ) from exc

    turnover_signals: dict[str, pd.DataFrame] = {}
    turnover_coverage: dict[str, int] = {}
    for name in sorted(contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES):
        try:
            signal = _evaluate_bound_idea(
                ideas_by_name[name],
                aquant_matrices=aquant_matrices,
                eligibility_mask=eligibility_mask,
            ).where(eligibility_mask)
            if (
                aquant_eval.matrix_hash_descriptor_v4_1(signal)
                != expected_signal_descriptors[name]
            ):
                raise FactorV4_1SameSnapshotRunnerError(
                    f"resolved candidate signal differs from bound proof: {name}"
                )
        except KeyError as exc:
            raise FactorV4_1SameSnapshotRunnerError(
                f"predeclared turnover candidate is not bound: {name}"
            ) from exc
        except Exception as exc:
            raise FactorV4_1SameSnapshotRunnerError(
                f"predeclared turnover evaluation failed: {name}:{_failure_reason(exc)}"
            ) from exc
        turnover_signals[name] = signal
        turnover_coverage[name] = len(
            _rank_ic_series(signal, forward_return, monthly_dates)
        )
    turnover_resolved = all(
        count >= MIN_VALID_MONTHS for count in turnover_coverage.values()
    )
    if turnover_resolved:
        for name, signal in turnover_signals.items():
            try:
                resolved[name] = _signal_summary(
                    signal,
                    eligibility_mask,
                    forward_return,
                    monthly_dates,
                    minimum_valid_rank_ic_months=MIN_VALID_MONTHS,
                )
            except Exception as exc:
                raise FactorV4_1SameSnapshotRunnerError(
                    "predeclared turnover candidates must resolve atomically: "
                    f"{name}:{_failure_reason(exc)}"
                ) from exc
    return resolved, turnover_resolved


def _validate_evaluation_profile(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    base_names: set[str],
    contract: Any,
) -> str:
    new_rows = [row for row in evaluations if row["name"] not in base_names]
    turnover_names = {
        row["name"]
        for row in new_rows
        if row["status"] == contract.STATUS_TURNOVER_BLOCKED
    }
    fundamental_names = {
        row["name"]
        for row in new_rows
        if row["status"] == contract.STATUS_FUNDAMENTAL_BLOCKED
    }
    evaluated_names = {
        row["name"]
        for row in new_rows
        if row["status"] == contract.STATUS_EVALUATED
    }
    all_new_names = {row["name"] for row in new_rows}
    original_names = all_new_names - set(contract.PREDECLARED_BLOCKED_CANDIDATE_NAMES)
    if (
        evaluated_names == original_names
        and turnover_names
        == set(contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES)
        and fundamental_names
        == set(contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES)
    ):
        profile = contract.ACCOUNTING_PROFILE_LEGACY
    elif (
        evaluated_names
        == original_names
        | set(contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES)
        and turnover_names
        == set(contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES)
        and not fundamental_names
    ):
        profile = contract.ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
    elif (
        evaluated_names == all_new_names
        and not turnover_names
        and not fundamental_names
    ):
        profile = contract.ACCOUNTING_PROFILE_FULLY_RESOLVED
    else:
        raise FactorV4_1SameSnapshotRunnerError(
            "new-candidate evaluations do not match an exact predefined "
            "27/2/8, 35/2/0, or 37/0/0 profile"
        )
    if any(
        row["status"] != contract.STATUS_EVALUATED
        and any(
            row[key] is not None
            for key in ("signal_sha256", "finite_ratio", "raw_p_value")
        )
        for row in new_rows
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "non-evaluated new candidates must have null statistics"
        )
    return profile


def _evaluate_candidates(
    preflight: Mapping[str, Any],
    governed: Mapping[str, Any],
    matrices: Mapping[str, pd.DataFrame],
    runtime: Mapping[str, Any],
    *,
    resolve_predeclared_input_fields: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    contract = _contract_module()
    base_by_name = {
        candidate.name: candidate for candidate in preflight["base_candidates"]
    }
    ideas_by_name = {idea["name"]: idea for idea in preflight["bound_ideas"]}
    no_label_rows = {
        row["name"]: row for row in preflight["diagnostic"]["rows"]
    }
    base_names = {
        row["name"] for row in preflight["base_catalog"]["candidates"]
    }
    eligibility_mask = governed["eligibility_mask"]
    context = runtime["signal_context"]
    forward_return = runtime["statistical_context"].forward_return
    monthly_dates = runtime["monthly_dates"]
    resolved_rows: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    turnover_resolved = False
    if resolve_predeclared_input_fields:
        resolved_rows, turnover_resolved = _precompute_resolved_new_candidates(
            contract=contract,
            ideas_by_name=ideas_by_name,
            runtime=runtime,
            eligibility_mask=eligibility_mask,
        )
    evaluations: list[dict[str, Any]] = []
    monthly_signals: dict[str, np.ndarray] = {}
    for catalog_row in preflight["formal_catalog"]["candidates"]:
        name = catalog_row["name"]
        if name in base_names:
            try:
                signal = compute_candidate_signal(
                    base_by_name[name],
                    context=context,
                    expression_inputs=runtime["expression_inputs"],
                    formulaic_primitives=runtime["formulaic_primitives"],
                ).where(eligibility_mask)
                summary, compact = _signal_summary(
                    signal, eligibility_mask, forward_return, monthly_dates
                )
                evaluations.append(
                    {
                        "name": name,
                        "status": contract.STATUS_EVALUATED,
                        **summary,
                        "failure_reason": None,
                    }
                )
                monthly_signals[name] = compact
            except Exception as exc:
                evaluations.append(
                    {
                        "name": name,
                        "status": contract.STATUS_COMPUTE_FAILED,
                        "signal_sha256": None,
                        "finite_ratio": None,
                        "raw_p_value": None,
                        "failure_reason": _failure_reason(exc),
                    }
                )
            continue

        row = no_label_rows[name]
        if row["status"] == no_label_contract.STATUS_TURNOVER_BLOCKED:
            if name not in contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES:
                raise FactorV4_1SameSnapshotRunnerError(
                    f"unexpected turnover-blocked predecessor candidate: {name}"
                )
            if resolve_predeclared_input_fields and turnover_resolved:
                summary, compact = resolved_rows[name]
                evaluations.append(
                    {
                        "name": name,
                        "status": contract.STATUS_EVALUATED,
                        **summary,
                        "failure_reason": None,
                    }
                )
                monthly_signals[name] = compact
                continue
            status = contract.STATUS_TURNOVER_BLOCKED
            reason = (
                "turnover_rate_rank_ic_coverage_below_three_months"
                if resolve_predeclared_input_fields
                else "turnover_rate_not_in_authorized_same_snapshot_projection"
            )
        elif row["status"] == no_label_contract.STATUS_FUNDAMENTAL_BLOCKED:
            if name not in contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES:
                raise FactorV4_1SameSnapshotRunnerError(
                    f"unexpected fundamental-blocked predecessor candidate: {name}"
                )
            if resolve_predeclared_input_fields:
                summary, compact = resolved_rows[name]
                evaluations.append(
                    {
                        "name": name,
                        "status": contract.STATUS_EVALUATED,
                        **summary,
                        "failure_reason": None,
                    }
                )
                monthly_signals[name] = compact
                continue
            status = contract.STATUS_FUNDAMENTAL_BLOCKED
            reason = "standalone_aquant_fundamental_semantics_not_proven"
        elif row["status"] == no_label_contract.STATUS_SIGNAL_DIAGNOSTIC:
            try:
                idea = ideas_by_name[name]
                signal = aquant_eval.evaluate_pinned_idea_v4_1(
                    idea=idea,
                    matrices={field: matrices[field] for field in idea["input_fields"]},
                    eligibility_mask=eligibility_mask,
                )
                summary, compact = _signal_summary(
                    signal, eligibility_mask, forward_return, monthly_dates
                )
                evaluations.append(
                    {
                        "name": name,
                        "status": contract.STATUS_EVALUATED,
                        **summary,
                        "failure_reason": None,
                    }
                )
                monthly_signals[name] = compact
            except Exception as exc:
                evaluations.append(
                    {
                        "name": name,
                        "status": contract.STATUS_COMPUTE_FAILED,
                        "signal_sha256": None,
                        "finite_ratio": None,
                        "raw_p_value": None,
                        "failure_reason": _failure_reason(exc),
                    }
                )
            continue
        else:
            raise FactorV4_1SameSnapshotRunnerError(
                f"unsupported no-label predecessor status: {row['status']}"
            )
        evaluations.append(
            {
                "name": name,
                "status": status,
                "signal_sha256": None,
                "finite_ratio": None,
                "raw_p_value": None,
                "failure_reason": reason,
            }
        )
    if len(evaluations) != EXPECTED_TOTAL_COUNT:
        raise FactorV4_1SameSnapshotRunnerError(
            "same-snapshot evaluation accounting is not exactly 267"
        )
    return evaluations, monthly_signals


def _monthly_correlation_rows(
    monthly_signals: Mapping[str, np.ndarray],
    monthly_dates: Sequence[pd.Timestamp],
    new_names: set[str],
) -> list[dict[str, Any]]:
    compact = _compact_correlation_evidence(
        monthly_signals, monthly_dates, new_names
    )
    return _expand_compact_correlation_rows(compact)


def _compact_correlation_evidence(
    monthly_signals: Mapping[str, np.ndarray],
    monthly_dates: Sequence[pd.Timestamp],
    new_names: set[str],
) -> dict[str, Any]:
    names = sorted(monthly_signals)
    if not names or not new_names.intersection(names):
        raise FactorV4_1SameSnapshotRunnerError(
            "correlation diagnostic has no evaluated new candidates"
        )
    shape = next(iter(monthly_signals.values())).shape
    if any(values.shape != shape for values in monthly_signals.values()) or shape[0] != len(
        monthly_dates
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "compact monthly signal axes are inconsistent"
        )
    expected_pairs = [
        (left, right)
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if left in new_names or right in new_names
    ]
    month_strings = [
        pd.Timestamp(month_end).strftime("%Y-%m-%d")
        for month_end in monthly_dates
    ]
    if month_strings != sorted(set(month_strings)):
        raise FactorV4_1SameSnapshotRunnerError(
            "correlation month axis must be sorted and distinct"
        )
    name_index = {name: index for index, name in enumerate(names)}
    pair_rows: list[dict[str, Any]] = [
        {
            "left_index": name_index[left],
            "right_index": name_index[right],
            "valid_month_indices": [],
            "abs_spearman": [],
            "valid_common_symbol_count": [],
        }
        for left, right in expected_pairs
    ]
    pair_counts = {pair: 0 for pair in expected_pairs}
    pair_by_name = dict(zip(expected_pairs, pair_rows, strict=True))
    for month_index, _month_end in enumerate(monthly_dates):
        frame = pd.DataFrame(
            {
                name: monthly_signals[name][month_index]
                for name in names
            }
        ).replace([np.inf, -np.inf], np.nan)
        valid = frame.notna().astype(np.int32)
        common = valid.T.dot(valid)
        correlations = frame.corr(
            method="spearman", min_periods=MIN_COMMON_SYMBOLS
        )
        for left, right in expected_pairs:
            count = int(common.loc[left, right])
            value = correlations.loc[left, right]
            if count < MIN_COMMON_SYMBOLS or pd.isna(value):
                continue
            pair_row = pair_by_name[(left, right)]
            pair_row["valid_month_indices"].append(int(month_index))
            pair_row["abs_spearman"].append(float(abs(value)))
            pair_row["valid_common_symbol_count"].append(count)
            pair_counts[(left, right)] += 1
    incomplete = [pair for pair, count in pair_counts.items() if count < MIN_VALID_MONTHS]
    if incomplete:
        raise FactorV4_1SameSnapshotRunnerError(
            f"correlation pairs lack three valid months: {incomplete[:5]}"
        )
    return {
        "schema_version": "factor-v4.1-compact-monthly-correlation-input.v1",
        "candidate_names": names,
        "month_end_dates": month_strings,
        "new_candidate_names": sorted(new_names.intersection(names)),
        "expected_pair_count": len(expected_pairs),
        "observed_pair_count": len(expected_pairs),
        "minimum_valid_symbol_count_per_month": MIN_COMMON_SYMBOLS,
        "minimum_valid_month_count": MIN_VALID_MONTHS,
        "pair_rows": pair_rows,
    }


def _expand_compact_correlation_rows(
    compact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    names = list(compact["candidate_names"])
    months = list(compact["month_end_dates"])
    rows: list[dict[str, Any]] = []
    for pair in compact["pair_rows"]:
        left = names[int(pair["left_index"])]
        right = names[int(pair["right_index"])]
        for month_index, value, count in zip(
            pair["valid_month_indices"],
            pair["abs_spearman"],
            pair["valid_common_symbol_count"],
            strict=True,
        ):
            rows.append(
                {
                    "left_name": left,
                    "right_name": right,
                    "month_end": months[int(month_index)],
                    "abs_spearman": float(value),
                    "valid_common_symbol_count": int(count),
                }
            )
    return rows


def _build_correlation_diagnostic(
    contract: Any,
    *,
    cycle_id: str,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    screening: Mapping[str, Any],
    compact_correlation: Mapping[str, Any],
) -> dict[str, Any]:
    builder = contract.build_correlation_diagnostic_v4_1
    parameters = inspect.signature(builder).parameters
    common = {
        "cycle_id": cycle_id,
        "base_ontology": base_ontology,
        "formal_ontology": formal_ontology,
        "base_catalog": base_catalog,
        "formal_catalog": formal_catalog,
        "screening": screening,
    }
    if "compact_correlation" in parameters:
        return builder(
            **common,
            compact_correlation=compact_correlation,
        )
    return builder(
        **common,
        monthly_rows=_expand_compact_correlation_rows(compact_correlation),
    )


def _calendar_sha256(dates: pd.DatetimeIndex) -> str:
    values = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]
    if not values or values != sorted(set(values)):
        raise FactorV4_1SameSnapshotRunnerError(
            "calendar SHA requires exact sorted unique session dates"
        )
    return _semantic_sha(
        {
            "schema_version": "factor-screening-open-session-calendar.v1",
            "open_session_dates": values,
        }
    )


def _matrix_context(
    governed: Mapping[str, Any], monthly_dates: Sequence[pd.Timestamp]
) -> dict[str, Any]:
    eligibility = governed["eligibility_mask"]
    descriptor = aquant_eval.matrix_hash_descriptor_v4_1(
        eligibility.astype(float)
    )
    closed_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d") for date in monthly_dates
    ]
    calendar_sha = _calendar_sha256(eligibility.index)
    return {
        "date_axis_sha256": descriptor["date_axis_sha256"],
        "symbol_axis_sha256": descriptor["symbol_axis_sha256"],
        "eligibility_matrix_sha256": descriptor["matrix_sha256"],
        "session_scope_sha256": governed["scope_binding"][
            "descriptor_semantic_sha256"
        ],
        "session_count": int(eligibility.shape[0]),
        "symbol_count": int(eligibility.shape[1]),
        "calendar_sha256": calendar_sha,
        "closed_month_end_dates": closed_dates,
        "closed_month_end_axis_sha256": _semantic_sha(closed_dates),
    }


def _source_bindings(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    controls: Mapping[str, Any],
    governed: Mapping[str, Any],
    matrix_context: Mapping[str, Any],
    resolution: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    resolved_mode = getattr(args, "resolve_predeclared_input_fields", False)
    if resolved_mode and resolution is None:
        raise FactorV4_1SameSnapshotRunnerError(
            "resolved run-config requires a validated input-resolution bundle"
        )
    if not resolved_mode and resolution is not None:
        raise FactorV4_1SameSnapshotRunnerError(
            "legacy run-config must not bind input-resolution evidence"
        )
    values = {
        "code_sha256": controls["code_sha256"],
        "registry_file_sha256": governed["registry_row"]["byte_sha256"],
        "latest_pointer_sha256": governed["latest_row"]["byte_sha256"],
        "manifest_sha256": _sha(
            args.expected_snapshot_manifest_sha256,
            "snapshot manifest SHA",
        ),
        "market_data_input_sha256": governed["market_data_input_sha256"],
        "pit_sha256": _sha(
            args.expected_pit_membership_sha256, "PIT membership SHA"
        ),
        "calendar_sha256": matrix_context["calendar_sha256"],
        "fundamental_manifest_sha256": governed[
            "fundamental_generation_manifest_sha256"
        ],
        "run_config_sha256": _run_config_semantic_sha(
            preflight,
            resolve_predeclared_input_fields=resolved_mode,
            input_resolution_artifact_path=(
                resolution["artifact_path"] if resolution is not None else None
            ),
            input_resolution_artifact_sha256=(
                resolution["artifact_sha256"] if resolution is not None else None
            ),
            input_resolution_semantic_sha256=(
                resolution["semantic_sha256"] if resolution is not None else None
            ),
        ),
    }
    if set(values) != set(screening_v4.SOURCE_BINDING_FIELDS):
        raise FactorV4_1SameSnapshotRunnerError(
            "same-snapshot source-binding map is incomplete"
        )
    return {key: _sha(value, f"source binding {key}") for key, value in values.items()}


def _precheck_exact_once_output(private_root: Path, run_id: str, contract: Any) -> None:
    suffix = tuple(contract.PRIVATE_ROOT_SUFFIX)
    if tuple(private_root.parts[-len(suffix) :]) != suffix:
        raise FactorV4_1SameSnapshotRunnerError(
            "private root does not match the same-snapshot contract suffix"
        )
    if os.path.lexists(private_root):
        metadata = os.lstat(private_root)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise FactorV4_1SameSnapshotRunnerError(
                "existing private root must be an owner-only real directory"
            )
    if os.path.lexists(private_root / run_id):
        raise FactorV4_1SameSnapshotRunnerError(
            "same-snapshot exact-once run path already exists"
        )


def _revalidate_all_inputs(
    args: argparse.Namespace,
    baseline_preflight: Mapping[str, Any],
    baseline_controls: Mapping[str, Any],
    baseline_governed: Mapping[str, Any],
    baseline_resolution: Mapping[str, Any] | None = None,
) -> None:
    current_preflight = _preflight_bundles(args)
    current_controls = _read_control_inputs(args, current_preflight)
    current_governed = _bind_governed_inputs(
        args, current_preflight, current_controls
    )
    current_resolution = _read_input_resolution_bundle(args)
    predecessor_keys = ("discovery", "formal", "no_label", "cutoff")
    if any(
        current_preflight[key]["report_semantic_sha256"]
        != baseline_preflight[key]["report_semantic_sha256"]
        for key in predecessor_keys
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "predecessor bundle changed before publication"
        )
    if (
        current_controls["code_sha256"] != baseline_controls["code_sha256"]
        or current_controls["protected_rows"]
        != baseline_controls["protected_rows"]
        or current_governed["market_inventory"]
        != baseline_governed["market_inventory"]
        or current_governed["scope_binding"]
        != baseline_governed["scope_binding"]
        or (
            current_resolution is None
            and baseline_resolution is not None
        )
        or (
            current_resolution is not None
            and baseline_resolution is None
        )
        or (
            current_resolution is not None
            and baseline_resolution is not None
            and (
                current_resolution["artifact"] != baseline_resolution["artifact"]
                or current_resolution["artifact_sha256"]
                != baseline_resolution["artifact_sha256"]
                or current_resolution["semantic_sha256"]
                != baseline_resolution["semantic_sha256"]
            )
        )
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "governed input changed before publication"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit-path research-only Factor v4.1 same-snapshot screen."
    )
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--discovery-bundle-path", required=True)
    parser.add_argument("--expected-discovery-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-discovery-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--formal-bundle-path", required=True)
    parser.add_argument("--expected-formal-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-formal-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--no-label-bundle-path", required=True)
    parser.add_argument("--expected-no-label-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-no-label-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--cutoff-bundle-path", required=True)
    parser.add_argument("--expected-cutoff-readback-report-sha256", required=True)
    parser.add_argument(
        "--expected-cutoff-readback-report-semantic-sha256", required=True
    )
    parser.add_argument("--snapshot-manifest-path", required=True)
    parser.add_argument("--expected-snapshot-manifest-sha256", required=True)
    parser.add_argument("--pit-membership-path", required=True)
    parser.add_argument("--expected-pit-membership-sha256", required=True)
    parser.add_argument("--pit-generation-manifest-path", required=True)
    parser.add_argument("--expected-pit-generation-manifest-sha256", required=True)
    parser.add_argument("--components-path", required=True)
    parser.add_argument("--expected-components-sha256", required=True)
    parser.add_argument("--table-root", required=True)
    parser.add_argument("--expected-table-inventory-sha256", required=True)
    parser.add_argument("--serving-root", required=True)
    parser.add_argument("--expected-serving-inventory-sha256", required=True)
    parser.add_argument(
        "--resolve-predeclared-input-fields",
        action="store_true",
        help=(
            "Opt into the proof-bound research-only resolution lane for the "
            "exact ten predeclared input-blocked candidates."
        ),
    )
    parser.add_argument("--input-resolution-bundle-path")
    parser.add_argument("--expected-input-resolution-artifact-sha256")
    parser.add_argument("--expected-input-resolution-semantic-sha256")
    parser.add_argument("--code-binding", action="append", required=True)
    parser.add_argument("--protected-binding", action="append", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build, independently read back, and return one private report bundle."""

    run_id = _run_id(args.run_id)
    private_root = _absolute_path(args.private_root, "private root")
    preflight = _preflight_bundles(args)
    contract = _contract_module()
    _precheck_exact_once_output(private_root, run_id, contract)
    controls = _read_control_inputs(args, preflight)
    governed = _bind_governed_inputs(args, preflight, controls)
    resolution = _read_input_resolution_bundle(args)
    resolved_mode = getattr(args, "resolve_predeclared_input_fields", False)
    matrices = _load_market_matrices(
        governed["table_root"],
        governed["table_inventory"],
        governed["eligibility_mask"],
        serving_root=governed["serving_root"] if resolved_mode else None,
        serving_inventory=(
            governed["serving_inventory"] if resolved_mode else None
        ),
        include_turnover_rate=resolved_mode,
    )
    runtime = _runtime_inputs(governed, matrices)
    evaluation_runtime = dict(runtime)
    if resolution is not None:
        evaluation_runtime.update(
            _bind_resolution_to_runtime(
                resolution,
                args=args,
                governed=governed,
                runtime=runtime,
            )
        )
    matrix_context = _matrix_context(governed, runtime["monthly_dates"])
    source_bindings = _source_bindings(
        args,
        preflight,
        controls,
        governed,
        matrix_context,
        resolution=resolution,
    )
    evaluations, monthly_signals = _evaluate_candidates(
        preflight,
        governed,
        matrices,
        evaluation_runtime,
        resolve_predeclared_input_fields=resolved_mode,
    )
    base_names = {
        row["name"] for row in preflight["base_catalog"]["candidates"]
    }
    accounting_profile = _validate_evaluation_profile(
        evaluations,
        base_names=base_names,
        contract=contract,
    )
    if (
        resolved_mode
        and accounting_profile == contract.ACCOUNTING_PROFILE_LEGACY
    ) or (
        not resolved_mode
        and accounting_profile != contract.ACCOUNTING_PROFILE_LEGACY
    ):
        raise FactorV4_1SameSnapshotRunnerError(
            "evaluation accounting profile is inconsistent with resolved-input mode"
        )

    # This is the runner's only call into full-catalog family BH.  The pure
    # contract converts every non-evaluated row to p=1 inside its family while
    # retaining the exact 267-row denominator.
    diagnostic_signal_sha256_by_name = {
        row["name"]: row["signal_matrix"]["matrix_sha256"]
        for row in preflight["diagnostic"]["rows"]
        if row["status"] == no_label_contract.STATUS_SIGNAL_DIAGNOSTIC
    }
    if len(diagnostic_signal_sha256_by_name) != 27:
        raise FactorV4_1SameSnapshotRunnerError(
            "no-label predecessor must bind exactly 27 diagnostic signal hashes"
        )
    screening = contract.build_same_snapshot_screening_v4_1(
        cycle_id=args.cycle_id,
        base_ontology=preflight["base_ontology"],
        formal_ontology=preflight["formal_ontology"],
        base_catalog=preflight["base_catalog"],
        formal_catalog=preflight["formal_catalog"],
        evaluations=evaluations,
        diagnostic_signal_sha256_by_name=diagnostic_signal_sha256_by_name,
        source_bindings=source_bindings,
        matrix_context=matrix_context,
        input_resolution_semantic_sha256=(
            resolution["semantic_sha256"] if resolution is not None else None
        ),
    )
    new_names = {
        row["name"] for row in preflight["formal_catalog"]["candidates"]
    } - {row["name"] for row in preflight["base_catalog"]["candidates"]}
    compact_correlation = _compact_correlation_evidence(
        monthly_signals, runtime["monthly_dates"], new_names
    )
    del evaluation_runtime
    del runtime
    del matrices
    gc.collect()
    correlation = _build_correlation_diagnostic(
        contract,
        cycle_id=args.cycle_id,
        base_ontology=preflight["base_ontology"],
        formal_ontology=preflight["formal_ontology"],
        base_catalog=preflight["base_catalog"],
        formal_catalog=preflight["formal_catalog"],
        screening=screening,
        compact_correlation=compact_correlation,
    )
    artifacts = {
        contract.SCREENING_FILENAME: screening,
        contract.CORRELATION_FILENAME: correlation,
    }
    bundle_contract = contract.build_private_bundle_contract_v4_1(
        expected_artifacts=artifacts,
        base_ontology=preflight["base_ontology"],
        formal_ontology=preflight["formal_ontology"],
        base_catalog=preflight["base_catalog"],
        formal_catalog=preflight["formal_catalog"],
    )

    def revalidate_inputs() -> None:
        _revalidate_all_inputs(
            args,
            preflight,
            controls,
            governed,
            baseline_resolution=resolution,
        )

    published = private_io.publish_private_bundle(
        private_root=private_root,
        run_id=run_id,
        artifacts=artifacts,
        contract=bundle_contract,
        revalidate_inputs=revalidate_inputs,
    )
    independent = private_io.readback_private_bundle(
        published["bundle_path"], contract=bundle_contract
    )
    if independent.get("accepted") is not True:
        raise FactorV4_1SameSnapshotRunnerError(
            "independent same-snapshot bundle readback failed"
        )
    result = {
        "accepted": True,
        "predecessors_validated": True,
        "bundle_path": independent["bundle_path"],
        "run_config_sha256": source_bindings["run_config_sha256"],
        "candidate_count": EXPECTED_TOTAL_COUNT,
        "evaluated_count": sum(
            row["status"] == contract.STATUS_EVALUATED for row in evaluations
        ),
        "compute_failed_count": sum(
            row["status"] == contract.STATUS_COMPUTE_FAILED for row in evaluations
        ),
        "turnover_data_blocked_count": sum(
            row["status"] == contract.STATUS_TURNOVER_BLOCKED
            for row in evaluations
        ),
        "fundamental_semantic_blocked_count": sum(
            row["status"] == contract.STATUS_FUNDAMENTAL_BLOCKED
            for row in evaluations
        ),
        "research_report_only": True,
        "new_risk_authorized": False,
    }
    if resolved_mode:
        result["accounting_profile"] = accounting_profile
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (FactorV4_1SameSnapshotRunnerError, ValueError, OSError) as exc:
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
