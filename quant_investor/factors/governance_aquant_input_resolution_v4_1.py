"""Fail-closed no-label input resolution for ten pinned A_quant v4.1 rows.

This contract records only that the exact input matrices and pinned expressions
were reproducible under myQuant's native PIT semantics.  It deliberately makes
no claim that those data semantics equal A_quant's data loader and grants no
screening, admission, registry, production, portfolio, or trading authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import pandas as pd

from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator
from quant_investor.factors import governance_private_bundle_io as private_io


PROTOCOL_VERSION = "v4.1"
EXPECTED_CYCLE_ID = "cn_full_a_v4_1_20260717"
SCHEMA_VERSION = "factor-governance-aquant-input-resolution.v4.1"
READBACK_SCHEMA_VERSION = (
    "factor-governance-aquant-input-resolution-readback.v4.1"
)
ARTIFACT_FILENAME = "aquant_input_resolution.v4_1.json"
READBACK_FILENAME = "aquant_input_resolution_readback.v4_1.json"
BUNDLE_INPUT_FILENAMES = (ARTIFACT_FILENAME,)
BUNDLE_FILENAMES = (ARTIFACT_FILENAME, READBACK_FILENAME)
PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_aquant_input_resolution",
)
SEMANTICS_MODE = "myquant_native_pit_semantics"
CLAIM_SCOPE = "exact_ten_input_resolution_no_label_only"

RESOLVED_CANDIDATE_NAMES = (
    "alpha_growth_quality_profit_roa",
    "alpha_quality_low_debt_assets",
    "alpha_quality_value_cash_fcf",
    "alpha_turnover_low_20d",
    "alpha_turnover_low_60d",
    "alpha_vwap_cash_quality_160",
    "alpha_vwap_growth_profit_160",
    "alpha_vwap_low_debt_160",
    "alpha_vwap_quality_roa_160",
    "alpha_vwap_quality_roe_160",
)

# These identities came from the exact formal/no-label predecessor chain.  They
# prevent a same-named expression or a substituted mapping from being accepted.
EXPECTED_CANDIDATES: dict[str, dict[str, Any]] = {
    "alpha_growth_quality_profit_roa": {
        "input_fields": ["fin_net_profit_yoy", "fin_roa"],
        "source_definition_sha256": "69f42b9ba40c36582fe01299a3a5333d8bb99dd45390c73c3a193e1cd4f795e1",
        "normalized_ast_sha256": "6244acfa806a48f251f7aeeb1a5a5c18e788bbda53010d18aabb3b9d51074abd",
        "catalog_definition_sha256": "b9a14024a8f2de68e7804ff1938f38867d1c0eef5acbabc7d3447f150503be9e",
        "mapping_semantic_sha256": "ca7d40b598ed6825454c10928c27e1c2ac3a1e62c2d9e47515ce7d98eab7d22c",
    },
    "alpha_quality_low_debt_assets": {
        "input_fields": ["fin_debt_to_assets"],
        "source_definition_sha256": "abfbd2b5c1074a8b6c882a7552b3b38166866bb7e9cb04ec3510b932c52db8f0",
        "normalized_ast_sha256": "3c39f29f9d2bd21bc1e162bef28f27a98bbee56982cb57bd7839955462714474",
        "catalog_definition_sha256": "669d42a935f30e14edd5645cbe1e614929e5b8f29c178bfe49a3cd2a279604aa",
        "mapping_semantic_sha256": "d0487031ee97e4df7d6c5d565a97699a49a940e40f979d6ab5f3e6a448d6345d",
    },
    "alpha_quality_value_cash_fcf": {
        "input_fields": ["fcf_to_price", "fin_ocf_to_profit"],
        "source_definition_sha256": "b69d0901f766fa09c090f6badf5954b18c99bc4b08a3867e8f0d1a567a720c30",
        "normalized_ast_sha256": "17e7cffb14066c86b4806b8c81015aba7bb6598abbb4e8045dda3bfb1a79b7cc",
        "catalog_definition_sha256": "e5ef6eeb318f4f1464ec0fb46fc75c3a6edcabc1ccdac2410790c6aa131b2a2f",
        "mapping_semantic_sha256": "eef303b6ddf2f6d298e7046c74707eac1400fa29ad23b9765887d29670ef27dc",
    },
    "alpha_turnover_low_20d": {
        "input_fields": ["turnover_rate"],
        "source_definition_sha256": "2be718d593223bd870a5de045befab935d8d0ea7887cb730d25eeecc1b9df9d7",
        "normalized_ast_sha256": "925586e9490b7c45f5c1551cb147bcb6f301aa3e687fc72a899de98b800b4e34",
        "catalog_definition_sha256": "d2f12508c62a8237269e4b9fcba88be702c56fd0173c8d9f6d472c710372131f",
        "mapping_semantic_sha256": "e8425ad521f13c4d93254705c4841b7a732d9dd6bb201f17d820cbcac0077012",
    },
    "alpha_turnover_low_60d": {
        "input_fields": ["turnover_rate"],
        "source_definition_sha256": "82932bffb0ed7fa59dc379cc50913152119ecccfabdd2745f76c93197a791dfb",
        "normalized_ast_sha256": "848e0ee1713fe989082d07601d8be58aff2ea97dd79abf696012aaff436b7b9c",
        "catalog_definition_sha256": "3101a0b1ae092c8ba4b56babbeb8a823f1cdfcc94a1650407de1ec9608fd0396",
        "mapping_semantic_sha256": "a270ebd1ada8b45e31a1cb85ed7e099766ae69d86ee9635ebb84a2cf14be9bd2",
    },
    "alpha_vwap_cash_quality_160": {
        "input_fields": ["close", "fin_ocf_to_profit", "vwap"],
        "source_definition_sha256": "5b4cd5785af064f2bc3e5fb592cdc6afc62230628909f0e62617f42781a8d889",
        "normalized_ast_sha256": "01c9771e40b035423f130fb625b1f5a2584215e615d422068c4ce6aabd47c39d",
        "catalog_definition_sha256": "d3f62db531cdd740f099cf7b547b59b017e26e02fcec4de1f89f1ed66044cf6c",
        "mapping_semantic_sha256": "ac683f5eef2307b2e0a7d072fe583293b22dd65708bb50e4a471eb5da27d5448",
    },
    "alpha_vwap_growth_profit_160": {
        "input_fields": ["close", "fin_net_profit_yoy", "vwap"],
        "source_definition_sha256": "7435d0f381de4756d1fcc53623d068f83ce5a717ab03953d4d2efa7cf47c9940",
        "normalized_ast_sha256": "2b42cb7cc5266ce0c157c1e5c62083ac5eecf330069eafbc1682a9f63b2e52ef",
        "catalog_definition_sha256": "4263a1a916f4029d562a2afec8cf969c1a653fe2a0c3762ff1a466dc4f0c0a52",
        "mapping_semantic_sha256": "bcacdf4a9271e75b5b5f178b1cba44eeeac4a23037649e31db07a91ed64b262d",
    },
    "alpha_vwap_low_debt_160": {
        "input_fields": ["close", "fin_debt_to_assets", "vwap"],
        "source_definition_sha256": "2693841fe296275d40ae42d7089c587032c0e0836e9b1a0fc078b7c56e3588b4",
        "normalized_ast_sha256": "749db87978d9defe87de2e00800309aa4f5a9e1de0e9645120a6f3454976f88b",
        "catalog_definition_sha256": "a0ef487743da1ac74ddcef1857c095236fbdab8eaac74a685aa9b8a36fefcb50",
        "mapping_semantic_sha256": "ff61c994799fe9365a9774f0bb0190eaf78f088fcbd9c29ff6f88d534b3940a6",
    },
    "alpha_vwap_quality_roa_160": {
        "input_fields": ["close", "fin_roa", "vwap"],
        "source_definition_sha256": "1adfb7502b4e0548cfa68147f511f76739cb5668f3d3ab1edb010ff9ba19219a",
        "normalized_ast_sha256": "9e7a33a2bae26cd2be9b1202b38fc7927f0380ba6596a09f4d7c08560a5c66ab",
        "catalog_definition_sha256": "c1b309a9fbf1cdf71d60575e4ba3cdabc2446eb343392da52745c7dd012ad38d",
        "mapping_semantic_sha256": "b6da96bc20cd117e59aaf39a9502a129f32ecaffa144e93f5e325da5cbcc2b12",
    },
    "alpha_vwap_quality_roe_160": {
        "input_fields": ["close", "fin_roe", "vwap"],
        "source_definition_sha256": "c1de4fefdce2486bacfced0f627bfec91aa059f0c96f18ac01c2c374708c2ac2",
        "normalized_ast_sha256": "3d165c051fbf011821a3d873cd397fe577df739154f4ff77c951d35f610a3d14",
        "catalog_definition_sha256": "696ec31ca9671a6df80aba715d9ea63e8b266eb6b66cd8b263c7f069e1f4e7b6",
        "mapping_semantic_sha256": "a9edb675ff92a15da5a10531e0a1536f6f79aa6678b9330e32087fb1c485a977",
    },
}

FIELD_POLICIES: dict[str, dict[str, str]] = {
    "close": {
        "source_surface": "strict_snapshot_table",
        "source_column": "close",
        "units": "CNY_per_share",
        "scaling_policy": "identity",
        "availability_policy": "same_trade_date_close_observation",
        "fallback_policy": "none",
        "zero_negative_policy": "nonpositive_values_rejected_as_missing",
        "missing_policy": "preserve_missing_no_fill",
    },
    "vwap": {
        "source_surface": "strict_snapshot_table",
        "source_column": "amount_times_10_divided_by_vol",
        "units": "CNY_per_share",
        "scaling_policy": "tushare_amount_kCNY_times_10_over_volume_hundred_shares",
        "availability_policy": "same_trade_date_close_observation",
        "fallback_policy": "none",
        "zero_negative_policy": "nonpositive_volume_or_nonpositive_result_rejected_as_missing",
        "missing_policy": "preserve_missing_no_fill",
    },
    "turnover_rate": {
        "source_surface": "strict_snapshot_serving",
        "source_column": "turnover_rate",
        "units": "percent_of_float_shares",
        "scaling_policy": "identity_no_ratio_conversion",
        "availability_policy": "same_trade_date_serving_observation",
        "fallback_policy": "none_table_forbidden_no_forward_fill",
        "zero_negative_policy": "zero_allowed_negative_rejected_as_missing",
        "missing_policy": "sparse_values_preserved_no_fill_ts_mean_min_periods_one",
    },
    "fin_roe": {
        "source_surface": "active_fundamental_generation",
        "source_column": "fundamental_daily.fin_roe",
        "units": "ratio",
        "scaling_policy": "myquant_percent_to_ratio",
        "availability_policy": "myquant_f_ann_date_then_ann_date_then_explicit_date_pit_asof",
        "fallback_policy": "direct_roe_dt_then_roe_no_legacy",
        "zero_negative_policy": "finite_zero_and_negative_preserved",
        "missing_policy": "preserve_missing_until_next_pit_available_value",
    },
    "fin_roa": {
        "source_surface": "active_fundamental_generation",
        "source_column": "fundamental_daily.fin_roa",
        "units": "ratio",
        "scaling_policy": "myquant_percent_to_ratio",
        "availability_policy": "myquant_f_ann_date_then_ann_date_then_explicit_date_pit_asof",
        "fallback_policy": "direct_roa_no_legacy",
        "zero_negative_policy": "finite_zero_and_negative_preserved",
        "missing_policy": "preserve_missing_until_next_pit_available_value",
    },
    "fin_debt_to_assets": {
        "source_surface": "active_fundamental_generation",
        "source_column": "fundamental_daily.fin_debt_to_assets",
        "units": "ratio",
        "scaling_policy": "myquant_percent_to_ratio",
        "availability_policy": "myquant_f_ann_date_then_ann_date_then_explicit_date_pit_asof",
        "fallback_policy": "direct_debt_to_assets_else_total_liabilities_over_positive_total_assets_no_legacy",
        "zero_negative_policy": "finite_zero_and_negative_preserved_positive_fallback_denominator",
        "missing_policy": "preserve_missing_until_next_pit_available_value",
    },
    "fin_net_profit_yoy": {
        "source_surface": "active_fundamental_generation",
        "source_column": "fundamental_daily.fin_net_profit_yoy",
        "units": "ratio",
        "scaling_policy": "myquant_percent_to_ratio",
        "availability_policy": "myquant_f_ann_date_then_ann_date_then_explicit_date_pit_asof",
        "fallback_policy": "direct_netprofit_yoy_else_report_period_change_over_positive_prior_profit_no_legacy",
        "zero_negative_policy": "finite_zero_and_negative_results_preserved_positive_fallback_denominator",
        "missing_policy": "preserve_missing_until_next_pit_available_value",
    },
    "fin_ocf_to_profit": {
        "source_surface": "active_fundamental_generation",
        "source_column": "fundamental_daily.fin_ocf_to_profit",
        "units": "ratio",
        "scaling_policy": "myquant_percent_to_ratio",
        "availability_policy": "myquant_f_ann_date_then_ann_date_then_explicit_date_pit_asof",
        "fallback_policy": "direct_ocf_to_profit_else_operating_cashflow_over_positive_profit_no_legacy",
        "zero_negative_policy": "finite_zero_and_negative_results_preserved_positive_fallback_denominator",
        "missing_policy": "preserve_missing_until_next_pit_available_value",
    },
    "fcf_to_price": {
        "source_surface": "active_fundamental_generation",
        "source_column": "fundamental_daily.fcf_to_price",
        "units": "ratio",
        "scaling_policy": "free_cash_flow_rmb_over_total_market_value_rmb",
        "availability_policy": "myquant_f_ann_date_then_ann_date_then_explicit_date_pit_asof",
        "fallback_policy": "free_cashflow_else_operating_cashflow_minus_capex_over_positive_market_value_no_legacy",
        "zero_negative_policy": "finite_zero_and_negative_numerator_preserved_positive_denominator",
        "missing_policy": "preserve_missing_until_next_pit_available_value",
    },
}

PREDECESSOR_BINDING_IDS = (
    "formal_catalog",
    "formal_catalog_readback",
    "no_label_diagnostic",
    "no_label_readback",
    "operator_runtime_equivalence_proof",
    "operator_runtime_equivalence_readback",
)
CODE_BINDING_IDS = (
    "aquant_no_label_evaluator",
    "formal_catalog_materialization",
    "fundamental_generation",
    "fundamental_mart",
    "no_label_diagnostic",
    "operator_runtime_equivalence",
    "pit_fundamentals",
    "private_bundle_io",
    "resolution_builder",
    "resolution_contract",
    "source_v4_1",
)
SOURCE_BINDING_IDS = (
    "fundamental_daily",
    "fundamental_generation_manifest",
    "fundamental_pointer",
    "latest_pointer",
    "pit_components",
    "pit_generation_manifest",
    "pit_membership",
    "serving_inventory",
    "snapshot_manifest",
    "table_inventory",
)

AUTHORITY_FIELDS = {
    "screening_authority": False,
    "screening_eligible": False,
    "bh_authority": False,
    "family_bh_authoritative": False,
    "qualification": False,
    "qualified": False,
    "admission_authority": False,
    "formal_admission_authority": False,
    "proposal_authority": False,
    "proposal_eligible": False,
    "registry_authority": False,
    "registry_entry_created": False,
    "production_apply_enabled": False,
    "new_risk_eligible": False,
    "new_risk_authorized": False,
    "portfolio_authority": False,
    "trading_authority": False,
}
SIDE_EFFECT_FIELDS = {
    "network": False,
    "provider": False,
    "label": False,
    "forward_return": False,
    "p_value": False,
    "bh": False,
    "replay": False,
    "proposal": False,
    "registry": False,
    "apply": False,
    "production": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
}

_SHA_CHARS = frozenset("0123456789abcdef")


class FactorGovernanceAquantInputResolutionV4_1Error(ValueError):
    """Raised when v4.1 input-resolution evidence fails closed."""


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
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_1(dict(value)) + b"\n"


def semantic_sha256_v4_1(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_1(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} must be a non-empty canonical string"
        )
    return value


def _absolute_normalized_path_text(value: Any, label: str) -> str:
    text = _text(value, label)
    path = Path(text)
    if (
        not path.is_absolute()
        or os.path.normpath(text) != text
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} must be an absolute normalized path"
        )
    return text


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} fields mismatch"
        )
    return copy.deepcopy(dict(value))


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if field in payload:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"self-hash field already exists: {field}"
        )
    payload[field] = semantic_sha256_v4_1(payload)
    return payload


def _validate_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    stored = _sha(value.get(field), f"{label}.{field}")
    unhashed = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if semantic_sha256_v4_1(unhashed) != stored:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} self-hash mismatch"
        )


def _binding_rows(
    value: Any,
    *,
    expected_ids: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} inventory mismatch"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(
            raw,
            {"binding_id", "path", "byte_sha256", "semantic_sha256"},
            f"{label}[{index}]",
        )
        _absolute_normalized_path_text(
            row["path"], f"{label}[{index}].path"
        )
        _sha(row["byte_sha256"], f"{label}[{index}].byte_sha256")
        _sha(row["semantic_sha256"], f"{label}[{index}].semantic_sha256")
        rows.append(row)
    if [row["binding_id"] for row in rows] != list(expected_ids):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} ids must be exact and ordered"
        )
    return rows


def _inventory_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} must be a non-empty inventory"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(
            raw,
            {"relative_path", "byte_sha256", "size_bytes", "dataset_member"},
            f"{label}[{index}]",
        )
        _text(row["relative_path"], f"{label}[{index}].relative_path")
        _sha(row["byte_sha256"], f"{label}[{index}].byte_sha256")
        if type(row["size_bytes"]) is not int or row["size_bytes"] <= 0:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"{label}[{index}].size_bytes must be positive"
            )
        if type(row["dataset_member"]) is not bool:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"{label}[{index}].dataset_member must be boolean"
            )
        rows.append(row)
    paths = [row["relative_path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} paths must be sorted and unique"
        )
    return rows


def _matrix_descriptor(value: Any, label: str) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "contract",
            "shape",
            "dtype",
            "date_axis_sha256",
            "symbol_axis_sha256",
            "matrix_sha256",
        },
        label,
    )
    if row["contract"] != evaluator.MATRIX_HASH_CONTRACT_VERSION:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} matrix contract mismatch"
        )
    if (
        row["dtype"] != "float64-little-endian"
        or not isinstance(row["shape"], list)
        or len(row["shape"]) != 2
        or any(type(item) is not int or item <= 0 for item in row["shape"])
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"{label} matrix shape/dtype mismatch"
        )
    for key in ("date_axis_sha256", "symbol_axis_sha256", "matrix_sha256"):
        _sha(row[key], f"{label}.{key}")
    return row


def _coverage(value: pd.DataFrame, mask: pd.DataFrame) -> dict[str, Any]:
    masked = value.where(mask)
    array = masked.to_numpy(dtype=float)
    finite = int(np.isfinite(array).sum())
    eligible = int(mask.to_numpy(dtype=bool).sum())
    outside = int(value.where(~mask).notna().sum().sum())
    return {
        "eligible_cell_count": eligible,
        "finite_count": finite,
        "finite_ratio": float(finite / eligible) if eligible else 0.0,
        "outside_mask_non_nan_count": outside,
    }


def _validate_field_rows(
    value: Any,
    *,
    matrix_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    names = tuple(FIELD_POLICIES)
    if not isinstance(value, list) or len(value) != len(names):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "field row inventory mismatch"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(
            raw,
            {
                "field_name",
                *FIELD_POLICIES[names[index]],
                "policy_semantic_sha256",
                "matrix",
                "eligible_cell_count",
                "finite_count",
                "finite_ratio",
                "outside_mask_non_nan_count",
            },
            f"field_rows[{index}]",
        )
        field_name = row["field_name"]
        if field_name != names[index]:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "field row names must be exact and ordered"
            )
        policy = {key: row[key] for key in FIELD_POLICIES[field_name]}
        if policy != FIELD_POLICIES[field_name]:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"field policy drift: {field_name}"
            )
        if row["policy_semantic_sha256"] != semantic_sha256_v4_1(
            {"field_name": field_name, **policy}
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"field policy hash mismatch: {field_name}"
            )
        descriptor = _matrix_descriptor(row["matrix"], f"field {field_name}")
        if descriptor["shape"] != [
            matrix_context["session_count"],
            matrix_context["symbol_count"],
        ]:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"field matrix shape mismatch: {field_name}"
            )
        if (
            descriptor["date_axis_sha256"] != matrix_context["date_axis_sha256"]
            or descriptor["symbol_axis_sha256"]
            != matrix_context["symbol_axis_sha256"]
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"field axes drift: {field_name}"
            )
        eligible = row["eligible_cell_count"]
        finite = row["finite_count"]
        ratio = row["finite_ratio"]
        if (
            type(eligible) is not int
            or eligible != matrix_context["eligible_cell_count"]
            or type(finite) is not int
            or not 0 <= finite <= eligible
            or type(ratio) is not float
            or not math.isclose(ratio, finite / eligible, rel_tol=0.0, abs_tol=1e-15)
            or row["outside_mask_non_nan_count"] != 0
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"field coverage mismatch: {field_name}"
            )
        rows.append(row)
    if rows[2]["finite_count"] <= 0:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "turnover resolution must contain finite serving observations"
        )
    return rows


def _validate_candidate_rows(
    value: Any,
    *,
    matrix_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(RESOLVED_CANDIDATE_NAMES):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "candidate row inventory mismatch"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(
            raw,
            {
                "candidate_id",
                "name",
                "expression",
                "input_fields",
                "source_definition_sha256",
                "normalized_ast_sha256",
                "catalog_definition_sha256",
                "mapping_semantic_sha256",
                "signal_matrix",
                "eligible_cell_count",
                "finite_count",
                "finite_ratio",
                "outside_mask_non_nan_count",
                "status",
                "row_semantic_sha256",
            },
            f"candidate_rows[{index}]",
        )
        name = RESOLVED_CANDIDATE_NAMES[index]
        expected = EXPECTED_CANDIDATES[name]
        if (
            row["name"] != name
            or row["candidate_id"] != f"aquant:{evaluator.PINNED_COMMIT}:{name}"
            or row["input_fields"] != expected["input_fields"]
            or row["source_definition_sha256"]
            != expected["source_definition_sha256"]
            or row["normalized_ast_sha256"] != expected["normalized_ast_sha256"]
            or row["catalog_definition_sha256"]
            != expected["catalog_definition_sha256"]
            or row["mapping_semantic_sha256"]
            != expected["mapping_semantic_sha256"]
            or evaluator.normalized_ast_sha256_v4_1(row["expression"])
            != expected["normalized_ast_sha256"]
            or row["status"] != "input_resolved_no_label_only"
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"candidate identity drift: {name}"
            )
        descriptor = _matrix_descriptor(row["signal_matrix"], f"candidate {name}")
        if (
            descriptor["shape"]
            != [matrix_context["session_count"], matrix_context["symbol_count"]]
            or descriptor["date_axis_sha256"] != matrix_context["date_axis_sha256"]
            or descriptor["symbol_axis_sha256"]
            != matrix_context["symbol_axis_sha256"]
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"candidate matrix axes drift: {name}"
            )
        eligible = row["eligible_cell_count"]
        finite = row["finite_count"]
        ratio = row["finite_ratio"]
        if (
            eligible != matrix_context["eligible_cell_count"]
            or type(finite) is not int
            or not 0 < finite <= eligible
            or type(ratio) is not float
            or not math.isclose(ratio, finite / eligible, rel_tol=0.0, abs_tol=1e-15)
            or row["outside_mask_non_nan_count"] != 0
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"candidate coverage mismatch: {name}"
            )
        unhashed = {key: copy.deepcopy(item) for key, item in row.items() if key != "row_semantic_sha256"}
        if row["row_semantic_sha256"] != semantic_sha256_v4_1(unhashed):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"candidate row hash mismatch: {name}"
            )
        rows.append(row)
    return rows


def build_input_resolution_artifact_v4_1(
    *,
    cycle_id: str,
    predecessor_bindings: Sequence[Mapping[str, Any]],
    code_bindings: Sequence[Mapping[str, Any]],
    source_bindings: Sequence[Mapping[str, Any]],
    table_root: str,
    table_inventory: Sequence[Mapping[str, Any]],
    serving_root: str,
    serving_inventory: Sequence[Mapping[str, Any]],
    snapshot_id: str,
    fundamental_generation_id: str,
    eligibility_mask: pd.DataFrame,
    field_matrices: Mapping[str, pd.DataFrame],
    bound_ideas: Sequence[Mapping[str, Any]],
    protected_stability: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic no-label proof from already-bound governed inputs."""

    if cycle_id != EXPECTED_CYCLE_ID:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "cycle identity mismatch"
        )
    if (
        not isinstance(eligibility_mask, pd.DataFrame)
        or eligibility_mask.empty
        or not isinstance(eligibility_mask.index, pd.DatetimeIndex)
        or not eligibility_mask.index.is_unique
        or not eligibility_mask.index.is_monotonic_increasing
        or not eligibility_mask.columns.is_unique
        or any(dtype != bool for dtype in eligibility_mask.dtypes)
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "eligibility mask axes/values are not canonical"
        )
    if set(field_matrices) != set(FIELD_POLICIES):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "field matrix inventory mismatch"
        )
    matrices: dict[str, pd.DataFrame] = {}
    for field in FIELD_POLICIES:
        matrix = field_matrices[field]
        if (
            not isinstance(matrix, pd.DataFrame)
            or not matrix.index.equals(eligibility_mask.index)
            or not matrix.columns.equals(eligibility_mask.columns)
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"field axes differ from eligibility axes: {field}"
            )
        values = matrix.astype(float)
        if field in {"close", "vwap"}:
            values = values.where(values > 0.0)
        elif field == "turnover_rate":
            values = values.where(values >= 0.0)
        matrices[field] = values.where(eligibility_mask)

    mask_descriptor = evaluator.matrix_hash_descriptor_v4_1(
        eligibility_mask.astype(float)
    )
    eligible_count = int(eligibility_mask.to_numpy(dtype=bool).sum())
    if eligible_count <= 0:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "eligibility mask contains no eligible cells"
        )
    matrix_context = {
        "snapshot_id": _text(snapshot_id, "snapshot_id"),
        "session_count": int(len(eligibility_mask.index)),
        "symbol_count": int(len(eligibility_mask.columns)),
        "eligible_cell_count": eligible_count,
        "start_session": eligibility_mask.index[0].strftime("%Y%m%d"),
        "end_session": eligibility_mask.index[-1].strftime("%Y%m%d"),
        "date_axis_sha256": mask_descriptor["date_axis_sha256"],
        "symbol_axis_sha256": mask_descriptor["symbol_axis_sha256"],
        "eligibility_mask": mask_descriptor,
    }
    field_rows: list[dict[str, Any]] = []
    for field, policy in FIELD_POLICIES.items():
        coverage = _coverage(matrices[field], eligibility_mask)
        field_rows.append(
            {
                "field_name": field,
                **copy.deepcopy(policy),
                "policy_semantic_sha256": semantic_sha256_v4_1(
                    {"field_name": field, **policy}
                ),
                "matrix": evaluator.matrix_hash_descriptor_v4_1(matrices[field]),
                **coverage,
            }
        )

    selected = {
        row.get("name"): copy.deepcopy(dict(row))
        for row in bound_ideas
        if isinstance(row, Mapping) and row.get("name") in EXPECTED_CANDIDATES
    }
    if set(selected) != set(RESOLVED_CANDIDATE_NAMES):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "bound ideas do not contain the exact ten resolution rows"
        )
    candidate_rows: list[dict[str, Any]] = []
    for name in RESOLVED_CANDIDATE_NAMES:
        idea = selected[name]
        expected = EXPECTED_CANDIDATES[name]
        if (
            idea.get("input_fields") != expected["input_fields"]
            or idea.get("source_definition_sha256")
            != expected["source_definition_sha256"]
            or idea.get("full_candidate_normalized_ast_sha256")
            != expected["normalized_ast_sha256"]
            or idea.get("catalog_definition_sha256")
            != expected["catalog_definition_sha256"]
            or idea.get("mapping_semantic_sha256")
            != expected["mapping_semantic_sha256"]
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"bound idea identity mismatch: {name}"
            )
        signal = evaluator.evaluate_pinned_idea_v4_1(
            idea=idea,
            matrices=matrices,
            eligibility_mask=eligibility_mask,
        )
        coverage = _coverage(signal, eligibility_mask)
        row = {
            "candidate_id": idea["candidate_id"],
            "name": name,
            "expression": idea["expression"],
            "input_fields": list(idea["input_fields"]),
            "source_definition_sha256": idea["source_definition_sha256"],
            "normalized_ast_sha256": idea[
                "full_candidate_normalized_ast_sha256"
            ],
            "catalog_definition_sha256": idea["catalog_definition_sha256"],
            "mapping_semantic_sha256": idea["mapping_semantic_sha256"],
            "signal_matrix": evaluator.matrix_hash_descriptor_v4_1(signal),
            **coverage,
            "status": "input_resolved_no_label_only",
        }
        row["row_semantic_sha256"] = semantic_sha256_v4_1(row)
        candidate_rows.append(row)

    normalized_protected: list[dict[str, Any]] = []
    if not isinstance(protected_stability, Sequence) or len(protected_stability) != len(SOURCE_BINDING_IDS):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "protected stability inventory mismatch"
        )
    for index, raw in enumerate(protected_stability):
        row = _exact(
            raw,
            {"binding_id", "path", "before_sha256", "after_sha256"},
            f"protected_stability[{index}]",
        )
        if row["binding_id"] != SOURCE_BINDING_IDS[index]:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "protected stability ids must be exact and ordered"
            )
        _text(row["path"], "protected path")
        before = _sha(row["before_sha256"], "protected before SHA")
        after = _sha(row["after_sha256"], "protected after SHA")
        if before != after:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"protected input changed: {row['binding_id']}"
            )
        normalized_protected.append(row)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": cycle_id,
        "claim_scope": CLAIM_SCOPE,
        "semantics_mode": SEMANTICS_MODE,
        "exact_aquant_data_equivalence_claimed": False,
        "no_label": True,
        "input_resolution_verified": True,
        "resolution_profile": "fully_resolved",
        "resolution_accounting": {
            "predeclared_candidate_count": 10,
            "turnover_resolved_count": 2,
            "fundamental_resolved_count": 8,
            "unresolved_count": 0,
        },
        "legacy_fallback_allowed": False,
        "turnover_source": "strict_snapshot_serving_only",
        "fundamental_source": "active_fundamental_generation_only",
        "fundamental_generation_id": _text(
            fundamental_generation_id, "fundamental_generation_id"
        ),
        "predecessor_bindings": [copy.deepcopy(dict(row)) for row in predecessor_bindings],
        "code_bindings": [copy.deepcopy(dict(row)) for row in code_bindings],
        "source_bindings": [copy.deepcopy(dict(row)) for row in source_bindings],
        "table_root": _text(table_root, "table_root"),
        "table_inventory": [copy.deepcopy(dict(row)) for row in table_inventory],
        "table_inventory_semantic_sha256": semantic_sha256_v4_1(list(table_inventory)),
        "serving_root": _text(serving_root, "serving_root"),
        "serving_inventory": [copy.deepcopy(dict(row)) for row in serving_inventory],
        "serving_inventory_semantic_sha256": semantic_sha256_v4_1(list(serving_inventory)),
        "matrix_context": matrix_context,
        "field_count": len(field_rows),
        "field_rows": field_rows,
        "candidate_count": len(candidate_rows),
        "candidate_order_sha256": semantic_sha256_v4_1(
            list(RESOLVED_CANDIDATE_NAMES)
        ),
        "candidate_rows": candidate_rows,
        "protected_stability": normalized_protected,
        "authority": dict(AUTHORITY_FIELDS),
        "side_effects": dict(SIDE_EFFECT_FIELDS),
    }
    return validate_input_resolution_artifact_v4_1(
        _seal(payload, "resolution_semantic_sha256")
    )


def validate_input_resolution_artifact_v4_1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "claim_scope",
        "semantics_mode",
        "exact_aquant_data_equivalence_claimed",
        "no_label",
        "input_resolution_verified",
        "resolution_profile",
        "resolution_accounting",
        "legacy_fallback_allowed",
        "turnover_source",
        "fundamental_source",
        "fundamental_generation_id",
        "predecessor_bindings",
        "code_bindings",
        "source_bindings",
        "table_root",
        "table_inventory",
        "table_inventory_semantic_sha256",
        "serving_root",
        "serving_inventory",
        "serving_inventory_semantic_sha256",
        "matrix_context",
        "field_count",
        "field_rows",
        "candidate_count",
        "candidate_order_sha256",
        "candidate_rows",
        "protected_stability",
        "authority",
        "side_effects",
        "resolution_semantic_sha256",
    }
    payload = _exact(value, fields, "input resolution artifact")
    _validate_self_hash(payload, "resolution_semantic_sha256", "input resolution")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["cycle_id"] != EXPECTED_CYCLE_ID
        or payload["claim_scope"] != CLAIM_SCOPE
        or payload["semantics_mode"] != SEMANTICS_MODE
        or payload["exact_aquant_data_equivalence_claimed"] is not False
        or payload["no_label"] is not True
        or payload["input_resolution_verified"] is not True
        or payload["resolution_profile"] != "fully_resolved"
        or payload["resolution_accounting"]
        != {
            "predeclared_candidate_count": 10,
            "turnover_resolved_count": 2,
            "fundamental_resolved_count": 8,
            "unresolved_count": 0,
        }
        or payload["legacy_fallback_allowed"] is not False
        or payload["turnover_source"] != "strict_snapshot_serving_only"
        or payload["fundamental_source"] != "active_fundamental_generation_only"
        or payload["field_count"] != len(FIELD_POLICIES)
        or payload["candidate_count"] != len(RESOLVED_CANDIDATE_NAMES)
        or payload["candidate_order_sha256"]
        != semantic_sha256_v4_1(list(RESOLVED_CANDIDATE_NAMES))
        or payload["authority"] != AUTHORITY_FIELDS
        or payload["side_effects"] != SIDE_EFFECT_FIELDS
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "input resolution identity/authority mismatch"
        )
    _text(payload["fundamental_generation_id"], "fundamental_generation_id")
    _absolute_normalized_path_text(payload["table_root"], "table_root")
    _absolute_normalized_path_text(payload["serving_root"], "serving_root")
    predecessor = _binding_rows(
        payload["predecessor_bindings"],
        expected_ids=PREDECESSOR_BINDING_IDS,
        label="predecessor_bindings",
    )
    code = _binding_rows(
        payload["code_bindings"],
        expected_ids=CODE_BINDING_IDS,
        label="code_bindings",
    )
    sources = _binding_rows(
        payload["source_bindings"],
        expected_ids=SOURCE_BINDING_IDS,
        label="source_bindings",
    )
    table_inventory = _inventory_rows(payload["table_inventory"], "table_inventory")
    serving_inventory = _inventory_rows(
        payload["serving_inventory"], "serving_inventory"
    )
    if (
        payload["table_inventory_semantic_sha256"]
        != semantic_sha256_v4_1(table_inventory)
        or payload["serving_inventory_semantic_sha256"]
        != semantic_sha256_v4_1(serving_inventory)
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "snapshot inventory semantic hash mismatch"
        )
    source_by_id = {row["binding_id"]: row for row in sources}
    if (
        source_by_id["table_inventory"]["semantic_sha256"]
        != payload["table_inventory_semantic_sha256"]
        or source_by_id["serving_inventory"]["semantic_sha256"]
        != payload["serving_inventory_semantic_sha256"]
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "inventory binding substitution detected"
        )
    matrix_context = _exact(
        payload["matrix_context"],
        {
            "snapshot_id",
            "session_count",
            "symbol_count",
            "eligible_cell_count",
            "start_session",
            "end_session",
            "date_axis_sha256",
            "symbol_axis_sha256",
            "eligibility_mask",
        },
        "matrix_context",
    )
    mask_descriptor = _matrix_descriptor(
        matrix_context["eligibility_mask"], "eligibility mask"
    )
    if (
        type(matrix_context["session_count"]) is not int
        or type(matrix_context["symbol_count"]) is not int
        or type(matrix_context["eligible_cell_count"]) is not int
        or matrix_context["session_count"] <= 0
        or matrix_context["symbol_count"] <= 0
        or not 0 < matrix_context["eligible_cell_count"] <= (
            matrix_context["session_count"] * matrix_context["symbol_count"]
        )
        or mask_descriptor["shape"]
        != [matrix_context["session_count"], matrix_context["symbol_count"]]
        or matrix_context["date_axis_sha256"]
        != mask_descriptor["date_axis_sha256"]
        or matrix_context["symbol_axis_sha256"]
        != mask_descriptor["symbol_axis_sha256"]
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "matrix context axes/counts mismatch"
        )
    _validate_field_rows(payload["field_rows"], matrix_context=matrix_context)
    _validate_candidate_rows(
        payload["candidate_rows"], matrix_context=matrix_context
    )
    protected = payload["protected_stability"]
    if not isinstance(protected, list) or len(protected) != len(SOURCE_BINDING_IDS):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "protected stability inventory mismatch"
        )
    for index, raw in enumerate(protected):
        row = _exact(
            raw,
            {"binding_id", "path", "before_sha256", "after_sha256"},
            f"protected_stability[{index}]",
        )
        source = sources[index]
        _absolute_normalized_path_text(
            row["path"], f"protected_stability[{index}].path"
        )
        if (
            row["binding_id"] != SOURCE_BINDING_IDS[index]
            or row["path"] != source["path"]
            or _sha(row["before_sha256"], "protected before SHA")
            != source["byte_sha256"]
            or _sha(row["after_sha256"], "protected after SHA")
            != source["byte_sha256"]
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                f"protected stability/source mismatch: {SOURCE_BINDING_IDS[index]}"
            )
    payload["predecessor_bindings"] = predecessor
    payload["code_bindings"] = code
    payload["source_bindings"] = sources
    payload["table_inventory"] = table_inventory
    payload["serving_inventory"] = serving_inventory
    return payload


def build_readback_report_v4_1(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(artifacts) != {ARTIFACT_FILENAME}:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "readback input artifact inventory mismatch"
        )
    artifact = validate_input_resolution_artifact_v4_1(
        artifacts[ARTIFACT_FILENAME]
    )
    bindings = [copy.deepcopy(dict(row)) for row in artifact_bindings]
    artifact_raw = canonical_file_bytes_v4_1(artifact)
    expected_bindings = [
        {
            "filename": ARTIFACT_FILENAME,
            "byte_sha256": hashlib.sha256(artifact_raw).hexdigest(),
            "size_bytes": len(artifact_raw),
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    ]
    if canonical_json_bytes_v4_1(bindings) != canonical_json_bytes_v4_1(
        expected_bindings
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "readback artifact binding identity mismatch"
        )
    payload = {
        "schema_version": READBACK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": _text(run_id, "run_id"),
        "artifact_bindings": bindings,
        "resolution_semantic_sha256": artifact["resolution_semantic_sha256"],
        "semantics_mode": SEMANTICS_MODE,
        "exact_aquant_data_equivalence_claimed": False,
        "input_resolution_verified": True,
        "resolution_profile": "fully_resolved",
        "authority": dict(AUTHORITY_FIELDS),
        "side_effects": dict(SIDE_EFFECT_FIELDS),
    }
    return _seal(payload, "readback_semantic_sha256")


def validate_readback_report_v4_1(
    value: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "run_id",
            "artifact_bindings",
            "resolution_semantic_sha256",
            "semantics_mode",
            "exact_aquant_data_equivalence_claimed",
            "input_resolution_verified",
            "resolution_profile",
            "authority",
            "side_effects",
            "readback_semantic_sha256",
        },
        "input resolution readback",
    )
    _validate_self_hash(payload, "readback_semantic_sha256", "input resolution readback")
    normalized = validate_input_resolution_artifact_v4_1(artifact)
    if (
        payload["schema_version"] != READBACK_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["resolution_semantic_sha256"]
        != normalized["resolution_semantic_sha256"]
        or payload["semantics_mode"] != SEMANTICS_MODE
        or payload["exact_aquant_data_equivalence_claimed"] is not False
        or payload["input_resolution_verified"] is not True
        or payload["resolution_profile"] != "fully_resolved"
        or payload["authority"] != AUTHORITY_FIELDS
        or payload["side_effects"] != SIDE_EFFECT_FIELDS
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "input resolution readback mismatch"
        )
    _text(payload["run_id"], "run_id")
    if not isinstance(payload["artifact_bindings"], list) or len(payload["artifact_bindings"]) != 1:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "readback artifact binding inventory mismatch"
        )
    expected = build_readback_report_v4_1(
        run_id=payload["run_id"],
        artifacts={ARTIFACT_FILENAME: normalized},
        artifact_bindings=payload["artifact_bindings"],
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "input resolution readback differs from exact recomputation"
        )
    return payload


def build_private_bundle_contract_v4_1(
    *, expected_artifact: Mapping[str, Any]
) -> private_io.PrivateBundleContract:
    expected = validate_input_resolution_artifact_v4_1(expected_artifact)

    def validate_artifact(filename: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if filename == ARTIFACT_FILENAME:
            normalized = validate_input_resolution_artifact_v4_1(value)
            if canonical_json_bytes_v4_1(normalized) != canonical_json_bytes_v4_1(expected):
                raise FactorGovernanceAquantInputResolutionV4_1Error(
                    "resolution artifact differs from expected bytes"
                )
            return normalized
        if filename == READBACK_FILENAME:
            return copy.deepcopy(dict(value))
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            f"unexpected bundle artifact: {filename}"
        )

    def validate_complete(
        values: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        if set(values) != set(BUNDLE_FILENAMES):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "resolution bundle inventory mismatch"
            )
        artifact = validate_input_resolution_artifact_v4_1(values[ARTIFACT_FILENAME])
        report = validate_readback_report_v4_1(
            values[READBACK_FILENAME], artifact=artifact
        )
        return {ARTIFACT_FILENAME: artifact, READBACK_FILENAME: report}

    return private_io.PrivateBundleContract(
        root_suffix=PRIVATE_ROOT_SUFFIX,
        input_filenames=BUNDLE_INPUT_FILENAMES,
        readback_report_filename=READBACK_FILENAME,
        canonicalize=canonical_file_bytes_v4_1,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_readback_report_v4_1,
    )


def _stable_file_bytes(path: str | os.PathLike[str]) -> bytes:
    target = Path(path)
    if not target.is_absolute():
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "artifact path must be absolute"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "artifact descriptor open failed"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size <= 0
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "artifact must be one owner-private 0600 non-empty regular file"
            )
        raw = b""
        while len(raw) < before.st_size:
            chunk = os.read(descriptor, before.st_size - len(raw))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
        if len(raw) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "artifact changed during stable read"
            )
        return raw
    finally:
        os.close(descriptor)


def validate_input_resolution_bundle_v4_1(
    *,
    artifact_path: str | os.PathLike[str],
    expected_artifact_sha256: str,
    expected_semantic_sha256: str,
    readback_path: str | os.PathLike[str] | None = None,
    expected_readback_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently read and validate explicit byte/semantic bundle bindings."""

    artifact_raw = _stable_file_bytes(artifact_path)
    if hashlib.sha256(artifact_raw).hexdigest() != _sha(
        expected_artifact_sha256, "expected artifact SHA"
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "resolution artifact byte SHA mismatch"
        )
    try:
        value = json.loads(artifact_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "resolution artifact is not canonical JSON"
        ) from exc
    artifact = validate_input_resolution_artifact_v4_1(value)
    if artifact["resolution_semantic_sha256"] != _sha(
        expected_semantic_sha256, "expected semantic SHA"
    ):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "resolution artifact semantic SHA mismatch"
        )
    if artifact_raw != canonical_file_bytes_v4_1(artifact):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "resolution artifact bytes are not canonical"
        )
    result: dict[str, Any] = {"artifact": artifact, "artifact_path": str(artifact_path)}
    if (readback_path is None) != (expected_readback_sha256 is None):
        raise FactorGovernanceAquantInputResolutionV4_1Error(
            "readback path and expected SHA must be supplied together"
        )
    if readback_path is not None and expected_readback_sha256 is not None:
        raw = _stable_file_bytes(readback_path)
        if hashlib.sha256(raw).hexdigest() != _sha(
            expected_readback_sha256, "expected readback SHA"
        ):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "resolution readback byte SHA mismatch"
            )
        try:
            report_value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "resolution readback is not canonical JSON"
            ) from exc
        report = validate_readback_report_v4_1(report_value, artifact=artifact)
        if raw != canonical_file_bytes_v4_1(report):
            raise FactorGovernanceAquantInputResolutionV4_1Error(
                "resolution readback bytes are not canonical"
            )
        result["readback"] = report
        result["readback_path"] = str(readback_path)
    return result


__all__ = [
    "ARTIFACT_FILENAME",
    "AUTHORITY_FIELDS",
    "BUNDLE_FILENAMES",
    "CODE_BINDING_IDS",
    "EXPECTED_CANDIDATES",
    "EXPECTED_CYCLE_ID",
    "FIELD_POLICIES",
    "FactorGovernanceAquantInputResolutionV4_1Error",
    "PREDECESSOR_BINDING_IDS",
    "PRIVATE_ROOT_SUFFIX",
    "PROTOCOL_VERSION",
    "READBACK_FILENAME",
    "READBACK_SCHEMA_VERSION",
    "RESOLVED_CANDIDATE_NAMES",
    "SCHEMA_VERSION",
    "SEMANTICS_MODE",
    "SIDE_EFFECT_FIELDS",
    "SOURCE_BINDING_IDS",
    "build_input_resolution_artifact_v4_1",
    "build_private_bundle_contract_v4_1",
    "build_readback_report_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "semantic_sha256_v4_1",
    "validate_input_resolution_artifact_v4_1",
    "validate_input_resolution_bundle_v4_1",
    "validate_readback_report_v4_1",
]
