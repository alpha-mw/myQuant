"""Provider-free native Theme source replay from one exact handoff."""

from functools import partial
import hashlib
from pathlib import Path
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.market.tushare import theme_runtime as native
from quant_investor.market.tushare import derive_tdx_fallback_company_keyset
from quant_investor.cli import unified
from quant_investor.intelligence.storage import approved_theme_policy_v2
from .daily_contract import ContractError, validate_ref, utc_stamp
from .theme_core_binding import bind_theme_acquisition
from .theme_acquisition import validate_theme_acquisition_claim
from .theme_handoff import validate_theme_handoff, SCHEMA_V2, SCHEMA_V3, handoff_layout
from .research_timing import theme_acquisition_bound
from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2, split_theme_source


def read_theme_handoff(
    *, journal, request_ref: dict, core_handoff_ref: dict, handoff_ref: dict
) -> dict:
    workspace = Path(journal.storage._io.workspace_root)
    binding = bind_theme_acquisition(
        workspace=str(workspace), request_ref=request_ref, core_handoff_ref=core_handoff_ref
    )
    prefix = journal.root / "executions" / request_ref["sha256"] / "theme-source"
    observed = {}

    def read(ref, expected_path=None):
        checked = validate_ref(ref)
        if expected_path is not None and checked["path"] != str(expected_path):
            raise ContractError("THEME_HANDOFF_SOURCE_PATH_INVALID")
        stored = journal.storage.read(checked["path"])
        if stored is None or stored.byte_sha256 != checked["sha256"]:
            raise ContractError("THEME_HANDOFF_SOURCE_SHA_MISMATCH")
        observed[checked["path"]] = stored.data
        return stored.data

    schema, fields, filename = handoff_layout(binding)
    value = parse_canonical_json_bytes(read(handoff_ref, prefix / filename))
    if type(value) is not dict or set(value) != fields or value["schema_version"] != schema:
        raise ContractError("THEME_HANDOFF_CONTRACT_INVALID")
    claim = parse_canonical_json_bytes(
        read(value["claim_ref"], journal.root / "theme-acquisition.v1.json")
    )
    validate_theme_acquisition_claim(
        claim, trade_date=journal.trade_date, identity=binding["identity"]
    )

    def provider(label, companies, provider_name):
        path_label = label.replace("_", "-")
        plan_ref, cap_ref, parts = (
            value[label + "_plan_ref"],
            value[label + "_capture_ref"],
            value[label + "_partition_refs"],
        )
        read(plan_ref, prefix / (path_label + "-plan.json"))
        read(cap_ref, prefix / path_label / "capture.json")
        if type(parts) is not list or len(parts) != len(companies) + 1:
            raise ContractError("THEME_HANDOFF_PARTITION_SET_INVALID")
        for index, ref in enumerate(parts):
            read(ref, prefix / path_label / "partitions" / f"{index:05d}.json")
        plan = native.load_exact_plan(workspace / plan_ref["path"], plan_ref["sha256"])
        if (
            plan["company_keyset"] != companies
            or plan["provider"] != provider_name
            or plan["trade_date"] != journal.trade_date
        ):
            raise ContractError("THEME_HANDOFF_NATIVE_PLAN_SCOPE_INVALID")
        capture_root = workspace / prefix / path_label
        copy = {
            "path": str(prefix / path_label / "plan.json"),
            "sha256": hashlib.sha256(native.canonical_bytes(plan)).hexdigest(),
        }
        read(copy)
        native._validate_resume_root(capture_root, plan)
        loaded, capture, partitions = native.load_capture_root(capture_root)
        if loaded != plan:
            raise ContractError("THEME_HANDOFF_NATIVE_PLAN_CHANGED")
        if utc_stamp(plan["timestamp"]) < utc_stamp(claim["claimed_at"]):
            raise ContractError("THEME_HANDOFF_PLAN_PRECEDES_CLAIM")
        return plan, capture, partitions

    dc_plan, dc_capture, dc_parts = provider("dc", binding["company_keyset"], "TUSHARE_DC")
    fallback = derive_tdx_fallback_company_keyset(
        dc_plan=dc_plan, dc_capture=dc_capture, dc_partition_documents=dc_parts
    )
    focus_fallback = None
    focus_times = []
    if schema in {SCHEMA_V2, SCHEMA_V3}:
        fp, fc, parts = provider("special_dc", binding["special_company_keyset"], "TUSHARE_DC")
        focus_fallback = derive_tdx_fallback_company_keyset(
            dc_plan=fp, dc_capture=fc, dc_partition_documents=parts
        )
        focus_times.append(fc["timestamp"])
        if focus_fallback:
            _, fc, _ = provider("special_tdx", focus_fallback, "TUSHARE_TDX")
            focus_times.append(fc["timestamp"])
    validate_theme_handoff(
        value,
        journal=journal,
        binding=binding,
        claim_ref=value["claim_ref"],
        fallback_company_keyset=fallback,
        special_fallback_company_keyset=focus_fallback,
    )
    times = [
        claim["claimed_at"],
        binding["core_completed_at"],
        dc_capture["timestamp"],
        *focus_times,
    ]
    if utc_stamp(claim["claimed_at"]) < utc_stamp(binding["core_completed_at"]):
        raise ContractError("THEME_HANDOFF_CLAIM_PRECEDES_CORE")
    if fallback:
        _, capture, _ = provider("tdx", fallback, "TUSHARE_TDX")
        times.append(capture["timestamp"])
    if utc_stamp(value["sealed_at"]) < max(map(utc_stamp, times)):
        raise ContractError("THEME_HANDOFF_PREMATURE_SEAL")
    descriptor = parse_canonical_json_bytes(
        read(value["source_descriptor_ref"], prefix / "source.json")
    )
    expected = {
        name + suffix: value[name + source]
        for name in ("dc", "tdx")
        for suffix, source in (
            ("_plan", "_plan_ref"),
            ("_capture", "_capture_ref"),
            ("_partitions", "_partition_refs"),
        )
    }
    if schema in {SCHEMA_V2, SCHEMA_V3}:
        expected = {
            "schema_version": THEME_SOURCE_V2,
            "pool": expected,
            "pcb_ai_hardware": {
                name + suffix: value["special_" + name + source]
                for name in ("dc", "tdx")
                for suffix, source in (
                    ("_plan", "_plan_ref"),
                    ("_capture", "_capture_ref"),
                    ("_partitions", "_partition_refs"),
                )
            },
        }
    if descriptor != expected:
        raise ContractError("THEME_HANDOFF_DESCRIPTOR_MISMATCH")
    projection = unified._daily_theme_projection(
        {
            "as_of": theme_acquisition_bound(binding),
            "policy": approved_theme_policy_v2(),
            "theme_source": descriptor,
        },
        binding["company_keyset"],
        partial(unified._daily_source_document, str(workspace)),
    )
    if projection is None or projection["payload"]["blocker_codes"]:
        raise ContractError("THEME_HANDOFF_NATIVE_SOURCE_INCOMPLETE")
    if schema in {SCHEMA_V2, SCHEMA_V3}:
        _, focus_source, _ = split_theme_source(descriptor)
        focus_projection = unified._daily_theme_projection(
            {
                "as_of": theme_acquisition_bound(binding),
                "policy": approved_theme_policy_v2(),
                "theme_source": focus_source,
            },
            binding["special_company_keyset"],
            partial(unified._daily_source_document, str(workspace)),
        )
        if focus_projection is None:
            raise ContractError("THEME_HANDOFF_FOCUS_PROJECTION_MISSING")
    if (
        bind_theme_acquisition(
            workspace=str(workspace), request_ref=request_ref, core_handoff_ref=core_handoff_ref
        )
        != binding
    ):
        raise ContractError("THEME_HANDOFF_CORE_CHANGED")
    for path, raw in observed.items():
        current = journal.storage.read(path)
        if current is None or current.data != raw:
            raise ContractError("THEME_HANDOFF_CHANGED_DURING_REPLAY")
    return {"handoff": value, "descriptor": descriptor, "binding": binding}
