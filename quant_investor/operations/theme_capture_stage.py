"""Native DC/TDX capture stage under the caller's existing day lock."""

from datetime import datetime, timezone
from pathlib import Path

from quant_investor.market.tushare import (
    build_theme_provider_execution_plan,
    derive_tdx_fallback_company_keyset,
)
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.market.tushare import theme_runtime as native
from quant_investor.cli import unified
from quant_investor.intelligence.storage import approved_theme_policy_v2
from .daily_contract import ContractError, utc_stamp
from .daily_journal import DailyJournal
from .theme_core_binding import bind_theme_acquisition
from .theme_acquisition import reserve_theme_acquisition
from quant_investor.intelligence.theme_sources import THEME_SOURCE_V2
from .research_timing import theme_acquisition_bound


def capture_bound_theme(
    *, journal: DailyJournal, request_ref: dict, core_handoff_ref: dict
) -> dict:
    """Explicit EXECUTE producer only; passive replay must not call this function."""
    journal._require_lock()
    workspace = Path(journal.storage._io.workspace_root)
    binding = bind_theme_acquisition(
        workspace=str(workspace), request_ref=request_ref, core_handoff_ref=core_handoff_ref
    )
    return _capture_validated_theme(
        journal=journal,
        request_ref=request_ref,
        core_handoff_ref=core_handoff_ref,
        binding=binding,
    )


def _capture_validated_theme(*, journal, request_ref, core_handoff_ref, binding):
    """Only called with the local result of the owning native core binder."""
    journal._require_lock()
    workspace = Path(journal.storage._io.workspace_root)
    if binding["trade_date"] != journal.trade_date:
        raise ContractError("THEME_CAPTURE_DAY_MISMATCH")
    source_bound = theme_acquisition_bound(binding)
    cutoff = utc_stamp(source_bound)
    if datetime.now(timezone.utc) < utc_stamp(binding["core_completed_at"]):
        raise ContractError("THEME_ACQUISITION_CORE_FUTURE")

    def now():
        value = datetime.now(timezone.utc)
        if value > cutoff:
            raise ContractError("THEME_ACQUISITION_CUTOFF_EXPIRED")
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Do not consume the day claim for an already-expired new acquisition.
    if journal.storage.read(str(journal.root / "theme-acquisition.v1.json")) is None:
        now()
    claim = reserve_theme_acquisition(journal=journal, identity=binding["identity"])
    claim_document = parse_canonical_json_bytes(journal.storage.read(claim["path"]).data)
    prefix = journal.root / "executions" / request_ref["sha256"] / "theme-source"

    def ref(path):
        relative = str(path.relative_to(workspace))
        stored = journal.storage.read(relative)
        if stored is None:
            raise ContractError("THEME_CAPTURE_FILE_MISSING")
        return {"path": relative, "sha256": stored.byte_sha256}

    def provider(name, companies, label):
        path = str(prefix / (label + "-plan.json"))
        saved = journal.storage.read(path)
        if saved is None:
            stamp = now()
            plan = build_theme_provider_execution_plan(
                provider=name,
                trade_date=journal.trade_date,
                company_keyset=companies,
                document_observed_at=stamp,
                created_at=stamp,
            )
            saved = journal.storage.write(path, native.canonical_bytes(plan))
        plan = native.load_exact_plan(workspace / path, saved.byte_sha256)
        if (
            plan["provider"] != name
            or plan["trade_date"] != journal.trade_date
            or plan["company_keyset"] != companies
        ):
            raise ContractError("THEME_CAPTURE_PLAN_SCOPE_CHANGED")
        if utc_stamp(plan["timestamp"]) > min(cutoff, datetime.now(timezone.utc)):
            raise ContractError("THEME_CAPTURE_PLAN_FUTURE")
        if utc_stamp(plan["timestamp"]) < utc_stamp(claim_document["claimed_at"]):
            raise ContractError("THEME_CAPTURE_PLAN_PRECEDES_CLAIM")
        root = workspace / prefix / label
        if not (root / "capture.json").exists():
            now()
            native.capture_theme_plan(
                plan_path=workspace / path,
                plan_sha256=saved.byte_sha256,
                output_root=root,
                allow_live=True,
                resume=root.exists(),
                now=now,
            )
        native._validate_resume_root(root, plan)
        capture_ref = ref(root / "capture.json")
        partition_refs = [
            ref(root / "partitions" / f"{i:05d}.json") for i in range(1 + len(companies))
        ]
        loaded, capture, partitions = native.load_capture_root(root)
        if loaded != plan:
            raise ContractError("THEME_CAPTURE_PLAN_CHANGED")
        return (
            plan,
            capture,
            partitions,
            {
                "plan": {"path": path, "sha256": saved.byte_sha256},
                "capture": capture_ref,
                "partitions": partition_refs,
            },
        )

    dc_plan, dc_capture, dc_parts, dc_refs = provider("TUSHARE_DC", binding["company_keyset"], "dc")
    fallback = derive_tdx_fallback_company_keyset(
        dc_plan=dc_plan, dc_capture=dc_capture, dc_partition_documents=dc_parts
    )
    tdx_refs = {"plan": None, "capture": None, "partitions": []}
    if fallback:
        _, _, _, tdx_refs = provider("TUSHARE_TDX", fallback, "tdx")
    descriptor = {
        "dc_plan": dc_refs["plan"],
        "dc_capture": dc_refs["capture"],
        "dc_partitions": dc_refs["partitions"],
        "tdx_plan": tdx_refs["plan"],
        "tdx_capture": tdx_refs["capture"],
        "tdx_partitions": tdx_refs["partitions"],
    }
    from functools import partial

    projection = unified._daily_theme_projection(
        {
            "as_of": source_bound,
            "policy": approved_theme_policy_v2(),
            "theme_source": descriptor,
        },
        binding["company_keyset"],
        partial(unified._daily_source_document, str(workspace)),
    )
    if projection is None or projection["payload"]["blocker_codes"]:
        raise ContractError("THEME_ACQUISITION_NATIVE_SOURCE_INCOMPLETE")
    focus_descriptor = None
    focus_projection = None
    focus_fallback = []
    focus_refs = []
    if "special_company_keyset" in binding:
        focus_companies = binding["special_company_keyset"]
        fp, fc, parts, fr = provider("TUSHARE_DC", focus_companies, "special-dc")
        focus_fallback = derive_tdx_fallback_company_keyset(
            dc_plan=fp, dc_capture=fc, dc_partition_documents=parts
        )
        ft = {"plan": None, "capture": None, "partitions": []}
        if focus_fallback:
            _, _, _, ft = provider("TUSHARE_TDX", focus_fallback, "special-tdx")
        focus_descriptor = {
            "dc_plan": fr["plan"],
            "dc_capture": fr["capture"],
            "dc_partitions": fr["partitions"],
            "tdx_plan": ft["plan"],
            "tdx_capture": ft["capture"],
            "tdx_partitions": ft["partitions"],
        }
        focus_projection = unified._daily_theme_projection(
            {
                "as_of": source_bound,
                "policy": approved_theme_policy_v2(),
                "theme_source": focus_descriptor,
            },
            focus_companies,
            partial(unified._daily_source_document, str(workspace)),
        )
        # Valid retained but incomplete focus captures must reach the partial report.
        if focus_projection is None:
            raise ContractError("THEME_FOCUS_NATIVE_PROJECTION_MISSING")
        focus_refs = [fr["plan"], fr["capture"], *fr["partitions"]]
        if focus_fallback:
            focus_refs.extend([ft["plan"], ft["capture"], *ft["partitions"]])
    if (
        bind_theme_acquisition(
            workspace=str(workspace), request_ref=request_ref, core_handoff_ref=core_handoff_ref
        )
        != binding
    ):
        raise ContractError("THEME_CAPTURE_CORE_CHANGED")
    captured_refs = [
        claim,
        dc_refs["plan"],
        dc_refs["capture"],
        *dc_refs["partitions"],
        *focus_refs,
    ]
    if fallback:
        captured_refs.extend([tdx_refs["plan"], tdx_refs["capture"], *tdx_refs["partitions"]])
    for captured_ref in captured_refs:
        current = journal.storage.read(captured_ref["path"])
        if current is None or current.byte_sha256 != captured_ref["sha256"]:
            raise ContractError("THEME_CAPTURE_SOURCE_CHANGED")
    return {
        "binding": binding,
        "claim_ref": claim,
        "descriptor": (
            descriptor
            if focus_descriptor is None
            else {
                "schema_version": THEME_SOURCE_V2,
                "pool": descriptor,
                "pcb_ai_hardware": focus_descriptor,
            }
        ),
        "fallback_company_keyset": fallback,
        "projection": projection,
        **(
            {
                "focus_projection": focus_projection,
                "special_fallback_company_keyset": focus_fallback,
            }
            if focus_descriptor is not None
            else {}
        ),
    }
