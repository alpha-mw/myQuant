"""Read exact native research input refs for descriptive source-time ledger rows.

Native business/PIT projections must still replay before ledger admission. Source
publication is never inferred from created_at, file metadata or target dates.
"""

from pathlib import Path
from quant_investor.cli.unified import _daily_source_file
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref
from quant_investor.intelligence.fundamental_time import availability_instant
from quant_investor.intelligence.theme_sources import split_theme_source


def read_research_source_times(*, workspace: str, request_ref: dict) -> dict:
    root = Path(workspace).resolve(strict=True)
    request_ref = validate_ref(request_ref)
    storage = SecureSystemStorage(workspace)
    raw = storage.read_workspace_file_bytes(request_ref["path"], maximum_bytes=8 * 1024 * 1024)
    if raw.byte_sha256 != request_ref["sha256"]:
        raise ContractError("PROSPECTIVE_RESEARCH_REQUEST_SHA_MISMATCH")
    request = parse_canonical_json_bytes(raw.data)
    if request.get("schema_version") in {
        "cn-daily-research-request.v2",
        "cn-daily-research-request.v3",
    }:
        from .research_request import load_research_request

        request = load_research_request(workspace=workspace, reference=request_ref)["document"]
    evidence = request.get("company_evidence") or {}
    result = {name: {} for name in ("industry", "theme", "exposure", "fundamental", "macro")}
    observed = {}

    def add(node, ref, declared=None):
        checked = validate_ref(ref)
        if declared is not None:
            availability_instant(declared)
        _daily_source_file(root, checked, code="PROSPECTIVE_SOURCE_SHA_INVALID")
        identity = (checked["path"], checked["sha256"])
        observed[identity] = checked
        prior = result[node].get(identity)
        # Multiple companies may cite one file. Retain the latest declaration,
        # never choose an earlier timestamp to make availability look timely.
        if prior is None or (
            declared is not None
            and (
                prior["declared_available_at"] is None
                or availability_instant(declared)
                > availability_instant(prior["declared_available_at"])
            )
        ):
            result[node][identity] = {"source_ref": checked, "declared_available_at": declared}

    for node, singles, lists in (
        (
            "industry",
            ("membership_capture", "membership_plan", "taxonomy_capture", "taxonomy_plan"),
            ("membership_partitions",),
        ),
        (
            "theme",
            ("dc_capture", "dc_plan", "tdx_capture", "tdx_plan"),
            ("dc_partitions", "tdx_partitions"),
        ),
    ):
        source = request[node + "_source"]
        sources = [source]
        if node == "theme":
            pool, focus, _ = split_theme_source(source)
            sources = [pool, focus]
        for selected in sources:
            if selected is None:
                continue
            if type(selected) is not dict or set(selected) != set(singles + lists):
                raise ContractError("PROSPECTIVE_SOURCE_SHAPE_INVALID")
            for name in singles:
                if selected[name] is not None:
                    add(node, selected[name])
            for name in lists:
                if type(selected[name]) is not list:
                    raise ContractError("PROSPECTIVE_SOURCE_SHAPE_INVALID")
                for ref in selected[name]:
                    add(node, ref)
    rows = evidence.get("exposure_rows")
    if rows is not None:
        if type(rows) is not list:
            raise ContractError("PROSPECTIVE_SOURCE_SHAPE_INVALID")
        for row in rows:
            add("exposure", row["source"], row["available_at"])
    fundamental = evidence.get("fundamental_source")
    if fundamental is not None:
        if type(fundamental) is not dict or set(fundamental) != {
            "pointer",
            "daily_parquet",
            "available_at",
        }:
            raise ContractError("PROSPECTIVE_SOURCE_SHAPE_INVALID")
        add("fundamental", fundamental["pointer"])
        add("fundamental", fundamental["daily_parquet"], fundamental["available_at"])
    macro = evidence.get("macro_risk")
    if macro is not None:
        if type(macro) is not dict or set(macro) != {"classification", "source"}:
            raise ContractError("PROSPECTIVE_SOURCE_SHAPE_INVALID")
        add("macro", macro["source"])
    for ref in observed.values():
        _daily_source_file(root, ref, code="PROSPECTIVE_SOURCE_CHANGED_DURING_READ")
    if storage.read_workspace_file_bytes(request_ref["path"], maximum_bytes=8 * 1024 * 1024) != raw:
        raise ContractError("PROSPECTIVE_RESEARCH_REQUEST_CHANGED")
    return {node: [rows[key] for key in sorted(rows)] for node, rows in result.items()}


def native_cutoff_source_times(
    *, request, source_document, source_file, workspace, macro_admission
):
    """Descriptive rows after native projection, using only owning source-time fields."""
    from quant_investor.strategy_records.corporate_contracts import instant

    rows = []

    def add(role, subject, ref, stamp, semantics):
        instant(stamp)
        rows.append(
            {
                "role": role,
                "subject_id": subject,
                "source_ref": validate_ref(ref),
                "original_time": stamp,
                "time_semantics": semantics,
            }
        )

    industry = request["industry_source"]
    for part, role in (("taxonomy", "INDUSTRY_TAXONOMY"), ("membership", "INDUSTRY_MEMBERSHIP")):
        ref = industry[part + "_capture"]
        capture = source_document(ref)
        add(role, "ALL", ref, capture["timestamp"], "PROVIDER_CAPTURE")
    pool, focus, _ = split_theme_source(request["theme_source"])
    for scope, descriptor in (("POOL", pool), ("FOCUS", focus)):
        if descriptor is None:
            continue
        for provider in ("dc", "tdx"):
            ref = descriptor[provider + "_capture"]
            if ref is not None:
                capture = source_document(ref)
                add(
                    "THEME_" + scope + "_" + provider.upper(),
                    "ALL",
                    ref,
                    capture["timestamp"],
                    "PROVIDER_CAPTURE",
                )
    evidence = request["company_evidence"]
    for row in evidence["exposure_rows"] or []:
        add(
            "EXPOSURE_DECLARATION",
            row["company_code"],
            row["source"],
            row["available_at"],
            "SOURCE_DECLARED",
        )
    fundamental = evidence["fundamental_source"]
    pointer = source_document(fundamental["pointer"])
    if pointer.get("schema_version") != "cn-fundamental-pointer.v1":
        raise ContractError("CUTOFF_NATIVE_FUNDAMENTAL_REQUIRED")
    from quant_investor.market.fundamental_generation import inspect_fundamental_pointer_bytes

    _, pointer_bytes, pointer_ref = source_file(
        fundamental["pointer"], code="CUTOFF_FUNDAMENTAL_TIME_SOURCE_INVALID"
    )
    verified = inspect_fundamental_pointer_bytes(
        Path(workspace) / "data/parquet/cn",
        pointer_bytes=pointer_bytes,
        expected_pointer_sha256=pointer_ref["sha256"],
    )["pointer"]
    path = Path(verified["manifest_path"])
    if not path.is_absolute():
        path = Path(workspace) / "data/parquet/cn" / path
    manifest_ref = {
        "path": path.relative_to(Path(workspace)).as_posix(),
        "sha256": verified["derivation_binding"]["manifest_sha256"],
    }
    manifest = source_document(manifest_ref)
    if manifest != verified["manifest"]:
        raise ContractError("CUTOFF_FUNDAMENTAL_MANIFEST_CHANGED")
    generation = pointer["generation_id"]
    add(
        "FUNDAMENTAL_DECLARATION",
        generation,
        fundamental["daily_parquet"],
        fundamental["available_at"],
        "SOURCE_DECLARED",
    )
    native_times = []
    for ref, derivation in (
        (manifest_ref, manifest["metadata"].get("provider_manifest", {}).get("derivation", {})),
        (fundamental["pointer"], pointer["metadata"].get("derivation", {})),
    ):
        if "derivation_timestamp" in derivation:
            native_times.append(derivation["derivation_timestamp"])
            add(
                "FUNDAMENTAL_NATIVE_DERIVATION",
                generation,
                ref,
                derivation["derivation_timestamp"],
                "LOCAL_CLOSURE",
            )
    if not native_times:
        raise ContractError("CUTOFF_NATIVE_FUNDAMENTAL_TIME_MISSING")
    for ref in (fundamental["pointer"], fundamental["daily_parquet"], manifest_ref):
        source_file(ref, code="CUTOFF_FUNDAMENTAL_TIME_SOURCE_INVALID")
    add(
        "MACRO_CLOSURE",
        macro_admission["target_date"],
        evidence["macro_risk"]["source"],
        macro_admission["available_at"],
        "LOCAL_CLOSURE",
    )
    return rows
