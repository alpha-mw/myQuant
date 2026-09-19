"""Select declared exposure facts from an explicit catalog after native cohort proof."""

from copy import deepcopy
from decimal import Decimal
from pathlib import PurePosixPath

from quant_investor.contracts import artifact_byte_sha256
from quant_investor.intelligence._common import (
    IntelligenceError,
    company_code,
    decimal_value,
    identifier,
    timestamp,
)
from quant_investor.intelligence.daily_evidence import EXPOSURE_SOURCE_TYPES
from quant_investor.intelligence.pcb_ai_hardware import FOCUS_COMPANIES
from quant_investor.intelligence.storage import DailyResearchPoolStore
from quant_investor.intelligence.theme_sources import descriptor_refs, split_theme_source
from .daily_contract import ContractError, validate_ref
from .dependency_diagnostics import DependencyInputError
from .research_file_readback import parse_research_json
from .research_recipes import derive_focus_context

SCHEMA = "cn-daily-exposure-catalog.v1"
ROW_FIELDS = frozenset(
    {
        "available_at",
        "company_code",
        "primary_theme_id",
        "source",
        "source_page",
        "source_type",
        "theme_revenue_share",
    }
)


def _validate_catalog_rows(value):
    """Validate declarations only; never open their physical source files."""
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "rows"}
        or value["schema_version"] != SCHEMA
        or type(value["rows"]) is not list
    ):
        raise ContractError("EXPOSURE_CATALOG_SCHEMA_INVALID")
    seen = set()
    for row in value["rows"]:
        if type(row) is not dict or set(row) != ROW_FIELDS:
            raise ContractError("EXPOSURE_CATALOG_ROW_INVALID")
        company = company_code(row["company_code"])
        if company in seen:
            raise ContractError("EXPOSURE_CATALOG_DUPLICATE_COMPANY")
        seen.add(company)
        validate_ref(row["source"])
        timestamp(row["available_at"], label="available_at")
        identifier(row["primary_theme_id"], label="primary_theme_id")
        source_type = identifier(row["source_type"], label="source_type")
        if source_type not in EXPOSURE_SOURCE_TYPES:
            raise ContractError("EXPOSURE_CATALOG_SOURCE_TYPE_INVALID")
        page = row["source_page"]
        if page is not None and (type(page) is not int or page <= 0):
            raise ContractError("EXPOSURE_CATALOG_SOURCE_PAGE_INVALID")
        decimal_value(
            row["theme_revenue_share"],
            label="theme_revenue_share",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
    return value["rows"]


def validate_catalog(value):
    """Use the existing schema diagnosis for invalid catalog declarations."""
    try:
        return _validate_catalog_rows(value)
    except (ContractError, IntelligenceError) as exc:
        raise DependencyInputError("RESEARCH_COMPANY_SOURCE_SCHEMA_INVALID") from exc


def select_exposure_rows(value, *, recipe, workspace, pool_ref, theme_source, files):
    """Keep legacy lists exact; derive catalog rows only from verified native inputs.

    Caller retains/rechecks the original catalog ref. All cohort/focus reads are
    enrolled in the supplied native file reader; only final native projection may
    open selected declaration sources. No unselected source is evidence.
    """
    if value is None or type(value) is list:
        return value
    from .execution_recipe import SCHEMA_V5, SCHEMA_V6
    from .research_timing import CURRENT

    rows = validate_catalog(value)
    if recipe["schema_version"] not in {SCHEMA_V5, SCHEMA_V6}:
        raise DependencyInputError("RESEARCH_SOURCE_SCHEMA_INVALID")
    if recipe.get("research_timing", {}).get("mode") != CURRENT:
        raise DependencyInputError("CUTOFF_TIMING_POLICY_MODE_MISMATCH")
    _, _, enabled = split_theme_source(theme_source)
    if not enabled:
        raise DependencyInputError("CUTOFF_FOCUS_SOURCE_DECLARATION_REQUIRED")

    def read(ref):
        _, raw, _ = files.source_file(ref, code="CUTOFF_SOURCE_REF_INVALID")
        return parse_research_json(raw, label="exposure catalog cohort")

    pool = read(pool_ref)
    rank_ref = {
        "path": str(PurePosixPath(pool_ref["path"]).parent / "factor_research_rank.json"),
        "sha256": pool["payload"]["rank_byte_sha256"],
    }
    rank = read(rank_ref)
    native = DailyResearchPoolStore(workspace)
    verified = native.verify(
        rank=rank,
        expected_policy_sha256=pool["payload"]["policy_byte_sha256"],
        policy_path=pool["payload"]["policy_path"],
    )
    if verified["manifest.json"] != pool_ref or verified["factor_research_rank.json"] != rank_ref:
        raise DependencyInputError("RESEARCH_SOURCE_POOL_BINDING_INVALID")
    for ref in verified.values():
        files.source_file(ref, code="CUTOFF_SOURCE_REF_INVALID")
    files.source_file(
        {"path": pool["payload"]["policy_path"], "sha256": pool["payload"]["policy_byte_sha256"]},
        code="CUTOFF_SOURCE_REF_INVALID",
    )
    day = rank["payload"]["signal_date"]
    for observation in native._observations(rank, None):
        alias = observation["payload"]["factor_alias"]
        files.source_file(
            {
                "path": f"results/factors/observations/{day[:4]}/{day[4:6]}/{day[6:]}/{alias}.json",
                "sha256": artifact_byte_sha256(observation),
            },
            code="CUTOFF_SOURCE_REF_INVALID",
        )
    for ref in descriptor_refs(theme_source):
        files.source_file(ref, code="CUTOFF_SOURCE_REF_INVALID")
    focus = derive_focus_context(
        workspace=str(workspace),
        request={"theme_source": theme_source},
        rank=rank,
        pool_store=native,
    )
    if focus is None:
        raise DependencyInputError("CUTOFF_FOCUS_SOURCE_DECLARATION_REQUIRED")
    for ref in focus["source_refs"]:
        files.source_file(ref, code="CUTOFF_FOCUS_PIT_REF_INVALID")
    companies = {row["symbol"] for row in rank["payload"]["pool_rows"]} | set(FOCUS_COMPANIES)
    return deepcopy(
        sorted(
            (row for row in rows if row["company_code"] in companies),
            key=lambda row: row["company_code"],
        )
    )
