"""Bulk Tushare Fundamental v4 evidence and reconciliation contracts."""

from .acquisition import acquire_fundamental_vip_v4
from .contracts import (
    build_logical_symbol_table_coverage_v4,
    build_provider_physical_request_receipt_v4,
    build_raw_table_evidence_v4,
    validate_logical_symbol_table_coverage_v4,
    validate_provider_physical_request_receipt_v4,
    validate_raw_table_evidence_v4,
)
from .comparison import (
    build_fundamental_comparison_policy,
    compare_fundamental_raw_tables,
    validate_fundamental_comparison_policy,
)
from .models import FundamentalV4ContractError
from .official_partition_acquisition import (
    OFFICIAL_PARTITION_REQUEST_RECEIPT_V1,
    acquire_official_partition_fundamental_vip_v4,
    validate_official_partition_request_receipt,
)
from .official_partition_plan import (
    OFFICIAL_PARTITION_PLAN_V1,
    build_official_partition_execution_plan,
    validate_official_partition_execution_plan,
)
from .manifest import (
    build_fundamental_provider_manifest_v4,
    validate_fundamental_provider_manifest_v4,
)
from .fileset import (
    REQUIRED_EVIDENCE_PATHS,
    build_provider_evidence_fileset_manifest,
    validate_provider_evidence_fileset_manifest,
)
from .reconciliation import (
    build_fundamental_reconciliation_receipt,
    validate_fundamental_reconciliation_receipt,
)
from .promotion import (
    append_promotion_journal_event,
    build_promotion_event,
    classify_promotion_recovery,
    create_promotion_journal,
    read_promotion_journal,
    validate_promotion_event,
    validate_promotion_event_chain,
)
from .schedule import (
    build_fundamental_execution_closure_v4,
    build_fundamental_request_plan_v4,
    validate_fundamental_execution_closure_v4,
    validate_fundamental_request_plan_v4,
)
from .storage import capture_provider_evidence_directory
from .shadow import (
    build_fundamental_shadow_bundle_v4,
    build_logical_coverages_from_shadow_v4,
    derive_fundamental_shadow_v4,
    materialize_fundamental_v4_staging_generation,
    write_fundamental_shadow_bundle_v4,
)

__all__ = [
    "FundamentalV4ContractError",
    "OFFICIAL_PARTITION_PLAN_V1",
    "OFFICIAL_PARTITION_REQUEST_RECEIPT_V1",
    "REQUIRED_EVIDENCE_PATHS",
    "acquire_official_partition_fundamental_vip_v4",
    "acquire_fundamental_vip_v4",
    "build_fundamental_comparison_policy",
    "build_fundamental_execution_closure_v4",
    "build_fundamental_provider_manifest_v4",
    "build_fundamental_request_plan_v4",
    "build_fundamental_shadow_bundle_v4",
    "build_logical_coverages_from_shadow_v4",
    "build_official_partition_execution_plan",
    "derive_fundamental_shadow_v4",
    "materialize_fundamental_v4_staging_generation",
    "build_fundamental_reconciliation_receipt",
    "build_provider_evidence_fileset_manifest",
    "build_promotion_event",
    "build_logical_symbol_table_coverage_v4",
    "build_provider_physical_request_receipt_v4",
    "build_raw_table_evidence_v4",
    "capture_provider_evidence_directory",
    "classify_promotion_recovery",
    "create_promotion_journal",
    "append_promotion_journal_event",
    "read_promotion_journal",
    "compare_fundamental_raw_tables",
    "validate_fundamental_comparison_policy",
    "validate_fundamental_execution_closure_v4",
    "validate_fundamental_provider_manifest_v4",
    "validate_fundamental_request_plan_v4",
    "validate_fundamental_reconciliation_receipt",
    "validate_provider_evidence_fileset_manifest",
    "validate_promotion_event",
    "validate_promotion_event_chain",
    "validate_logical_symbol_table_coverage_v4",
    "validate_official_partition_request_receipt",
    "validate_official_partition_execution_plan",
    "validate_provider_physical_request_receipt_v4",
    "validate_raw_table_evidence_v4",
    "write_fundamental_shadow_bundle_v4",
]
