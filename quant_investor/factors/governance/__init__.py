"""Stable, versionless Factor governance public API.

All builders below are pure, hash-bound candidate or observation operations.
Only :mod:`quant_investor.system` can make a Factor set official by activating a
verified generation.
"""

from .admission import (
    ADMITTED_SET_KIND,
    EVALUATION_KIND,
    PROSPECTIVE_ADMISSION_ROUTE,
    validate_admitted_factor_set,
    validate_preregistration_evaluation,
)
from .bootstrap import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    BOOTSTRAP_LANE,
    BOOTSTRAP_SET_KIND,
    CANONICAL_PARQUET,
    LOW_DOLLAR_VOLUME,
    NOT_CLAIMED,
    PROSPECTIVE_LANE,
    bootstrap_factor_definitions,
    build_bootstrap_factor_set,
    compute_bootstrap_signals,
    validate_bootstrap_factor_set,
)
from .bootstrap_evidence import (
    BOOTSTRAP_ADMISSION_ROUTE,
    BOOTSTRAP_DECISION_SOURCE_ID,
    BOOTSTRAP_EVIDENCE_KIND,
    build_bootstrap_exception_evidence,
    validate_bootstrap_exception_evidence,
)
from .errors import FactorGovernanceError
from .custody import (
    COMPOSITE_STATE_KIND,
    CUSTODY_RECORD_KIND,
    validate_composite_state,
    validate_custody_record,
)
from .contextual import CONTEXTUAL_RESULT_KIND, validate_contextual_result
from .execution import (
    EXECUTION_EVIDENCE_KIND,
    validate_execution_turnover_evidence,
)
from .lineage import (
    OBSERVATION_HEAD_KIND,
    append_observation_cas,
    read_observation_head,
    validate_observation_head,
)
from .manifest import VALIDATOR_MANIFEST_KIND, validate_validator_manifest
from .prospective import (
    OBSERVATION_KIND,
    PREREGISTRATION_KIND,
    SELECTION_KIND,
    SIGNAL_CAPTURE_KIND,
    validate_configuration_selection,
    validate_observation,
    validate_preregistration,
    validate_signal_capture,
)
from .receipt import (
    VALIDATION_RECEIPT_KIND,
    validate_factor_validation_receipt,
)
from .source import SOURCE_DECODE_ATTESTATION_KIND, validate_source_decode_attestation
from .status import (
    FACTOR_BLOCKED,
    FACTOR_READY,
    FACTOR_STATUS_KIND,
    validate_factor_status,
)
from .store import (
    BootstrapClosure,
    FactorValidationStore,
    bootstrap_validation_namespace_id,
    prospective_validation_namespace_id,
)

__all__ = [
    "ADMITTED_SET_KIND",
    "BLEND_W75_CONTROL",
    "BLEND_W80",
    "BOOTSTRAP_ADMISSION_ROUTE",
    "BOOTSTRAP_DECISION_SOURCE_ID",
    "BOOTSTRAP_EVIDENCE_KIND",
    "BOOTSTRAP_LANE",
    "BOOTSTRAP_SET_KIND",
    "BootstrapClosure",
    "CANONICAL_PARQUET",
    "COMPOSITE_STATE_KIND",
    "CONTEXTUAL_RESULT_KIND",
    "CUSTODY_RECORD_KIND",
    "EVALUATION_KIND",
    "EXECUTION_EVIDENCE_KIND",
    "FACTOR_BLOCKED",
    "FACTOR_READY",
    "FACTOR_STATUS_KIND",
    "FactorGovernanceError",
    "FactorValidationStore",
    "LOW_DOLLAR_VOLUME",
    "NOT_CLAIMED",
    "OBSERVATION_KIND",
    "OBSERVATION_HEAD_KIND",
    "PREREGISTRATION_KIND",
    "PROSPECTIVE_ADMISSION_ROUTE",
    "PROSPECTIVE_LANE",
    "SELECTION_KIND",
    "SIGNAL_CAPTURE_KIND",
    "SOURCE_DECODE_ATTESTATION_KIND",
    "VALIDATION_RECEIPT_KIND",
    "VALIDATOR_MANIFEST_KIND",
    "bootstrap_factor_definitions",
    "bootstrap_validation_namespace_id",
    "append_observation_cas",
    "build_bootstrap_exception_evidence",
    "build_bootstrap_factor_set",
    "compute_bootstrap_signals",
    "read_observation_head",
    "prospective_validation_namespace_id",
    "validate_admitted_factor_set",
    "validate_bootstrap_exception_evidence",
    "validate_bootstrap_factor_set",
    "validate_configuration_selection",
    "validate_contextual_result",
    "validate_composite_state",
    "validate_custody_record",
    "validate_factor_status",
    "validate_factor_validation_receipt",
    "validate_validator_manifest",
    "validate_execution_turnover_evidence",
    "validate_observation",
    "validate_observation_head",
    "validate_preregistration",
    "validate_preregistration_evaluation",
    "validate_signal_capture",
    "validate_source_decode_attestation",
]
