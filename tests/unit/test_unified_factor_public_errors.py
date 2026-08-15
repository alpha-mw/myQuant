from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from quant_investor.factors.governance import (
    FactorGovernanceError,
    validate_bootstrap_exception_evidence,
    validate_bootstrap_factor_set,
    validate_factor_status,
    validate_factor_validation_receipt,
    validate_observation_head,
    validate_preregistration,
)
from quant_investor.factors.governance.common import artifact_ref


@pytest.mark.parametrize(
    "validator",
    [
        validate_bootstrap_exception_evidence,
        validate_bootstrap_factor_set,
        validate_factor_status,
        validate_factor_validation_receipt,
        validate_observation_head,
        validate_preregistration,
        artifact_ref,
    ],
)
def test_malformed_envelopes_use_the_factor_validation_boundary(
    validator: Callable[[bytes], Any],
) -> None:
    with pytest.raises(FactorGovernanceError) as captured:
        validator(b"{}")
    assert captured.value.code == "FACTOR_VALIDATION_FAILED"
    assert captured.value.exit_code == 2
    assert captured.value.public_fields == {}
    assert str(captured.value) == ("FACTOR_VALIDATION_FAILED:artifact envelope is invalid")
