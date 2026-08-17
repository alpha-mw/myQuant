from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.system import (
    ACTIVE_POINTER_PATH,
    MIGRATION_MARKER_PATH,
    SystemActivationAuthorizationError,
)
from test_unified_system_bootstrap import _closure
from unified_activation_helpers import prepare_initial_activation


def test_generic_operational_generation_cannot_cross_initial_activation_gate(
    tmp_path: Path,
) -> None:
    closure = _closure(tmp_path)
    store = closure["store"]
    generation = store.assemble_generation(**closure["kwargs"])
    prepared = prepare_initial_activation(store, generation, closure["release_ref"])

    with pytest.raises(
        SystemActivationAuthorizationError,
        match="lacks valid production target closure",
    ):
        store.activate_initial_generation(**prepared)

    assert not (store.workspace_root / str(ACTIVE_POINTER_PATH)).exists()
    assert not (store.workspace_root / str(MIGRATION_MARKER_PATH)).exists()
