"""Recipe versions select one immutable path without fallback or extra authority."""

from pathlib import PurePosixPath
from types import SimpleNamespace
import hashlib

import pytest
from quant_investor.contracts import canonical_json_bytes

from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.execution_recipe import SCHEMA, SCHEMA_V2
from quant_investor.operations.materialization_contract import (
    FIELDS_V1,
    FIELDS_V2,
    SCHEMA_V1 as MATERIALIZATION_V1,
    SCHEMA_V2 as MATERIALIZATION_V2,
    layout_for_recipe,
    read_selected_materialization,
    validate_bound_materialization,
    validate_materialization_shape,
    validate_materialization_version,
)

EXECUTION = PurePosixPath("days/20260908/executions/" + "a" * 64)


def document(version):
    fields = FIELDS_V1 if version == MATERIALIZATION_V1 else FIELDS_V2
    return {**dict.fromkeys(fields), "schema_version": version}


@pytest.mark.parametrize("policy_version", [1, 2])
@pytest.mark.parametrize("handoff_version", [1, 2])
def test_native_materialization_uses_exact_policy_version(
    tmp_path, policy_version, handoff_version
):
    from quant_investor.factors.production_pit import FOCUS_COMPANIES

    policy = {
        "schema_version": f"cn-daily-theme-acquisition.v{policy_version}",
        "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
        "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
        "maximum_companies": 100,
        **({"special_company_keyset": list(FOCUS_COMPANIES)} if policy_version == 2 else {}),
    }
    raw = canonical_json_bytes(policy)
    source = tmp_path / "policy.json"
    source.write_bytes(raw)
    source.chmod(0o600)
    recipe = {
        "schema_version": SCHEMA_V2,
        "theme_acquisition_ref": {"path": "policy.json", "sha256": hashlib.sha256(raw).hexdigest()},
    }
    value = document(MATERIALIZATION_V2)
    value["theme_source_handoff_ref"] = {
        "path": str(EXECUTION / f"theme-source/handoff.v{handoff_version}.json"),
        "sha256": "b" * 64,
    }
    kwargs = dict(
        workspace=str(tmp_path),
        value=value,
        recipe=recipe,
        execution=EXECUTION,
        path=str(EXECUTION / "materialization.v2.json"),
    )
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    if policy_version == handoff_version:
        validate_bound_materialization(**kwargs)
    else:
        with pytest.raises(ContractError, match="HANDOFF_PATH"):
            validate_bound_materialization(**kwargs)
    assert before == (source.read_bytes(), source.stat().st_mtime_ns)
    recipe["theme_acquisition_ref"]["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="POLICY_SHA"):
        validate_bound_materialization(**kwargs)


@pytest.mark.parametrize(
    "recipe_version,version", [(SCHEMA, MATERIALIZATION_V1), (SCHEMA_V2, MATERIALIZATION_V2)]
)
def test_selected_path_never_falls_back_to_other_version(recipe_version, version):
    recipe = {"schema_version": recipe_version, "theme_acquisition_ref": None}
    _, filename, _ = layout_for_recipe(recipe)
    selected = str(EXECUTION / filename)
    other = str(
        EXECUTION
        / ("materialization.v2.json" if recipe_version == SCHEMA else "materialization.v1.json")
    )
    files = {selected: b"original"}
    reads = []

    def read(path):
        reads.append(path)
        return files.get(path)

    journal = SimpleNamespace(storage=SimpleNamespace(read=read))
    assert (
        read_selected_materialization(journal=journal, execution=EXECUTION, recipe=recipe)
        == b"original"
    )
    assert reads == [
        str(EXECUTION / name)
        for name in (
            "materialization.v1.json",
            "materialization.v2.json",
            "materialization.v3.json",
            "materialization.v4.json",
            "materialization.v5.json",
            "materialization.v6.json",
        )
        if str(EXECUTION / name) != selected
    ] + [selected]
    value = document(version)
    validate_materialization_version(value, recipe=recipe, execution=EXECUTION, path=selected)
    for contents in (b"conflict", b""):
        files[other] = contents
        files.pop(selected, None)
        with pytest.raises(ContractError, match="VERSION_CONFLICT"):
            read_selected_materialization(journal=journal, execution=EXECUTION, recipe=recipe)
    files.clear()
    assert (
        read_selected_materialization(journal=journal, execution=EXECUTION, recipe=recipe) is None
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_version",
        "unknown_version",
        "version_mismatch",
        "wrong_path",
        "extra_field",
        "missing_field",
        "missing_mode",
        "unexpected_handoff",
        "missing_handoff",
        "wrong_handoff",
        "bad_sha",
    ],
)
def test_malformed_version_or_theme_binding_rejected(fault):
    recipe = {"schema_version": SCHEMA_V2, "theme_acquisition_ref": None}
    value = document(MATERIALIZATION_V2)
    path = str(EXECUTION / "materialization.v2.json")
    if fault == "missing_version":
        recipe.pop("schema_version")
    elif fault == "unknown_version":
        recipe["schema_version"] = "cn-daily-execution-recipe.v3"
    elif fault == "version_mismatch":
        value = document(MATERIALIZATION_V1)
    elif fault == "wrong_path":
        path = str(EXECUTION / "materialization.v1.json")
    elif fault == "extra_field":
        value["extra"] = True
    elif fault == "missing_field":
        value.pop("theme_source_handoff_ref")
    elif fault == "missing_mode":
        recipe.pop("theme_acquisition_ref")
    elif fault == "unexpected_handoff":
        value["theme_source_handoff_ref"] = {
            "path": str(EXECUTION / "theme-source/handoff.v1.json"),
            "sha256": "b" * 64,
        }
    else:
        recipe["theme_acquisition_ref"] = {"path": "policy.json", "sha256": "c" * 64}
        if fault != "missing_handoff":
            value["theme_source_handoff_ref"] = {
                "path": (
                    "other.json"
                    if fault == "wrong_handoff"
                    else str(EXECUTION / "theme-source/handoff.v1.json")
                ),
                "sha256": "bad" if fault == "bad_sha" else "b" * 64,
            }
    with pytest.raises(ContractError):
        validate_materialization_version(value, recipe=recipe, execution=EXECUTION, path=path)


def test_acquired_theme_requires_exact_request_namespace():
    value = document(MATERIALIZATION_V2)
    value["theme_source_handoff_ref"] = {
        "path": str(EXECUTION / "theme-source/handoff.v1.json"),
        "sha256": "b" * 64,
    }
    validate_materialization_version(
        value,
        recipe={
            "schema_version": SCHEMA_V2,
            "theme_acquisition_ref": {"path": "policy.json", "sha256": "c" * 64},
        },
        execution=EXECUTION,
        path=str(EXECUTION / "materialization.v2.json"),
    )


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": "unknown"}])
def test_unknown_shapes_rejected(value):
    with pytest.raises(ContractError, match="CONTRACT_INVALID"):
        validate_materialization_shape(value)
