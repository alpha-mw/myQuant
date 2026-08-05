from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import quant_investor.v17_v4_runtime.research_factor_set as factor_set_module
from quant_investor.v17_v4_contract import (
    artifact_identity_field,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_runtime.research_factor_set import (
    CANDIDATE_CATALOG_SHA256,
    CATALOG_RESOURCE_SHA256,
    FACTOR_SET_POINTER,
    GATE_ORDER,
    IMPLEMENTATION_RESOURCE_SHA256,
    ResearchFactorSetCrash,
    ResearchFactorSetError,
    ResearchFactorSetStore,
    assert_research_factor_set_reread,
    build_research_factor_input_bundle,
    build_research_shadow_factor_set,
    research_factor_catalog_bindings,
    validate_research_factor_input_bundle,
    validate_research_shadow_factor_set,
)
from quant_investor.v17_v4_runtime.source_storage import EMPTY_SHA256

STRATEGY = "cn-shadow-research"
CUTOFF = "2026-07-28T10:00:00Z"
PUBLISHED_AT = "2026-07-28T10:01:00Z"
AUDIT_SESSION = "2026-07-28"
OPEN_SESSIONS = ["2026-07-28", "2026-07-29", "2026-07-30"]


def test_research_factor_set_artifact_identity_is_factor_set_id() -> None:
    assert (
        artifact_identity_field("myquant.v17.v4.research-shadow-factor-set.v1") == "factor_set_id"
    )


def _ref(
    artifact_id: str,
    version: str,
    path: str,
    *,
    cutoff: str = CUTOFF,
    byte_sha256: str | None = None,
    semantic_sha256: str | None = None,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": byte_sha256 or hashlib.sha256(path.encode()).hexdigest(),
        "cutoff": cutoff,
        "relative_path": path,
        "semantic_sha256": semantic_sha256
        or hashlib.sha256((path + ":semantic").encode()).hexdigest(),
        "strategy_id": STRATEGY,
    }


def _selection(name: str, **gate_overrides: bool) -> dict[str, Any]:
    bindings = research_factor_catalog_bindings()
    gates = {gate_id: True for gate_id in GATE_ORDER}
    gates.update(gate_overrides)
    factor = bindings["factors"][name]
    return {
        "candidate_catalog_sha256": CANDIDATE_CATALOG_SHA256,
        "catalog_resource_sha256": CATALOG_RESOURCE_SHA256,
        "definition_sha256": factor["definition_sha256"],
        "implementation_resource_sha256": IMPLEMENTATION_RESOURCE_SHA256,
        "implementation_sha256": factor["implementation_sha256"],
        "name": name,
        "selection_gates": gates,
    }


def _factor_set(
    factor_set_id: str,
    names: list[str],
    *,
    previous: dict[str, str] | None = None,
    disabled: set[str] = frozenset(),
    cutoff: str = CUTOFF,
    published_at: str = PUBLISHED_AT,
) -> dict[str, Any]:
    rows = [
        _selection(
            name,
            source_review_accepted=name not in disabled,
        )
        for name in names
    ]
    return build_research_shadow_factor_set(
        factor_set_id=factor_set_id,
        strategy_id=STRATEGY,
        cutoff=cutoff,
        audit_session=AUDIT_SESSION,
        selected_at=cutoff,
        published_at=published_at,
        open_sessions=OPEN_SESSIONS,
        monthly_audit_ref=_ref(
            "monthly-audit-202607",
            "myquant.factor-governance.monthly-audit.v4",
            "data/private/v17_v4_sources/audits/monthly-audit-202607.json",
            cutoff=cutoff,
        ),
        previous_factor_set_ref=previous,
        selection_rows=rows,
        expected_candidate_catalog_sha256=CANDIDATE_CATALOG_SHA256,
        expected_catalog_resource_sha256=CATALOG_RESOURCE_SHA256,
        expected_implementation_resource_sha256=(IMPLEMENTATION_RESOURCE_SHA256),
    )


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return seal_semantic(payload)


def test_selection_is_bounded_deterministic_and_effective_next_open() -> None:
    names = sorted(research_factor_catalog_bindings()["factors"])[:9]
    factor_set = _factor_set("research-set-1", list(reversed(names)))

    assert factor_set["eligible_factor_count"] == 9
    assert factor_set["eligible_distinct_slot_count"] == 9
    assert factor_set["target_cardinality"] == 8
    assert factor_set["effective_from_session"] == "2026-07-29"
    assert factor_set["published_at"] == PUBLISHED_AT
    assert [row["name"] for row in factor_set["selected_factors"]] == names[:8]
    assert len({row["slot"] for row in factor_set["selected_factors"]}) == 8
    assert all(row["selection_score"] == 90 for row in factor_set["selected_factors"])
    assert factor_set["authority"] == {
        "broker": False,
        "execution": False,
        "mainline_authority": False,
        "order": False,
        "production": False,
        "research_only": True,
        "trade": False,
    }
    assert factor_set["shadow_only"] is True
    assert factor_set["performance_evidence_eligible"] is False


def test_preclose_publication_can_start_same_shanghai_session() -> None:
    name = sorted(research_factor_catalog_bindings()["factors"])[0]
    factor_set = _factor_set(
        "research-set-same-session",
        [name],
        cutoff="2026-07-28T02:00:00Z",
        published_at="2026-07-28T02:01:00Z",
    )

    assert factor_set["effective_from_session"] == "2026-07-28"

    forged = copy.deepcopy(factor_set)
    forged.pop("published_at")
    with pytest.raises(ResearchFactorSetError, match="factor_set_timing"):
        validate_research_shadow_factor_set(_reseal(forged))


def test_same_session_publish_requires_current_process_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = sorted(research_factor_catalog_bindings()["factors"])[0]
    factor_set = _factor_set(
        "research-set-same-session",
        [name],
        cutoff="2026-07-28T02:00:00Z",
        published_at="2026-07-28T02:01:00Z",
    )
    monkeypatch.setattr(
        factor_set_module,
        "_publication_now_utc",
        lambda: datetime(2026, 7, 28, 2, 7, tzinfo=timezone.utc),
    )

    with pytest.raises(ResearchFactorSetError, match="same_session_publication_clock"):
        ResearchFactorSetStore(str(tmp_path.resolve())).publish(
            factor_set,
            expected_pointer_sha256=EMPTY_SHA256,
        )
    monkeypatch.setattr(
        factor_set_module,
        "_publication_now_utc",
        lambda: datetime(2026, 7, 28, 2, 5, tzinfo=timezone.utc),
    )
    publication = ResearchFactorSetStore(str(tmp_path.resolve())).publish(
        factor_set,
        expected_pointer_sha256=EMPTY_SHA256,
    )
    assert publication.factor_set_ref["artifact_id"] == "research-set-same-session"


def test_rotation_publishes_immutable_set_before_pointer_and_exact_reread(
    tmp_path: Path,
) -> None:
    names = sorted(research_factor_catalog_bindings()["factors"])[:9]
    store = ResearchFactorSetStore(str(tmp_path.resolve()))
    first_set = _factor_set("research-set-1", names)
    first = store.publish(
        first_set,
        expected_pointer_sha256=EMPTY_SHA256,
    )

    second_set = _factor_set(
        "research-set-2",
        names,
        previous=dict(first.factor_set_ref),
        disabled={names[0]},
    )
    second = store.publish(
        second_set,
        expected_pointer_sha256=first.pointer_ref["byte_sha256"],
    )
    state = assert_research_factor_set_reread(
        str(tmp_path.resolve()),
        expected_pointer_byte_sha256=second.pointer_ref["byte_sha256"],
        expected_factor_set_ref=second.factor_set_ref,
        expected_factor_set_byte_sha256=second.factor_set_ref["byte_sha256"],
    )

    assert state.factor_set["factor_set_id"] == "research-set-2"
    assert state.factor_set["previous_factor_set_ref"] == first.factor_set_ref
    assert names[0] not in {row["name"] for row in state.factor_set["selected_factors"]}
    first_path = tmp_path / first.factor_set_ref["relative_path"]
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == (
        first.factor_set_ref["byte_sha256"]
    )


def test_tamper_and_unsupported_fields_fail_closed() -> None:
    name = sorted(research_factor_catalog_bindings()["factors"])[0]
    factor_set = _factor_set("research-set-1", [name])

    tampered = copy.deepcopy(factor_set)
    tampered["selected_factors"][0]["slot"] = "primitive:forged"
    with pytest.raises(ResearchFactorSetError, match="definition"):
        validate_research_shadow_factor_set(_reseal(tampered))

    unsupported = copy.deepcopy(factor_set)
    unsupported["production_ready"] = False
    with pytest.raises(ResearchFactorSetError, match="factor_set_fields"):
        validate_research_shadow_factor_set(_reseal(unsupported))

    row = _selection(name)
    row["unsupported"] = False
    with pytest.raises(ResearchFactorSetError, match="selection_row_fields"):
        build_research_shadow_factor_set(
            factor_set_id="research-set-2",
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            audit_session=AUDIT_SESSION,
            selected_at=CUTOFF,
            published_at=PUBLISHED_AT,
            open_sessions=OPEN_SESSIONS,
            monthly_audit_ref=factor_set["monthly_audit_ref"],
            previous_factor_set_ref=None,
            selection_rows=[row],
            expected_candidate_catalog_sha256=CANDIDATE_CATALOG_SHA256,
            expected_catalog_resource_sha256=CATALOG_RESOURCE_SHA256,
            expected_implementation_resource_sha256=(IMPLEMENTATION_RESOURCE_SHA256),
        )


def test_pointer_cas_crash_recovery_and_true_third_state_abort(
    tmp_path: Path,
) -> None:
    names = sorted(research_factor_catalog_bindings()["factors"])[:2]
    store = ResearchFactorSetStore(str(tmp_path.resolve()))
    factor_set = _factor_set("research-set-1", names)

    with pytest.raises(ResearchFactorSetCrash, match="pointer CAS"):
        store.publish(
            factor_set,
            expected_pointer_sha256=EMPTY_SHA256,
            crash_after="cas",
        )
    recovered = store.publish(
        factor_set,
        expected_pointer_sha256=EMPTY_SHA256,
    )
    assert recovered.recovered is True
    before = (tmp_path / str(FACTOR_SET_POINTER)).read_bytes()

    competing = _factor_set(
        "research-set-2",
        names,
        previous=dict(recovered.factor_set_ref),
    )
    with pytest.raises(ResearchFactorSetError, match="pointer_third_state"):
        store.publish(
            competing,
            expected_pointer_sha256=EMPTY_SHA256,
        )
    assert (tmp_path / str(FACTOR_SET_POINTER)).read_bytes() == before


def test_reader_detects_immutable_set_tamper(tmp_path: Path) -> None:
    name = sorted(research_factor_catalog_bindings()["factors"])[0]
    store = ResearchFactorSetStore(str(tmp_path.resolve()))
    publication = store.publish(
        _factor_set("research-set-1", [name]),
        expected_pointer_sha256=EMPTY_SHA256,
    )
    path = tmp_path / publication.factor_set_ref["relative_path"]
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"shadow_only":true', b'"shadow_only":false'))

    with pytest.raises(ResearchFactorSetError, match="factor_set_exact_read"):
        store.read_current()


def test_input_bundle_binds_exact_run_local_fields_and_rejects_unknown() -> None:
    factor_set = _factor_set(
        "research-set-inputs",
        ["cn_earnings_yield_ex_shell_30pct"],
    )
    factor_raw = canonical_resource_bytes(factor_set)
    factor_ref = _ref(
        "research-set-inputs",
        factor_set["version"],
        ("data/private/v17_v4_sources/research_factor_sets/" "sets/research-set-inputs.json"),
        byte_sha256=hashlib.sha256(factor_raw).hexdigest(),
        semantic_sha256=factor_set["semantic_sha256"],
    )
    slices = [
        {
            "available_at": "2026-07-28T09:59:00Z",
            "field_name": field_name,
            "first_session": "2025-07-01",
            "last_session": "2026-07-29",
            "row_count": 100,
            "slice_ref": _ref(
                f"run-1-{field_name}",
                "myquant.v17.v4.research-factor-input-slice.v1",
                ("data/private/v17_v4_runs/run-1/" f"research_factor_inputs/{field_name}.parquet"),
            ),
        }
        for field_name in ["pe", "total_mv"]
    ]
    bundle = build_research_factor_input_bundle(
        bundle_id="input-bundle-1",
        run_id="run-1",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        decision_session="2026-07-29",
        factor_set=factor_set,
        factor_set_ref=factor_ref,
        research_source_locator_ref=_ref(
            "source-locator-1",
            "myquant.v17.v4.research-source-locator.v2",
            "data/private/v17_v4_sources/locators/source-locator-1.json",
        ),
        field_slices=slices,
    )
    assert bundle["required_fields"] == ["pe", "total_mv"]
    assert [row["field_name"] for row in bundle["field_slices"]] == [
        "pe",
        "total_mv",
    ]

    unsupported = copy.deepcopy(bundle)
    unsupported["field_slices"][0]["inferred"] = False
    with pytest.raises(ResearchFactorSetError, match="field_slice_fields"):
        validate_research_factor_input_bundle(
            _reseal(unsupported),
            factor_set=factor_set,
        )


def test_new_schemas_are_additive_strict_resources() -> None:
    schema_root = Path(__file__).parents[2] / "quant_investor/v17_v4_contract/schemas"
    expected = {
        "research_shadow_factor_set.v1.schema.json": (
            "myquant.v17.v4.research-shadow-factor-set.v1"
        ),
        "research_shadow_factor_set_pointer.v1.schema.json": (
            "myquant.v17.v4.research-shadow-factor-set-pointer.v1"
        ),
        "research_factor_input_bundle.v1.schema.json": (
            "myquant.v17.v4.research-factor-input-bundle.v1"
        ),
    }
    for filename, version in expected.items():
        schema = json.loads((schema_root / filename).read_text())
        assert schema["additionalProperties"] is False
        assert schema["properties"]["version"]["const"] == version
