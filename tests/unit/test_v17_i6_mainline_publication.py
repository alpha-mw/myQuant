from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

import pytest

from quant_investor.intelligence_v2._core import canonical_bytes as v2_canonical_bytes
from quant_investor.intelligence_v2._core import common_fields, content_ref, seal
from quant_investor.intelligence_v2.portfolio import (
    build_market_risk_projection,
    build_portfolio_risk_policy,
)
from quant_investor.intelligence_v2.publication import (
    PUBLICATION_CLOSURE_VERSION,
    PUBLICATION_CLOSURE_VERSIONS,
    build_action_permit,
    build_activation_sidecar,
    build_legacy_marker,
    build_legacy_marker_profile,
    build_preactivation_receipt,
    build_publication_closure,
    build_publication_owner_policy,
    derive_publication_paths,
    permit_message,
    validate_legacy_marker_profile,
    validate_publication_owner_policy,
)
from quant_investor.v17_mainline import MainlineStore, MainlineStorageSecurityError
from quant_investor.v17_mainline import MainlineBlocker, derive_mainline_state
from quant_investor.v17_mainline.contracts import canonical_bytes, parse_canonical
from quant_investor.v17_mainline import intelligence_v2_reader
from quant_investor.v17_mainline.runtime import read_public_run
from quant_investor.v17_mainline.testing import (
    write_synthetic_fixture_for_tests,
    write_synthetic_governed_bytes_for_tests,
)

STRATEGY = "cn-mainline"
LEGACY_DTO_SHA256 = "63c9ece4913e75bed8a559c32bb54fbc4a9df89c2503458e35490dc4def3d235"
V2_PATH = "results/v17_intelligence_v2/fixtures/publication-closure.json"
AT = "2026-08-09T01:00:00Z"
PERMIT_END = "2026-08-09T01:05:00Z"

_IDENTITY_FIELDS = {
    "DECISION_V2": "decision_id",
    "EVIDENCE_GRAPH_V2": "graph_id",
    "GRADUATION": "graduation_id",
    "GRADUATION_POLICY": "policy_id",
    "I5_ADVISORY_RANK": "advisory_rank_id",
    "I5_PRIVATE_CAPABILITY": "private_capability_id",
    "LEGACY_MARKER_PROFILE": "profile_id",
    "MARKET_RISK_PROJECTION": "projection_id",
    "PAPER_CAPITAL_GATE": "receipt_id",
    "PAPER_EXECUTION_POLICY": "policy_id",
    "PAPER_LEDGER": "ledger_id",
    "PORTFOLIO": "receipt_id",
    "PORTFOLIO_POLICY": "policy_id",
    "PREACTIVATION": "preactivation_id",
    "PUBLICATION_OWNER_POLICY": "policy_id",
}


def _generic_v2(key: str, **payload: object) -> dict:
    return seal(
        {
            **common_fields(timestamp_value=AT),
            **payload,
            "version": PUBLICATION_CLOSURE_VERSIONS[key],
        },
        identity_field=_IDENTITY_FIELDS[key],
    )


def _write_v2_document(
    tmp_path: Path,
    document: dict,
    *,
    identity_field: str,
    relative_path: str,
) -> dict[str, str]:
    raw = v2_canonical_bytes(document)
    stored = write_synthetic_governed_bytes_for_tests(
        tmp_path,
        relative_path=relative_path,
        raw=raw,
        synthetic_only=True,
    )
    return {
        "artifact_id": document[identity_field],
        "artifact_version": document["version"],
        "available_at": AT,
        "byte_sha256": stored.byte_sha256,
        "cutoff": document["timestamp"],
        "relative_path": relative_path,
        "semantic_sha256": document["semantic_sha256"],
    }


def test_legacy_no_marker_public_dto_bytes_remain_frozen(tmp_path: Path) -> None:
    write_synthetic_fixture_for_tests(
        tmp_path,
        strategy_id=STRATEGY,
        synthetic_only=True,
    )
    raw = canonical_bytes(read_public_run(tmp_path, strategy_id=STRATEGY))
    assert hashlib.sha256(raw).hexdigest() == LEGACY_DTO_SHA256


@pytest.mark.parametrize(
    "profile",
    ["INTELLIGENCE_V2_PUBLICATION_V1", "UNKNOWN_PUBLICATION_PROFILE"],
)
def test_marked_run_without_exact_v2_closure_never_becomes_active(
    tmp_path: Path,
    profile: str,
) -> None:
    write_synthetic_fixture_for_tests(
        tmp_path,
        strategy_id=STRATEGY,
        formal_overrides={"publication_profile": profile},
        synthetic_only=True,
    )

    result = derive_mainline_state(tmp_path, strategy_id=STRATEGY)

    assert result.blocker is MainlineBlocker.INTELLIGENCE_V2_PUBLICATION_INVALID
    assert result.public_run is None


def _owner_policy_and_signer():  # type: ignore[no-untyped-def]
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_id = hashlib.sha256(public_key).hexdigest()
    policy = build_publication_owner_policy(
        created_at=AT,
        maximum_permit_lifetime_seconds=600,
        keys=[
            {
                "actions": ["ACTIVATE"],
                "algorithm": "ED25519",
                "key_id": key_id,
                "not_after": "2026-08-10T01:00:00Z",
                "not_before": AT,
                "public_key_base64": base64.b64encode(public_key).decode("ascii"),
                "revoked_at": None,
            }
        ],
    )
    return policy, private_key, key_id


def _research_nodes(tmp_path: Path) -> tuple[dict[str, dict], dict, object, str]:
    owner_policy, private_key, key_id = _owner_policy_and_signer()
    profile = build_legacy_marker_profile(created_at=AT, canonical_strategy_id=STRATEGY)
    graph_one = _generic_v2("EVIDENCE_GRAPH_V2", company_code="000001.SZ")
    graph_two = _generic_v2("EVIDENCE_GRAPH_V2", company_code="600000.SH")
    graph_one_ref = content_ref(graph_one, identity_field="graph_id")
    graph_two_ref = content_ref(graph_two, identity_field="graph_id")
    decision_one = _generic_v2(
        "DECISION_V2",
        company_code="000001.SZ",
        graph_ref=graph_one_ref,
        state="PAPER_CANDIDATE",
    )
    decision_two = _generic_v2(
        "DECISION_V2",
        company_code="600000.SH",
        graph_ref=graph_two_ref,
        state="PAPER_CANDIDATE",
    )
    decision_refs = [
        content_ref(decision_one, identity_field="decision_id"),
        content_ref(decision_two, identity_field="decision_id"),
    ]
    portfolio = _generic_v2(
        "PORTFOLIO",
        admitted_decision_refs=decision_refs,
        final_portfolio={
            "blocker_codes": [],
            "cash_weight": "0.600000000000",
            "gross_weight": "0.400000000000",
            "ordering": ["000001.SZ", "600000.SH"],
            "status": "COMPLETE",
            "targets": [
                {
                    "company_code": "000001.SZ",
                    "final_weight": "0.150000000000",
                },
                {
                    "company_code": "600000.SH",
                    "final_weight": "0.250000000000",
                },
            ],
        },
    )
    graduation_policy = _generic_v2("GRADUATION_POLICY")
    paper_policy = _generic_v2("PAPER_EXECUTION_POLICY")
    portfolio_policy = build_portfolio_risk_policy(
        created_at=AT,
        target_positions=2,
        target_gross="0.40",
        cash_floor="0.60",
        per_security_cap="0.25",
        industry_cap="0.40",
        theme_cap="0.40",
        max_adv_participation="0.10",
        turnover_cap="1",
        weight_quantum="0.01",
        drawdown_threshold="0.30",
        risk_threshold="0.80",
        hard_veto_codes=[],
        macro_regime_rules=[
            {
                "cash_floor": "0.60",
                "gross_cap": "0.40",
                "regime": "NORMAL",
                "risk_multiplier": "1",
                "veto_codes": [],
            }
        ],
        fundamental_staleness_allowance_sessions=1,
        paper_execution_policy_ref=content_ref(paper_policy, identity_field="policy_id"),
        graduation_policy_ref=content_ref(graduation_policy, identity_field="policy_id"),
    )
    market_inputs = []
    for kind in (
        "CANONICAL_DAILY",
        "CANONICAL_DAILY_BASIC",
        "CANONICAL_LIMIT",
        "CANONICAL_SUSPEND",
    ):
        digest = hashlib.sha256(kind.encode()).hexdigest()
        market_inputs.append(
            {
                "freshness": "FRESH",
                "kind": kind,
                "source_ref": {
                    "artifact_id": kind.lower(),
                    "artifact_version": "myquant.test.market-source.v1",
                    "available_at": AT,
                    "byte_sha256": digest,
                    "cutoff": AT,
                    "relative_path": f"results/v17_intelligence_v2/fixtures/{kind.lower()}.json",
                    "semantic_sha256": digest,
                },
                "status": "COMPLETE",
                "target_session": "20260809",
            }
        )
    market_projection = build_market_risk_projection(
        portfolio_policy=portfolio_policy,
        target_session="20260809",
        input_rows=market_inputs,
        projected_gross_cap="0.40",
        projected_cash_floor="0.60",
        projected_security_cap="0.25",
        projected_veto_codes=[],
        as_of=AT,
    )
    documents = {
        "DECISION_V2": decision_one,
        "EVIDENCE_GRAPH_V2": graph_one,
        "GRADUATION": _generic_v2("GRADUATION", status="GRADUATED"),
        "GRADUATION_POLICY": graduation_policy,
        "I5_ADVISORY_RANK": _generic_v2("I5_ADVISORY_RANK"),
        "I5_PRIVATE_CAPABILITY": _generic_v2("I5_PRIVATE_CAPABILITY"),
        "LEGACY_MARKER_PROFILE": profile,
        "MARKET_RISK_PROJECTION": market_projection,
        "PAPER_CAPITAL_GATE": _generic_v2("PAPER_CAPITAL_GATE"),
        "PAPER_EXECUTION_POLICY": paper_policy,
        "PAPER_LEDGER": _generic_v2("PAPER_LEDGER"),
        "PORTFOLIO": portfolio,
        "PORTFOLIO_POLICY": portfolio_policy,
        "PUBLICATION_OWNER_POLICY": owner_policy,
    }
    refs: dict[str, dict] = {}
    for key, document in documents.items():
        path = f"results/v17_intelligence_v2/fixtures/{key.lower()}.json"
        refs[key] = _write_v2_document(
            tmp_path,
            document,
            identity_field=_IDENTITY_FIELDS[key],
            relative_path=path,
        )
    extra_refs = []
    for label, document, identity_field in (
        ("decision-two", decision_two, "decision_id"),
        ("graph-two", graph_two, "graph_id"),
    ):
        extra_refs.append(
            _write_v2_document(
                tmp_path,
                document,
                identity_field=identity_field,
                relative_path=f"results/v17_intelligence_v2/fixtures/{label}.json",
            )
        )
    candidates = sorted(
        [
            refs["DECISION_V2"],
            refs["EVIDENCE_GRAPH_V2"],
            refs["GRADUATION"],
            refs["PORTFOLIO"],
            *extra_refs,
        ],
        key=lambda row: (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        ),
    )
    preactivation = build_preactivation_receipt(
        candidate_refs=candidates,
        expected_pointer_sha256="EMPTY",
        rollback_target_ref=None,
        blocker_codes=[],
        evaluated_at=AT,
    )
    refs["PREACTIVATION"] = _write_v2_document(
        tmp_path,
        preactivation,
        identity_field="preactivation_id",
        relative_path="results/v17_intelligence_v2/fixtures/preactivation.json",
    )
    documents["PREACTIVATION"] = preactivation
    return refs, documents, private_key, key_id


def _write_valid_marked_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs, documents, private_key, key_id = _research_nodes(tmp_path)
    publication_closure = build_publication_closure(
        canonical_strategy_id=STRATEGY,
        transaction_id="tx-001",
        closure_refs=refs,
        outcome_refs=[],
        built_at=AT,
    )
    closure_ref = _write_v2_document(
        tmp_path,
        publication_closure,
        identity_field="closure_id",
        relative_path=publication_closure["closure_path"],
    )
    fixture = write_synthetic_fixture_for_tests(
        tmp_path,
        strategy_id=STRATEGY,
        timestamp=AT,
        formal_overrides={"publication_profile": "INTELLIGENCE_V2_PUBLICATION_V1"},
        additional_formal_evidence_refs=[
            {
                "schema_id": PUBLICATION_CLOSURE_VERSION,
                "relative_path": closure_ref["relative_path"],
                "byte_sha256": closure_ref["byte_sha256"],
            }
        ],
        synthetic_only=True,
    )
    store = MainlineStore(tmp_path)
    run_stored = store.read(fixture.run_path, fixture.run_sha256)
    pointer_stored = store.read(fixture.pointer_path, fixture.pointer_sha256)
    run = parse_canonical(run_stored.data)
    pointer = parse_canonical(pointer_stored.data)
    legacy_run_ref = {
        "artifact_id": fixture.run_id,
        "artifact_version": run["schema_id"],
        "available_at": AT,
        "byte_sha256": run_stored.byte_sha256,
        "cutoff": run["created_at"],
        "relative_path": run_stored.relative_path,
        "semantic_sha256": run["semantic_sha256"],
    }
    target_pointer_ref = {
        "artifact_id": fixture.run_id,
        "artifact_version": pointer["schema_id"],
        "available_at": AT,
        "byte_sha256": pointer_stored.byte_sha256,
        "cutoff": pointer["updated_at"],
        "relative_path": pointer_stored.relative_path,
        "semantic_sha256": pointer["semantic_sha256"],
    }
    marker_closure = {
        "profile": documents["LEGACY_MARKER_PROFILE"],
        "transaction_id": "tx-001",
        "legacy_run_ref": legacy_run_ref,
        "target_pointer_ref": target_pointer_ref,
        "portfolio_ref": content_ref(documents["PORTFOLIO"], identity_field="receipt_id"),
        "risk_ref": content_ref(documents["EVIDENCE_GRAPH_V2"], identity_field="graph_id"),
        "graduation_ref": content_ref(documents["GRADUATION"], identity_field="graduation_id"),
        "built_at": AT,
    }
    marker = build_legacy_marker(**marker_closure)
    _write_v2_document(
        tmp_path,
        marker,
        identity_field="marker_id",
        relative_path=marker["marker_path"],
    )
    sidecar = build_activation_sidecar(
        marker=marker,
        marker_validation_closure=marker_closure,
        publication_closure=publication_closure,
        built_at=AT,
    )
    _write_v2_document(
        tmp_path,
        sidecar,
        identity_field="sidecar_id",
        relative_path=sidecar["sidecar_path"],
    )
    permit_args = {
        "action": "ACTIVATE",
        "canonical_strategy_id": STRATEGY,
        "subject_ref": content_ref(sidecar, identity_field="sidecar_id"),
        "expected_pointer_sha256": "EMPTY",
        "target_pointer_sha256": pointer_stored.byte_sha256,
        "issued_at": AT,
        "not_before": AT,
        "expires_at": PERMIT_END,
        "nonce": hashlib.sha256(b"reader-nonce").hexdigest(),
        "signer_key_id": key_id,
    }
    unsigned = build_action_permit(
        **permit_args,
        signature_base64=base64.b64encode(bytes(64)).decode("ascii"),
    )
    signature = private_key.sign(permit_message(unsigned["claims"]))
    permit = build_action_permit(
        **permit_args,
        signature_base64=base64.b64encode(signature).decode("ascii"),
    )
    permit_path = derive_publication_paths(
        strategy_id=STRATEGY,
        transaction_id="tx-001",
        target_pointer_sha256=pointer_stored.byte_sha256,
        run_id=fixture.run_id,
    )["activation_permit"]
    write_synthetic_governed_bytes_for_tests(
        tmp_path,
        relative_path=permit_path,
        raw=v2_canonical_bytes(permit),
        synthetic_only=True,
    )
    monkeypatch.setattr(
        intelligence_v2_reader,
        "_SELF_VALIDATORS",
        {
            "LEGACY_MARKER_PROFILE": validate_legacy_marker_profile,
            "PUBLICATION_OWNER_POLICY": validate_publication_owner_policy,
        },
    )


def test_valid_signed_marked_run_preserves_legacy_public_dto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_marked_fixture(tmp_path, monkeypatch)

    result = derive_mainline_state(tmp_path, strategy_id=STRATEGY)

    assert result.is_active is True
    assert result.blocker is None
    assert result.public_run is not None
    assert result.public_run["schema_id"] == "myquant.v17.v4.mainline-public-run.v1"


def test_v2_root_is_read_governed_but_not_write_authority(tmp_path: Path) -> None:
    raw = b'{"fixture":true}'
    stored = write_synthetic_governed_bytes_for_tests(
        tmp_path,
        relative_path=V2_PATH,
        raw=raw,
        synthetic_only=True,
    )
    assert MainlineStore(tmp_path).read(V2_PATH, stored.byte_sha256).data == raw
    with pytest.raises(MainlineStorageSecurityError, match="cannot publish external authority"):
        MainlineStore(tmp_path).write_exact_once(
            "results/v17_intelligence_v2/fixtures/forbidden.json",
            raw,
        )


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "casefold"])
def test_v2_read_root_rejects_filesystem_alias_attacks(
    tmp_path: Path,
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"fixture":true}'
    stored = write_synthetic_governed_bytes_for_tests(
        tmp_path,
        relative_path=V2_PATH,
        raw=raw,
        synthetic_only=True,
    )
    target = tmp_path / V2_PATH
    if attack == "casefold":
        listdir = os.listdir

        def colliding_listdir(value):  # type: ignore[no-untyped-def]
            names = listdir(value)
            if "publication-closure.json" in names:
                return [*names, "Publication-Closure.json"]
            return names

        monkeypatch.setattr(os, "listdir", colliding_listdir)
    else:
        original = tmp_path / "original.json"
        original.write_bytes(raw)
        original.chmod(0o600)
        target.unlink()
        if attack == "symlink":
            target.symlink_to(original)
        else:
            os.link(original, target)
    with pytest.raises(MainlineStorageSecurityError):
        MainlineStore(tmp_path).read(V2_PATH, stored.byte_sha256)


def test_synthetic_v2_writer_is_never_implicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="synthetic_only"):
        write_synthetic_governed_bytes_for_tests(
            tmp_path,
            relative_path=V2_PATH,
            raw=b"{}",
        )
    assert tuple(tmp_path.iterdir()) == ()
