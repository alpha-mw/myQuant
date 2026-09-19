"""Production proof custody units; native capture admission is an explicit seam."""

import hashlib

import pytest

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.market import production_calendar_proof as proof
from quant_investor.market.next_session_proof import read_next_session_proof
from quant_investor.market import production_calendar_evidence as evidence
from quant_investor.operations.daily_contract import ContractError
from test_production_calendar_evidence import bound  # noqa: F401


@pytest.fixture
def publication(bound, monkeypatch):  # noqa: F811 - imported pytest fixture
    journal, info, transport = bound
    root = info["execution"]["payload"]["capture_root_name"]
    with journal.locked():
        for role in ("execution", "success"):
            raw = canonical_json_bytes(info[role])
            leaf = f"capture-{role}.json"
            journal.storage.write(str(journal.root / "calendar-future/captures" / root / leaf), raw)
            info[role + "_ref"] = {
                "relative_path": root + "/" + leaf,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
            transport[role + "_ref"] = info[role + "_ref"]
        path = evidence.transport_evidence_path(journal, info["execution_ref"])
        stored = journal.storage.write(path, canonical_json_bytes(transport))
        transport_ref = {"path": path, "sha256": stored.byte_sha256}
    native_ref = lambda leaf: {"relative_path": root + "/" + leaf, "byte_sha256": "e" * 64}
    info.update(
        eod_trade_date=journal.trade_date,
        observed_through_date="20260923",
        next_open_session="20260903",
        capture_root_ref={"capture_parent": "unit-only", "capture_root_name": root},
        transaction_ref=native_ref("capture-transaction.json"),
        provider_capture_refs=[],
        projection_sha256="f" * 64,
        policy_ref=native_ref("policy.json"),
        capability_ref=native_ref("capability.json"),
        source_limitations=["UNIT_CAPTURE_ADMISSION_SEAM"],
        projection=[],
        calendar_policy={},
    )
    monkeypatch.setattr(proof, "inspect_next_session_capture", lambda **kw: info)
    args = dict(
        workspace=str(journal.storage._io.workspace_root),
        eod_trade_date=journal.trade_date,
        execution=info["execution"],
        execution_ref=info["execution_ref"],
        success=info["success"],
        success_ref=info["success_ref"],
        transport_evidence_ref=transport_ref,
    )
    return journal, info, args


def test_publication_dispatch_and_idempotent_recovery(publication):
    journal, info, args = publication
    ref = proof.publish_production_next_session_proof(**args)
    before = journal.storage.read(ref["path"]).data
    assert proof.publish_production_next_session_proof(**args) == ref
    assert journal.storage.read(ref["path"]).data == before
    result = read_next_session_proof(
        workspace=args["workspace"], eod_trade_date=args["eod_trade_date"], publication_ref=ref
    )
    assert result["synthetic"] is False
    assert result["live_eligible"] is True  # Contract unit seam; not real-source acceptance.
    assert result["consumer_admission"] is False
    assert result["proof"]["source_classification"] == "INSTALLED_NATIVE_HTTPS_OBSERVED"
    assert result["proof"]["transport_evidence_ref"] == args["transport_evidence_ref"]


@pytest.mark.parametrize("schema", ["unknown", "cn-next-session-calendar-proof-publication.v1"])
def test_cross_version_or_unknown_publication_rejected(publication, schema):
    journal, _, args = publication
    ref = proof.publish_production_next_session_proof(**args)
    original = parse_canonical_json_bytes(journal.storage.read(ref["path"]).data)
    value = {**original, "schema_version": schema}
    # Same canonical expected publication path with resealed bytes via controlled
    # read seam: version dispatch must not reinterpret the production proof as v1.
    from quant_investor.market import next_session_proof as dispatcher

    read = dispatcher._read

    def replaced(j, r):
        return value if r == ref else read(j, r)

    from unittest.mock import patch

    with patch.object(dispatcher, "_read", replaced):
        with pytest.raises(ContractError):
            read_next_session_proof(
                workspace=args["workspace"],
                eod_trade_date=args["eod_trade_date"],
                publication_ref=ref,
            )


def test_completed_eod_blocks_any_new_publication(publication):
    journal, _, args = publication
    with journal.locked():
        journal.storage.write(str(journal.root / "completion.v1.json"), b"{}")
    with pytest.raises(ContractError, match="ALREADY_SELECTED"):
        proof.publish_production_next_session_proof(**args)
