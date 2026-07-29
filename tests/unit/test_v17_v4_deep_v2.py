from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.deep_v2 import (
    ASSESSMENT_VERSION,
    DEEP_BUNDLE_V2,
    DeepV2Error,
    compile_deep_v2,
    revalidate_deep_v2_bundle,
)
from quant_investor.v17_v4_runtime.source_storage import (
    ExactReferenceReader,
    GovernedStore,
)
from tests.unit.test_v17_v4_contract import _formal_intent

NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
STRATEGY_ID = "cn-aggressive-tech-manufacturing"
CUTOFF = "2026-07-28T07:00:00Z"
RUN_ID = "shadow-deep-v2-run"
MODULE_ORDER = (
    "financial_reconciliation",
    "business_model",
    "industry",
    "competition",
    "management",
    "valuation",
    "catalysts",
    "contrary_evidence",
    "falsification_conditions",
    "monitoring",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reference(
    artifact: dict[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(artifact)
    identity_field = artifact_identity_field(artifact["version"])
    return {
        "artifact_id": artifact[identity_field],
        "artifact_version": artifact["version"],
        "byte_sha256": _sha(raw),
        "cutoff": artifact["cutoff"],
        "relative_path": relative_path,
        "semantic_sha256": artifact["semantic_sha256"],
        "strategy_id": artifact["strategy_id"],
    }


def _fake_promotion_ref() -> dict[str, str]:
    return {
        "artifact_id": "promotion-shadow-v2",
        "artifact_version": ("myquant.v17.v4.fusion-promotion-receipt.v1"),
        "byte_sha256": "1" * 64,
        "cutoff": CUTOFF,
        "relative_path": ("data/private/v17_v4_runs/shadow-deep-v2-run/" "promotion.json"),
        "semantic_sha256": "2" * 64,
        "strategy_id": STRATEGY_ID,
    }


def _fusion() -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "output_id": "fusion-shadow-deep-v2",
            "promotion_receipt_ref": _fake_promotion_ref(),
            "protocol_version": "myquant.v17.v4",
            "rows": [
                {
                    "base_target": "0.03",
                    "fused_score": str(25 - index),
                    "rank": index,
                    "symbol": f"{index:06d}.SZ",
                }
                for index in range(1, 25)
            ],
            "run_id": RUN_ID,
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.fusion-top24.v1",
        }
    )


def _manifest(
    fusion_ref: dict[str, str],
    raw_documents: dict[str, tuple[str, bytes]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(1, 25):
        symbol = f"{index:06d}.SZ"
        if index == 24:
            rows.append(
                {
                    "blocker_codes": ["official_evidence_unavailable"],
                    "event_flags": [],
                    "modules": [],
                    "raw_documents": [],
                    "status": "UNAVAILABLE",
                    "symbol": symbol,
                }
            )
            continue
        raw_path, raw = raw_documents[symbol]
        evidence_id = f"filing-{index:02d}"
        rows.append(
            {
                "blocker_codes": [],
                "event_flags": (["financial_fraud_or_material_restatement"] if index == 23 else []),
                "modules": [
                    {
                        "conclusion": "POSITIVE",
                        "evidence_ids": [evidence_id],
                        "finding": f"{module_id} supported",
                        "module_id": module_id,
                        "score": "0.5",
                    }
                    for module_id in MODULE_ORDER
                ],
                "raw_documents": [
                    {
                        "available_at": CUTOFF,
                        "captured_at": CUTOFF,
                        "evidence_id": evidence_id,
                        "evidence_kind": "FILING",
                        "official_source_id": "cninfo",
                        "parser_id": "pdf-text",
                        "parser_sha256": "3" * 64,
                        "parser_version": "v1",
                        "published_at": CUTOFF,
                        "raw_byte_sha256": _sha(raw),
                        "raw_relative_path": raw_path,
                        "raw_size": len(raw),
                    }
                ],
                "status": "COMPLETE",
                "symbol": symbol,
            }
        )
    return seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "fusion_top24_ref": fusion_ref,
            "protocol_version": "myquant.v17.v4",
            "request_id": "deep-assessment-request",
            "rows": rows,
            "run_id": RUN_ID,
            "strategy_id": STRATEGY_ID,
            "version": ASSESSMENT_VERSION,
        }
    )


@pytest.fixture()
def deep_v2_workspace(tmp_path: Path) -> dict[str, Any]:
    store = GovernedStore(tmp_path)
    store.initialize()
    fusion = _fusion()
    validate_artifact(fusion)
    fusion_path = "data/private/v17_v4_runs/shadow-deep-v2-run/" "fusion_top24.json"
    store.write_exact_once(
        fusion_path,
        canonical_resource_bytes(fusion),
    )
    fusion_ref = _reference(fusion, relative_path=fusion_path)
    raw_documents: dict[str, tuple[str, bytes]] = {}
    for index in range(1, 24):
        symbol = f"{index:06d}.SZ"
        raw = f"official filing bytes for {symbol}".encode("ascii")
        raw_path = "data/private/v17_v4_sources/deep_raw/" f"{symbol}/filing.bin"
        store.write_exact_once(raw_path, raw)
        raw_documents[symbol] = (raw_path, raw)
    manifest = _manifest(fusion_ref, raw_documents)
    validate_artifact(manifest)
    manifest_path = "data/private/v17_v4_runs/shadow-deep-v2-run/" "deep_assessment_manifest.json"
    manifest_raw = canonical_resource_bytes(manifest)
    store.write_exact_once(manifest_path, manifest_raw)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha(manifest_raw),
        "raw_documents": raw_documents,
        "root": tmp_path,
    }


def test_compile_deep_v2_builds_replayable_shadow_only_closure(
    deep_v2_workspace: dict[str, Any],
) -> None:
    result = compile_deep_v2(
        deep_v2_workspace["root"],
        assessment_manifest_path=deep_v2_workspace["manifest_path"],
        expected_assessment_manifest_sha256=(deep_v2_workspace["manifest_sha256"]),
        created_at=CUTOFF,
    )
    assert result["status"] == "DEEP_V2_COMPILED"
    assert result["shadow_only"] is True
    assert result["formal_activation_eligible"] is False
    assert result["canary_evidence_eligible"] is False

    reader = ExactReferenceReader(deep_v2_workspace["root"])
    loader = lambda reference: reader.read(
        reference["relative_path"],
        reference["byte_sha256"],
    )
    bundle, fusion, manifest = revalidate_deep_v2_bundle(
        result["bundle_ref"],
        artifact_loader=loader,
    )
    assert bundle["version"] == DEEP_BUNDLE_V2
    assert fusion["run_id"] == RUN_ID
    assert manifest["version"] == ASSESSMENT_VERSION
    assert len(bundle["rows"]) == 24
    assert bundle["rows"][0]["signal"] == "0.5"
    assert bundle["rows"][0]["buy_veto"] is False
    assert bundle["rows"][0]["target_after_deep"] == "0.03"
    assert bundle["rows"][22]["buy_veto"] is True
    assert bundle["rows"][22]["target_after_deep"] == "0"
    assert bundle["rows"][23] == {
        "blocker_codes": ["official_evidence_unavailable"],
        "buy_veto": True,
        "event_scan_ref": None,
        "issuer_dossier_ref": None,
        "official_evidence_refs": [],
        "signal": None,
        "status": "UNAVAILABLE",
        "symbol": "000024.SZ",
        "target_after_deep": "0",
    }
    raw = loader(result["bundle_ref"])
    assert (
        load_canonical_artifact(
            raw,
            expected_version=DEEP_BUNDLE_V2,
            artifact_loader=loader,
        ).payload
        == bundle
    )


def test_compile_deep_v2_is_exact_once_idempotent(
    deep_v2_workspace: dict[str, Any],
) -> None:
    first = compile_deep_v2(
        deep_v2_workspace["root"],
        assessment_manifest_path=deep_v2_workspace["manifest_path"],
        expected_assessment_manifest_sha256=(deep_v2_workspace["manifest_sha256"]),
        created_at=CUTOFF,
    )
    second = compile_deep_v2(
        deep_v2_workspace["root"],
        assessment_manifest_path=deep_v2_workspace["manifest_path"],
        expected_assessment_manifest_sha256=(deep_v2_workspace["manifest_sha256"]),
        created_at=CUTOFF,
    )
    assert first["bundle_ref"] == second["bundle_ref"]
    assert first["created"] is True
    assert second["created"] is False


def test_compile_deep_v2_fails_closed_after_raw_evidence_tamper(
    deep_v2_workspace: dict[str, Any],
) -> None:
    raw_path, _ = deep_v2_workspace["raw_documents"]["000001.SZ"]
    absolute = deep_v2_workspace["root"] / raw_path
    absolute.write_bytes(b"tampered official filing")
    with pytest.raises(DeepV2Error, match="raw_evidence_readback"):
        compile_deep_v2(
            deep_v2_workspace["root"],
            assessment_manifest_path=(deep_v2_workspace["manifest_path"]),
            expected_assessment_manifest_sha256=(deep_v2_workspace["manifest_sha256"]),
            created_at=CUTOFF,
        )


def test_deep_assessment_rejects_unknown_module_evidence(
    deep_v2_workspace: dict[str, Any],
) -> None:
    manifest = deepcopy(deep_v2_workspace["manifest"])
    manifest.pop("semantic_sha256")
    manifest["rows"][0]["modules"][0]["evidence_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_artifact(seal_semantic(manifest))


def test_formal_v1_intent_rejects_shadow_only_deep_v2_reference() -> None:
    intent = deepcopy(_formal_intent())
    intent.pop("semantic_sha256")
    intent["deep_bundle_ref"]["artifact_version"] = DEEP_BUNDLE_V2
    with pytest.raises(ValueError, match="version mismatch"):
        validate_artifact(seal_semantic(intent))
