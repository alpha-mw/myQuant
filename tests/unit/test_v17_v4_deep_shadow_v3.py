from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.deep_v3 import (
    ASSESSMENT_VERSION,
    DEEP_BUNDLE_V3,
    FUSION_TOP24_V2,
    MODULE_ORDER,
    DeepV3Error,
    compile_deep_v3,
)
from quant_investor.v17_v4_runtime.forward_shadow import (
    ForwardShadowError,
    build_shadow_readiness_v2,
    forward_artifact_formal_eligible,
    publish_forward_shadow,
    read_forward_shadow_session,
    reject_forward_artifact_for_formal,
)
from quant_investor.v17_v4_runtime.forward_fusion import (
    build_forward_fusion_top24,
    build_shadow_fusion_policy,
)
from quant_investor.v17_v4_runtime.source_storage import (
    ExactReferenceReader,
    GovernedStore,
)

NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}
STRATEGY = "forward-shadow"
CUTOFF = "2026-07-29T07:00:00Z"
SESSION = "2026-07-29"
RUN_ID = "forward-shadow-run"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ref(
    document: dict[str, Any],
    *,
    path: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(document)
    identity = artifact_identity_field(document["version"])
    return {
        "artifact_id": document[identity],
        "artifact_version": document["version"],
        "byte_sha256": _sha(raw),
        "cutoff": document["cutoff"],
        "relative_path": path,
        "semantic_sha256": document["semantic_sha256"],
        "strategy_id": document["strategy_id"],
    }


def _fusion_v2() -> dict[str, Any]:
    symbols = [f"{index:06d}.SZ" for index in range(1, 25)]
    policy = build_shadow_fusion_policy(
        policy_id="fusion-forward-policy",
        strategy_id=STRATEGY,
        effective_from_session=SESSION,
        created_at=CUTOFF,
    )
    return build_forward_fusion_top24(
        output_id="fusion-forward-v2",
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        pool_symbols=symbols,
        quant_scores={symbol: 25 - index for index, symbol in enumerate(symbols, start=1)},
        fundamental_scores={symbol: 25 - index for index, symbol in enumerate(symbols, start=1)},
        source_locator_semantic_sha256="4" * 64,
        input_bundle_sha256="2" * 64,
        factor_set_byte_sha256="1" * 64,
        policy=policy,
    )


def _manifest(
    fusion_ref: dict[str, str],
    *,
    raw_path: str,
    raw: bytes,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(1, 25):
        symbol = f"{index:06d}.SZ"
        if index > 1:
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
        rows.append(
            {
                "blocker_codes": [],
                "event_flags": [],
                "modules": [
                    {
                        "conclusion": "POSITIVE",
                        "evidence_ids": ["filing-one"],
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
                        "evidence_id": "filing-one",
                        "evidence_kind": "FILING",
                        "official_source_id": "cninfo",
                        "parser_id": "pdf-text",
                        "parser_sha256": "5" * 64,
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
            "request_id": "assessment-forward-v2",
            "rows": rows,
            "run_id": RUN_ID,
            "strategy_id": STRATEGY,
            "version": ASSESSMENT_VERSION,
        }
    )


def test_deep_v3_replays_complete_and_explicit_unavailable_rows(
    tmp_path: Path,
) -> None:
    store = GovernedStore(tmp_path)
    store.initialize()
    fusion = _fusion_v2()
    validate_artifact(fusion)
    fusion_path = "data/private/v17_v4_runs/forward-shadow-run/" "fusion_top24.v2.json"
    store.write_exact_once(
        fusion_path,
        canonical_resource_bytes(fusion),
    )
    raw = b"owner-prepositioned official filing"
    raw_path = "data/private/v17_v4_sources/deep_raw/" "000001.SZ/filing.bin"
    store.write_exact_once(raw_path, raw)
    manifest = _manifest(
        _ref(fusion, path=fusion_path),
        raw_path=raw_path,
        raw=raw,
    )
    validate_artifact(manifest)
    manifest_path = (
        "data/private/v17_v4_runs/forward-shadow-run/" "deep_assessment_manifest.v2.json"
    )
    manifest_raw = canonical_resource_bytes(manifest)
    store.write_exact_once(manifest_path, manifest_raw)
    result = compile_deep_v3(
        str(tmp_path),
        assessment_manifest_path=manifest_path,
        expected_assessment_manifest_sha256=_sha(manifest_raw),
        created_at=CUTOFF,
    )
    reader = ExactReferenceReader(tmp_path)
    bundle = load_canonical_resource(
        reader.read(
            result["bundle_ref"]["relative_path"],
            result["bundle_ref"]["byte_sha256"],
        ),
        label=DEEP_BUNDLE_V3,
    )
    assert bundle["rows"][0]["status"] == "COMPLETE"
    assert [row["status"] for row in bundle["rows"][0]["modules"]] == ["COMPLETE"] * 10
    unavailable = bundle["rows"][1]
    assert unavailable["buy_veto"] is True
    assert unavailable["target_after_deep"] == "0"
    assert [row["status"] for row in unavailable["modules"]] == ["UNAVAILABLE"] * 10


def test_missing_manifest_writes_only_blocked_readiness(
    tmp_path: Path,
) -> None:
    with pytest.raises(DeepV3Error, match="assessment_manifest"):
        compile_deep_v3(
            str(tmp_path),
            assessment_manifest_path=(
                "data/private/v17_v4_runs/missing/" "deep_assessment_manifest.v2.json"
            ),
            expected_assessment_manifest_sha256="9" * 64,
            created_at=CUTOFF,
        )
    result = build_shadow_readiness_v2(
        str(tmp_path),
        readiness_id="missing-deep-readiness",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        decision_session=SESSION,
        created_at=CUTOFF,
        blocker_codes=["deep_assessment_manifest_missing"],
        factor_refs_present=True,
    )
    assert result["model_output_present"] is False
    assert result["state"] == "FORWARD_SHADOW_BLOCKED"
    assert not (tmp_path / "results/v17_v4_shadow/strategies" / STRATEGY / "sessions").exists()


def _fake_ref(
    version: str,
    identity: str,
) -> dict[str, str]:
    return {
        "artifact_id": identity,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(f"{version}:{identity}".encode("ascii")).hexdigest(),
        "cutoff": CUTOFF,
        "relative_path": ("data/private/v17_v4_runs/forward-shadow-run/" f"{identity}.json"),
        "semantic_sha256": hashlib.sha256(
            f"semantic:{version}:{identity}".encode("ascii")
        ).hexdigest(),
        "strategy_id": STRATEGY,
    }


def _publication_refs() -> dict[str, dict[str, str]]:
    return {
        "source_locator_ref": _fake_ref(
            "myquant.v17.v4.research-source-locator.v2",
            "source",
        ),
        "initial_pool_ref": _fake_ref(
            "myquant.v17.v4.research-initial-pool-output.v2",
            "initial",
        ),
        "quant_branch_ref": _fake_ref(
            "myquant.v17.v4.research-quant-branch-output.v2",
            "quant",
        ),
        "fundamental_branch_ref": _fake_ref(
            ("myquant.v17.v4." "research-fundamental-branch-output.v2"),
            "fundamental",
        ),
        "fusion_top24_ref": _fake_ref(
            FUSION_TOP24_V2,
            "fusion",
        ),
        "deep_bundle_ref": _fake_ref(
            DEEP_BUNDLE_V3,
            "deep",
        ),
        "holdings_snapshot_ref": _fake_ref(
            "myquant.v17.v4.holdings-snapshot.v1",
            "holdings",
        ),
        "factor_set_pointer_ref": _fake_ref(
            ("myquant.v17.v4." "research-shadow-factor-set-pointer.v1"),
            "factor-pointer",
        ),
        "factor_set_ref": _fake_ref(
            "myquant.v17.v4.research-shadow-factor-set.v1",
            "factor-set",
        ),
        "fusion_observation_ref": _fake_ref(
            "myquant.v17.v4.shadow-fusion-observation.v1",
            "observation",
        ),
    }


def _patch_forward_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_investor.v17_v4_runtime import forward_shadow

    monkeypatch.setattr(
        forward_shadow,
        "_load_forward_inputs",
        lambda **_: {},
    )
    monkeypatch.setattr(
        forward_shadow,
        "_validate_closure",
        lambda **_: (["factor-a", "factor-b"], "7" * 64),
    )

    def reread_run(
        run_ref: dict[str, str],
        *,
        reader: ExactReferenceReader,
    ) -> dict[str, Any]:
        return load_canonical_resource(
            reader.read(
                run_ref["relative_path"],
                run_ref["byte_sha256"],
            ),
            label="shadow-run-v3",
        )

    monkeypatch.setattr(
        forward_shadow,
        "revalidate_forward_shadow_run",
        reread_run,
    )


def test_shadow_crash_or_pointer_drift_leave_no_session_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_forward_closure(monkeypatch)
    refs = _publication_refs()

    def crash() -> None:
        raise RuntimeError("injected crash before session")

    with pytest.raises(RuntimeError, match="injected crash"):
        publish_forward_shadow(
            str(tmp_path),
            shadow_run_id=RUN_ID,
            override_id="factor-assertion",
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            decision_session=SESSION,
            created_at=CUTOFF,
            factor_state_reread=crash,
            **refs,
        )
    session_path = (
        tmp_path / "results/v17_v4_shadow/strategies" / STRATEGY / "sessions" / f"{SESSION}.json"
    )
    assert not session_path.exists()
    with pytest.raises(
        ForwardShadowError,
        match="factor_pointer_drift",
    ):
        publish_forward_shadow(
            str(tmp_path),
            shadow_run_id=RUN_ID,
            override_id="factor-assertion",
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            decision_session=SESSION,
            created_at=CUTOFF,
            factor_state_reread=lambda: (
                {**refs["factor_set_pointer_ref"], "byte_sha256": "8" * 64},
                refs["factor_set_ref"],
            ),
            **refs,
        )
    assert not session_path.exists()
    published = publish_forward_shadow(
        str(tmp_path),
        shadow_run_id=RUN_ID,
        override_id="factor-assertion",
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        decision_session=SESSION,
        created_at=CUTOFF,
        factor_state_reread=lambda: (
            refs["factor_set_pointer_ref"],
            refs["factor_set_ref"],
        ),
        **refs,
    )
    assert published["run_created"] is False
    assert published["session_created"] is True
    assert session_path.exists()


def test_forward_family_is_rejected_for_formal_use() -> None:
    reference = _fake_ref(DEEP_BUNDLE_V3, "deep")
    assert forward_artifact_formal_eligible(reference) is False
    with pytest.raises(
        ForwardShadowError,
        match="formal_artifact_rejected",
    ):
        reject_forward_artifact_for_formal(reference)
