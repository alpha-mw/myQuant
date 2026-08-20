from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest
from quant_investor.contracts import (
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    parse_canonical_json_bytes,
    seal_artifact,
)

from quant_investor.factors.governance import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    FactorGovernanceError,
)
from quant_investor.factors.governance import production_authority as production_authority
from quant_investor.factors.governance import factor_production_prepare as prepare_module
from quant_investor.factors.governance.factor_production_prepare import (
    prepare_factor_production,
)
from quant_investor.factors.governance import legacy_zero_call as legacy_scanner_module
from quant_investor.factors.governance.legacy_zero_call import (
    _scan_release_legacy_zero_call_with_runner,
    scan_release_legacy_zero_call,
)
from quant_investor.factors.governance.bootstrap import build_bootstrap_factor_set
from quant_investor.factors.governance.bootstrap_evidence import (
    _DECISION_DOCUMENT,
    build_bootstrap_exception_evidence,
)
from quant_investor.factors.governance.receipt import _build_factor_validation_receipt
from quant_investor.factors.governance.implementations import (
    installed_implementation_rows,
    installed_semantic_row,
)
from quant_investor.factors.governance.source import role_schema
from quant_investor.factors.governance.production_authority import (
    FUNDAMENTAL_ADVISORY,
    FUNDAMENTAL_NOT_USED,
    build_factor_legacy_zero_call_certificate,
    build_factor_legacy_zero_call_certificate_for_release,
    build_factor_calendar_capture_custody_attestation,
    build_factor_production_market_input,
    build_factor_production_generation,
    build_factor_production_recomputation_evidence,
    build_factor_production_source_closure,
    recompute_factor_production_signals,
    replay_factor_production_recomputation_evidence,
    replay_factor_production_generation,
    validate_factor_production_generation,
    validate_factor_legacy_zero_call_certificate,
)
from quant_investor.market.tushare_calendar_authority import (
    build_trusted_provider_calendar_compilation,
)
from quant_investor.market.pit_universe import PITUniverseRecord
from quant_investor.factors.governance.bootstrap_selection import build_market_pit_selection
from quant_investor.system.components import seal_installed_component_manifest
from quant_investor.system.release_install import build_release_install_evidence
from quant_investor.system.store import SystemStore
from quant_investor.factors.governance.production_authority import system_store_source_resolver
from quant_investor.market.tushare_calendar_authority import build_calendar_authority_policy
from tests.unit.test_tushare_calendar_authority import (
    _captured_provider_production_case,
)
from tests.unit import test_tushare_calendar_authority as tushare_calendar_test_module
from tests.unit.test_tushare_calendar_authority import _raw_resolver as _trusted_raw_resolver

SYMBOLS = ["000001.SZ", "000002.SZ", "430001.BJ", "600000.SH"]


def _fixed_git_runner(
    *,
    final_commit: str,
    final_tree: str,
    extra_files: dict[str, bytes] | None = None,
):
    files = {
        "pyproject.toml": b"[project]\nname='myquant'\n",
        "quant_investor/factors/governance/legacy_zero_call.py": Path(
            "quant_investor/factors/governance/legacy_zero_call.py"
        ).read_bytes(),
        **(extra_files or {}),
    }

    def run(arguments, **_kwargs):
        if arguments == ["git", "rev-parse", f"{final_commit}^{{tree}}"]:
            return SimpleNamespace(returncode=0, stdout=(final_tree + "\n").encode(), stderr=b"")
        if arguments == ["git", "ls-tree", "-r", "--name-only", final_commit]:
            return SimpleNamespace(
                returncode=0,
                stdout=("\n".join(sorted(files)) + "\n").encode(),
                stderr=b"",
            )
        prefix = ["git", "show"]
        if arguments[:2] == prefix:
            path = arguments[2].split(":", 1)[1]
            return SimpleNamespace(returncode=0, stdout=files[path], stderr=b"")
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"unexpected")

    return run


def _published_calendar_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    def exact_release_install_closure(
        local_tmp_path: Path,
        *,
        release: dict[str, object] | None = None,
    ) -> tuple[bytes, dict[str, object], dict[str, object]]:
        assert release is not None
        release_payload = release["payload"]
        evidence = build_release_install_evidence(
            final_commit="4" * 40,
            final_tree="5" * 40,
            code_tree_sha256_value=release_payload["code_sha256"],
            git_code_manifest_sha256_value=release_payload["code_manifest_sha256"],
            release_ref=_ref(release),
            source_archive={
                "path": str(local_tmp_path / "release.tar.gz"),
                "byte_sha256": "6" * 64,
                "size": 1,
            },
            wheel={
                "path": str(local_tmp_path / "release.whl"),
                "byte_sha256": release_payload["wheel_sha256"],
                "size": 1,
            },
            install_root=local_tmp_path / "installed",
            python_executable=local_tmp_path / "installed/bin/python",
            python_executable_sha256="8" * 64,
            import_origin=local_tmp_path / "installed/quant_investor/__init__.py",
            installed_code_manifest_sha256=release_payload["code_manifest_sha256"],
            contract_catalog_sha256_value="a" * 64,
            lockfile_sha256="b" * 64,
            created_at="2026-08-14T00:00:00Z",
        )
        verification = {
            "wheel_sha256": release_payload["wheel_sha256"],
            "installed_code_manifest_sha256": release_payload["code_manifest_sha256"],
            "contract_catalog_sha256": evidence["payload"]["contract_catalog_sha256"],
            "import_origin": evidence["payload"]["import_origin"],
        }
        raw = canonical_json_bytes(
            {"release_install_evidence": evidence, "deployed_release": release}
        )
        return raw, release, {"evidence": evidence, "verification": verification}

    monkeypatch.setattr(
        tushare_calendar_test_module,
        "_fake_release_install_closure",
        exact_release_install_closure,
    )
    _workspace, input_root, release_ref, files, capture = _captured_provider_production_case(
        tmp_path, monkeypatch
    )
    capture_root = Path(capture["capture_root"])
    raw_by_ref: dict[bytes, bytes] = {}
    for path in sorted(capture_root.iterdir(), key=lambda value: value.name):
        raw = path.read_bytes()
        raw_by_ref[
            canonical_json_bytes(
                {
                    "relative_path": f"{capture_root.name}/{path.name}",
                    "byte_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        ] = raw
    for field in ("calendar_runtime_json_file_ref", "exchange_calendar_file_ref"):
        reference = files[field]
        raw = (input_root / reference["relative_path"]).read_bytes()
        raw_by_ref[canonical_json_bytes(reference)] = raw
    policy = parse_canonical_json_bytes(
        (input_root / files["calendar_authority_policy_file_ref"]["relative_path"]).read_bytes()
    )
    capability = parse_canonical_json_bytes(
        (
            input_root / files["trusted_provider_calendar_capability_file_ref"]["relative_path"]
        ).read_bytes()
    )
    captures = [
        parse_canonical_json_bytes((input_root / row["relative_path"]).read_bytes())
        for row in files["trusted_provider_calendar_capture_file_refs"]
    ]
    runtime = (
        pq.ParquetFile(input_root / files["exchange_calendar_file_ref"]["relative_path"])
        .read()
        .to_pandas()
    )
    market_sessions = [value.isoformat() for value in runtime["open_session"]][-100:]
    release_install_input_raw = (
        input_root / capture["release_install_input_file_ref"]["relative_path"]
    ).read_bytes()
    return {
        "workspace": _workspace,
        "input_root": input_root,
        "created_at": policy["created_at"],
        "capture_root": capture_root,
        "capture": capture,
        "raw_by_ref": raw_by_ref,
        "market_sessions": market_sessions,
        "release_ref": release_ref,
        "policy": policy,
        "capability": capability,
        "captures": captures,
        "docs": (capture_root / "documentation.raw").read_bytes(),
        "json_ref": files["calendar_runtime_json_file_ref"],
        "parquet_ref": files["exchange_calendar_file_ref"],
        "release_install_input_raw": release_install_input_raw,
        "release_install_input": parse_canonical_json_bytes(release_install_input_raw),
    }


def _write(path: Path, rows: list[dict[str, object]], *, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=role_schema(role)), path)
    path.chmod(0o600)


def _artifact_ref(identity: str) -> dict[str, str]:
    artifact = seal_artifact(
        "system.release",
        {
            "release_id": identity,
            "state": "OPERATIONAL",
            "code_sha256": "a" * 64,
            "wheel_sha256": "b" * 64,
            "code_manifest_sha256": "c" * 64,
        },
        created_at="2026-04-02T00:00:00Z",
    )
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _ref(artifact: dict[str, object]) -> dict[str, str]:
    return {
        "kind": str(artifact["kind"]),
        "contract_sha256": str(artifact["contract_sha256"]),
        "artifact_id": str(artifact["artifact_id"]),
        "semantic_sha256": str(artifact["semantic_sha256"]),
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _contract_artifact(
    kind: str,
    identity: str,
    *,
    payload_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    contract = get_contract(kind)
    payload: dict[str, object] = {field: None for field in contract.required_payload_fields}
    payload[contract.identity_field] = identity
    if payload_overrides is not None:
        payload.update(payload_overrides)
    return seal_artifact(kind, payload, created_at="2026-04-02T00:00:00Z")


def _structural_market_input_ref() -> dict[str, str]:
    return _ref(_contract_artifact("factor.production_market_input", "structural-market-input"))


def _structural_ref(kind: str, identity: str) -> dict[str, str]:
    return _ref(_contract_artifact(kind, identity))


def _release_install_evidence(
    release_ref: dict[str, str],
    root: Path,
    *,
    code_sha256: str = "a" * 64,
    code_manifest_sha256: str = "c" * 64,
    wheel_sha256: str = "b" * 64,
) -> dict[str, object]:
    return build_release_install_evidence(
        final_commit="a" * 40,
        final_tree="b" * 40,
        code_tree_sha256_value=code_sha256,
        git_code_manifest_sha256_value=code_manifest_sha256,
        release_ref=release_ref,
        source_archive={"path": str(root / "source.tar.gz"), "byte_sha256": "d" * 64, "size": 1},
        wheel={"path": str(root / "package.whl"), "byte_sha256": wheel_sha256, "size": 1},
        install_root=root / "install",
        python_executable=root / "install/bin/python",
        python_executable_sha256="e" * 64,
        import_origin=root / "install/quant_investor/__init__.py",
        installed_code_manifest_sha256=code_manifest_sha256,
        contract_catalog_sha256_value="f" * 64,
        lockfile_sha256="1" * 64,
        created_at="2026-04-02T00:00:00Z",
    )


def _release_install_verification(
    release_ref: dict[str, str],
    *,
    code_sha256: str = "a" * 64,
    code_manifest_sha256: str = "c" * 64,
    wheel_sha256: str = "b" * 64,
) -> dict[str, object]:
    return {
        "state": "PASS",
        "release_ref": release_ref,
        "source_archive_sha256": "d" * 64,
        "wheel_sha256": wheel_sha256,
        "code_tree_sha256": code_sha256,
        "installed_code_manifest_sha256": code_manifest_sha256,
        "contract_catalog_sha256": "f" * 64,
        "import_origin": "/tmp/install/quant_investor/__init__.py",
    }


def _calendar_custody_artifact(
    release_ref: dict[str, str],
    leaves: list[dict[str, object]],
    *,
    identity: str,
) -> dict[str, object]:
    rows = sorted(
        [
            {
                "relative_path": (
                    str(row["relative_path"])
                    if str(row["relative_path"]).startswith(identity + "/")
                    else f"{identity}/{row['relative_path']}"
                ),
                "byte_sha256": row["byte_sha256"],
                "size": row["size"],
            }
            for row in leaves
        ],
        key=lambda row: str(row["relative_path"]),
    )
    body = {
        "state": "VERIFIED",
        "activation_scope": "FACTOR_PRODUCTION",
        "capture_root_name": identity,
        "deployed_release_ref": release_ref,
        "capture_transaction_ref": _structural_ref(
            "system.trusted_provider_calendar_capture_transaction", identity + "-transaction"
        ),
        "capture_execution_ref": _structural_ref(
            "system.trusted_provider_calendar_capture_execution", identity + "-execution"
        ),
        "capture_success_ref": _structural_ref(
            "system.trusted_provider_calendar_capture_success", identity + "-success"
        ),
        "published_root_device": 1,
        "published_root_inode": 1,
        "published_leaf_manifest": rows,
        "published_leaf_manifest_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "verified_at": "2026-04-02T00:00:00Z",
    }
    return seal_artifact(
        "factor.production_calendar_capture_custody_attestation",
        {
            "calendar_capture_custody_attestation_id": "factor-calendar-custody-"
            + hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
            **body,
        },
        created_at="2026-04-02T00:00:00Z",
    )


def _inputs(tmp_path: Path, *, constant: bool = False) -> tuple[Path, Path, Path, str]:
    start = date(2026, 1, 2)
    sessions = [start + timedelta(days=index) for index in range(91)]
    as_of = sessions[-1].strftime("%Y%m%d")
    calendar = tmp_path / "calendar.parquet"
    pit = tmp_path / "pit.parquet"
    market = tmp_path / "market.parquet"
    _write(
        calendar,
        [
            {
                "ordinal": index,
                "open_session": session,
                "opens_at_utc": datetime.combine(session, time(1, 30), tzinfo=timezone.utc),
                "closes_at_utc": datetime.combine(session, time(7, 0), tzinfo=timezone.utc),
            }
            for index, session in enumerate(sessions)
        ],
        role="exchange_calendar",
    )
    _write(
        pit,
        [
            {
                "signal_session": sessions[-1],
                "symbol": symbol,
                "industry": "test",
                "total_mv": float(1_000_000 + index),
                "tradable": True,
            }
            for index, symbol in enumerate(SYMBOLS)
        ],
        role="pit_universe",
    )
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for session_index, session in enumerate(sessions):
            scalar = 1.0 if constant else float(symbol_index + 1)
            rows.append(
                {
                    "trade_date": session,
                    "symbol": symbol,
                    "adj_close": 10.0 + scalar * session_index * 0.01,
                    "amount": 1000.0 + scalar * (session_index + 1),
                    "vol": 100.0 + scalar * (session_index % 19),
                }
            )
    _write(market, rows, role="market_history")
    return calendar, pit, market, as_of


def _source_shas(calendar: Path, pit: Path, market: Path) -> dict[str, str]:
    return {
        "exchange_calendar_sha256": hashlib.sha256(calendar.read_bytes()).hexdigest(),
        "pit_universe_sha256": hashlib.sha256(pit.read_bytes()).hexdigest(),
        "market_history_sha256": hashlib.sha256(market.read_bytes()).hexdigest(),
    }


def _market_pit_selection(
    *,
    as_of: str,
    pit_path: Path,
) -> dict[str, object]:
    def file_ref(path: str, raw: bytes) -> dict[str, str]:
        return {"relative_path": path, "byte_sha256": hashlib.sha256(raw).hexdigest()}

    pit_raw = pit_path.read_bytes()
    membership_ref = file_ref(pit_path.name, pit_raw)
    pit_manifest_ref = file_ref("pit-manifest.json", b"pit-manifest")
    market_bound_ref = file_ref("market-bound-pit.json", b"market-bound-pit")
    observed_ref = file_ref("observed-current-pit.json", b"observed-current-pit")
    pointer_ref = file_ref("market-pointer.json", b"market-pointer")
    snapshot_ref = file_ref("market-snapshot.json", b"market-snapshot")
    expected_scope_sha = hashlib.sha256("\n".join(sorted(SYMBOLS)).encode("utf-8")).hexdigest()
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "classification_sets_disjoint": True,
        "true_missing_symbols": [],
        "expected_scope_count": len(SYMBOLS),
        "coverage_complete_count": len(SYMBOLS),
        "non_blocking_absent_symbols": [],
        "latest_complete_trade_date": as_of,
        "coverage_trade_date": as_of,
        "upsert_target_trade_date": as_of,
        "expected_scope_sha256": expected_scope_sha,
        "pit_generation_id": f"pit-{as_of}-factor-test",
        "pit_generation_manifest_sha256": pit_manifest_ref["byte_sha256"],
        "pit_membership_sha256": membership_ref["byte_sha256"],
    }
    pointer = {
        "status": "OK",
        "blockers": [],
        "snapshot_id": f"market-{as_of}-factor-test",
        "latest_complete_trade_date": as_of,
        "coverage": coverage,
    }
    market_bound = {
        "discovery_schema_version": "cn_pit_universe_latest.v1",
        "generation_id": coverage["pit_generation_id"],
        "generation_manifest_sha256": pit_manifest_ref["byte_sha256"],
        "canonical_sha256": membership_ref["byte_sha256"],
    }
    return build_market_pit_selection(
        as_of=as_of,
        market_pointer_file_ref=pointer_ref,
        market_snapshot_manifest_file_ref=snapshot_ref,
        market_bound_pit_pointer_file_ref=market_bound_ref,
        pit_generation_manifest_file_ref=pit_manifest_ref,
        pit_membership_file_ref=membership_ref,
        observed_current_pit_pointer_file_ref=observed_ref,
        market_pointer=pointer,
        market_snapshot_manifest=pointer,
        market_bound_pit_pointer=market_bound,
        pit_generation_manifest={
            "generation_id": coverage["pit_generation_id"],
            "canonical_sha256": membership_ref["byte_sha256"],
        },
        observed_current_pit_pointer={
            "discovery_schema_version": "cn_pit_universe_latest.v1",
            "generation_id": f"pit-{as_of}-observed-current",
        },
        created_at="2026-04-02T00:00:00Z",
    )


def test_strict_factor_production_recomputes_low_w80_and_keeps_w75_control(
    tmp_path: Path,
) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path)
    result = recompute_factor_production_signals(
        exchange_calendar_path=calendar,
        pit_universe_path=pit,
        market_history_path=market,
        as_of=as_of,
        **_source_shas(calendar, pit, market),
    )
    assert {row["factor_id"]: row["weight"] for row in result["active_factor_rows"]} == {
        LOW_DOLLAR_VOLUME: "0.500000000000",
        BLEND_W80: "0.500000000000",
    }
    assert result["control_rows"] == [
        {
            "factor_id": BLEND_W75_CONTROL,
            "spec_id": result["control_rows"][0]["spec_id"],
            "direction": "HIGHER_IS_BETTER",
            "required_source_roles": ["EXCHANGE_CALENDAR", "MARKET", "PIT_MEMBERSHIP"],
            "weight": "0.000000000000",
            "role": "CONTROL_ONLY",
            "selectable": False,
        }
    ]
    assert [row["factor_id"] for row in result["signal_statistics"]] == [
        LOW_DOLLAR_VOLUME,
        BLEND_W80,
    ]
    assert all(row["finite_count"] == len(SYMBOLS) for row in result["signal_statistics"])
    assert all(row["distinct_finite_count"] > 1 for row in result["signal_statistics"])
    assert result["fundamental_dependency_state"] == FUNDAMENTAL_NOT_USED
    assert result["fundamental_freshness_policy"] == FUNDAMENTAL_ADVISORY


def test_strict_factor_production_rejects_constant_low_or_w80_signal(tmp_path: Path) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path, constant=True)
    with pytest.raises(FactorGovernanceError, match="empty or constant"):
        recompute_factor_production_signals(
            exchange_calendar_path=calendar,
            pit_universe_path=pit,
            market_history_path=market,
            as_of=as_of,
            **_source_shas(calendar, pit, market),
        )


def test_factor_production_requires_exact_calendar_cutoff(tmp_path: Path) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path)
    with pytest.raises(FactorGovernanceError, match="calendar"):
        recompute_factor_production_signals(
            exchange_calendar_path=calendar,
            pit_universe_path=pit,
            market_history_path=market,
            as_of="2026-04-03",
            **_source_shas(calendar, pit, market),
        )


def test_market_expected_scope_mismatch_is_terminal() -> None:
    symbols = ["000001.SZ", "600000.SH"]
    with pytest.raises(FactorGovernanceError, match="expected scope"):
        production_authority._require_market_scope_sha256(symbols, "f" * 64)


def test_market_scope_must_be_inside_canonical_pit_and_equal_factor_projection() -> None:
    scope = ["000001.SZ", "600000.SH"]
    with pytest.raises(FactorGovernanceError, match="outside canonical PIT"):
        production_authority._require_market_pit_scope_relation(
            market_scope_symbols=scope,
            canonical_pit_symbols=["000001.SZ"],
            factor_projection_symbols=scope,
        )
    with pytest.raises(FactorGovernanceError, match="projection cohort differs"):
        production_authority._require_market_pit_scope_relation(
            market_scope_symbols=scope,
            canonical_pit_symbols=[*scope, "430001.BJ"],
            factor_projection_symbols=["000001.SZ", "430001.BJ"],
        )


def test_factor_production_artifacts_keep_fundamental_not_used_and_no_authority(
    tmp_path: Path,
) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path)
    recomputation = recompute_factor_production_signals(
        exchange_calendar_path=calendar,
        pit_universe_path=pit,
        market_history_path=market,
        as_of=as_of,
        **_source_shas(calendar, pit, market),
    )
    refs = [_artifact_ref(f"ref-{index}") for index in range(12)]
    certificate = build_factor_legacy_zero_call_certificate(
        final_commit="a" * 40,
        final_tree="b" * 40,
        resolver_inventory_ref=refs[0],
        verification_module_path="quant_investor/factors/governance/production_authority.py",
        verification_module_sha256="d" * 64,
        verification_command="quant-investor factor verify-legacy-zero-call",
        stdout_sha256="e" * 64,
        stderr_sha256="f" * 64,
        verified_at="2026-04-02T00:00:00Z",
    )
    certificate_ref = {
        "kind": certificate["kind"],
        "contract_sha256": certificate["contract_sha256"],
        "artifact_id": certificate["artifact_id"],
        "semantic_sha256": certificate["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(certificate),
    }
    closure = build_factor_production_source_closure(
        deployed_release_ref=refs[2],
        release_install_evidence_ref=_structural_ref(
            "system.release_install_evidence", "release-install-source"
        ),
        release_install_input_source_ref=_structural_ref(
            "system.source_object", "release-install-input-source"
        ),
        release_install_verification=_release_install_verification(refs[2]),
        market_pit_selection_ref=_structural_ref(
            "factor.production_market_pit_selection", "selection-source"
        ),
        market_scope_source_ref=_structural_ref("system.source_object", "scope-source"),
        calendar_authority_policy_ref=refs[4],
        calendar_compilation_ref=refs[5],
        calendar_capture_custody_attestation_ref=_structural_ref(
            "factor.production_calendar_capture_custody_attestation", "calendar-custody"
        ),
        factor_source_bundle_ref=_structural_ref("system.source_bundle", "factor-sources"),
        factor_policy_ref=refs[7],
        factor_active_set_ref=refs[8],
        factor_validation_attestation_ref=refs[9],
        factor_implementation_refs=sorted(refs[10:12], key=lambda row: row["artifact_id"]),
        legacy_zero_call_ref=certificate_ref,
        market_input_ref=_structural_market_input_ref(),
        created_at="2026-04-02T00:00:00Z",
    )
    evidence = build_factor_production_recomputation_evidence(
        source_closure=closure,
        deployed_release_ref=refs[2],
        factor_active_set_ref=refs[8],
        recomputation=recomputation,
        created_at="2026-04-02T00:00:00Z",
    )
    assert closure["payload"]["fundamental_dependency_state"] == FUNDAMENTAL_NOT_USED
    assert closure["payload"]["system_authority"] == "NONE"
    assert evidence["payload"]["admission_route"] == "BOOTSTRAP_EXCEPTION"
    assert evidence["payload"]["producer_identity"] == "NOT_CLAIMED"
    generation = build_factor_production_generation(
        source_closure=closure,
        recomputation_evidence=evidence,
        created_at="2026-04-02T00:00:00Z",
    )
    assert validate_factor_production_generation(generation)["payload"]["state"] == "OPERATIONAL"


def test_recomputation_evidence_rejects_policy_row_mutation(tmp_path: Path) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path)
    recomputation = recompute_factor_production_signals(
        exchange_calendar_path=calendar,
        pit_universe_path=pit,
        market_history_path=market,
        as_of=as_of,
        **_source_shas(calendar, pit, market),
    )
    refs = [_artifact_ref(f"ref-{index}") for index in range(12)]
    certificate = build_factor_legacy_zero_call_certificate(
        final_commit="a" * 40,
        final_tree="b" * 40,
        resolver_inventory_ref=refs[0],
        verification_module_path="quant_investor/factors/governance/production_authority.py",
        verification_module_sha256="d" * 64,
        verification_command="quant-investor factor verify-legacy-zero-call",
        stdout_sha256="e" * 64,
        stderr_sha256="f" * 64,
        verified_at="2026-04-02T00:00:00Z",
    )
    closure = build_factor_production_source_closure(
        deployed_release_ref=refs[2],
        release_install_evidence_ref=_structural_ref(
            "system.release_install_evidence", "release-install-signal"
        ),
        release_install_input_source_ref=_structural_ref(
            "system.source_object", "release-install-input-source-2"
        ),
        release_install_verification=_release_install_verification(refs[2]),
        market_pit_selection_ref=_structural_ref(
            "factor.production_market_pit_selection", "selection-source-2"
        ),
        market_scope_source_ref=_structural_ref("system.source_object", "scope-source-2"),
        calendar_authority_policy_ref=refs[4],
        calendar_compilation_ref=refs[5],
        calendar_capture_custody_attestation_ref=_structural_ref(
            "factor.production_calendar_capture_custody_attestation", "calendar-custody-2"
        ),
        factor_source_bundle_ref=_structural_ref("system.source_bundle", "factor-sources-2"),
        factor_policy_ref=refs[7],
        factor_active_set_ref=refs[8],
        factor_validation_attestation_ref=refs[9],
        factor_implementation_refs=sorted(refs[10:12], key=lambda row: row["artifact_id"]),
        legacy_zero_call_ref=_ref(certificate),
        market_input_ref=_structural_market_input_ref(),
        created_at="2026-04-02T00:00:00Z",
    )
    tampered = dict(recomputation)
    tampered["active_factor_rows"] = [dict(row) for row in recomputation["active_factor_rows"]]
    tampered["active_factor_rows"][0]["weight"] = "0.900000000000"
    with pytest.raises(FactorGovernanceError, match="active/control policy"):
        build_factor_production_recomputation_evidence(
            source_closure=closure,
            deployed_release_ref=refs[2],
            factor_active_set_ref=refs[8],
            recomputation=tampered,
            created_at="2026-04-02T00:00:00Z",
        )


def test_recomputation_evidence_rejects_immutable_signal_value_mutation(
    tmp_path: Path,
) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path)
    recomputation = recompute_factor_production_signals(
        exchange_calendar_path=calendar,
        pit_universe_path=pit,
        market_history_path=market,
        as_of=as_of,
        **_source_shas(calendar, pit, market),
    )
    refs = [_artifact_ref(f"signal-ref-{index}") for index in range(12)]
    certificate = build_factor_legacy_zero_call_certificate(
        final_commit="a" * 40,
        final_tree="b" * 40,
        resolver_inventory_ref=refs[0],
        verification_module_path="quant_investor/factors/governance/production_authority.py",
        verification_module_sha256="d" * 64,
        verification_command="quant-investor factor verify-legacy-zero-call",
        stdout_sha256="e" * 64,
        stderr_sha256="f" * 64,
        verified_at="2026-04-02T00:00:00Z",
    )
    closure = build_factor_production_source_closure(
        deployed_release_ref=refs[2],
        release_install_evidence_ref=_structural_ref(
            "system.release_install_evidence", "release-install-signal"
        ),
        release_install_input_source_ref=_structural_ref(
            "system.source_object", "release-install-input-signal"
        ),
        release_install_verification=_release_install_verification(refs[2]),
        market_pit_selection_ref=_structural_ref(
            "factor.production_market_pit_selection", "signal-selection"
        ),
        market_scope_source_ref=_structural_ref("system.source_object", "signal-scope"),
        calendar_authority_policy_ref=refs[4],
        calendar_compilation_ref=refs[5],
        calendar_capture_custody_attestation_ref=_structural_ref(
            "factor.production_calendar_capture_custody_attestation", "signal-custody"
        ),
        factor_source_bundle_ref=_structural_ref("system.source_bundle", "signal-sources"),
        factor_policy_ref=refs[7],
        factor_active_set_ref=refs[8],
        factor_validation_attestation_ref=refs[9],
        factor_implementation_refs=sorted(refs[10:12], key=lambda row: row["artifact_id"]),
        legacy_zero_call_ref=_ref(certificate),
        market_input_ref=_structural_market_input_ref(),
        created_at="2026-04-02T00:00:00Z",
    )
    tampered = dict(recomputation)
    tampered["signal_values"] = {
        factor_id: dict(values) for factor_id, values in recomputation["signal_values"].items()
    }
    first_symbol = next(iter(tampered["signal_values"][LOW_DOLLAR_VOLUME]))
    tampered["signal_values"][LOW_DOLLAR_VOLUME][first_symbol] = (999.0).hex()
    with pytest.raises(FactorGovernanceError, match="immutable signal SHA"):
        build_factor_production_recomputation_evidence(
            source_closure=closure,
            deployed_release_ref=refs[2],
            factor_active_set_ref=refs[8],
            recomputation=tampered,
            created_at="2026-04-02T00:00:00Z",
        )


def test_legacy_zero_call_certificate_replays_fixed_release_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver_ref = _artifact_ref("resolver-inventory")
    runner = _fixed_git_runner(final_commit="a" * 40, final_tree="b" * 40)
    monkeypatch.setattr(legacy_scanner_module.subprocess, "run", runner)
    certificate = build_factor_legacy_zero_call_certificate_for_release(
        repository_root=Path.cwd(),
        final_commit="a" * 40,
        final_tree="b" * 40,
        resolver_inventory_ref=resolver_ref,
        verified_at="2026-04-02T00:00:00Z",
    )
    validate_factor_legacy_zero_call_certificate(
        certificate,
        repository_root=Path.cwd(),
    )


@pytest.mark.parametrize(
    ("path", "raw", "field"),
    [
        (
            "quant_investor/live_import.py",
            b"import quant_investor.v17_v4_runtime.foo\n",
            "active_legacy_import_count",
        ),
        (
            "quant_investor/live_dynamic_import.py",
            b'import importlib\nimportlib.import_module("quant_investor.v17_v4_runtime.foo")\n',
            "active_legacy_import_count",
        ),
        (
            "quant_investor/live_dynamic_import_alias.py",
            b'import importlib as il\nil.import_module("quant_investor.v17_v4_contract.foo")\n',
            "active_legacy_import_count",
        ),
        (
            "quant_investor/live_static_from.py",
            b"from quant_investor import v17_v4_runtime as retired\n",
            "active_legacy_import_count",
        ),
        (
            "quant_investor/live_call.py",
            b"quant_investor.v17_v4_runtime.run()\n",
            "active_legacy_call_count",
        ),
        (
            "quant_investor/live_hash.py",
            b'import hashlib\nhashlib.sha256("quant_investor/v17_v4_runtime/x")\n',
            "active_legacy_path_hash_count",
        ),
        (
            "quant_investor/live_open.py",
            b'from pathlib import Path\nPath("quant_investor/v17_v4_runtime/x.py").read_bytes()\n',
            "active_legacy_path_hash_count",
        ),
        (
            "quant_investor/live_open_alias.py",
            b'from builtins import open as op\nop("quant_investor/v17_v4_contract/x.py")\n',
            "active_legacy_path_hash_count",
        ),
        (
            "quant_investor/live_subprocess.py",
            b"import subprocess\n"
            b'subprocess.run(["python", "quant_investor/v17_v4_runtime/x.py"])\n',
            "active_legacy_path_hash_count",
        ),
        (
            "quant_investor/live_subprocess_alias.py",
            b'import subprocess as sp\nsp.run(["python", "quant_investor/v17_v4_runtime/x.py"])\n',
            "active_legacy_path_hash_count",
        ),
        (
            "pyproject.toml",
            b'[project.scripts]\nlegacy="quant_investor.v17_v4_runtime:main"\n',
            "legacy_entrypoint_count",
        ),
    ],
)
def test_fixed_legacy_scanner_detects_each_nonzero_lane(
    path: str,
    raw: bytes,
    field: str,
) -> None:
    result = _scan_release_legacy_zero_call_with_runner(
        repository_root=Path.cwd(),
        final_commit="a" * 40,
        final_tree="b" * 40,
        resolver_inventory_ref=_artifact_ref("resolver-inventory-negative"),
        process_runner=_fixed_git_runner(
            final_commit="a" * 40,
            final_tree="b" * 40,
            extra_files={path: raw},
        ),
    )
    assert result[field] > 0


def test_public_legacy_scanner_cannot_accept_a_forged_runner() -> None:
    with pytest.raises(TypeError, match="process_runner"):
        scan_release_legacy_zero_call(
            repository_root=Path.cwd(),
            final_commit="a" * 40,
            final_tree="b" * 40,
            resolver_inventory_ref=_artifact_ref("resolver-inventory-forged"),
            process_runner=_fixed_git_runner(final_commit="a" * 40, final_tree="b" * 40),
        )


def test_factor_source_topology_requires_stable_owner_only_double_read(tmp_path: Path) -> None:
    calendar, pit, market, _as_of = _inputs(tmp_path)
    source_paths = {
        "exchange_calendar": calendar,
        "market_history": market,
        "pit_universe": pit,
    }
    artifacts: dict[str, dict[str, object]] = {}
    source_rows: list[dict[str, object]] = []
    for role, path in source_paths.items():
        artifact = seal_artifact(
            "system.source_object",
            {
                "source_object_id": role,
                "source_root_id": "factor-test-root",
                "relative_path": path.name,
                "media_type": "application/vnd.apache.parquet",
                "source_format": "PARQUET",
                "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            created_at="2026-04-02T00:00:00Z",
        )
        ref = _ref(artifact)
        artifacts[ref["byte_sha256"]] = artifact
        source_rows.append({"role": role, "source_ref": ref})
    calendar_bundle = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": "factor-calendar-capture",
            "state": "IMMUTABLE",
            "sources": [source_rows[0]],
        },
        created_at="2026-04-02T00:00:00Z",
    )
    pit_bundle = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": "factor-pit-capture",
            "state": "IMMUTABLE",
            "sources": [source_rows[2]],
        },
        created_at="2026-04-02T00:00:00Z",
    )
    market_bundle = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": "factor-market-capture",
            "state": "IMMUTABLE",
            "sources": [
                {"role": "factor-market-history", "source_ref": source_rows[1]["source_ref"]}
            ],
        },
        created_at="2026-04-02T00:00:00Z",
    )
    calendar_bundle_ref = _ref(calendar_bundle)
    pit_bundle_ref = _ref(pit_bundle)
    market_bundle_ref = _ref(market_bundle)
    artifacts[calendar_bundle_ref["byte_sha256"]] = calendar_bundle
    artifacts[pit_bundle_ref["byte_sha256"]] = pit_bundle
    artifacts[market_bundle_ref["byte_sha256"]] = market_bundle
    bundle = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": "factor-inputs",
            "state": "IMMUTABLE",
            "sources": [
                {"role": "exchange_calendar", "source_ref": calendar_bundle_ref},
                {"role": "market_history", "source_ref": market_bundle_ref},
                {"role": "pit_universe", "source_ref": pit_bundle_ref},
            ],
        },
        created_at="2026-04-02T00:00:00Z",
    )
    bundle_ref = _ref(bundle)
    artifacts[bundle_ref["byte_sha256"]] = bundle

    def artifact_resolver(ref: dict[str, str]) -> dict[str, object]:
        return artifacts[ref["byte_sha256"]]

    def source_resolver(ref: dict[str, str], maximum_bytes: int) -> tuple[dict[str, object], bytes]:
        source = artifacts[ref["byte_sha256"]]
        payload = source["payload"]
        path = source_paths[
            next(
                role for role, item in source_paths.items() if item.name == payload["relative_path"]
            )
        ]
        raw = path.read_bytes()
        metadata = path.stat()
        return (
            {
                "source_object_ref": ref,
                "source_root_id": payload["source_root_id"],
                "relative_path": payload["relative_path"],
                "media_type": payload["media_type"],
                "source_format": payload["source_format"],
                "byte_sha256": payload["byte_sha256"],
                "size": len(raw),
                "stat_identity": {
                    "st_ctime_ns": metadata.st_ctime_ns,
                    "st_dev": metadata.st_dev,
                    "st_gid": metadata.st_gid,
                    "st_ino": metadata.st_ino,
                    "st_mode": metadata.st_mode,
                    "st_mtime_ns": metadata.st_mtime_ns,
                    "st_nlink": metadata.st_nlink,
                    "st_size": metadata.st_size,
                    "st_uid": metadata.st_uid,
                },
            },
            raw,
        )

    inputs, branches = production_authority._factor_source_topology(
        bundle_ref,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
    )
    assert inputs == {}
    assert set(branches) == {"exchange_calendar", "market_history", "pit_universe"}
    assert branches["market_history"][0]["raw"] == market.read_bytes()

    def link_drift(ref: dict[str, str], maximum_bytes: int) -> tuple[dict[str, object], bytes]:
        descriptor, raw = source_resolver(ref, maximum_bytes)
        descriptor["stat_identity"] = dict(descriptor["stat_identity"])
        descriptor["stat_identity"]["st_nlink"] = 2
        return descriptor, raw

    with pytest.raises(FactorGovernanceError, match="storage identity"):
        production_authority._factor_source_topology(
            bundle_ref,
            artifact_resolver=artifact_resolver,
            source_resolver=link_drift,
        )


def test_deep_factor_replay_recomputes_exact_custody_bytes(
    tmp_path: Path,
) -> None:
    calendar, pit, market, as_of = _inputs(tmp_path)
    source_paths = {
        "exchange_calendar": calendar,
        "market_history": market,
        "pit_universe": pit,
    }
    artifacts: dict[str, dict[str, object]] = {}

    def stash(artifact: dict[str, object]) -> dict[str, str]:
        ref = _ref(artifact)
        artifacts[ref["byte_sha256"]] = artifact
        return ref

    source_rows: list[dict[str, object]] = []
    for role, path in source_paths.items():
        source_rows.append(
            {
                "role": role,
                "source_ref": stash(
                    seal_artifact(
                        "system.source_object",
                        {
                            "source_object_id": role,
                            "source_root_id": "factor-deep-test-root",
                            "relative_path": path.name,
                            "media_type": "application/vnd.apache.parquet",
                            "source_format": "PARQUET",
                            "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        },
                        created_at="2026-04-02T00:00:00Z",
                    )
                ),
            }
        )
    source_bundle_ref = stash(
        seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": "factor-deep-inputs",
                "state": "IMMUTABLE",
                "sources": source_rows,
            },
            created_at="2026-04-02T00:00:00Z",
        )
    )
    selection_ref = stash(_market_pit_selection(as_of=as_of, pit_path=pit))
    release_ref = stash(
        seal_artifact(
            "system.release",
            {
                "release_id": "factor-deep-release",
                "state": "OPERATIONAL",
                "code_sha256": "a" * 64,
                "wheel_sha256": "b" * 64,
                "code_manifest_sha256": "c" * 64,
            },
            created_at="2026-04-02T00:00:00Z",
        )
    )
    calendar_policy_ref = stash(
        build_calendar_authority_policy(
            created_at="2026-04-02T00:00:00Z",
            authority_route="EXCHANGE_OFFICIAL",
            pit_exchange_ids=["BSE", "SSE", "SZSE"],
        )
    )
    calendar_source = artifacts[source_rows[0]["source_ref"]["byte_sha256"]]["payload"]
    calendar_compilation_ref = stash(
        _contract_artifact(
            "system.exchange_calendar_compilation",
            "factor-deep-calendar",
            payload_overrides={
                "policy_ref": calendar_policy_ref,
                "cutoff_date": as_of,
                "calendar_parquet_file_ref": {
                    "relative_path": calendar_source["relative_path"],
                    "byte_sha256": calendar_source["byte_sha256"],
                },
                "pit_exchange_ids": ["BSE", "SSE", "SZSE"],
            },
        )
    )
    resolver_inventory_ref = release_ref
    legacy = build_factor_legacy_zero_call_certificate(
        final_commit="a" * 40,
        final_tree="b" * 40,
        resolver_inventory_ref=resolver_inventory_ref,
        verification_module_path="quant_investor/factors/governance/production_authority.py",
        verification_module_sha256="d" * 64,
        verification_command="quant-investor factor verify-legacy-zero-call",
        stdout_sha256="e" * 64,
        stderr_sha256="f" * 64,
        verified_at="2026-04-02T00:00:00Z",
    )
    legacy_ref = stash(legacy)
    source_artifacts: dict[str, dict[str, object]] = {"code": artifacts[release_ref["byte_sha256"]]}
    for role in (
        "decision_source",
        "exchange_calendar",
        "implementation",
        "market",
        "pit_universe",
        "recomputation",
        "source_generation",
    ):
        sources: list[dict[str, object]] = []
        inner_role = {
            "decision_source": "bootstrap_decision",
            "implementation": "implementation_tree_manifest",
        }.get(role)
        if inner_role is not None:
            leaf = seal_artifact(
                "system.source_object",
                {
                    "source_object_id": f"factor-deep-{inner_role}",
                    "source_root_id": "factor-deep-test-root",
                    "relative_path": f"{inner_role}.json",
                    "media_type": "application/json",
                    "source_format": "JSON",
                    "byte_sha256": hashlib.sha256(inner_role.encode("utf-8")).hexdigest(),
                },
                created_at="2026-04-02T00:00:00Z",
            )
            sources = [{"role": inner_role, "source_ref": stash(leaf)}]
        source_artifacts[role] = seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": f"factor-deep-evidence-{role}",
                "state": "IMMUTABLE",
                "sources": sources,
            },
            created_at="2026-04-02T00:00:00Z",
        )
        stash(source_artifacts[role])
    policy = build_bootstrap_exception_evidence(
        decision_source_bytes=canonical_json_bytes(_DECISION_DOCUMENT),
        source_artifacts=source_artifacts,
        implementation_source_sha256="d" * 64,
        created_at="2026-04-02T00:00:00Z",
    )
    policy_ref = stash(policy)
    active_set = build_bootstrap_factor_set(
        bootstrap_exception_evidence=policy,
        created_at="2026-04-02T00:00:00Z",
    )
    active_set_ref = stash(active_set)
    attestation = _build_factor_validation_receipt(
        policy=policy,
        active_set=active_set,
        evidence_artifacts=[source_artifacts[role] for role in sorted(source_artifacts)],
        trusted_at="2026-04-02T00:00:00Z",
    )
    attestation_ref = stash(attestation)
    implementation_refs = []
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        implementation = installed_semantic_row(factor_id)
        implementation_refs.append(
            stash(
                seal_installed_component_manifest(
                    component_id=implementation["implementation_id"],
                    component_role="SOURCE_IMPLEMENTATION",
                    package_name="quant_investor.factors.governance",
                    module_names=[implementation["module_name"]],
                    entrypoint_specs=[
                        (implementation["module_name"], implementation["qualified_name"])
                    ],
                    release_manifest_ref=release_ref,
                    allowed_source_formats=["PARQUET"],
                    fallback_allowed=False,
                    created_at="2026-04-02T00:00:00Z",
                )
            )
        )
    implementation_refs.sort(key=lambda row: row["artifact_id"])
    calendar_leaf_payload = artifacts[source_rows[0]["source_ref"]["byte_sha256"]]["payload"]
    calendar_custody_ref = stash(
        _calendar_custody_artifact(
            release_ref,
            [
                {
                    "relative_path": calendar_leaf_payload["relative_path"],
                    "byte_sha256": calendar_leaf_payload["byte_sha256"],
                    "size": len(calendar.read_bytes()),
                }
            ],
            identity="factor-flat-calendar",
        )
    )
    release_install = _release_install_evidence(release_ref, tmp_path)
    release_install_ref = stash(release_install)
    closure = build_factor_production_source_closure(
        deployed_release_ref=release_ref,
        release_install_evidence_ref=release_install_ref,
        release_install_input_source_ref=_structural_ref(
            "system.source_object", "flat-release-install-input"
        ),
        release_install_verification={
            "state": "PASS",
            "release_ref": release_ref,
            "source_archive_sha256": release_install["payload"]["source_archive"]["byte_sha256"],
            "wheel_sha256": release_install["payload"]["wheel"]["byte_sha256"],
            "code_tree_sha256": release_install["payload"]["code_tree_sha256"],
            "installed_code_manifest_sha256": release_install["payload"][
                "installed_code_manifest_sha256"
            ],
            "contract_catalog_sha256": release_install["payload"]["contract_catalog_sha256"],
            "import_origin": release_install["payload"]["import_origin"],
        },
        market_pit_selection_ref=selection_ref,
        market_scope_source_ref=_structural_ref("system.source_object", "flat-market-scope"),
        calendar_authority_policy_ref=calendar_policy_ref,
        calendar_compilation_ref=calendar_compilation_ref,
        calendar_capture_custody_attestation_ref=calendar_custody_ref,
        factor_source_bundle_ref=source_bundle_ref,
        factor_policy_ref=policy_ref,
        factor_active_set_ref=active_set_ref,
        factor_validation_attestation_ref=attestation_ref,
        factor_implementation_refs=implementation_refs,
        legacy_zero_call_ref=legacy_ref,
        market_input_ref=stash(
            _contract_artifact("factor.production_market_input", "structural-market-input")
        ),
        created_at="2026-04-02T00:00:00Z",
    )
    closure_ref = stash(closure)

    def artifact_resolver(ref: dict[str, str]) -> dict[str, object]:
        return artifacts[ref["byte_sha256"]]

    def source_resolver(ref: dict[str, str], maximum_bytes: int) -> tuple[dict[str, object], bytes]:
        source = artifacts[ref["byte_sha256"]]
        payload = source["payload"]
        path = source_paths[
            next(
                role for role, item in source_paths.items() if item.name == payload["relative_path"]
            )
        ]
        raw = path.read_bytes()
        metadata = path.stat()
        return (
            {
                "source_object_ref": ref,
                "source_root_id": payload["source_root_id"],
                "relative_path": payload["relative_path"],
                "media_type": payload["media_type"],
                "source_format": payload["source_format"],
                "byte_sha256": payload["byte_sha256"],
                "size": len(raw),
                "stat_identity": {
                    "st_ctime_ns": metadata.st_ctime_ns,
                    "st_dev": metadata.st_dev,
                    "st_gid": metadata.st_gid,
                    "st_ino": metadata.st_ino,
                    "st_mode": metadata.st_mode,
                    "st_mtime_ns": metadata.st_mtime_ns,
                    "st_nlink": metadata.st_nlink,
                    "st_size": metadata.st_size,
                    "st_uid": metadata.st_uid,
                },
            },
            raw,
        )

    with pytest.raises(FactorGovernanceError, match="calendar capture bundle is absent"):
        production_authority.validate_factor_production_source_closure(
            closure,
            artifact_resolver=artifact_resolver,
            source_resolver=source_resolver,
        )
    assert closure_ref["kind"] == "factor.production_source_closure"


def test_deep_factor_replay_rebuilds_trusted_calendar_and_market_binding(  # noqa: C901
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _published_calendar_case(tmp_path, monkeypatch)
    created_at = case["created_at"]
    as_of = case["market_sessions"][-1].replace("-", "")
    source_root = tmp_path / "canonical-source"
    source_root.mkdir(mode=0o700)
    source_paths: dict[str, Path] = {}
    artifacts: dict[str, dict[str, object]] = {}

    def stash(artifact: dict[str, object]) -> dict[str, str]:
        ref = _ref(artifact)
        artifacts[ref["byte_sha256"]] = artifact
        return ref

    def source_object(
        *,
        relative_path: str,
        raw: bytes,
        source_format: str,
        media_type: str,
        source_object_id: str | None = None,
    ) -> dict[str, str]:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(raw)
        path.chmod(0o600)
        source_paths[relative_path] = path
        return stash(
            seal_artifact(
                "system.source_object",
                {
                    "source_object_id": source_object_id
                    or "factor-source-"
                    + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24],
                    "source_root_id": "factor-native-calendar-test-root",
                    "relative_path": relative_path,
                    "media_type": media_type,
                    "source_format": source_format,
                    "byte_sha256": hashlib.sha256(raw).hexdigest(),
                },
                created_at=created_at,
            )
        )

    def raw_sha(source_ref: dict[str, str]) -> str:
        return str(artifacts[source_ref["byte_sha256"]]["payload"]["byte_sha256"])

    calendar_rows: list[dict[str, object]] = []
    for encoded_ref, raw in case["raw_by_ref"].items():
        file_ref = parse_canonical_json_bytes(encoded_ref, label="calendar fixture ref")
        relative = file_ref["relative_path"]
        if relative.endswith(".parquet"):
            source_format = "PARQUET"
            media_type = "application/vnd.apache.parquet"
        elif relative.endswith(".json"):
            source_format = "JSON"
            media_type = "application/json"
        else:
            source_format = "BINARY"
            media_type = "application/octet-stream"
        calendar_rows.append(
            {
                "role": "raw-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16],
                "source_ref": source_object(
                    relative_path=relative,
                    raw=raw,
                    source_format=source_format,
                    media_type=media_type,
                ),
            }
        )
    calendar_rows.sort(key=lambda row: row["role"])
    calendar_bundle_ref = stash(
        seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": "factor-native-calendar-capture",
                "state": "IMMUTABLE",
                "sources": calendar_rows,
            },
            created_at=created_at,
        )
    )

    sessions = [date.fromisoformat(value) for value in case["market_sessions"][-91:]]
    symbols = ["000001.SZ", "600000.SH", "430001.BJ"]
    pit_path = "pit/membership.parquet"
    pit_raw_path = source_root / pit_path
    pit_raw_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "signal_session": sessions[-1],
                    "symbol": symbol,
                    "industry": "test",
                    "total_mv": float(1_000_000 + index),
                    "tradable": True,
                }
                for index, symbol in enumerate(symbols)
            ],
            schema=role_schema("pit_universe"),
        ),
        pit_raw_path,
    )
    pit_raw_path.chmod(0o600)
    pit_ref = source_object(
        relative_path=pit_path,
        raw=pit_raw_path.read_bytes(),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    factor_pit_path = "pit/factor-pit-universe.parquet"
    factor_pit_ref = source_object(
        relative_path=factor_pit_path,
        raw=pit_raw_path.read_bytes(),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    pit_manifest_raw = canonical_json_bytes(
        {"generation_id": "pit-native-calendar", "canonical_sha256": raw_sha(pit_ref)}
    )
    pit_manifest_ref = source_object(
        relative_path="pit/manifest.json",
        raw=pit_manifest_raw,
        source_format="JSON",
        media_type="application/json",
    )
    bound_pit_raw = canonical_json_bytes(
        {
            "discovery_schema_version": "cn_pit_universe_latest.v1",
            "generation_id": "pit-native-calendar",
            "generation_manifest_sha256": raw_sha(pit_manifest_ref),
            "canonical_sha256": raw_sha(pit_ref),
        }
    )
    bound_pit_ref = source_object(
        relative_path="pit/market-bound.json",
        raw=bound_pit_raw,
        source_format="JSON",
        media_type="application/json",
    )
    observed_pit_raw = canonical_json_bytes(
        {"discovery_schema_version": "cn_pit_universe_latest.v1", "generation_id": "pit-observed"}
    )
    observed_pit_ref = source_object(
        relative_path="pit/observed-current.json",
        raw=observed_pit_raw,
        source_format="JSON",
        media_type="application/json",
    )
    pit_bundle_ref = stash(
        seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": "factor-native-pit-capture",
                "state": "IMMUTABLE",
                "sources": [
                    {"role": "bound-pointer", "source_ref": bound_pit_ref},
                    {"role": "factor-pit-universe", "source_ref": factor_pit_ref},
                    {"role": "membership", "source_ref": pit_ref},
                    {"role": "observed-pointer", "source_ref": observed_pit_ref},
                    {"role": "pit-manifest", "source_ref": pit_manifest_ref},
                ],
            },
            created_at=created_at,
        )
    )

    market_path = "market/history.parquet"
    market_raw_path = source_root / market_path
    market_raw_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    market_rows = [
        {
            "trade_date": session,
            "symbol": symbol,
            "adj_close": 10.0 + (symbol_index + 1) * session_index * 0.01,
            "amount": 1000.0 + (symbol_index + 1) * (session_index + 1),
            "vol": 100.0 + (symbol_index + 1) * (session_index % 19),
        }
        for symbol_index, symbol in enumerate(symbols)
        for session_index, session in enumerate(sessions)
    ]
    pq.write_table(
        pa.Table.from_pylist(market_rows, schema=role_schema("market_history")), market_raw_path
    )
    market_raw_path.chmod(0o600)
    market_history_ref = source_object(
        relative_path=market_path,
        raw=market_raw_path.read_bytes(),
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_scope_raw = canonical_json_bytes(
        {"full_a": sorted(symbols), "stats": {"full_a": len(symbols)}}
    )
    market_scope_ref = source_object(
        relative_path="market/scope.json",
        raw=market_scope_raw,
        source_format="JSON",
        media_type="application/json",
    )
    scope_sha = hashlib.sha256("\n".join(sorted(symbols)).encode("utf-8")).hexdigest()
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "classification_sets_disjoint": True,
        "true_missing_symbols": [],
        "expected_scope_count": len(symbols),
        "coverage_complete_count": len(symbols),
        "non_blocking_absent_symbols": [],
        "latest_complete_trade_date": as_of,
        "coverage_trade_date": as_of,
        "upsert_target_trade_date": as_of,
        "expected_scope_sha256": scope_sha,
        "pit_generation_id": "pit-native-calendar",
        "pit_generation_manifest_sha256": raw_sha(pit_manifest_ref),
        "pit_membership_sha256": raw_sha(pit_ref),
    }
    pointer_document = {
        "status": "OK",
        "blockers": [],
        "snapshot_id": "market-native-calendar",
        "latest_complete_trade_date": as_of,
        "coverage": coverage,
    }
    pointer_raw = canonical_json_bytes(pointer_document)
    pointer_ref = source_object(
        relative_path="market/pointer.json",
        raw=pointer_raw,
        source_format="JSON",
        media_type="application/json",
    )
    manifest_raw = canonical_json_bytes(pointer_document)
    manifest_ref = source_object(
        relative_path="market/manifest.json",
        raw=manifest_raw,
        source_format="JSON",
        media_type="application/json",
    )

    selection = build_market_pit_selection(
        as_of=as_of,
        market_pointer_file_ref={
            "relative_path": "market/pointer.json",
            "byte_sha256": raw_sha(pointer_ref),
        },
        market_snapshot_manifest_file_ref={
            "relative_path": "market/manifest.json",
            "byte_sha256": raw_sha(manifest_ref),
        },
        market_bound_pit_pointer_file_ref={
            "relative_path": "pit/market-bound.json",
            "byte_sha256": raw_sha(bound_pit_ref),
        },
        pit_generation_manifest_file_ref={
            "relative_path": "pit/manifest.json",
            "byte_sha256": raw_sha(pit_manifest_ref),
        },
        pit_membership_file_ref={"relative_path": pit_path, "byte_sha256": raw_sha(pit_ref)},
        observed_current_pit_pointer_file_ref={
            "relative_path": "pit/observed-current.json",
            "byte_sha256": raw_sha(observed_pit_ref),
        },
        market_pointer=pointer_document,
        market_snapshot_manifest=pointer_document,
        market_bound_pit_pointer=parse_canonical_json_bytes(bound_pit_raw),
        pit_generation_manifest=parse_canonical_json_bytes(pit_manifest_raw),
        observed_current_pit_pointer=parse_canonical_json_bytes(observed_pit_raw),
        created_at=created_at,
    )
    selection_ref = stash(selection)
    release = case["release_install_input"]["deployed_release"]
    release_ref = stash(release)
    assert release_ref == case["release_ref"]
    release_install = case["release_install_input"]["release_install_evidence"]
    release_install_ref = stash(release_install)
    stash(case["capture"]["capture_transaction"])
    stash(case["capture"]["capture_execution"])
    stash(case["capture"]["capture_success"])
    stash(case["capability"])
    for capture in case["captures"]:
        stash(capture)
    policy_ref = stash(case["policy"])
    compilation = build_trusted_provider_calendar_compilation(
        compilation_id="factor-native-calendar-compilation",
        policy=case["policy"],
        capability=case["capability"],
        capture_documents=case["captures"],
        docs_raw=case["docs"],
        raw_resolver=_trusted_raw_resolver(case),
        release_ref=release_ref,
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
        market_session_dates=case["market_sessions"][-91:],
        cutoff_date=case["market_sessions"][-1],
        calendar_json_file_ref=case["json_ref"],
        calendar_parquet_file_ref=case["parquet_ref"],
        created_at=created_at,
    )
    compilation_ref = stash(compilation)
    market_input = build_factor_production_market_input(
        market_pit_selection=selection,
        market_pointer_source=artifacts[pointer_ref["byte_sha256"]],
        market_snapshot_manifest_source=artifacts[manifest_ref["byte_sha256"]],
        market_scope_source=artifacts[market_scope_ref["byte_sha256"]],
        market_history_source=artifacts[market_history_ref["byte_sha256"]],
        market_pointer_raw=pointer_raw,
        market_snapshot_manifest_raw=manifest_raw,
        created_at=created_at,
    )
    market_input_ref = stash(market_input)
    market_bundle_ref = stash(
        seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": "factor-native-market-capture",
                "state": "IMMUTABLE",
                "sources": [
                    {"role": "factor-market-history", "source_ref": market_history_ref},
                    {"role": "market-scope", "source_ref": market_scope_ref},
                ],
            },
            created_at=created_at,
        )
    )
    source_bundle_ref = stash(
        seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": "factor-native-source-closure",
                "state": "IMMUTABLE",
                "sources": [
                    {"role": "exchange_calendar", "source_ref": calendar_bundle_ref},
                    {"role": "market_history", "source_ref": market_bundle_ref},
                    {"role": "pit_universe", "source_ref": pit_bundle_ref},
                ],
            },
            created_at=created_at,
        )
    )
    recomputation = recompute_factor_production_signals(
        exchange_calendar_path=source_paths[case["parquet_ref"]["relative_path"]],
        pit_universe_path=source_paths[factor_pit_path],
        market_history_path=source_paths[market_path],
        exchange_calendar_sha256=case["parquet_ref"]["byte_sha256"],
        pit_universe_sha256=raw_sha(factor_pit_ref),
        market_history_sha256=raw_sha(market_history_ref),
        as_of=as_of,
    )
    implementation_refs = []
    implementation_component_refs: dict[str, dict[str, str]] = {}
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        implementation = installed_semantic_row(factor_id)
        component_ref = stash(
            seal_installed_component_manifest(
                component_id=implementation["implementation_id"],
                component_role="SOURCE_IMPLEMENTATION",
                package_name="quant_investor.factors.governance",
                module_names=[implementation["module_name"]],
                entrypoint_specs=[
                    (implementation["module_name"], implementation["qualified_name"])
                ],
                release_manifest_ref=release_ref,
                allowed_source_formats=["PARQUET"],
                fallback_allowed=False,
                created_at=created_at,
            )
        )
        implementation_refs.append(component_ref)
        implementation_component_refs[factor_id] = component_ref
    implementation_refs.sort(key=lambda row: row["artifact_id"])
    implementation_rows = installed_implementation_rows(
        implementation_component_refs=implementation_component_refs
    )
    implementation_raw = canonical_json_bytes(
        {
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": implementation_rows,
        }
    )
    decision_raw = canonical_json_bytes(_DECISION_DOCUMENT)
    decision_source_ref = source_object(
        relative_path="operations/unified_cutover/bootstrap-decision.json",
        raw=decision_raw,
        source_format="JSON",
        media_type="application/json",
        source_object_id="factor-bootstrap-decision",
    )
    implementation_source_ref = source_object(
        relative_path="bootstrap/implementation-tree.json",
        raw=implementation_raw,
        source_format="JSON",
        media_type="application/json",
        source_object_id="factor-bootstrap-implementation",
    )
    recomputation_source_ref = source_object(
        relative_path="bootstrap/recomputation.json",
        raw=canonical_json_bytes(
            {
                "authority": "NON_AUTHORIZING",
                "domain": "myquant-bootstrap-recomputation",
                "result": "EXACT_MATCH",
                "recomputation": recomputation,
                "source_sha256s": {
                    "exchange_calendar": case["parquet_ref"]["byte_sha256"],
                    "market_history": raw_sha(market_history_ref),
                    "pit_universe": raw_sha(factor_pit_ref),
                },
            }
        ),
        source_format="JSON",
        media_type="application/json",
        source_object_id="factor-bootstrap-recomputation",
    )
    runtime_calendar_ref = next(
        row["source_ref"]
        for row in calendar_rows
        if artifacts[row["source_ref"]["byte_sha256"]]["payload"]["relative_path"]
        == case["parquet_ref"]["relative_path"]
    )
    source_generation_rows = sorted(
        [
            {
                "role": role,
                "source_ref": source_ref,
                "source_byte_sha256": raw_sha(source_ref),
            }
            for role, source_ref in (
                ("exchange_calendar", runtime_calendar_ref),
                ("market", market_history_ref),
                ("pit_universe", factor_pit_ref),
            )
        ],
        key=lambda row: row["role"],
    )
    source_generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": {
            "reader": "MarketDataReader",
            "market": "CN",
            "mode_policy": "strict",
            "source_format": "PARQUET",
            "fallback_allowed": False,
        },
        "source_rows": source_generation_rows,
    }
    source_generation_ref = source_object(
        relative_path="bootstrap/source-generation.json",
        raw=canonical_json_bytes(
            {
                **source_generation_body,
                "generation_sha256": hashlib.sha256(
                    canonical_json_bytes(source_generation_body)
                ).hexdigest(),
            }
        ),
        source_format="JSON",
        media_type="application/json",
        source_object_id="factor-bootstrap-source-generation",
    )

    def bootstrap_bundle(
        role: str,
        inner_role: str,
        source_ref: dict[str, str],
    ) -> dict[str, object]:
        bundle = seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": f"factor-native-bootstrap-{role}",
                "state": "IMMUTABLE",
                "sources": [{"role": inner_role, "source_ref": source_ref}],
            },
            created_at=created_at,
        )
        stash(bundle)
        return bundle

    source_artifacts: dict[str, dict[str, object]] = {
        "code": release,
        "decision_source": bootstrap_bundle("decision", "bootstrap_decision", decision_source_ref),
        "exchange_calendar": bootstrap_bundle("calendar", "calendar", runtime_calendar_ref),
        "implementation": bootstrap_bundle(
            "implementation", "implementation_tree_manifest", implementation_source_ref
        ),
        "market": bootstrap_bundle("market", "market", market_history_ref),
        "pit_universe": bootstrap_bundle("pit", "pit", factor_pit_ref),
        "recomputation": bootstrap_bundle(
            "recomputation", "recomputation", recomputation_source_ref
        ),
        "source_generation": bootstrap_bundle(
            "source-generation", "source_generation", source_generation_ref
        ),
    }
    policy = build_bootstrap_exception_evidence(
        decision_source_bytes=decision_raw,
        source_artifacts=source_artifacts,
        implementation_source_sha256=raw_sha(implementation_source_ref),
        created_at=created_at,
    )
    policy_closure_ref = stash(policy)
    active_set = build_bootstrap_factor_set(
        bootstrap_exception_evidence=policy,
        created_at=created_at,
    )
    active_set_ref = stash(active_set)
    attestation = _build_factor_validation_receipt(
        policy=policy,
        active_set=active_set,
        evidence_artifacts=[source_artifacts[role] for role in sorted(source_artifacts)],
        trusted_at=created_at,
    )
    attestation_ref = stash(attestation)

    repository_root = Path(
        case["capture"]["capture_execution"]["payload"]["release_repository_root"]
    )
    legacy_runner = _fixed_git_runner(
        final_commit=release_install["payload"]["final_commit"],
        final_tree=release_install["payload"]["final_tree"],
        extra_files={
            "operations/unified_cutover/bootstrap-decision.json": canonical_json_bytes(
                _DECISION_DOCUMENT
            )
        },
    )
    monkeypatch.setattr(legacy_scanner_module.subprocess, "run", legacy_runner)
    legacy = build_factor_legacy_zero_call_certificate_for_release(
        repository_root=repository_root,
        final_commit=release_install["payload"]["final_commit"],
        final_tree=release_install["payload"]["final_tree"],
        resolver_inventory_ref=release_ref,
        verified_at=created_at,
    )
    legacy_ref = stash(legacy)
    calendar_custody_ref = stash(
        build_factor_calendar_capture_custody_attestation(
            capture_parent=case["capture_root"].parent,
            capture_execution=case["capture"]["capture_execution"],
            capture_execution_file_ref=case["capture"]["capture_execution_file_ref"],
            capture_success=case["capture"]["capture_success"],
            capture_success_file_ref=case["capture"]["capture_success_file_ref"],
            deployed_release_ref=release_ref,
            verified_at=created_at,
        )
    )
    release_install_input_source_ref = next(
        row["source_ref"]
        for row in calendar_rows
        if str(artifacts[row["source_ref"]["byte_sha256"]]["payload"]["relative_path"]).endswith(
            "release-install-input.json"
        )
    )
    closure = build_factor_production_source_closure(
        deployed_release_ref=release_ref,
        release_install_evidence_ref=release_install_ref,
        release_install_input_source_ref=release_install_input_source_ref,
        release_install_verification={
            "state": "PASS",
            "release_ref": release_ref,
            "source_archive_sha256": release_install["payload"]["source_archive"]["byte_sha256"],
            "wheel_sha256": release_install["payload"]["wheel"]["byte_sha256"],
            "code_tree_sha256": release_install["payload"]["code_tree_sha256"],
            "installed_code_manifest_sha256": release_install["payload"][
                "installed_code_manifest_sha256"
            ],
            "contract_catalog_sha256": release_install["payload"]["contract_catalog_sha256"],
            "import_origin": release_install["payload"]["import_origin"],
        },
        market_pit_selection_ref=selection_ref,
        market_scope_source_ref=market_scope_ref,
        calendar_authority_policy_ref=policy_ref,
        calendar_compilation_ref=compilation_ref,
        calendar_capture_custody_attestation_ref=calendar_custody_ref,
        factor_source_bundle_ref=source_bundle_ref,
        factor_policy_ref=policy_closure_ref,
        factor_active_set_ref=active_set_ref,
        factor_validation_attestation_ref=attestation_ref,
        factor_implementation_refs=implementation_refs,
        legacy_zero_call_ref=legacy_ref,
        market_input_ref=market_input_ref,
        created_at=created_at,
    )
    closure_ref = stash(closure)

    def artifact_resolver(ref: dict[str, str]) -> dict[str, object]:
        return artifacts[ref["byte_sha256"]]

    def source_resolver(ref: dict[str, str], maximum_bytes: int) -> tuple[dict[str, object], bytes]:
        source = artifacts[ref["byte_sha256"]]
        payload = source["payload"]
        raw = source_paths[payload["relative_path"]].read_bytes()
        metadata = source_paths[payload["relative_path"]].stat()
        return (
            {
                "source_object_ref": ref,
                "source_root_id": payload["source_root_id"],
                "relative_path": payload["relative_path"],
                "media_type": payload["media_type"],
                "source_format": payload["source_format"],
                "byte_sha256": payload["byte_sha256"],
                "size": len(raw),
                "stat_identity": {
                    "st_ctime_ns": metadata.st_ctime_ns,
                    "st_dev": metadata.st_dev,
                    "st_gid": metadata.st_gid,
                    "st_ino": metadata.st_ino,
                    "st_mode": metadata.st_mode,
                    "st_mtime_ns": metadata.st_mtime_ns,
                    "st_nlink": metadata.st_nlink,
                    "st_size": metadata.st_size,
                    "st_uid": metadata.st_uid,
                },
            },
            raw,
        )

    evidence = build_factor_production_recomputation_evidence(
        source_closure=closure,
        deployed_release_ref=release_ref,
        factor_active_set_ref=active_set_ref,
        recomputation=recomputation,
        created_at=created_at,
    )
    stash(evidence)
    generation = build_factor_production_generation(
        source_closure=closure,
        recomputation_evidence=evidence,
        created_at=created_at,
    )
    stash(generation)
    production_authority.validate_factor_production_source_closure(
        closure,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
    )
    replay_factor_production_recomputation_evidence(
        evidence,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
    )
    replay_factor_production_generation(
        generation,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        validation_mode="HISTORICAL_RECOVERY",
    )
    custody_workspace = tmp_path / "native-calendar-custody"
    custody_workspace.mkdir(mode=0o700)
    custody = SystemStore(
        custody_workspace,
        source_root=source_root,
        source_root_id="factor-native-calendar-test-root",
    )
    for stored_artifact in artifacts.values():
        custody.put_object(stored_artifact)
    custody_source_resolver = system_store_source_resolver(custody)
    production_authority.validate_factor_production_source_closure(
        closure,
        artifact_resolver=custody.get_object,
        source_resolver=custody_source_resolver,
    )
    replay_factor_production_recomputation_evidence(
        evidence,
        artifact_resolver=custody.get_object,
        source_resolver=custody_source_resolver,
    )
    replay_factor_production_generation(
        generation,
        artifact_resolver=custody.get_object,
        source_resolver=custody_source_resolver,
        validation_mode="HISTORICAL_RECOVERY",
    )
    assert closure_ref["kind"] == "factor.production_source_closure"

    def tampered_calendar(
        ref: dict[str, str], maximum_bytes: int
    ) -> tuple[dict[str, object], bytes]:
        descriptor, raw = source_resolver(ref, maximum_bytes)
        if descriptor["relative_path"].endswith("response-sse.raw"):
            return descriptor, raw + b"x"
        return descriptor, raw

    with pytest.raises(FactorGovernanceError, match="source raw SHA"):
        production_authority.validate_factor_production_source_closure(
            closure,
            artifact_resolver=artifact_resolver,
            source_resolver=tampered_calendar,
        )

    def tampered_market(ref: dict[str, str], maximum_bytes: int) -> tuple[dict[str, object], bytes]:
        descriptor, raw = source_resolver(ref, maximum_bytes)
        if descriptor["relative_path"] == market_path:
            return descriptor, raw + b"x"
        return descriptor, raw

    with pytest.raises(FactorGovernanceError, match="source raw SHA"):
        production_authority.validate_factor_production_source_closure(
            closure,
            artifact_resolver=artifact_resolver,
            source_resolver=tampered_market,
        )


def test_calendar_capture_root_tamper_blocks_factor_custody_ingress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _published_calendar_case(tmp_path, monkeypatch)
    capture_root = case["capture_root"]
    target = capture_root / "response-sse.raw"
    original = target.read_bytes()
    target.chmod(0o600)
    target.write_bytes(original + b"x")
    target.chmod(0o600)
    with pytest.raises(Exception, match="hash|bytes|authority|capture"):
        build_factor_calendar_capture_custody_attestation(
            capture_parent=capture_root.parent,
            capture_execution=case["capture"]["capture_execution"],
            capture_execution_file_ref=case["capture"]["capture_execution_file_ref"],
            capture_success=case["capture"]["capture_success"],
            capture_success_file_ref=case["capture"]["capture_success_file_ref"],
            deployed_release_ref=case["release_ref"],
            verified_at=case["created_at"],
        )


def _factor_production_operator_fixture(  # noqa: C901
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    case = _published_calendar_case(tmp_path, monkeypatch)
    workspace = case["workspace"]
    release = case["release_install_input"]["deployed_release"]
    release_ref = _ref(release)
    release_install = case["release_install_input"]["release_install_evidence"]
    sessions = [date.fromisoformat(value) for value in case["market_sessions"][-91:]]
    cutoff = sessions[-1].strftime("%Y%m%d")
    symbols = ["000001.SZ", "000002.SZ", "430001.BJ", "600000.SH"]
    scope_sha = hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest()
    market_root = tmp_path / "strict-market"
    market_scope_path = market_root / "cn_universe/cn_index_components.json"
    market_scope_path.parent.mkdir(parents=True, mode=0o700)
    market_scope_raw = canonical_json_bytes({"full_a": symbols, "stats": {"full_a": len(symbols)}})
    market_scope_path.write_bytes(market_scope_raw)
    market_scope_path.chmod(0o644)
    reference_root = market_root / "parquet/cn/reference"
    generation_id = "pit-factor-prepare"
    generation_root = reference_root / "_generations" / generation_id
    generation_root.mkdir(parents=True, mode=0o700)
    pit_path = generation_root / "stock_basic_membership.parquet"
    records = {
        symbol: PITUniverseRecord(
            symbol=symbol,
            industry="test",
            source_list_status="L",
            list_date="20200101",
            observed_at="2026-08-07T00:00:00Z",
            source_run_id="factor-prepare-pit",
        )
        for symbol in symbols
    }
    pq.write_table(
        pa.Table.from_pylist([record.to_dict() for record in records.values()]), pit_path
    )
    pit_path.chmod(0o600)
    pit_sha = hashlib.sha256(pit_path.read_bytes()).hexdigest()
    pit_manifest_path = generation_root / "manifest.json"
    pit_manifest = {
        "generation_id": generation_id,
        "canonical_path": str(pit_path),
        "canonical_sha256": pit_sha,
    }
    pit_manifest_path.write_bytes(canonical_json_bytes(pit_manifest))
    pit_manifest_path.chmod(0o600)
    pit_manifest_sha = hashlib.sha256(pit_manifest_path.read_bytes()).hexdigest()
    reference_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    pit_pointer_path = reference_root / "stock_basic_membership_latest.json"
    pit_pointer = {
        "discovery_schema_version": "cn_pit_universe_latest.v1",
        "generation_id": generation_id,
        "generation_manifest_path": str(pit_manifest_path),
        "generation_manifest_sha256": pit_manifest_sha,
        "canonical_path": str(pit_path),
        "canonical_sha256": pit_sha,
    }
    pit_pointer_path.write_bytes(canonical_json_bytes(pit_pointer))
    pit_pointer_path.chmod(0o600)
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "classification_sets_disjoint": True,
        "true_missing_symbols": [],
        "expected_scope_count": len(symbols),
        "coverage_complete_count": len(symbols),
        "non_blocking_absent_symbols": [],
        "latest_complete_trade_date": cutoff,
        "coverage_trade_date": cutoff,
        "upsert_target_trade_date": cutoff,
        "expected_scope_sha256": scope_sha,
        "pit_generation_id": generation_id,
        "pit_generation_manifest_path": str(pit_manifest_path),
        "pit_generation_manifest_sha256": pit_manifest_sha,
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": pit_sha,
    }
    market_document = {
        "status": "OK",
        "blockers": [],
        "snapshot_id": "factor-prepare-market",
        "latest_complete_trade_date": cutoff,
        "coverage": coverage,
    }
    market_pointer_path = market_root / "parquet/cn/_latest.json"
    market_manifest_path = market_root / "parquet/cn/_snapshots/factor-prepare-market.json"
    market_pointer_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    market_manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in (market_pointer_path, market_manifest_path):
        path.write_bytes(canonical_json_bytes(market_document))
        path.chmod(0o644)
    frames = {
        symbol: SimpleNamespace(
            frame=pd.DataFrame(
                [
                    {
                        "trade_date": session,
                        "symbol": symbol,
                        "ts_code": symbol,
                        "adj_close": 10.0 + symbol_index + ordinal * 0.01,
                        "amount": 1000.0 + (symbol_index + 1) * (ordinal + 1),
                        "vol": 100.0 + symbol_index + ordinal / 10.0,
                        "total_mv": 1_000_000.0 + symbol_index,
                    }
                    for ordinal, session in enumerate(sessions)
                ]
            )
        )
        for symbol_index, symbol in enumerate(symbols)
    }

    class FakeReader:
        def __init__(self, **_: object) -> None:
            pass

        def clean_snapshot_gate(self, *, refresh: bool = False) -> dict[str, object]:
            del refresh
            return {
                "healthy": True,
                "status": "ok",
                "latest_complete_trade_date": cutoff,
                "latest_pointer_path": str(market_pointer_path),
                "manifest_path": str(market_manifest_path),
            }

        def coverage_bound_pit(self, *, refresh: bool = False) -> dict[str, object]:
            del refresh
            return {
                "status": "passed",
                "generation_id": generation_id,
                "generation_manifest_path": str(pit_manifest_path),
                "generation_manifest_sha256": pit_manifest_sha,
                "canonical_path": str(pit_path),
                "canonical_sha256": pit_sha,
                "records": records,
            }

        def list_symbols(self, universe_key: str) -> list[str]:
            assert universe_key == "full_a"
            return list(symbols)

        def read_symbol_frames(self, requested: list[str], **_: object) -> dict[str, object]:
            return {symbol: frames[symbol] for symbol in requested}

    monkeypatch.setattr(prepare_module, "MarketDataReader", FakeReader)
    repository_root = Path(
        case["capture"]["capture_execution"]["payload"]["release_repository_root"]
    )
    legacy_runner = _fixed_git_runner(
        final_commit=release_install["payload"]["final_commit"],
        final_tree=release_install["payload"]["final_tree"],
        extra_files={
            "operations/unified_cutover/bootstrap-decision.json": canonical_json_bytes(
                _DECISION_DOCUMENT
            )
        },
    )
    release_verification = {
        "state": "PASS",
        "release_ref": release_ref,
        "source_archive_sha256": release_install["payload"]["source_archive"]["byte_sha256"],
        "wheel_sha256": release_install["payload"]["wheel"]["byte_sha256"],
        "code_tree_sha256": release_install["payload"]["code_tree_sha256"],
        "installed_code_manifest_sha256": release_install["payload"][
            "installed_code_manifest_sha256"
        ],
        "contract_catalog_sha256": release_install["payload"]["contract_catalog_sha256"],
        "import_origin": release_install["payload"]["import_origin"],
    }
    verified_inputs: list[bytes] = []

    def verify_release(raw: bytes, *, repository_root: Path) -> dict[str, object]:
        assert Path(repository_root) == Path(
            case["capture"]["capture_execution"]["payload"]["release_repository_root"]
        )
        assert raw == case["release_install_input_raw"]
        verified_inputs.append(raw)
        return dict(release_verification)

    monkeypatch.setattr(prepare_module, "verify_running_release_install_input", verify_release)
    monkeypatch.setattr(
        production_authority,
        "verify_running_release_install_input",
        verify_release,
    )
    monkeypatch.setattr(legacy_scanner_module.subprocess, "run", legacy_runner)
    from quant_investor.system import store as system_store_module

    monkeypatch.setattr(system_store_module, "_verify_installed_release", lambda _release: None)
    decision_path = repository_root / "operations/unified_cutover/bootstrap-decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_bytes(canonical_json_bytes(_DECISION_DOCUMENT))
    decision_path.chmod(0o600)

    arguments = {
        "workspace_root": workspace,
        "market_data_root": market_root,
        "calendar_capture_root": case["capture_root"],
        "expected_calendar_success_sha256": hashlib.sha256(
            (case["capture_root"] / "capture-success.json").read_bytes()
        ).hexdigest(),
    }
    factor_pointer = workspace / "results/factors/_active.json"
    factor_marker = workspace / "results/factors/_production_complete.json"
    system_pointer = workspace / "results/system/_active.json"
    capture_target = case["capture_root"] / "response-sse.raw"
    capture_original = capture_target.read_bytes()
    capture_target.write_bytes(capture_original + b"x")
    capture_target.chmod(0o600)
    with pytest.raises(Exception, match="hash|bytes|capture|authority"):
        prepare_factor_production(**arguments)
    capture_target.write_bytes(capture_original)
    capture_target.chmod(0o600)
    assert (
        not factor_pointer.exists() and not factor_marker.exists() and not system_pointer.exists()
    )
    market_scope_path.write_bytes(
        canonical_json_bytes(
            {
                "full_a": sorted([*symbols, "600001.SH"], key=lambda value: value.encode("utf-8")),
                "stats": {"full_a": len(symbols) + 1},
            }
        )
    )
    market_scope_path.chmod(0o644)
    with pytest.raises(FactorGovernanceError, match="scope SHA|outside Market-bound PIT"):
        prepare_factor_production(**arguments)
    market_scope_path.write_bytes(market_scope_raw)
    market_scope_path.chmod(0o644)
    assert (
        not factor_pointer.exists() and not factor_marker.exists() and not system_pointer.exists()
    )

    original_rename = prepare_module._rename_no_replace
    monkeypatch.setattr(
        prepare_module,
        "_rename_no_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected publication stop")),
    )
    with pytest.raises(FactorGovernanceError, match="atomic publication failed"):
        prepare_factor_production(**arguments)
    preparations = workspace / "results/factors/preparations"
    assert not [path for path in preparations.iterdir() if not path.name.startswith(".")]
    assert (
        not factor_pointer.exists() and not factor_marker.exists() and not system_pointer.exists()
    )
    monkeypatch.setattr(prepare_module, "_rename_no_replace", original_rename)

    market_pointer_path.chmod(0o664)
    with pytest.raises(FactorGovernanceError, match="bounded regular file"):
        prepare_factor_production(**arguments)
    market_pointer_path.chmod(0o644)
    assert (
        not factor_pointer.exists() and not factor_marker.exists() and not system_pointer.exists()
    )

    fundamental_pointer = workspace / "data/fundamental/_latest.json"
    fundamental_pointer.parent.mkdir(parents=True, mode=0o700)
    fundamental_pointer.write_bytes(b"deliberately-invalid-and-stale")
    fundamental_pointer.chmod(0o600)
    fundamental_before = fundamental_pointer.read_bytes()
    first = prepare_factor_production(**arguments)
    second = prepare_factor_production(**arguments)
    assert first == second
    assert fundamental_pointer.read_bytes() == fundamental_before
    for governed_directory in (
        workspace / "results",
        workspace / "results/factors",
        workspace / "results/factors/preparations",
        workspace / "results/factors/preparations" / first["operation_id"],
        workspace / "results/factors/preparations" / first["operation_id"] / "sources",
    ):
        assert stat.S_IMODE(governed_directory.stat().st_mode) == 0o700
    assert len(verified_inputs) == 3
    assert first["status"] == "PREPARED"
    assert first["factor_pointer_writes"] == 0
    source_root_id = hashlib.sha256(
        canonical_json_bytes({"domain": "factor-production-prepare", "id": first["operation_id"]})
    ).hexdigest()
    prepared_store = SystemStore(
        workspace,
        source_root=(
            workspace / "results/factors/preparations" / first["operation_id"] / "sources"
        ),
        source_root_id=source_root_id,
    )
    generation = prepared_store.get_object(first["factor_production_generation_ref"])
    before_current_gate = len(verified_inputs)
    replay_factor_production_generation(
        generation,
        artifact_resolver=prepared_store.get_object,
        source_resolver=system_store_source_resolver(prepared_store),
        validation_mode="PRE_CAS_CURRENT",
        current_release_root=repository_root,
    )
    assert len(verified_inputs) == before_current_gate + 1
    replay_factor_production_generation(
        generation,
        artifact_resolver=prepared_store.get_object,
        source_resolver=system_store_source_resolver(prepared_store),
        validation_mode="HISTORICAL_RECOVERY",
    )
    assert len(verified_inputs) == before_current_gate + 1
    receipt_path = (
        workspace / "results/factors/preparations" / first["operation_id"] / "prepared.json"
    )
    original_receipt_raw = receipt_path.read_bytes()
    original_receipt = parse_canonical_json_bytes(original_receipt_raw)
    tamper_rows = [
        ("as_of", "19990101"),
        ("market_snapshot_id", "tampered-snapshot"),
        ("market_pit_selection_ref", release_ref),
        ("low_signal_sha256", "f" * 64),
        ("system_pointer_writes", 1),
        ("factor_production_generation_ref", release_ref),
        (
            "operation_inputs_ref",
            {"relative_path": "operation-inputs.json", "byte_sha256": "f" * 64},
        ),
    ]
    for field, value in tamper_rows:
        tampered_receipt = dict(original_receipt)
        tampered_receipt[field] = value
        receipt_path.write_bytes(canonical_json_bytes(tampered_receipt))
        receipt_path.chmod(0o600)
        with pytest.raises(FactorGovernanceError):
            prepare_factor_production(**arguments)
        receipt_path.write_bytes(original_receipt_raw)
        receipt_path.chmod(0o600)
    market_manifest_path.write_text(
        json.dumps(market_document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    market_manifest_path.chmod(0o600)
    changed = prepare_factor_production(**arguments)
    assert changed["operation_inputs_sha256"] != first["operation_inputs_sha256"]
    assert changed["operation_id"] != first["operation_id"]
    assert (
        not factor_pointer.exists() and not factor_marker.exists() and not system_pointer.exists()
    )
    return {
        "workspace": workspace,
        "market_root": market_root,
        "capture_root": case["capture_root"],
        "calendar_success_sha256": arguments["expected_calendar_success_sha256"],
        "release_repository_root": repository_root,
        "prepare_result": first,
        "public_prepare_inputs": {
            "calendar_capture_root": str(case["capture_root"]),
            "expected_calendar_success_sha256": arguments["expected_calendar_success_sha256"],
        },
        "activation_inputs": {
            "workspace_root": str(workspace),
            "source_root": first["source_root"],
            "source_root_id": first["source_root_id"],
            "current_release_root": str(repository_root),
            "factor_generation_ref": first["factor_production_generation_ref"],
            "source_closure_ref": first["factor_production_source_closure_ref"],
            "recomputation_evidence_ref": first["factor_production_recomputation_ref"],
            "legacy_zero_call_ref": first["legacy_zero_call_ref"],
            "deployed_release_ref": first["deployed_release_ref"],
        },
    }


def test_prepare_factor_production_is_offline_retry_safe_and_deep_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _factor_production_operator_fixture(tmp_path, monkeypatch)
    assert fixture["prepare_result"]["status"] == "PREPARED"


def test_deep_bootstrap_receipt_rejects_cross_root_source_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally valid receipt cannot mix an outside source root."""

    fixture = _factor_production_operator_fixture(tmp_path, monkeypatch)
    prepared = fixture["prepare_result"]
    workspace = fixture["workspace"]
    store = SystemStore(
        workspace,
        source_root=workspace / prepared["source_root"],
        source_root_id=prepared["source_root_id"],
    )
    closure = store.get_object(prepared["factor_production_source_closure_ref"])
    closure_payload = closure["payload"]
    policy = store.get_object(closure_payload["factor_policy_ref"])
    source_refs = {row["role"]: row["ref"] for row in policy["payload"]["source_refs"]}

    foreign_root = tmp_path / "foreign-bootstrap-source-root"
    foreign_root.mkdir(mode=0o700)
    foreign_decision = foreign_root / "bootstrap" / "bootstrap-decision.json"
    foreign_decision.parent.mkdir(mode=0o700)
    foreign_decision.write_bytes(canonical_json_bytes(_DECISION_DOCUMENT))
    foreign_decision.chmod(0o600)
    foreign_store = SystemStore(
        workspace,
        source_root=foreign_root,
        source_root_id="foreign-bootstrap-source-root",
    )
    foreign_decision_ref = foreign_store.put_source_file(
        "bootstrap/bootstrap-decision.json",
        source_object_id="foreign-bootstrap-decision",
        source_format="JSON",
        media_type="application/json",
        created_at=policy["created_at"],
    )
    foreign_bundle = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": "foreign-bootstrap-decision-bundle",
            "state": "IMMUTABLE",
            "sources": [{"role": "bootstrap_decision", "source_ref": foreign_decision_ref}],
        },
        created_at=policy["created_at"],
    )
    foreign_bundle_ref = store.put_object(foreign_bundle)

    source_artifacts = {
        role: store.get_object(reference) for role, reference in source_refs.items()
    }
    source_artifacts["decision_source"] = store.get_object(foreign_bundle_ref)
    replacement_policy = build_bootstrap_exception_evidence(
        decision_source_bytes=canonical_json_bytes(_DECISION_DOCUMENT),
        source_artifacts=source_artifacts,
        implementation_source_sha256=policy["payload"]["factor_rows"][0]["implementation_sha256"],
        created_at=policy["created_at"],
    )
    replacement_policy_ref = store.put_object(replacement_policy)
    replacement_active = build_bootstrap_factor_set(
        bootstrap_exception_evidence=replacement_policy,
        created_at=policy["created_at"],
    )
    replacement_active_ref = store.put_object(replacement_active)
    replacement_receipt = _build_factor_validation_receipt(
        policy=replacement_policy,
        active_set=replacement_active,
        evidence_artifacts=[source_artifacts[role] for role in sorted(source_artifacts)],
        trusted_at=policy["created_at"],
    )
    replacement_receipt_ref = store.put_object(replacement_receipt)
    replacement_closure = build_factor_production_source_closure(
        deployed_release_ref=closure_payload["deployed_release_ref"],
        release_install_evidence_ref=closure_payload["release_install_evidence_ref"],
        release_install_input_source_ref=closure_payload["release_install_input_source_ref"],
        release_install_verification=closure_payload["release_install_verification"],
        market_pit_selection_ref=closure_payload["market_pit_selection_ref"],
        market_scope_source_ref=closure_payload["market_scope_source_ref"],
        calendar_authority_policy_ref=closure_payload["calendar_authority_policy_ref"],
        calendar_compilation_ref=closure_payload["calendar_compilation_ref"],
        calendar_capture_custody_attestation_ref=closure_payload[
            "calendar_capture_custody_attestation_ref"
        ],
        factor_source_bundle_ref=closure_payload["factor_source_bundle_ref"],
        factor_policy_ref=replacement_policy_ref,
        factor_active_set_ref=replacement_active_ref,
        factor_validation_attestation_ref=replacement_receipt_ref,
        factor_implementation_refs=closure_payload["factor_implementation_refs"],
        legacy_zero_call_ref=closure_payload["legacy_zero_call_ref"],
        market_input_ref=closure_payload["market_input_ref"],
        created_at=closure["created_at"],
    )

    primary_source_resolver = system_store_source_resolver(store)
    foreign_source_resolver = system_store_source_resolver(foreign_store)

    def source_resolver(reference: dict[str, str], maximum_bytes: int):
        artifact = store.get_object(reference)
        if artifact["payload"].get("source_root_id") == foreign_store.source_root_id:
            return foreign_source_resolver(reference, maximum_bytes)
        return primary_source_resolver(reference, maximum_bytes)

    with pytest.raises(FactorGovernanceError, match="span multiple source roots"):
        production_authority.validate_factor_production_source_closure(
            replacement_closure,
            artifact_resolver=store.get_object,
            source_resolver=source_resolver,
        )


def test_deep_bootstrap_receipt_rejects_adversarial_1_plus_7_matrix(  # noqa: C901
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every alternate Bootstrap identity, not only a foreign root."""

    fixture = _factor_production_operator_fixture(tmp_path, monkeypatch)
    prepared = fixture["prepare_result"]
    workspace = fixture["workspace"]
    source_root = workspace / prepared["source_root"]
    store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id=prepared["source_root_id"],
    )
    closure = store.get_object(prepared["factor_production_source_closure_ref"])
    closure_payload = closure["payload"]
    policy = store.get_object(closure_payload["factor_policy_ref"])
    receipt = store.get_object(closure_payload["factor_validation_attestation_ref"])
    created_at = policy["created_at"]
    source_refs = {row["role"]: row["ref"] for row in policy["payload"]["source_refs"]}

    def bundle_source_ref(role: str) -> dict[str, str]:
        bundle = store.get_object(source_refs[role])
        return dict(bundle["payload"]["sources"][0]["source_ref"])

    def source_bytes(reference: dict[str, str]) -> tuple[dict[str, object], bytes]:
        return store.read_source_object_bytes(reference, maximum_bytes=8 * 1024 * 1024)

    def stage_alternate(
        name: str,
        reference: dict[str, str],
        *,
        raw: bytes | None = None,
    ) -> dict[str, str]:
        payload, observed = source_bytes(reference)
        relative = str(payload["relative_path"])
        suffix = Path(relative).suffix
        target = f"adversarial/{name}{suffix}"
        path = source_root / target
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(observed if raw is None else raw)
        path.chmod(0o600)
        return store.put_source_file(
            target,
            source_object_id=f"adversarial-bootstrap-{name}",
            source_format=str(payload["source_format"]),
            media_type=str(payload["media_type"]),
            created_at=created_at,
        )

    def bundle(
        name: str,
        *,
        inner_role: str,
        source_ref: dict[str, str],
    ) -> dict[str, object]:
        document = seal_artifact(
            "system.source_bundle",
            {
                "source_bundle_id": f"adversarial-bootstrap-{name}",
                "state": "IMMUTABLE",
                "sources": [{"role": inner_role, "source_ref": source_ref}],
            },
            created_at=created_at,
        )
        store.put_object(document)
        return document

    def closure_for(
        policy_ref: dict[str, str],
        active_ref: dict[str, str],
        receipt_ref: dict[str, str],
    ) -> dict[str, object]:
        return build_factor_production_source_closure(
            deployed_release_ref=closure_payload["deployed_release_ref"],
            release_install_evidence_ref=closure_payload["release_install_evidence_ref"],
            release_install_input_source_ref=closure_payload["release_install_input_source_ref"],
            release_install_verification=closure_payload["release_install_verification"],
            market_pit_selection_ref=closure_payload["market_pit_selection_ref"],
            market_scope_source_ref=closure_payload["market_scope_source_ref"],
            calendar_authority_policy_ref=closure_payload["calendar_authority_policy_ref"],
            calendar_compilation_ref=closure_payload["calendar_compilation_ref"],
            calendar_capture_custody_attestation_ref=closure_payload[
                "calendar_capture_custody_attestation_ref"
            ],
            factor_source_bundle_ref=closure_payload["factor_source_bundle_ref"],
            factor_policy_ref=policy_ref,
            factor_active_set_ref=active_ref,
            factor_validation_attestation_ref=receipt_ref,
            factor_implementation_refs=closure_payload["factor_implementation_refs"],
            legacy_zero_call_ref=closure_payload["legacy_zero_call_ref"],
            market_input_ref=closure_payload["market_input_ref"],
            created_at=closure["created_at"],
        )

    def rebuilt_lane(
        *,
        source_overrides: dict[str, dict[str, object]] | None = None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, object]]:
        source_artifacts = {
            role: store.get_object(reference) for role, reference in source_refs.items()
        }
        source_artifacts.update(source_overrides or {})
        implementation_bundle = source_artifacts["implementation"]
        implementation_ref = implementation_bundle["payload"]["sources"][0]["source_ref"]
        implementation_payload = store.get_object(implementation_ref)["payload"]
        replacement_policy = build_bootstrap_exception_evidence(
            decision_source_bytes=canonical_json_bytes(_DECISION_DOCUMENT),
            source_artifacts=source_artifacts,
            implementation_source_sha256=implementation_payload["byte_sha256"],
            created_at=created_at,
        )
        replacement_policy_ref = store.put_object(replacement_policy)
        replacement_active = build_bootstrap_factor_set(
            bootstrap_exception_evidence=replacement_policy,
            created_at=created_at,
        )
        replacement_active_ref = store.put_object(replacement_active)
        replacement_receipt = _build_factor_validation_receipt(
            policy=replacement_policy,
            active_set=replacement_active,
            evidence_artifacts=[source_artifacts[role] for role in sorted(source_artifacts)],
            trusted_at=created_at,
        )
        replacement_receipt_ref = store.put_object(replacement_receipt)
        return (
            replacement_policy_ref,
            replacement_active_ref,
            replacement_receipt_ref,
            closure_for(
                replacement_policy_ref,
                replacement_active_ref,
                replacement_receipt_ref,
            ),
        )

    def altered_json(reference: dict[str, str], mutate) -> bytes:
        _payload, raw = source_bytes(reference)
        value = parse_canonical_json_bytes(raw, label="adversarial Bootstrap JSON")
        mutate(value)
        return canonical_json_bytes(value)

    def malformed_receipt(name: str, refs: list[dict[str, str]]) -> dict[str, object]:
        payload = dict(receipt["payload"])
        payload["evidence_refs"] = refs
        document = seal_artifact(
            "factor.validation_receipt",
            payload,
            created_at=receipt["created_at"],
        )
        return closure_for(
            closure_payload["factor_policy_ref"],
            closure_payload["factor_active_set_ref"],
            store.put_object(document),
        )

    decision_ref = bundle_source_ref("decision_source")
    implementation_ref = bundle_source_ref("implementation")
    calendar_ref = bundle_source_ref("exchange_calendar")
    market_ref = bundle_source_ref("market")
    pit_ref = bundle_source_ref("pit_universe")
    recomputation_ref = bundle_source_ref("recomputation")
    source_generation_ref = bundle_source_ref("source_generation")

    alternate_decision_bundle = bundle(
        "same-root-decision",
        inner_role="bootstrap_decision",
        source_ref=stage_alternate("same-root-decision", decision_ref),
    )
    altered_implementation_bundle = bundle(
        "implementation-rows",
        inner_role="implementation_tree_manifest",
        source_ref=stage_alternate(
            "implementation-rows",
            implementation_ref,
            raw=altered_json(
                implementation_ref,
                lambda value: value.__setitem__("implementation_rows", []),
            ),
        ),
    )
    alternate_calendar_bundle = bundle(
        "calendar-alias",
        inner_role="calendar",
        source_ref=stage_alternate("calendar-alias", calendar_ref),
    )
    alternate_market_bundle = bundle(
        "market-alias",
        inner_role="market",
        source_ref=stage_alternate("market-alias", market_ref),
    )
    alternate_pit_bundle = bundle(
        "pit-alias",
        inner_role="pit",
        source_ref=stage_alternate("pit-alias", pit_ref),
    )
    altered_recomputation_bundle = bundle(
        "recomputation",
        inner_role="recomputation",
        source_ref=stage_alternate(
            "recomputation",
            recomputation_ref,
            raw=altered_json(
                recomputation_ref,
                lambda value: value.__setitem__("result", "NOT_EXACT_MATCH"),
            ),
        ),
    )
    altered_source_generation_rows = bundle(
        "source-generation-rows",
        inner_role="source_generation",
        source_ref=stage_alternate(
            "source-generation-rows",
            source_generation_ref,
            raw=altered_json(
                source_generation_ref,
                lambda value: value.__setitem__("source_rows", []),
            ),
        ),
    )
    altered_source_generation_reader = bundle(
        "source-generation-reader",
        inner_role="source_generation",
        source_ref=stage_alternate(
            "source-generation-reader",
            source_generation_ref,
            raw=altered_json(
                source_generation_ref,
                lambda value: value["reader_contract"].__setitem__("reader", "other"),
            ),
        ),
    )
    altered_source_generation_hash = bundle(
        "source-generation-hash",
        inner_role="source_generation",
        source_ref=stage_alternate(
            "source-generation-hash",
            source_generation_ref,
            raw=altered_json(
                source_generation_ref,
                lambda value: value.__setitem__("generation_sha256", "0" * 64),
            ),
        ),
    )
    extra_bundle = bundle(
        "receipt-extra",
        inner_role="bootstrap_decision",
        source_ref=stage_alternate("receipt-extra", decision_ref),
    )
    extra_bundle_ref = store.put_object(extra_bundle)
    receipt_refs = [dict(value) for value in receipt["payload"]["evidence_refs"]]
    extra_receipt_refs = sorted(
        [*receipt_refs, extra_bundle_ref],
        key=lambda value: (
            value["kind"],
            value["contract_sha256"],
            value["artifact_id"],
            value["semantic_sha256"],
            value["byte_sha256"],
        ),
    )
    substituted_receipt_refs = [
        extra_bundle_ref if value == source_refs["decision_source"] else value
        for value in receipt_refs
    ]
    substituted_receipt_refs.sort(
        key=lambda value: (
            value["kind"],
            value["contract_sha256"],
            value["artifact_id"],
            value["semantic_sha256"],
            value["byte_sha256"],
        )
    )
    release_payload = dict(store.get_object(source_refs["code"])["payload"])
    release_payload["release_id"] = "adversarial-bootstrap-release"
    alternate_release = seal_artifact("system.release", release_payload, created_at=created_at)
    store.put_object(alternate_release)
    alternate_release_lane = rebuilt_lane(source_overrides={"code": alternate_release})
    alternate_decision_lane = rebuilt_lane(
        source_overrides={"decision_source": alternate_decision_bundle}
    )

    cases = [
        (
            "same-root alternate decision",
            lambda: rebuilt_lane(source_overrides={"decision_source": alternate_decision_bundle})[
                3
            ],
        ),
        (
            "altered implementation tree",
            lambda: rebuilt_lane(
                source_overrides={"implementation": altered_implementation_bundle}
            )[3],
        ),
        (
            "calendar bundle not primary alias",
            lambda: rebuilt_lane(source_overrides={"exchange_calendar": alternate_calendar_bundle})[
                3
            ],
        ),
        (
            "market bundle not primary alias",
            lambda: rebuilt_lane(source_overrides={"market": alternate_market_bundle})[3],
        ),
        (
            "pit bundle not primary alias",
            lambda: rebuilt_lane(source_overrides={"pit_universe": alternate_pit_bundle})[3],
        ),
        (
            "altered recomputation JSON",
            lambda: rebuilt_lane(source_overrides={"recomputation": altered_recomputation_bundle})[
                3
            ],
        ),
        (
            "altered source-generation rows",
            lambda: rebuilt_lane(
                source_overrides={"source_generation": altered_source_generation_rows}
            )[3],
        ),
        (
            "altered source-generation reader",
            lambda: rebuilt_lane(
                source_overrides={"source_generation": altered_source_generation_reader}
            )[3],
        ),
        (
            "altered source-generation hash",
            lambda: rebuilt_lane(
                source_overrides={"source_generation": altered_source_generation_hash}
            )[3],
        ),
        ("receipt missing ref", lambda: malformed_receipt("missing", receipt_refs[:-1])),
        (
            "receipt duplicate ref",
            lambda: malformed_receipt("duplicate", [receipt_refs[0], *receipt_refs]),
        ),
        (
            "receipt reordered refs",
            lambda: malformed_receipt("reordered", list(reversed(receipt_refs))),
        ),
        (
            "receipt additional ref",
            lambda: malformed_receipt("additional", extra_receipt_refs),
        ),
        (
            "receipt substituted ref",
            lambda: malformed_receipt("substituted", substituted_receipt_refs),
        ),
        (
            "policy release cross binding",
            lambda: closure_for(
                alternate_release_lane[0], alternate_release_lane[1], alternate_release_lane[2]
            ),
        ),
        (
            "active cross binding",
            lambda: closure_for(
                closure_payload["factor_policy_ref"],
                alternate_decision_lane[1],
                alternate_decision_lane[2],
            ),
        ),
        (
            "receipt cross binding",
            lambda: closure_for(
                closure_payload["factor_policy_ref"],
                closure_payload["factor_active_set_ref"],
                alternate_decision_lane[2],
            ),
        ),
    ]
    for label, build_closure in cases:
        try:
            production_authority.validate_factor_production_source_closure(
                build_closure(),
                artifact_resolver=store.get_object,
                source_resolver=system_store_source_resolver(store),
            )
        except FactorGovernanceError:
            continue
        raise AssertionError(f"adversarial Bootstrap case was accepted: {label}")
