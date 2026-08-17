from __future__ import annotations

import base64
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.governance.production import assemble_production_bootstrap
from quant_investor.market.exchange_calendar_closure import (
    runtime_json_bytes,
    runtime_parquet_bytes,
)
from quant_investor.migration import run_cutover_gate, validate_cutover_gate_evidence
from quant_investor.system import (
    SystemPreconditionError,
    object_ref_for_artifact,
    prepare_operational_release,
    validate_release_install_evidence,
    verify_release_install_input,
)
from quant_investor.market.tushare_calendar_authority import (
    SOURCE_LIMITATIONS,
    build_trusted_provider_calendar_compilation,
    capture_trusted_provider_calendar_evidence,
)
from quant_investor.system import SystemStore
from tests.unit.test_tushare_calendar_authority import _runtime
import tests.unit.test_unified_production_bootstrap_operator as production_test_module
from tests.unit.test_unified_production_bootstrap_operator import (
    _byte_ref,
    _inputs,
    _request,
    _write,
)
from unified_activation_helpers import prepare_initial_activation

BASE = "2026-08-16T00:00:00Z"


def _git(root: Path, *arguments: str) -> str:
    return (
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("ascii")
        .strip()
    )


def test_frozen_release_build_install_and_exact_origin_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[2]
    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(repository)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    commit = _git(repository, "rev-parse", "HEAD^{commit}")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    release_root = tmp_path / "release"
    release_root.mkdir(mode=0o700)
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "--detach", "-q", commit],
        check=True,
        stdin=subprocess.DEVNULL,
    )

    prepared = prepare_operational_release(
        repository_root=repository,
        release_root=release_root,
        final_commit=commit,
        final_tree=tree,
        created_at=BASE,
    )
    release = prepared["release"]
    evidence = validate_release_install_evidence(prepared["release_install_evidence"])
    assert evidence["payload"]["release_ref"] == object_ref_for_artifact(release)
    exact_input = canonical_json_bytes(
        {"release_install_evidence": evidence, "deployed_release": release}
    )
    assert verify_release_install_input(exact_input, repository_root=repository)["state"] == "PASS"
    assert Path(evidence["payload"]["import_origin"]).is_relative_to(
        Path(evidence["payload"]["install_root"])
    )
    assert not Path(evidence["payload"]["import_origin"]).is_relative_to(repository)

    gate = run_cutover_gate(
        repository_root=repository,
        gate_id="release_install_origin",
        final_commit=commit,
        final_tree=tree,
        subject_ref=object_ref_for_artifact(evidence),
        release_install_evidence=evidence,
        deployed_release=release,
    )
    gate_payload = validate_cutover_gate_evidence(gate)["payload"]
    assert gate_payload["state"] == "PASS", base64.b64decode(
        gate_payload["batch_results"][0]["stderr_base64"]
    ).decode("utf-8", errors="replace")
    assert (
        gate_payload["batch_results"][0]["stdin_sha256"]
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )

    operator_workspace = tmp_path / "operator-workspace"
    authority_root = operator_workspace / "authority"
    operator_workspace.mkdir(mode=0o700)
    authority_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        production_test_module,
        "SYMBOLS",
        ["000001.SZ", "000002.SZ", "430001.BJ", "600000.SH", "600001.SH"],
    )
    capture_parent = tmp_path / "provider-production-inputs"
    files = _inputs(capture_parent)
    authority_root.chmod(0o700)
    capture_parent.chmod(0o700)
    release_input_path = authority_root / "release-install-input.json"
    release_input_path.write_bytes(exact_input)
    release_input_path.chmod(0o600)
    release_input_sha = hashlib.sha256(exact_input).hexdigest()
    installed_python = Path(evidence["payload"]["python_executable"])
    script = textwrap.dedent(r"""
        from datetime import date, timedelta
        import json
        import sys

        import quant_investor.market.tushare_calendar_authority as authority
        from quant_investor.cli.main import main
        from quant_investor.market.tushare_transport import replay_tushare_response_bytes

        DOCS = b'''<!doctype html><html><body>
        <h2>\xe4\xba\xa4\xe6\x98\x93\xe6\x97\xa5\xe5\x8e\x86</h2><p>\xe6\x8e\xa5\xe5\x8f\xa3\xef\xbc\x9atrade_cal</p>
        <h3>\xe8\xbe\x93\xe5\x85\xa5\xe5\x8f\x82\xe6\x95\xb0</h3><table>
        <tr><th>\xe5\x90\x8d\xe7\xa7\xb0</th><th>\xe6\x8f\x8f\xe8\xbf\xb0</th></tr>
        <tr><td>exchange</td><td>SSE\xe4\xb8\x8a\xe4\xba\xa4\xe6\x89\x80,SZSE\xe6\xb7\xb1\xe4\xba\xa4\xe6\x89\x80,CFFEX \xe4\xb8\xad\xe9\x87\x91\xe6\x89\x80,SHFE \xe4\xb8\x8a\xe6\x9c\x9f\xe6\x89\x80,CZCE \xe9\x83\x91\xe5\x95\x86\xe6\x89\x80,DCE \xe5\xa4\xa7\xe5\x95\x86\xe6\x89\x80,INE \xe4\xb8\x8a\xe8\x83\xbd\xe6\xba\x90</td></tr>
        </table>
        <h3>\xe8\xbe\x93\xe5\x87\xba\xe5\x8f\x82\xe6\x95\xb0</h3><table>
        <tr><th>\xe5\x90\x8d\xe7\xa7\xb0</th><th>\xe6\x8f\x8f\xe8\xbf\xb0</th></tr>
        <tr><td>exchange</td><td>SSE\xe4\xb8\x8a\xe4\xba\xa4\xe6\x89\x80 SZSE\xe6\xb7\xb1\xe4\xba\xa4\xe6\x89\x80</td></tr>
        <tr><td>cal_date</td><td>\xe6\x97\xa5\xe5\x8e\x86\xe6\x97\xa5\xe6\x9c\x9f</td></tr>
        <tr><td>is_open</td><td>\xe6\x98\xaf\xe5\x90\xa6\xe4\xba\xa4\xe6\x98\x93</td></tr>
        <tr><td>pretrade_date</td><td>\xe4\xb8\x8a\xe4\xb8\x80\xe4\xb8\xaa\xe4\xba\xa4\xe6\x98\x93\xe6\x97\xa5</td></tr>
        </table></body></html>'''

        def raw(exchange):
            rows = []
            if exchange != "BSE":
                prior = ""
                cursor = date(2023, 12, 1)
                cutoff = date(2026, 8, 7)
                while cursor <= cutoff:
                    opened = cursor.weekday() < 5
                    rows.append([
                        exchange,
                        cursor.strftime("%Y%m%d"),
                        1 if opened else 0,
                        prior,
                    ])
                    if opened:
                        prior = cursor.strftime("%Y%m%d")
                    cursor += timedelta(days=1)
            return json.dumps({
                "code": 0,
                "data": {
                    "count": 0,
                    "fields": list(authority.EXPECTED_FIELDS),
                    "has_more": False,
                    "items": rows,
                },
                "detail": "",
                "msg": "",
                "request_id": "installed-" + exchange.lower(),
            }, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()

        class FakeClient:
            def __init__(self, **kwargs):
                assert kwargs["max_response_bytes"] == 4 * 1024 * 1024

            def request(self, *, api_name, params, expected_fields):
                return replay_tushare_response_bytes(
                    raw(params["exchange"]),
                    api_name=api_name,
                    expected_fields=expected_fields,
                    strict_decimal_decode=True,
                )

        authority._official_documentation_fetch = lambda: (
            DOCS,
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        )
        authority.OfficialTushareHttpsClient = FakeClient
        main([
            "system",
            "calendar-capture",
            "--workspace-root", sys.argv[1],
            "--capture-parent", sys.argv[2],
            "--capture-root-name", "installed-cli-capture",
            "--cutoff-date", "2026-08-07",
            "--release-repository-root", sys.argv[5],
            "--release-install-input", sys.argv[3],
            "--expected-release-install-input-sha256", sys.argv[4],
        ])
        """)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            str(installed_python),
            "-c",
            script,
            str(operator_workspace),
            str(capture_parent),
            release_input_path.relative_to(operator_workspace).as_posix(),
            release_input_sha,
            str(repository),
        ],
        check=True,
        cwd=Path(evidence["payload"]["install_root"]),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    capture_result = json.loads(completed.stdout)
    assert capture_result["status"] == "CAPTURED"
    assert capture_result["network_call_count"] == 4
    assert (
        capture_result["capture_execution"]["payload"]["installed_import_origin"]
        == evidence["payload"]["import_origin"]
    )
    capture_root = capture_parent / "installed-cli-capture"
    assert (capture_root / "capture-success.json").is_file()
    assert len(list(capture_root.iterdir())) == 13

    cutoff = date(2026, 8, 7)
    runtime = _runtime(cutoff=cutoff)
    calendar_json = runtime_json_bytes(runtime)
    calendar_parquet = runtime_parquet_bytes(runtime)
    _write(capture_parent, "strict/exchange_calendar.json", calendar_json)
    _write(capture_parent, "strict/exchange_calendar.parquet", calendar_parquet)
    json_ref = _byte_ref(capture_parent, "strict/exchange_calendar.json")
    parquet_ref = _byte_ref(capture_parent, "strict/exchange_calendar.parquet")
    provider_raw_refs = capture_result["trusted_provider_calendar_raw_file_refs"]
    provider_capture_refs = capture_result["trusted_provider_calendar_capture_file_refs"]
    raw_by_ref = {
        canonical_json_bytes(reference): (capture_parent / reference["relative_path"]).read_bytes()
        for reference in provider_raw_refs
    }
    raw_by_ref[canonical_json_bytes(json_ref)] = calendar_json
    raw_by_ref[canonical_json_bytes(parquet_ref)] = calendar_parquet

    def raw_resolver(reference: dict) -> bytes:
        return raw_by_ref[canonical_json_bytes(reference)]

    policy = json.loads(
        (
            capture_parent / capture_result["calendar_authority_policy_file_ref"]["relative_path"]
        ).read_bytes()
    )
    capability = json.loads(
        (
            capture_parent
            / capture_result["trusted_provider_calendar_capability_file_ref"]["relative_path"]
        ).read_bytes()
    )
    captures = [
        json.loads((capture_parent / reference["relative_path"]).read_bytes())
        for reference in provider_capture_refs
    ]
    docs_raw = (capture_parent / provider_raw_refs[0]["relative_path"]).read_bytes()
    market_sessions = [
        row["date"] for row in runtime if row["status"] == "OPEN" and row["date"] >= "2024-01-02"
    ][-100:]
    release_ref = object_ref_for_artifact(release)
    compilation = build_trusted_provider_calendar_compilation(
        compilation_id="installed-two-root-provider-compilation",
        policy=policy,
        capability=capability,
        capture_documents=captures,
        docs_raw=docs_raw,
        raw_resolver=raw_resolver,
        release_ref=release_ref,
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
        market_session_dates=market_sessions,
        cutoff_date=cutoff.isoformat(),
        calendar_json_file_ref=json_ref,
        calendar_parquet_file_ref=parquet_ref,
        created_at=capture_result["capture_execution"]["payload"]["observed_completed_at"],
    )
    compilation_path = "closure/installed-provider-calendar-compilation.json"
    _write(capture_parent, compilation_path, canonical_json_bytes(compilation))
    files.update(
        {
            "exchange_calendar_file_ref": parquet_ref,
            "calendar_runtime_json_file_ref": json_ref,
            "calendar_compilation_file_ref": _byte_ref(capture_parent, compilation_path),
            "calendar_authority_policy_file_ref": capture_result[
                "calendar_authority_policy_file_ref"
            ],
            "official_calendar_raw_file_refs": [],
            "official_calendar_capture_file_refs": [],
            "official_calendar_decoder_admission_file_refs": [],
            "official_calendar_index_closure_file_refs": [],
            "trusted_provider_calendar_raw_file_refs": provider_raw_refs,
            "trusted_provider_calendar_capture_file_refs": provider_capture_refs,
            "trusted_provider_calendar_capability_file_ref": capture_result[
                "trusted_provider_calendar_capability_file_ref"
            ],
            "trusted_provider_calendar_capture_transaction_file_ref": capture_result[
                "capture_transaction_file_ref"
            ],
            "trusted_provider_calendar_capture_execution_file_ref": capture_result[
                "capture_execution_file_ref"
            ],
            "trusted_provider_calendar_capture_success_file_ref": capture_result[
                "capture_success_file_ref"
            ],
            "trusted_provider_release_install_input_file_ref": capture_result[
                "release_install_input_file_ref"
            ],
        }
    )
    production_workspace = tmp_path / "attached-production-workspace"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(production_workspace)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(production_workspace), "checkout", "-q", "-B", "production-main", commit],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    assert _git(production_workspace, "symbolic-ref", "-q", "HEAD") == (
        "refs/heads/production-main"
    )
    store = SystemStore(production_workspace)
    assert store.put_object(release) == release_ref
    assembled = assemble_production_bootstrap(
        workspace_root=production_workspace,
        input_root=capture_parent,
        request_raw=_request(
            workspace_root=production_workspace,
            release_ref=release_ref,
            files=files,
            operation_id="installed-two-root-provider-bootstrap",
            trusted_at=capture_result["capture_execution"]["payload"]["observed_completed_at"],
        ),
    )
    assert assembled["status"] == "OFFLINE_VERIFIED"
    rules_path = production_workspace / "operations/unified_cutover/rules.json"
    original_rules = rules_path.read_bytes()
    activation = prepare_initial_activation(
        store,
        assembled["generation"],
        release_ref,
    )
    rules_path.write_bytes(original_rules)
    for relative in (
        "authority/_active.json",
        "caller.py",
        "custody/archive.bin",
        "legacy.py",
        "shadow/source.json",
        "src/main.py",
        "strategy/source.json",
    ):
        (production_workspace / relative).unlink()
    assert _git(production_workspace, "status", "--porcelain=v1") == ""
    activated = store.activate_initial_generation(**activation)
    assert activated["generation_state"] == "OPERATIONAL"
    active = store.read_active(deployed_release_ref=release_ref)
    assert active is not None
    assert active["factor_status"]["payload"]["readiness"] == "READY"
    status = store.status()
    assert status["state"] == "PARTIAL"
    assert status["calendar_source_limitations"] == list(SOURCE_LIMITATIONS)

    wrong_origin_parent = tmp_path / "wrong-origin"
    wrong_origin_parent.mkdir(mode=0o700)
    with pytest.raises(SystemPreconditionError, match="not running installed release"):
        capture_trusted_provider_calendar_evidence(
            capture_parent=wrong_origin_parent,
            capture_root_name="wrong-origin",
            cutoff_date="2026-08-07",
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=release_input_sha,
            release_repository_root=repository,
        )
    assert not (wrong_origin_parent / "wrong-origin").exists()

    wheel = Path(evidence["payload"]["wheel"]["path"])
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(SystemPreconditionError, match="archive exact bytes"):
        verify_release_install_input(exact_input, repository_root=repository)


def test_release_preparation_requires_exact_clean_detached_checkout(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2]
    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(repository)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    commit = _git(repository, "rev-parse", "HEAD^{commit}")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    release_root = tmp_path / "release"
    release_root.mkdir(mode=0o700)

    with pytest.raises(SystemPreconditionError, match="attached to a branch"):
        prepare_operational_release(
            repository_root=repository,
            release_root=release_root,
            final_commit=commit,
            final_tree=tree,
            created_at=BASE,
        )

    subprocess.run(
        ["git", "-C", str(repository), "checkout", "--detach", "-q", commit],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    (repository / "untracked-release-drift").write_text("drift", encoding="utf-8")
    with pytest.raises(SystemPreconditionError, match="not clean"):
        prepare_operational_release(
            repository_root=repository,
            release_root=release_root,
            final_commit=commit,
            final_tree=tree,
            created_at=BASE,
        )
