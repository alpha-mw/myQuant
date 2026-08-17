from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration import run_cutover_gate, validate_cutover_gate_evidence
from quant_investor.system import (
    SystemPreconditionError,
    object_ref_for_artifact,
    prepare_operational_release,
    validate_release_install_evidence,
    verify_release_install_input,
)
from quant_investor.market.tushare_calendar_authority import (
    capture_trusted_provider_calendar_evidence,
)

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


def test_frozen_release_build_install_and_exact_origin_replay(tmp_path: Path) -> None:
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

    authority_root = repository / "results/system/authority"
    capture_parent = repository / "results/system/calendar-captures"
    authority_root.mkdir(parents=True, mode=0o700)
    capture_parent.mkdir(parents=True, mode=0o700)
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
                cutoff = date(2025, 7, 1)
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
            "--cutoff-date", "2025-07-01",
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
            str(repository),
            str(capture_parent),
            release_input_path.relative_to(repository).as_posix(),
            release_input_sha,
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

    wrong_origin_parent = repository / "results/system/wrong-origin"
    wrong_origin_parent.mkdir(mode=0o700)
    with pytest.raises(SystemPreconditionError, match="not running installed release"):
        capture_trusted_provider_calendar_evidence(
            capture_parent=wrong_origin_parent,
            capture_root_name="wrong-origin",
            cutoff_date="2025-07-01",
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=release_input_sha,
            repository_root=repository,
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
