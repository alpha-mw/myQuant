from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

import quant_investor.factors.governance_canonical_replay as replay
import quant_investor.factors.governance_protocol_v2 as protocol
import scripts.daily_factor_mining_automation as daily
import scripts.factor_health_automation as health
import scripts.mine_quant_branch_factors as mining


EXPECTED_PRODUCTION_REGISTRY_SHA = (
    "b8369dfef7d27156999e93e3a1a12020e072db0296532fee10b0335d8bddca2f"
)
BLOCKER = "forward_factor_apply_not_authorized_pr4"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_REGISTRY = (
    REPO_ROOT / "quant_investor/factor_registry/mined_factors.json"
)


def _load_fixture_module() -> ModuleType:
    fixture_path = Path(__file__).with_name(
        "test_factor_governance_canonical_replay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_pr4_canonical_replay_fixture", fixture_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("canonical replay fixture module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = _load_fixture_module()


@pytest.fixture()
def replay_case(tmp_path: Path) -> dict[str, Any]:
    return FIXTURES.replay_fixture.__wrapped__(tmp_path)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return replay.canonical_json_bytes(payload) + b"\n"


def _registry_state() -> tuple[str, int, int]:
    value = PRODUCTION_REGISTRY.stat()
    return (
        _sha(PRODUCTION_REGISTRY.read_bytes()),
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
    )


@contextmanager
def _production_registry_unchanged() -> Iterator[None]:
    before = _registry_state()
    assert before == (EXPECTED_PRODUCTION_REGISTRY_SHA, 0o644, 1)
    try:
        yield
    finally:
        assert _registry_state() == before


def _published_paths(case: dict[str, Any]) -> tuple[Path, Path]:
    registry_sha = _sha(case["registry"].read_bytes())
    receipt = case["private"] / "receipts" / f"{registry_sha}.json"
    bundle = case["private"] / "bundles" / "evidence-20260107-1.json"
    return receipt, bundle


def test_normal_replay_second_run_is_exactly_idempotent(
    replay_case: dict[str, Any],
) -> None:
    with _production_registry_unchanged():
        first = replay.produce_canonical_replay(
            private_root=replay_case["private"],
            registry_path=replay_case["registry"],
            draft_path=replay_case["draft"],
        )
        receipt, bundle = _published_paths(replay_case)
        receipt_raw = receipt.read_bytes()
        bundle_raw = bundle.read_bytes()
        receipt_payload = json.loads(receipt_raw)
        bundle_payload = json.loads(bundle_raw)

        assert receipt.name == f'{first["registry_sha256"]}.json'
        assert receipt_payload["bundle_sha256"] == _sha(bundle_raw)
        assert receipt_raw == _canonical_bytes(receipt_payload)
        assert bundle_raw == _canonical_bytes(bundle_payload)
        assert [
            (item["arm"], item["stage"])
            for item in bundle_payload["stages"]
        ] == [
            (arm, stage)
            for arm in replay.ARM_NAMES
            for stage in replay.CONTROL_CHAIN_STAGES
        ]
        assert len(bundle_payload["stages"]) == 20

        second = replay.produce_canonical_replay(
            private_root=replay_case["private"],
            registry_path=replay_case["registry"],
            draft_path=replay_case["draft"],
        )
        assert second == first
        assert receipt.read_bytes() == receipt_raw
        assert bundle.read_bytes() == bundle_raw
        assert list((replay_case["private"] / "receipts").glob("*.json")) == [
            receipt
        ]
        assert list((replay_case["private"] / "bundles").glob("*.json")) == [
            bundle
        ]


@pytest.mark.parametrize(
    "category",
    [
        "registry",
        "bundle",
        "receipt",
        "snapshot_pointer",
        "snapshot_manifest",
        "calendar",
        "pit_manifest",
        "pit_canonical",
        "code_config_manifest",
        "code_config_file",
        "stage",
    ],
)
def test_post_publish_byte_drift_matrix_fails_closed(
    replay_case: dict[str, Any], category: str
) -> None:
    with _production_registry_unchanged():
        replay.produce_canonical_replay(
            private_root=replay_case["private"],
            registry_path=replay_case["registry"],
            draft_path=replay_case["draft"],
        )
        receipt, bundle = _published_paths(replay_case)
        draft = replay_case["draft_payload"]
        if category == "registry":
            target = replay_case["registry"]
        elif category == "bundle":
            target = bundle
        elif category == "receipt":
            target = receipt
        elif category == "code_config_file":
            manifest = json.loads(
                Path(draft["code_config_manifest"]["path"]).read_bytes()
            )
            target = Path(manifest["files"][0]["path"])
        elif category == "stage":
            target = Path(draft["stages"][0]["path"])
        else:
            target = Path(draft[category]["path"])
        target.write_bytes(target.read_bytes() + b" ")

        with pytest.raises(replay.CanonicalReplayError):
            replay.verify_canonical_replay(
                private_root=replay_case["private"],
                registry_path=replay_case["registry"],
            )


@pytest.mark.parametrize(
    "attack", ["receipt_symlink", "receipt_hardlink", "receipt_mode", "bundle_fifo"]
)
def test_exact_readback_namespace_attacks_fail_closed(
    replay_case: dict[str, Any], attack: str
) -> None:
    with _production_registry_unchanged():
        replay.produce_canonical_replay(
            private_root=replay_case["private"],
            registry_path=replay_case["registry"],
            draft_path=replay_case["draft"],
        )
        receipt, bundle = _published_paths(replay_case)
        if attack == "receipt_symlink":
            target = replay_case["private"] / "receipt-target.json"
            target.write_bytes(receipt.read_bytes())
            target.chmod(0o600)
            receipt.unlink()
            receipt.symlink_to(target)
        elif attack == "receipt_hardlink":
            os.link(receipt, replay_case["private"] / "receipt-alias.json")
        elif attack == "receipt_mode":
            receipt.chmod(0o640)
        else:
            bundle.unlink()
            os.mkfifo(bundle, 0o600)

        with pytest.raises(replay.CanonicalReplayError):
            replay.verify_canonical_replay(
                private_root=replay_case["private"],
                registry_path=replay_case["registry"],
            )


class InjectedPublishFault(OSError):
    pass


def _install_publish_fault(
    patch: pytest.MonkeyPatch,
    crash_point: str,
    destination: Path,
    temp: Path,
) -> None:
    if crash_point == "umask_probe":
        patch.setattr(
            replay,
            "_preflight_publish_umask",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedPublishFault("umask probe")
            ),
        )
        return

    if crash_point == "temp_write":
        real_write = os.write
        failed = False

        def fail_partial_write(fd: int, raw: Any) -> int:
            nonlocal failed
            if not failed:
                failed = True
                real_write(fd, raw[: max(1, len(raw) // 2)])
                raise InjectedPublishFault("temp write")
            return real_write(fd, raw)

        patch.setattr(os, "write", fail_partial_write)
        return

    if crash_point == "temp_fsync":
        real_fsync = os.fsync
        failed = False

        def fail_temp_fsync(fd: int) -> None:
            nonlocal failed
            value = os.fstat(fd)
            if not failed and stat.S_ISREG(value.st_mode) and value.st_size:
                failed = True
                raise InjectedPublishFault("temp fsync")
            real_fsync(fd)

        patch.setattr(os, "fsync", fail_temp_fsync)
        return

    if crash_point == "link":
        patch.setattr(
            os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedPublishFault("link")
            ),
        )
        return

    if crash_point == "parent_fsync":
        real_link = os.link
        real_fsync = os.fsync
        linked = False

        def observe_link(*args: Any, **kwargs: Any) -> None:
            nonlocal linked
            real_link(*args, **kwargs)
            linked = True

        def fail_parent_fsync(fd: int) -> None:
            if linked and stat.S_ISDIR(os.fstat(fd).st_mode):
                raise InjectedPublishFault("parent fsync")
            real_fsync(fd)

        patch.setattr(os, "link", observe_link)
        patch.setattr(os, "fsync", fail_parent_fsync)
        return

    if crash_point == "unlink":
        real_unlink = os.unlink

        def fail_temp_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            if path == temp.name:
                raise InjectedPublishFault("temp unlink")
            real_unlink(path, *args, **kwargs)

        patch.setattr(os, "unlink", fail_temp_unlink)
        return

    if crash_point == "post_unlink_parent_fsync":
        real_unlink = os.unlink
        real_fsync = os.fsync
        temp_unlinked = False

        def observe_temp_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal temp_unlinked
            real_unlink(path, *args, **kwargs)
            if path == temp.name:
                temp_unlinked = True

        def fail_post_unlink_parent_fsync(fd: int) -> None:
            if temp_unlinked and stat.S_ISDIR(os.fstat(fd).st_mode):
                raise InjectedPublishFault("post-unlink parent fsync")
            real_fsync(fd)

        patch.setattr(os, "unlink", observe_temp_unlink)
        patch.setattr(os, "fsync", fail_post_unlink_parent_fsync)
        return

    if crash_point == "final_readback":
        real_read = replay.SafeReadSession.read_bytes

        def fail_final_readback(self: Any, path: Any) -> bytes:
            if os.fspath(path) == str(destination) and destination.exists():
                raise InjectedPublishFault("final readback")
            return real_read(self, path)

        patch.setattr(replay.SafeReadSession, "read_bytes", fail_final_readback)
        return
    raise AssertionError(f"unknown crash point: {crash_point}")


@pytest.mark.parametrize(
    "crash_point",
    [
        "umask_probe",
        "temp_write",
        "temp_fsync",
        "link",
        "parent_fsync",
        "unlink",
        "post_unlink_parent_fsync",
        "final_readback",
    ],
)
def test_publish_crash_boundaries_never_leave_torn_final_and_retry_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    with _production_registry_unchanged():
        private = tmp_path / "private"
        bundles = private / "bundles"
        private.mkdir(mode=0o700)
        bundles.mkdir(mode=0o700)
        payload = {"value": 1}
        raw = _canonical_bytes(payload)
        destination = bundles / "value.json"
        _, temp = FIXTURES._publish_reserved_paths(
            bundles, destination.name, payload
        )

        with monkeypatch.context() as fault:
            _install_publish_fault(
                fault, crash_point, destination, temp
            )
            with pytest.raises(InjectedPublishFault):
                replay.publish_immutable_json(
                    private, "bundles/value.json", payload
                )

        if destination.exists():
            assert destination.read_bytes() == raw
            assert destination.stat().st_nlink in {1, 2}

        result = replay.publish_immutable_json(
            private, "bundles/value.json", payload
        )
        assert result == {"sha256": _sha(raw), "size": len(raw)}
        assert destination.read_bytes() == raw
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        assert destination.stat().st_nlink == 1
        assert not temp.exists()


class Poison:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"forward gate touched poison: {name}")

    def __fspath__(self) -> str:
        raise AssertionError("forward gate converted a path")

    def __iter__(self) -> Any:
        raise AssertionError("forward gate iterated an input")


def _explode(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("forward gate crossed into I/O or side effects")


def _poison_forward_surfaces(patch: pytest.MonkeyPatch) -> None:
    for module, names in (
        (
            protocol,
            (
                "Path",
                "load_registry_snapshot_strict",
                "protocol_hash",
                "canonical_replay_producer_control",
                "reserve_monthly_mutation_budget",
                "apply_factor_record_patch",
            ),
        ),
        (
            daily,
            (
                "Path",
                "_now_shanghai",
                "latest_download_report",
                "run_mining",
                "write_outputs",
                "load_registry_snapshot_strict",
                "load_governance_replay_evidence",
                "apply_governed_transition",
            ),
        ),
        (health, ("Path", "load_registry_snapshot_strict", "build_runtime_smoke")),
        (mining, ("Path", "load_registry_snapshot_strict", "run_mining")),
    ):
        for name in names:
            if hasattr(module, name):
                patch.setattr(module, name, _explode)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "protocol",
        "daily_run",
        "daily_cli",
        "health_cli",
        "mining_candidate",
        "mining_family",
        "mining_cli",
    ],
)
def test_forward_apply_matrix_blocks_before_io_or_external_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entrypoint: str,
) -> None:
    with _production_registry_unchanged():
        daily_argv = [
            "--apply-governed-transitions",
            "--protocol-version",
            "v2",
            "--expected-protocol-hash",
            protocol.protocol_hash(),
            "--governed-evidence-json",
            "poison.json",
        ]
        daily_args = daily.parse_args(daily_argv)
        with monkeypatch.context() as poison:
            _poison_forward_surfaces(poison)
            if entrypoint == "protocol":
                result = protocol.apply_governed_transition(
                    Poison(),
                    Poison(),
                    expected_protocol_hash=Poison(),
                    valid_trading_days=Poison(),
                    write=True,
                )
                assert result["blockers"] == [BLOCKER]
            elif entrypoint == "daily_run":
                result = daily.run_daily_automation(daily_args)
                assert result["factor_protocol"]["blockers"] == [BLOCKER]
            elif entrypoint == "daily_cli":
                assert daily.main(daily_argv) == 2
            elif entrypoint == "health_cli":
                assert health.main(["--apply-registry-actions"]) == 2
            elif entrypoint == "mining_candidate":
                result = mining.apply_production_candidate_registry_updates(
                    registry_path=Poison(),
                    qualified_results=Poison(),
                    run_timestamp=Poison(),
                    run_id=Poison(),
                    report_path=Poison(),
                    owner=Poison(),
                    source_notes=Poison(),
                    journal_path=Poison(),
                    write=True,
                )
                assert result["blockers"] == [BLOCKER]
            elif entrypoint == "mining_family":
                result = mining.apply_production_family_governance(
                    registry_path=Poison(),
                    results=Poison(),
                    run_timestamp=Poison(),
                    run_id=Poison(),
                    report_path=Poison(),
                    journal_path=Poison(),
                    write=True,
                )
                assert result["blockers"] == [BLOCKER]
            else:
                assert mining.main(["--write-production-candidates"]) == 2
        if entrypoint.endswith("cli"):
            assert BLOCKER in capsys.readouterr().err
