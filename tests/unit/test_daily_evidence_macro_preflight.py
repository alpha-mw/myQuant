"""Native transaction-envelope preflight: no INTENT, filesystem writes or locks."""

import hashlib
import pytest
from quant_investor.macro import maintenance_transaction as native
from test_macro_maintenance_transaction import _fixture, _authority_args


def inventory(root):
    return {
        str(p.relative_to(root)): (
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        )
        for p in root.rglob("*")
    }


@pytest.mark.parametrize(
    "mutation",
    [None, "input", "market", "pit", "release", "observations", "candidate", "target", "sha"],
)
def test_native_preflight_is_readonly_and_rejects_drift(tmp_path, monkeypatch, mutation):
    source = tmp_path / "bound-input.json"
    source.write_bytes(b"{}\n")
    source.chmod(0o600)
    fixture = _fixture(
        tmp_path,
        input_bindings={
            "source": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        },
    )
    kwargs = {
        "prepared_path": fixture["prepared_path"],
        "expected_prepared_sha256": fixture["prepared_sha"],
        "expected_target_date": "20260819",
        **_authority_args(fixture),
    }
    if mutation == "input":
        source.write_bytes(b"changed\n")
    elif mutation in {"market", "pit"}:
        fixture[mutation + "_pointer"].write_bytes(b"changed\n")
    elif mutation in {"release", "observations"}:
        (fixture[mutation] / "_latest.json").write_bytes(b"changed\n")
    elif mutation == "candidate":
        (tmp_path / "candidate-release/_generations/release-child/manifest.json").write_bytes(
            b"changed"
        )
    elif mutation == "target":
        kwargs["expected_target_date"] = "20260820"
    elif mutation == "sha":
        kwargs["expected_prepared_sha256"] = "a" * 64

    def forbidden(*args, **kwargs):
        pytest.fail("preflight attempted write, lock or commit")

    for name in (
        "_write_exclusive",
        "_journal_directory",
        "_transaction_locks",
        "commit_prepared_macro_transaction",
    ):
        monkeypatch.setattr(native, name, forbidden)
    before = inventory(tmp_path)
    if mutation is None:
        result = native._preflight_prepared_commit(**kwargs)
        assert result == {
            "prepared_path": fixture["prepared_path"],
            "prepared_sha256": fixture["prepared_sha"],
        }
    else:
        with pytest.raises(native.MacroMaintenanceTransactionError):
            native._preflight_prepared_commit(**kwargs)
    assert inventory(tmp_path) == before
    assert not (fixture["journal"] / "run-1").exists()


def test_candidate_authority_cannot_enter_first_commit_preflight(tmp_path):
    fixture = _fixture(tmp_path, authority_mode="candidate")
    with pytest.raises(native.MacroMaintenanceTransactionError, match="canonical_required"):
        native._preflight_prepared_commit(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            expected_target_date="20260819",
            **_authority_args(fixture),
        )
    assert not (fixture["journal"] / "run-1").exists()
