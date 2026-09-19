"""The release test snapshot must match running source, including staged changes."""

from _native_daily_release_fixture import snapshot_repository, git


def test_snapshot_includes_staged_edits_deletions_and_untracked_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet")
    package = source / "quant_investor"
    package.mkdir()
    (package / "edited.py").write_text("original\n")
    (package / "deleted.py").write_text("delete me\n")
    git(source, "add", "quant_investor")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    (package / "edited.py").write_text("staged\n")
    (package / "deleted.py").unlink()
    git(source, "add", "-u")
    (package / "new.py").write_text("new\n")
    (source / ".env").write_text("private fixture\n")
    status = git(source, "status", "--porcelain")
    target = tmp_path / "snapshot"
    snapshot_repository(source, target)
    assert (target / "quant_investor/edited.py").read_text() == "staged\n"
    assert (target / "quant_investor/new.py").read_text() == "new\n"
    assert not (target / "quant_investor/deleted.py").exists()
    assert not (target / ".env").exists()
    assert git(target, "status", "--porcelain") == ""
    assert git(source, "status", "--porcelain") == status
