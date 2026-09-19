"""Real research input failures keep exact causes without changing writer custody."""

import hashlib
from types import SimpleNamespace

import pytest

from quant_investor.cli import unified
from quant_investor.cli.output import CommandError, command_boundary
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.dependency_diagnostics import DependencyInputError
from quant_investor.operations.research_file_readback import ResearchFileReadback
from quant_investor.operations.research_request import load_research_request
from quant_investor.operations.research_source_bundle import SourceBundle, FIELDS, SCHEMA
from quant_investor.system.errors import SystemSecurityError, SystemStorageError
from quant_investor.system.storage import SecureSystemStorage
from test_daily_evidence_runner import setup

LABEL = "CUTOFF_SOURCE_REF_INVALID"


def put(root, raw, name="source.json", mode=0o600):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize("path", ["missing.json", "missing-parent/source.json"])
def test_safe_missing_research_wrapper_has_exact_category(tmp_path, path):
    with pytest.raises(DependencyInputError) as error:
        load_research_request(workspace=str(tmp_path), reference={"path": path, "sha256": "a" * 64})
    assert error.value.reason_code == "RESEARCH_SOURCE_MISSING"
    assert error.value.failure_code == "INPUT_MISSING"


@pytest.mark.parametrize(
    "raw,wrong_sha,reason,category",
    [
        (b"not JSON", True, "RESEARCH_REQUEST_SHA_MISMATCH", "SHA_MISMATCH"),
        (b"not JSON", False, "RESEARCH_SOURCE_JSON_INVALID", "SCHEMA_MISMATCH"),
        (b"[]", False, "RESEARCH_REQUEST_DOCUMENT_INVALID", "SCHEMA_MISMATCH"),
        (
            b'{"schema_version":"unknown"}',
            False,
            "RESEARCH_REQUEST_WRAPPER_INVALID",
            "SCHEMA_MISMATCH",
        ),
    ],
)
def test_wrapper_bytes_are_validated_in_native_order(tmp_path, raw, wrong_sha, reason, category):
    ref = put(tmp_path, raw)
    if wrong_sha:
        ref["sha256"] = "0" * 64
    with pytest.raises(DependencyInputError) as error:
        load_research_request(workspace=str(tmp_path), reference=ref)
    assert (error.value.reason_code, error.value.failure_code) == (reason, category)


@pytest.mark.parametrize("fault", ["missing", "sha"])
def test_safe_native_file_failures_translate_only_owned_labels(tmp_path, fault):
    ref = put(tmp_path, b"source")
    if fault == "missing":
        (tmp_path / ref["path"]).unlink()
    else:
        ref["sha256"] = "0" * 64
    with pytest.raises(DependencyInputError) as error:
        ResearchFileReadback(str(tmp_path)).source_file(ref, code=LABEL)
    assert error.value.reason_code == (
        "RESEARCH_NATIVE_SOURCE_MISSING"
        if fault == "missing"
        else "RESEARCH_NATIVE_SOURCE_SHA_MISMATCH"
    )
    with pytest.raises(CommandError) as public:
        ResearchFileReadback(str(tmp_path)).source_file(ref, code="UNTRUSTED_LABEL")
    assert public.value.to_dict() == {"status": "BLOCKED", "blocker_code": "UNTRUSTED_LABEL"}


@pytest.mark.parametrize("category", ["SAFE_SOURCE_MISSING", "SAFE_SOURCE_SHA_MISMATCH"])
def test_internal_native_error_preserves_public_stdout_and_exit(category, capsys):
    for cls in (CommandError, unified._DailySourceFileError):
        kwargs = {} if cls is CommandError else {"source_failure": category}

        def fail():
            raise cls("PUBLIC_SOURCE_INVALID", **kwargs)

        with pytest.raises(SystemExit) as result:
            command_boundary(fail)
        captured = capsys.readouterr()
        assert result.value.code == 2
        assert captured.out == '{"blocker_code":"PUBLIC_SOURCE_INVALID","status":"BLOCKED"}\n'
        assert captured.err == ""


@pytest.mark.parametrize(
    "fault",
    ["symlink", "parent_symlink", "hardlink", "mode", "unsafe_mode", "case_alias", "not_directory"],
)
def test_native_security_ambiguity_is_not_a_typed_input_failure(tmp_path, fault):
    ref = put(tmp_path, b"source")
    path = tmp_path / ref["path"]
    if fault == "symlink":
        (tmp_path / "alias.json").symlink_to(path)
        ref["path"] = "alias.json"
    elif fault == "parent_symlink":
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        ref["path"] = "alias/source.json"
    elif fault == "hardlink":
        (tmp_path / "alias.json").hardlink_to(path)
    elif fault == "mode":
        path.chmod(0o644)
    elif fault == "unsafe_mode":
        path.chmod(0o666)
    elif fault == "case_alias":
        ref["path"] = "SOURCE.json"
    else:
        ref["path"] = "source.json/child.json"
    ref["sha256"] = "0" * 64
    with pytest.raises(CommandError) as error:
        ResearchFileReadback(str(tmp_path)).source_file(ref, code=LABEL)
    assert not isinstance(error.value, unified._DailySourceFileError)


def test_secure_security_error_with_missing_cause_is_preserved(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise SystemSecurityError("unsafe ownership") from FileNotFoundError("private")

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", denied)
    with pytest.raises(SystemSecurityError):
        load_research_request(
            workspace=str(tmp_path), reference={"path": "missing.json", "sha256": "0" * 64}
        )
    with pytest.raises(CommandError) as native:
        ResearchFileReadback(str(tmp_path)).source_file(
            {"path": "missing.json", "sha256": "0" * 64}, code=LABEL
        )
    assert not isinstance(native.value, unified._DailySourceFileError)


def test_changed_bytes_during_native_confirmation_stay_generic(tmp_path, monkeypatch):
    ref = put(tmp_path, b"first")
    ref["sha256"] = "0" * 64
    monkeypatch.setattr(
        SecureSystemStorage,
        "read_workspace_file_bytes",
        lambda *args, **kwargs: SimpleNamespace(data=b"replaced", byte_sha256="f" * 64),
    )
    with pytest.raises(CommandError) as error:
        ResearchFileReadback(str(tmp_path)).source_file(ref, code=LABEL)
    assert not isinstance(error.value, unified._DailySourceFileError)


def test_source_bundle_day_failure_precedes_source_reads(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260827")
    document = dict.fromkeys(FIELDS)
    document.update(schema_version=SCHEMA, trade_date="20260826")
    with pytest.raises(DependencyInputError) as error:
        SourceBundle(journal=journal, document=document)
    assert error.value.reason_code == "CUTOFF_SOURCE_BUNDLE_DAY_INVALID"
    assert error.value.failure_code == "DATE_MISMATCH"


@pytest.mark.parametrize("raw", [b"not JSON", b'{"a":1,"a":2}'])
def test_declared_source_json_rejection_is_owned_by_its_decoder(tmp_path, raw):
    sources = object.__new__(SourceBundle)
    sources.files = ResearchFileReadback(str(tmp_path))
    with pytest.raises(DependencyInputError) as error:
        sources.read(put(tmp_path, raw))
    assert error.value.reason_code == "RESEARCH_SOURCE_JSON_INVALID"
    assert error.value.failure_code == "SCHEMA_MISMATCH"


def test_real_research_file_error_reaches_downstream_without_writer(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)
    reader = ResearchFileReadback(str(tmp_path))
    ref = put(tmp_path, b"source")
    ref["sha256"] = "0" * 64

    def probe(request):
        reader.source_file(ref, code=LABEL)

    adapters["industry"].probe = probe
    result = runner.run(templates)
    row = result["nodes"]["industry"]
    assert row["state"] == "BLOCKED" and row["attempt"] == 0
    assert row["blocking_reason"] == "SHA_MISMATCH" and row["retryable"] is False
    assert row["failure"]["owner_action_required"] is True
    assert row["failure"]["recommended_next_node"] == "industry"
    assert result["nodes"]["decision"]["upstream_blockers"] == [
        {
            "node_id": "industry",
            "failure_code": "SHA_MISMATCH",
            "reason_code": "RESEARCH_NATIVE_SOURCE_SHA_MISMATCH",
        }
    ]
    assert "industry" not in calls and "decision" not in calls


def test_resolver_constructor_error_rethrows_without_fake_attempt(tmp_path):
    runner, adapters, templates, calls = setup(tmp_path)

    def resolve(node, completed):
        return load_research_request(
            workspace=str(tmp_path), reference={"path": "missing.json", "sha256": "0" * 64}
        )

    with runner.journal.locked():
        original = runner.journal.begin(templates["calendar"])
        with pytest.raises(DependencyInputError):
            runner.run_locked({}, resolve=resolve)
        assert runner.journal.inspect(templates["calendar"]) == original
    assert calls == []


@pytest.mark.parametrize("fault", ["payload_sha", "wrapper_reread"])
def test_payload_hash_is_typed_but_wrapper_reread_race_is_not(tmp_path, monkeypatch, fault):
    from quant_investor.operations import research_cutoff
    from quant_investor.operations.daily_contract import ContractError
    from quant_investor.operations.research_request import WRAPPER_SCHEMA
    from quant_investor.operations.exposure_completion import POLICY

    payload_ref = put(tmp_path, canonical_json_bytes({"synthetic": True}), "payload.json")
    wrapper = {
        "schema_version": WRAPPER_SCHEMA,
        "trade_date": "20260827",
        "native_request_ref": payload_ref,
        "cutoff_ref": {"path": "cutoff.json", "sha256": "c" * 64},
        "source_completion_policy": POLICY,
    }
    wrapper_ref = put(tmp_path, canonical_json_bytes(wrapper), "wrapper.json")
    monkeypatch.setattr(
        research_cutoff,
        "read_cutoff_inputs",
        lambda **kwargs: {
            "research_request_ref": wrapper_ref,
            "native_request_ref": payload_ref,
            "receipt": {"schema_version": "cn-daily-research-cutoff.v1"},
        },
    )
    original = SecureSystemStorage.read_workspace_file_bytes
    calls = []

    def read(self, path, **kwargs):
        stored = original(self, path, **kwargs)
        calls.append(path)
        if fault == "wrapper_reread" and calls.count("wrapper.json") == 2:
            return SimpleNamespace(data=b"changed")
        return stored

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", read)
    if fault == "payload_sha":
        (tmp_path / "payload.json").write_bytes(b"changed")
    with pytest.raises(ContractError) as error:
        load_research_request(workspace=str(tmp_path), reference=wrapper_ref)
    if fault == "payload_sha":
        assert isinstance(error.value, DependencyInputError)
        assert error.value.reason_code == "RESEARCH_REQUEST_PAYLOAD_SHA_MISMATCH"
        assert calls.count("wrapper.json") == 1
    else:
        assert not isinstance(error.value, DependencyInputError)
        assert str(error.value) == "RESEARCH_REQUEST_CHANGED"


def test_nested_storage_cause_cannot_be_classified_as_missing(tmp_path, monkeypatch):
    def nested(*args, **kwargs):
        try:
            raise SystemStorageError("inner") from FileNotFoundError("private")
        except SystemStorageError as inner:
            raise SystemStorageError("outer") from inner

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", nested)
    with pytest.raises(SystemStorageError):
        load_research_request(
            workspace=str(tmp_path), reference={"path": "missing.json", "sha256": "0" * 64}
        )


@pytest.mark.parametrize("fault", ["ownership", "permission", "unstable_read"])
def test_native_confirmation_denial_stays_generic(tmp_path, monkeypatch, fault):
    ref = put(tmp_path, b"source")
    ref["sha256"] = "0" * 64

    def denied(*args, **kwargs):
        if fault == "permission":
            raise PermissionError("source access")
        raise SystemSecurityError(fault)

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", denied)
    with pytest.raises(CommandError) as error:
        ResearchFileReadback(str(tmp_path)).source_file(ref, code=LABEL)
    assert not isinstance(error.value, unified._DailySourceFileError)


@pytest.mark.parametrize("fault", ["absolute", "traversal", "escape", "missing_parent"])
def test_native_path_boundary_before_missing_category(tmp_path, fault):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if fault == "absolute":
        name = str(tmp_path / "missing.json")
    elif fault == "traversal":
        name = "../missing.json"
    elif fault == "escape":
        (workspace / "escape").symlink_to(tmp_path, target_is_directory=True)
        name = "escape/missing.json"
    else:
        name = "safe-parent/missing.json"
    with pytest.raises(CommandError) as error:
        unified._daily_source_file(workspace, {"path": name, "sha256": "0" * 64}, code=LABEL)
    assert isinstance(error.value, unified._DailySourceFileError) is (fault == "missing_parent")


def test_missing_file_appearing_during_confirmation_stays_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SecureSystemStorage,
        "read_workspace_file_bytes",
        lambda *a, **k: SimpleNamespace(data=b"appeared"),
    )
    with pytest.raises(CommandError) as error:
        ResearchFileReadback(str(tmp_path)).source_file(
            {"path": "missing.json", "sha256": "0" * 64}, code=LABEL
        )
    assert not isinstance(error.value, unified._DailySourceFileError)


def test_oversized_native_source_keeps_native_size_error(tmp_path):
    from quant_investor.migration.errors import UnifiedCutoverError, FILE_TOO_LARGE

    path = tmp_path / "large.bin"
    with path.open("wb") as stream:
        stream.truncate(512 * 1024 * 1024 + 1)
    path.chmod(0o600)
    with pytest.raises(UnifiedCutoverError) as error:
        ResearchFileReadback(str(tmp_path)).source_file(
            {"path": "large.bin", "sha256": "0" * 64}, code=LABEL
        )
    assert error.value.code == FILE_TOO_LARGE


@pytest.mark.parametrize("fault", ["missing", "sha", "schema"])
def test_retained_cutoff_errors_are_split_before_native_replay(tmp_path, fault):
    from quant_investor.operations.research_cutoff import read_cutoff_inputs

    journal = DailyJournal(str(tmp_path), "20260827")
    ref = {"path": str(journal.root / "cutoff.json"), "sha256": "0" * 64}
    if fault != "missing":
        with journal.locked():
            journal.storage.write(ref["path"], b"bad json")
        if fault == "schema":
            ref["sha256"] = hashlib.sha256(b"bad json").hexdigest()
    with pytest.raises(DependencyInputError) as error:
        read_cutoff_inputs(journal=journal, cutoff_ref=ref)
    assert (
        error.value.reason_code
        == {
            "missing": "CUTOFF_RETAINED_REF_MISSING",
            "sha": "CUTOFF_RETAINED_REF_SHA_MISMATCH",
            "schema": "RESEARCH_SOURCE_JSON_INVALID",
        }[fault]
    )
