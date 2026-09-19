"""Configured production Calendar: fresh capture and separate network-free recovery."""

from pathlib import Path

from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_revisions import selected_binding
from quant_investor.operations.future_calendar_binding import future_calendar_outputs
from quant_investor.system.errors import SystemPreconditionError, SystemSecurityError
from .future_calendar_context import PRODUCTION_CONTEXT_SCHEMA, future_mode, validate_future_state
from .future_calendar_producer import (
    _retained_capture,
    _retained_native_failure,
    _projection_failure_code,
)
from .next_session_calendar import inspect_next_session_capture
from .next_session_proof import read_next_session_proof
from .next_session_failure import publish_next_session_failure
from .production_calendar_evidence import read_transport_evidence
from .production_calendar_proof import publish_production_next_session_proof


class _Binding:
    def __init__(self, *, workspace, context, context_sha, state, save_state):
        self.workspace = str(workspace)
        self.context, self.context_sha, self.state, self.save = (
            context,
            context_sha,
            state,
            save_state,
        )
        self.day = state["trade_date"]
        self.mode = future_mode(context)
        if context["schema_version"] != PRODUCTION_CONTEXT_SCHEMA:
            raise ContractError("DAILY_FACTOR_STATE_SCHEMA_MISMATCH")
        self.validate()
        if self.mode != "DISABLED":
            from ._calendar_fixture_capability import _ACTIVE as fixture_active

            if fixture_active.get() is not None:
                raise SystemSecurityError("NEXT_SESSION_TRANSPORT_ROUTE_UNAVAILABLE")
        self.journal = DailyJournal(self.workspace, self.day)
        self.install_sha = context["release_install_input_ref"]["sha256"]

    def validate(self):
        validate_future_state(
            self.state,
            context_sha256=self.context_sha,
            trade_date=self.day,
            mode=self.mode,
            context_schema=PRODUCTION_CONTEXT_SCHEMA,
        )

    def persist(self, **changes):
        self.state.update(changes)
        self.validate()
        self.save(self.state)

    def ready(self):
        if self.state["phase"] in {"FUTURE_BOUND", "CORE_PUBLISHED"}:
            ref = self.state["next_session_calendar_proof_ref"]
            if ref is not None:
                proof = read_next_session_proof(
                    workspace=self.workspace, eod_trade_date=self.day, publication_ref=ref
                )
                if proof["synthetic"] is not False or proof["live_eligible"] is not True:
                    raise ContractError("NEXT_SESSION_PRODUCTION_SCHEMA_INVALID")
                core = proof["proof"]
                captures = self.state["future_capture_refs"]
                if (
                    core["transport_evidence_ref"] != self.state["transport_evidence_ref"]
                    or core["execution_ref"] != captures["execution_ref"]
                    or core["success_ref"] != captures["success_ref"]
                    or Path(core["execution_ref"]["relative_path"]).parts[0]
                    != "future-" + self.day + "-" + self.install_sha[:16]
                ):
                    raise ContractError("NEXT_SESSION_TRANSPORT_BINDING_INVALID")
            future_calendar_outputs(
                workspace=self.workspace,
                trade_date=self.day,
                proof_ref=ref,
                failure_ref=self.state["next_session_calendar_failure_ref"],
            )
            return True
        if (
            self.journal.storage.read(str(self.journal.root / "completion.v1.json")) is not None
            or selected_binding(self.journal.storage, str(self.journal.root / "nodes/calendar"))[0]
            is not None
        ):
            raise ContractError("NEXT_SESSION_CALENDAR_ALREADY_SELECTED")
        if self.mode == "DISABLED":
            self.persist(phase="FUTURE_BOUND")
            return True
        return False

    def capture_refs(self, captured):
        return {
            "execution_ref": captured["capture_execution_file_ref"],
            "success_ref": captured["capture_success_file_ref"],
        }

    def inspect(self, captured):
        return inspect_next_session_capture(
            workspace=self.workspace,
            eod_trade_date=self.day,
            execution=captured["capture_execution"],
            success=captured["capture_success"],
            **self.capture_refs(captured),
        )

    def failure(self, code, captured=None, native_failure_ref=None):
        refs = {} if captured is None else self.capture_refs(captured)
        ref = publish_next_session_failure(
            workspace=self.workspace,
            eod_trade_date=self.day,
            phase="ACQUISITION" if captured is None else "POST_CAPTURE_VALIDATION",
            failure_code=code,
            native_failure_ref=native_failure_ref,
            **refs,
        )
        self.persist(
            phase="FUTURE_BOUND",
            next_session_calendar_failure_ref=ref,
            future_capture_refs=None if captured is None else refs,
        )
        return self.state

    def proof(self, captured):
        ref = publish_production_next_session_proof(
            workspace=self.workspace,
            eod_trade_date=self.day,
            execution=captured["capture_execution"],
            success=captured["capture_success"],
            transport_evidence_ref=self.state["transport_evidence_ref"],
            **self.capture_refs(captured),
        )
        self.persist(phase="FUTURE_BOUND", next_session_calendar_proof_ref=ref)
        return self.state


def recover_production_calendar(**kwargs):
    """Retained reads/binding only; no capture/recorder/transport operation reachable."""
    binding = _Binding(**kwargs)
    if binding.ready():
        return binding.state
    captured = _retained_capture(
        binding.journal, binding.install_sha, binding.state["future_capture_refs"]
    )
    if captured is None:
        if binding.state["phase"] != "CORE_READY":
            raise ContractError("NEXT_SESSION_CAPTURE_FILE_REF_MISMATCH")
        return binding.failure(
            "ACQUISITION_FAILED",
            native_failure_ref=_retained_native_failure(binding.journal, binding.install_sha),
        )
    try:
        info = binding.inspect(captured)
        ref, _ = read_transport_evidence(
            binding.journal, info, binding.state["transport_evidence_ref"]
        )
    except ContractError as exc:
        code = _projection_failure_code(exc)
        if code is None:
            raise
        if code == "PROVENANCE_UNAVAILABLE" and binding.state["transport_evidence_ref"] is not None:
            raise SystemSecurityError("NEXT_SESSION_TRANSPORT_BINDING_INVALID") from exc
        return binding.failure(code, captured)
    binding.persist(
        phase="TRANSPORT_BOUND",
        future_capture_refs=binding.capture_refs(captured),
        transport_evidence_ref=ref,
    )
    return binding.proof(captured)


def bind_fresh_production_calendar(**kwargs):
    """The sole fresh branch; scope activation and capture are unreachable in recovery."""
    binding = _Binding(**kwargs)
    if binding.ready():
        return binding.state
    if binding.state["phase"] != "CORE_READY":
        return recover_production_calendar(**kwargs)
    from quant_investor.factors.production_rollover import _read_owner_file
    from ._calendar_production_transport import _production_transport_scope
    from .next_session_acquisition import capture_next_session_calendar
    from .production_calendar_evidence import publish_transport_evidence

    ref = binding.context["release_install_input_ref"]
    raw, sha = _read_owner_file(
        Path(binding.workspace) / ref["path"],
        root=Path(binding.workspace),
        label="production future Calendar release",
    )
    if sha != binding.install_sha:
        raise ContractError("NEXT_SESSION_RELEASE_INPUT_SHA_MISMATCH")
    try:
        with _production_transport_scope(
            workspace=binding.workspace,
            day=binding.day,
            install_raw=raw,
            install_ref=ref,
            repository=binding.context["release_repository_root"],
        ) as session:
            binding.save(binding.state)
            result = capture_next_session_calendar(
                workspace=binding.workspace,
                eod_trade_date=binding.day,
                release_install_input_raw=raw,
                expected_release_install_input_sha256=sha,
                release_repository_root=binding.context["release_repository_root"],
            )
            captured = result["capture"]
            binding.persist(
                phase="CAPTURE_SEALED", future_capture_refs=binding.capture_refs(captured)
            )
            evidence = publish_transport_evidence(
                workspace=binding.workspace,
                trade_date=binding.day,
                captured=captured,
                session=session,
            )
            binding.persist(phase="TRANSPORT_BOUND", transport_evidence_ref=evidence)
    except SystemPreconditionError as exc:
        native_ref = exc.public_fields.get("capture_failure_file_ref")
        if native_ref is None:
            raise
        if native_ref != _retained_native_failure(binding.journal, binding.install_sha):
            raise ContractError("NEXT_SESSION_FAILURE_SOURCE_SHA_MISMATCH") from exc
        return binding.failure("ACQUISITION_FAILED", native_failure_ref=native_ref)
    except ContractError as exc:
        code = _projection_failure_code(exc)
        if code is None:
            raise
        captured = _retained_capture(binding.journal, binding.install_sha)
        if captured is None:
            raise
        return binding.failure(code, captured)
    return binding.proof(captured)
