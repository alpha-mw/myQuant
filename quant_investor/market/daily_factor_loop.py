"""Deterministic Factor wiring for the existing CN daily launcher and maintainer."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from quant_investor.factors.production_authority import FactorProductionStore
from quant_investor.factors.production_observation import register_factor_production_observations
from quant_investor.factors.production_outcomes import settle_production_observations
from quant_investor.factors.production_rollover import (
    validate_daily_maintenance_receipt,
    _read_owner_file,
)
from quant_investor.system.release_install import verify_running_release_install_input
from .daily_maintenance import _canonical_json_bytes, _write_once, _owner_only_directory
from .future_calendar_context import (
    FUTURE_CONTEXT_SCHEMAS,
    FUTURE_STATE_SCHEMAS,
    PRODUCTION_MODE,
    PRODUCTION_CONTEXT_SCHEMA,
)


def read_paper_comparison(workspace: Path) -> dict[str, Any]:
    """Read the existing registered strategy result; never infer Factor attribution."""
    from quant_investor.strategy_records.performance import load_performance_history
    from quant_investor.strategy_records import load_registered_catalog

    root = workspace / "results/strategy_records/CN/aggressive_tech_manufacturing"
    try:
        loaded = load_registered_catalog(root)
        if loaded is None:
            raise ValueError("STRATEGY_RECORDS_NOT_REGISTERED")
        pointer, catalog = loaded
        performance = load_performance_history(root, catalog["performance_history_ref"])
        manifest = performance["manifest"]
        if manifest["final_record_id"] != pointer["active_record_id"]:
            raise ValueError("STRATEGY_PERFORMANCE_ACTIVE_RECORD_MISMATCH")
        series_ref = performance["ref"]["series"]
        return {
            "state": "READ_ONLY_REFERENCE",
            "factor_attribution": "UNAVAILABLE",
            "reason": "NO_VERIFIED_FACTOR_CONSUMPTION_BINDING",
            "writes": 0,
            "strategy_record_id": pointer["active_record_id"],
            "record_ref": {"path": str(root / series_ref["path"]), "sha256": series_ref["sha256"]},
            "valuation_date": manifest["seed_end_date"],
            "total_value": manifest["last_raw_nav_cny"],
            "portfolio_return": manifest["cumulative_return"],
            "max_drawdown": manifest["max_drawdown"],
        }
    except Exception as exc:
        return {"state": "UNAVAILABLE", "reason": str(exc), "writes": 0}


def read_factor_loop_context(
    *, workspace_root: str, context_path: str, context_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the existing installed loop context without creating its run directory."""
    workspace = Path(workspace_root).resolve(strict=True)
    raw, sha = _read_owner_file(Path(context_path), root=workspace, label="daily Factor context")
    if sha != context_sha256:
        raise ValueError("DAILY_FACTOR_CONTEXT_SHA_MISMATCH")
    context = json.loads(raw)
    from .future_calendar_context import validate_loop_context, future_mode

    validate_loop_context(context)
    if future_mode(context) == "SYNTHETIC_FIXTURE_ONLY":
        from ._calendar_fixture_capability import require_fixture_capability

        require_fixture_capability(
            workspace=workspace,
            install_sha=context["release_install_input_ref"]["sha256"],
        )
    if future_mode(context) == PRODUCTION_MODE:
        from ._calendar_production_transport import _guard_route

        _guard_route()
    release_ref = context["release_install_input_ref"]
    release_raw, release_sha = _read_owner_file(
        workspace / release_ref["path"], root=workspace, label="daily Factor release"
    )
    if release_sha != release_ref["sha256"]:
        raise ValueError("DAILY_FACTOR_RELEASE_SHA_MISMATCH")
    installation = verify_running_release_install_input(
        release_raw, repository_root=context["release_repository_root"]
    )
    if installation.get("state") != "PASS":
        raise ValueError("DAILY_FACTOR_INSTALLED_RUNTIME_REQUIRED")
    return context, installation


class DailyFactorLoop:
    """One installed context; no authority is inferred from readiness flags or stdout."""

    _core_handoff_completed: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    _require_existing_calendar_capture: bool = False

    def __init__(
        self,
        *,
        workspace_root: str,
        run_root: str,
        context_path: str,
        context_sha256: str,
        core_handoff_completed: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ):
        self.workspace = Path(workspace_root).resolve(strict=True)
        self.context, self.installation = read_factor_loop_context(
            workspace_root=workspace_root, context_path=context_path, context_sha256=context_sha256
        )
        if core_handoff_completed is not None and not callable(core_handoff_completed):
            raise ValueError("DAILY_CORE_HANDOFF_CALLBACK_INVALID")
        self.context_sha256 = context_sha256
        self._core_handoff_completed = core_handoff_completed
        self.run_root = _owner_only_directory(Path(run_root), create=True)
        self.store = FactorProductionStore(self.workspace)
        self.stages: dict[str, Any] = {}
        self.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _core_release_ref(self) -> dict[str, str]:
        """Bind DAG receipts to the installed release artifact, not its input envelope."""
        from quant_investor.contracts import validate_artifact
        from quant_investor.system.store import object_ref_for_artifact

        expected = self.installation["release_ref"]
        if expected["kind"] != "system.release":
            raise ValueError("DAILY_CORE_RELEASE_KIND_INVALID")
        path = f"results/factors/objects/system.release/{expected['byte_sha256']}.json"
        stored = self.store.read(path)
        if stored is None or stored.byte_sha256 != expected["byte_sha256"]:
            raise ValueError("DAILY_CORE_RELEASE_ARTIFACT_MISSING_OR_CHANGED")
        if object_ref_for_artifact(validate_artifact(stored.data)) != expected:
            raise ValueError("DAILY_CORE_RELEASE_ARTIFACT_MISMATCH")
        return {"path": path, "sha256": stored.byte_sha256}

    def _attempt(self, stage: str, callback: Any) -> dict[str, Any]:
        try:
            result = dict(callback())
        except Exception as exc:
            result = {"status": "BLOCKED", "blocker": str(exc), "error_class": type(exc).__name__}
        self.stages[stage] = result
        return result

    def _settle(self, ref: Mapping[str, str]) -> dict[str, Any]:
        return settle_production_observations(
            workspace_root=str(self.workspace),
            calendar_receipt=ref["path"],
            expected_calendar_sha256=ref["sha256"],
            limit=32,
        )

    def recover(self) -> dict[str, Any]:
        ref = self.context["initial_calendar_receipt_ref"]
        state_path = self.run_root / "factor-loop-state.json"
        state = None
        if state_path.exists():
            raw, _ = _read_owner_file(state_path, root=self.workspace, label="daily Factor state")
            state = json.loads(raw)
            if state.get("schema_version") in FUTURE_STATE_SCHEMAS:
                from .future_calendar_context import validate_future_state, future_mode

                validate_future_state(
                    state,
                    context_sha256=self.context_sha256,
                    trade_date=state.get("trade_date"),
                    mode=future_mode(self.context),
                    context_schema=self.context["schema_version"],
                )
            elif self.context.get("schema_version") in FUTURE_CONTEXT_SCHEMAS:
                from .future_calendar_context import future_mode

                if (
                    self.context.get("schema_version") == PRODUCTION_CONTEXT_SCHEMA
                    or future_mode(self.context) != "DISABLED"
                ):
                    raise ValueError("DAILY_FACTOR_STATE_SCHEMA_MISMATCH")
            ref = state["calendar_receipt_ref"]
        self._attempt(
            "observation_recovery",
            lambda: register_factor_production_observations(
                str(self.workspace), recover_history=True
            ),
        )
        if state is not None and state.get("core_observation_refs"):
            self._attempt("core_handoff_recovery", lambda: self._recover_core(state))
        self._attempt("historical_settlement", lambda: self._settle(ref))
        raw, _ = _read_owner_file(Path(ref["path"]), root=self.workspace, label="recovery Calendar")
        calendar = json.loads(raw)
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        if calendar["calendar_end_date"] < today:
            self.stages["calendar_freshness"] = {
                "status": "BLOCKED",
                "blocker": "CURRENT_EXPECTED_SESSION_UNCONFIRMED",
                "verified_calendar_through": calendar["calendar_end_date"],
            }
        return self.report(maintenance=None)

    def _recover_core(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Recover only from explicitly retained native observation refs, never a scan."""
        from quant_investor.factors.production_observation import (
            validate_factor_production_observation,
        )
        from quant_investor.operations.core_pool import publish_core_pool

        refs = state["core_observation_refs"]
        if type(refs) is not dict or set(refs) != {"LOW", "W80"}:
            raise ValueError("CORE_RECOVERY_OBSERVATION_SET_INVALID")
        observations = []
        for alias in ("LOW", "W80"):
            ref = refs[alias]
            path = Path(ref["path"])
            if not path.is_absolute():
                path = self.workspace / path
            raw, sha = _read_owner_file(
                path, root=self.workspace, label="core recovery observation"
            )
            if sha != ref["sha256"]:
                raise ValueError("CORE_RECOVERY_OBSERVATION_SHA_MISMATCH")
            value = validate_factor_production_observation(raw)["payload"]
            day = value["signal_date"]
            expected = (
                self.workspace
                / f"results/factors/observations/{day[:4]}/{day[4:6]}/{day[6:]}/{alias}.json"
            )
            if path != expected or value["factor_alias"] != alias:
                raise ValueError("CORE_RECOVERY_OBSERVATION_PATH_MISMATCH")
            observations.append(value)
        low, w80 = observations
        if any(
            low[k] != w80[k]
            for k in ("signal_date", "factor_pointer_sha256", "factor_generation_sha256")
        ):
            raise ValueError("CORE_RECOVERY_OBSERVATION_BINDING_MISMATCH")
        from .future_calendar_producer import bind_future_calendar, core_future_arguments

        if state.get("schema_version") in FUTURE_STATE_SCHEMAS:
            if state.get("trade_date") != low["signal_date"]:
                raise ValueError("DAILY_FACTOR_STATE_DATE_MISMATCH")
            state = bind_future_calendar(
                workspace=self.workspace,
                context=self.context,
                context_sha=self.context_sha256,
                state=dict(state),
                save_state=self._save_state,
                recovery=True,
            )
        return publish_core_pool(
            workspace=str(self.workspace),
            trade_date=low["signal_date"],
            factor_pointer_sha256=low["factor_pointer_sha256"],
            release_ref=self._core_release_ref(),
            **core_future_arguments(state),
        )

    def _replay_core_completed(self, checkpoint: Mapping[str, str]) -> dict[str, Any]:
        """Repair from existing capture only; a finalized attempt grants no new provider call."""
        previous = self._require_existing_calendar_capture
        self._require_existing_calendar_capture = True
        try:
            return self.core_completed(checkpoint)
        finally:
            self._require_existing_calendar_capture = previous

    def core_completed(self, checkpoint: Mapping[str, str]) -> dict[str, Any]:
        # Called synchronously by daily-maintain with its lock held, before auxiliaries.
        maintenance = validate_daily_maintenance_receipt(
            workspace_root=self.workspace,
            receipt_path=checkpoint["path"],
            expected_receipt_sha256=checkpoint["sha256"],
        )
        close_ref = maintenance["close_session_receipt_ref"]
        target = maintenance["target_date"]
        if self.context.get("schema_version") in FUTURE_CONTEXT_SCHEMAS:
            from .future_calendar_context import validate_future_state, future_mode

            retained = self.run_root / "factor-loop-state.json"
            if retained.exists():
                raw, _ = _read_owner_file(retained, root=self.workspace, label="daily Factor state")
                prior_state = json.loads(raw)
                if prior_state.get("schema_version") in FUTURE_STATE_SCHEMAS:
                    validate_future_state(
                        prior_state,
                        context_sha256=self.context_sha256,
                        trade_date=prior_state.get("trade_date"),
                        mode=future_mode(self.context),
                        context_schema=self.context["schema_version"],
                    )
                    if prior_state["trade_date"] == target and prior_state[
                        "core_checkpoint_ref"
                    ] != dict(checkpoint):
                        raise ValueError("DAILY_FACTOR_STATE_CHECKPOINT_MISMATCH")
                elif (
                    self.context.get("schema_version") == PRODUCTION_CONTEXT_SCHEMA
                    or future_mode(self.context) != "DISABLED"
                ):
                    raise ValueError("DAILY_FACTOR_STATE_SCHEMA_MISMATCH")
        self.stages["core_data"] = {
            "status": "CORE_COMPLETE",
            "target_session": target,
            "checkpoint_ref": dict(checkpoint),
            "provider_calls": False,
        }
        predecessor = self._attempt(
            "pre_rollover_observation",
            lambda: register_factor_production_observations(
                str(self.workspace), recover_history=True
            ),
        )

        def advance() -> dict[str, Any]:
            from quant_investor.cli.unified import (
                factor_production_rollover,
                system_calendar_capture,
            )

            current = self.store.read("results/factors/_active.json")
            name = (
                f"tushare-calendar-{target}-{self.context['release_commit'][:7]}-"
                f"{checkpoint['sha256'][:12]}"
            )
            parent = _owner_only_directory(
                Path(self.context["calendar_capture_parent"]),
                create=not self._require_existing_calendar_capture,
            )
            capture_root = parent / name
            release = self.context["release_install_input_ref"]
            from .tushare_transport import operation_provider_summary

            before = operation_provider_summary()
            capture = None
            try:
                if capture_root.exists():
                    from quant_investor.market.tushare_calendar_authority import (
                        validate_published_trusted_provider_calendar_capture_root,
                    )

                    execution_path = capture_root / "capture-execution.json"
                    success_path = capture_root / "capture-success.json"
                    execution_raw, execution_sha = _read_owner_file(
                        execution_path, root=parent, label="existing Calendar execution"
                    )
                    success_raw, success_sha = _read_owner_file(
                        success_path, root=parent, label="existing Calendar success"
                    )
                    leaves = validate_published_trusted_provider_calendar_capture_root(
                        capture_parent=parent,
                        capture_execution=execution_raw,
                        capture_execution_file_ref={
                            "relative_path": name + "/capture-execution.json",
                            "byte_sha256": execution_sha,
                        },
                        capture_success=success_raw,
                        capture_success_file_ref={
                            "relative_path": name + "/capture-success.json",
                            "byte_sha256": success_sha,
                        },
                    )
                    if (
                        hashlib.sha256(leaves["release-install-input.json"]).hexdigest()
                        != release["sha256"]
                    ):
                        raise ValueError("CALENDAR_RELEASE_INPUT_DRIFT")
                    transaction = json.loads(leaves["capture-transaction.json"])["payload"]
                    if (
                        transaction["cutoff_date"]
                        != datetime.strptime(target, "%Y%m%d").date().isoformat()
                    ):
                        raise ValueError("CALENDAR_TARGET_DRIFT")
                    capture = {"status": "VERIFIED_REPLAY"}
                else:
                    if self._require_existing_calendar_capture:
                        raise ValueError("CORE_RECOVERY_CALENDAR_CAPTURE_MISSING")
                    capture = system_calendar_capture(
                        workspace_root=str(self.workspace),
                        capture_parent=str(parent),
                        capture_root_name=name,
                        cutoff_date=datetime.strptime(target, "%Y%m%d").date().isoformat(),
                        release_repository_root=self.context["release_repository_root"],
                        release_install_input_path=release["path"],
                        expected_release_install_input_sha256=release["sha256"],
                    )
            finally:
                after = operation_provider_summary()
                count = sum(after["provider_request_attempts"].values()) - sum(
                    before["provider_request_attempts"].values()
                )
                self.stages["factor_calendar"] = {
                    "status": capture.get("status") if capture else "BLOCKED",
                    "provider_calls": (
                        count > 0 if after["provider_calls"] != "UNCONFIRMED" else "UNCONFIRMED"
                    ),
                    "observed_provider_request_count": count,
                    "capture_root": str(capture_root),
                }
            success = capture_root / "capture-success.json"
            success_sha = hashlib.sha256(success.read_bytes()).hexdigest()
            self.stages["factor_calendar"]["success_ref"] = {
                "path": str(success),
                "sha256": success_sha,
            }
            return factor_production_rollover(
                workspace_root=str(self.workspace),
                market_data_root=str(self.workspace / "data"),
                calendar_capture_root=str(capture_root),
                expected_calendar_success_sha256=success_sha,
                maintenance_receipt=checkpoint["path"],
                expected_maintenance_receipt_sha256=checkpoint["sha256"],
                expected_current_pointer_sha256=current.byte_sha256,
            )

        if predecessor.get("status") == "BLOCKED":
            self.stages["factor_rollover"] = {
                "status": "BLOCKED",
                "blocker": "PREDECESSOR_OBSERVATION_NOT_CLOSED",
            }
        else:
            self._attempt("factor_rollover", advance)
        if self.stages["factor_rollover"].get("status") == "BLOCKED":
            self.stages["observation_registration"] = {"status": "SKIPPED_DEPENDENCY_BLOCKED"}
        else:
            self._attempt(
                "observation_registration",
                lambda: register_factor_production_observations(
                    str(self.workspace), recover_history=True
                ),
            )
        # The pool is part of the native core handoff, before slow auxiliaries.
        # Missing children cannot be repaired by a scheduler guessing next steps.
        observation_result = self.stages["observation_registration"]
        state = {
            "schema_version": "cn-daily-factor-state.v1",
            "calendar_receipt_ref": close_ref,
            "core_checkpoint_ref": dict(checkpoint),
        }
        if observation_result.get("command_status") not in {"REGISTERED", "NO_ACTION"}:
            self.stages["top100_publication"] = {"status": "SKIPPED_DEPENDENCY_BLOCKED"}
        else:
            from quant_investor.operations.core_pool import publish_core_pool

            state["core_observation_refs"] = {
                row["factor_alias"]: {
                    "path": row["observation_path"],
                    "sha256": row["observation_sha256"],
                }
                for row in observation_result["observations"]
            }
            from .future_calendar_producer import (
                new_future_state,
                bind_future_calendar,
                core_future_arguments,
            )

            if self.context.get("schema_version") in FUTURE_CONTEXT_SCHEMAS:
                retained = self.run_root / "factor-loop-state.json"
                recovery = self._require_existing_calendar_capture
                if retained.exists():
                    raw, _ = _read_owner_file(
                        retained, root=self.workspace, label="daily Factor state"
                    )
                    previous = json.loads(raw)
                    if previous.get("trade_date") == target:
                        if previous.get("core_checkpoint_ref") != dict(checkpoint):
                            raise ValueError("DAILY_FACTOR_STATE_CHECKPOINT_MISMATCH")
                        state = previous
                        recovery = True
                    else:
                        state = new_future_state(
                            day=target,
                            context_sha=self.context_sha256,
                            state=state,
                            context_schema=self.context["schema_version"],
                        )
                else:
                    state = new_future_state(
                        day=target,
                        context_sha=self.context_sha256,
                        state=state,
                        context_schema=self.context["schema_version"],
                    )
                state = bind_future_calendar(
                    workspace=self.workspace,
                    context=self.context,
                    context_sha=self.context_sha256,
                    state=state,
                    save_state=self._save_state,
                    recovery=recovery,
                )
            # Keep the exact recovery inputs before starting the pool writer.
            self._save_state(state)

            self._attempt(
                "top100_publication",
                lambda: publish_core_pool(
                    workspace=str(self.workspace),
                    trade_date=target,
                    factor_pointer_sha256=self.store.read(
                        "results/factors/_active.json"
                    ).byte_sha256,
                    release_ref=self._core_release_ref(),
                    **core_future_arguments(state),
                ),
            )
        self._handoff_before_settlement(state, close_ref)
        return self.report(maintenance={"status": "IN_PROGRESS", "target_date": target})

    def _handoff_before_settlement(
        self, state: dict[str, Any], close_ref: Mapping[str, str]
    ) -> None:
        pool_result = self.stages["top100_publication"]
        if pool_result.get("core_handoff_ref"):
            state["core_handoff_ref"] = pool_result["core_handoff_ref"]
            if state.get("schema_version") in FUTURE_STATE_SCHEMAS:
                state["phase"] = "CORE_PUBLISHED"
        self._save_state(state)
        if state.get("core_handoff_ref") and self._core_handoff_completed is not None:
            # Fixed coordinator hook; never supplied through request JSON. Publish
            # the recovery anchor before settlement or maintenance auxiliaries.
            self.stages["maintenance_handoff_publication"] = dict(
                self._core_handoff_completed(dict(state))
            )
        # A failed/newly absent signal still permits independent settlement.
        self._attempt("outcome_settlement", lambda: self._settle(close_ref))

    def _save_state(self, state: Mapping[str, Any]) -> None:
        # Derived recovery context, written only under the existing daily lock.
        import os

        if state.get("schema_version") in FUTURE_STATE_SCHEMAS:
            from .future_calendar_context import validate_future_state, future_mode

            validate_future_state(
                dict(state),
                context_sha256=self.context_sha256,
                trade_date=state["trade_date"],
                mode=future_mode(self.context),
                context_schema=self.context["schema_version"],
            )
        raw = _canonical_json_bytes(state)
        immutable = self.run_root / ("factor-state-" + hashlib.sha256(raw).hexdigest() + ".json")
        if not immutable.exists():
            _write_once(immutable, raw)
        target = self.run_root / "factor-loop-state.json"
        temporary = self.run_root / ".factor-loop-state.pending"
        if temporary.exists():
            if temporary.read_bytes() != raw:
                raise ValueError("DAILY_FACTOR_STATE_IN_DOUBT")
        else:
            _write_once(temporary, raw)
        os.replace(temporary, target)
        fd = os.open(self.run_root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def report(self, *, maintenance: Mapping[str, Any] | None) -> dict[str, Any]:
        root = _owner_only_directory(self.run_root / "factor_reports", create=True)
        diagnostic = self.stages.get(
            "outcome_settlement", self.stages.get("historical_settlement", {})
        )
        blocked = [
            name
            for name, value in self.stages.items()
            if value.get("status") == "BLOCKED"
            or value.get("errors")
            or any("INVALID" in counts for counts in value.get("horizon_states", {}).values())
        ]
        summary = {
            "schema_version": "cn-daily-factor-report.v1",
            "actual_run_time": self.started_at,
            "installation": self.installation,
            "expected_session": (
                maintenance.get("target_date") if maintenance else diagnostic.get("as_of")
            ),
            "OPERATIONAL_RESULT": {
                "run_scope": (
                    "DAILY_MAINTENANCE_AND_FACTOR" if maintenance else "HISTORICAL_RECOVERY_ONLY"
                ),
                "scheduled": "UNCONFIRMED_CURRENT_DISPATCH",
                "started": True,
                "state": "PARTIAL" if blocked else "COMPLETED",
                "maintenance": dict(maintenance) if maintenance else None,
                "stages": self.stages,
                "blockers": blocked,
            },
            "FACTOR_DIAGNOSTIC_RESULT": diagnostic,
            "PAPER_COMPARISON_RESULT": read_paper_comparison(self.workspace),
            "effectiveness": "FACTOR_EFFECTIVENESS_INSUFFICIENT_EVIDENCE",
            "other_authority": "NONE",
        }
        handoffs = [
            self.stages[name]["core_handoff_ref"]
            for name in ("top100_publication", "core_handoff_recovery")
            if self.stages.get(name, {}).get("core_handoff_ref") is not None
        ]
        if handoffs:
            from quant_investor.operations.daily_contract import validate_ref

            for ref in handoffs:
                validate_ref(ref)
            if any(ref != handoffs[0] for ref in handoffs):
                raise ValueError("DAILY_FACTOR_REPORT_HANDOFF_CONFLICT")
            summary["core_handoff_ref"] = dict(handoffs[0])
        raw = _canonical_json_bytes(summary)
        sha = hashlib.sha256(raw).hexdigest()
        path = root / f"{sha}.json"
        if not path.exists():
            _write_once(path, raw)
        markdown = (
            f"# Daily Factor evidence — {summary['expected_session']}\n\n"
            f"Actual run: {self.started_at}\n\n"
            f"Operations: {summary['OPERATIONAL_RESULT']['state']}\n\n"
            f"New evaluations: {diagnostic.get('new_evaluation_count', 'UNAVAILABLE')}\n\n"
            "Horizon states: "
            f"{json.dumps(diagnostic.get('horizon_states', {}), ensure_ascii=False)}\n\n"
            "Factor results are descriptive raw-price diagnostics. "
            "Economic/executable returns and paper attribution require their own evidence.\n"
        )
        md = root / f"{sha}.md"
        if not md.exists():
            _write_once(md, markdown.encode())
        return {
            "status": summary["OPERATIONAL_RESULT"]["state"],
            "report_ref": {"path": str(path), "sha256": sha},
            "markdown_path": str(md),
            "blockers": blocked,
            "new_evaluation_count": diagnostic.get("new_evaluation_count", 0),
            **({"core_handoff_ref": summary["core_handoff_ref"]} if handoffs else {}),
        }
