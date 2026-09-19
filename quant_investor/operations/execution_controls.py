"""Bind pre-maintenance request/recipe bytes; this grants no execution authority."""

from dataclasses import dataclass
from pathlib import Path
import re

from quant_investor.cli.unified import _daily_source_file
from quant_investor.contracts import parse_canonical_json_bytes
from .daily_contract import ContractError, validate_ref
from .production_request import validate_production_request
from .execution_recipe import validate_execution_recipe
from .dependency_diagnostics import DependencyInputError


def _validate_producer_loop_context(context: dict, *, verified_commit: str) -> None:
    """EXECUTE-only dependency check; never creates paths or grants write access."""
    from quant_investor.market.future_calendar_context import validate_loop_context

    # This internal dependency diagnostic also accepts a projection containing
    # only commit/capture-parent. Public readers validate full context first.
    if "schema_version" in context:
        validate_loop_context(context)
    commit = context.get("release_commit")
    if commit is None:
        raise DependencyInputError("EXECUTION_CONTEXT_RELEASE_COMMIT_MISSING")
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise DependencyInputError("EXECUTION_CONTEXT_RELEASE_COMMIT_INVALID")
    if commit != verified_commit:
        raise DependencyInputError("EXECUTION_CONTEXT_RELEASE_COMMIT_MISMATCH")
    parent = context.get("calendar_capture_parent")
    if type(parent) is not str or not parent or "\x00" in parent or "\\" in parent:
        raise DependencyInputError("EXECUTION_CONTEXT_CALENDAR_PARENT_INVALID")
    path = Path(parent)
    try:
        valid = (
            path.is_absolute()
            and str(path) == parent
            and ".." not in path.parts
            and path.resolve(strict=False) == path
            and (not path.exists() or path.is_dir())
        )
    except (OSError, RuntimeError, ValueError):
        valid = False
    if not valid:
        raise DependencyInputError("EXECUTION_CONTEXT_CALENDAR_PARENT_INVALID")


@dataclass(frozen=True)
class ExecutionControls:
    workspace: str
    request_ref: tuple[str, str]
    recipe_ref: tuple[str, str]
    sources: tuple[tuple[str, str, bytes], ...]

    def document(self, reference: tuple[str, str]):
        for path, sha, raw in self.sources:
            if (path, sha) == reference:
                return parse_canonical_json_bytes(raw)
        raise ContractError("EXECUTION_CONTROL_REF_NOT_CAPTURED")

    def recheck(self) -> None:
        for path, sha, raw in self.sources:
            _, current, _ = _daily_source_file(
                Path(self.workspace),
                {"path": path, "sha256": sha},
                code="EXECUTION_CONTROL_CHANGED",
            )
            if current != raw:
                raise ContractError("EXECUTION_CONTROL_CHANGED")


def read_execution_controls(*, workspace: str, request_ref: dict) -> ExecutionControls:
    """Read declared refs only, before any loop construction, lock or provider call.

    Native install, policy, bootstrap/previous completion and Store admission checks
    must still run before invoking maintenance. No produced refs are discovered here.
    """
    root = Path(workspace).resolve(strict=True)
    observed = {}

    def capture(ref):
        checked = validate_ref(ref)
        _, raw, _ = _daily_source_file(root, checked, code="EXECUTION_CONTROL_SHA_INVALID")
        old = observed.get(checked["path"])
        if old is not None and old != (checked["sha256"], raw):
            raise ContractError("EXECUTION_CONTROL_CONFLICTING_REF")
        observed[checked["path"]] = (checked["sha256"], raw)
        return raw

    request = parse_canonical_json_bytes(capture(request_ref))
    request = validate_production_request(
        request, release_install_ref=request["release_install_ref"]
    )
    if request["action"] not in {"PLAN", "EXECUTE"}:
        raise ContractError("EXECUTION_CONTROL_INITIAL_ACTION_REQUIRED")
    recipe = validate_execution_recipe(
        parse_canonical_json_bytes(capture(request["recipe_ref"])), request=request
    )

    def walk(value):
        if type(value) is dict:
            if set(value) == {"path", "sha256"}:
                capture(value)
            else:
                for child in value.values():
                    walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)

    walk(recipe)
    acquisition = recipe.get("theme_acquisition_ref")
    if acquisition is not None:
        from .theme_acquisition import (
            validate_theme_acquisition_policy,
            validate_theme_policy_profile,
        )

        policy = validate_theme_acquisition_policy(parse_canonical_json_bytes(capture(acquisition)))
        validate_theme_policy_profile(recipe, policy)
    result = ExecutionControls(
        str(root),
        (request_ref["path"], request_ref["sha256"]),
        (request["recipe_ref"]["path"], request["recipe_ref"]["sha256"]),
        tuple((path, sha, raw) for path, (sha, raw) in sorted(observed.items())),
    )
    result.recheck()
    return result


def verify_execution_install_and_research_policies(controls: ExecutionControls) -> dict:
    """Native running-install/policy gate; anchor and Store checks remain required."""
    controls.recheck()
    request = controls.document(controls.request_ref)
    recipe = controls.document(controls.recipe_ref)
    if request["action"] != "EXECUTE":
        raise ContractError("EXECUTION_INSTALL_EXECUTE_REQUIRED")
    validate_production_request(request, release_install_ref=request["release_install_ref"])
    validate_execution_recipe(recipe, request=request)

    def document(ref):
        return controls.document((ref["path"], ref["sha256"]))

    result = verify_recipe_static_controls(
        workspace=controls.workspace, recipe=recipe, document=document
    )
    controls.recheck()
    return result


def verify_recipe_static_controls(*, workspace: str, recipe: dict, document) -> dict:
    """Shared installed/policy gate for an exact recipe or unmaterialized template."""
    from quant_investor.contracts import canonical_json_bytes, validate_artifact
    from quant_investor.system.store import object_ref_for_artifact
    from quant_investor.market.daily_factor_loop import read_factor_loop_context
    from quant_investor.intelligence.storage import approved_theme_policy_v2
    from .prediction_policy import validate_prediction_policy
    from .execution_recipe import SCHEMA_V5, SCHEMA_V6
    from .research_timing import validate_research_timing_policy

    if recipe["schema_version"] in {SCHEMA_V5, SCHEMA_V6}:
        timing = recipe["research_timing"]
        policy = validate_research_timing_policy(document(timing["policy_ref"]))
        if policy["mode"] != timing["mode"]:
            raise ContractError("EXECUTION_TIMING_POLICY_MODE_MISMATCH")

    ref = recipe["factor_loop_context_ref"]
    retained_context = document(ref)
    if retained_context.get("release_install_input_ref") != recipe["release_install_ref"]:
        raise ContractError("EXECUTION_CONTEXT_INSTALL_REF_MISMATCH")
    context, installation = read_factor_loop_context(
        workspace_root=workspace,
        context_path=str(Path(workspace) / ref["path"]),
        context_sha256=ref["sha256"],
    )
    if context != retained_context or installation.get("state") != "PASS":
        raise ContractError("EXECUTION_RUNNING_INSTALL_INVALID")
    _validate_producer_loop_context(
        context,
        verified_commit=document(recipe["release_install_ref"])["release_install_evidence"][
            "payload"
        ]["final_commit"],
    )
    release = validate_artifact(canonical_json_bytes(document(recipe["release_ref"])))
    if release["kind"] != "system.release" or object_ref_for_artifact(release) != installation.get(
        "release_ref"
    ):
        raise ContractError("EXECUTION_RELEASE_ARTIFACT_MISMATCH")
    if document(recipe["policy_refs"]["research"]) != approved_theme_policy_v2():
        raise ContractError("EXECUTION_RESEARCH_POLICY_INVALID")
    policy = recipe["policy_refs"]["prospective"]
    if policy is not None:
        validate_prediction_policy(document(policy), trade_date=recipe["target_trade_date"])
    return {"context": context, "installation": installation}
