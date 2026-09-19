"""Exact bootstrap launcher transport and retained-content identities."""

import hashlib
from pathlib import PurePosixPath

from quant_investor.contracts import parse_canonical_json_bytes
from .bootstrap import validate_bootstrap_declaration
from .catchup_binding import BindingSources
from .daily_contract import ContractError, validate_ref
from .execution_recipe import SCHEMA_V5, validate_execution_recipe
from .production_request import SCHEMA, validate_production_request


def validate_bootstrap_launch_profile(request, recipe):
    validate_production_request(request, release_install_ref=request["release_install_ref"])
    validate_execution_recipe(recipe, request=request)
    if (
        request["schema_version"] != SCHEMA
        or request["action"] != "EXECUTE"
        or recipe["schema_version"] != SCHEMA_V5
        or recipe["bootstrap_ref"] is None
        or recipe["previous_completion_ref"] is not None
    ):
        raise ContractError("BOOTSTRAP_LAUNCH_PROFILE_REQUIRED")
    return recipe


def execution_paths(request_ref, request):
    ref = validate_ref(request_ref)
    root = PurePosixPath("results/operations/daily_production/CN") / request["target_trade_date"]
    execution = root / "executions" / ref["sha256"]
    recipe = validate_ref(request["recipe_ref"])
    return {
        "execution": str(execution),
        "handoff": str(execution / "maintenance-handoff.v1.json"),
        "cutoff": str(execution / "research-cutoff.v1.json"),
        "request_ref": {
            "path": str(execution / "inputs" / f"request-{ref['sha256']}.json"),
            "sha256": ref["sha256"],
        },
        "recipe_ref": {
            "path": str(execution / "inputs" / f"recipe-{recipe['sha256']}.json"),
            "sha256": recipe["sha256"],
        },
    }


def validate_retained_bootstrap_identity(
    *,
    request_ref,
    request_raw,
    recipe_raw,
    handoff_ref,
    handoff,
    retained_request_raw,
    retained_recipe_raw,
    declaration,
):
    request = parse_canonical_json_bytes(request_raw)
    recipe = parse_canonical_json_bytes(recipe_raw)
    validate_bootstrap_launch_profile(request, recipe)
    validate_bootstrap_declaration(declaration, recipe=recipe)
    paths = execution_paths(request_ref, request)
    if (
        hashlib.sha256(request_raw).hexdigest() != request_ref["sha256"]
        or hashlib.sha256(recipe_raw).hexdigest() != request["recipe_ref"]["sha256"]
        or retained_request_raw != request_raw
        or retained_recipe_raw != recipe_raw
        or validate_ref(handoff_ref)["path"] != paths["handoff"]
        or handoff["request_ref"] != paths["request_ref"]
        or handoff["recipe_ref"] != paths["recipe_ref"]
        or handoff["trade_date"] != request["target_trade_date"]
        or handoff["release_install_ref"] != request["release_install_ref"]
        or handoff["release_ref"] != recipe["release_ref"]
    ):
        raise ContractError("BOOTSTRAP_RETAINED_IDENTITY_MISMATCH")
    return paths


class BootstrapLaunchInputs:
    def __init__(self, *, workspace, request_ref, release_install_ref):
        self.source = BindingSources(workspace)
        self.workspace = workspace
        self.request_ref = validate_ref(request_ref)
        self.request_raw = self.source.raw(self.request_ref)
        self.request = validate_production_request(
            parse_canonical_json_bytes(self.request_raw), release_install_ref=release_install_ref
        )
        self.recipe_raw = self.source.raw(self.request["recipe_ref"])
        self.recipe = validate_bootstrap_launch_profile(
            self.request, parse_canonical_json_bytes(self.recipe_raw)
        )
        self.declaration = validate_bootstrap_declaration(
            self.source.document(self.recipe["bootstrap_ref"]), recipe=self.recipe
        )
        self.paths = execution_paths(self.request_ref, self.request)
        self.day = self.request["target_trade_date"]
        self.source.recheck()

    def bind(self, recovered):
        handoff = recovered["handoff"]
        validate_retained_bootstrap_identity(
            request_ref=self.request_ref,
            request_raw=self.request_raw,
            recipe_raw=self.recipe_raw,
            handoff_ref=recovered["handoff_ref"],
            handoff=handoff,
            retained_request_raw=self.source.raw(handoff["request_ref"]),
            retained_recipe_raw=self.source.raw(handoff["recipe_ref"]),
            declaration=self.declaration,
        )
        if recovered["request"] != self.request or recovered["recipe"] != self.recipe:
            raise ContractError("BOOTSTRAP_RETAINED_DOCUMENT_MISMATCH")
        self.source.recheck()

    def recheck(self):
        self.source.recheck()
