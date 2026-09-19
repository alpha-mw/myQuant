"""Bind a retained execution recipe to exact post-maintenance research inputs."""

from pathlib import PurePosixPath
import hashlib
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref
from .daily_journal import DailyJournal
from .exposure_catalog import select_exposure_rows
from .research_file_readback import ResearchFileReadback


def collect_research_input(
    *, journal: DailyJournal, recovered: dict, auxiliary: dict, verify_only: bool = False
) -> dict:
    """Caller has replayed the handoff and captured auxiliary refs before this day lock.

    Only exact PINNED refs or an explicit native stage research_source_ref may
    supply Fundamental/Macro input. Missing sources remain None and block their
    native DAG nodes; they cannot be relabeled ready or found from current heads.
    """
    if type(verify_only) is not bool:
        raise ContractError("MATERIALIZATION_VERIFY_MODE_INVALID")
    if not verify_only:
        journal._require_lock()
    workspace = str(journal.storage._io.workspace_root)
    storage = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        ref = validate_ref(ref)
        stored = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=16 * 1024 * 1024)
        if stored.byte_sha256 != ref["sha256"]:
            raise ContractError("MATERIALIZATION_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = stored.data
        return parse_canonical_json_bytes(stored.data)

    handoff = read(recovered["handoff_ref"])
    recipe = read(handoff["recipe_ref"])
    if handoff != recovered["handoff"] or recipe != recovered["recipe"]:
        raise ContractError("MATERIALIZATION_HANDOFF_CONTEXT_CHANGED")
    if handoff["trade_date"] != journal.trade_date:
        raise ContractError("MATERIALIZATION_TRADE_DATE_MISMATCH")
    execution = PurePosixPath(recovered["handoff_ref"]["path"]).parent
    expected = journal.root / "executions" / handoff["request_ref"]["sha256"]
    if execution != expected:
        raise ContractError("MATERIALIZATION_EXECUTION_PATH_INVALID")
    sources = recipe["research_sources"]
    policy = read(recipe["policy_refs"]["research"])
    if policy != approved_theme_policy_v2():
        raise ContractError("MATERIALIZATION_RESEARCH_POLICY_INVALID")
    core = read(handoff["core_handoff_ref"])
    core_nodes = core["node_refs"]
    selected = {
        name: read(core_nodes[name]) for name in ("low_observation", "w80_observation", "top100")
    }
    if any(row.get("state") != "SUCCEEDED" for row in selected.values()):
        raise ContractError("MATERIALIZATION_CORE_NODE_INCOMPLETE")
    obs = {
        "low": selected["low_observation"]["output_refs"]["LOW"],
        "w80": selected["w80_observation"]["output_refs"]["W80"],
    }
    pool = selected["top100"]["output_refs"]["manifest.json"]
    read(pool)
    for ref in obs.values():
        read(ref)
    stage_refs, missing = {}, []

    def optional(name):
        ref = sources[name]
        if ref is None:
            missing.append(name)
            return None
        return read(ref)

    def auxiliary_source(name):
        declaration = sources[name]
        if declaration["mode"] == "PINNED":
            stage_refs[name] = None
            return read(declaration["source_ref"])
        if declaration["mode"] != "MAINTENANCE_STAGE" or declaration["source_ref"] is not None:
            raise ContractError("MATERIALIZATION_SOURCE_MODE_INVALID")
        row = auxiliary["stages"][name]
        stage_refs[name] = row["ref"]
        if row["state"] == "MISSING":
            if row["ref"] is not None or row["document"] is not None:
                raise ContractError("MATERIALIZATION_MISSING_STAGE_HAS_REF")
            missing.append(name + ":NATIVE_STAGE_MISSING")
            return None
        if row["state"] != "RECORDED" or row["ref"] is None:
            raise ContractError("MATERIALIZATION_STAGE_STATE_INVALID")
        expected_stage = "FUNDAMENTAL" if name == "fundamental" else "MACRO_RELEASE"
        attempt = PurePosixPath(handoff["maintenance_core_ref"]["path"]).parent
        if row["ref"]["path"] != str(attempt / f"stage-{expected_stage}.json"):
            raise ContractError("MATERIALIZATION_STAGE_ATTEMPT_MISMATCH")
        stage = read(row["ref"])
        if stage != row["document"]:
            raise ContractError("MATERIALIZATION_STAGE_CHANGED")
        result = stage["result"]
        if result.get("stage") != expected_stage:
            raise ContractError("MATERIALIZATION_STAGE_KIND_MISMATCH")
        if result.get("status") not in {"READY", "NO_ACTION"} or result.get("blockers") != []:
            missing.append(name + ":NATIVE_STAGE_NOT_READY")
            return None
        evidence = result.get("evidence")
        ref = evidence.get("research_source_ref") if isinstance(evidence, dict) else None
        if ref is None:
            missing.append(name + ":PRODUCER_SOURCE_REF_UNAVAILABLE")
            return None
        return read(ref)

    theme_handoff_ref = None
    theme_source = None
    if recipe.get("theme_acquisition_ref") is not None:
        from .execution_recipe import SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6
        from .theme_handoff_readback import read_theme_handoff
        from .theme_acquisition import (
            validate_theme_acquisition_policy,
            POLICY_SCHEMA_V2,
            POLICY_SCHEMA_V3,
        )

        if (
            recipe["schema_version"] not in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6}
            or sources["theme_source_ref"] is not None
        ):
            raise ContractError("MATERIALIZATION_THEME_MODE_INVALID")
        acquisition = validate_theme_acquisition_policy(read(recipe["theme_acquisition_ref"]))
        filename = (
            "handoff.v3.json"
            if acquisition["schema_version"] == POLICY_SCHEMA_V3
            else (
                "handoff.v2.json"
                if acquisition["schema_version"] == POLICY_SCHEMA_V2
                else "handoff.v1.json"
            )
        )
        theme_path = str(execution / "theme-source" / filename)
        stored_theme = journal.storage.read(theme_path)
        if stored_theme is None:
            raise ContractError("MATERIALIZATION_THEME_HANDOFF_REQUIRED")
        theme_handoff_ref = {"path": theme_path, "sha256": stored_theme.byte_sha256}
        replayed_theme = read_theme_handoff(
            journal=journal,
            request_ref=handoff["request_ref"],
            core_handoff_ref=handoff["core_handoff_ref"],
            handoff_ref=theme_handoff_ref,
        )
        read(theme_handoff_ref)
        theme_source = read(replayed_theme["handoff"]["source_descriptor_ref"])
        if theme_source != replayed_theme["descriptor"]:
            raise ContractError("MATERIALIZATION_THEME_DESCRIPTOR_CHANGED")
    else:
        theme_source = optional("theme_source_ref")

    exposure_files = ResearchFileReadback(workspace)
    exposure_rows = select_exposure_rows(
        optional("exposure_rows_ref"),
        recipe=recipe,
        workspace=workspace,
        pool_ref=pool,
        theme_source=theme_source,
        files=exposure_files,
    )
    document = {
        "as_of": sources["as_of"],
        "strategy_id": recipe["strategy_id"],
        "policy": policy,
        "expected_trade_date": journal.trade_date,
        "expected_factor_pointer_sha256": handoff["factor_pointer_ref"]["sha256"],
        "industry_source": optional("industry_source_ref"),
        "theme_source": theme_source,
        "company_evidence": {
            "exposure_rows": exposure_rows,
            "fundamental_source": auxiliary_source("fundamental"),
            "macro_risk": auxiliary_source("macro"),
        },
    }
    fundamental = document["company_evidence"]["fundamental_source"]
    if fundamental is not None:
        if type(fundamental) is not dict or set(fundamental) != {
            "available_at",
            "pointer",
            "daily_parquet",
        }:
            raise ContractError("MATERIALIZATION_FUNDAMENTAL_DESCRIPTOR_INVALID")
        pointer_ref = validate_ref(fundamental["pointer"])
        snapshot_path = str(
            execution / "inputs" / f"fundamental-pointer-{pointer_ref['sha256']}.json"
        )
        saved = journal.storage.read(snapshot_path)
        if saved is not None:
            if saved.byte_sha256 != pointer_ref["sha256"]:
                raise ContractError("MATERIALIZATION_FUNDAMENTAL_SNAPSHOT_CONFLICT")
            observed[snapshot_path] = saved.data
        else:
            if verify_only:
                raise ContractError("MATERIALIZATION_FUNDAMENTAL_SNAPSHOT_MISSING")
            pointer = storage.read_workspace_file_bytes(
                pointer_ref["path"], maximum_bytes=16 * 1024 * 1024
            )
            if pointer.byte_sha256 != pointer_ref["sha256"]:
                raise ContractError("MATERIALIZATION_FUNDAMENTAL_POINTER_SHA_MISMATCH")
            observed[pointer_ref["path"]] = pointer.data
            saved = journal.storage.write(snapshot_path, pointer.data)
        document["company_evidence"]["fundamental_source"] = {
            **fundamental,
            "pointer": {"path": saved.relative_path, "sha256": saved.byte_sha256},
        }
    for alias, ref in obs.items():
        document[alias + "_observation_path"] = ref["path"]
        document[alias + "_observation_sha256"] = ref["sha256"]
    for path, raw in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=16 * 1024 * 1024).data != raw:
            raise ContractError("MATERIALIZATION_SOURCE_CHANGED_DURING_READ")
    exposure_files.recheck()
    return {
        "document": document,
        "pool_manifest_ref": pool,
        "theme_source_handoff_ref": theme_handoff_ref,
        "auxiliary_stage_refs": stage_refs,
        "missing_sources": missing,
        "native_replay_required": True,
        "execution_authorized": False,
    }


def publish_research_input(
    *, journal: DailyJournal, recovered: dict, auxiliary: dict, verify_only: bool = False
) -> dict:
    """Legacy native payload writer; new cutoff transactions first collect in memory."""
    from .execution_recipe import SCHEMA_V5, SCHEMA_V6

    if recovered["recipe"].get("schema_version") in {SCHEMA_V5, SCHEMA_V6}:
        raise ContractError("MATERIALIZATION_CUTOFF_COMMITMENT_REQUIRED")
    collected = collect_research_input(
        journal=journal, recovered=recovered, auxiliary=auxiliary, verify_only=verify_only
    )
    document = collected.pop("document")
    if document["as_of"] is None:
        raise ContractError("MATERIALIZATION_CUTOFF_COMMITMENT_REQUIRED")
    execution = PurePosixPath(recovered["handoff_ref"]["path"]).parent
    raw = canonical_json_bytes(document)
    digest = hashlib.sha256(raw).hexdigest()
    path = str(execution / "inputs" / f"research-{digest}.json")
    if verify_only:
        stored = journal.storage.read(path)
        if stored is None or stored.data != raw:
            raise ContractError("MATERIALIZATION_RESEARCH_REPLAY_MISMATCH")
    else:
        stored = journal.storage.write(path, raw)
    return {
        "research_request_ref": {"path": path, "sha256": stored.byte_sha256},
        **collected,
    }
