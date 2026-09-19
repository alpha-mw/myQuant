"""Immutable Theme source handoff publication inside the existing day lock."""

from datetime import datetime, timezone
from quant_investor.contracts import canonical_json_bytes
from .daily_contract import GRAPH_SHA256
from .daily_journal import FALSE_AUTHORITY
from .theme_handoff import SCHEMA_V2, SCHEMA_V3, PIT_FIELDS, handoff_layout
from .theme_handoff_readback import read_theme_handoff
from .theme_core_binding import bind_theme_acquisition
from quant_investor.intelligence.theme_sources import split_theme_source


def publish_theme_handoff(*, journal, request_ref: dict, core_handoff_ref: dict) -> dict:
    from .theme_capture_stage import _capture_validated_theme

    journal._require_lock()
    prefix = journal.root / "executions" / request_ref["sha256"] / "theme-source"
    binding = bind_theme_acquisition(
        workspace=str(journal.storage._io.workspace_root),
        request_ref=request_ref,
        core_handoff_ref=core_handoff_ref,
    )
    schema, _, filename = handoff_layout(binding)
    path = str(prefix / filename)
    existing = journal.storage.read(path)
    if existing is not None:
        ref = {"path": path, "sha256": existing.byte_sha256}
        read_theme_handoff(
            journal=journal,
            request_ref=request_ref,
            core_handoff_ref=core_handoff_ref,
            handoff_ref=ref,
        )
        return ref
    captured = _capture_validated_theme(
        journal=journal,
        request_ref=request_ref,
        core_handoff_ref=core_handoff_ref,
        binding=binding,
    )
    binding, descriptor = captured["binding"], captured["descriptor"]
    source = journal.storage.write(str(prefix / "source.json"), canonical_json_bytes(descriptor))
    value = {
        "schema_version": schema,
        "trade_date": journal.trade_date,
        "graph_sha256": GRAPH_SHA256,
        **binding["identity"],
        "claim_ref": captured["claim_ref"],
        "company_keyset": binding["company_keyset"],
        "source_descriptor_ref": {"path": source.relative_path, "sha256": source.byte_sha256},
        "sealed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": FALSE_AUTHORITY,
    }
    pool_descriptor, focus_descriptor, v2 = split_theme_source(descriptor)
    if v2 != (schema in {SCHEMA_V2, SCHEMA_V3}):
        raise ValueError("Theme handoff source version changed")
    scopes = [("", pool_descriptor)]
    if v2:
        value.update(
            {
                name: binding[name]
                for name in (
                    "special_company_keyset",
                    "special_company_set_sha256",
                    *PIT_FIELDS,
                )
            }
        )
        scopes.append(("special_", focus_descriptor))
    for scope, sources in scopes:
        for label in ("dc", "tdx"):
            for suffix, key in (
                ("plan_ref", "plan"),
                ("capture_ref", "capture"),
                ("partition_refs", "partitions"),
            ):
                value[scope + label + "_" + suffix] = sources[label + "_" + key]
    stored = journal.storage.write(path, canonical_json_bytes(value))
    ref = {"path": path, "sha256": stored.byte_sha256}
    read_theme_handoff(
        journal=journal, request_ref=request_ref, core_handoff_ref=core_handoff_ref, handoff_ref=ref
    )
    return ref
