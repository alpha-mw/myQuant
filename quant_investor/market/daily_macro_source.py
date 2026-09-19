"""Publish the existing research descriptor from one exact native Macro terminal."""

import hashlib
from pathlib import Path

from quant_investor.contracts import canonical_json_bytes
from quant_investor.macro import readiness_closure as native
from quant_investor.macro.maintenance_transaction import _postcheck


def attach_macro_source(context, layout, result: dict) -> dict:
    """Source failure never retries the transaction or changes component health."""
    from .daily_maintenance import _write_once

    if context.mode != "execute" or result["status"] not in {"READY", "NO_ACTION"}:
        return result
    result = {**result, "evidence": dict(result["evidence"])}
    evidence = result["evidence"]
    if layout.state == "LEGACY":
        evidence["research_source_blockers"] = ["MACRO_LEGACY_TERMINAL_NOT_ADMISSIBLE"]
        return result
    root = context.workspace_root.resolve(strict=True)
    terminal = layout.journal_root / layout.journal_id / "0007-terminal.json"
    try:
        relative = terminal.relative_to(root)
        raw = native._read(root, relative, code="MACRO_SOURCE_TERMINAL_UNAVAILABLE")
        sha = hashlib.sha256(raw).hexdigest()
        closure = native.build_macro_readiness_closure(
            workspace_root=root, terminal_path=relative.as_posix(), terminal_sha256=sha
        )
        if closure["target_date"] != context.target_date:
            raise native.MacroReadinessClosureError("MACRO_SOURCE_TARGET_DIFFERS")

        def recheck_heads():
            for row in closure["frozen_pointers"].values():
                native._read(
                    root,
                    Path(row["current_path"]),
                    code="MACRO_SOURCE_CURRENT_HEAD_DIFFERS",
                    expected_sha256=row["frozen_ref"]["sha256"],
                )

        recheck_heads()
        pointers = closure["frozen_pointers"]
        _postcheck(
            *[
                {
                    "canonical_root": root / Path(pointers[name]["current_path"]).parent,
                    "new_pointer_sha256": pointers[name]["frozen_ref"]["sha256"],
                    "generation_id": pointers[name]["generation_id"],
                }
                for name in ("release", "observations")
            ]
        )
        sealed = native.seal_macro_readiness_closure(
            workspace_root=root, terminal_path=relative.as_posix(), terminal_sha256=sha
        )
        if sealed["closure"] != closure:
            raise native.MacroReadinessClosureError("MACRO_SOURCE_CHANGED_DURING_SEAL")
        recheck_heads()
        descriptor = {
            "classification": "CANONICAL_MACRO_READY",
            "source": {"path": sealed["closure_path"], "sha256": sealed["closure_sha256"]},
        }
        path = context.attempt_root / "macro-research-source.json"
        digest = _write_once(path, canonical_json_bytes(descriptor))
        evidence["research_source_ref"] = {"path": str(path.relative_to(root)), "sha256": digest}
        evidence["research_evidence_write_performed"] = True
    except Exception:
        evidence.pop("research_source_ref", None)
        evidence["research_source_blockers"] = ["MACRO_RESEARCH_SOURCE_PUBLICATION_FAILED"]
        evidence["research_evidence_write_status"] = "UNCONFIRMED"
    return result
