from __future__ import annotations

import json
from typing import Any, Mapping

from quant_investor.agent_protocol import BranchVerdict, DataQualityIssue, ModelRoleMetadata, ShortlistItem
from quant_investor.llm_gateway import estimate_message_tokens
from quant_investor.market.dag.common import _as_mapping, _dedupe_texts


MASTER_EVIDENCE_PACK_TOKEN_LIMIT = 8_000
MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT = 8
MASTER_EVIDENCE_PACK_FIELD_LIMIT = 7


def _compact_trace_fragments(
    *,
    model_roles: ModelRoleMetadata | Mapping[str, Any] | Any,
    resolver_snapshot: Mapping[str, Any],
    data_quality_issues: list[DataQualityIssue],
    shortlist: list[ShortlistItem],
) -> dict[str, Any]:
    role_payload = _as_mapping(model_roles)
    return {
        "model_roles": {
            "branch_model": str(role_payload.get("branch_model", "")),
            "master_model": str(role_payload.get("master_model", "")),
            "branch_fallback_used": bool(role_payload.get("branch_fallback_used", False)),
            "master_fallback_used": bool(role_payload.get("master_fallback_used", False)),
            "master_reasoning_effort": str(role_payload.get("master_reasoning_effort", "")),
        },
        "resolver": {
            "resolution_strategy": str(resolver_snapshot.get("resolution_strategy", "")),
            "directory_priority": list(resolver_snapshot.get("directory_priority", []) or []),
            "physical_directories_used_for_full_a": list(
                resolver_snapshot.get("physical_directories_used_for_full_a", []) or []
            )[:4],
        },
        "data_quality_issue_count": len(data_quality_issues),
        "quarantined_count": len(data_quality_issues),
        "selected_count": len(shortlist),
    }


def _build_master_evidence_pack(
    *,
    shortlist: list[ShortlistItem],
    branch_summaries: Mapping[str, BranchVerdict],
    macro_verdict: BranchVerdict,
    risk_constraints: Mapping[str, Any],
    model_roles: Mapping[str, Any],
    resolver_snapshot: Mapping[str, Any],
    data_quality_issues: list[DataQualityIssue],
    company_name_map: Mapping[str, str],
    top_k: int,
) -> dict[str, Any]:
    shortlist_limit = max(1, min(int(top_k or 1), MASTER_EVIDENCE_PACK_SHORTLIST_LIMIT))
    shortlisted_items = shortlist[:shortlist_limit]

    def _shortlist_entry(item: ShortlistItem) -> dict[str, Any]:
        posterior_meta = dict(item.metadata or {})
        return {
            "symbol": item.symbol,
            "company_name": company_name_map.get(item.symbol, ""),
            "category": item.category,
            "rank_score": round(float(item.rank_score), 4),
            "action": item.action.value if hasattr(item.action, "value") else str(item.action),
            "confidence": round(float(item.confidence), 4),
            "expected_upside": round(float(item.expected_upside), 4),
            "suggested_weight": round(float(item.suggested_weight), 4),
            "posterior_action_score": round(float(posterior_meta.get("posterior_action_score", item.rank_score)), 4),
            "posterior_win_rate": round(float(posterior_meta.get("posterior_win_rate", 0.0)), 4),
            "posterior_confidence": round(float(posterior_meta.get("posterior_confidence", item.confidence)), 4),
            "posterior_edge_after_costs": round(float(posterior_meta.get("posterior_edge_after_costs", 0.0)), 4),
            "posterior_capacity_penalty": round(float(posterior_meta.get("posterior_capacity_penalty", 0.0)), 4),
            "rationale": list(item.rationale[:3]),
            "risk_flags": list(item.risk_flags[:3]),
        }

    def _branch_entry(name: str, verdict: BranchVerdict) -> dict[str, Any]:
        return {
            "branch": name,
            "score": round(float(verdict.final_score), 4),
            "confidence": round(float(verdict.final_confidence), 4),
            "thesis": str(verdict.thesis)[:240],
            "top_risks": list(verdict.investment_risks[:3]),
            "coverage_notes": list(verdict.coverage_notes[:2]),
        }

    key_risks = _dedupe_texts(
        _dedupe_texts([str(item) for item in risk_constraints.get("risk_notes", [])])[:4]
        + [issue.message for issue in data_quality_issues[:6]]
        + [item for verdict in branch_summaries.values() for item in verdict.investment_risks[:1]]
    )[:8]

    evidence_pack: dict[str, Any] = {
        "version": "evidence_pack.v1",
        "shortlist": [_shortlist_entry(item) for item in shortlisted_items],
        "branch_consensus": {
            name: _branch_entry(name, verdict)
            for name, verdict in branch_summaries.items()
        },
        "key_risks": key_risks,
        "portfolio_constraints": {
            "gross_exposure_cap": float(risk_constraints.get("gross_exposure_cap", 0.0) or 0.0),
            "target_exposure_cap": float(risk_constraints.get("target_exposure_cap", 0.0) or 0.0),
            "max_weight": float(risk_constraints.get("max_weight", 0.0) or 0.0),
            "action_cap": str(risk_constraints.get("action_cap", "hold")),
            "blocked_symbols": list(risk_constraints.get("blocked_symbols", [])[:10]),
            "selected_count": len(shortlisted_items),
        },
        "macro_summary": {
            "symbol": str(macro_verdict.symbol or ""),
            "score": round(float(macro_verdict.final_score), 4),
            "confidence": round(float(macro_verdict.final_confidence), 4),
            "thesis": str(macro_verdict.thesis)[:240],
            "regime": str(macro_verdict.metadata.get("regime", "neutral")),
        },
        "trace_fragments": _compact_trace_fragments(
            model_roles=model_roles,
            resolver_snapshot=resolver_snapshot,
            data_quality_issues=data_quality_issues,
            shortlist=shortlisted_items,
        ),
    }

    evidence_pack["trace_fragments"]["budget"] = {
        "field_limit": MASTER_EVIDENCE_PACK_FIELD_LIMIT,
        "token_limit": MASTER_EVIDENCE_PACK_TOKEN_LIMIT,
        "field_count": len(evidence_pack),
        "shortlist_count": len(shortlisted_items),
        "company_name_coverage": sum(1 for item in shortlisted_items if company_name_map.get(item.symbol, "")),
    }

    def _token_count(payload: dict[str, Any]) -> int:
        return estimate_message_tokens(
            [
                {"role": "system", "content": "evidence-pack"},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )

    token_count = _token_count(evidence_pack)
    if token_count <= MASTER_EVIDENCE_PACK_TOKEN_LIMIT:
        evidence_pack["trace_fragments"]["budget"]["token_count"] = token_count
        evidence_pack["trace_fragments"]["budget"]["truncated"] = False
        return evidence_pack

    trimmed = dict(evidence_pack)
    trimmed_shortlist = [_shortlist_entry(item) for item in shortlisted_items[: max(1, shortlist_limit // 2)]]
    trimmed["shortlist"] = trimmed_shortlist
    trimmed["branch_consensus"] = {
        name: {
            "branch": name,
            "score": round(float(verdict.final_score), 4),
            "confidence": round(float(verdict.final_confidence), 4),
            "thesis": str(verdict.thesis)[:120],
        }
        for name, verdict in branch_summaries.items()
    }
    trimmed["key_risks"] = key_risks[:5]
    trimmed["trace_fragments"] = {
        "model_roles": evidence_pack["trace_fragments"]["model_roles"],
        "resolver": evidence_pack["trace_fragments"]["resolver"],
        "data_quality_issue_count": len(data_quality_issues),
        "selected_count": len(trimmed_shortlist),
        "budget": {
            "field_limit": MASTER_EVIDENCE_PACK_FIELD_LIMIT,
            "token_limit": MASTER_EVIDENCE_PACK_TOKEN_LIMIT,
            "field_count": len(trimmed),
            "shortlist_count": len(trimmed_shortlist),
            "company_name_coverage": sum(1 for item in trimmed_shortlist if item.get("company_name")),
        },
    }
    token_count = _token_count(trimmed)
    trimmed["trace_fragments"]["budget"]["token_count"] = token_count
    trimmed["trace_fragments"]["budget"]["truncated"] = True
    return trimmed
