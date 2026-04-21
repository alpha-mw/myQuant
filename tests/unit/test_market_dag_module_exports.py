from __future__ import annotations

from quant_investor.market.dag.assembly import (
    _aggregate_branch_summaries as assembly_aggregate_branch_summaries,
    _build_branch_results as assembly_build_branch_results,
)
from quant_investor.market import dag_executor
from quant_investor.market.dag.context import (
    _prepare_market_context as context_prepare_market_context,
)
from quant_investor.market.dag.decision import (
    _run_bayesian_selection_phase as decision_run_bayesian_selection_phase,
    _run_portfolio_construction_phase as decision_run_portfolio_construction_phase,
)
from quant_investor.market.dag.evidence import (
    _build_master_evidence_pack as evidence_build_master_evidence_pack,
    _compact_trace_fragments as evidence_compact_trace_fragments,
)
from quant_investor.market.dag.review import (
    _portfolio_master_advisory as review_portfolio_master_advisory,
)
from quant_investor.market.dag.packets import (
    _build_market_snapshot as packets_build_market_snapshot,
    _build_symbol_bundle as packets_build_symbol_bundle,
    _build_symbol_research_packet as packets_build_symbol_research_packet,
)
from quant_investor.market.dag.research import (
    _run_candidate_research_phase as research_run_candidate_research_phase,
)
from quant_investor.market.dag.reporting import (
    _build_reporting_artifacts as reporting_build_reporting_artifacts,
)
from quant_investor.market.dag.shortlist import (
    _build_shortlist as shortlist_build_shortlist,
    _build_shortlist_from_bayesian_records as shortlist_build_shortlist_from_bayesian_records,
)


def test_market_dag_helper_modules_reexport_canonical_callables():
    assert dag_executor._compact_trace_fragments is evidence_compact_trace_fragments
    assert dag_executor._build_master_evidence_pack is evidence_build_master_evidence_pack
    assert dag_executor._build_shortlist is shortlist_build_shortlist
    assert dag_executor._build_shortlist_from_bayesian_records is shortlist_build_shortlist_from_bayesian_records
    assert dag_executor._portfolio_master_advisory is review_portfolio_master_advisory
    assert dag_executor._build_market_snapshot is packets_build_market_snapshot
    assert dag_executor._build_symbol_research_packet is packets_build_symbol_research_packet
    assert dag_executor._build_symbol_bundle is packets_build_symbol_bundle
    assert dag_executor._aggregate_branch_summaries is assembly_aggregate_branch_summaries
    assert dag_executor._build_branch_results is assembly_build_branch_results
    assert dag_executor._prepare_market_context is context_prepare_market_context
    assert dag_executor._run_candidate_research_phase is research_run_candidate_research_phase
    assert dag_executor._run_bayesian_selection_phase is decision_run_bayesian_selection_phase
    assert dag_executor._run_portfolio_construction_phase is decision_run_portfolio_construction_phase
    assert dag_executor._build_reporting_artifacts is reporting_build_reporting_artifacts
