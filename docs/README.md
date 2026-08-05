# Documentation Index

V17 v4 is the repository's only supported decision mainline. These pages
describe the active-pointer contract, operator workflow, strict data boundary,
and the separate Shadow forward-evidence lane.

## Start here

- [V17 v4 mainline contract](architecture/v17_v4_production_research_contract.md)
- [V17 v4 operations](runbooks/v17_v4_operations.md)
- [Entrypoints and versioning](architecture/entrypoints_and_versioning.md)
- [Forward-evidence runtime](architecture/v17_v4_forward_evidence_runtime.md)
- [Investment Intelligence I0](architecture/v17_i0_investment_intelligence.md)
- [Forward Research Evaluator R2.2](architecture/v17_r22_forward_research_evaluator.md)

## Supporting contracts

- [Research pipeline and protocols](architecture/research_pipeline_and_protocols.md)
- [Factor Governance v4](factor_governance_v4.md)
- [Module map](modules/module_map.md)
- [Macro risk reference](modules/macro_risk_reference.md)

Public result commands resolve one exact strategy pointer below
`results/v17_mainline/`. They never scan result directories or substitute a
Shadow session. Missing authority is an unavailable result, not permission to
create one.
