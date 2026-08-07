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
- [Legacy configuration cleanup and migration](runbooks/v17_legacy_configuration_cleanup.md)

## Supporting contracts

- [Research pipeline and protocols](architecture/research_pipeline_and_protocols.md)
- [Factor Governance v4](factor_governance_v4.md)
- [Evidence archive boundary](architecture/evidence_archive_boundary.md) — why
  `factors/governance*` is 32% of the package, unreachable from every entry
  point, and still must not be deleted, moved or edited
- [Module map](modules/module_map.md)

## Standalone and historical references

- [Macro risk reference](modules/macro_risk_reference.md) documents a manual
  legacy helper. It is not current V17 mainline authority.

Public result commands resolve one exact strategy pointer below
`results/v17_mainline/`. They never scan result directories or substitute a
Shadow session. Missing authority is an unavailable result, not permission to
create one.

The repository has no public production publisher or activation command. I0
and R2.2 are explicit research-only consumers: the I0 regime model is a causal,
one-step Market/Industry/Theme Markov filter, and `research-evaluate` is an
offline stdout-only evaluator rather than a daily scheduler or memory writer.
