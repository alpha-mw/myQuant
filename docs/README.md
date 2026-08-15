# Documentation Index

Factor Governance, Intelligence, and Mainline now share one stable public
runtime. Numeric runtime names and split command surfaces are retired.

## Start here

- [Unified module map](modules/module_map.md)
- [Factor governance](factor_governance.md)
- [Cutover command mapping](migrations/unified-cutover/cli-mapping.md)
- [Cutover safety and recovery](migrations/unified-cutover/README.md)
- [Replacement-test map](migrations/unified-cutover/replacement-test-map.md)
- [CN Fundamental safe-successor operations](runbooks/cn_fundamental_safe_successor.md)

## Independent domains

- [Macro risk reference](modules/macro_risk_reference.md) documents a manual,
  non-authoritative helper.
- [Trading discipline](trading_discipline.md) remains Strategy Record Store
  policy and does not acquire System-pointer authority.
- [CN weekly portfolio review automation](cn_weekly_review_automation.md) —
  Store-v3 evidence, weekly narrative inputs, formal-advisory gate, and
  decision-log boundary
- [CN aggressive Dashboard contract](../portfolio_dashboard/README.md) —
  registered current/previous holdings and canonical performance closure

Public result commands resolve exactly one generation through
`results/system/_active.json`. They never scan for a latest run, substitute a
research candidate, or create authority during a read. Missing, blocked, or
suspended state is returned as canonical JSON and remains fail closed.
