# Injection Surface Failure Polarity

QuantInvestor treats external narrative text as risk input, not authority.
LLM summaries, news snippets, and review-layer notes may enter RiskGuard through
`risk_texts`, where they can only add risk reasons, trigger conservative veto
keywords, or tighten exposure through existing deterministic controls.

This is intentional failure polarity: a malicious or noisy text input may cause
a false negative, such as an unnecessary veto or lower exposure, but it must not
create a false positive, bypass a hard veto, loosen deterministic constraints,
raise target weights, or override `RiskGuard -> ICCoordinator ->
PortfolioConstructor`.

Operationally:

- Keep LLM and news-derived text advisory and auditable.
- Route ambiguous or untrusted text into conservative risk fields only.
- Do not let text injection write branch scores, target weights, tradability, or
  risk limits.
- Treat any future path that turns narrative text into permissive portfolio
  action as a security regression.
