# V17 v4 Sanitized Daily Research Loop

The sanitized daily loop is an additive, research-only scheduler built directly
on the Forward Research Release provisional runtime. It accepts one explicit
provisional request path and byte SHA, runs `run-provisional-forward`, and
seals three content-addressed outputs: a daily receipt, a research-memory
entry, and (only after a complete forward manifest exists) an experiment
registry entry.

The loop never discovers `latest`, invokes a provider, selects a factor or
universe, writes Factor Governance, or creates Source Truth. Its label-maturity
helper recognizes only the fixed 1/5/10/20/60 horizons and requires an explicit
calendar reference plus the exact future-session sequence before reporting a
label as matured. Missing future closure remains `PENDING`; malformed or
unbound closure is `BLOCKED`. Historical backfill is always false.

Research memory and experiment registration are immutable observations, not
recommendations, weights, tiers, lifecycle actions, production evidence, or
trading instructions. A partial forward failure preserves already published
upstream refs and records `RUN_PARTIAL`; it never promotes them to a complete
experiment. A failure before any upstream artifact records `RUN_BLOCKED`.

This runtime was reimplemented on the Forward Release lineage. It does not
merge or depend on the Source Truth, Authority Root, Census, Security Identity,
Security Master, Trusted Pin, Tushare capture, or historical daily-loop
lineages.

Authority remains fixed:

```text
default_protocol_state = V15_DEFAULT
global_activation_state = INACTIVE
research_only = true
research_runtime_default = false
factor_governance_write = false
provider_calls = false
execution = false
broker = false
order = false
trade = false
```
