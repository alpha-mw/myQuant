# Theme Policy Event JSONL

Theme Policy Catalyst reads local JSONL policy-event caches only. The first
maintained production path is `data/theme_policy_events.jsonl`; the template is
`data/theme_policy_events.example.jsonl`.

This layer is a ThemeScanner sidecar. It can add policy catalyst metadata and a
capped theme score component when explicitly enabled, but it does not create
candidate pools, branches, Bayesian likelihoods, RiskGuard decisions, portfolio
weights, or buy/sell instructions.

## Schema

Each line is one JSON object.

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `event_id` | yes | string | Stable unique id across the file. Duplicate ids are validation errors. |
| `title` | yes | string | Short policy title. |
| `issuer` | yes | string | Issuing body, such as `工业和信息化部`. |
| `publish_date` | yes | string | `YYYY-MM-DD` or `YYYYMMDD`. |
| `effective_date` | no | string | Same date formats; scanner falls back to `publish_date`. |
| `policy_level` | no | enum | Authority bucket. Unknown values are warnings. |
| `policy_type` | no | enum | Event type. Unknown values are warnings. |
| `theme_tags` | conditional | array[string] | At least one of `theme_tags`, `industry_tags`, or `symbol_tags` must be non-empty. |
| `industry_tags` | conditional | array[string] | Use local `industry_map` names where possible. |
| `symbol_tags` | conditional | array[string] | Optional A-share symbols such as `688012.SH`; used only for beneficiary clarity. |
| `evidence_text` | no | string | Compact local summary. Empty text is a warning. |
| `source_url` | no | string | Source pointer or local cache id. Empty value is a warning; it is not fetched. |

Valid `policy_level` values:

```text
central
ministry
local
association
exchange
other
```

Valid `policy_type` values:

```text
plan
notice
subsidy
standard
pilot
procurement
consultation
project_list
funding
tax
other
```

## Maintenance

1. Copy one line from `data/theme_policy_events.example.jsonl`.
2. Assign a stable `event_id`.
3. Keep tags aligned with local theme or industry names. Prefer explicit
   `theme_tags` plus `symbol_tags` when the policy beneficiaries are clear.
4. Keep `evidence_text` short and factual. Do not write trade instructions.
5. Store source context as a local cache pointer or URL string in `source_url`.
   The validator and scanner do not open it.
6. Run local validation before using the file:

```bash
./.venv/bin/python -m pytest tests/unit/test_theme_policy_validation.py -v
```

For ad hoc validation from Python:

```python
from quant_investor.themes.policy_validation import validate_policy_event_jsonl

issues = validate_policy_event_jsonl("data/theme_policy_events.jsonl")
for issue in issues:
    print(issue.to_dict())
```

## Enabling In Review

Policy catalyst is default-off. For a local paper or diagnostic review:

```bash
THEME_POLICY_CATALYST_ENABLED=1
THEME_POLICY_EVENT_PATH=data/theme_policy_events.jsonl
```

`THEME_POLICY_CATALYST_WEIGHT` caps the maximum score component. The default is
`0.16`, so the policy component can add at most 16 theme-score points.

## Risk Boundaries

- No external policy websites are called.
- No LLM is used.
- Policy text does not directly generate buy, sell, switch, or weight advice.
- Policy events do not create official `candidate_pool` entries.
- Policy events do not add a canonical branch.
- Policy events do not add `theme_likelihood`.
- Bayesian posterior math, RiskGuard, PortfolioConstructor, and v13 DAG
  execution remain unchanged.
