# Phase 13: source and preparation failure metadata

Status: bounded design draft, not implemented or reviewed.

The source/preparation inventory contains 103 literal reason codes across eight
files (`.agent/acceptance/phase13-source-preparation-reason-inventory.json`).
Their CLI boundaries preserve exact blocker codes and useful missing-date refs,
but omit the existing retryable, owner-action and recommended-node fields.
The DAG itself already attaches that metadata to its node failures.

## Intended smallest repair

Add a pure, code-owned, explicit reason-to-failure/next-node registry for these
two preparation entry points. Compose the existing `daily_contract.failure`
metadata into their expected error JSON as an additive `failure` field. Preserve
the top-level status/blocker code and all existing safe date/ref fields, return
codes, successful result shapes, source identities, request modes and authority.
Do not modify retry loops, consume another request marker, select another request,
alter failure custody or call a provider as part of classification.

The registry must enumerate exact codes. Do not classify runtime exception text
with substrings or prefix heuristics. Missing immutable custody, exhausted source
budgets and unconfirmed historical owner facts must not be described as ordinary
retryable source acquisition. Unknown expected input failures retain the existing
non-retrying generic diagnosis. Unexpected exceptions remain exit3/internal error
and must not disclose their text, paths or credentials.

Calendar, Store, Factor and Theme suggestions must refer to existing graph nodes;
there is no invented Event/benchmark node. Document that the suggestion is an
inspection/resolution target, not an automatic permission or dispatch instruction.

## Required acceptance

- Every inventoried source/preparation reason has one reviewed exact mapping.
- Real local CLI failures preserve their blocker code and expected exit while
  adding schema-valid metadata: missing historical facts, changed Store pointer,
  source SHA mismatch, foreign ACTIVE request, late Calendar and exhausted budget.
- Missing/invalid install and native-source failures stay fail closed; classify
  only from an explicit typed owning boundary, never raw exception text.
- Source and preparation success/recovery/no-provider shell paths are unchanged.
- No metadata argument can replace the original status, blocker code or authority.
- Existing journal RUNNING/POST_WRITE_IN_DOUBT and completed replay remain intact.
- CLI malformed-argument/shared global error handling is a separate existing
  interface; this change must not silently rewrite every command in the project.

Before implementation, attach the complete explicit mapping and inspect every
affected public error boundary, then obtain Architect followed by Critic review.
This draft does not by itself certify the remaining Phase13 producer coverage.
