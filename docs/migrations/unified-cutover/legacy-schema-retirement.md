# Legacy Version Schema Retirement

This plan separates callable runtime, active control-plane language, mutable
forward contracts, immutable historical evidence, and migration guards.  A
text match is not sufficient reason to rewrite or delete evidence.

## Current classification

- Callable version-named runtime files: zero.
- Active daily and weekly automation prompts: version-neutral.
- Weekly exporter/checker and runbook: version-neutral unified
  System/Mainline projection.
- Mutable source references remaining: nine `quant_investor` files and five
  scripts.  They contain three different classes:
  - current identity or provider schema identifiers;
  - negative-authority fields retained for Store/Dashboard compatibility;
  - migration and zero-call guards that prove retired imports stay absent.
- Immutable `results/` matches: audit evidence.  They are never rewritten to
  make a search count reach zero.

## Retirement phases

### Phase 1 — user-facing and control-plane cleanup (complete)

- Remove version labels from active automation prompts, weekly domains,
  blockers, permissions, and report fields.
- Preserve exact historical bytes and retired-import guards.

### Phase 2 — neutral forward-write fields

- Add version-neutral authority fields to new Store, Dashboard, performance,
  and decision-log artifacts.
- Readers accept the historical negative-authority field only on immutable
  pre-cutover artifacts; new writers emit only the neutral field.
- Require exact fixtures for old-read/new-write behavior before deployment.

### Phase 3 — owner identity declaration

- Prepare a new owner-approved version-neutral identity declaration and bind
  its exact SHA to all current consumers.
- Cut over by explicit declaration/pointer transaction.  Do not mutate the
  existing declaration or reinterpret its protocol string.

### Phase 4 — provider evidence schemas

- Introduce neutral schema IDs for newly produced Fundamental provider
  evidence while keeping strict readers for historical IDs.
- Migrate producers first, then consumers, then remove compatibility reads
  after no active pointer references the old schema.

### Phase 5 — guard consolidation

- Keep one explicit migration manifest and negative-import test proving retired
  entrypoints remain absent.
- Remove redundant human-facing references only after phases 2–4 have no active
  dependencies.  Never delete immutable results merely to reduce grep counts.

## Stop conditions

- Stop before any owner identity, Store pointer, performance generation,
  Fundamental pointer, or immutable result mutation without its separately
  approved transaction.
- Stop if a proposed cleanup changes the meaning of `authority=false`, permits
  a retired import, or makes historical bytes unverifiable.
