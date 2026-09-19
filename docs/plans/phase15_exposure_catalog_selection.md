# Bind a pre-Core exposure catalog to the native research cohort

The installed configured run at
`/private/tmp/myquant-configured-clock-20260913T192045Z` completed native
Factor/LOW/W80/Top100, maintenance and Theme acquisition, then failed cutoff
preview with `focus exposure union contains an unknown or duplicate company`.
Its fixture supplied3000 declarations as a plain list. The native list contract
deliberately accepts only Top100 plus the fixed focus companies. Keep that
rejection and preserve the failed immutable request/receipts.

The intended configured entry must exist before the day's Top100. Add an explicit
versioned source representation instead of silently reinterpreting existing lists
or reducing the supplied economic evidence to make the test pass:

```
{"schema_version":"cn-daily-exposure-catalog.v1","rows":[<existing declaration>, ...]}
```

Bounded implementation ownership:

- New `operations/exposure_catalog.py` owns exact catalog shape, row identity
  validation, and deterministic selection from an independently native-verified
  original Top100 manifest/rank. It may read exact original refs, never write or
  scan latest results. For this v5/v6 profile the verified modern focus descriptor is mandatory;
  a non-focus/legacy descriptor rejects. The selected set is native Top100 union
  the existing fixed FOCUS_COMPANIES. No caller-selected company set grants authority.
- `research_materialization.collect_research_input` recognizes the new catalog
  only for current recipe v5/v6, after its exact native Top100/Theme bindings are
  available, and derives ordinary native exposure rows. None and legacy list
  inputs retain their exact existing values and strict downstream behavior.
- `research_source_bundle.SourceBundle._verify_origins` independently repeats
  the same catalog-to-native-cohort derivation from the original catalog ref and
  original pool ref, and compares exact selected rows to frozen native fields.
  Original full catalog bytes/SHA and all native pool read refs participate in
  existing custody rechecks. No unversioned field or profile reinterpretation.
- All final native builders, `partition_exposure_evidence`, optional-missing
  policy, source SHA/time checks, financial state and OOS exclusion stay unchanged.
  The emitted native request contains selected declarations only; its existing
  source-time derivation therefore covers exactly the used native evidence.

Catalog admission: reject extra envelope keys, wrong versions/non-list rows,
invalid company/ref/row shape, and duplicates anywhere in the catalog. Validate
the existing declaration field grammar using existing native primitives; selected
rows still undergo full native semantic/physical/time validation. Unselected
source files are not read or claimed as verified evidence. Unselected future
declarations cannot affect the selected report or create an earlier availability
claim. Missing selected-company facts retain the existing explicit missing policy;
no rows are invented. Sort selected rows by canonical company code. The actual
Top100 native manifest/rank, release/Factor binding and focus PIT remain authoritative.

Verification before another full installed run:

1. Native projection tests with a broad catalog and genuine native pool/source
   fixtures: exact Top100/focus selection, overlap once, deterministic order,
   duplicate/malformed/unknown-version refusal, selected missing/SHA/future refusal,
   catalog and pool drift refusal, and unchanged strict legacy-list rejection.
2. Both collection and frozen source-bundle replay must use the same derivation;
   exercise an altered selected row/cohort/source-ref as a negative, not just
   helper unit tests. Preserve original source inventory and timestamps.
3. Run focused materialization/cutoff/Exposure/registered and timing tests plus
   applicable static checks. Use retained D native inputs for a clearly labeled
   read-only candidate preview where feasible; it is not an installed EOD proof.
4. Change only the synthetic config's input document to the explicit catalog.
   Build/verify a current isolated release and exercise the full configured entry,
   four same-config successors, five-day replay, Morning and boundary cases.
   The old release/request cannot be relabeled as having run the new code.

Architect then Critic review is required for this explicit source-schema extension.
No producer/provider, financial mutation, production install or schedule cutover
is authorized by this plan itself. The overall Phase0–15 acceptance remains open.

## Architect refinements accepted

Architect APPROVE_WITH_CHANGES: implement exactly the current v5/v6 focus profile,
not a new Top100-only catalog route. Reuse native pool verification with the exact
manifest-named rank and its factor/policy/observation bindings, and the existing
Theme/focus PIT derivation in BOTH collection and SourceBundle replay. The source
functions must enrol native manifest/rank and Theme/focus refs in their existing
byte rechecks, not merely extract company codes from JSON.

Full catalog validation is structural/declarative only: exact existing row keys,
canonical company and available_at, native ref/path grammar, approved source type,
page grammar and finite admissible metric value, plus global uniqueness. It may
not open any row source file or call _daily_exposure_evidence. Only the selected
rows reach that unchanged native builder for physical, SHA, semantic and future
time admission. Entire catalog bytes remain an original retained input; changing
an unused row changes that input SHA and must invalidate replay. Its unused
physical source path/time must never be presented as verified evidence.

The catalog envelope is administrative custody, with no aggregate fact timestamp.
Current cutoff is sampled after catalog and selected source reads; selected native
availability and physical custody remain authoritative. Historical/late/synthetic
custody stays non-prospective. No late fact becomes an earlier OOS observation.

Keep runtime changes to exposure_catalog.py, research_materialization.py and
research_source_bundle.py. Do not change native projections, ResearchSources,
partition_exposure_evidence, focus builders or optional completion policy. Static
pre-Core source UI/schema validation is not required by this bounded repair.

## Implementation checkpoint

Architect refinements accepted; Critic APPROVE. The three runtime files now
implement the catalog route and independent replay. Catalog grammar/profile and
cohort failures reuse the existing dependency failure taxonomy without expanding
its authority. Existing native projection/union/optional-completion code is
unchanged. Current v5/v6 collector+source-bundle, selected missing/SHA/future,
unused future-file exclusion, catalog/pool drift and unchanged legacy-list checks
passed37tests/79.84s. Earlier related62tests passed202.07s; scopes overlap. Native
retained-D candidate selection verified3000->102 (100 pool+2 focus), with no unused
source reads or workspace writes. Independent native Theme/Exposure projections
for both cohorts are READY. These are local/component previews, not acceptance of
D's failed original request. New isolated release bf078d3235de3ade1379e8082d5d4a49e109f7d5
installed/origin verification passed; its configured native run is pending.
