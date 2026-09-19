# Phase 4: explicit PCB / AI hardware evidence in the daily path

The source design requires DC-primary, registered-TDX-fallback daily evidence
that distinguishes membership, economic exposure, confidence and source evidence,
and explicitly covers 002463.SZ 沪电股份 and 002384.SZ 东山精密. Current sources
are scoped only to the ranked Top100, so these two companies can disappear from
daily coverage. Their inclusion in a focus report is an evidence requirement,
not a claim that either stock is in Top100 or qualifies for investment.

## Existing path and authority

Extend the existing acquisition → Theme → Industry/Exposure source adapters.
No new factor, rank, weight, strategy, automation, trade, portfolio state or
external provider is introduced. Keep native project_tushare_theme_source,
derive_tdx_fallback_company_keyset and source-bound exposure validation
authoritative. Local verification uses native synthetic capture fixtures and
zero live calls. Existing historical receipts remain immutable.

The ranked pool remains exactly 100 companies. A separate code-owned focus scope
is the sorted unique set [002384.SZ, 002463.SZ]. Source acquisition may query that
scope even when a company is outside the ranked pool; it cannot insert those rows
into rank/selected_symbols or the ordinary pool membership projection.

## Explicit source and acquisition versions

Add an exact descriptor `cn-daily-theme-evidence-source.v2` with only:

- schema_version;
- pool: the existing exact six-ref DC/TDX source descriptor;
- pcb_ai_hardware: a separate exact six-ref descriptor, or null for explicitly
  missing supplied focus evidence.

Both scopes are validated through the same native DC/registered-fallback reader,
with exact company set, trade date, cutoff, partition order and byte references.
The ordinary projection still contains only Top100. A v2 descriptor always
creates an explicit two-company focus result, including rows for missing inputs.
Legacy six-ref descriptors remain readable as legacy evidence and do not prove
completion of the new Phase4 requirement. The final deployed daily configuration
must select the new contract; final acceptance cannot count legacy coverage.

Add acquisition policy `cn-daily-theme-acquisition.v2`, preserving existing
provider_priority, fallback_mode and maximum_companies=100 fields and adding
special_company_keyset with the exact two-company set. The same existing producer
captures pool DC/fallback and focus DC/fallback under one day claim. Its existing
claim identity already binds the exact acquisition policy SHA and pool source
refs; preserve that authority and prove both requested scopes before calls.
Keep v1 passive/historical readback. V2 never exceeds 100 pool + 2 focus company
partitions per provider, with fallback still restricted to native derived sets.

Add a versioned Theme handoff for the extra scope, retaining the existing strict
source path/ref checks. The v2 handoff binds the focus company set and exact focus
DC/TDX plan/capture/partition refs; source_descriptor_ref names the wrapper above.
Keep v1 handoff interpretation unchanged. Captures remain in the same execution
root with separate code-owned focus prefixes, not another discovery/latest path.
Replay reconstructs both scopes and requires exact unchanged source bytes.

## Evidence outputs

The Theme node publishes its unchanged ordinary membership artifact plus a new
registered `pcb_ai_hardware_membership` artifact for a v2 descriptor. That artifact
has exactly the two required company rows, the focus labels PCB / AI_HARDWARE as
requested research topics, exact native membership values/provider, cutoff/date,
source refs, missing codes and a completion_state of SUCCEEDED or
PARTIAL_WITH_EXPLICIT_MISSING. Topic labels are not assertions of membership.
Do not persist a second ordinary theme projection with the same logical artifact
ID; the focus artifact has its own kind and identity bound to scope and sources.

At the existing Exposure node, publish a registered `pcb_ai_hardware_evidence`
artifact with exact two-company rows and four separate fields:

- membership: exact native focus membership, including provider and membership
  status, without converting concept membership to revenue exposure;
- economic_exposure: existing source-backed exposure state and qualified facts;
- confidence: deterministic evidence-completeness category with reasons, not an
  invented probability or investment conviction;
- source_evidence: original source refs, source types/pages/times and missing
  items, never references fabricated from a company name or topic label.

Use the existing full Industry source descriptor to project the two companies
independently of the ordinary Top100 industry output. Use existing exposure rows
and source validators for those companies. Missing industry, absent financial
sources or unavailable membership remain explicit. Reuse the existing source
facts and qualification rules; do not introduce hand-written revenue claims.
The report has SUCCEEDED or PARTIAL_WITH_EXPLICIT_MISSING and exact missing codes.
The node maps the latter to the existing PARTIAL state and INPUT_MISSING taxonomy;
it does not claim a successful full evidence chain from missing focus evidence.

Research source recipes carry the v2 focus requirements only when declared by the
source wrapper/policy, and bind them to immutable inputs/upstream terminal refs.
Keep legacy recipe reconstruction exact for historical completed EODs. The
completion replay regenerates both new artifacts from the recorded inputs and
compares complete output ref maps. Decision retains its native research states;
new evidence cannot grant trades, portfolio admission or owner threshold changes.

## Verification and acceptance

Architect then Critic review is required for these public source/artifact shapes.
Acceptance must exercise the existing acquisition and daily adapter entrypoints,
not only an isolated report builder:

1. Both companies appear exactly once when outside Top100; rank/selection bytes
   and ordinary membership company set remain unchanged.
2. DC primary success never consults TDX for the same company; only native
   missing-partition fallback and registered aliases can enter focus membership.
   No dual-source voting or unregistered alias admission.
3. Native capture/date/partition/SHA drift, wrong focus company set, future
   evidence and invalid exposure sources reject without false success.
4. Positive native source-backed facts yield separate membership/exposure/source
   confidence fields. Missing and partially available inputs retain both rows,
   precise missing codes and PARTIAL_WITH_EXPLICIT_MISSING; no invented values.
5. Real Theme/Exposure source adapters and completed-source replay bind all new
   artifacts; changing one focus ref invalidates its dependent proof.
6. Same policy/request/source replay makes zero provider/writer calls; old v1
   source/handoff/recipe fixtures remain readable and are not counted as Phase4
   coverage. Installed final acceptance must explicitly use the v2 path.
7. Focused native Theme/fallback/exposure/acquisition/handoff/adapter regressions
   and changed-file checks; no repeated whole-suite runs for this phase alone.

Historical catch-up's separate Dashboard failure is retained for its own routing
and recovery work: it reproduced DASHBOARD_STALE:benchmark under a current-page
intent. This phase must not rewrite that immutable request or change its selector.

## Accepted Architect amendments

### PIT and overlap

Resolve the recorded Factor generation's `market_pit_selection_ref`, then its
exact `pit_generation_manifest_file_ref` and `pit_membership_file_ref`. These file
refs are logical capture aliases (for example pit/canonical-membership.parquet),
not workspace paths. Resolve them only through the owning Factor source bundle,
`_mirrored_source_resolver` and native deep Market/PIT selection replay. Verify
all SHA bindings against the same observations/core generation used by Top100.
A bounded read-only owner helper may expose the exact retained manifest/membership
bytes and physical mirror refs after native validation; it cannot acquire, copy,
publish or consult heads. Preserve the exact source-object/mirror metadata chain.
Decode the validated canonical retained PIT bytes with the native record schema,
row-count/records-SHA checks and PITUniverseRecord parser, then call
`filter_symbols_by_pit_status(..., required=True)` for the two codes. Never join
logical aliases to workspace, depend on original capture paths, call latest-record
or current-PIT lookup, or replace native byte/record checks with a new permissive
parser. The three explicit PIT refs identify actual retained physical bytes,
not unresolvable logical aliases.
Preserve each native listing-status result, including missing/pre-listing/delisted
or outside-frozen-scope reasons. A missing row remains in both focus artifacts
with FOCUS_PIT_NOT_ELIGIBLE and the original PIT reason. A missing/corrupt PIT
file or wrong SHA rejects the binding before provider calls.

V2 acquisition binding explicitly carries pit_selection_ref,
pit_generation_manifest_ref and pit_membership_ref alongside both company sets
and their hashes. These are derived from the verified core; the existing claim
still binds their immutable ancestry through core_handoff_ref and policy SHA.

Always capture the full two-company focus scope separately under fixed focus
prefixes. If a company is also in Top100, compare the two native membership rows
on provider, status, theme_ids and technology_theme_ids. Matching rows are
reported with both exact source chains. A mismatch becomes explicit
FOCUS_POOL_MEMBERSHIP_CONFLICT, with neither row admitted as reconciled focus
membership. Preserve raw source facts; no union, voting or stronger-source pick.

### Exact handoff dispatch

V2 uses `handoff.v2.json` selected from the validated acquisition-policy schema;
v1 uses its existing exact path. Never probe/fall back between versions.
FIELDS_V2 is exactly the existing v1 fields plus:
special_company_keyset, special_company_set_sha256, pit_selection_ref,
pit_generation_manifest_ref, pit_membership_ref, special_dc_plan_ref,
special_dc_capture_ref, special_dc_partition_refs, special_tdx_plan_ref,
special_tdx_capture_ref, special_tdx_partition_refs.
The existing company_keyset/company_set_sha256 continue to mean Top100 only.
Pool source prefixes remain dc/tdx; focus prefixes are special-dc/special-tdx.
For each scope validate exact native DC and independently derived TDX fallback
company sets, partition ordering/count, paths, time bounds, and registered aliases.
V1 rejects extra v2 fields; v2 requires every extra field. Source-wrapper fields
remain exactly schema_version, pool, pcb_ai_hardware; non-null nested descriptors
have exactly the existing six fields. Unknown versions reject.

### Exact focus artifact schemas

Both new artifact kinds use the common research-only inactive envelope. Their
payloads additionally contain: evidence_id, as_of, trade_date, focus_topics,
company_set_sha256, pool_manifest_ref, pit_selection_ref,
pit_generation_manifest_ref, pit_membership_ref, source_refs, company_rows,
completion_state, missing_codes. Identity binds kind, date/cutoff, exact scope,
policy/source refs and the original pool ref. focus_topics is exactly
[PCB, AI_HARDWARE], and company_rows is sorted by canonical company_code with
exactly 002384.SZ and 002463.SZ. Names are the user-provided display labels only.

Membership row fields are exactly company_code, company_name, in_top100,
pit_status, membership, source_evidence, missing_codes. pit_status is the native
listing-status object; membership is either the exact validated native company
row (company_code/provider/status/theme_ids/technology_theme_ids) or null when
not admitted. source_evidence has exactly membership_rows, membership_refs,
overlap_membership_rows and overlap_membership_refs. Row lists contain only the
validated native company row from the corresponding scope (or are empty); ref
lists contain sorted unique exact {path, sha256} native descriptor input refs.
This retains raw facts when a PIT or overlap gate prevents admission.

Final evidence row fields are exactly company_code, company_name, in_top100,
pit_status, membership, industry, economic_exposure, confidence, source_evidence,
missing_codes. industry retains the native industry company row or null.
economic_exposure retains the native exposure row and qualified company evidence
refs, or null if no native derivation can be admitted. HIGH/MEDIUM/LOW/UNVERIFIED
remain the original native categories; never infer them from focus topics.
source_evidence is an exact object with membership_refs, industry_refs,
company_evidence_refs and unqualified_company_evidence_refs, each a sorted unique
list. Membership/industry refs are exact {path, sha256} input-file refs; company
evidence refs are exact native five-field artifact refs. Global source_refs is
the sorted unique list of physical {path, sha256} inputs. Raw company facts can remain in the last
list without granting qualified exposure.

confidence is exactly {category, reason_codes}. Categories are finite:
COMPLETE_SOURCE_BOUND, MEMBERSHIP_ONLY, SOURCE_ONLY, MISSING. COMPLETE_SOURCE_BOUND
requires eligible PIT, admitted native membership, available native industry and
qualified native exposure with no native blockers. Otherwise use MEMBERSHIP_ONLY
when membership is admitted, SOURCE_ONLY when only validated source facts exist,
and MISSING when neither is available. reason_codes is the sorted union of row
missing codes and original native blocker/reason codes. It is never a probability.

Code-owned missing codes are FOCUS_SOURCE_MISSING, FOCUS_PIT_NOT_ELIGIBLE,
FOCUS_MEMBERSHIP_UNAVAILABLE, FOCUS_POOL_MEMBERSHIP_CONFLICT,
FOCUS_INDUSTRY_SOURCE_MISSING, FOCUS_INDUSTRY_UNAVAILABLE,
FOCUS_EXPOSURE_SOURCE_MISSING and FOCUS_EXPOSURE_NOT_QUALIFIED. Global missing_codes
are sorted strings CODE:COMPANY_CODE, exactly derived from the rows. Unknown codes
reject. Preserve original native PIT, membership and exposure reasons separately.
completion_state is SUCCEEDED iff no row missing codes remain, otherwise
PARTIAL_WITH_EXPLICIT_MISSING. Missing focus inputs never drop a company row.

### Actual DAG and replay lifecycle

For a v2 wrapper, the Theme node always emits its ordinary artifact plus the
focus membership artifact. Its lifecycle follows the ordinary Top100 projection;
focus missingness is visible in the focus artifact, allowing Exposure to run.
The Exposure node always emits its ordinary projection/evidence followed by the
focus evidence artifact and required qualified/unqualified source artifacts.
Its lifecycle is PARTIAL/INPUT_MISSING if either ordinary exposure or focus
evidence is partial. Thus full EOD is blocked while the requested focus report
is still created. No missing focus result can silently count as full success.

Use explicit named output refs `pcb_ai_hardware_membership` on Theme and
`pcb_ai_hardware_evidence` on Exposure, while preserving existing ordinary output
names. Capture artifact ordering is deterministic: ordinary primary first,
ordinary source artifacts in native order, focus report, then distinct focus
source artifacts sorted by (kind, artifact_id, byte SHA). Replay verifies the
entire map and both new kinds. Existing immutable v1 captures remain unchanged.

The v2 materialized source recipe binds pool/PIT refs, focus wrapper and needed
Industry/exposure sources so the report can be regenerated from its exact
original inputs. Update completed-source recipe reconstruction and allowed kinds.
Existing journal revision gates remain authoritative: no migration/relabeling of
a successful legacy terminal, running attempt, or old day claim. An incompatible
same-day request conflicts; only existing permitted linked input revisions may
advance. Decision compilation consumes the ordinary Top100 projections only.

For v2 input, first validate all exposure rows and reject duplicate companies or
companies outside the union of Top100 and focus. Then partition the validated
evidence by company scope before calling native builders. Ordinary Exposure and
Decision must not receive out-of-pool focus facts, because even unused extra refs
would change native evidence identities. Overlapping companies reuse the exact
same validated company-evidence artifacts. Deduplicate equal artifact refs before
capture publication; conflicting refs for one semantic identity reject. Preserve
legacy input behavior when reconstructing original v1 recipes.

Additional required negatives: missing/out-of-scope PIT row; wrong PIT manifest
or membership SHA; focus/Top100 overlap matching and conflicting memberships;
Theme emits a missing focus row while Exposure still runs and emits a partial
focus report; unqualified financial facts never create HIGH/MEDIUM/LOW.

## Implementation precision

Source-node captures recognize the two newly registered kinds and named refs.
Decision's separate `ResearchCapture.ALLOWED_KINDS` intentionally remains
unchanged: focus-only artifacts must not enter ordinary Decision compilation.
The journal's research-only input-revision allowlist includes the two kinds with
explicitly false inactivity/authority checks. This satisfies source capture
support without widening Decision's artifact contract.

Materialization, EOD and ledger callers use a bound version validator that reads
the exact acquisition policy bytes/SHA. The pure shape/version validator retains
legacy behavior unless given the validated policy. The bound path selects one
handoff version and rejects both wrong-version refs and policy-byte drift; it
does not accept both filenames as interchangeable alternatives.

## Local acceptance

COMPLETED_LOCAL_VALIDATION. V2 acquisition, strict PIT/source binding, handoff,
materialization, two-company reports, partial-data lifecycle, deterministic
output names, input revisions, source timing and completed-source readers are
integrated. Native 100-company source-adapter cases passed for both complete and
missing focus data. 142 focused regressions passed (two already-passed slow cases
excluded); 10 focused checks passed after CI complexity extraction. Exact source
hashes, test scopes and limitations are in `.agent/acceptance/phase4-validation.json`.

The final installed whole-DAG run and repository-wide CI are not inferred from
these component checks. No scheduler, provider connection, actual holdings,
trades or existing immutable production receipts were changed.
