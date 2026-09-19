# Phase 7: adopt an already committed native daily close

Status: Architect amendments accepted; Critic APPROVE; COMPLETED_LOCAL_VALIDATION.

Final current-source verification: 58 focused native Store/materialization/
portfolio/Decision-recipe tests passed, including 16 adoption cases. Coverage
includes corrupt-source, symlink, partial-legacy, pre-commit custody, corrupt
pending recovery and complete-legacy read-only compatibility. Three older mocked
adapter fixtures fail during setup with missing `derived_serving_root`; all three
also fail on untouched frozen 3c46 source before any adoption logic, independently
verified with cache/bytecode writing disabled. They remain a separate fixture debt.
Six source files passed mypy; seven files passed Black; scoped flake8 passed with
the repository's 100-character limit, with fatal-error lint on the native batch
script. The current-source isolated native reproduction now returns PLAN_ADOPTED,
preserves original outputs and every business byte/mtime, and forbids all current
head/CAS/recovery calls during replay. Pending recovery now validates any retained
source before proceeding, and cannot recover incomplete legacy custody after
the head advanced. Exact receipts are
`.agent/acceptance/phase7-existing-close-validation.json` and
`.agent/acceptance/phase7-already-closed-after.json`. This closes only the native
batch adoption defect; broader Phase7/8 and whole installed-path acceptance remain.

## Demonstrated failure and scope

`.agent/acceptance/phase7-already-closed-before.json` executes a real isolated
native close, then invokes the intended planner again with current preimages.
The native result is NO_ACTION with no plan reference, while the materializer
requires PLAN_PREPARED. A fully valid existing daily state therefore cannot enter
new DAG materialization. The Store pointer remains unchanged in the reproduction.

Repair this existing native batch-transaction path. Do not synthesize an old
plan, reconstruct an absent old pointer, relabel nonempty events as empty, rewrite
financial records, or change any actual holdings/cash/thresholds. Adoption of
other manual/financial record schemas remains a separately audited requirement;
this plan cannot certify that broader scope from native no-action tests.

## Preserve native pre-close pointer custody

When the existing native close planner writes/reads a prepared plan under its
operation lock, retain the exact current source pointer bytes at the transaction's
fixed `source-pointer.v1.json`. Its SHA must equal the plan's existing
preimages.store_pointer_sha256 and its catalog/active closure must pass the native
Store reader. The plan already binds this SHA; do not change old plan fields,
input fingerprints or receipt hashes. Write exact canonical bytes once; conflict
rejects. Do this before any financial staging or Store CAS. Prepare-only and
execute share this path; read-only planning must not write it.

A legacy plan lacking source-pointer bytes can only gain them while its exact
original source pointer is still selected. After that head advances, no pointer
is reconstructed. An already retained Phase6 portfolio source also remains valid.

## Exact existing-close adoption

The code-owned Store planner wrapper keeps normal PLAN_PREPARED behavior. On a
native NO_ACTION, consider adoption only when the requested/current Market date
is exactly the official Store date and the selected native catalog has exactly
one batch-close receipt for its active record. Use that receipt's transaction ID
to derive the existing fixed plan path; no directory/mtime/latest scan.

Read and validate the original native plan, then call the owning
inspect_close_commit with its exact SHA, original source-pointer SHA and target.
This proves committed pointer/catalog/performance/receipt/holdings ancestry.
Require the proof's pointer SHA to equal the caller's current Store preimage,
requested target and active record to match, and every other original preimage
(Market, benchmark, event, Calendar, policy and retrospective declaration) to match
the caller exactly. A same-date source change is a conflict/restatement issue,
not a permit to reuse the old valuation. Missing/malformed/native-unknown receipts
or incomplete commit proof reject; no new business writer runs.

Return internal status PLAN_ADOPTED with the exact original plan ref, plan and
validated commit proof, plus its mandatory fixed source-pointer ref. This is
an internal script composition result, not a new public authority or command.
The existing StoreCloseAdapter can probe the original committed plan and expose
its original outputs without executing another close/CAS.

## Materialization and Decision source integration

For normal preparation, all existing recipe/plan preimage equality checks remain.
For PLAN_ADOPTED, the only permitted mismatch is the Store source preimage:
original plan source SHA versus the execution recipe's observed committed SHA.
The exact native commit proof must explain this relation. Other preimages remain
identical. The loaded Store arguments derive from the original immutable plan,
so its existing adapter validates the original committed transaction.

Keep store_plan_ref pointing to that original native plan. Existing native-input
v3 and outer materialization shapes need no extra caller-supplied mode; replay
re-derives adoption from the two bound Store SHAs and validates the same native
commit proof. Update materialization readback to validate this relation explicitly
rather than relying on the historical assumption that preparation did it. Unknown
mismatches reject. Valid legacy equal-preimage materializations remain unchanged.

Phase6 portfolio binding must still use the original pre-close state, never the
already closed T state. If its journal state already exists, replay it. Otherwise
copy and validate the native transaction's exact source-pointer bytes. An optional
internal retained-pointer input to freeze_portfolio_state may be supplied only
by this validated adoption path and must equal the original plan preimage SHA.
Its actual custody time is new/late, not backdated. Missing historical source
pointer evidence blocks the dependent Decision/materialization claim.

Materialized held-security price refs should derive from that same verified
portfolio state rather than selecting the now-advanced current Store head.
Preserve the existing strict frozen Market path/SHA reader and held-symbol set.
Both ordinary and adoption paths use one source book. No financial calculations
or mutations are added by materialization.

On replay after another day's head advancement, validate the original native
retained committed pointer, original plan and retained source book; never reread
a mutable current Store pointer. The recipe's old current-pointer path is an
identity/expected-SHA declaration, not a fresh source selection.

## Verification and stop conditions

- Native close then repeat planner: PLAN_ADOPTED and exact original plan/commit
  refs; zero second CAS, no changed business bytes/mtimes.
- Real materializer produces v3 inputs and pre-close portfolio for an already
  closed native day, preserving the original holding/cash/cost identities and
  current closed state. Existing current-day state never replaces T-1 context.
- Later-head replay and crash after source-pointer retention remain idempotent;
  missing retained old source rejects without reconstruction.
- Wrong target, changed Market/benchmark/event/Calendar/policy, ambiguous receipt,
  wrong plan/SHA/checkpoint, incomplete commit, unknown receipt schema and missing
  old preimage fail closed.
- Normal fresh daily close, five no-action sessions, multi-day backlog, crash
  adoption, v1/v2/v3 materialization and Decision portfolio regressions pass.
- No actual account operation, provider call, scheduler/deployment or owner-policy
  mutation during verification. Full nonempty event/other-record adoption and
  named corporate-action reconciliation remain open; final whole-DAG acceptance
  is Phase15.


## Accepted Architect precision

Initial adoption and later replay are different trust operations. Initial
PLAN_ADOPTED must validate currently registered ancestry via inspect_close_commit.
A new owning frozen commit validator shares the native plan/completion/catalog/
receipt/holdings/performance logic but requires the fixed source-pointer,
committed-pointer and completion files and never calls _assert_registered_ancestor
or current.v1.json. It verifies source native catalog/active record, committed
previous_pointer_sha256=source SHA, completion.pointer_sha256=committed SHA,
requested target and all original plan bindings. It rechecks exact input bytes.

PLAN_ADOPTED always requires the complete three-file custody set and returns the
source ref. Missing old custody after the head advances cannot be repaired by
reconstruction. Initial adoption requires existing fully complete metadata and
never calls a recovery writer. All other preimages remain equal to the caller;
only source Store SHA to proved committed SHA can differ.

Adapter probing validates a fully retained strict commit before looking at the
current Store head. Invalid strict proof never falls through to execution or
recovery. Legacy pre-custody completed transactions remain readable only through
the existing registered-commit check when no adopted binding is in use; that
branch is read-only and cannot return safe_to_execute, recover metadata, create
an adopted materialization or invoke CAS. Incomplete legacy metadata fails closed.
New or replayed adopted materializations always revalidate strict proof, so deleting
source custody cannot downgrade their behavior to this legacy reader. Every new
transaction must retain source bytes before financial execution; failure blocks it.

No native plan-version change is needed: the exact adopted materialization
relationship between its recipe's observed Store SHA and its original plan's
source SHA is the durable discriminator. Materialization readback must enforce
this before accepting/reusing its loaded context or invoking native adapters.
Ordinary equality uses the existing path. Adoption inequality must equal the
strict frozen proof's committed SHA. Any third relation fails.

Chronology is explicit: plan.effective_at <= completion.cas_observed_at <= actual
adoption/materialization custody. Portfolio custody is actual new time, not the
old close time. Existing late/non-prospective classification is retained.
Materialized held-symbol Market refs derive from the same frozen pre-close
portfolio state. The original Store output refs remain identical.

Add focused tests for source custody deletion after adoption, partial legacy
metadata, committed proof before current-head lookup, later-head absence, exact
output identity and all non-Store preimage mismatches. A missing source may not
be restored from today's holdings or from a guessed old pointer.
