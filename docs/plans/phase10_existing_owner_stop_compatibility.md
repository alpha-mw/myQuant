# Phase 10: read the existing owner-revalidated legacy stop

Status: Architect exact-event amendment accepted; Critic APPROVE;
IMPLEMENTED_LOCAL_VALIDATION. No owner policy mutation.

Current-workspace combined validation: 97 tests passed in 19.64 seconds, covering
the two row profiles, native synthetic threshold projection, prior Morning
behavior, v3 receipts and consumers. The actual owner policy's two rows now parse;
SHA `11eb7018ff6abde2d276c7b997e0e61e2cd6b170407872734bb8d407b334c178`
and original mtime are unchanged. Scoped Black, flake8 and mypy passed. Evidence:
`.agent/acceptance/phase10-existing-owner-stop-final-tests.log` and
`.agent/acceptance/phase10-existing-owner-stop-after.json`.

## Reproduced problem

The actual `initial-risk-stop.v1` policy at
`results/policies/risk/aggressive_tech_manufacturing/initial-risk-stop.v1/owner-stop-policy-20260828-v1.json`
contains both an ordinary `CONFIRMED` row and a
`CONFIRMED_OWNER_REVALIDATION` legacy row. The current
`validate_initial_stop_policy` rejects the latter with
`MORNING_STOP_POLICY_ROW_INVALID`, preventing the whole Morning policy read.
The synthetic Morning fixture only contains the ordinary row.

The legacy row explicitly declares no effective entry date and no executable
trailing anchor. Its owner-confirmed fixed stop is an independent policy.
Accepting its declared shape must not create an entry date, trailing threshold,
new stop price, financial state, or execution permission.

## Scoped implementation

Only change the shared initial-stop reader in
`quant_investor/strategy_records/risk_policy_contract.py` and focused Morning
tests/fixtures. Preserve the existing policy-level authority, Store binding,
owner-confirmation/effective clocks and trigger checks.

Split row validation into two explicit, closed field profiles, selected by the
existing `initial_stop_state`. The ordinary `CONFIRMED` profile stays unchanged.
The legacy profile requires exactly the current row's common fields plus
`entry_anchor_state`, `legacy_stop_revalidated`, `trailing_stop_state`,
`retired_unexecutable_trailing_stop_cny` and
`sina_price_cny_at_20260828_143135`; it has no `calibration` field.
Require `CONFIRMED_OWNER_REVALIDATION`, `OWNER_EXPLICIT_CONFIRMATION`,
`effective_entry_trade_date=null`, `UNAVAILABLE_LEGACY_POSITION`, boolean true
revalidation, and `UNAVAILABLE_MISSING_EFFECTIVE_ENTRY_ANCHOR`. Require the exact
event string `LEGACY_POSITION_OWNER_STOP_REVALIDATION_20260828`, not a suffix
match or a generalized future event grammar. Validate finite positive diagnostic prices.
The two historical diagnostic prices remain unused metadata. Reject unknown,
mixed or incomplete profiles, invented dates, enabled authorities and mismatched
trigger amounts/quantities.

The existing Morning stop-window code already derives availability from owner
policy clocks and validates unchanged position, native close history and
corporate changes independently of entry anchors. Leave those semantics intact.
The trailing lane continues to require its own valid anchor and reconciliation.
If focused tests reveal a dependent reader assumption, amend this plan before
changing another runtime module.

## Acceptance and limits

- An actual-file read validates both rows and leaves policy SHA/mtime unchanged.
- Synthetic native Morning replay with a legacy fixed stop and no trailing
  anchor produces an independently bound fixed stop while trailing remains
  unavailable. It never uses the retired trailing number or quote metadata.
- Before-effective/late-confirmed policy, changed shares/cost, missing close
  history and corporate changes retain their existing stop blockers.
- Malformed/mixed profile, non-null legacy entry date, false revalidation,
  unknown fields, nonfinite metadata and enabled automatic execution reject.
- A different event string with the same `_20260828` suffix rejects.
- The actual legacy row selects only its owner fixed-stop price `35.32`;
  diagnostic values `163.51` and `43.47` cannot become any threshold, comparison
  input or trailing anchor. Exercise native synthetic Morning replay separately;
  a real full-EOD Morning proof cannot be invented while the actual EOD is absent.
- Existing ordinary-policy and Morning receipt/consumer tests remain passing.
- Focused format/lint/type checks pass. No real Morning execution, quote/provider
  call, policy write, broker operation or installed/unattended claim.

Implementation was prepared and tested in an isolated copy while the Phase15
baseline suite ran. Its 914 source/build/test files were verified unchanged at
termination before integrating the three owned files. The baseline suite completed
with 4532 passed, 7 failed and 3 skipped; its separate fixture/install failures
and the full Phase7/12/13/15 goal are not closed by this focused reader repair.
