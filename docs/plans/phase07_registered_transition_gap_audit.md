# Phase 7 registered financial transition reachability

Status: REVISE after two bounded read-only Architect audits on 2026-09-13.
Implementation has not started. Phase7 and the full Goal remain incomplete.

## Confirmed gaps

`daily_store_adoption` only adopts a native batch-close transaction. A registered
manual/applied/funding state cannot reach that adapter through the current source,
preparation and corporate contracts.

If Store has already advanced when `source_slot_inputs.build_plan` runs, its
pointer is accepted as the preimage, but a missing target Event still selects
`CREATE_CURRENT_EMPTY`. A registered nonempty state must not be declared empty.
If Store advances after that plan, `daily_source_inputs._source_mode` correctly
rejects the changed pointer. The source route is not deployed; resolve this gap
before deployment, including a negative test against false empty publication.

Other empty-only assumptions occur in `daily_preparation._native_sources`,
`research_corporate_inputs.derive_corporate_inputs`,
`daily_store_materialization.verify_initial_store_controls`, `portfolio_binding`
and `StoreCloseAdapter`. Merely adding an adapter branch is insufficient.

## Native capabilities and constraints

The native `command_seal_publish` remains the financial registration/CAS owner.
It checks direct predecessor, effective Parquet ledger, financial state and
performance. Funding requires its exact supplement and cash-flow declaration.
Funding corrections remain unsupported by that publisher. Its applied-trade
validator is not a complete arbitrary fill/fee/cash-bridge contract.

Phase8 can validate one isolated corporate application against registered
before/after records. It rejects mixed or unattributed deltas. Generic manual
changes and arbitrary historical trade formats do not have an exact owning
contract and cannot be admitted implicitly.

A registered applied record may still have `official_valuation=false`. Such an
intraday state is not Store READY. Adoption cannot invent the missing official
close or alter holdings/cash. Any integration plan must distinguish this case
from a fully valued registered state and test both.

## Required integration direction

Use a versioned sibling registered-transition binding, preserving Event v1's
empty-only contract. The previous EOD's exact retained Store pointer provides
T-1 custody. A supported direct T successor must bind original pointer bytes,
catalog, lineage, record inventory, strict-close valuation and performance.
Unknown ancestry, missing original custody, changed preimages, multiple
unaccounted transitions and a simultaneous T `CLOSED_EMPTY` must block.

Carry the mutually exclusive empty/registered evidence choice through source
plan, preparation, recipe, cutoff, corporate gate, portfolio and completed
readback. The Decision book remains T-1. The Store node only reads and adopts
the registered T financial state, with zero additional financial CAS calls.
Replay after a later head must use frozen refs exclusively.

Before implementation, specify exact supported trade/funding/corporate shapes,
closed input schemas, source timing, original pointer custody, native strict
valuation proof and all affected version dispatchers. Review that concrete plan
with Architect then Critic. This audit is not implementation approval or an
assertion that unsupported manual formats satisfy the full Phase7 requirement.

## Follow-up: registered intraday source needs official valuation

Current native catalog readback found two registered applied records, both with
`official_valuation=false`: `20260810_1625` and `20260820_1321`. Their recorded dates
are not proof of a completed official close. An already-official-only adoption
profile is therefore insufficient for the ordinary owner-reported-fill flow.

The follow-up Architect recommendation is to reuse the existing writer for
`previous EOD -> registered intraday source -> official T close`. The existing
`build_record` already accepts the modern
`owner_declared_manual_execution_applied` predecessor and preserves its shares,
average cost, cost basis and cash; it changes valuation only. The older
`owner_declared_actual_filled` profile remains unsupported.

A concrete implementation plan must provide:

- A versioned batch v2 registered-source profile and a separate transition
  binding; preserve ordinary batch v1 and its empty-only receipt semantics.
- Exact T-1 Decision baseline custody separate from the registered writer
  source. Retain both original pointer byte sequences before final Store CAS.
- A closed modern owner-trade row validator covering quantities, price, fees,
  value, cash bridge and pre/post ledger identity. Do not infer unreported
  corporate/manual/funding dimensions merely from missing fields.
- Explicit seven-domain evidence/authority for the source choice, with original
  registration and capture times. Historical missing declarations remain blocked.
- A narrowly validated same-date official finalization mode in the existing
  performance helper. Replace the intraday performance row with official close,
  preserve units and already-applied external flows, and retain both records in
  Store lineage. Do not relabel it as a correction.
- Source/preparation/cutoff/corporate/portfolio/adapter/replay dispatch for the
  mutually exclusive profiles, and proof of both pointer transitions. The
  Decision book remains T-1; the financial writer starts from the already-applied
  source. No fill application, order generation or new holdings writer.

The first implementation can reject unsupported/mixed/multiple-hop inputs, but
must disclose those limits. Funding and isolated corporate posting remain part of
the full Phase7 goal and need their owning exact contracts. This architecture
recommendation does not yet approve a concrete schema or authorize real execution.
