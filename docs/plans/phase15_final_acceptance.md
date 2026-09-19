# Final acceptance against the original Phase 0–15 design

Audit date: 2026-09-14. Goal remains incomplete.
Authority: `daily_evidence_dag_chat_design.md`, especially the original final
execution instruction and the ten Phase15 cases. Earlier receipts remain evidence
for their exact source versions/scopes; they are not current whole-path proof.

The original objective is reliable connection of existing components. It forbids
new factors/weights/strategies, broker/real-trade or actual holdings changes,
owner-stop changes and weaker SHA/PIT/Calendar checks. Broad future financial
writer functionality must not be added solely because an earlier continuation
listed it. Existing trade/fill/cash/corporate inputs still need correct native
owner handling or an explicit evidence-backed refusal; unsupported actual inputs
cannot be silently ignored to pass daily continuity.

## Required acceptance evidence

| Original requirement | Existing authoritative evidence | Remaining acceptance |
|---|---|---|
| Phase0 inventory and precise original break reason | Architecture and dated baseline; phase0 audit | Keep dated baseline unchanged; final runtime state is separate |
| Phase1–2 nine states, metadata, dependency/identity gates | Current journal/runner/resolver; focused phase1/2 receipts | Current full unit/static gates and native integrated node-state readback |
| Phase3 Factor/LOW/W80 -> idempotent Top100 Parquet/manifest | Real native pool tests and old installed full days | Current installed automatic missing-Top100 and interrupted-Top100 cases |
| Phase4 native DC/registered TDX, membership/exposure/source distinctions and two focus companies | Source adapters, phase4 native component cases | Current installed full source chain and report refs |
| Phase5 PIT-valid low-frequency data, lag warnings, critical missing blocks | Four-state native Fundamental/Macro readers and phase5 cases | Full Decision continues with legitimate lag; stale/missing is explicit |
| Phase6 research-only Decision bound to eight evidence domains | Decision report v2, portfolio custody and native compiler cases | Current full-DAG output and immutable refs, no trade authority |
| Phase7 official Store continuity, no-trade state, existing financial inputs | Native five-day/backlog/CAS and registered BUY custody/recovery cases | Current full-DAG Store dates/invariants; integrate actual relevant inputs through their owners |
| Phase8 corporate types and unresolved moving thresholds | Native reconciliation/report/risk source tests | Full valuation may complete while unresolved moving threshold stays non-executable |
| Phase9 completed-DAG Dashboard/date/expiry | Current Python/Node/HTTP UI and registered Dashboard component proofs | Current installed end-to-end publication; file-browser policy limitation remains explicit |
| Phase10 quote-first Morning, previous EOD only, no maintenance | Native risk/quote/source/receipt tests and old installed Morning | Current installed consumer after completed EOD, producers forbidden |
| Phase11 ordered gap recovery/idempotence | Real native financial cases; automatic/origin/cross-day coordinator tests | Current installed ordered missing-date interruption/recovery, no skipped days |
| Phase12 unified producer/fallback/Morning automation | Configured launcher/source preparation and shell/CLI tests | Real verified installation/configuration and exact automation migration/readback within authorization |
| Phase13 at least the named taxonomy and blocker metadata | Taxonomy/resolver/research boundary mappings | Audit every required category and metadata in reachable failures; do not require unrelated exhaustive provider mappings without a demonstrated gap |
| Phase14 native timestamps, synthetic/backfilled exclusion and evaluator use | Native ledger/OOS admission and current evaluator tests | Current integrated ledger and evaluator readback; never label synthetic as prospective/live |
| Phase15 ten integration cases and at least synthetic five-session continuity | Older 3c46 five-session and eaa single-day native proofs | Fresh current-runtime installed proof; exact source comparison and final full gates |

Phase15 cases are: normal all-node success; Factor success with missing Top100;
Top100 crash/resume without rerunning Factor; missing prior Store date; no trades
still creates daily Store; unresolved corporate action keeps moving threshold
non-executable; legitimate Fundamental lag only warns; Market/PIT SHA mismatch
fails closed; repeat same day is idempotent; closed day produces no false financial
state. The fifth-session check also needs the subsequent Morning consumer.

The final source instruction explicitly requires at least a synthetic uninterrupted
five-trading-day demonstration. Its earlier operational Done Definition describes
five automatically completed trading days without manual repairs, stale pointers,
fake backfills or silent failures. A clarification about whether additional real
unattended days are required is pending; complete all independent technical gates
now, retain both evidence categories, and do not claim real days from synthetic runs.

## Execution order

1. Freeze current code/tests in an isolated local repository and verify its real
   installed release. Run current full unit/static gates against that immutable
   snapshot; preserve exact command, source inventory, exit and log.
2. Extend the existing native integration fixtures to cover the current intended
   configuration/preparation/automatic entry, instead of counting the older
   downstream-assembled legacy scenario as proof of the new entry.
3. Exercise the ten cases, five ordered synthetic sessions and subsequent Morning
   against a current verified installation. Inject only explicit synthetic data,
   transport and clocks; native business/validation/installation owners stay real.
4. Reconcile source deltas and final gates. Prepare exact production configuration
   and automation changes before any remaining deployment approval boundary.
5. Inspect actual migration/receipts and apply the resolved final observation
   requirement. Never rewrite older proof, actual financial history or authority
   policies to make a gate pass.

## 2026-09-14T13:26:29.049295+00:00 — Current full gates and retained Store recovery verified

Frozen c3c0b382 full gates passed: 4931 unit tests passed, three explicitly documented skips, and all static/access/Node checks passed. Current runtime source matches the frozen manifest. Evidence: `.agent/acceptance/phase15-c3-final-gates-acceptance.json`.

Case4's bf078 installed retained recovery also passed independent native16-node replay with exactly one financial CAS and unchanged upstream/Factor/Top100/cutoff artifacts. Its original test interruption and unavailable first public return remain disclosed. Evidence: `.agent/acceptance/phase15-missing-store-final-acceptance.json`. This does not replace the current c3 uninterrupted five-day requirement.

The current c3 first day has all16 native journal nodes SUCCEEDED but its final native EOD proof is still pending. The original process continues after host sleep; no restart or whole-goal completion is claimed.
