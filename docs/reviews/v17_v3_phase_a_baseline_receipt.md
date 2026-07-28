# V17 v3 Phase A Baseline Receipt

- Recorded at: `2026-07-27T04:10:59Z`
- Branch: `codex/v17-v3-quant-first`
- Frozen source commit: `3a5d69934b686087d4d734f5b5e8254de9dd4a6d`
- Frozen Git tree: `db7e908b8b92af73cee7e5b0cace15e328fc51b9`
- Protocol: `myquant.v17.v3`
- Scope: research-only Phase A baseline
- Production/default change: forbidden

## Frozen files

The complete 85-file SHA-256 inventory is
`docs/reviews/v17_v3_phase_a_baseline_files.sha256`.

The inventory contains only source, schemas, packaged policy resources, tests,
scripts, and documentation. It excludes private data, holdings, provider raw
responses, tokens, `.env`, runtime workspaces, caches, and test residue.

## Validation evidence

- `verify_package()`: 42 assets verified.
- `verify_runtime_build()`: 17 runtime sources verified.
- `scripts/run_v17_v3_phase_a_gate.py`: `NOT_ACTIVATED_DATA_BLOCKED`.
- V17 v3 unit/integration suite: 71 passed.
- V15 public/default and frozen V17 v2 regression suite: 1659 passed.
- Changed-file secret scan: 85 files scanned, 0 configured-token or private-key
  hits.
- V17 production-code forbidden import scan: 0 broker/execution/order/trade
  imports.
- V17 production-code forbidden call scan: 0 order/trade/execution/broker
  client calls.
- `git diff --check`: passed before the frozen source commit.

## Manifest and frozen-boundary evidence

- Package manifest SHA-256:
  `e8dc183437a631f60aa06e17dd59f10dccce21dd3c4d576adfa479dad6b2674c`
- Runtime manifest SHA-256:
  `c7b86c3dc0d04a3dbe95ebd2034d76772a2c3312ac51d0cfe119258ad69eebf5`
- V17 v2 contract tree: 46 files,
  `58de840f785feea78d3f7d69f5199bfe45f2a60f70102f62f3821d5f523657e2`
- V17 v2 runtime tree: 19 files,
  `72e577bcb959ffc029feca3191bf3a10ec2ab79257dc0af13060b3ccb1816eff`
- V15 CLI before/after SHA-256:
  `bbdc16c0a6df93fb5ac4405806d5d510e376fccf84dbb3ab4e77569335e1004b`
- Dashboard Contract v3 before/after SHA-256:
  `a7c70208ce291683184eb895067b5d3206a3e4d415958f13b85c8b2ee9631268`

The Phase A gate also verified that all provider, LLM, broker, order, and trade
call flags are false; all authority flags are false; and the V15 default
entrypoint is unchanged.

This receipt does not activate V17, promote formal research, alter a production
data pointer, change a schedule, or change the default research runtime.
