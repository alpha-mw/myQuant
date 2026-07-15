# v14 Intelligence Retirement Evidence Baseline

- Captured at: `2026-07-14` (`Asia/Shanghai`)
- Source repository: `/Users/maxwell/mySpace/myQuant`
- Isolated implementation worktree: `/private/tmp/myQuant-v14-three-branch`
- Implementation branch: `codex/v14-three-branch-no-intelligence`
- Isolation base commit: `18a5cf8`
- Target contract: three research branches (`quant`, `fundamental`, `macro`) and two Bayesian likelihoods (`quant`, `fundamental`)

This receipt makes the retirement boundary explicit. Runtime code, configuration,
CLI/API fields, and new storage contracts are removed directly; no disabled or
neutral Intelligence compatibility shell is retained. Historical artifacts stay
read-only evidence and must not be copied into a v14 write namespace.

## Immutable data evidence

| Path | Bytes | SHA-256 | Notes |
| --- | ---: | --- | --- |
| `data/parquet/cn/intelligence_daily/part.parquet` | 95,410 | `aa81955db7c7692d2f908e50fa18a850efc33c49d11f048cdea862e253043b07` | 5,531 rows and symbols; dates `2025-08-12` through `2026-06-05` |
| `data/parquet/cn/_latest.json` | 3,216 | `1c9e7af7c764d5c90d5d660c20572d91183a3c40de54cd84e1ae90e1d764a17d` | Must remain byte-identical during activation |
| `data/parquet/cn/_catalog.json` | 37,837 | `f3b45a0bf9d141c13c03b1f19fe6a46ed42937a0979098a2b7b8190fb5fee2de` | Pre-cutover baseline; only the catalog CAS activation may change it |

The catalog activation is intentionally separate from code construction. It may
remove only the `intelligence_daily` catalog requirement after the v14 code gate
passes and only when the expected pre-cutover catalog hash above still matches.
It must not edit `_latest.json` or the historical mart.

The guarded implementation is `scripts/retire_intelligence_catalog.py`. Run it
without `--apply` for the read-only CAS preview. The production CLI has compiled
canonical paths and hashes and deliberately provides no path/hash override.

`--confirm-token RETIRE_INTELLIGENCE_CATALOG_V14` is only an anti-accident
token; it is not Maxwell approval. Apply additionally requires both files below:

1. `reports/intelligence_retirement/v14_activation_receipt.json`, using schema
   `myquant.intelligence-retirement-activation-receipt.v14`. It must bind the
   final clean candidate commit, exact architecture/branch/likelihood/report
   versions, ordered `quant/fundamental/macro` branches, ordered
   `quant/fundamental` likelihoods, the three immutable file hashes, all four
   report-tree digests/counts, and passed gates for code, replay, no new BUY,
   unchanged RiskGuard authority, integrated parallel work, quiesced runtimes,
   and removal of retired bytecode.
2. The repository-external, mode-`0600` Maxwell attestation at
   `~/.config/myquant/approvals/v14_intelligence_catalog_retirement.json`, using
   schema `myquant.maxwell-intelligence-retirement-approval.v14`. It must contain
   a random 64+ hex-character nonce, `explicit_confirmation: true`, and bind the
   activation-receipt SHA, candidate commit, and pre-cutover catalog SHA. It
   must also contain an exact `runtime_quiescence` object with:
   `observed_at_utc` ending in `Z`, `max_age_seconds: 300`, the same candidate
   commit/catalog SHA, scope
   `daily_daemon/web_runtime/market_runtime/research_runtime/retired_source_and_pyc`,
   and empty `active_processes` and `residue` arrays. Observations older than
   five minutes or more than 30 seconds in the future are rejected.

Neither file is fabricated by the implementation worktree. They may be created
only after the final replay and explicit approval gates have actually passed.

## Immutable historical report evidence

Directory digests hash the sorted output of `shasum -a 256` over every file. The
file count and allocation are recorded to make accidental rewrites visible.

| Directory | Files | KiB | Tree digest |
| --- | ---: | ---: | --- |
| `reports/daily` | 8 | 196 | `e5b7459a38234225daaf51522c7df885ac7e9496fc6dada1717453adb6adf4de` |
| `reports/branch_readiness` | 2,016 | 159,440 | `b256e1c7eb5b1c0abbc5e926ef208d6e01b5484ae8ca1a05016a4154b26e39a2` |
| `reports/branch_readiness_clean_parquet_smoke` | 3 | 556 | `f4b5ae8bec66aafcd3de8d2a55b6ad7f99f4a059a040e879c3edb626a8671cc6` |
| `reports/holdings_dag_review` | 6 | 1,472 | `df00f19ee8c2bfa1cb4770ebc30d07b5ad01cbfa6a347ed221603beebae4269f` |

These directories and `reports/intelligence_retirement` are rejected by all
cleanup gates and executors, even if a deletion whitelist and confirmation token
would otherwise permit removal.

They are frozen v13 evidence trees, not current output locations. v14 branch
readiness defaults to `reports/v14/branch_readiness`, and v14 daily automation
defaults to `reports/v14/daily`. Direct writes to the two old active roots are
rejected so a normal v14 run cannot invalidate the retirement evidence.

Apply acquires the non-blocking exclusive `fcntl.flock` at
`data/parquet/cn/._catalog.json.intelligence-retirement.lock` before the initial
validation and holds it through write, fresh-reader probe, post-readback, and any
rollback. Lock contention blocks immediately. This script is the repository's
only production `_catalog.json` writer; any future writer must use the same lock.
Catalog replacement uses a mode-preserving, safely unique temporary file.

Inside the lock, the catalog, `_latest`, mart, and every report-tree digest are
validated once initially and again before `os.replace`. The receipt and external
approval hashes are rebound at that boundary, followed by a catalog SHA check
immediately adjacent to replacement. A concurrent update is retained and blocks
activation rather than being overwritten.

Initial, pre-replace, and post-apply runtime scans must all find no known daily
daemon, Web runtime, market runtime, or research runtime process. They also scan
the five retired source paths and their direct/`__pycache__` bytecode variants.
An unavailable process scan is itself a blocker. After replace, all evidence is
read back and a newly constructed strict `MarketDataReader` must fail specifically
because `intelligence_daily` is absent from the catalog. Any write/readback/probe
failure restores the original catalog atomically and verifies the pre-cutover SHA
before returning a blocked status.

## Activation stop conditions

Activation must stop closed if any of the following is true:

1. An Intelligence constructor, toggle, DTO field, branch slot, likelihood slot,
   write path, or current-protocol report field remains.
2. A v13 calibration, outcome-ledger, posterior-overlay, or report namespace can
   be loaded as a current v14 artifact.
3. `_latest.json` or any immutable evidence digest changes.
4. The v14 candidate creates a new BUY, weakens a RiskGuard veto, or changes a
   surviving action without an explained three-branch/two-likelihood cause.
5. The parallel Quant, Fundamental, and Macro work is not integrated and replayed
   on the final clean candidate commit.
6. Runtime writers/services are not quiesced, retired `.pyc` files remain, or a
   fresh strict reader does not fail closed after the catalog CAS.
7. The exclusive catalog lock is busy, the process scan is unavailable, or the
   catalog SHA changes at the final in-lock CAS boundary.

Main-branch merge and catalog activation require Maxwell's explicit confirmation.
