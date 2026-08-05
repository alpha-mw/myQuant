# V17 Sprint I0 Historical Intelligence Inventory

## Audit boundary

- Audit baseline: `967e4b47c7a1c0a688db85ce4195efe7c877ecb1`.
- Method: read-only scan of 102 local refs and 428 reachable commits, followed by
  exact-tree inspection of the candidate commits below.
- Migration rule: no historical commit, branch, module, runtime, writer, risk
  overlay, selector, or authority object was cherry-picked or merged.
- Classification: A means an isolated pure algorithm can be reused; B means the
  idea is usable only after a new contract and dependency isolation; C means the
  implementation must not be reused and only its failure mode or invariant may
  inform I0.

## Inventory

| Component | Historical commit and date | Representative refs | Original paths | Algorithm and data contract | Dependencies and I/O | Historical authority | Tests/evidence found | Class | I0 decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bayesian evidence | `4b24547bf11e36ba99e986dfe81389a23081e04f`, 2026-07-16 | `codex/v15-no-theme-integration` and descendants | `quant_investor/bayesian/prior.py`, `likelihood.py`, `posterior.py`, `types.py`, plus calibration, overlay and outcome-ledger modules | Prior, likelihood and posterior decomposition existed, but its records were not the I0 content-addressed Evidence Receipt contract | Coupled to V15 calibration, ledger and overlay surfaces; included persistence and downstream overlay behavior | Could influence a live-era shortlist/overlay path | Historical module tests and downstream integration coverage were present on the audited refs | B | Reimplement only Bayes odds and likelihood-ratio arithmetic with fixed Decimal rules; reject calibration, overlay, ledger and runtime coupling |
| Markov regime package | `45b4de74bf5266011d7470b7e6e3e0df021b5824`, 2026-07-16 | `codex/v15-no-theme-integration` and descendants | `quant_investor/regime/engine.py`, `features.py`, `transition.py`, `scope.py`, `persistence.py`, `types.py` | Feature extraction, transition matrices and state inference existed | Persistence, scope policy and production caps were mixed into the package | Production cap behavior existed for full-market records | Markov scope/validation history and downstream runtime checks were present | B | Reimplement one-step causal filtering only; introduce three fixed layers; reject persistence, caps, hidden smoothing and production state |
| Two Branch orchestration | `3a5d69934b686087d4d734f5b5e8254de9dd4a6d`, 2026-07-27 | `codex/v17-v3-quant-first` and V17 descendants | `_branch_and_fusion` in `quant_investor/v17_v3_runtime/pipeline.py` | Quant-first branch orchestration and branch availability invariants | Coupled to the V17 v3 runtime pipeline, its artifacts and its research lifecycle | Pipeline-level selection/research behavior | V17 v3 baseline tests | C | Preserve only the invariants “Quant required” and “missing optional branch is explicit”; do not reuse pipeline code or artifact writers |
| Branch fusion algorithm | `3a5d69934b686087d4d734f5b5e8254de9dd4a6d`, 2026-07-27 | `codex/v17-v3-quant-first` and V17 descendants | `quant_investor/v17_v3_runtime/algorithms/branch_fusion.py` | Isolated deterministic fusion arithmetic | Pure helper inside a broader V17 v3 package | No independent execution authority in the helper itself | V17 v3 algorithm/runtime tests | A, algorithm only | Reuse the idea of normalized evidence mass, but define a new I0 formula from availability, confidence and reliability; do not import or copy the old module |
| Legacy single-layer regime detector | `76965a80f8744478ec464a95a3fc1b14be7201e5`, 2026-07-14 | `codex/v14-main-integration` and descendants | `quant_investor/regime_detector.py` | Single market regime detection combined with portfolio behavior | Mixed state detection with position, stop and weight decisions | Position sizing, stop and allocation influence | Legacy detector/runtime tests | C | Do not reuse; its coupling is the explicit reason I0 regime output has no portfolio, selector, risk, order or trade fields |
| V4 causal regime evidence | `b4fa743b7005b2b0ea216f387df36e8ff484126b`, 2026-07-30 | `codex/v17-v4-bounded-live-tushare-probe` and V17 descendants | `quant_investor/v17_v4_runtime/regime_evidence_v3.py` | Point-in-time, bounded, content-addressed causal evidence chain | V4 runtime, governed storage and exact artifact closure | Shadow/research-only, but still a separate V4 runtime surface | V4 regime checkpoint-chain tests | A for contract reference, C for runtime reuse | Use its causal/no-future and exact-reference principles as a read-only reference; do not import its runtime or writer |

## Findings

1. The historical repository contains useful mathematical primitives, but no
   historical package satisfies the I0 combination of content addressing,
   explicit contrary evidence, three regime layers, append-only pure memory and
   zero authority.
2. Historical Bayesian calibration, Markov persistence, regime-driven position
   logic and runtime writers are intentionally excluded.
3. The only A-class material is an isolated algorithmic idea or a read-only
   contract principle. I0 contains an independent implementation and has no
   import dependency on those historical modules.
4. Git verification for this sprint must show no merge commit and no cherry-pick
   of any audited commit.
