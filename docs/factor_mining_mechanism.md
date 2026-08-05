# Factor mining mechanism: current state and proposed redesign

Written 2026-08-05 after a mining run over the full calendar returned zero
qualified candidates for the fourth consecutive week. The run was not unlucky.
This document records why the current mechanism cannot produce a new production
factor, and what to replace it with.

## 1. What the current mechanism does

`scripts/mine_quant_branch_factors.py` enumerates a fixed catalogue of
candidates — roughly 230 of them, formed as `family x window` over a
hand-written list of price/volume, fundamental and formulaic families — scores
each one standalone on a single full-sample path, and admits the ones that pass
all eight gates in `quant_investor/factors/governance.py`.

Three properties of that design matter more than any parameter inside it.

**It is an enumeration, not a search.** The candidate list is deterministic. Re-
running the miner against the same data produces exactly the same 230
candidates with nearly the same metrics. There is no generative step, so
"mining again" cannot discover anything the first run did not already see. The
only thing that changes between weekly runs is the data tail.

**Every candidate is scored standalone.** Gate 8 checks redundancy against the
existing pool, but only at the very end, as a rejection. Nothing in the search
is pointed at *incremental* value. The consequence is visible in the
2026-08-01 run: of 230 candidates, exactly 9 showed a positive portfolio
increment, and all 9 had 0.81-1.00 correlation with factors already in the
pool. The mechanism spent its entire budget rediscovering `low_dollar_volume`
and `amihud_illiquidity` at slightly different windows.

**There is no accounting for the number of trials.** Family-wise
Benjamini-Hochberg is applied to RankIC p-values, but those p-values come from
an iid t-test on daily RankIC against a 30-day forward return. Overlapping
labels inflate that statistic by roughly `sqrt(30)`. Measured directly on the
current production set, two of the five factors that "passed 8/8" fail once the
overlap is corrected by Newey-West at lag 30 or by non-overlapping 30-session
cohort means.

## 2. Why the run returned zero, mechanically

Two separate blockages, both now diagnosed.

**Structural.** Commit `acae1fb` (2026-07-13) deliberately pinned the evidence
for Gates 1, 5, 6, 7 and 8 to fail-closed values inside `candidate_metrics`,
with the comment that the legacy retest does not emit the versioned audits
those gates require. Unknown evidence is treated as failure, which is correct
policy — but it means no candidate can score above 3/8 until real evidence
producers exist. Observed maximum was 2/8.

That commit landed one day after the 2026-07-12 run that wrote the current five
production factors with "8/8 passed". Those records were produced by the code
this commit judged untrustworthy, which independently corroborates the
measurement finding above.

**Data.** The exposure loader read three loose raw tables plus a hand-written
`_catalog.json`. That source stopped being written on 2026-06-11; the
`_catalog.json` has since disappeared entirely; and while it was current it
reconstructed 27% of its market caps by multiplying one as-of `total_share`
snapshot by close, which is not point-in-time. With exposure blocked, Gate 2
also fails, because sector and size concentration are computed from those maps.

## 3. What was fixed on 2026-08-05

The exposure blockage is repaired. `quant_investor/factors/exposure_maps.py`
now reads sector and point-in-time market cap from the governed fundamental
generation (`_fundamental_latest.json` -> `fundamental_daily`), hash-bound to
the generation manifest. On the current mart this is `ready`: 100% point-in-time
size pairs against the previous 73%, zero reconstruction against the previous
27%, coverage through the live trade date rather than stopping two months back.

Size buckets are recomputed here as cross-sectional terciles rather than taken
from the generation's own `size_bucket` column, which uses fixed absolute
thresholds — under those, a rising market sweeps the whole cross-section into
`large` and neutralization stops removing any size exposure at all.

Gate 6 now reads the real sector x size neutralized ICIR, conditional on the
exposure evidence saying `ready`; when it is not ready the gate still fails
closed. `style_exposure_only` became a real diagnostic: a factor is flagged as
pure style when demeaning inside sector x size buckets flips its direction or
eats more than half its ICIR.

The mining analysis window is now clamped to the exposure coverage start, so
the miner no longer scores rebalance dates it cannot neutralize.

Repairing the exposure source surfaced two latent defects that had been
invisible while it was blocked, and either alone would have kept Gate 2 and
Gate 6 failing even with perfect evidence:

- The production evidence blocker read
  `float(exposure.get("reconstructed_size_pair_ratio", 1.0) or 1.0)`. A
  legitimate 0.0 — no reconstruction at all, the best possible result — is
  falsy, so it was replaced by the 1.0 default and reported as
  `factor_exposure_reconstruction_above_35pct`. The legacy source never reached
  0.0, so the bug had never fired.
- `_restrict_context_from_start` did not carry `sector_by_symbol`,
  `size_bucket_by_symbol`, `size_bucket_by_date` or `exposure_metadata` onto
  the narrowed context, and per-candidate maturity contexts were derived from
  `full_context`, which never received the exposure at all. Every candidate was
  therefore scored as though no exposure existed: Gate 2 saw one sector holding
  100% of the cross-section and Gate 6 saw a neutralized ICIR of zero.

Gate 7 was then rebuilt on combinatorial purged cross-validation — see Stage 1
below, which is now implemented in `quant_investor/factors/purged_cv.py`.

Gates 1, 5 and 8 remain fail-closed, as does `parameter_stability` inside Gate
7. They need new producers, described below.

### Result of the repairs

Run `factor_v4_mining_20260805_final`, 230 candidates over 2021-08-04 to
2026-06-22 (the analysis start was clamped to the exposure coverage start):

| | before | after |
|---|---|---|
| production market evidence | `factor_exposure_evidence_not_ready` | passes, no blocker |
| best gate score | 2/8 | 5/8, reached by 24 candidates |
| gate score distribution | 56 at 0, 54 at 1, 120 at 2 | 8/16/44/45/93/24 across 0-5 |
| `max_sector_coverage_share` | 1.0 (no exposure) | ~0.063 (real) |
| neutralized ICIR | 0.0 for all | real; 207/230 are not pure style |
| out-of-sample evidence | contiguous folds, pinned to 0.0 | 45 purged CPCV paths, 174/230 above the 0.55 floor |
| fail-closed reason | `factor_exposure_evidence_not_ready` | `no_qualified_positive_candidate` |

The remaining failure is now a statement about the candidates rather than about
missing plumbing. Every 5/8 candidate fails on exactly Gates 1, 5 and 8 — the
three that still have no evidence producer.

Worth noting for whoever picks this up: `pv_volatility_penalty_5d` and
`pv_volatility_penalty_10d` reach 5/8 with a neutralized ICIR near 1.0 and only
0.39-0.41 correlation with the existing pool, which is well inside the 0.70
redundancy ceiling. Under the old exposure-blind scoring they were
indistinguishable from the rest. They are not admissible yet, but they are the
first non-redundant leads this pipeline has produced.

## 4. Proposed mechanism

The literature converged some time ago on a shape that this repository does not
yet have. The four ingredients are: purged multi-path validation, explicit trial
accounting, a set-level objective, and an actual search over an expression
space.

### Stage 1 — Combinatorial purged cross-validation (fixes Gate 7) — implemented

`quant_investor/factors/purged_cv.py` splits the session calendar into 10
blocks and tests on all `C(10,2) = 45` block pairs. For each split it purges
training sessions whose 30-session label window reaches into a test block, then
embargoes the 30 sessions immediately following each test block. Blocks and
purge/embargo are counted in sessions, so the 30-day purge and embargo Gate 7
demands are literal; each path then keeps whichever month-end RankIC
observations fall inside its test blocks.

Gate 7 now reads real evidence: `oos_positive_ratio` is the fraction of the 45
paths with positive test IC, with `walk_forward_purged=True`, `purge_days=30`,
`embargo_days=30`, `fold_count=45` and a content-bound evidence hash. No new
data was required.

One caveat found while testing it, which matters for Stage 2: the 45 paths are
heavily overlapping — each block appears in nine of them — so
`positive_path_ratio` is not a mean of independent trials. A pure-noise signal
can and does land above the 0.55 floor on individual draws; only across many
draws does it centre near 0.5. The path ratio is therefore necessary but not
sufficient, which is precisely why the trial-aware statistics below are the
next step rather than an optional refinement.

`parameter_stability` stays fail-closed inside `candidate_metrics` and is
filled in by `_set_parameter_stability`, a run-level post-pass that can see a
candidate's neighbours in the same family. Per-candidate metrics cannot compute
it, so that split is the right one.

### Stage 2 — Trial-aware admission — implemented

`quant_investor/factors/trial_correction.py` adds three corrections on top of
the eight gates. They are strictly an extra bar: `qualified_candidates` requires
both a passing gate review and a passing correction, and nothing in the
correction can qualify a candidate the gates rejected.

- **Deflated Sharpe Ratio.** The run records how many candidates it scored and
  how widely their ICIRs varied, computes the ICIR the best of N worthless
  trials would be expected to reach, and deflates the observed ICIR by it,
  adjusting for the skew, kurtosis and length of the IC series. Because the
  "returns" series here is the per-rebalance RankIC series, its Sharpe *is* the
  ICIR the pipeline already reports. Admission floor 0.95.
- **Probability of backtest overfitting.** Combinatorially symmetric
  cross-validation over the whole candidate set: 10 time blocks, every balanced
  split into in-sample and out-of-sample halves (252 of them), nominate the
  in-sample winner, ask where it ranks out of sample. Ceiling 0.5.
- **Non-overlapping cohort t-statistic against a hurdle of 3.0**, not 2.0.
  Cohorts are exactly one 30-session horizon wide, so they do not share a label
  window at all — the overlap is removed rather than modelled.

Because a mining run reports the single best candidate it produced, the relevant
error rate is family-wise, not merely false-discovery, which is why this sits
alongside the existing family-BH rather than replacing it.

One caveat established while testing PBO: every split reuses the same
performance matrix, so a config that happens to draw a high mean across all
blocks wins in sample *and* ranks high out of sample. Under the null, PBO
averages 0.5 across matrices, but a single draw ranges roughly 0.10 to 0.89
(sd about 0.19). Read one run's PBO as a diagnostic with real sampling error,
not as a precise threshold — the DSR and the t-hurdle are the sharper of the
three.

A unit bug worth recording, because it is the same class of mistake twice: the
cohort width for the non-overlap t-test was fixed at 30, but 30 is a count of
*trading sessions* while the IC series is sampled at *month ends*. A monthly
series of ~40 observations was therefore always shorter than the 60 the test
demanded, and every one of the 230 candidates silently failed closed with
`t = 0.0`. `infer_cohort_size` now derives the width from the series' own
median business-day gap: 30 observations for a daily series, 2 for a month-end
one. After the fix, 94 of 230 candidates clear `t > 3.0`.

### What the correction says about this run

Run `factor_v4_mining_20260805_stage3`:

| | value |
|---|---|
| trials | 230 |
| spread of trial ICIRs | 0.380 |
| **expected best-of-230 ICIR under the null** | **1.067** |
| **best ICIR actually observed** | **0.724** |
| candidates clearing `t > 3.0` | 94 / 230 |
| candidates clearing DSR ≥ 0.95 | 0 / 230 |
| PBO | 0.048 (median OOS rank 0.91) |

The two headline numbers are not in conflict, and reading them together is the
point. PBO of 0.048 says the ranking is *stable*: the in-sample leaders land in
the top decile out of sample, so the selection is not fitting noise. The
deflated Sharpe says the effect sizes are nonetheless *too small for the size of
the search*: searching 230 candidates whose ICIRs spread by 0.38 would be
expected to throw up a worthless winner at ICIR 1.067, and the best real
candidate reaches 0.724.

That is a directional finding, not just a rejection. It says the way forward is
**not** to enumerate more candidates — a bigger search raises the bar it has to
clear. It is either to cut the trial count by pre-registering a smaller,
theory-driven candidate set, or to find genuinely larger effects.

### Effective trials — implemented, and the answer is uncomfortable

The trial count started as all 230 candidates, but they are nowhere near
independent: 70 are `volume_stability_smooth` at different smoothing pairs, 10
are `low_dollar_volume` at different windows. Bailey and López de Prado's N is
the number of *effectively independent* trials.

`effective_trial_count` clusters candidates by the absolute correlation of their
IC series — the series already computed for the non-overlap test, so it costs no
extra pass over the panel — at the same 0.70 floor the gate policy uses for
factor redundancy. Sign is ignored: a factor and its negation are one bet.

On run `factor_v4_mining_20260805_effective` that collapses 230 trials to **7**,
the bar from an expected best-of-N ICIR of 1.067 down to **0.527**, and the best
candidate's deflated Sharpe from 0.052 up to **0.827**. The best observed ICIR
of 0.724 now comfortably exceeds the null bar. Still nothing clears 0.95.

**But the clustering is single-linkage, and it chained.** The largest of the
seven clusters contains **224 of the 230 candidates**: A correlates with B, B
with C, and the whole set fuses even where A and C do not correlate at all. So 7
is an aggressive floor on the effective count in the same way 230 was an
aggressive ceiling. The honest reading is a bracket, and the verdict is
genuinely sensitive across it:

| effective N | expected best-of-N ICIR | DSR of the leader |
|---|---|---|
| 3 | 0.324 | 0.987 |
| 7 (single-linkage) | 0.527 | 0.864 |
| 15 | 0.672 | 0.613 |
| 30 | 0.787 | 0.363 |
| 230 (naive) | 1.067 | 0.028 |

The leader clears DSR ≥ 0.95 only if the effective trial count is 4 or fewer.
Since the run explores at least a dozen genuinely distinct ideas — momentum,
reversal, illiquidity, volume stability, several fundamental ratios — the
plausible range is the middle of that table, and the leader does not clear the
bar there.

The next refinement is average-linkage or an eigenvalue-based effective count,
neither of which chains. But note what the sensitivity table already settles:
no reasonable choice of N makes the current candidate set admissible, so this is
a precision question about *how far short* the run falls, not about whether it
falls short.

### Stage 3 — Search over an expression space, with a set-level objective

This repository already has `quant_investor/factors/aquant_expression.py`, an
expression evaluator. The search space therefore exists; only the search is
missing. Replace the fixed 230-candidate list with generation over
operator x field x window expression trees, in the Alpha158/Alpha360 idiom —
a curated operator set expanded across multiple rolling horizons.

For the search itself, genetic programming is the established baseline; the
current literature reports better results from RL (AlphaGen, AlphaQCM), from a
surrogate-scored generative-predictive loop (AlphaForge), and from LLM-guided
MCTS, which is a natural fit here because the project already runs an LLM
provider. Whichever is chosen, carry over the **frequent subtree avoidance**
idea — penalise candidates that reuse subtrees already common in the pool — to
stop the search collapsing onto one formula shape.

**The single most important change is the objective — implemented 2026-08-05.**
`quant_investor/factors/incremental_alpha.py` projects the production pool out
of each candidate cross-section by cross-section (both sides ranked first, so
the projection is scale-free) and `candidate_metrics` reports the residual's
RankIC, ICIR, retention against the standalone ICIR, and its own CPCV path
ratio. `rank_candidates` then orders on gate score, then residualised ICIR,
with standalone ICIR demoted to a tie-break.

This is what AlphaGen calls a synergistic set objective. It directly addresses
the 2026-08-01 failure, where every positive candidate was a near-duplicate of
something already held: redundancy stops being a late Gate 8 rejection and
becomes the thing the ranking optimises against.

The separation it buys is sharp. On a fixture with two equally strong but
orthogonal edges — one already in the pool, one new — standalone `mean_rankic`
cannot tell the clone from the new factor (they differ by less than 0.05), while
the residualised RankIC collapses to under 0.01 for the clone and stays above
0.05 for the new factor. The residual CPCV path ratio separates them the same
way: above 0.55 for the new factor, below it for the clone.

Run `factor_v4_mining_20260805_incremental` shows the same thing on real data.
`pv_low_dollar_volume_5d` carries the highest standalone RankIC in the entire
run at 0.1206, which is what put it and its siblings at the top of the old
ranking. It is also already in production, and its correlation with the pool is
1.00. Residualised, its RankIC is 0.0112 and it falls to position 95 of 230.
The whole family degrades exactly as window length increases its overlap with
the pool:

| candidate | standalone RankIC | residual RankIC | retention | pool corr |
|---|---|---|---|---|
| `pv_low_dollar_volume_5d` | 0.1206 | 0.0112 | 0.47 | 1.00 |
| `pv_low_dollar_volume_10d` | 0.1189 | 0.0135 | 0.25 | 0.98 |
| `pv_low_dollar_volume_15d` | 0.1148 | 0.0059 | 0.10 | 0.97 |
| `pv_low_dollar_volume_20d` | 0.1124 | 0.0036 | 0.06 | 0.95 |
| `pv_low_dollar_volume_25d` | 0.1107 | 0.0036 | 0.06 | 0.94 |

What rises in their place is more interesting than what falls.
`fund_fin_ocf_to_profit` reaches position 4 with a *retention above 1.0* — its
residual RankIC of 0.0246 exceeds its standalone 0.0222, because the pool was
working against it — and a pool correlation of 0.024. Standalone ranking buried
it; it is the only fundamental family near the top, and family diversity is a
hard requirement of the five-factor baseline.

Across the run, 25 of 230 candidates now carry retention below 0.35, i.e. the
pool already explains two thirds or more of them. None of that was visible
before.

### Stage 4 — The remaining hard gates

- **Gate 1** needs a versioned PIT/tradability audit per candidate. The
  generation already carries `availability_date` per row, so most of this is
  mechanical: assert every input a candidate touches was knowable on its
  rebalance date, and record the audit hash.
- **Gate 5** needs a real slippage model — a participation-rate curve over the
  traded-amount panel — instead of the current flat 1bp assumption, plus an
  execution-feasibility check against suspension and limit states.
- **Gate 8** needs the A/B/C/D replay through Quant -> Bayesian -> RiskGuard ->
  PortfolioConstructor with per-arm hashes. This is the largest piece and the
  one that genuinely gates production admission.

## 5. Recommended order

1. ~~Stage 1 (CPCV)~~ — done 2026-08-05.
2. ~~The set-level residualised objective from Stage 3~~ — done 2026-08-05.
3. ~~Stage 2 (DSR/PBO/t>3)~~ — done 2026-08-05.
4. ~~Effective-trial clustering~~ — done 2026-08-05, with the single-linkage
   chaining caveat above.
5. Replace single-linkage with average-linkage or an eigenvalue-based effective
   count, to tighten the N bracket. **This is the next piece of work**, and it
   is a precision improvement, not a blocker.
6. Stage 4 gates, Gate 8 last.
7. Stage 3's search engine — but see below.

The ordering logic has changed as a result of what Stage 2 measured. Generating
a larger candidate space *raises* the trial count and therefore raises the
deflated-Sharpe bar every candidate must clear. On this evidence a bigger search
would make admission harder, not easier. Build the search engine only alongside
a pre-registration discipline that keeps the effective trial count down;
otherwise it manufactures exactly the factor zoo this document opens by
describing.

Stage 3's search engine can wait until 1-3 are in place: there is no value in
generating more candidates before the evaluation can tell a real one from a
lucky one.

## References

- Harvey, Liu and Zhu, ["… and the Cross-Section of Expected Returns"](https://academic.oup.com/rfs/article/29/1/5/1843824), *Review of Financial Studies* 29(1), 2016 — the t > 3.0 hurdle and the multiple-testing haircut.
- Bailey and López de Prado, [The Deflated Sharpe Ratio](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio) — selection bias, backtest overfitting, non-normality.
- [Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation) and [purged-cross-validation](https://github.com/eslazarev/purged-cross-validation) — purging, embargo, CPCV and DSR in a scikit-learn-compatible implementation.
- [Cross Validation in Finance: Purging, Embargoing, Combinatorial](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/) — practical CPCV walkthrough.
- Yu et al., [Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning](https://www.semanticscholar.org/paper/Generating-Synergistic-Formulaic-Alpha-Collections-Yu-Xue/6720ff18a0a3fb567549945da6d61bb8b4322271) (AlphaGen, KDD 2023) and its [reference implementation](https://github.com/RL-MLDM/alphagen/) — set-level objective.
- [AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors](https://arxiv.org/pdf/2406.18394), AAAI 2025 — generative-predictive mining with test-time combination.
- [Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining](https://arxiv.org/pdf/2505.11122) — backtest-feedback-guided MCTS and frequent subtree avoidance.
- [AlphaEval: A Comprehensive and Efficient Evaluation Framework for Formula Alpha Mining](https://arxiv.org/pdf/2508.13174) — why IC plus backtest return alone is an insufficient evaluation.
- [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) — Alpha158/Alpha360 as the reference A-share operator x field x window search space.
