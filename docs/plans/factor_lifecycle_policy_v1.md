# Factor lifecycle policy v1

Status: **accepted parameters, implementation landed, not activated.** The
governed builders live in `quant_investor/factors/governance/lifecycle.py` and
the contracts `factor.lifecycle_policy` / `factor.lifecycle_decision` are
compiled. Nothing here changes the sealed admission policy in
`docs/factor_governance.md`, the active LOW/W80 set, `results/factors/_active.json`,
or any weight. Applying a sealed weight proposal still requires a release that
ships this code, a monthly decision seal, and separate Factor production
authorization.

## Problem

Admission is strict (390 planned sessions, t > 3 on disjoint cohorts,
DSR >= 0.95, PBO <= 0.5, ...) but there is no exit: an admitted factor is never
monitored for decay, down-weighted or retired. A fixed review window is the
wrong tool for both ends:

- too short cannot separate an effect from noise: with disjoint cohorts the
  cohorts needed for a t-threshold `t*` are about `(t* * sigma / mu)^2`;
- too long keeps a dead factor at full weight while its evidence accumulates.

This policy separates *estimating the level* of a factor's IC from *deciding
its weight*, accumulates evidence continuously, and makes decisions on a fixed
cadence.

## Evidence behind the parameters

Research report `reports/factor_lifecycle/backtest-20261002T004041Z-70ea02e27f3e.json`
(snapshot `parquet/cn/_snapshots/20261002T004041Z.json`; 176 month-ends,
2012-01 to 2026-08; 20-session label entered at the next close; 10 factors,
5 effective trials). Produced by `scripts/research_factor_lifecycle_backtest.py`
from `quant_investor/factors/lifecycle_backtest.py`. Research replay only: no
costs, no ST/limit filter, IC-weighted IC is signal quality rather than return.

1. **Level estimation wants long memory.** One-step-ahead MSE of next-month IC
   falls monotonically up to 12 months and is flat beyond; the expanding mean
   is best overall (24 months in the first half). Short windows mostly fit noise.
2. **Weight decisions benefit from medium memory.** Over a 64-rule grid, mean
   excess IC over a fair baseline (weights proportional to the positive
   expanding-mean IC, no states) is +0.0122 at a 3-month decision half-life,
   +0.0075 at 6, +0.0036 at 12 and +0.0020 at 24. Three months churns (about 12
   retirements over 14 years, turnover 17.8); six months has the best average
   IR (0.759) and worst-12-month IC sum (0.372) with half the retirements.
3. **Part of the rule value is concentration.** The best uncapped rule retired
   nine factors by 2017 and ended almost entirely in `pv_low_dollar_volume_5d`,
   whose IC correlates 0.76 with the small-minus-large return spread. With a
   0.35 per-factor cap the best excess halves (0.026 to 0.0126 IC per month)
   but survives deflation over 64 trials (DSR 0.959; best capped rule uses a
   6-month half-life). A cap is therefore part of the policy, not an option.
4. **Permanent retirement has a cost.** Low-volatility factors retired after the
   2015 crash recovered strongly in 2017 and 2021. In-sample, re-entry lowered
   mean excess (+0.0039 vs +0.0088) mainly by diluting the concentrated winner,
   so the re-entry rule is kept for governance reasons and its cost is stated.
5. **Probation length and watch threshold did not matter** in this grid
   (6 vs 12 months and 0.70 vs 0.80 change mean excess by under 0.0001), so the
   more conservative values are chosen. A stricter retire threshold (0.50 over
   0.40) helped.
6. **Regime exposure is visible and large.** Over the 2024-01..02 microcap crash
   `pv_low_dollar_volume_5d` IC averaged -0.076, `pv_amihud_20d` -0.092; in the
   production monitor LOW's signal has a -0.74 rank correlation with log total
   market value. The value-spread crowding proxy has weak links to the next IC
   (correlations -0.19 to +0.05) and is reported, not used as a trigger.

## States and transitions

```text
CANDIDATE -> PREREGISTERED -> PROBATION -> ACTIVE <-> WATCH -> RETIRED
                                                         RETIRED -> (new preregistration) PREREGISTERED
```

| State | Enter when | Weight scale | Production |
|---|---|---|---|
| PREREGISTERED | Candidate sealed before its first outcome, in a quarterly batch with recorded trial count | 0 | no |
| PROBATION | At least 12 monthly cohorts of prospective evidence, posterior `P(IC > 0) >= 0.80`, and positive residual IC against the current set | 0.5 | **no** (paper / research only) |
| ACTIVE | The existing 390-session admission passes in full; thresholds unchanged | 1.0 | yes |
| WATCH | From ACTIVE when posterior `P(IC > 0) < 0.70`, or the monitor's CUSUM alarms | 0.5 | yes, at reduced scale |
| RETIRED | Posterior `P(IC > 0) < 0.50` for 3 consecutive monthly decisions | 0 | no |

- Return from WATCH to ACTIVE requires `P(IC > 0) >= 0.80` and no live CUSUM alarm.
- A retired factor keeps being observed at zero weight. It can come back only
  through a new preregistration after at least 12 months, with its evidence
  window restarted; the old evidence is not reused.
- Bootstrap factors (LOW, W80) never passed prospective admission. They are
  monitored with WATCH/RETIRED semantics, but any weight change for them needs
  explicit authorization (`bootstrap_weight_change_requires_authorization`).

## Statistics

- **Level and reference** (prior and CUSUM reference): expanding mean, or a
  half-life of at least 24 months, of disjoint-cohort ICs, with a 50%
  publication haircut on historical evidence (McLean and Pontiff). The priors
  file written by the research script implements this for LOW and W80.
- **Decision posterior**: normal-normal update of disjoint-cohort ICs with a
  6-month (about 120-session) exponential half-life, Kish effective sample size,
  and an observation-sd floor of 0.01 so short smooth series cannot claim
  certainty.
- **CUSUM**: one-sided on standardized cohort means against the reference,
  slack 0.5, threshold 4 (in-control run length about 170 cohorts; detects a
  one-sigma drop in about 9).
- **Minimum evidence for any monitor state**: 60 origin sessions and 8 disjoint
  cohorts; below that the state is `INSUFFICIENT_EVIDENCE`.
- **Source gaps** (unavailable RankIC) are `SOURCE_BLOCKED`, never alpha failure.
- Implemented and tested in `quant_investor/factors/lifecycle_monitor.py`
  (`tests/unit/test_factor_lifecycle_monitor.py`).

## Weights and cadence

| Activity | Cadence |
|---|---|
| Outcomes and monitor statistics | Daily, as labels mature |
| State decisions | Monthly, after the month-end review |
| Weight changes | Quarterly, at most 0.5 total absolute weight change per rebalance |
| Structural event (rule change, liquidity crisis) | Manual review may move a factor to WATCH early |

- Target score `max(posterior mean, 0) x state scale`, normalized, then capped at
  0.35 per factor with the excess redistributed pro rata. With fewer than three
  eligible factors the remainder stays uninvested.
- Applying a sealed weight proposal requires exact `cost_result` and capacity
  evidence; missing either is a blocker (`cost_evidence_required_to_apply`,
  `capacity_evidence_required_to_apply`). The proposal itself never
  self-authorizes (`application_authorized` is always false at seal time).
- The current two-factor set cannot satisfy the cap; diversification depends on
  the candidate batch `factor-candidates-2026q4-b1`
  (`docs/plans/factor_candidate_batch_2026q4.md`).

## Trial accounting

Every parameter above was chosen from a 64-rule grid; that grid size is charged
in the deflated statistic and must be added to by any later re-tuning. Candidate
batches record nominal and family trial counts; cross-batch comparisons charge
the union.

## Relationship to existing contracts

- The prospective validation contract is sealed verbatim into each
  preregistration and checked for exact equality, so v1 applies only to
  preregistrations sealed under a release that carries it.
- The Bootstrap implementation tree replayed by Factor production authority
  binds only LOW/W80. Prospective price-volume candidates live in
  `PROSPECTIVE_FACTOR_IDS` inside
  `quant_investor/factors/governance/implementations.py` and are sealed through
  `FactorValidationStore.build_prospective_validator_manifest`. The fundamental
  candidate stays research-only until a governed `FUNDAMENTAL` source role
  exists.
- Monitor and lifecycle outputs are non-authorizing. Factor production
  activation and rollover remain the only Factor pointer writers; System
  activation remains the only System pointer writer.

## Implementation

1. `quant_investor/factors/governance/lifecycle.py`: artifact kinds
   `factor.lifecycle_policy` (sealed parameters above) and
   `factor.lifecycle_decision` (monthly per-factor state, posterior, CUSUM,
   source-gap counts and refs to the outcome heads used), with intrinsic
   validators and exact replay.
2. Transition function `transition_lifecycle_state` is a pure function of the
   previous decision, the policy and the newly matured cohorts.
3. Weight proposals are separate non-authorizing fields inside the decision;
   applying them goes through the existing Factor production writers with
   explicit authorization, cost evidence and capacity evidence.
4. Tests in `tests/unit/test_unified_factor_lifecycle.py`; flake8, black and mypy
   for the governance package as in CI.
5. Monthly Hermes job `myquant-factor-v4` runs the non-authorizing lifecycle
   monitor after the existing monthly auditor.

## Accepted review decisions

- **Probation weight stays paper/research only** until the 390-session admission
  gate passes. Putting 0.5 into production before admission would create a
  parallel production path around the sealed gate. Probation may still be used
  in paper ledgers to prove residual IC.
- **Do not move LOW/W80 to WATCH for weight changes now.** The production
  monitor is still `INSUFFICIENT_EVIDENCE` (origins well below 60). Report
  small-cap concentration as a diagnostic. Bootstrap factors enter WATCH
  semantics only after minimum evidence and a posterior/CUSUM trip; any weight
  change still needs explicit authorization.
- **Cost and capacity gate application, not the step limit.** Keep the 0.5
  quarterly absolute weight-change ceiling as a hard governance limit. Seal
  proposals without inventing a cost model; refuse to apply them when exact
  cost or capacity evidence is missing.
