# Factor candidate batch `factor-candidates-2026q4-b1`

Status: research definitions only. Nothing here is preregistered, selected,
admitted or weighted. The executable definitions live in
`quant_investor/factors/lifecycle_candidates.py`; this document records the
hypotheses and the trial accounting that any later statistic over the batch
must charge.

## Why a separate batch

The prospective admission clock (390 planned sessions, 300 valid RankIC
sessions, 12 month-ends, 8 disjoint cohorts) only starts when a candidate is
preregistered. No prospective preregistration has been sealed, so every month
without a batch is a month of lost evidence. Each quarterly batch is kept small
and theory-led so the deflated-Sharpe hurdle stays clearable
(`docs/factor_mining_mechanism.md`, section 4).

## Candidates

| factor_id | family | operator, window | hypothesis |
|---|---|---|---|
| `pv_volatility_penalty_5d` | LOW_VOLATILITY | negative std of daily adjusted returns, 5 | Low-risk anomaly (Ang et al. 2006; Frazzini and Pedersen 2014) |
| `pv_volatility_penalty_10d` | LOW_VOLATILITY | same, 10 | Window variant of the row above, not an independent idea |
| `pv_downside_volatility_20d` | LOW_VOLATILITY | negative std of negative-clipped returns, 20 | Downside and crash risk (Ang, Chen and Xing 2006) |
| `pv_short_reversal_5d` | SHORT_TERM_REVERSAL | negative 5-session return | Liquidity provision and overreaction (Jegadeesh 1990; Lehmann 1990) |
| `pv_max_return_20d` | LOTTERY | negative max daily return, 20 | MAX effect (Bali, Cakici and Whitelaw 2011) |
| `fund_fin_ocf_to_profit` | EARNINGS_QUALITY | latest PIT-available value | Accrual anomaly (Sloan 1996) |

All are `HIGHER_IS_BETTER`. Price candidates read only `trade_date` and
`adj_close`; a value exists only when every return in its window is finite.
The fundamental candidate requires rows with `available_date` and a `PIT`
source classification, and rejects any row available after the cutoff instead
of dropping it.

The two leads named in the mining review (`pv_volatility_penalty_5d/10d`, about
0.4 correlation with the bootstrap pool, and `fund_fin_ocf_to_profit`, about
0.02) are included. Reversal and lottery were added for family diversity; both
are long-documented A-share effects rather than mined shapes.

## Trial accounting

- Nominal trials: 6.
- Families: 4 (LOW_VOLATILITY 3, SHORT_TERM_REVERSAL 1, LOTTERY 1,
  EARNINGS_QUALITY 1).
- A DSR over this batch must use at least the effective count measured on the
  realized IC series (`effective_trial_count`), never fewer than the 4 families,
  and must add every later batch's trials when candidates are compared across
  batches.

## Path to a governed preregistration

The installed registry is now split:

- Bootstrap (`BOOTSTRAP_FACTOR_IDS`): exactly LOW and W80. Default
  `installed_implementation_rows()` returns only this set, so Factor production
  authority still replays the sealed `implementation-tree.json` byte-for-byte.
- Prospective (`PROSPECTIVE_FACTOR_IDS`): the five price-volume candidates in
  this batch. Build their validator manifest with
  `FactorValidationStore.build_prospective_validator_manifest`, then
  preregister through `factor mine` under a release that ships the split.

Still required before sealing the fundamental row:

1. Add a governed `FUNDAMENTAL` source role with PIT decoding for
   `fund_fin_ocf_to_profit`. Until that exists, the fundamental candidate stays
   in research replay only (`lifecycle_candidates.compute_pit_fundamental_candidate`).
2. Ship the split in a release, then preregister the price-volume subset with
   separate authorization. Do not mix Bootstrap and prospective IDs in one
   validator manifest.
