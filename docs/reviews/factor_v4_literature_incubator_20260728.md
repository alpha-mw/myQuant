# Factor v4 literature-backed incubator

Date: 2026-07-28
Status: research-only; not preregistered; no Factor v4 authority

## Outcome

The literature pass now freezes eleven candidate definitions plus explicit raw
accounting-data blockers. The result is narrower than the source list:
FIP continuous direction, conventional low prior-month total skewness, and the
market-adjusted distributional tail-asymmetry proxy remain eligible for later
preregistration drafts. FIP is 97.05% computable and passes every routed
0.70 correlation diagnostic; its China source is aggregate-market relevance,
not direct cross-sectional support. The source-exact China VaR1 left-tail
signal is 97.77% computable but stopped: its four-closed-month median absolute
Spearman correlation with formal `pv_downside_volatility_60d` is 0.7405,
above the frozen 0.70 line. Raw low beta is a control; low prior-month MAX is
stopped because its median monthly absolute Spearman correlation with the
protected v4.4 `pv_low_vol_of_vol_20d` is 0.8188; the exact value-weighted
Hou-Moskowitz D1 price-delay definition is blocked because current strict
Parquet has at most 10 valid regression-design observations in a 52-week
window, below the frozen minimum of 40. The 52-week-high momentum signal is
97.56% computable but stopped because its correlation with existing
`pv_momentum_120d` is 0.7134. Residual momentum, factor momentum, and change in
salience are recorded as blocked routes instead of being translated from
missing inputs or incomplete formulas. Earnings yield still waits for common
closed-month samples, same-month seasonality waits for all 12 stable
calendar-month buckets, and the quality composite remains stopped for
excessive correlation. None has Factor v4 production authority.

## Source-to-definition map

| Primary source | Local translation | Current status |
| --- | --- | --- |
| [Liu, Stambaugh and Yuan, Size and Value in China](https://www.nber.org/papers/w24458) | `cn_earnings_yield_ex_shell_30pct`: positive E/P, excluding the smallest 30% by same-date PIT market cap | Implementable, research-only |
| [Frazzini and Pedersen, Betting Against Beta](https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta) | `cn_low_beta_252d`: negative 252-session beta to the PIT equal-weight market, minimum 126 observations | Implementable as a control only |
| [Hou and Moskowitz, Market Frictions, Price Delay, and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhi023) ([author copy](https://www.ruf.rice.edu/~jgsfss/moskowitz.pdf)) | Foundational D1: `1 - R²(restricted contemporaneous market return) / R²(unrestricted contemporaneous plus four weekly lags)`, using prior-year Wednesday-to-Wednesday returns and a value-weighted market | Exact method reference |
| [Qian, Sun and Yu, High turnover with high price delay?](https://www.sciencedirect.com/science/article/abs/pii/S1544612316302781) ([DOI](https://doi.org/10.1016/j.frl.2017.06.004)) | `cn_high_price_delay_d1_52w`: rolling causal translation of D1, retaining exact Wednesday anchors, prior-week market-cap weights, 52 response weeks, four market lags, and 40 minimum observations | Structurally implementable, but blocked by insufficient historical PIT `total_mv`; no equal-weight fallback |
| [George and Hwang, The 52-Week High and Momentum Investing](https://doi.org/10.1111/j.1540-6261.2004.00695.x) ([author copy](https://www.bauer.uh.edu/TGeorge/papers/gh4-paper.pdf)) | Foundational monthly ranking: current month-end price divided by the highest price during the trailing 12 months | Exact method reference |
| [Zhou, Liu and Guo, The 52-week High Momentum Strategy and Economic Policy Uncertainty: Evidence from China](https://www.tandfonline.com/doi/full/10.1080/1540496X.2021.1904880) ([DOI](https://doi.org/10.1080/1540496X.2021.1904880)) | `cn_52_week_high_momentum_12m`: causal daily price-to-trailing-365-calendar-day-high rank, measured at closed month ends for governance | Direct China support, but high-EPU periods have virtually no momentum; local signal stopped on correlation before this missing-PIT-state blocker |
| [Blitz, Hanauer and van Vliet, The Volatility Effect in China](https://link.springer.com/article/10.1057/s41260-021-00218-0) ([DOI](https://doi.org/10.1057/s41260-021-00218-0)) | China A-share low-risk result is attributed to volatility rather than beta | Adverse evidence against raw low beta |
| [Chen, Huang and Qiu, Heterogeneous Beliefs and the Beta Anomaly in the Chinese A-share Stock Market](https://www.tandfonline.com/doi/full/10.1080/1540496X.2020.1822809) ([DOI](https://doi.org/10.1080/1540496X.2020.1822809)) | China beta anomaly is conditional on disagreement and arbitrage limits | Conditional support; required PIT state inputs are unavailable |
| [Zhao and Lin, Does behavioral-motivated volatility effect explain the beta anomaly?](https://www.sciencedirect.com/science/article/abs/pii/S154461232100307X) ([DOI](https://doi.org/10.1016/j.frl.2021.102265)) | China beta anomaly is mainly explained by volatility, MAX lottery demand, or idiosyncratic risk | Adverse to raw beta; supports a MAX mechanism test |
| [Gao, Han and Xiong, Loss from the chasing of MAX stocks: Evidence from China](https://www.sciencedirect.com/science/article/abs/pii/S1062940821000966) ([DOI](https://doi.org/10.1016/j.najef.2021.101475)) | `cn_low_max_return_20d`: rank the negative maximum daily return over the prior 20 sessions | Implementable; direct China evidence with sentiment and short-sale regime risk |
| [Wang, Wang and Wu, The role of anchoring on investors' gambling preference: Evidence from China](https://www.sciencedirect.com/science/article/abs/pii/S0927538X23001208) ([DOI](https://doi.org/10.1016/j.pacfin.2023.102054)) | `cn_low_total_skewness_20d`: rank the negative unbiased sample skewness of daily adjusted returns over the prior 20 sessions, requiring at least 15 observations | Implementable; direct China evidence conditional on distance from the 52-week high, with sentiment and arbitrage-risk mechanism states |
| [Jiang, Wu, Zhou and Zhu, Stock Return Asymmetry: Beyond Skewness](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/stock-return-asymmetry-beyond-skewness/6DB44C0DB241D0030AAA7F885CD078DB) ([DOI](https://doi.org/10.1017/S0022109019000206), [author copy](https://www.cb.cityu.edu.hk/ef/doc/2016%20Sofie/Papers/5_Wu_Stock%20Return%20Asymmetry%20Beyond%20Skewness.pdf)) | Foundational distributional measure `P(z>1)-P(z<-1)` | Method reference; not China-specific |
| [Chen, Wu and Zhu, Stock return asymmetry in China](https://www.sciencedirect.com/science/article/abs/pii/S0927538X2200052X) ([DOI](https://doi.org/10.1016/j.pacfin.2022.101757)) | `cn_low_market_adjusted_tail_asymmetry_252d`: rank the negative excess-tail probability over 252 sessions using PIT equal-weight-market-adjusted returns | Implementable proxy; direct China support, but not the paper's exact CH-3/CH-4 idiosyncratic IE replication |
| [Asness, Frazzini and Pedersen, Quality Minus Junk](https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk) | `cn_quality_cash_low_leverage`: equal-weight ranks of PIT ROE, cash conversion, and low leverage | Stopped: proxy is too correlated with existing quality/cash/leverage factors |
| [Meng, Du and Shu, Return seasonalities in the Chinese stock market](https://research.birmingham.ac.uk/en/publications/return-seasonalities-in-the-chinese-stock-market/) ([DOI](https://doi.org/10.1016/j.pacfin.2024.102391)) | `cn_same_month_seasonality_5y`: rank the mean PIT equal-weight-market-adjusted return from the same calendar month in the prior one to five years, requiring at least three annual observations | Implementable, research-only; direct China-market evidence |
| [Heston and Sadka, Seasonality in the cross-section of stock returns](https://www.sciencedirect.com/science/article/pii/S0304405X0700195X) ([author copy](https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/063f-sadka.pdf)) | Foundational same-calendar-month lag construction | Method reference |
| [Da, Gurun and Warachka, Frog in the Pan: Continuous Information and Momentum](https://academicweb.nd.edu/~zda/Frog.pdf) ([DOI](https://doi.org/10.1093/rfs/hhu003)) | Source `ID=sign(PRET) * (% negative formation days - % positive formation days)`; `cn_fip_continuous_direction_12m` translates the source double-sort to `sign(PRET) * (1-ID) / 2` over 231 sessions after a 21-session skip | Implementable causal daily translation; foundational cross-sectional method, not an exact source portfolio replication |
| [Zhang, Chen and Feng, Information Discreteness and Stock Market Returns: Evidence from China](https://www.tandfonline.com/doi/full/10.1080/1540496X.2026.2641079) ([DOI](https://doi.org/10.1080/1540496X.2026.2641079)) | Information discreteness negatively predicts aggregate China excess market returns | China market-relevance evidence only; not direct cross-sectional A-share support |
| [Atilgan, Bali, Demirtas and Gunaydin, Left-tail momentum](https://www.sciencedirect.com/science/article/pii/S0304405X19301795) ([DOI](https://doi.org/10.1016/j.jfineco.2019.07.006)) | Foundational nonparametric lower-tail method using prior-year daily returns | Method reference; China direction is taken only from the direct China source |
| [Zhen, Ruan and Zhang, Left-tail risk in China](https://www.sciencedirect.com/science/article/pii/S0927538X20301797) ([DOI](https://doi.org/10.1016/j.pacfin.2020.101391)) | `cn_low_left_tail_var1_250d`: `cs_rank(-VaR1)`, where `VaR1=-q1%(daily return)` over 250 trading days with at least 200 observations | Direct China support and exactly computable; stopped because correlation with downside volatility is 0.7405 |
| [Lin, Residual momentum and the cross-section of stock returns: Chinese evidence](https://www.sciencedirect.com/science/article/pii/S1544612318303325) ([DOI](https://doi.org/10.1016/j.frl.2018.07.009)) | Rolling China Fama-French three-factor residual momentum from monthly excess returns | Blocked: exact PIT monthly risk-free and China FF3 histories are unavailable; no equal-weight-market-only substitute |
| [Lin, Understanding idiosyncratic momentum in the Chinese stock market](https://www.sciencedirect.com/science/article/pii/S104244312100175X) ([DOI](https://doi.org/10.1016/j.intfin.2021.101469)) | Partial-least-squares aggregation of eight multifactor residual-momentum measures | Blocked: the eight model-factor histories and source PLS design inputs are unavailable |
| [Ma, Liao and Jiang, Factor momentum in the Chinese stock market](https://www.sciencedirect.com/science/article/pii/S0927539823001251) ([DOI](https://doi.org/10.1016/j.jempfin.2023.101458)) | Prior-one-year momentum of ten non-momentum factor portfolios | Factor-timing idea, not a new stock-level cross-sectional signal; blocked on factor-portfolio histories |
| [Zhang, Ma, Yang and Fan, The change in salience and the cross-section of stock returns](https://www.sciencedirect.com/science/article/pii/S0927538X24000702) ([DOI](https://doi.org/10.1016/j.pacfin.2024.102319)) | Preserve the reported negative China A-share predictor for later exact translation | Blocked: full change-in-salience aggregation and source parameters are not source-locked; no guessed formula |
| [Novy-Marx, The Other Side of Value](https://www.nber.org/papers/w15940) | gross profit / assets | Blocked: canonical daily mart lacks raw PIT gross-profit and total-assets history |
| [Hou, Xue and Zhang, Digesting Anomalies](https://academic.oup.com/rfs/article/28/3/650/1574802) | investment and profitability factors | Blocked: canonical daily mart lacks sufficiently bound PIT asset-growth history |

The China value finding changes the next test materially: do not substitute
book-to-price or a small-cap tilt for earnings yield, and do not let the
smallest shell-value segment drive the result.

For low risk, the China evidence is deliberately treated as conflicting rather
than averaged away. Raw beta remains computable, but it cannot become a formal
candidate unless it adds value beyond realized volatility, MAX/idiosyncratic
risk, size, sector, disagreement, and arbitrage-limit states. The current
canonical inputs do not contain the last two PIT state variables. The cleaner
test is therefore the directly supported low-MAX signal, with raw beta retained
only as a diagnostic control.

Conventional total skewness is kept separate from both low MAX and
distributional tail asymmetry. The local signal uses unbiased sample skewness
of daily adjusted total returns over 20 sessions, requires at least 15
observations, negates it, and cross-sectionally ranks it. The later formal test
must show incremental value beyond both sibling candidates, volatility, and
short-term momentum. It also locks a PIT 52-week-anchor interaction:
cross-sectional terciles of distance from the adjusted-close high over exact
sessions `t-251...t` must be reported separately, and the low-skewness spread
in the far-below-high tercile must exceed the near-high tercile. Sentiment and
arbitrage-risk states are mandatory only when valid PIT inputs exist; missing
inputs cannot be filled with inferred or non-PIT proxies.
Chen, Wu and Zhu is retained as adverse definition evidence for this candidate:
its China result favors a distributional asymmetry measure and does not make
conventional skewness interchangeable with that measure.

Distributional tail asymmetry is kept separate from low MAX and conventional
skewness. The local signal
standardizes each stock's PIT equal-weight-market-adjusted daily returns over
252 sessions, computes `P(z>1)-P(z<-1)`, negates it, and cross-sectionally
ranks it. It must not be described as conventional skewness or as an exact
replication of the paper's CH-3/CH-4 idiosyncratic IE construction. A formal
test must show incremental value beyond low MAX, realized/downside volatility,
and short-term momentum.

VaR1 is kept as an exact but stopped negative result. The local signal uses the
source's 250-trading-day window, 200-observation floor, and first daily-return
percentile; because `VaR1` is the negative of that percentile, ranking the
percentile directly prefers low left-tail risk. Governance measurement is at
closed month ends even though the pure helper can emit a causal daily series.
VaR5 and ES1 are robustness statistics only and cannot replace VaR1 after
measurement. The 0.7405 correlation with downside volatility makes its draft
policy `INAPPLICABLE_STOP_HIGH_CORRELATION`; neither the direct China result nor
97.77% coverage overrides dedup.

Residual momentum is not approximated with an equal-weight-market residual:
the directly supported China construction requires monthly excess returns,
rolling China FF3 regressions, a risk-free series, and factor histories.
Factor momentum is assigned to a separate future factor-timing lane rather
than mislabeled as a stock signal. Change in salience remains blocked until the
complete aggregation and parameters can be locked from a primary source.

The China seasonality paper also reports reversal outside the matching
calendar month. A later preregistration must therefore lock the matching-month
horizon as the target outcome and report other-month reversal separately as
adverse/mechanism evidence; pooling both horizons would change the hypothesis
after source review.

## Required Factor v4 route

1. Publish FIP continuous direction, conventional total skewness, and tail
   asymmetry only on a
   strict-full-A cutoff strictly after their 2026-07-28 selection date. Each
   owner-private preregistration must use zero initial weight, immutable
   definition identity, exact code/data hashes, and no inherited source-paper
   or local outcome statistics.
2. Open the 30-session embargo, then collect at least 240 post-embargo open
   sessions and 12 closed month ends.
3. Run the eight formal gates: data safety; coverage/stability; IC/RankIC;
   grouped returns; turnover/cost/capacity; sector/size neutralization; purged
   and embargoed walk-forward OOS; incremental canonical A/B/C/D replay.
4. Apply family-level multiple-testing control and structural/formal/high-
   correlation dedup. FIP must be tested against 20/120-session momentum,
   52-week high, price efficiency, short reversal, and all five protected
   v4.4 definitions. Total skewness and tail asymmetry must be tested against
   one another, the stopped low-MAX control, existing downside volatility,
   20-session momentum, volatility penalty, and all five protected v4.4
   definitions. Raw low beta stays outside the formal family unless a future
   cycle resolves its China-mechanism conflict. Price delay stays blocked until
   an approved strict source supplies enough historical PIT market-cap
   observations; it may not fall back to equal weights or inferred market caps.
   The 52-week-high candidate is retired from this cycle because its 0.7134
   diagnostic correlation with 120-session momentum exceeds the 0.70 stop;
   acquiring China EPU data cannot reverse that same-cycle stop.
   VaR1 is likewise retired from this cycle because its 0.7405 correlation
   with downside volatility breaches the same fixed line. VaR5, ES1, a shorter
   window, or a looser threshold would constitute a new cycle rather than a
   repair of this stopped candidate.
5. Only an accepted candidate-admission record, transaction plan, apply
   result, activation receipt, and fresh readiness readback may grant
   production authority.

No literature result substitutes for these gates. Statistical significance in
the historical 230/267-candidate screens is discovery evidence only.

## No-label strict-Parquet diagnostic

The current owner-private diagnostic
`reports/factor_governance/private/factor_v4_literature_incubator/factor_v4_literature_incubator_20260728T1748CST.json`
is bound to market snapshot `20260728T024704Z`, latest trade date
`2026-07-27`, PIT membership SHA
`639d8051340f685789885a3771129daf6147fa48e16ae540847ef7e6c164a03d`,
Fundamental generation `cn_fundamental_primary_20260714_v3_barbound`, and the
validated 267-factor catalog/ontology byte SHAs
`09cb6ac73590a48e826845f608e4bd733e27c183b6abaa2079436ba5bb2169ee`
and
`d734e2a397b33213e2417592699963f7a9caf7a3e5012be989943e968087159f`.
Its semantic SHA is
`3384255d226eecec1d5d5978282d9a671e1cc841a85be001ecfa334c50d53011`;
its canonical file SHA is
`a2383d9dcde59d93b8cb3d9e963b5819ada32e92293ea8ca9b2faaf33169e1c4`.
An independent repeat at `20260728T1749CST` is byte-identical.

| Candidate | Last computable date | Latest finite / eligible | Latest coverage |
| --- | --- | ---: | ---: |
| `cn_earnings_yield_ex_shell_30pct` | 2026-07-27 | 3,057 / 5,528 | 55.30% |
| `cn_fip_continuous_direction_12m` | 2026-07-27 | 5,365 / 5,528 | 97.05% |
| `cn_52_week_high_momentum_12m` | 2026-07-27 | 5,393 / 5,528 | 97.56% |
| `cn_high_price_delay_d1_52w` | unavailable | 0 / 0 | 0.00% |
| `cn_low_beta_252d` | 2026-07-27 | 5,452 / 5,528 | 98.63% |
| `cn_low_max_return_20d` | 2026-07-27 | 5,505 / 5,528 | 99.58% |
| `cn_low_market_adjusted_tail_asymmetry_252d` | 2026-07-27 | 5,452 / 5,528 | 98.63% |
| `cn_low_left_tail_var1_250d` | 2026-07-27 | 5,405 / 5,528 | 97.77% |
| `cn_low_total_skewness_20d` | 2026-07-27 | 5,499 / 5,528 | 99.48% |
| `cn_quality_cash_low_leverage` | 2026-07-14 | 4,011 / 5,528 | 72.56% |
| `cn_same_month_seasonality_5y` | 2026-07-27 | 5,113 / 5,528 | 92.49% |

The quality date is thirteen calendar days behind the market input because the
bound Fundamental generation stops at 2026-07-14. Six serving symbols without
rows in the bound PIT membership were excluded; missing PIT rows were not
failed open.

All eleven definitions passed structural checks against the 267-row catalog and
the protected v4.4 exact-five identities and slots: there was no identical
definition SHA, exact primitive set, or protected slot collision. The
diagnostic also recomputed all five protected signals through both the frozen
source DAG and an independent local engine and proved exact frame equality
before applying their frozen directions. This structural pass does not
override the economic-correlation stop on low MAX.

| Candidate | Routed comparison | Valid closed months | Median monthly absolute Spearman | Decision |
| --- | --- | ---: | ---: | --- |
| `cn_earnings_yield_ex_shell_30pct` | `fund_fcf_to_price` | 1 | unavailable | Wait for at least 3 common closed months |
| `cn_fip_continuous_direction_12m` | `pv_momentum_120d` | 28 | 0.3779 | Continue |
| `cn_fip_continuous_direction_12m` | `cn_52_week_high_momentum_12m` | 4 | 0.3972 | Continue |
| `cn_fip_continuous_direction_12m` | protected exact-five, maximum route | 13 | 0.1704 | Continue |
| `cn_52_week_high_momentum_12m` | `pv_momentum_120d` | 4 | 0.7134 | Stop |
| `cn_52_week_high_momentum_12m` | protected `alpha_range_position_momentum_20d` | 4 | 0.4898 | Below line |
| `cn_high_price_delay_d1_52w` | exact value-weighted design history | 0 | unavailable | Block: at most 10 valid design observations versus 40 required |
| `cn_low_beta_252d` | `pv_downside_volatility_60d` | 8 | 0.4986 | Control only: China mechanism conflict |
| `cn_low_beta_252d` | `pv_volatility_penalty_60d` | 8 | 0.4809 | Control only: China mechanism conflict |
| `cn_low_max_return_20d` | `pv_volatility_penalty_60d` | 14 | 0.6692 | Below line |
| `cn_low_max_return_20d` | protected `pv_low_overnight_gap_20d` | 13 | 0.6857 | Below line, close |
| `cn_low_max_return_20d` | protected `pv_low_vol_of_vol_20d` | 13 | 0.8188 | Stop |
| `cn_low_left_tail_var1_250d` | `pv_downside_volatility_60d` | 4 | 0.7405 | Stop |
| `cn_low_left_tail_var1_250d` | `pv_volatility_penalty_60d` | 4 | 0.6814 | Below line, close |
| `cn_low_left_tail_var1_250d` | `cn_low_max_return_20d` | 4 | 0.4153 | Below line |
| `cn_low_left_tail_var1_250d` | `cn_low_market_adjusted_tail_asymmetry_252d` | 4 | 0.2255 | Below line |
| `cn_low_left_tail_var1_250d` | protected exact-five, maximum route | 4 | 0.5219 | Below line |
| `cn_low_market_adjusted_tail_asymmetry_252d` | `cn_low_max_return_20d` | 8 | 0.1895 | Continue |
| `cn_low_market_adjusted_tail_asymmetry_252d` | `pv_downside_volatility_60d` | 8 | 0.2153 | Continue |
| `cn_low_market_adjusted_tail_asymmetry_252d` | `pv_momentum_20d` | 8 | 0.0772 | Continue |
| `cn_low_market_adjusted_tail_asymmetry_252d` | `pv_volatility_penalty_60d` | 8 | 0.2402 | Continue |
| `cn_low_market_adjusted_tail_asymmetry_252d` | protected exact-five, maximum route | 8 | 0.2134 | Continue |
| `cn_low_total_skewness_20d` | `cn_low_market_adjusted_tail_asymmetry_252d` | 8 | 0.0281 | Continue |
| `cn_low_total_skewness_20d` | `cn_low_max_return_20d` | 14 | 0.6139 | Continue |
| `cn_low_total_skewness_20d` | `pv_downside_volatility_60d` | 14 | 0.0647 | Continue |
| `cn_low_total_skewness_20d` | `pv_momentum_20d` | 14 | 0.3194 | Continue |
| `cn_low_total_skewness_20d` | `pv_volatility_penalty_60d` | 14 | 0.1814 | Continue |
| `cn_low_total_skewness_20d` | protected exact-five, maximum route | 13 | 0.3074 | Continue |
| `cn_quality_cash_low_leverage` | `formula_cash_growth_lowlev_w50` | 14 | 0.8514 | Stop |
| `cn_quality_cash_low_leverage` | `fund_quality_cash_combo` | 14 | 0.7616 | Stop |
| `cn_quality_cash_low_leverage` | `fund_quality_low_leverage_combo` | 14 | 0.7707 | Stop |
| `cn_same_month_seasonality_5y` | `pv_momentum_20d` | 6 | 0.0598 | Dedup pass; wait for coverage |
| `cn_same_month_seasonality_5y` | `pv_momentum_120d` | 6 | 0.0727 | Dedup pass; wait for coverage |

The 0.70 cutoff, minimum three closed months, and minimum 20 common symbols
match the formal dedup metric shape. This artifact is still diagnostic-only:
no labels or forward returns were loaded, no outcome statistics or family BH
were computed, and no formal dedup evidence, preregistration, or Factor v4
gate result was created.

Low MAX has definition identity
`c38f316ca1c68f01291a49e6708e44532317c871af0d4501d46dc9baea076037`.
Its embedded draft preregistration policy has semantic SHA
`0d3fd08d73c89689d3caba740c111be6fbb3a551a7ee94c90c68044e91f4c3c9`.
The policy locks the monthly primary test, a one-open-session execution lag,
20-open-session forward horizon, 30-session embargo, 240-session and
12-month-end maturity, family BH `q<=0.10`, exact three-factor dedup route plus
mutual dedup against tail asymmetry and total skewness, Gates 1-8, and the rule
that 15/25-session robustness variants cannot replace the 20-session primary
definition after measurement. The current report marks that historical draft
`INAPPLICABLE_STOP_HIGH_CORRELATION`; it is not a formal preregistration or
authority artifact and must not be promoted.

Price delay has definition identity
`b1a5fcbed619e689b7d98ddc47bbd7afd1ccbcff5a7f43651f0ee35e6bb2999b`.
The strict input audit found 263 Wednesday calendar anchors, but only 27 with a
minimum market-cap cross-section, 31 finite value-weighted market returns, and
at most 10 valid contemporaneous-plus-four-lag design rows in any 52-week
window. The definition requires 40. Its route is therefore
`BLOCKED_NOT_COMPUTABLE`; no policy draft was created and no equal-weight,
current-cap, or inferred-cap fallback was used.

The 52-week-high candidate has definition identity
`40d77e7cc6da81ca3c9fa8e0fded069ea94debd4180b4027d6e1ea35671fb499`.
It is computable for 5,393 of 5,528 eligible names on 2026-07-27 and has no
definition, primitive, family, slot, or protected exact-five identity
collision. Its four valid closed-month absolute Spearman correlations with
`pv_momentum_120d` are 0.5739, 0.6889, 0.7379, and 0.8368; the median is
0.7134, so the route is `STOP_HIGH_CORRELATION`. The China source also requires
low/moderate/high EPU state reporting, but the strict repository lacks a
release-lag-bound PIT China EPU series. Both limitations are recorded without
substituting inferred EPU or weakening the dedup threshold.

Total skewness has definition identity
`61f9decdf50929402f919a23330c32150cc34cc533ae5478b5b27f3a83fa02af`.
Its embedded draft preregistration policy has semantic SHA
`18dd3a66cc6681ac52a743cb27a490dea9e46aa5bab8eec2c0e9f3a96321c325`.
The policy locks the 20-session, minimum-15-observation conventional skewness
definition, mutual dedup against low MAX and distributional tail asymmetry, a
PIT 52-week-high tercile interaction, and strict handling of unavailable
sentiment or arbitrage-risk inputs. The 15/25-session robustness variants
cannot replace the primary after measurement.

Tail asymmetry has definition identity
`8d4437da6a3d4e7758a371922ad96849ebbab35f5799a83364b48856fb057dd3`.
Its embedded draft preregistration policy has semantic SHA
`6511448e0748c3696ab8fa2490e9cc8dad6fbffce93ba8aea79ab9be31ecc4b6`.
The policy locks the 252-session, one-sigma proxy as primary, explicitly
disclaims exact CH-3/CH-4 replication, requires mutual dedup and incremental
value beyond low MAX and total skewness, and prevents 126/504-session or
0.75/1.25-sigma robustness variants from replacing the primary after
measurement.

FIP continuous direction has definition identity
`50774665658fb239c4fcddfb4d4106932d800b2eb5c2a4b64c16d1dd5d98f9b3`.
Its embedded draft preregistration policy has semantic SHA
`d12673621d6da27c078165dbd1dd458e649f2263a452b9ecba969faf48246c3c`.
The policy locks the source ID formula, the 231-session formation period,
21-session skip, monthly primary measurement, one-session execution lag,
20-session forward horizon, 30-session embargo, 240-session/12-month maturity,
family BH `q<=0.10`, and Gates 1-8. It requires separate winner and loser
results plus zero-return, price-limit, suspension, and relisting sensitivity.
It explicitly disclaims exact replication of the source monthly double-sort
and disallows treating the 2026 China aggregate-market result as
cross-sectional support. The 210/252 formation and 20/22 skip variants are
robustness checks only and cannot replace the frozen primary after
measurement.

VaR1 has definition identity
`1eb65dbbd5bc33c6fddd9196fda08026faa10b1de7181e0a38ec27f1444824a9`.
Its embedded draft policy SHA is
`23e25f0cb366f637aa647b641ab5bc9e9f8e711690965b329329155e9162b7c0`.
The policy locks the exact 250/200/1% construction, monthly measurement,
one-session execution lag, 20-session forward horizon, 30-session embargo,
240-session/12-month maturity, family BH `q<=0.10`, Gates 1-8, and all sibling,
formal, and protected-factor dedup routes. The report marks it
`INAPPLICABLE_STOP_HIGH_CORRELATION`; VaR5 and ES1 cannot rescue or replace the
stopped primary definition.

Seasonality also has a separate 80% closed-month coverage floor by calendar
month. Only March, April, May, and June currently have at least one stable
closed-month observation; January, February, and July through December are
missing. The latest July signal is computable for 5,113 names, but that single
date does not cure the cross-calendar instability. Its current route is
`WAITING_FOR_12_STABLE_CALENDAR_MONTHS`, not preregistration.
