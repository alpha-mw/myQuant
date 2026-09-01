---
name: myquant
description: Use for /Users/maxwell/mySpace/myQuant A-share investment research and portfolio work, including full-market candidate discovery, stable factor research and governance, fundamental, Industry and Theme evidence, deterministic investment decisions, skeptical review, portfolio review, paper evaluation, outcome learning, and exact mainline reads.
---

# myQuant Investment Research

Operate `/Users/maxwell/mySpace/myQuant` as one stable A-share investment
research and portfolio-decision system. Investment judgment and portfolio
improvement are the objective; data, packaging, governance, publication, and
operations support that objective.

## Mission

Improve expected portfolio return after drawdown, liquidity, concentration,
turnover, valuation, evidence, and execution costs.

The intended research loop is:

```text
actual holdings + exact investable A-share universe
  -> verified system + governed admitted factor set
  -> source-bound Industry, Theme, Fundamental, and Macro evidence
  -> deterministic evidence closure and five-state decision
  -> skeptical advisory review
  -> constrained research portfolio and paper evaluation
  -> outcome review and explicit learning proposal
```

Use a capability only when `quant-investor system verify`, system status, the
intended checkout, and focused tests establish that exact capability. Otherwise
label it unavailable and continue only with clearly marked research diagnosis.

## Start with the investment question

- **Market opportunity:** discover and compare current A-share candidates.
- **Company research:** test the thesis, evidence, valuation, catalysts,
  Industry/Theme position, and downside.
- **Portfolio review:** evaluate exact holdings, cash, concentration,
  replacements, and action priorities.
- **Factor research:** assess robust incremental value and prospective evidence,
  not historical fit alone.
- **Paper review and learning:** preserve decisions, measure matured outcomes,
  and propose research or discipline improvements.
- **Maintenance:** perform system work only when explicitly requested.

Read `references/investment-research-and-portfolio.md` for investment work and
`references/operations-and-verification.md` when operational checks matter.

## Decision model

- **Quant** owns the exact full-market pool and deterministic ranking. Only a
  governed admitted factor set may feed it; diagnostics are not candidates.
- **Industry** supplies source-bound identity and evaluation. Unmapped or
  ambiguous identity is missing evidence.
- **Theme** supplies memberships, exposure, lifecycle, and concentration risk.
  Missing membership is not neutral alpha.
- **Fundamental** tests business quality, earnings, valuation, balance sheet,
  catalysts, governance, and evidence freshness.
- **Macro/regime** may tighten cash, gross exposure, and risk budgets or apply a
  veto. It cannot create company alpha or loosen hard limits.
- **Decision** binds exact hypotheses, evidence, risks, vetoes, and one of five
  research states.
- **Skeptical review** may challenge the deterministic result from validated
  facts. It is advisory-only.
- **Portfolio/paper** starts from eligible research decisions and exact holdings
  to build constrained research portfolios and paper outcomes without broker
  authority.

Record disagreements rather than averaging them away. Deterministic evidence,
Decision, portfolio, and risk controls remain authoritative.

## Research states and authority

The five research states are `THESIS_INVALIDATED`, `INSUFFICIENT_EVIDENCE`,
`WATCHLIST`, `RESEARCH_APPROVED`, and `PAPER_CANDIDATE`.

They are not buy, sell, hold, target-price, position, portfolio-admission,
order, or trade instructions. `PAPER_CANDIDATE` means only eligibility for the
next separately governed research or paper stage.

Public `QuantInvestor`, `research run`, `market analyze`, and `market run`
surfaces read one already-active, exactly validated generation. They do not
build, publish, replace, or activate one. `quant-investor system activate` is
the only normal `results/system/_active.json` writer and accepts only an exact
validated immutable generation with filesystem write permission. A builder,
fixture, clean checkout, or passing test does not prove operational activation.

Factor production authority is independent from System and Mainline. Read it
through `factor production-status`, `factor production-verify`, and
`factor production-signal`; LOW/W80 may be active while System and Mainline are
uninitialized. W75 is control-only. `production-activate --expected-empty`
creates only the immutable genesis pointer; `production-rollover` alone may
advance an existing Factor head from an exact execute-mode PIT/Market/History
receipt and expected pointer SHA. Neither grants portfolio or trading
authority.

Daily immutable Factor generations are production signal history, not the
registered prospective admission lifecycle. `factor production-observe` may
append exact-once OPEN LOW/W80 production observations after a verified daily
head. These are non-authorizing preregistration evidence, not the mature
`factor.prospective_observation` used for admission. Report their exact state
and SHA when present, but keep outcome evidence `WAITING_FOR_FUTURE_SESSIONS`
until target sessions arrive and keep prospective admission `NOT_CONFIGURED`
unless its separate exact lifecycle exists. Do not invent IC, RankIC, horizon
outcomes, maturity, or graduation evidence. Macro and Fundamental blockers
remain visible but do not block LOW/W80 when the sealed active Factor set does
not consume them.

The live CN maintenance automation uses the release-owned
`scripts/operations/run_cn_daily_slot.sh` for all four slots. It records only a
non-secret project `.env` access receipt, never reads macOS Keychain, and may
automatically recover only an exact zero-write `TUSHARE_TOKEN_MISSING` veto with
unchanged Market/PIT/Factor/Store preimages. The `.env` must be owner-only mode
`0600` with exactly one valid `TUSHARE_TOKEN`; it is parsed as data and never
sourced. Security, schema, lineage, partial-write and pointer-drift vetoes remain
operator-only.

Morning strategy is a separate research-only read lane over the prior close.
`research morning-strategy` requires prior-date Factor READY, exact OPEN LOW/W80
observations, verified Store holdings and a credential-free Sina capture from
the current trading date at or after 09:30 Asia/Shanghai. `09:47` classifies a
capture as delayed but does not block it; midday, afternoon and post-close
same-date snapshots are allowed only when their real request/provider times,
market session, timing status, delay and raw SHA remain exact and visible.
Missing Macro, Fundamental, Theme economic exposure, Top100 or
benchmark-relative evidence may make it PARTIAL; stale Factor, invalid Store or
missing quote blocks it. Its immutable receipt never grants broker, order,
execution or holdings authority. `research morning-cutover` seals the 20:20
core/auxiliary scheduling decision; source code never writes Codex scheduler
configuration directly.

CN aggressive Dashboard reads only Store-v3 and pointer-selected
`ledger_after_manual_switch.parquet`. A trailing benchmark gap may leave v1
`PARTIAL` while v2 marks current holdings/absolute performance independently;
missing relative fields stay `null/unavailable`. A failed refresh preserves the
last-good selector and appends an attempt receipt. Missing owner-controlled
same-day continuity keeps holdings `STALE`; no new Store record is not proof of
no manual trade.

Macro retrospective reconstruction is prepare-only unless an exact typed
transaction binds the execute receipt, capture inventory, per-date PIT evidence,
Market/PIT/Macro/Release pointer preimages, journal and veto. A deterministic
recovery candidate never authorizes pointer mutation or veto clearing.

PIT `stock_basic` has one separate endpoint-local evidence grammar for legacy
delisted provider identities: exact `^T[0-9]{6}\.SH$` rows in the D partition
may be retained only in the sealed exclusion inventory and never normalized,
mapped, admitted, or passed to Market/Factor. This PIT-only rule does not alter
Fundamental's provider-external identity grammar.

If active state or a required input is unavailable, keep the conclusion
`INSUFFICIENT_EVIDENCE` or a clearly labeled research diagnosis. Never recycle
an old candidate list or scan for a nearby result.

## Minimum evidence

Require only what makes the requested investment judgment credible:

- current strict CN Parquet market data and point-in-time investable membership;
- an exact decision date/cutoff and adequate symbol coverage;
- current Fundamental evidence for earnings, valuation, and quality claims;
- Industry, Theme, or Macro evidence only when it changes the thesis or risk;
- no CSV, stale, mock, inferred, cached, or latest-by-mtime fallback.

`market storage-validate` proves market/PIT health, not complete investment
readiness. Missing unrelated data may be recorded without expanding a company
question into a storage project.

## Portfolio rules

- Start from exact user-supplied holdings and cash or a verified current
  resolver. Never infer them from a recent filename or prior chat.
- Review thesis, valuation, catalysts, downside, liquidity, concentration,
  correlation, turnover, costs, and opportunity cost.
- Compare holdings with current candidates before proposing replacements.
- Distinguish hold, add, reduce, exit, watch, and insufficient evidence; give
  reasons, sizing logic, triggers, and invalidation conditions.
- Use `代码 公司名` for holdings and candidates.
- Keep current quotes, forecasts, paper fills, and broker fills separate.

An official target portfolio requires a valid active closure. Research
portfolios and paper ledgers remain labeled and cannot mutate actual holdings,
active state, or production state.

## Factor, forward research, paper, and learning

- Keep factor idea, bootstrap definition, preregistration, prospective evidence,
  admission, stock candidate, and portfolio position separate.
- Historical backtests cannot substitute for prospective matured evidence.
- Forward observations are evidence tools, never active portfolio or
  performance authority.
- Paper assumptions must be explicit and effective-dated. Missing calendar,
  T+1, lot, suspension, limit, price, fee, corporate-action, or cancellation
  semantics remain blocked.
- The independent `paper` CLI exposes the registered sell-only Paper risk-exit
  writer. Release A is code-ready but must remain `PAPER_ACCOUNT_NOT_REGISTERED`
  until a separate owner-authorized account genesis exists. `writer-status`,
  `account-status`, `risk-exit-preview`, and `verify` are read-only.
  `account-register` and `risk-exit-run` require `--allow-write`, exact
  installed-release evidence, and Paper-only pointer authority; they never
  clone actual Store, connect a broker, create a real order, or mutate actual
  holdings. Factor/Fundamental readiness is not a writer dependency; any
  upstream dependency must already be sealed in the explicit Paper intent.
- Graduation is a sealed research record and learning proposal. It does not
  silently change a factor, policy, portfolio, active generation, or Memory.

## Source and authority hygiene

- Verify the exact branch, commit, package, entrypoint, tests, and artifacts
  used for any capability claim.
- Treat modified, untracked, other-branch, fixture-only, and historical material
  as non-current until proven otherwise.
- Keep research permission, paper mutation, publication, activation, portfolio
  mutation, broker connection, order creation, and trading as separate scopes.
- Default to local, offline verification. Never call live providers or models,
  publish, activate, connect a broker, create an order, execute, or trade without
  separate exact authority.

## Reporting contract

Lead with the investment conclusion, then provide:

1. company or portfolio conclusion;
2. holdings and candidates in `代码 公司名` format;
3. thesis, catalysts, valuation, Industry/Theme context, risks, and evidence;
4. proposed action, sizing/risk logic, and invalidation conditions;
5. official versus research-only status;
6. missing evidence and the next useful investment step;
7. only operational blockers that materially affect the conclusion.

Do not bury the investment answer beneath storage, schema, Git, or publication
details.
