# Phase12 — current automation boundary audit

Status: READ_ONLY_AUDIT. No task, schedule, model, deployment or credentials changed.
Exact saved configs and hashes: `.agent/acceptance/phase12-automation-before.json`.
The app's view operation recognized all four IDs and displayed their cards; it
did not return effective unattended permissions or a new scheduler-run receipt.

| ID | Existing task name | Saved cadence | Current route |
| --- | --- | --- | --- |
| automation | A股量化投资与日度复盘 | Weekdays09:45 | DUAL_RUN Morning snapshot; no maintenance; old consumer/manual review sequence |
| cn-daily-data-update | A股收盘生产与数据更新 | Weekdays16:20,17:20,18:20,20:20 | First three slots maintenance only;20:20 native Factor loop then separately described downstream closure |
| cn-evening-review-fallback | cn-evening-review-fallback | Weekdays21:00 | Recover existing logical2020 slot; completed work verify/NO_ACTION |
| cn-dashboard | CN aggressive 持仓与业绩 Dashboard | Weekdays21:30 | Explicit read-only portfolio/report consumer; no external data or Store writer |

All four saved configs use gpt-6-astra and the myQuant project. Preserve configured
model, reasoning effort, notification preference and unrelated weekly/A_quant tasks.
The existing myQuant ten-day observation heartbeat is separate from these daily
production responsibilities and is not an inferred new scheduling target.

`scripts/operations/run_cn_daily_slot.sh` currently exposes maintenance/scope-
transition/Factor-context arguments. It verifies installed import origin, records
launcher STARTED/ENDED, performs the existing pre-credential recovery path and
uses PROJECT_ENV credential preflight before `market daily-maintain`. It does
not yet invoke the new automatic `production daily-close` contract. The saved
producer/fallback prompts still bind the frozen7b26ac2 Factor context. Updating
prompt wording alone would therefore leave the execution path incomplete.

The saved producer/fallback prompts contain owner standing authorization for
their exact scheduled slots, official Tushare endpoint and specified existing
maintenance/close/research/publication writes. They explicitly exclude ad-hoc
off-slot expansion, custom endpoints, brokers/orders/trades, actual portfolio
mutation and System/Mainline changes. Preserve that boundary; do not infer an
unattended success from ACTIVE config or rerun live providers during local tests.
No credential value was read during this audit.

The source design calls for EOD production,21:00 fallback and09:45 Morning as the
three daily responsibilities. Required preparation is to repair the existing
launcher path, supply exact immutable Calendar/recipe/native-input prerequisites
and make EOD/fallback reuse one automatic request. Morning must consume the prior
completed EOD and its v3 policy/quote contract without maintenance. A concrete
cutover must address the existing21:30 read-only task explicitly and preserve its
authorization until that cutover is reviewable. Do not create duplicate tasks.

New production invocation must wait for a matching current-source installed
release and real input provisioning. Missing named event/owner evidence remains
missing; it cannot be replaced with invented CLOSED_EMPTY events, copied stale
inputs or a manually repaired SHA. Phase7/manual-event acceptance and final
Phase15 installed DAG checks remain separate requirements.

The official scheduling documentation supports testing the prompt first and
reviewing initial scheduled results; local-project runs also depend on the local
machine/app and configured permissions. These platform facts do not establish
this project's effective scheduler state. See [official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app).

Next: inspect the exact retained installed context/launcher receipt contracts and
existing request-input builders; prepare a minimal reviewed launcher/input and
three-role cutover plan, validate offline, then perform only the authorized
concrete scheduler transition. No new task or recurring monitor has been created.
