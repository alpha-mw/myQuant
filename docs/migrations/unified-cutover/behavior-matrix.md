# Migration Behavior Matrix

This matrix is the review template for every source classified by the unified
cutover rules. Add one row per source identity; never merge rows merely because
two paths look equivalent.

| Classification | Pre-cutover source | Required post-cutover behavior | Allowed data movement | Required proof |
|---|---|---|---|---|
| `replace` | A supported version-named runtime source | Resolve once to its declared stable target; the old source is absent afterward | Exact bytes or an explicitly declared deterministic transformation | Source identity, target identity, transformation receipt, positive stable test, negative legacy test |
| `archive` | Immutable historical evidence | Preserve exact bytes under the declared archive target; never import or execute them | Byte-for-byte only | Source and target byte hashes, archive containment, non-importability test |
| `remove` | Obsolete executable, alias, generated wrapper, cache, or unsupported surface | Remove it from the current runtime with no replacement fallback | None | Absence test and stable error from any former public invocation |

## Runtime behavior

| Condition | Result | Writes permitted |
|---|---|---:|
| All required stable packages and contracts verify | Stable command may perform only its documented operation | Command-specific |
| Required source or target is absent | Cutover blocked | 0 |
| Source identity is ambiguous or duplicated | Cutover blocked | 0 |
| Canonical bytes or declared hash do not match | Cutover blocked | 0 |
| Dynamic import is not explicitly allowlisted | Import blocked | 0 |
| A legacy executable or subcommand is invoked | Explicit unsupported-command failure | 0 |
| Active generation is missing or invalid | Stable read unavailable or blocked | 0 |
| Exact valid generation is activated with the expected pointer prevalue | `system activate` CAS-writes `results/system/_active.json`, retains pointer history, and reads back the exact closure | One governed pointer transaction |
| System closure is incomplete but safely diagnosable | `PARTIAL` with explicit blockers | 0 from status |
| System is intentionally suspended | `SUSPENDED`; external routing must not infer active authority | 0 from status |
| Automation routing is missing, disabled, or drifted | `UNCONFIRMED`, `SYSTEM_ACTIVE_AUTOMATION_DISABLED`, or `SYSTEM_EXTERNAL_ROUTING_DRIFT` | 0 |
| Mainline identity is absent, invalid, or not closed | `MAINLINE_UNINITIALIZED`, `MAINLINE_ARGUMENTS_INVALID`, or `MAINLINE_BLOCKED` | 0 |
| Backtest is requested on the stable mainline | `BACKTEST_UNAVAILABLE` | 0 |
| Factor evidence is incomplete or immature | Factor status reports blocked or insufficient evidence | 0 |
| Research evidence is incomplete | Research readiness/evaluation reports its blocker | 0 |
| Repository Codex deployment copy verifies | Copy is eligible for a separate external deployment review | 0 external writes |
| Exact pre-cutover dirty capture still contains any `UNCONFIRMED` row | Capture readback verifies that user files remain untouched, then reports `PRE_CUTOVER_DIRTY_INVENTORY_UNCONFIRMED` | 0 |
| Captured HEAD, porcelain status, path set, size, or bytes drift | Pre-cutover verification blocked; recapture or confirmation is never inferred | 0 |
| Clean descendant integration commit has a user-confirmed disposition and reason for every captured row | Dirty-inventory gate is satisfied; checkout is eligible only for the next preflight gate | 0 from verification; final build and CAS remain unauthorized |

## Mapping row template

Copy this row for additions to the machine-controlled source-to-target rules:

| Source identity | Classification | Stable/archive target | Byte policy | Collision policy | Positive test | Negative test | Owner |
|---|---|---|---|---|---|---|---|
| `<exact repository path or entrypoint>` | `<replace|archive|remove>` | `<exact target or none>` | `<exact|deterministic transform|none>` | `block` | `<test node>` | `<test node>` | `<module owner>` |

Every field is required. `latest`, `nearest`, wildcard import targets, inferred
hashes, and empty compatibility shims are invalid values.
