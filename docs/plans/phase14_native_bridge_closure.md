# Repair the existing native bridge import closure

Status: IMPLEMENTED_LOCAL_VALIDATION. Architect APPROVE then Critic APPROVE.
This repairs the existing Phase7/9/12 path;
it does not change operations, provider permission, publication gates or writers.

An actual archived completion replay failed in the recorded child. Its bounded
stderr is being captured to identify the exact rejection. Independent AST/source
inspection proves the current bridge's closed module map omits dependencies already
required by registered native operations. The old eaa60b2d snapshot has the same
first-level omissions. Do not modify that immutable snapshot or override its map.

Add exactly these code-owned keys to operations/native_bridge.py::_MODULES:

| Key | Existing source | Required by |
| --- | --- | --- |
| scripts.daily_dashboard_publication | scripts/daily_dashboard_publication.py | daily_catchup, daily_completion, daily_materialization |
| scripts.daily_dashboard_sealed | scripts/daily_dashboard_sealed.py | daily_completion, daily_native_registry |
| scripts.daily_store_adoption | scripts/daily_store_adoption.py | daily_materialization, daily_production_store_adapter, daily_store_materialization |
| scripts.cn_dashboard_v2_selector | scripts/cn_dashboard_v2_selector.py | transitive qualified import from Dashboard publication |
| scripts.export_cn_aggressive_dashboard_data | scripts/export_cn_aggressive_dashboard_data.py | transitive qualified import from Dashboard publication |

The latter two files already have their bare-name entries. Preserve both names
because their callers use both explicit forms; this is the existing closed-loader
pattern, not a dynamic alias or fallback. The three new files contain definitions
and constants at import, no top-level provider/financial/publication invocation.
Native validation, exact Git blob verification, preload audit, sealed late-import
refusal, context isolation and fixed _OPERATIONS stay intact. No unknown module
discovery, automatic allowlist generation, retired imports or arbitrary operation.

Verification: one bounded read-only source closure audit; existing closed-loader
mechanics including dirty/preloaded/late/unlisted rejection; a fixture copying the
exact current closed source set into a disposable Git repository and preloading
the full real module list with only the installed-runtime verifier seam explicitly
controlled. Forbid provider/network and canonical writers during this import-only
test, assert cleanup and fixed operation availability. This is import-closure
proof, not installed release or full-DAG acceptance. Existing shadow/synthetic
outcome/native-route regressions remain. Final current-source installed/public
acceptance stays Phase15. An old release that cannot load its own fixed bridge
remains INVALID native evidence; no current-source retry or source substitution.

Exact archived trace confirmed: replay_native_completion -> replay_completed_ledger
-> daily_materialization._recorded:395 imports scripts.daily_store_adoption and
the old installed bridge rejects NATIVE_BRIDGE_MODULE_NOT_ALLOWED. Diagnostic
`.agent/acceptance/phase14-archived-native-diagnostic.json` and bounded stderr
`.agent/acceptance/phase14-archived-child-stderr.txt` are retained. All14,862 workspace
files stayed byte/mtime identical in the first readback. Both old calls terminal;
do not repeat them.

Architect APPROVE: the full-preload fixture must keep real Git blob verification,
finder/audit and second source recheck; verify every mapped origin, equal bytes for
aliases, unchanged operations keys/functions, strict unlisted/preloaded rejection,
zero provider/canonical writers, and full sys.modules/path/meta/cache/prefix cleanup.
