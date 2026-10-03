# Repointing the scheduled CN jobs to a new frozen release

The three launchd jobs (`daily-factor-loop` 20:25, `evening-close` 21:15/22:30,
`prune-market-serving` 22:00) and the Hermes contracts that still run the
installed interpreter take their release from one file:

```
operations/releases/active.env
```

It is plain `KEY=VALUE`, never evaluated, and both readers
(`scripts/operations/release_pointer.sh`, `quant_investor/operations/release_pointer.py`)
validate every reference before a job proceeds: the install and checkout
directory names must start with `RELEASE_COMMIT`, `bin/python` must exist,
exactly one `lib/python3.*/site-packages` must exist, and the factor-loop
context must hash to `FACTOR_LOOP_CONTEXT_SHA256`. A failed check is a named
`RELEASE_POINTER_*` blocker, an alert in `logs/alerts.jsonl`, and no run.

launchd executes these scripts from the main working tree, so a change to the
pointer takes effect the next time a job fires once it is in that checkout.
Keep a repoint to a commit of its own.

## Steps

1. **Land the code.** Everything the new release must carry is committed on
   `main`; the working tree you build from is clean.
2. **Build the release** from a clean detached checkout of that commit
   (`quant-investor system release-prepare`, see
   `docs/runbooks/factor_production.md`). It returns `installed_python`, the
   workspace-relative release-install input and its SHA. The install lands
   under `~/mySpace/myQuant-release-authority/<commit>-unified-runtime/installs/`,
   the checkout under `~/mySpace/myQuant-release-checkouts/<commit>-unified-runtime`.
3. **Verify the install**: `verify_running_release_install_input(...)` → `PASS`
   from the installed interpreter with an empty `PYTHONPATH`.
4. **Write the factor-loop context** for the new commit at
   `data/private/cn_daily_maintenance/factor_loop_contexts/<commit>.json`
   (schema `cn-daily-factor-loop.v3`, binding `release_commit`,
   `release_install_input_ref`, `release_repository_root` and
   `calendar_capture_parent`), owner-only permissions, and record its sha256.
   There is no writer command for this file; copy the previous context and
   change only the release fields.
5. **Migrate the retained factor-loop state** with `market native-seal` on the
   old release and `market native-continue` on the new one, following
   `docs/plans/cross_release_native_migration_protocol.md`. A refusal such as
   `CROSS_RELEASE_CAPTURE_REBIND_REQUIRED` means the retained calendar capture
   is bound to the old install: wait for the next session's core to complete
   under the old release and try again. Do not recapture.
6. **Edit `operations/releases/active.env`** in one commit: `RELEASE_COMMIT`,
   `RELEASE_INSTALL_DIR`, `RELEASE_CHECKOUT_DIR`, `RELEASE_INSTALL_INPUT_SHA256`,
   `FACTOR_LOOP_CONTEXT`, `FACTOR_LOOP_CONTEXT_SHA256`. Drop
   `PRUNE_RELEASE_INSTALL_DIR` once the active release has the
   `prune-market-serving` command. Then run both readers against it:

   ```bash
   zsh -c 'source scripts/operations/release_pointer.sh && read_release_pointer operations/releases/active.env "$PWD" && echo "$RELEASE_COMMIT $INSTALLED_PYTHON"'
   ```

   ```bash
   uv run python scripts/operations/run_cn_evening_close.py --trade-date <next session>
   ```

   The second is a plan with no writes; its receipt must carry the new commit.
7. **Repoint the Hermes contracts** in `~/.hermes/profiles/myquant/automations/`
   that still name a release: replace the paths with "read
   `operations/releases/active.env`" and remove the release section from any
   paused job that launchd has replaced.
8. **Retire.** After the first full week on the new release, keep only the two
   newest release directories and checkouts; remove the rest together with
   their worktrees (`git worktree list` / `git worktree remove`).

## Rolling back

Revert the pointer commit. The retained factor-loop state must be migrated
back the same way (step 5); the pointer alone does not move it.
