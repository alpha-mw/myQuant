# Repoint to commit 2259530 after the 2026-10-08 session

This is the deferred half of `docs/runbooks/release_repoint.md`. The generic
runbook still governs; this file fixes the values and the ordering for this
particular switch.

## Why it is deferred

Step 5 (native migration) seals the retained `factor-loop-state.json`, and
`market native-seal` replays that state's published next-session proof. The
retained state is the 2026-09-30 session, whose capture was published before the
2026-10-03 23:23 reboot. The capture records `published_root_device` 16777230
while APFS now reports 16777231 for the volume (inodes are unchanged), so
`validate_published_trusted_provider_calendar_capture_root` refuses it:

```text
SYSTEM_STORAGE_SECURITY: trusted-provider published root identity differs
```

Repointing `active.env` without that migration makes the next session fail
closed: `daily_factor_loop.py` validates the retained state against the *active*
context sha and raises `DAILY_FACTOR_STATE_INVALID` when they differ.

Relaxing the device comparison is a security-control change and is **not**
authorized; instead the fix is to wait for a session whose capture was published
under the current boot, then migrate.

## Preconditions

1. The 2026-10-08 core session completed under the **old** release (660f066) and
   published a fresh next-session capture. Verify:

   ```bash
   cd /Users/maxwell/mySpace/myQuant
   python3 -c "import json;d=json.load(open('data/private/cn_daily_maintenance/factor-loop-state.json'));print(d['trade_date'], d['phase'], d['context_sha256'])"
   ```

   Expect `20261008` with a terminal phase and `context_sha256` equal to the old
   context (864e4e78…). The state's `next_session_calendar_proof_ref` must point
   at a capture whose `capture-success.json` records the device the volume
   reports **now** (`stat -f %d` on that capture directory).

2. The target release is installed and verified (built 2026-10-04 from commit
   `2259530d7d5dfd11215f9f804306c06a32444e7e`):

   ```text
   install   /Users/maxwell/mySpace/myQuant-release-authority/2259530d7d5dfd11215f9f804306c06a32444e7e-unified-runtime/installs/2259530d7d5dfd11215f9f804306c06a32444e7e-d8748d0bf32228e78cd9070568a3bd20ac84e226c68e3c62936232686a6ea2ef
   checkout  /Users/maxwell/mySpace/myQuant-release-checkouts/2259530d7d5dfd11215f9f804306c06a32444e7e-unified-runtime
   input     results/releases/7b88feb4beb0f7c389e4da9359bc2aed29f724a17c0f7d612090a849493d62ce/release-install-input.json
   input sha 7b88feb4beb0f7c389e4da9359bc2aed29f724a17c0f7d612090a849493d62ce
   context   data/private/cn_daily_maintenance/factor_loop_contexts/2259530d7d5dfd11215f9f804306c06a32444e7e.json (0600)
   context sha 3cc62a7e9e905ef316bfd214250794c29680e2f12d230b547082258d3c47df10
   ```

## Steps

Set the shell variables once; every command below is a single command (no shell
concatenation inside the CLI invocations).

```bash
CP=/Users/maxwell/mySpace/myQuant/data/private/cn_daily_maintenance
FROM_STATE_SHA=$(shasum -a 256 "$CP/factor-loop-state.json" | cut -d' ' -f1)
OLD_INSTALL=/Users/maxwell/mySpace/myQuant-release-authority/660f066fb11e3bc100c89992a989f844c0c3a975-unified-runtime/installs/660f066fb11e3bc100c89992a989f844c0c3a975-f29e64e2eb6adcb364e9a54b9b8f2e72fd6408e68d88097f134c2b9eb03e4cc6
NEW_INSTALL=/Users/maxwell/mySpace/myQuant-release-authority/2259530d7d5dfd11215f9f804306c06a32444e7e-unified-runtime/installs/2259530d7d5dfd11215f9f804306c06a32444e7e-d8748d0bf32228e78cd9070568a3bd20ac84e226c68e3c62936232686a6ea2ef
```

1. **Seal the retained state** (old release, owner-only receipt in the run root):

   ```bash
   PYTHONPATH= $OLD_INSTALL/bin/python -I -m quant_investor market native-seal \
     --workspace-root /Users/maxwell/mySpace/myQuant \
     --run-root "$CP" \
     --from-state-sha256 "$FROM_STATE_SHA" \
     --from-context-sha256 864e4e7801549b74e499e563cc3f8d2b666027edcc3760609afb020ba4945053 \
     --from-release-install-input-ref '{"path":"results/releases/f2a06c61cdc73e03f300044692e34d202e41c6dbbcb3bab0e7dec556dd6f725a/release-install-input.json","sha256":"f2a06c61cdc73e03f300044692e34d202e41c6dbbcb3bab0e7dec556dd6f725a"}' \
     --from-trade-date 20261008 \
     --from-mode PRODUCTION_INSTALLED_CAPTURE
   ```

   Expect `{"status":"SEALED","seal_receipt_ref":{...}}`. `NO_ACTION` with the
   same receipt is idempotent success. The receipt path is
   `$CP/factor-state-seal-$FROM_STATE_SHA.json`; copy its `seal_receipt_ref`.

2. **Continue onto the new release** (new install):

   ```bash
   PYTHONPATH= $NEW_INSTALL/bin/python -I -m quant_investor market native-continue \
     --workspace-root /Users/maxwell/mySpace/myQuant \
     --run-root "$CP" \
     --seal-receipt-ref '{"path":"<seal receipt path>","sha256":"<seal receipt sha>"}' \
     --to-release-commit 2259530d7d5dfd11215f9f804306c06a32444e7e \
     --to-release-install-input-ref '{"path":"results/releases/7b88feb4beb0f7c389e4da9359bc2aed29f724a17c0f7d612090a849493d62ce/release-install-input.json","sha256":"7b88feb4beb0f7c389e4da9359bc2aed29f724a17c0f7d612090a849493d62ce"}' \
     --to-context-ref '{"path":"data/private/cn_daily_maintenance/factor_loop_contexts/2259530d7d5dfd11215f9f804306c06a32444e7e.json","sha256":"3cc62a7e9e905ef316bfd214250794c29680e2f12d230b547082258d3c47df10"}' \
     --to-repository-root /Users/maxwell/mySpace/myQuant-release-checkouts/2259530d7d5dfd11215f9f804306c06a32444e7e-unified-runtime
   ```

   A refusal naming a capture rebind means the precondition in §Preconditions 1
   is not met — do not recapture, wait for the next session and retry.

3. **Confirm the state rebinds**: re-read `$CP/factor-loop-state.json`; its
   `context_sha256` must now be `3cc62a7e…` and the previous state must survive
   as an immutable `factor-state-<sha>.json` sibling.

4. **Edit `operations/releases/active.env`** in one commit: `RELEASE_COMMIT`,
   `RELEASE_INSTALL_DIR`, `RELEASE_CHECKOUT_DIR`, `RELEASE_INSTALL_INPUT_SHA256`,
   `FACTOR_LOOP_CONTEXT`, `FACTOR_LOOP_CONTEXT_SHA256`. Commit 2259530 carries
   `prune_snapshot_serving_layers`, so **drop `PRUNE_RELEASE_INSTALL_DIR`** and
   its comment.

5. **Verify the pointer** with both readers:

   ```bash
   zsh -c 'source scripts/operations/release_pointer.sh && read_release_pointer operations/releases/active.env "$PWD" && echo "$RELEASE_COMMIT $INSTALLED_PYTHON"'
   ```

   ```bash
   uv run python scripts/operations/run_cn_evening_close.py --trade-date <next session>
   ```

   The second is a plan with no writes; its receipt must name commit `2259530`.

## Rolling back

Revert the pointer commit, then migrate the retained state back the same way
(seal under the new release, continue onto the old release/context). The pointer
alone does not move the retained state.

## Known follow-up

The `published_root_device` comparison will keep failing for every artifact
published before the most recent reboot, and the same pattern may exist in other
sealed readers. That durability question is tracked separately and was
deliberately **not** changed here.
