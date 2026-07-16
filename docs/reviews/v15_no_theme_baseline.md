# v15 No-Theme Baseline

- Captured at: 2026-07-16 CST
- Source branch: `main`
- Source commit: `48a899c338b8ebdd82a2e0ca62493443a37c2d6a`
- Source worktree status: clean
- Integration worktree: `/tmp/myquant-v15-integration`
- Integration branch: `codex/v15-no-theme-integration`

## Protected state

| State | SHA-256 |
| --- | --- |
| Factor registry | `b8369dfef7d27156999e93e3a1a12020e072db0296532fee10b0335d8bddca2f` |
| Strict catalog | `8969f9683cd68ac3f2ef481cec297e5bf50d340286aff6275d2b5cc76c90c086` |
| CN market pointer | `b7d9387ded8109ad742a9052bcc9ba77d716a5ecbf102eb965b4449706ccd349` |
| Fundamental pointer | `eeb5d2f584f5351f024f520125894843c53ff7dc9cfcdaa165c1002345e37bbd` |
| Strategy-record file/hash aggregate | `e25170a394d539eeef553aa8ccbaf06f0cddb0fb14541374cb2159d17ad08595` |

The ignored strategy-record tree, its contained ledgers and manifests, Git
history, provider pointers, and Factor registry are protected throughout the
code migration.  The integration worktree does not contain the ignored private
data roots.  Any purge, live provider operation, production activation, or
registry mutation requires a later independent authorization.
