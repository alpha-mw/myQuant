# Stable CLI Mapping

The hard cutover removes the version-named executable. Rewrite invocations to
the single `quant-investor` command; do not retain shell aliases or wrapper
fallbacks.

| Removed invocation | Stable invocation | Stable meaning |
|---|---|---|
| `quant-investor-v17-v4 verify` | `quant-investor system verify` | Verify registered contracts and stable package/system integrity without activation |
| `quant-investor-v17-v4 status` | `quant-investor system status` | Read the exact system generation and its blockers |
| `quant-investor-v17-v4 factor-set-status` | `quant-investor factor status` | Read factor-set health and evidence; never admit or activate factor state |
| `quant-investor-v17-v4 run-forward` | `quant-investor research forward` | Run the governed forward-research observation path |
| `quant-investor-v17-v4 research-evaluate` | `quant-investor research evaluate` | Evaluate one exact research request |
| `quant-investor-v17-v4 deep-v3-compile` | `quant-investor research compile-evidence` | Compile the stable research-evidence closure |
| `quant-investor-v17-v4 forward-shadow-readiness` | `quant-investor research readiness` | Check exact forward-research prerequisites |
| `quant-investor-v17-v4 forward-shadow-status` | `quant-investor research inspect` | Inspect one exact research run or session |

Existing stable market and public-read commands remain under the same
executable:

```bash
quant-investor system verify --help
quant-investor system status --help
quant-investor system bootstrap-assemble --help
quant-investor system activate --help
quant-investor market maintain --help
quant-investor market storage-validate --help
quant-investor research run --help
quant-investor market analyze --help
quant-investor market run --help
quant-investor portfolio cycle-status --help
```

Initial `system activate` is the only first-pointer writer. It consumes exact
bytes and expected SHAs for the detached migration receipt, final cutover
authorization, activation authorization, target pointer, and deployed release.
`system suspend` is a separate non-empty-to-suspended lane and likewise consumes
an exact presealed target pointer. Its only accepted target is the suspended
generation and manifest sealed into the initial controller. The System resolves
that fixed target from raw pointer history without requiring or modifying the
permanent marker, while the installed emergency controller itself has no pointer
write authority. Every read, verify, factor, research-status, and public-result
route performs no activation.

Use `--help` from the intended checkout before any operator run. Arguments are
not inferred from the old command name. Missing required paths, identities,
hashes, cutoffs, or authorization stay blocked; the CLI does not search for a
substitute.

`system calendar-capture` is the installed-release-only, read-only network
entrypoint for the `TRUSTED_PROVIDER_DEGRADED` route. It atomically retains the
official Tushare `trade_cal` documentation plus exact SSE, SZSE, and BSE probe
responses and seals an all-leaves capture transaction, installed-release
execution receipt, and success marker. The command requires the exact
release-install input path and SHA plus a distinct exact clean detached
`--release-repository-root`; the attached production workspace is never
substituted for that release checkout. Caller-supplied timestamps or transport
objects are not accepted. The production request must bind that transaction,
execution receipt, success marker, release-install input, and the original
owner-only capture-root inode. Assembly reopens the fixed thirteen-leaf root
through one pinned no-follow descriptor before copying; isolated or rehomed
raw/capture/receipt files are insufficient.
Tushare does not sign response bodies: this proves exact retained bytes,
reviewed installed operator identity, and local custody topology, but it is not
a cryptographic proof against a malicious same-UID process.
The BSE response must be exact-empty and confers no direct calendar authority.
No provider write, broker, order, trade, funds, portfolio, Strategy
Record, System pointer, generation, or activation write is reachable from this
command.

`system bootstrap-assemble` is offline-only and requires a sealed request whose
PIT pointer and complete Fundamental safe-successor v3 fileset are explicit
path/SHA inputs. The fileset includes the canonical Fundamental pointer,
manifest, all three Parquet tables, and every provider-evidence file. The
calendar side is one exact tagged route. `EXCHANGE_OFFICIAL` uses a detached
`system.exchange_calendar_compilation`, raw issuer responses, decoder
admissions, and complete notice-index/page/body closures.
`TRUSTED_PROVIDER_DEGRADED` uses a detached
`system.trusted_provider_calendar_compilation`, provider capability artifact,
exact three-response capture set, the immutable four-call capture transaction,
installed-release execution receipt, terminal success marker, policy artifact,
runtime JSON and strict Parquet. Official and provider fields
are mandatory tombstones for the route
that is not selected; mixed, missing, duplicated, or old topology fails. The
assembler reruns the selected decoder and requires regenerated JSON and
Parquet bytes to equal the requested files. The superseded
`calendar_manifest_file_ref` and project-authored `DAILY_STATUS` projection are
not accepted request fields or calendar authority. The assembler reconstructs
the canonical layout under owner-only staging and
replays predecessor, target-source, derivation, readiness, resource, provider,
table and response-evidence bindings. A homogeneous v1 claim, partial fileset,
or market/PIT/Fundamental cutoff drift fails before generation publication.
The staging directory is always the workspace-relative
`data/private/system_source_staging/<operation-id>` authority beneath the
default `SystemStore`; the request's `source_root_id` is only an assertion of
that System-derived identity and cannot select another root. The detached
production receipt binds the complete acyclic generation intent, including
timestamp, assembly identity, catalog, mandatory-null migration tombstones and
`mainline_ref = null`. Source blockers are the exact repository-owned
projection of replayed Fundamental machine states, never operator labels.

## Mapping template

| Removed invocation | Stable invocation | Argument mapping | Output mapping | No-write assertion | Replacement test |
|---|---|---|---|---|---|
| `<old executable and subcommand>` | `<quant-investor group command>` | `<old flag -> stable flag>` | `<old field -> stable field or removed>` | `<protected roots and hashes>` | `<pytest node>` |

An unmapped flag or output field is a migration blocker, not permission to pass
unknown arguments through.
