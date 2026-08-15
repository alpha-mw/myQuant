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
an exact presealed target pointer. The installed emergency controller verifies
those bytes but has no pointer write authority. Every read, verify, factor,
research-status, and public-result route performs no activation.

Use `--help` from the intended checkout before any operator run. Arguments are
not inferred from the old command name. Missing required paths, identities,
hashes, cutoffs, or authorization stay blocked; the CLI does not search for a
substitute.

`system bootstrap-assemble` is offline-only and requires a sealed request whose
PIT pointer and complete Fundamental safe-successor v3 fileset are explicit
path/SHA inputs. The fileset includes the canonical Fundamental pointer,
manifest, all three Parquet tables, and every provider-evidence file. The
assembler reconstructs the canonical layout under owner-only staging and
replays predecessor, target-source, derivation, readiness, resource, provider,
table and response-evidence bindings. A homogeneous v1 claim, partial fileset,
or market/PIT/Fundamental cutoff drift fails before generation publication.

## Mapping template

| Removed invocation | Stable invocation | Argument mapping | Output mapping | No-write assertion | Replacement test |
|---|---|---|---|---|---|
| `<old executable and subcommand>` | `<quant-investor group command>` | `<old flag -> stable flag>` | `<old field -> stable field or removed>` | `<protected roots and hashes>` | `<pytest node>` |

An unmapped flag or output field is a migration blocker, not permission to pass
unknown arguments through.
