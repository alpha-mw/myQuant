# Equivalent canonical control-character scan optimization

Status: Architect APPROVE and Critic APPROVE; minimal change implemented, native validation pending.

## Evidence and required outcome

Frozen d2 final CI:4995 passed/1 failed/3 skipped; the full prospective contextual
validation exceeded the unchanged180-second worker bound. Isolated exact-test
reproduction also failed after747.70 seconds total setup+validation. Its retained
workspace is /private/tmp/myquant-contextual-timeout-repro-20260916T144304Z.
A second diagnostic reused those same stored inputs directly through the unchanged
native callback worker, with a profiler; it generated no validation receipt.
The150-second profile snapshot has1,173,588,019 calls. Control-character generator
expressions at contracts/core.py:82 and:124 account for511million generator calls;
_validate_string's cumulative recorded time is69.99seconds, total canonical-value
validation107.56seconds. _reject_casefold_alias is1.36seconds cumulative. Profiling
adds overhead, so these figures identify a candidate bottleneck, not unprofiled
speedup or complete causal proof. Retained source comparison confirms relevant
validation/contextual/store/test files match prior c3 and new a46 versions.

## Minimal implementation

Only quant_investor/contracts/core.py changes:
- Add a compiled constant regex with the exact Unicode codepoint range U+0000
  through U+001F: `re.compile(r"[\x00-\x1f]")`.
- Replace the two `any(ord(character) < 0x20 for character in ...)` conditions
  in `_validate_string` and dictionary-key validation with an explicit
  `pattern.search(value) is not None` predicate.
- Preserve check order, exact exception class/message, strict UTF-8 encoding,
  byte bounds, NFC validation, key ASCII/nonempty checks, recursion/cycle checks,
  JSON serialization, schema/catalog hashes and all source/custody replay.
- No cache, memoization, directory-scan optimization, early acceptance, alternate
  serializer, source omission, timeout change or callback-bound change.
- Leave governance/common.py and timestamp validation unchanged; do not broaden
  optimization to other locations without evidence.

A single bounded character class is linear and matches exactly the previous
predicate for every Python Unicode codepoint. Surrogates remain rejected by the
pre-existing strict UTF-8 encoding step before this predicate for string values;
non-ASCII keys remain rejected by the pre-existing key guard.

## Verification

1. Unprofiled microbenchmark old predicate against candidate regex using strings
   from the retained synthetic test artifacts; record corpus and results, not an
   acceptance claim. Establish useful speedup before runtime edits.
2. Differential predicate check across all Python Unicode codepoints, plus all
   control positions, empty strings, ASCII keys, non-ASCII/NFC/non-NFC/surrogate
   strings, byte-bound cases and nested/cyclic objects. Assert unchanged accepted
   bytes and unchanged exception type/message against a test-only old reference.
3. Existing canonical-contract/golden-hash tests and stable formatting/type checks.
4. Reuse the retained exact native request/candidate under the original180-second
   worker bound without profiling. This diagnostic is not a new full test pass;
   then run the actual original test through its intended path on changed source.
5. Full required CI and installed/current-source final acceptance remain needed;
   preserve failed old proofs and current frozen running jobs. No hotpatch.

Stop if accepted inputs, output bytes, error semantics, compiled contracts, limits
or resource defenses change; if meaningful speedup is absent; or if a broader
trust/cache/serialization redesign is needed. Re-review before broadening scope.

## Verification checkpoint

- Exhaustive1,114,112-codepoint predicate comparison:0differences.
- Unprofiled retained corpus6855strings/193686characters:median5.67x predicate
  improvement (not full callback speedup); exact samples retained in acceptance.
- Canonical differential plus existing contracts:140tests PASS; Black and stable
  mypy88files PASS. Catalog SHA unchanged:
  bf33e0e64f9c316edf78ba6289535e2d7c39311a4c1feac70ce2da43afc981fb.
- Reusing old retained callback after the code change correctly refused with
  IMPLEMENTATION_IDENTITY_MISMATCH after47.581seconds. All7244retained files
  unchanged. This is not a timeout result or permission to rewrite old identities.
- Therefore full runtime verification uses the original test to generate a fresh
  native fixture under the new frozen installed candidate5060dee2. No timestamp,
  source SHA, component identity or native180-second bound is manually adjusted.
