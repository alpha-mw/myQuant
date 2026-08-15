# Codex External Deployment Copies

This directory is the repository-controlled source for the myQuant Codex skill
and the `myquant-2` automation after the unified runtime cutover. The files are
inert copies: repository verification does not write `~/.codex`, update a
schedule, or activate an automation.

## Exact targets

| Repository source | External installed target |
|---|---|
| `operations/codex/skills/myquant/` | `/Users/maxwell/.codex/skills/myquant/` |
| `operations/codex/automations/myquant-2/automation.toml` | `/Users/maxwell/.codex/automations/myquant-2/automation.toml` |

The exact file-by-file mapping and content hashes are in
`projection-manifest.json`. No glob or directory scan may add a file to the
skill projection: the manifest allowlist is complete and extra files block
deployment.

## Canonical hashes

All hashes use SHA-256.

- A file `byte_sha256` hashes its exact bytes.
- `skill_tree_identity_sha256` hashes canonical UTF-8 JSON for the ordered array
  of `{relative_path, mode, byte_sha256}` skill rows. Mode is the exact
  four-digit regular-file mode, currently `0644`. JSON is serialized with keys
  sorted, no optional whitespace, and Unicode preserved.
- `semantic_projection_sha256` hashes canonical UTF-8 JSON for the automation
  object containing exactly the manifest's 11 `included_fields`. The excluded
  fields are exactly `status`, `created_at`, and `updated_at`.
- `contract_sha256` hashes canonical UTF-8 JSON for the entire manifest after
  removing only the root `contract_sha256` field. The manifest file itself is
  that exact compact, sorted-key canonical JSON byte sequence, with no trailing
  newline.

The automation source itself must contain `status = "PAUSED"`. Status is
excluded from semantic identity because it is operational state, but the
repository copy is not eligible for review if it is anything other than
`PAUSED`. `activation_performed=false` and
`external_deployment_performed=false` are assertions about this repository
cutover, not instructions to change external state.

## Read-only verification

Verification must:

1. require canonical bytes for `projection-manifest.json` and parse it as JSON;
2. require the sealed `agents/openai.yaml` bytes and parse `automation.toml` as
   TOML;
3. require the exact source and installed-target paths in the manifest;
4. reject symlinks, non-regular files, mode drift, missing files, extra skill
   files, hash mismatches, field-set drift, any automation status other than
   `PAUSED`, and any removed command in the deployment sources;
5. recompute the tree, semantic projection, and contract hashes;
6. read back the same canonical manifest bytes and report their byte hash;
7. stop before copying or activation.

Run the repository verifier from the repository root:

```bash
uv run python operations/codex/verify_projection.py
```

External installation requires a separate explicit request, a fresh readback
of installed pre-state, compare-before-replace protection, post-copy byte
comparison, and automation readback that remains `PAUSED`. Those operations are
outside this repository-only cutover.
