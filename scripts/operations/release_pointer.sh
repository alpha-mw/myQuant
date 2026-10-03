#!/bin/zsh
# Read and validate operations/releases/active.env from zsh.
#
# Source this file, then call `read_release_pointer <pointer-file> <workspace-root>`.
# On success it defines RELEASE_COMMIT, RELEASE_INSTALL_DIR, RELEASE_CHECKOUT_DIR,
# RELEASE_INSTALL_INPUT_SHA256, FACTOR_LOOP_CONTEXT (absolute), FACTOR_LOOP_CONTEXT_SHA256,
# PRUNE_RELEASE_INSTALL_DIR (may be empty), INSTALLED_PYTHON and EXPECTED_IMPORT_ROOT.
# On any problem it prints one blocker code on stderr and returns 2. The file is
# parsed line by line as KEY=VALUE; nothing in it is ever evaluated.

read_release_pointer() {
  local file="$1" workspace="$2" line key value
  local -a known=(RELEASE_COMMIT RELEASE_INSTALL_DIR RELEASE_CHECKOUT_DIR
    RELEASE_INSTALL_INPUT_SHA256 FACTOR_LOOP_CONTEXT FACTOR_LOOP_CONTEXT_SHA256
    PRUNE_RELEASE_INSTALL_DIR)
  for key in "${known[@]}"; do
    typeset -g "$key"=""
  done
  if [[ -z "$file" || ! -f "$file" || -z "$workspace" || "$workspace" != /* ]]; then
    print -u2 -- "RELEASE_POINTER_MISSING"
    return 2
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ "$line" != *=* || ! "$key" =~ '^[A-Z][A-Z0-9_]*$' ]]; then
      print -u2 -- "RELEASE_POINTER_LINE_INVALID"
      return 2
    fi
    if (( ${known[(Ie)$key]} == 0 )); then
      print -u2 -- "RELEASE_POINTER_KEY_UNKNOWN:$key"
      return 2
    fi
    typeset -g "$key"="$value"
  done < "$file"

  if [[ ! "$RELEASE_COMMIT" =~ '^[0-9a-f]{40}$' ]]; then
    print -u2 -- "RELEASE_POINTER_COMMIT_INVALID"
    return 2
  fi
  if [[ "$RELEASE_INSTALL_DIR" != /* || "${RELEASE_INSTALL_DIR:t}" != "$RELEASE_COMMIT"-* ||
        ! -x "$RELEASE_INSTALL_DIR/bin/python" ]]; then
    print -u2 -- "RELEASE_POINTER_INSTALL_INVALID"
    return 2
  fi
  local -a site_packages=("$RELEASE_INSTALL_DIR"/lib/python3.*/site-packages(N/))
  if (( ${#site_packages} != 1 )); then
    print -u2 -- "RELEASE_POINTER_IMPORT_ROOT_AMBIGUOUS"
    return 2
  fi
  if [[ "$RELEASE_CHECKOUT_DIR" != /* || "${RELEASE_CHECKOUT_DIR:t}" != "$RELEASE_COMMIT"-* ||
        ! -d "$RELEASE_CHECKOUT_DIR" ]]; then
    print -u2 -- "RELEASE_POINTER_CHECKOUT_INVALID"
    return 2
  fi
  if [[ ! "$RELEASE_INSTALL_INPUT_SHA256" =~ '^[0-9a-f]{64}$' ||
        ! "$FACTOR_LOOP_CONTEXT_SHA256" =~ '^[0-9a-f]{64}$' ]]; then
    print -u2 -- "RELEASE_POINTER_SHA_INVALID"
    return 2
  fi
  if [[ "$FACTOR_LOOP_CONTEXT" != /* ]]; then
    FACTOR_LOOP_CONTEXT="$workspace/$FACTOR_LOOP_CONTEXT"
  fi
  if [[ ! -f "$FACTOR_LOOP_CONTEXT" ||
        "$(/usr/bin/shasum -a 256 "$FACTOR_LOOP_CONTEXT" | /usr/bin/awk '{print $1}')" != "$FACTOR_LOOP_CONTEXT_SHA256" ]]; then
    print -u2 -- "RELEASE_POINTER_CONTEXT_SHA_MISMATCH"
    return 2
  fi
  if [[ -n "$PRUNE_RELEASE_INSTALL_DIR" &&
        ( "$PRUNE_RELEASE_INSTALL_DIR" != /* || ! -x "$PRUNE_RELEASE_INSTALL_DIR/bin/python" ) ]]; then
    print -u2 -- "RELEASE_POINTER_PRUNE_INSTALL_INVALID"
    return 2
  fi
  typeset -g INSTALLED_PYTHON="$RELEASE_INSTALL_DIR/bin/python"
  typeset -g EXPECTED_IMPORT_ROOT="${site_packages[1]}"
  return 0
}
