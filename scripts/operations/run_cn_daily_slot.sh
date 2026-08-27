#!/bin/zsh
set -eu
set +x
trap 'unset slot_token TUSHARE_TOKEN TUSHARE_URL 2>/dev/null || true' EXIT

installed_python=""
workspace_root=""
run_root=""
attempt_slot=""
expected_import_root=""

while (( $# > 0 )); do
  case "$1" in
    --python) installed_python="$2"; shift 2 ;;
    --workspace-root) workspace_root="$2"; shift 2 ;;
    --run-root) run_root="$2"; shift 2 ;;
    --attempt-slot) attempt_slot="$2"; shift 2 ;;
    --expected-import-root) expected_import_root="$2"; shift 2 ;;
    *) print -u2 -- "CN_SLOT_LAUNCHER_ARGUMENT_INVALID"; exit 2 ;;
  esac
done

if [[ "$attempt_slot" != "1620" && "$attempt_slot" != "1720" && \
      "$attempt_slot" != "1820" && "$attempt_slot" != "2020" ]]; then
  print -u2 -- "CN_SLOT_LAUNCHER_SLOT_INVALID"
  exit 2
fi
if [[ "$installed_python" != /* || ! -x "$installed_python" || \
      "$workspace_root" != /* || "$run_root" != /* || "$expected_import_root" != /* ]]; then
  print -u2 -- "CN_SLOT_LAUNCHER_PATH_INVALID"
  exit 2
fi

import_origin="$($installed_python -I -c 'import pathlib,quant_investor; print(pathlib.Path(quant_investor.__file__).resolve())')"
if [[ "$import_origin" != "$expected_import_root"/* ]]; then
  print -u2 -- "CN_SLOT_LAUNCHER_IMPORT_ORIGIN_MISMATCH"
  exit 2
fi

receipt_id="slot-${attempt_slot}-$(date -u +%Y%m%dT%H%M%SZ)"
preflight_path="$run_root/credential_preflight/$receipt_id.json"
env_file="$workspace_root/.env"

slot_token="$("$installed_python" -I -c '
import sys
from quant_investor.market.credential_preflight import read_project_env_token
try:
    token = read_project_env_token(sys.argv[1])
except Exception:
    raise SystemExit(3)
sys.stdout.write(token)
' "$env_file" 2>/dev/null || true)"
if [[ -z "$slot_token" ]]; then
  "$installed_python" -I -m quant_investor market credential-preflight \
    --run-root "$run_root" --attempt-slot "$attempt_slot" \
    --receipt-id "$receipt_id" --access-state BLOCKED
  unset slot_token
  print -u2 -- "CN_SLOT_LAUNCHER_ENV_UNAVAILABLE"
  exit 3
fi

"$installed_python" -I -m quant_investor market credential-preflight \
  --run-root "$run_root" --attempt-slot "$attempt_slot" \
  --receipt-id "$receipt_id" --access-state READY
preflight_sha="$(/usr/bin/shasum -a 256 "$preflight_path" | /usr/bin/awk '{print $1}')"

veto_path="$run_root/WRITE_VETO.json"
if [[ -f "$veto_path" ]]; then
  veto_sha="$(/usr/bin/shasum -a 256 "$veto_path" | /usr/bin/awk '{print $1}')"
  env TUSHARE_TOKEN="$slot_token" \
    "$installed_python" -I -m quant_investor market recover-transient-write-veto \
      --workspace-root "$workspace_root" --run-root "$run_root" \
      --expected-veto-sha256 "$veto_sha" \
      --credential-preflight-receipt "$preflight_path" \
      --expected-credential-preflight-sha256 "$preflight_sha"
fi

env TUSHARE_TOKEN="$slot_token" \
  TUSHARE_URL="https://api.tushare.pro/dataapi" \
  PYTHONPATH="" \
  "$installed_python" -I -m quant_investor market daily-maintain \
    --market CN --workspace-root "$workspace_root" --run-root "$run_root" \
    --mode execute --attempt-slot "$attempt_slot"
exit_code=$?
unset slot_token
exit "$exit_code"
