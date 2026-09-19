#!/bin/zsh
set -eu
set +x
umask 077
set -C
trap 'unset slot_token TUSHARE_TOKEN 2>/dev/null || true' EXIT

installed_python=""
workspace_root=""
run_root=""
attempt_slot=""
expected_import_root=""
scope_transition_request=""
scope_transition_sha=""
retire_coverage_sha=""
factor_loop_context=""
factor_loop_context_sha=""
daily_request=""
daily_request_sha=""
daily_source_config=""
daily_source_config_sha=""
release_repository_root=""
release_install_input=""
release_install_sha=""
daily_profile=0

while (( $# > 0 )); do
  if (( $# < 2 )); then
    print -u2 -- "CN_SLOT_LAUNCHER_ARGUMENT_INVALID"
    exit 2
  fi
  case "$1" in
    --retire-coverage-declaration-sha256) retire_coverage_sha="$2"; shift 2 ;;
    --scope-transition-request) scope_transition_request="$2"; shift 2 ;;
    --expected-scope-transition-sha256) scope_transition_sha="$2"; shift 2 ;;
    --python) installed_python="$2"; shift 2 ;;
    --workspace-root) workspace_root="$2"; shift 2 ;;
    --run-root) run_root="$2"; shift 2 ;;
    --attempt-slot) attempt_slot="$2"; shift 2 ;;
    --expected-import-root) expected_import_root="$2"; shift 2 ;;
    --factor-loop-context) factor_loop_context="$2"; shift 2 ;;
    --expected-factor-loop-context-sha256) factor_loop_context_sha="$2"; shift 2 ;;
    --daily-production-request) daily_request="$2"; shift 2 ;;
    --expected-daily-production-request-sha256) daily_request_sha="$2"; shift 2 ;;
    --daily-source-config) daily_source_config="$2"; shift 2 ;;
    --expected-daily-source-config-sha256) daily_source_config_sha="$2"; shift 2 ;;
    --release-repository-root) release_repository_root="$2"; shift 2 ;;
    --release-install-input) release_install_input="$2"; shift 2 ;;
    --expected-release-install-input-sha256) release_install_sha="$2"; shift 2 ;;
    *) print -u2 -- "CN_SLOT_LAUNCHER_ARGUMENT_INVALID"; exit 2 ;;
  esac
done

is_daily_ref() {
  [[ -n "$1" && "$1" =~ '^[ -~]+$' && "$1" != /* && "$1" != *\\* &&
     "$1" != */ && "$1" != *//* && "$1" != . && "$1" != .. &&
     "$1" != ./* && "$1" != ../* && "$1" != */./* && "$1" != */../* &&
     "$1" != */. && "$1" != */.. ]]
}

if [[ -n "$daily_request" || -n "$daily_request_sha" || -n "$daily_source_config" ||
      -n "$daily_source_config_sha" || -n "$release_repository_root" ||
      -n "$release_install_input" || -n "$release_install_sha" ]]; then
  daily_profile=1
  if [[ "$attempt_slot" != "2020" || "$release_repository_root" != /* ||
        "$run_root" != "$workspace_root/data/private/cn_daily_maintenance" ||
        ${#release_install_sha} != 64 || "$release_install_sha" == *[^0-9a-f]* ||
        -n "$factor_loop_context" || -n "$factor_loop_context_sha" ||
        -n "$scope_transition_request" || -n "$scope_transition_sha" || -n "$retire_coverage_sha" ]] ||
      ! is_daily_ref "$release_install_input"; then
    print -u2 -- "CN_DAILY_PRODUCTION_ARGUMENTS_INVALID"
    exit 2
  fi
  if [[ -n "$daily_source_config" || -n "$daily_source_config_sha" ]]; then
    if [[ -n "$daily_request" || -n "$daily_request_sha" ||
          ${#daily_source_config_sha} != 64 || "$daily_source_config_sha" == *[^0-9a-f]* ]] ||
        ! is_daily_ref "$daily_source_config"; then
      print -u2 -- "CN_DAILY_SOURCE_ARGUMENTS_INVALID"
      exit 2
    fi
  elif [[ ${#daily_request_sha} != 64 || "$daily_request_sha" == *[^0-9a-f]* ]] ||
      ! is_daily_ref "$daily_request"; then
    print -u2 -- "CN_DAILY_PRODUCTION_ARGUMENTS_INVALID"
    exit 2
  fi
fi

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

if [[ ( -n "$scope_transition_request" || -n "$scope_transition_sha" || -n "$retire_coverage_sha" ) &&
      ( -n "$factor_loop_context" || -n "$factor_loop_context_sha" ) ]]; then
  print -u2 -- "DAILY_MAINTENANCE_MODES_CONFLICT"
  exit 2
fi
if [[ -n "$factor_loop_context" || -n "$factor_loop_context_sha" ]]; then
  if [[ "$factor_loop_context" != /* || ${#factor_loop_context_sha} != 64 ||
        "$factor_loop_context_sha" == *[^0-9a-f]* || "$attempt_slot" != "2020" ]]; then
    print -u2 -- "DAILY_FACTOR_CONTEXT_ARGUMENTS_INVALID"
    exit 2
  fi
fi

if [[ -n "$scope_transition_request" || -n "$scope_transition_sha" ]]; then
  if [[ "$scope_transition_request" != /* || ${#scope_transition_sha} != 64 || "$scope_transition_sha" == *[^0-9a-f]* || "$attempt_slot" != "2020" ]]; then
    print -u2 -- "SCOPE_TRANSITION_ARGUMENTS_REQUIRED_TOGETHER"
    exit 2
  fi
fi

if [[ -n "$retire_coverage_sha" && ( -z "$scope_transition_request" || ${#retire_coverage_sha} != 64 || "$retire_coverage_sha" == *[^0-9a-f]* ) ]]; then
  print -u2 -- "SCOPE_TRANSITION_REQUEST_REQUIRED_FOR_DECLARATION_RETIREMENT"
  exit 2
fi

if (( daily_profile )); then
  # The inspected DAG branch selects credentials explicitly; inherit none.
  unset TUSHARE_TOKEN
fi

import_origin="$($installed_python -I -c 'import pathlib,quant_investor; print(pathlib.Path(quant_investor.__file__).resolve())')"
if [[ "$import_origin" != "$expected_import_root"/* ]]; then
  print -u2 -- "CN_SLOT_LAUNCHER_IMPORT_ORIGIN_MISMATCH"
  exit 2
fi

receipt_id="slot-${attempt_slot}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
launcher_root="$run_root/launcher_attempts/$receipt_id"
"$installed_python" -I -c '
import json,sys
from quant_investor.market.daily_maintenance import write_launcher_record
print(json.dumps(write_launcher_record(run_root=sys.argv[1],receipt_id=sys.argv[2],phase="STARTED")))
' "$run_root" "$receipt_id"
finish_launcher() {
  local launcher_code=$?
  unset slot_token TUSHARE_TOKEN 2>/dev/null || true
  if "$installed_python" -I -c '
import json,sys
from quant_investor.market.daily_maintenance import write_launcher_record
print(json.dumps(write_launcher_record(run_root=sys.argv[1],receipt_id=sys.argv[2],phase="ENDED",exit_code=int(sys.argv[3]))))
' "$run_root" "$receipt_id" "$launcher_code"; then
    :
  elif (( daily_profile )); then
    print -u2 -- "CN_DAILY_LAUNCHER_END_RECEIPT_UNAVAILABLE"
    trap - EXIT
    exit 3
  fi
}
trap finish_launcher EXIT

prepare_project_credentials() {
  if [[ -n "${slot_token:-}" ]]; then
    return 0
  fi
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
}

if [[ -n "$daily_source_config" ]]; then
  source_args=(--workspace-root "$workspace_root" --config "$daily_source_config"
    --expected-config-sha256 "$daily_source_config_sha"
    --release-repository-root "$release_repository_root"
    --release-install-input "$release_install_input"
    --expected-release-install-input-sha256 "$release_install_sha")
  source_plain() {
    env -u TUSHARE_TOKEN "$installed_python" -I -m quant_investor.cli.daily_sources "${source_args[@]}" "$@"
  }
  source_fail() {
    local source_exit="$1"
    cat "$launcher_root/$2.stdout.json"
    cat "$launcher_root/$2.stderr.log" >&2
    [[ "$source_exit" == "2" ]] && exit 2
    exit 3
  }
  source_saved="source-inspection"
  if source_plain --mode inspect > "$launcher_root/source-inspection.stdout.json" 2> "$launcher_root/source-inspection.stderr.log"; then
    source_code=0
  else
    source_code=$?
  fi
  if [[ "$source_code" == "10" || "$source_code" == "11" ]]; then
    source_saved="source-provision"
    if [[ "$source_code" == "11" ]]; then
      prepare_project_credentials
      if TUSHARE_TOKEN="$slot_token" PYTHONPATH="" "$installed_python" -I -m quant_investor.cli.daily_sources \
          "${source_args[@]}" --mode provision > "$launcher_root/source-provision.stdout.json" 2> "$launcher_root/source-provision.stderr.log"; then
        source_code=0
      else
        source_code=$?
      fi
    else
      if source_plain --mode provision --no-providers > "$launcher_root/source-provision.stdout.json" 2> "$launcher_root/source-provision.stderr.log"; then
        source_code=0
      else
        source_code=$?
      fi
    fi
  fi
  [[ "$source_code" == "0" ]] || source_fail "$source_code" "$source_saved"
  source_relative="data/private/cn_daily_maintenance/launcher_attempts/$receipt_id/$source_saved.stdout.json"
  source_sha="$(/usr/bin/shasum -a 256 "$launcher_root/$source_saved.stdout.json" | /usr/bin/awk '{print $1}')"
  source_saved_args=(--inspection "$source_relative" --expected-inspection-sha256 "$source_sha")
  if source_plain --mode emit "${source_saved_args[@]}" > "$launcher_root/source-validated.stdout.json" 2> "$launcher_root/source-validated.stderr.log"; then
    :
  else
    source_fail "$?" "source-validated"
  fi
  if source_plain --mode select "${source_saved_args[@]}" > "$launcher_root/source-selection.txt" 2> "$launcher_root/source-selection.stderr.log"; then
    source_selection_code=0
  else
    source_selection_code=$?
  fi
  if [[ "$source_selection_code" == "10" ]]; then
    cat "$launcher_root/source-validated.stdout.json"
    exit 0
  fi
  if [[ "$source_selection_code" != "0" ]]; then
    cat "$launcher_root/source-selection.txt"
    cat "$launcher_root/source-selection.stderr.log" >&2
    [[ "$source_selection_code" == "2" ]] && exit 2
    exit 3
  fi
  source_selection="$(cat "$launcher_root/source-selection.txt")"
  source_fields=("${(@f)source_selection}")
  if [[ ${#source_fields} != 2 ]] || ! is_daily_ref "${source_fields[1]}" ||
      [[ ${#source_fields[2]} != 64 || "${source_fields[2]}" == *[^0-9a-f]* ]]; then
    print -u2 -- "CN_DAILY_SOURCE_SELECTION_INVALID"
    exit 2
  fi
  daily_request="${source_fields[1]}"
  daily_request_sha="${source_fields[2]}"
fi

if (( daily_profile )); then
  daily_args=(--workspace-root "$workspace_root" --request "$daily_request"
    --expected-request-sha256 "$daily_request_sha"
    --release-repository-root "$release_repository_root"
    --release-install-input "$release_install_input"
    --expected-release-install-input-sha256 "$release_install_sha")
  inspect_relative="data/private/cn_daily_maintenance/launcher_attempts/$receipt_id/inspection.stdout.json"
  launch_inspection() {
    env -u TUSHARE_TOKEN "$installed_python" -I -c '
import sys
from quant_investor.cli.daily_launch import main
sys.exit(main())
' "${daily_args[@]}" "$@"
  }
  if launch_inspection --mode inspect > "$launcher_root/inspection.stdout.json" 2> "$launcher_root/inspection.stderr.log"; then
    inspection_code=0
  else
    inspection_code=$?
  fi
  if [[ "$inspection_code" != "0" && "$inspection_code" != "10" && "$inspection_code" != "11" ]]; then
    cat "$launcher_root/inspection.stdout.json"
    cat "$launcher_root/inspection.stderr.log" >&2
    [[ "$inspection_code" == "2" ]] && exit 2
    exit 3
  fi
  inspection_sha="$(/usr/bin/shasum -a 256 "$launcher_root/inspection.stdout.json" | /usr/bin/awk '{print $1}')"
  inspection_args=(--inspection "$inspect_relative" --expected-inspection-sha256 "$inspection_sha")
  if launch_inspection --mode validate "${inspection_args[@]}" > "$launcher_root/inspection-validation.stdout.json" 2> "$launcher_root/inspection-validation.stderr.log"; then
    validated_code=0
  else
    validated_code=$?
  fi
  if [[ "$validated_code" != "$inspection_code" ]]; then
    cat "$launcher_root/inspection-validation.stdout.json"
    cat "$launcher_root/inspection-validation.stderr.log" >&2
    print -u2 -- "CN_DAILY_INSPECTION_DECISION_CHANGED"
    [[ "$validated_code" == "3" ]] && exit 3
    exit 2
  fi
  if [[ "$validated_code" == "0" ]]; then
    if launch_inspection --mode emit "${inspection_args[@]}" > "$launcher_root/daily-close.stdout.json" 2> "$launcher_root/daily-close.stderr.log"; then
      daily_code=0
    else
      daily_code=$?
    fi
  elif [[ "$validated_code" == "10" ]]; then
    if launch_inspection --mode recover "${inspection_args[@]}" > "$launcher_root/daily-close.stdout.json" 2> "$launcher_root/daily-close.stderr.log"; then
      daily_code=0
    else
      daily_code=$?
    fi
  else
    prepare_project_credentials
    if TUSHARE_TOKEN="$slot_token" PYTHONPATH="" "$installed_python" -I -m quant_investor production daily-close \
        "${daily_args[@]}" > "$launcher_root/daily-close.stdout.json" 2> "$launcher_root/daily-close.stderr.log"; then
      daily_code=0
    else
      daily_code=$?
    fi
    unset slot_token
  fi
  cat "$launcher_root/daily-close.stdout.json"
  cat "$launcher_root/daily-close.stderr.log" >&2
  [[ "$daily_code" == "0" || "$daily_code" == "2" || "$daily_code" == "3" ]] || daily_code=3
  exit "$daily_code"
fi

factor_args=()
if [[ "$attempt_slot" == "2020" && -n "$factor_loop_context" ]]; then
  factor_args=(--factor-loop-context "$factor_loop_context"
    --expected-factor-loop-context-sha256 "$factor_loop_context_sha")
  # Historical settlement runs before token access, independently of new signals.
  if "$installed_python" -I -m quant_investor market daily-maintain \
      --market CN --workspace-root "$workspace_root" --run-root "$run_root" \
      --mode execute --attempt-slot "$attempt_slot" "${factor_args[@]}" --recover-only > "$launcher_root/recovery.stdout.json" 2> "$launcher_root/recovery.stderr.log"; then
    :
  else
    print -u2 -- "CN_SLOT_HISTORICAL_RECOVERY_BLOCKED"
  fi
  cat "$launcher_root/recovery.stdout.json"
  cat "$launcher_root/recovery.stderr.log" >&2
fi

prepare_project_credentials

veto_path="$run_root/WRITE_VETO.json"
if [[ -f "$veto_path" && -z "$scope_transition_request" ]]; then
  veto_sha="$(/usr/bin/shasum -a 256 "$veto_path" | /usr/bin/awk '{print $1}')"
  if env TUSHARE_TOKEN="$slot_token" \
    "$installed_python" -I -m quant_investor market recover-transient-write-veto \
      --workspace-root "$workspace_root" --run-root "$run_root" \
      --expected-veto-sha256 "$veto_sha" \
      --credential-preflight-receipt "$preflight_path" \
      --expected-credential-preflight-sha256 "$preflight_sha" > "$launcher_root/veto-recovery.stdout.json" 2> "$launcher_root/veto-recovery.stderr.log"; then
    cat "$launcher_root/veto-recovery.stdout.json"
    cat "$launcher_root/veto-recovery.stderr.log" >&2
  else
    veto_code=$?
    cat "$launcher_root/veto-recovery.stdout.json"
    cat "$launcher_root/veto-recovery.stderr.log" >&2
    exit "$veto_code"
  fi
fi

transition_args=()
if [[ -n "$scope_transition_request" ]]; then
  transition_args=(--scope-transition-request "$scope_transition_request"
    --expected-scope-transition-sha256 "$scope_transition_sha")
fi

if [[ -n "$retire_coverage_sha" ]]; then
  transition_args+=(--retire-coverage-declaration-sha256 "$retire_coverage_sha")
fi

if env TUSHARE_TOKEN="$slot_token" \
  PYTHONPATH="" \
  "$installed_python" -I -m quant_investor market daily-maintain \
    --market CN --workspace-root "$workspace_root" --run-root "$run_root" \
    --mode execute --attempt-slot "$attempt_slot" "${factor_args[@]}" "${transition_args[@]}" > "$launcher_root/maintenance.stdout.json" 2> "$launcher_root/maintenance.stderr.log"; then
  exit_code=0
else
  exit_code=$?
fi
cat "$launcher_root/maintenance.stdout.json"
cat "$launcher_root/maintenance.stderr.log" >&2
unset slot_token
exit "$exit_code"
