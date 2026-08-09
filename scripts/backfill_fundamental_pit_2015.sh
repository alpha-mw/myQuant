#!/bin/bash
# Guarded Fundamental PIT rebuild launcher. Offline/dry-run by default.
# It never selects latest generations, never promotes a pointer, and never
# supplies default data paths. A live rebuild needs --execute and the separate
# --allow-live-provider acknowledgement.

set -euo pipefail

REPO=""

MARKET_POINTER=""
MEMBERSHIP=""
SCOPE=""
STAGING_ROOT=""
CHECKPOINT_ROOT=""
BACKUP_DIR=""
COMPLETION_RECEIPT=""
AS_OF=""
RUN_ID=""
YEARS=""
EXECUTE=0
ALLOW_LIVE=0

usage() {
  echo "usage: $0 --repo-root ABS --market-pointer ABS --membership ABS --scope ABS --staging-root ABS --checkpoint-root ABS --backup-dir ABS --completion-receipt ABS --as-of YYYYMMDD --years N --run-id ID [--execute --allow-live-provider]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root) REPO="${2:-}"; shift 2 ;;
    --market-pointer) MARKET_POINTER="${2:-}"; shift 2 ;;
    --membership) MEMBERSHIP="${2:-}"; shift 2 ;;
    --scope) SCOPE="${2:-}"; shift 2 ;;
    --staging-root) STAGING_ROOT="${2:-}"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="${2:-}"; shift 2 ;;
    --backup-dir) BACKUP_DIR="${2:-}"; shift 2 ;;
    --completion-receipt) COMPLETION_RECEIPT="${2:-}"; shift 2 ;;
    --as-of) AS_OF="${2:-}"; shift 2 ;;
    --years) YEARS="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --allow-live-provider) ALLOW_LIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1"; usage; exit 2 ;;
  esac
done

exact_abs() {
  local label="$1"
  local value="$2"
  case "${value}" in
    /*) ;;
    *) echo "${label} must be an explicit absolute path"; exit 2 ;;
  esac
  case "${value}" in
    *'*'*|*'?'*|*'['*|*']'*) echo "${label} must not contain glob syntax"; exit 2 ;;
  esac
  if [ -L "${value}" ]; then
    echo "${label} must not be a symlink: ${value}"
    exit 2
  fi
}

for item in REPO MARKET_POINTER MEMBERSHIP SCOPE STAGING_ROOT CHECKPOINT_ROOT BACKUP_DIR COMPLETION_RECEIPT; do
  value="${!item}"
  if [ -z "${value}" ]; then echo "required path argument is missing: ${item}"; usage; exit 2; fi
  exact_abs "${item}" "${value}"
done
if [ ! -d "${REPO}" ]; then echo "REPO must name an existing directory"; exit 2; fi
CLI="${REPO}/.venv/bin/quant-investor"
PY="${REPO}/.venv/bin/python"
if ! [[ "${AS_OF}" =~ ^[0-9]{8}$ ]]; then echo "--as-of must be YYYYMMDD"; exit 2; fi
if ! [[ "${YEARS}" =~ ^[1-9][0-9]*$ ]]; then echo "--years must be a positive integer"; exit 2; fi
if ! [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then echo "--run-id is invalid"; exit 2; fi
for source in "${MARKET_POINTER}" "${MEMBERSHIP}" "${SCOPE}"; do
  if [ ! -f "${source}" ]; then echo "missing exact input: ${source}"; exit 2; fi
done

echo "mode=$([ "${EXECUTE}" -eq 1 ] && echo execute || echo dry-run) run_id=${RUN_ID} as_of=${AS_OF} years=${YEARS}"
echo "staging_root=${STAGING_ROOT} checkpoint_root=${CHECKPOINT_ROOT}"
if [ "${EXECUTE}" -ne 1 ]; then
  echo "DRY RUN - offline; provider and filesystem mutation disabled"
  exit 0
fi
if [ "${ALLOW_LIVE}" -ne 1 ]; then
  echo "--execute additionally requires --allow-live-provider"
  exit 2
fi
if [ ! -x "${CLI}" ] || [ ! -x "${PY}" ]; then
  echo "repo-local quant-investor and python executables are required"
  exit 2
fi

GENERATION_FILE="${STAGING_ROOT}/_fundamental_generations/${RUN_ID}/fundamental_daily.parquet"
if [ -f "${COMPLETION_RECEIPT}" ]; then
  "${PY}" -c 'import hashlib,json,pathlib,sys; receipt=json.loads(pathlib.Path(sys.argv[1]).read_text()); target=pathlib.Path(sys.argv[2]); expected=sys.argv[3]; actual=hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else ""; sys.exit(0 if receipt.get("run_id")==expected and receipt.get("generation_sha256")==actual else "existing completion receipt does not validate")' "${COMPLETION_RECEIPT}" "${GENERATION_FILE}" "${RUN_ID}"
  echo "ALREADY_COMPLETE - exact completion receipt and generation readback validated"
  exit 0
fi

for target in "${STAGING_ROOT}" "${CHECKPOINT_ROOT}" "${BACKUP_DIR}" "${COMPLETION_RECEIPT}"; do
  if [ -e "${target}" ]; then echo "refusing to overwrite existing target: ${target}"; exit 2; fi
done

POINTER_AS_OF="$("${PY}" -c 'import json,pathlib,sys; print(str(json.loads(pathlib.Path(sys.argv[1]).read_text())["latest_complete_trade_date"]).replace("-", ""))' "${MARKET_POINTER}")"
if [ "${POINTER_AS_OF}" != "${AS_OF}" ]; then
  echo "--as-of differs from exact market pointer: ${AS_OF} != ${POINTER_AS_OF}"
  exit 2
fi
"${PY}" -c 'import hashlib,json,pathlib,sys; payload=json.loads(pathlib.Path(sys.argv[1]).read_text()); expected=str(payload.get("coverage",{}).get("pit_membership_sha256") or "").lower(); actual=hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(); sys.exit(0 if expected and expected==actual else "membership SHA does not bind to exact pointer")' "${MARKET_POINTER}" "${MEMBERSHIP}"

mkdir -m 700 "${BACKUP_DIR}"
cp -p "${MARKET_POINTER}" "${BACKUP_DIR}/market_pointer.json"
cp -p "${MEMBERSHIP}" "${BACKUP_DIR}/membership.parquet"
cp -p "${SCOPE}" "${BACKUP_DIR}/scope.json"
cmp -s "${MARKET_POINTER}" "${BACKUP_DIR}/market_pointer.json"
cmp -s "${MEMBERSHIP}" "${BACKUP_DIR}/membership.parquet"
cmp -s "${SCOPE}" "${BACKUP_DIR}/scope.json"

"${CLI}" market fundamental-maintain \
  --market CN \
  --universes full_a \
  --years "${YEARS}" \
  --as-of "${AS_OF}" \
  --run-id "${RUN_ID}" \
  --allow-live \
  --authoritative-full-rebuild \
  --data-root "${STAGING_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --canonical-scope-path "${SCOPE}" \
  --canonical-market-pointer-path "${MARKET_POINTER}" \
  --canonical-membership-path "${MEMBERSHIP}" \
  --requests-per-second 8.0 \
  --workers 4

if [ ! -f "${GENERATION_FILE}" ]; then
  echo "exact generation file missing after rebuild: ${GENERATION_FILE}"
  exit 1
fi
GENERATION_SHA="$(shasum -a 256 "${GENERATION_FILE}" | awk '{print $1}')"
READBACK_SHA="$(shasum -a 256 "${GENERATION_FILE}" | awk '{print $1}')"
if [ "${GENERATION_SHA}" != "${READBACK_SHA}" ]; then echo "generation readback mismatch"; exit 1; fi

RECEIPT_TMP="${COMPLETION_RECEIPT}.tmp.$$"
"${PY}" -c 'import json,pathlib,sys; path=pathlib.Path(sys.argv[1]); payload={"schema_version":"guarded-fundamental-rebuild-completion.v1","run_id":sys.argv[2],"generation_file":sys.argv[3],"generation_sha256":sys.argv[4],"promoted":False}; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,sort_keys=True,separators=(",", ":"))+"\n")' "${RECEIPT_TMP}" "${RUN_ID}" "${GENERATION_FILE}" "${GENERATION_SHA}"
if ! ln "${RECEIPT_TMP}" "${COMPLETION_RECEIPT}"; then
  echo "completion receipt appeared concurrently"
  exit 1
fi
rm "${RECEIPT_TMP}"
RECEIPT_SHA="$(shasum -a 256 "${COMPLETION_RECEIPT}" | awk '{print $1}')"
RECEIPT_READBACK_SHA="$(shasum -a 256 "${COMPLETION_RECEIPT}" | awk '{print $1}')"
if [ "${RECEIPT_SHA}" != "${RECEIPT_READBACK_SHA}" ]; then echo "receipt readback mismatch"; exit 1; fi

echo "rebuild complete; NOT PROMOTED; completion_receipt=${COMPLETION_RECEIPT}"
