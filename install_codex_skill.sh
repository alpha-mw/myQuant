#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/skill"
CODEX_HOME_DIR="${CODEX_HOME:-${HOME}/.codex}"
DEST_DIR="${CODEX_HOME_DIR}/skills/quant-investor"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "未找到 skill 源目录: ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${CODEX_HOME_DIR}/skills"

if [[ -e "${DEST_DIR}" ]]; then
  BACKUP_DIR="${DEST_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
  mv "${DEST_DIR}" "${BACKUP_DIR}"
  echo "已备份现有 skill 到: ${BACKUP_DIR}"
fi

cp -R "${SOURCE_DIR}" "${DEST_DIR}"

echo "已安装到: ${DEST_DIR}"
echo "重启 Codex 后即可识别新 skill。"
