#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMLM_DIR="${REPO_ROOT}/../LMLM"

export PATH="${HOME}/.local/bin:${PATH}"

log() {
  printf '[setup_repository.sh] %s\n' "$*"
}

log "Bootstrapping repository tools"
"${SCRIPT_DIR}/uv.sh"

if [[ ! -d "${LMLM_DIR}" ]]; then
  cat <<EOF >&2
[setup_repository.sh] Missing local LMLM checkout at:
  ${LMLM_DIR}

This repository expects the upstream LMLM repo as a sibling directory because
pyproject.toml uses:
  lmlm = { path = "../LMLM", editable = true }

Clone it first, then rerun this script:
  git clone https://github.com/kilian-group/LMLM.git "${LMLM_DIR}"
EOF
  exit 1
fi

log "Found sibling LMLM checkout at ${LMLM_DIR}"
cd "${REPO_ROOT}"

log "Installing Python 3.10 and syncing dependencies"
uv python install 3.10
uv sync --python 3.10

cat <<'EOF'

Repository setup complete.

Useful next commands:
  source .venv/bin/activate
  .venv/bin/pytest
  ./scripts/claude.sh

EOF

log "Done"