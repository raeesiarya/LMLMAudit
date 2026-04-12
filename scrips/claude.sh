#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PATH="${HOME}/.local/bin:${PATH}"

log() {
  printf '[claude.sh] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[claude.sh] Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

install_claude() {
  require_command curl
  log "Installing Claude Code via the official installer"
  curl -fsSL https://claude.ai/install.sh | bash
  export PATH="${HOME}/.local/bin:${PATH}"
}

if ! command -v claude >/dev/null 2>&1; then
  install_claude
else
  log "Claude Code is already installed: $(claude --version)"
fi

require_command node
require_command npm

log "Checking Claude Flow / Ruflo CLI availability"
npx -y @claude-flow/cli@latest --version

if [[ -f "${REPO_ROOT}/.mcp.json" ]]; then
  log "Project MCP configuration already exists at .mcp.json"
fi

if [[ -f "${REPO_ROOT}/.claude/settings.local.json" ]]; then
  log "Project Claude local settings already exist"
fi

cat <<'EOF'

Next steps:
  1. Run `claude` once to authenticate if you have not already.
  2. Optional: start the Claude Flow daemon with:
       npx -y @claude-flow/cli@latest daemon start
  3. Optional: verify MCP visibility with:
       claude mcp list

EOF

log "Done"