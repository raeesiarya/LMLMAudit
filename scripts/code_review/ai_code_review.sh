#!/usr/bin/env bash
# ai_code_review.sh — multi-agent code review
# Agents: security | architecture | testing | ml-correctness
# Output: outputs/review/ai_code_review_YYYY-MM-DD.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

BASE_REF="${1:-origin/main}"
OUTPUT_DIR="${REPO_ROOT}/outputs/review"
DATE="$(date +%Y-%m-%d)"
REPORT_FILE="${OUTPUT_DIR}/ai_code_review_${DATE}.md"
WORK_DIR="$(mktemp -d)"
DIFF_FILE="${WORK_DIR}/diff.patch"
trap 'rm -rf "${WORK_DIR}"' EXIT

log() { printf '[ai_code_review] %s\n' "$*"; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || { log "Missing required command: $1" >&2; exit 1; }
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/ai_code_review.sh [base-ref]

Examples:
  ./scripts/ai_code_review.sh
  ./scripts/ai_code_review.sh origin/main

Output:
  outputs/review/ai_code_review_YYYY-MM-DD.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

require_command git
require_command claude
require_command python3

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

if ! git rev-parse --verify "${BASE_REF}" >/dev/null 2>&1; then
  log "Base ref '${BASE_REF}' not found locally. Try: git fetch origin ${BASE_REF#origin/}"
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
MERGE_BASE="$(git merge-base "${BASE_REF}" HEAD)"

log "Branch: ${CURRENT_BRANCH:-DETACHED}  Base: ${BASE_REF}  Merge-base: ${MERGE_BASE}"

# Exclude large data files — only review source code, tests, scripts, configs
DIFF_PATHSPEC=(
  ':(exclude)data/'
  ':(exclude)*.jsonl'
  ':(exclude)*.json'
  ':(exclude)coverage.*'
  ':(exclude)badges/'
  ':(exclude)*.svg'
  ':(exclude)*.xml'
)

git --no-pager diff --stat "${BASE_REF}...HEAD" -- "${DIFF_PATHSPEC[@]}"
git --no-pager diff "${BASE_REF}...HEAD" -- "${DIFF_PATHSPEC[@]}" > "${DIFF_FILE}"

if [[ ! -s "${DIFF_FILE}" ]]; then
  log "No diff found between ${BASE_REF} and HEAD (after excluding data files)."
  exit 0
fi

log "Diff size: $(wc -l < "${DIFF_FILE}") lines"

# Truncate to 4000 lines to stay within Claude's context for each agent
DIFF_CONTENT="$(head -n 4000 "${DIFF_FILE}")"
if [[ "$(wc -l < "${DIFF_FILE}")" -gt 4000 ]]; then
  DIFF_CONTENT="${DIFF_CONTENT}"$'\n\n[... diff truncated at 4000 lines ...]'
  log "Warning: diff truncated to 4000 lines"
fi
CONTEXT="Base: ${BASE_REF} | Head: ${CURRENT_BRANCH:-HEAD} | Merge-base: ${MERGE_BASE}"

# ── helper: run one agent ──────────────────────────────────────────────────
run_agent() {
  local role="$1"
  local focus="$2"
  local out_file="${WORK_DIR}/${role}.json"

  local prompt
  prompt="$(cat <<EOF
You are the ${role} reviewer in a multi-agent code review panel.

Context: ${CONTEXT}

Your focus: ${focus}

Instructions:
- Review ONLY the diff below through your specific lens.
- Be concrete: reference file names and line ranges when relevant.
- Rate overall severity for your domain: LOW | MEDIUM | HIGH | CRITICAL.
- Return a JSON object with exactly these fields:
    "severity": one of LOW|MEDIUM|HIGH|CRITICAL,
    "findings": array of objects each with "file", "issue", "recommendation",
    "summary": one-paragraph markdown summary of your review.

<diff>
${DIFF_CONTENT}
</diff>
EOF
)"

  log "Running ${role} agent..."
  if echo "${prompt}" | claude -p \
    --output-format json \
    --model sonnet \
    --permission-mode bypassPermissions \
    --json-schema '{
      "type":"object",
      "properties":{
        "severity":{"type":"string","enum":["LOW","MEDIUM","HIGH","CRITICAL"]},
        "findings":{"type":"array","items":{"type":"object","properties":{"file":{"type":"string"},"issue":{"type":"string"},"recommendation":{"type":"string"}},"required":["file","issue","recommendation"],"additionalProperties":false}},
        "summary":{"type":"string"}
      },
      "required":["severity","findings","summary"],
      "additionalProperties":false
    }' > "${out_file}" 2>"${WORK_DIR}/${role}.err"; then
    log "${role} agent done."
  else
    log "WARNING: ${role} agent failed (exit $?). Error: $(cat "${WORK_DIR}/${role}.err")"
    echo '{"severity":"UNKNOWN","findings":[],"summary":"Agent failed to run."}' > "${out_file}"
  fi
}

# ── spawn the four agents (sequential to stay within rate limits) ──────────
run_agent "security" \
  "Identify security vulnerabilities: injection, insecure deserialization, hardcoded secrets, unsafe file I/O, dependency risks, authentication/authorization issues."

run_agent "architecture" \
  "Evaluate structural quality: separation of concerns, coupling, cohesion, abstraction layers, naming conventions, API design, and maintainability."

run_agent "testing" \
  "Assess test coverage and quality: missing unit/integration tests, brittle assertions, improper mocking, untested edge cases, and regression gaps introduced by this diff."

run_agent "ml-correctness" \
  "Review ML/data correctness: data leakage, train/val/test splits, metric calculation, reproducibility (seeds), model loading/saving integrity, numerical stability, and evaluation fairness."

# ── synthesise into markdown ───────────────────────────────────────────────
log "Synthesising report..."
python3 - <<'PY' \
  "${WORK_DIR}/security.json" \
  "${WORK_DIR}/architecture.json" \
  "${WORK_DIR}/testing.json" \
  "${WORK_DIR}/ml-correctness.json" \
  "${REPORT_FILE}" \
  "${BASE_REF}" \
  "${CURRENT_BRANCH}" \
  "${DATE}"

import json, sys
from pathlib import Path

roles = ["Security", "Architecture", "Testing", "ML Correctness"]
files = sys.argv[1:5]
report_path = Path(sys.argv[5])
base_ref, branch, date = sys.argv[6], sys.argv[7], sys.argv[8]

agents = []
for role, f in zip(roles, files):
    try:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        agents.append((role, data))
    except Exception as e:
        agents.append((role, {"severity": "UNKNOWN", "findings": [], "summary": f"Agent failed: {e}"}))

severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
overall = min((a[1].get("severity", "UNKNOWN") for a in agents),
              key=lambda s: severity_order.get(s, 4))

# ── report ─────────────────────────────────────────────────────────────────
lines = [
    f"# AI Code Review — {date}",
    "",
    f"| Field | Value |",
    f"|---|---|",
    f"| **Base ref** | `{base_ref}` |",
    f"| **Head ref** | `{branch or 'HEAD'}` |",
    f"| **Overall severity** | **{overall}** |",
    "",
    "---",
    "",
    "## Agent Summaries",
    "",
    "| Agent | Severity | Summary |",
    "|---|---|---|",
]
for role, data in agents:
    sev = data.get("severity", "UNKNOWN")
    summary = data.get("summary", "").replace("\n", " ").replace("|", "\\|")
    lines.append(f"| {role} | {sev} | {summary} |")

lines += ["", "---", "", "## Detailed Findings", ""]

for role, data in agents:
    findings = data.get("findings", [])
    lines.append(f"### {role}")
    lines.append("")
    if not findings:
        lines.append("_No findings._")
    else:
        lines.append("| File | Issue | Recommendation |")
        lines.append("|---|---|---|")
        for f in findings:
            file_ = str(f.get("file", "—")).replace("|", "\\|")
            issue = str(f.get("issue", "—")).replace("|", "\\|").replace("\n", " ")
            rec   = str(f.get("recommendation", "—")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{file_}` | {issue} | {rec} |")
    lines.append("")

report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written to {report_path}")
PY

log "Report saved to ${REPORT_FILE}"
