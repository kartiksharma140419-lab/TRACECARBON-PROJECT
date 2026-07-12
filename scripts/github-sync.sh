#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TraceCarbon — GitHub Sync Script
# Usage:  bash scripts/github-sync.sh "optional commit message"
#
# What it does:
#   1. Re-injects the current GITHUB_PERSONAL_ACCESS_TOKEN into the remote URL
#      (so a rotated token never silently breaks pushes)
#   2. Stages all changes (respecting .gitignore — secrets/model files excluded)
#   3. Commits with a timestamped message (or your custom message)
#   4. Pushes to origin/main
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Token & remote ────────────────────────────────────────────────────────────
if [[ -z "${GITHUB_PERSONAL_ACCESS_TOKEN:-}" ]]; then
  echo "❌  GITHUB_PERSONAL_ACCESS_TOKEN is not set. Add it as a Replit Secret."
  exit 1
fi

GH_USER="kartiksharma140419-lab"
GH_REPO="TRACECARBON-PROJECT"
REMOTE_URL="https://${GH_USER}:${GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git"

git remote set-url origin "$REMOTE_URL"
echo "🔑  Remote URL refreshed with current token."

# ── Stage ────────────────────────────────────────────────────────────────────
git add -A

if git diff --cached --quiet; then
  echo "✅  Nothing to commit — working tree is clean."
  exit 0
fi

# ── Commit ───────────────────────────────────────────────────────────────────
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
MSG="${1:-"chore: auto-sync ${TIMESTAMP}"}"

echo "📝  Committing: ${MSG}"
git commit -m "${MSG}"

# ── Push ─────────────────────────────────────────────────────────────────────
echo "🚀  Pushing to origin/main …"
git push origin main

SHORT_SHA="$(git rev-parse --short HEAD)"
echo "✅  GitHub sync complete — commit ${SHORT_SHA}"
echo "🔗  https://github.com/${GH_USER}/${GH_REPO}/commit/${SHORT_SHA}"
