#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TraceCarbon — GitHub Sync Script
# Usage: bash scripts/github-sync.sh "optional commit message"
# Stages all tracked/new files (respecting .gitignore), commits, and pushes.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MSG="${1:-"chore: auto-sync $(date -u '+%Y-%m-%dT%H:%M:%SZ')"}"

echo "🔍  Checking git status …"
git add -A

if git diff --cached --quiet; then
  echo "✅  Nothing to commit — working tree clean."
  exit 0
fi

echo "📝  Committing: $MSG"
git commit -m "$MSG"

echo "🚀  Pushing to origin/main …"
git push origin main

echo "✅  GitHub sync complete — $(git rev-parse --short HEAD)"
