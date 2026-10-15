#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# setup.sh — one-time setup on a new machine.
#
# Rebuilds the three Python virtualenvs from the requirements files. The
# venvs are deliberately NOT included in the transfer archive: a venv hard-
# codes the absolute path it was created at, so a copied one breaks on a
# machine with a different username, and the installed packages are large
# (~1 GB) but perfectly reproducible from a few text files.
#
# Usage:
#   ./setup.sh          # create/refresh all three venvs
#   ./setup.sh --check  # verify an existing setup, install nothing
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; N=""
fi

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "${B}Gurugram Traffic Predictor — setup${N}"
echo

# ── 1. Python ───────────────────────────────────────────────────────────
# On a Mac without the Command Line Tools installed, /usr/bin/python3 exists
# as a stub that only pops the CLT installer and exits non-zero — so testing
# for the binary is not enough, it has to actually run.
clt_help() {
  echo "  Run this, click through the installer, then re-run ./setup.sh:"
  echo
  echo "      xcode-select --install"
  echo
  echo "  It installs Apple's Command Line Tools, which provide git and python3."
  echo "  Nothing else needs installing — no Homebrew, no Node."
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "${R}✗ python3 not found.${N}"
  clt_help
  exit 1
fi

if ! PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"; then
  echo "${R}✗ python3 is present but cannot run.${N}"
  echo "  This is the normal state of a new Mac: /usr/bin/python3 is only a stub"
  echo "  until Apple's Command Line Tools are installed."
  echo
  clt_help
  exit 1
fi
echo "${G}✓${N} python3 $PYV  ($(command -v python3))"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)'; then
  echo "${R}✗ Python 3.9 or newer is required (found $PYV).${N}"
  exit 1
fi

# ── 2. The virtualenvs ──────────────────────────────────────────────────
# name → requirements files to install into it
make_venv() {
  local name="$1"; shift
  local reqs=("$@")

  if [ "$CHECK_ONLY" = "1" ]; then
    if [ -x "$name/bin/python" ]; then
      echo "${G}✓${N} $name present"
    else
      echo "${Y}○${N} $name missing — run ./setup.sh"
    fi
    return
  fi

  echo
  echo "${B}→ $name${N}"
  if [ ! -x "$name/bin/python" ]; then
    python3 -m venv "$name" || { echo "${R}✗ could not create $name${N}"; return 1; }
  fi
  "$name/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1
  for r in "${reqs[@]}"; do
    echo "   installing $r"
    if ! "$name/bin/python" -m pip install --quiet -r "$r"; then
      echo "${R}✗ pip install -r $r failed for $name${N}"
      return 1
    fi
  done
  echo "${G}✓${N} $name ready"
}

# .venv_backend — what run.sh uses to serve the API and run the tests.
make_venv .venv_backend backend/requirements.txt

# .venv_bootstrap — the one-time grid sweep + model training.
make_venv .venv_bootstrap requirements.txt

# .venv_forecast — live collection and the gated residual model.
make_venv .venv_forecast requirements.txt requirements-collect.txt

# ── 3. API key ──────────────────────────────────────────────────────────
echo
if [ -f .env ] && grep -q '^TOMTOM_API_KEY=..' .env; then
  echo "${G}✓${N} .env present with a TOMTOM_API_KEY"
else
  echo "${Y}○${N} No TOMTOM_API_KEY in .env."
  echo "  Only the data collectors need it. Serving the site and running the"
  echo "  tests work fine without one — they read the CSVs already in data/."
  echo "  To add one:  echo 'TOMTOM_API_KEY=your-key' > .env"
fi

chmod +x run.sh setup.sh 2>/dev/null

# ── 4. Done ─────────────────────────────────────────────────────────────
if [ "$CHECK_ONLY" = "1" ]; then
  exit 0
fi
echo
echo "${B}Setup complete.${N}"
echo
echo "  ./run.sh                                    start the site"
echo "  .venv_backend/bin/python -m pytest backend/test_api.py -q    run the tests"
echo
