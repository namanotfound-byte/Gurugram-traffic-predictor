#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run.sh — one-command launcher for the Gurugram Traffic Predictor.
#
# Starts the Flask backend (backend/app.py) using the pre-built
# .venv_backend virtualenv, serves frontend/ over plain HTTP (never
# file://), opens the browser at the right URL, and cleans up both
# processes on Ctrl-C.
#
# Usage:
#   ./run.sh                 # start everything, open the browser
#   ./run.sh --no-open       # start everything, don't open a browser
#   BACKEND_PORT=5050 ./run.sh
#   FRONTEND_PORT=9000 ./run.sh
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-5000}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
  esac
done

VENV="$ROOT/.venv_backend"
BACKEND_PY="$VENV/bin/python"

BACKEND_PID=""
FRONTEND_PID=""
REUSED_BACKEND=0
LOG_DIR="$(mktemp -d /tmp/gtp-run.XXXXXX)"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# ── colors (disabled if not a terminal) ────────────────────────────────
if [ -t 1 ]; then
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_BLD=$'\033[1m'; C_RST=$'\033[0m'
else
  C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_BLD=""; C_RST=""
fi
info(){ echo "${C_CYN}==>${C_RST} $*"; }
ok(){ echo "${C_GRN}✓${C_RST} $*"; }
warn(){ echo "${C_YEL}!${C_RST} $*"; }
fail(){ echo "${C_RED}✗ $*${C_RST}" >&2; }

# ── sanity: backend virtualenv ─────────────────────────────────────────
if [[ ! -x "$BACKEND_PY" ]]; then
  fail "Backend virtualenv not found at .venv_backend (expected $BACKEND_PY)."
  echo "Create it and install dependencies, then re-run ./run.sh:" >&2
  echo "" >&2
  echo "  python3 -m venv .venv_backend" >&2
  echo "  .venv_backend/bin/pip install -r backend/requirements.txt" >&2
  echo "" >&2
  exit 1
fi

if [[ ! -f "$ROOT/backend/app.py" ]]; then
  fail "backend/app.py not found. Are you running this from the repo root?"
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
  fail "lsof is required (used to detect ports already in use) but was not found on PATH."
  exit 1
fi

PY_STATIC="python3"
command -v python3 >/dev/null 2>&1 || PY_STATIC="python"
if ! command -v "$PY_STATIC" >/dev/null 2>&1; then
  fail "No python3/python found on PATH — needed to serve frontend/ over HTTP."
  exit 1
fi

# ── helpers ─────────────────────────────────────────────────────────────
port_in_use() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

find_free_port() {
  local p="$1"
  for _ in $(seq 1 50); do
    if ! port_in_use "$p"; then echo "$p"; return 0; fi
    p=$((p + 1))
  done
  return 1
}

# Is whatever is listening on $1 actually OUR Flask backend already?
looks_like_our_backend() {
  local p="$1"
  local body
  body="$(curl -s -m 2 "http://localhost:$p/health" 2>/dev/null)" || return 1
  [[ "$body" == *'"status"'* ]] && [[ "$body" == *'corridors'* ]]
}

cleanup() {
  echo ""
  info "Shutting down..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null
    wait "$FRONTEND_PID" 2>/dev/null
    ok "Stopped frontend server (pid $FRONTEND_PID)"
  fi
  if [[ "$REUSED_BACKEND" -eq 0 ]] && [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null
    wait "$BACKEND_PID" 2>/dev/null
    ok "Stopped backend server (pid $BACKEND_PID)"
  elif [[ "$REUSED_BACKEND" -eq 1 ]]; then
    warn "Backend on port $BACKEND_PORT was already running before this script started — left it running."
  fi
  exit 0
}
trap cleanup INT TERM

echo ""
echo "${C_BLD}Gurugram Traffic Predictor — launcher${C_RST}"
echo "──────────────────────────────────────"

# ── backend port: reuse if it's already our backend, else find a free one ─
if port_in_use "$BACKEND_PORT"; then
  if looks_like_our_backend "$BACKEND_PORT"; then
    warn "Port $BACKEND_PORT already has a running backend that answers /health — reusing it instead of starting a second copy."
    REUSED_BACKEND=1
  else
    warn "Port $BACKEND_PORT is already in use by something else (on macOS this is often AirPlay Receiver)."
    NEW_PORT="$(find_free_port $((BACKEND_PORT + 1)))"
    if [[ -z "$NEW_PORT" ]]; then
      fail "Could not find a free port for the backend near $BACKEND_PORT."
      exit 1
    fi
    info "Using port $NEW_PORT for the backend instead."
    BACKEND_PORT="$NEW_PORT"
  fi
fi

# ── start backend ─────────────────────────────────────────────────────
if [[ "$REUSED_BACKEND" -eq 0 ]]; then
  info "Starting backend (Flask) on port $BACKEND_PORT..."
  PORT="$BACKEND_PORT" "$BACKEND_PY" "$ROOT/backend/app.py" >"$BACKEND_LOG" 2>&1 &
  BACKEND_PID=$!

  # poll /health until it responds or the process dies
  READY=0
  for _ in $(seq 1 60); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      break
    fi
    if curl -s -m 1 -o /dev/null "http://localhost:$BACKEND_PORT/health" 2>/dev/null; then
      READY=1
      break
    fi
    sleep 0.25
  done

  if [[ "$READY" -ne 1 ]]; then
    fail "Backend did not come up on port $BACKEND_PORT. Last lines of its log ($BACKEND_LOG):"
    tail -n 25 "$BACKEND_LOG" >&2 2>/dev/null
    kill "$BACKEND_PID" 2>/dev/null
    exit 1
  fi
  ok "Backend is up (pid $BACKEND_PID) — http://localhost:$BACKEND_PORT"
fi

# ── frontend port: never reuse blindly, just find something free ────────
if port_in_use "$FRONTEND_PORT"; then
  warn "Port $FRONTEND_PORT is already in use."
  NEW_PORT="$(find_free_port $((FRONTEND_PORT + 1)))"
  if [[ -z "$NEW_PORT" ]]; then
    fail "Could not find a free port for the frontend near $FRONTEND_PORT."
    cleanup
  fi
  info "Using port $NEW_PORT for the frontend instead."
  FRONTEND_PORT="$NEW_PORT"
fi

# ── start frontend static server ────────────────────────────────────────
info "Starting frontend static server on port $FRONTEND_PORT..."
# --directory (not `cd` + a subshell) so $! is the http.server process
# itself, not a wrapper shell — kill "$FRONTEND_PID" in cleanup() must be
# able to reach the actual listening process directly, or it leaks an
# orphaned server on $FRONTEND_PORT every time this script exits.
"$PY_STATIC" -m http.server --directory "$ROOT/frontend" "$FRONTEND_PORT" >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

READY=0
for _ in $(seq 1 40); do
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then break; fi
  if curl -s -m 1 -o /dev/null "http://localhost:$FRONTEND_PORT/index.html" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.25
done

if [[ "$READY" -ne 1 ]]; then
  fail "Frontend server did not come up on port $FRONTEND_PORT. Last lines of its log ($FRONTEND_LOG):"
  tail -n 25 "$FRONTEND_LOG" >&2 2>/dev/null
  cleanup
fi
ok "Frontend is up (pid $FRONTEND_PID) — http://localhost:$FRONTEND_PORT"

APP_URL="http://localhost:$FRONTEND_PORT/index.html?api=http://localhost:$BACKEND_PORT"

echo ""
echo "${C_BLD}Ready.${C_RST}"
echo "  Backend  : http://localhost:$BACKEND_PORT $( [[ "$REUSED_BACKEND" -eq 1 ]] && echo "(already running, reused)" )"
echo "  Frontend : http://localhost:$FRONTEND_PORT"
echo "  Open     : $APP_URL"
echo "  Logs     : $LOG_DIR"
echo ""
echo "  Press Ctrl-C to stop."
echo ""

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$APP_URL"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$APP_URL" >/dev/null 2>&1
  else
    warn "Could not detect a way to open a browser automatically — open the URL above manually."
  fi
fi

# Keep the script alive so Ctrl-C hits our trap and we can clean up.
wait
