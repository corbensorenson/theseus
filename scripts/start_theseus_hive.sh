#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

DASHBOARD_PORT="${DASHBOARD_PORT:-8787}"
HIVE_PORT="${HIVE_PORT:-8791}"
RELAY_PORT="${RELAY_PORT:-8793}"
PYTHON="${PYTHON:-python3}"
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [ -x "$ROOT/.venv-puffer/bin/python" ]; then
  PYTHON="$ROOT/.venv-puffer/bin/python"
fi

if [ -z "${THESEUS_RUNTIME_ROOT:-}" ]; then
  if [ -d /Volumes/ProjectTheseus ]; then
    THESEUS_RUNTIME_ROOT="/Volumes/ProjectTheseus/runtime"
  elif [ "$(uname -s 2>/dev/null || printf unknown)" = "Darwin" ]; then
    THESEUS_RUNTIME_ROOT="$HOME/Library/Application Support/Project Theseus Hive/runtime"
  else
    THESEUS_RUNTIME_ROOT="$HOME/.local/share/project-theseus/runtime"
  fi
fi
export THESEUS_RUNTIME_ROOT
export THESEUS_DATA_DIR="${THESEUS_DATA_DIR:-$THESEUS_RUNTIME_ROOT/data}"
export THESEUS_CACHE_DIR="${THESEUS_CACHE_DIR:-$THESEUS_RUNTIME_ROOT/cache}"
export THESEUS_REPORTS_DIR="${THESEUS_REPORTS_DIR:-$THESEUS_RUNTIME_ROOT/reports}"
export THESEUS_CHECKPOINTS_DIR="${THESEUS_CHECKPOINTS_DIR:-$THESEUS_RUNTIME_ROOT/checkpoints}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$THESEUS_RUNTIME_ROOT/cargo-target}"
"$PYTHON" scripts/runtime_paths.py status --create >/dev/null 2>&1 || true
LOG_DIR="${THESEUS_RUNTIME_ROOT}/logs"
mkdir -p "$LOG_DIR"

if [ "${THESEUS_FOREGROUND:-0}" = "1" ]; then
  if [ "${THESEUS_NO_DASHBOARD:-0}" != "1" ]; then
    "$PYTHON" scripts/sparkstream_dashboard.py --host 127.0.0.1 --port "$DASHBOARD_PORT" >"$LOG_DIR/project-theseus-dashboard.log" 2>&1 &
  fi

  if [ "${THESEUS_START_RELAY:-0}" = "1" ]; then
    "$PYTHON" scripts/hive_relay.py --port "$RELAY_PORT" >"$LOG_DIR/project-theseus-hive-relay.log" 2>&1 &
  fi

  (
    "$PYTHON" scripts/hive_scheduler.py --out reports/hive_scheduler.json --sync-artifacts >/dev/null 2>&1 || true
  ) &
  (
    "$PYTHON" scripts/hive_version_manager.py status --out reports/hive_version_status.json >/dev/null 2>&1 || true
  ) &
  (
    "$PYTHON" scripts/update_manager.py check --if-enabled-on-start --respect-interval --out reports/update_checkin.json >/dev/null 2>&1 || true
  ) &

  printf '%s\n' "Project Theseus dashboard: http://127.0.0.1:$DASHBOARD_PORT"
  printf '%s\n' "Project Theseus Hive node: http://127.0.0.1:$HIVE_PORT/api/hive/status"
  if [ "${THESEUS_START_RELAY:-0}" = "1" ]; then
    printf '%s\n' "Project Theseus Hive relay: http://127.0.0.1:$RELAY_PORT/mobile"
  fi

  if [ "${THESEUS_NO_HIVE:-0}" != "1" ]; then
    exec "$PYTHON" scripts/hive_node.py daemon --port "$HIVE_PORT"
  fi

  wait
  exit 0
fi

if [ "${THESEUS_NO_DASHBOARD:-0}" != "1" ]; then
  "$PYTHON" scripts/sparkstream_dashboard.py --host 127.0.0.1 --port "$DASHBOARD_PORT" >"$LOG_DIR/project-theseus-dashboard.log" 2>&1 &
fi

if [ "${THESEUS_NO_HIVE:-0}" != "1" ]; then
  "$PYTHON" scripts/hive_node.py daemon --port "$HIVE_PORT" >"$LOG_DIR/project-theseus-hive.log" 2>&1 &
fi

if [ "${THESEUS_START_RELAY:-0}" = "1" ]; then
  "$PYTHON" scripts/hive_relay.py --port "$RELAY_PORT" >"$LOG_DIR/project-theseus-hive-relay.log" 2>&1 &
fi

"$PYTHON" scripts/hive_scheduler.py --out reports/hive_scheduler.json --sync-artifacts >/dev/null 2>&1 || true
"$PYTHON" scripts/hive_version_manager.py status --out reports/hive_version_status.json >/dev/null 2>&1 || true
"$PYTHON" scripts/update_manager.py check --if-enabled-on-start --respect-interval --out reports/update_checkin.json >/dev/null 2>&1 || true

printf '%s\n' "Project Theseus dashboard: http://127.0.0.1:$DASHBOARD_PORT"
printf '%s\n' "Project Theseus Hive node: http://127.0.0.1:$HIVE_PORT/api/hive/status"
if [ "${THESEUS_START_RELAY:-0}" = "1" ]; then
  printf '%s\n' "Project Theseus Hive relay: http://127.0.0.1:$RELAY_PORT/mobile"
fi
