#!/usr/bin/env sh
set -eu

SOURCE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
INSTALL_ROOT="${THESEUS_LINUX_INSTALL_ROOT:-$HOME/.local/share/project-theseus-hive/app/current}"
RUNTIME_ROOT="${THESEUS_RUNTIME_ROOT:-$HOME/.local/share/project-theseus-hive/runtime}"
CACHE_ROOT="${THESEUS_CACHE_DIR:-$HOME/.cache/project-theseus-hive}"
INVITE=""
RELAY_URL=""
COORDINATOR_URL=""
START_NOW="0"
COPY_PAYLOAD="1"
REGISTER_COMMUNITY="1"
PUBLIC_MODE="off"
PUBLIC_GATEWAY_URL=""
PUBLIC_WORKER_NAME=""
ALLOW_PUBLIC="0"
INSTALL_SERVICE="1"
ENABLE_SERVICE="0"
AUTO_UPDATE_SOFT="0"
VACATION_MODE_SERVICE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      SOURCE_ROOT="${2:-}"
      shift 2
      ;;
    --install-root)
      INSTALL_ROOT="${2:-}"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="${2:-}"
      shift 2
      ;;
    --invite)
      INVITE="${2:-}"
      shift 2
      ;;
    --relay-url)
      RELAY_URL="${2:-}"
      shift 2
      ;;
    --coordinator-url)
      COORDINATOR_URL="${2:-}"
      shift 2
      ;;
    --start)
      START_NOW="1"
      shift
      ;;
    --no-copy)
      COPY_PAYLOAD="0"
      shift
      ;;
    --skip-registration)
      REGISTER_COMMUNITY="0"
      shift
      ;;
    --public-mode)
      PUBLIC_MODE="${2:-off}"
      shift 2
      ;;
    --public-gateway-url)
      PUBLIC_GATEWAY_URL="${2:-}"
      shift 2
      ;;
    --public-worker-name)
      PUBLIC_WORKER_NAME="${2:-}"
      shift 2
      ;;
    --allow-public)
      ALLOW_PUBLIC="1"
      shift
      ;;
    --install-service)
      INSTALL_SERVICE="1"
      shift
      ;;
    --no-install-service)
      INSTALL_SERVICE="0"
      shift
      ;;
    --enable-service)
      INSTALL_SERVICE="1"
      ENABLE_SERVICE="1"
      shift
      ;;
    --auto-update-soft)
      AUTO_UPDATE_SOFT="1"
      shift
      ;;
    --vacation-mode-service)
      VACATION_MODE_SERVICE="1"
      shift
      ;;
    *)
      printf '%s\n' "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SOURCE_ROOT="$(CDPATH= cd -- "$SOURCE_ROOT" && pwd)"
mkdir -p "$INSTALL_ROOT" "$RUNTIME_ROOT" "$RUNTIME_ROOT/logs" "$CACHE_ROOT"

LOCAL_CONFIG_BACKUP=""
backup_local_configs() {
  LOCAL_CONFIG_BACKUP="${TMPDIR:-/tmp}/theseus_hive_local_configs_$$"
  rm -rf "$LOCAL_CONFIG_BACKUP"
  mkdir -p "$LOCAL_CONFIG_BACKUP"
  if [ -d "$INSTALL_ROOT/configs" ]; then
    find "$INSTALL_ROOT/configs" -maxdepth 1 -type f \( -name '*.local.json' -o -name '*.secret.json' \) -exec cp {} "$LOCAL_CONFIG_BACKUP/" \; 2>/dev/null || true
  fi
}

restore_local_configs() {
  if [ -n "$LOCAL_CONFIG_BACKUP" ] && [ -d "$LOCAL_CONFIG_BACKUP" ]; then
    mkdir -p "$INSTALL_ROOT/configs"
    find "$LOCAL_CONFIG_BACKUP" -maxdepth 1 -type f -exec cp {} "$INSTALL_ROOT/configs/" \; 2>/dev/null || true
    rm -rf "$LOCAL_CONFIG_BACKUP"
  fi
}

if [ "$COPY_PAYLOAD" = "1" ]; then
  backup_local_configs
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude '.git/' \
      --exclude '.attd_tmp/' \
      --exclude '.venv*/' \
      --exclude 'dist/' \
      --exclude 'games/' \
      --exclude 'node_modules/' \
      --exclude 'target/' \
      --exclude 'reports/' \
      --exclude 'checkpoints/' \
      --exclude 'configs/*.local.json' \
      --exclude 'configs/*.secret.json' \
      --exclude 'tmp/' \
      --exclude 'updates/' \
      --exclude 'vendor/' \
      --exclude 'data/.cache/' \
      --exclude 'data/external_benchmark_candidates/' \
      --exclude 'data/local_roms/' \
      --exclude 'data/public_benchmarks/' \
      --exclude 'data/rom_manifests/' \
      --exclude 'data/synthetic/' \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      "$SOURCE_ROOT/" "$INSTALL_ROOT/"
  else
    rm -rf "$INSTALL_ROOT"
    mkdir -p "$(dirname "$INSTALL_ROOT")"
    cp -R "$SOURCE_ROOT" "$INSTALL_ROOT"
    rm -rf "$INSTALL_ROOT/.git" "$INSTALL_ROOT/.attd_tmp" "$INSTALL_ROOT"/.venv* "$INSTALL_ROOT/dist" "$INSTALL_ROOT/games" "$INSTALL_ROOT/node_modules" "$INSTALL_ROOT/target" "$INSTALL_ROOT/reports" "$INSTALL_ROOT/checkpoints" "$INSTALL_ROOT/tmp" "$INSTALL_ROOT/updates" "$INSTALL_ROOT/vendor"
    rm -f "$INSTALL_ROOT"/configs/*.local.json "$INSTALL_ROOT"/configs/*.secret.json
    rm -rf "$INSTALL_ROOT/data/.cache" "$INSTALL_ROOT/data/external_benchmark_candidates" "$INSTALL_ROOT/data/local_roms" "$INSTALL_ROOT/data/public_benchmarks" "$INSTALL_ROOT/data/rom_manifests" "$INSTALL_ROOT/data/synthetic"
  fi
  restore_local_configs
fi

cd "$INSTALL_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$INSTALL_ROOT/.venv-puffer/bin/python" ]; then
  printf '%s\n' "python3 is required. Install Python 3, then rerun this installer." >&2
  exit 2
fi

if [ ! -x "$INSTALL_ROOT/.venv-puffer/bin/python" ]; then
  "$PYTHON" -m venv "$INSTALL_ROOT/.venv-puffer"
  PYTHON="$INSTALL_ROOT/.venv-puffer/bin/python"
  "$PYTHON" -m pip install --upgrade pip wheel setuptools >/dev/null
  "$PYTHON" -m pip install numpy >/dev/null
else
  PYTHON="$INSTALL_ROOT/.venv-puffer/bin/python"
fi

"$PYTHON" - "$INSTALL_ROOT" <<'PY' >/dev/null
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
for pattern in ("*.sh", "*.command", "*.desktop"):
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        data = path.read_bytes()
        fixed = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if fixed != data:
            path.write_bytes(fixed)
        try:
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
PY

mkdir -p "$RUNTIME_ROOT/reports" "$RUNTIME_ROOT/checkpoints" "$RUNTIME_ROOT/cargo-target" "$RUNTIME_ROOT/data" "$CACHE_ROOT"
for link_name in reports checkpoints target; do
  case "$link_name" in
    reports) target_dir="$RUNTIME_ROOT/reports" ;;
    checkpoints) target_dir="$RUNTIME_ROOT/checkpoints" ;;
    target) target_dir="$RUNTIME_ROOT/cargo-target" ;;
  esac
  if [ -L "$link_name" ]; then
    rm -f "$link_name"
  elif [ -d "$link_name" ]; then
    if [ "$(find "$link_name" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')" != "0" ]; then
      mv "$link_name" "$target_dir/migrated-$(date +%Y%m%d%H%M%S)"
    else
      rmdir "$link_name"
    fi
  fi
  [ -e "$link_name" ] || ln -s "$target_dir" "$link_name"
done

export THESEUS_RUNTIME_ROOT="$RUNTIME_ROOT"
export THESEUS_DATA_DIR="$RUNTIME_ROOT/data"
export THESEUS_CACHE_DIR="$CACHE_ROOT"
export THESEUS_REPORTS_DIR="$RUNTIME_ROOT/reports"
export THESEUS_CHECKPOINTS_DIR="$RUNTIME_ROOT/checkpoints"
export CARGO_TARGET_DIR="$RUNTIME_ROOT/cargo-target"

"$PYTHON" scripts/runtime_paths.py init --runtime-root "$RUNTIME_ROOT" >/dev/null
if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
  "$PYTHON" scripts/update_manager.py configure \
    --mode auto_soft \
    --check-on-start \
    --auto-install-soft \
    --no-auto-install-hard \
    --out reports/update_client_configure_install.json >/dev/null
fi
if [ "$REGISTER_COMMUNITY" = "1" ] && [ ! -f "configs/theseus_registration.local.json" ]; then
  "$PYTHON" scripts/license_manager.py register \
    --usage personal_homelab \
    --accept-terms \
    --out reports/license_registration_linux_install.json >/dev/null
fi
if [ -n "$INVITE" ]; then
  "$PYTHON" scripts/hive_invite.py apply --invite "$INVITE" --write-local-config --out reports/hive_join_apply_last.json >/dev/null
fi
if [ -n "$RELAY_URL" ] || [ -n "$COORDINATOR_URL" ]; then
  "$PYTHON" scripts/hive_invite.py configure-local \
    --relay-url "$RELAY_URL" \
    --coordinator-url "$COORDINATOR_URL" \
    --out reports/hive_join_configure_last.json >/dev/null
fi
if [ "$PUBLIC_MODE" != "off" ] || [ -n "$PUBLIC_GATEWAY_URL" ]; then
  if [ "$ALLOW_PUBLIC" = "1" ]; then
    "$PYTHON" scripts/public_hive_contributor.py configure \
      --mode "$PUBLIC_MODE" \
      --gateway-url "$PUBLIC_GATEWAY_URL" \
      --worker-name "$PUBLIC_WORKER_NAME" \
      --allow \
      --out reports/public_hive_contribution_status.json >/dev/null
  else
    "$PYTHON" scripts/public_hive_contributor.py configure \
      --mode "$PUBLIC_MODE" \
      --gateway-url "$PUBLIC_GATEWAY_URL" \
      --worker-name "$PUBLIC_WORKER_NAME" \
      --out reports/public_hive_contribution_status.json >/dev/null
  fi
fi

if [ "$INSTALL_SERVICE" = "1" ] && command -v systemctl >/dev/null 2>&1; then
  USER_DIR="$HOME/.config/systemd/user"
  mkdir -p "$USER_DIR"
  cat > "$USER_DIR/project-theseus-hive.service" <<EOF
[Unit]
Description=Project Theseus Hive node
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_ROOT
ExecStart=$INSTALL_ROOT/scripts/start_theseus_hive.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
  if [ "$VACATION_MODE_SERVICE" = "1" ]; then
    cat > "$USER_DIR/project-theseus-vacation.service" <<EOF
[Unit]
Description=Project Theseus Vacation Mode Supervisor
After=network-online.target project-theseus-hive.service

[Service]
Type=oneshot
WorkingDirectory=$INSTALL_ROOT
ExecStart=$PYTHON $INSTALL_ROOT/scripts/vacation_mode_supervisor.py --cycles 1 --execute --start-services --out reports/vacation_mode_supervisor.json --markdown-out reports/vacation_mode_supervisor.md
EOF
    cat > "$USER_DIR/project-theseus-vacation.timer" <<EOF
[Unit]
Description=Run Project Theseus Vacation Mode periodically

[Timer]
OnBootSec=5min
OnUnitActiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
EOF
  fi
  if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
    cat > "$USER_DIR/project-theseus-update.service" <<EOF
[Unit]
Description=Project Theseus Hive update check
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$INSTALL_ROOT
ExecStart=$PYTHON $INSTALL_ROOT/scripts/update_manager.py check --apply --respect-interval --out reports/update_checkin.json
EOF
    cat > "$USER_DIR/project-theseus-update.timer" <<EOF
[Unit]
Description=Check Project Theseus private Hive updates periodically

[Timer]
OnBootSec=3min
OnUnitActiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
EOF
  fi
  systemctl --user daemon-reload || true
  if [ "$ENABLE_SERVICE" = "1" ]; then
    systemctl --user enable --now project-theseus-hive.service || true
    if [ "$VACATION_MODE_SERVICE" = "1" ]; then
      systemctl --user enable --now project-theseus-vacation.timer || true
    fi
    if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
      systemctl --user enable --now project-theseus-update.timer || true
    fi
  fi
fi

DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/project-theseus-setup.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Project Theseus Setup
Comment=Set up or join a Project Theseus Hive
Exec=$PYTHON $INSTALL_ROOT/scripts/theseus_setup_wizard.py --open
Path=$INSTALL_ROOT
Terminal=false
Categories=Utility;Development;
EOF

"$PYTHON" scripts/hive_node.py probe --out reports/hive_status.json >/dev/null
"$PYTHON" scripts/hive_scheduler.py --out reports/hive_scheduler.json >/dev/null || true

if [ "$START_NOW" = "1" ]; then
  "$INSTALL_ROOT/scripts/start_theseus_hive.sh"
fi

printf '%s\n' "Project Theseus payload installed at: $INSTALL_ROOT"
printf '%s\n' "Project Theseus runtime root: $RUNTIME_ROOT"
printf '%s\n' "Desktop launcher written: $DESKTOP_DIR/project-theseus-setup.desktop"
if [ "$INSTALL_SERVICE" = "1" ] && command -v systemctl >/dev/null 2>&1; then
  printf '%s\n' "Systemd user service written: $HOME/.config/systemd/user/project-theseus-hive.service"
  printf '%s\n' "Enable later with: systemctl --user enable --now project-theseus-hive.service"
  if [ "$VACATION_MODE_SERVICE" = "1" ]; then
    printf '%s\n' "Systemd user timer written: $HOME/.config/systemd/user/project-theseus-vacation.timer"
  fi
  if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
    printf '%s\n' "Systemd user update timer written: $HOME/.config/systemd/user/project-theseus-update.timer"
  fi
fi
