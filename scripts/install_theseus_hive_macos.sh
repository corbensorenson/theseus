#!/usr/bin/env sh
set -eu

SOURCE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
INSTALL_ROOT="${THESEUS_MACOS_INSTALL_ROOT:-$HOME/Library/Application Support/Project Theseus Hive/app/current}"
SUPPORT_ROOT="${THESEUS_MACOS_SUPPORT_ROOT:-$HOME/Library/Application Support/Project Theseus Hive}"
RUNTIME_ROOT="${THESEUS_RUNTIME_ROOT:-$SUPPORT_ROOT/runtime}"
CACHE_ROOT="${THESEUS_CACHE_DIR:-$HOME/Library/Caches/Project Theseus Hive}"
APPLICATIONS_DIR=""
INVITE=""
RELAY_URL=""
COORDINATOR_URL=""
START_NOW="0"
INSTALL_DEPS="1"
REQUIRE_MLX="0"
COPY_PAYLOAD="1"
REGISTER_COMMUNITY="1"
PUBLIC_MODE="off"
PUBLIC_GATEWAY_URL=""
PUBLIC_WORKER_NAME=""
ALLOW_PUBLIC="0"
INSTALL_SERVICE="0"
ENABLE_SERVICE="0"
AUTO_UPDATE_SOFT="0"
VACATION_MODE_SERVICE="0"
MACOS_MIN_VERSION="${THESEUS_MACOS_MIN_VERSION:-10.15}"

log_step() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

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
    --applications-dir)
      APPLICATIONS_DIR="${2:-}"
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
    --skip-deps)
      INSTALL_DEPS="0"
      shift
      ;;
    --require-mlx)
      REQUIRE_MLX="1"
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
if [ -z "$INVITE" ]; then
  for invite_candidate in "$SOURCE_ROOT/ProjectTheseusHive.join.json" "$SOURCE_ROOT/hive_join_bundle.json" "$SOURCE_ROOT/reports/hive_join_bundle.json"; do
    if [ -f "$invite_candidate" ]; then
      INVITE="$invite_candidate"
      log_step "Found bundled Hive join profile: $INVITE"
      break
    fi
  done
fi
if [ -z "$APPLICATIONS_DIR" ]; then
  if [ -d /Applications ] && [ -w /Applications ]; then
    APPLICATIONS_DIR="/Applications"
  else
    APPLICATIONS_DIR="$HOME/Applications"
  fi
fi
log_step "Project Theseus Hive macOS install starting."
log_step "Source: $SOURCE_ROOT"
log_step "Install root: $INSTALL_ROOT"
log_step "Runtime root: $RUNTIME_ROOT"
log_step "Applications: $APPLICATIONS_DIR"

PYTHON="${PYTHON:-python3}"
BUILD_VERSION_TMP="${TMPDIR:-/tmp}/theseus_hive_build_version_$$.json"
rm -f "$BUILD_VERSION_TMP"
if [ -f "$SOURCE_ROOT/scripts/hive_version_manager.py" ] && command -v "$PYTHON" >/dev/null 2>&1; then
  log_step "Writing packaged Hive build manifest."
  (cd "$SOURCE_ROOT" && "$PYTHON" scripts/hive_version_manager.py build-manifest --out "$BUILD_VERSION_TMP" >/dev/null 2>&1) || true
fi

log_step "Creating install, runtime, cache, and app directories."
mkdir -p "$INSTALL_ROOT" "$RUNTIME_ROOT" "$RUNTIME_ROOT/logs" "$CACHE_ROOT" "$APPLICATIONS_DIR"

if [ "$COPY_PAYLOAD" = "1" ]; then
  backup_local_configs
  if command -v rsync >/dev/null 2>&1; then
    log_step "Copying application payload with rsync."
    rsync -a --delete \
      --exclude '.git/' \
      --exclude '.attd_tmp/' \
      --exclude '.venv-puffer/' \
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
    log_step "Copying application payload with ditto."
    rm -rf "$INSTALL_ROOT"
    mkdir -p "$(dirname "$INSTALL_ROOT")"
    ditto "$SOURCE_ROOT" "$INSTALL_ROOT"
    rm -rf "$INSTALL_ROOT/.git" "$INSTALL_ROOT/.attd_tmp" "$INSTALL_ROOT"/.venv* "$INSTALL_ROOT/dist" "$INSTALL_ROOT/games" "$INSTALL_ROOT/node_modules" "$INSTALL_ROOT/target" "$INSTALL_ROOT/reports" "$INSTALL_ROOT/checkpoints" "$INSTALL_ROOT/tmp" "$INSTALL_ROOT/updates" "$INSTALL_ROOT/vendor"
    rm -f "$INSTALL_ROOT"/configs/*.local.json "$INSTALL_ROOT"/configs/*.secret.json
    rm -rf "$INSTALL_ROOT/data/.cache" "$INSTALL_ROOT/data/external_benchmark_candidates" "$INSTALL_ROOT/data/local_roms" "$INSTALL_ROOT/data/public_benchmarks" "$INSTALL_ROOT/data/rom_manifests" "$INSTALL_ROOT/data/synthetic"
  fi
  restore_local_configs
fi

cd "$INSTALL_ROOT"

log_step "Preparing runtime links."
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
if [ -d "$SOURCE_ROOT/dist" ]; then
  rm -rf "$INSTALL_ROOT/dist"
  ln -s "$SOURCE_ROOT/dist" "$INSTALL_ROOT/dist"
fi
if [ -f "$BUILD_VERSION_TMP" ]; then
  mkdir -p "$INSTALL_ROOT/configs"
  cp "$BUILD_VERSION_TMP" "$INSTALL_ROOT/configs/hive_build_version.json"
  rm -f "$BUILD_VERSION_TMP"
fi

PYTHON="${PYTHON:-python3}"
if [ -x "$INSTALL_ROOT/.venv-puffer/bin/python" ]; then
  PYTHON="$INSTALL_ROOT/.venv-puffer/bin/python"
fi
if ! command -v python3 >/dev/null 2>&1 && [ ! -x "$INSTALL_ROOT/.venv-puffer/bin/python" ]; then
  if command -v brew >/dev/null 2>&1; then
    brew install python
  else
    printf '%s\n' "python3 is required. Install Xcode Command Line Tools or Homebrew Python, then rerun this installer." >&2
    printf '%s\n' "Tip: xcode-select --install" >&2
    exit 2
  fi
fi

"$PYTHON" - "$INSTALL_ROOT" <<'PY' >/dev/null
from pathlib import Path
import os
import stat
import sys

root = Path(sys.argv[1])
patterns = ["*.sh", "*.command", "*.desktop"]
for pattern in patterns:
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

export THESEUS_RUNTIME_ROOT="$RUNTIME_ROOT"
export THESEUS_DATA_DIR="$RUNTIME_ROOT/data"
export THESEUS_CACHE_DIR="$CACHE_ROOT"
export THESEUS_REPORTS_DIR="$RUNTIME_ROOT/reports"
export THESEUS_CHECKPOINTS_DIR="$RUNTIME_ROOT/checkpoints"
export CARGO_TARGET_DIR="$RUNTIME_ROOT/cargo-target"

if [ "$INSTALL_DEPS" = "1" ]; then
  log_step "Bootstrapping macOS dependencies. First install may take several minutes while Python packages download."
  if [ "$REQUIRE_MLX" = "1" ]; then
    "$PYTHON" scripts/macos_dependency_bootstrap.py \
      --venv "$INSTALL_ROOT/.venv-puffer" \
      --runtime-root "$RUNTIME_ROOT" \
      --install-missing \
      --require-mlx
  else
    "$PYTHON" scripts/macos_dependency_bootstrap.py \
      --venv "$INSTALL_ROOT/.venv-puffer" \
      --runtime-root "$RUNTIME_ROOT" \
      --install-missing
  fi
else
  log_step "Skipping dependency bootstrap by request."
fi

if [ -x "$INSTALL_ROOT/.venv-puffer/bin/python" ]; then
  PYTHON="$INSTALL_ROOT/.venv-puffer/bin/python"
fi

log_step "Initializing runtime paths."
"$PYTHON" scripts/runtime_paths.py init --runtime-root "$RUNTIME_ROOT" >/dev/null
if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
  log_step "Configuring safe soft auto-update client."
  "$PYTHON" scripts/update_manager.py configure \
    --mode auto_soft \
    --check-on-start \
    --auto-install-soft \
    --no-auto-install-hard \
    --out reports/update_client_configure_install.json >/dev/null
fi
if [ "$REGISTER_COMMUNITY" = "1" ] && [ ! -f "configs/theseus_registration.local.json" ]; then
  log_step "Registering local personal/homelab license profile."
  "$PYTHON" scripts/license_manager.py register \
    --usage personal_homelab \
    --accept-terms \
    --out reports/license_registration_macos_install.json >/dev/null
fi
if [ -n "$INVITE" ]; then
  log_step "Applying Hive invite bundle."
  "$PYTHON" scripts/hive_invite.py apply --invite "$INVITE" --write-local-config --out reports/hive_join_apply_last.json >/dev/null
fi
if [ -n "$RELAY_URL" ] || [ -n "$COORDINATOR_URL" ]; then
  log_step "Configuring Hive coordinator/relay URLs."
  "$PYTHON" scripts/hive_invite.py configure-local \
    --relay-url "$RELAY_URL" \
    --coordinator-url "$COORDINATOR_URL" \
    --out reports/hive_join_configure_last.json >/dev/null
fi
if [ "$PUBLIC_MODE" != "off" ] || [ -n "$PUBLIC_GATEWAY_URL" ]; then
  log_step "Configuring public Hive contribution mode."
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
if [ -f "scripts/hive_macos_canary.py" ]; then
  log_step "Assigning local Mac Hive roles."
  "$PYTHON" scripts/hive_macos_canary.py roles --write-local-config --out reports/macos_role_assignment.json >/dev/null || true
fi

write_hive_menu_fallback() {
  fallback_executable_path="$1"
  cat > "$fallback_executable_path" <<EOF_HIVE_FALLBACK
#!/usr/bin/env sh
export THESEUS_APP_SUPPORT_ROOT="$SUPPORT_ROOT"
export THESEUS_RUNTIME_ROOT="$RUNTIME_ROOT"
export THESEUS_DATA_DIR="$RUNTIME_ROOT/data"
export THESEUS_CACHE_DIR="$CACHE_ROOT"
export THESEUS_REPORTS_DIR="$RUNTIME_ROOT/reports"
export THESEUS_CHECKPOINTS_DIR="$RUNTIME_ROOT/checkpoints"
export CARGO_TARGET_DIR="$RUNTIME_ROOT/cargo-target"
cd "$INSTALL_ROOT"
exec "$INSTALL_ROOT/scripts/start_theseus_hive.sh"
EOF_HIVE_FALLBACK
}

binary_supports_current_arch() {
  binary_path="$1"
  current_arch="$(uname -m)"
  [ -f "$binary_path" ] || return 1
  if head -c 2 "$binary_path" 2>/dev/null | grep -q '^#!'; then
    return 0
  fi
  if command -v lipo >/dev/null 2>&1; then
    binary_arches="$(lipo -archs "$binary_path" 2>/dev/null || true)"
    for binary_arch in $binary_arches; do
      [ "$binary_arch" = "$current_arch" ] && return 0
    done
  fi
  if command -v file >/dev/null 2>&1; then
    binary_file_info="$(file "$binary_path" 2>/dev/null || true)"
    case "$binary_file_info" in
      *"$current_arch"*|*"universal binary"*|*"Mach-O universal"*) return 0 ;;
    esac
  fi
  return 1
}

compile_hive_menu_for_current_arch() {
  menu_compile_source="$1"
  menu_compile_output="$2"
  current_arch="$(uname -m)"
  [ -f "$menu_compile_source" ] || return 1
  command -v swiftc >/dev/null 2>&1 || return 1
  rm -f "$menu_compile_output"
  if swiftc -parse-as-library -O -target "$current_arch-apple-macosx$MACOS_MIN_VERSION" -o "$menu_compile_output" "$menu_compile_source" >/dev/null 2>&1; then
    return 0
  fi
  swiftc -parse-as-library -O -o "$menu_compile_output" "$menu_compile_source" >/dev/null 2>&1
}

create_app() {
  app_name="$1"
  executable_name="$2"
  command_body="$3"
  app_dir="$APPLICATIONS_DIR/$app_name.app"
  bundle_suffix="$(printf '%s' "$executable_name" | tr '[:upper:] ' '[:lower:]-' | tr -cd '[:alnum:]._-')"
  mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources"
  cat > "$app_dir/Contents/MacOS/$executable_name" <<EOF_APP
#!/usr/bin/env sh
export THESEUS_APP_SUPPORT_ROOT="$SUPPORT_ROOT"
export THESEUS_RUNTIME_ROOT="$RUNTIME_ROOT"
export THESEUS_DATA_DIR="$RUNTIME_ROOT/data"
export THESEUS_CACHE_DIR="$CACHE_ROOT"
export THESEUS_REPORTS_DIR="$RUNTIME_ROOT/reports"
export THESEUS_CHECKPOINTS_DIR="$RUNTIME_ROOT/checkpoints"
export CARGO_TARGET_DIR="$RUNTIME_ROOT/cargo-target"
cd "$INSTALL_ROOT"
$command_body
EOF_APP
  chmod +x "$app_dir/Contents/MacOS/$executable_name"
  cat > "$app_dir/Contents/Info.plist" <<EOF_PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$executable_name</string>
  <key>CFBundleIdentifier</key>
  <string>local.project-theseus.$bundle_suffix</string>
  <key>CFBundleName</key>
  <string>$app_name</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MACOS_MIN_VERSION</string>
</dict>
</plist>
EOF_PLIST
}

create_hive_menu_bar_app() {
  app_name="Project Theseus Hive"
  executable_name="Project Theseus Hive"
  app_dir="$APPLICATIONS_DIR/$app_name.app"
  executable_path="$app_dir/Contents/MacOS/$executable_name"
  menu_source="$INSTALL_ROOT/packaging/macos/TheseusHiveMenuBar.swift"
  prebuilt_binary="$INSTALL_ROOT/packaging/macos/TheseusHiveMenuBar"
  mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources"
  if [ -x "$prebuilt_binary" ] && binary_supports_current_arch "$prebuilt_binary"; then
    cp "$prebuilt_binary" "$executable_path"
  elif compile_hive_menu_for_current_arch "$menu_source" "$executable_path"; then
    :
  else
    write_hive_menu_fallback "$executable_path"
  fi
  chmod +x "$executable_path"
  cat > "$app_dir/Contents/Info.plist" <<EOF_HIVE_PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$executable_name</string>
  <key>CFBundleIdentifier</key>
  <string>local.project-theseus.hive-menubar</string>
  <key>CFBundleName</key>
  <string>$app_name</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MACOS_MIN_VERSION</string>
  <key>LSUIElement</key>
  <true/>
</dict>
</plist>
EOF_HIVE_PLIST
  if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "$app_dir" >/dev/null 2>&1 || true
  fi
}

log_step "Creating Project Theseus app launchers."
create_hive_menu_bar_app
create_app "Project Theseus Setup" "Project Theseus Setup" "exec \"$PYTHON\" \"$INSTALL_ROOT/scripts/theseus_setup_wizard.py\" --open"
create_app "Project Theseus Doctor" "Project Theseus Doctor" "exec \"$PYTHON\" \"$INSTALL_ROOT/scripts/macos_dependency_bootstrap.py\" --venv \"$INSTALL_ROOT/.venv-puffer\" --runtime-root \"$RUNTIME_ROOT\""

if [ "$INSTALL_SERVICE" = "1" ]; then
  log_step "Writing Hive LaunchAgent service files."
  LAUNCH_DIR="$HOME/Library/LaunchAgents"
  mkdir -p "$LAUNCH_DIR"
  cat > "$LAUNCH_DIR/local.project-theseus.hive.plist" <<EOF_LAUNCH
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.project-theseus.hive</string>
  <key>ProgramArguments</key>
  <array>
    <string>$INSTALL_ROOT/scripts/start_theseus_hive.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$INSTALL_ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>THESEUS_FOREGROUND</key>
    <string>1</string>
    <key>THESEUS_APP_SUPPORT_ROOT</key>
    <string>$SUPPORT_ROOT</string>
    <key>THESEUS_RUNTIME_ROOT</key>
    <string>$RUNTIME_ROOT</string>
    <key>THESEUS_DATA_DIR</key>
    <string>$RUNTIME_ROOT/data</string>
    <key>THESEUS_CACHE_DIR</key>
    <string>$CACHE_ROOT</string>
    <key>THESEUS_REPORTS_DIR</key>
    <string>$RUNTIME_ROOT/reports</string>
    <key>THESEUS_CHECKPOINTS_DIR</key>
    <string>$RUNTIME_ROOT/checkpoints</string>
    <key>CARGO_TARGET_DIR</key>
    <string>$RUNTIME_ROOT/cargo-target</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$RUNTIME_ROOT/logs/project-theseus-hive-launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>$RUNTIME_ROOT/logs/project-theseus-hive-launchd.err.log</string>
</dict>
</plist>
EOF_LAUNCH
  if [ "$ENABLE_SERVICE" = "1" ]; then
    log_step "Loading Hive node LaunchAgent."
    launchctl unload "$LAUNCH_DIR/local.project-theseus.hive.plist" >/dev/null 2>&1 || true
    launchctl load "$LAUNCH_DIR/local.project-theseus.hive.plist" >/dev/null 2>&1 || true
  fi
  cat > "$LAUNCH_DIR/local.project-theseus.hive-menubar.plist" <<EOF_MENUBAR
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.project-theseus.hive-menubar</string>
  <key>ProgramArguments</key>
  <array>
    <string>$APPLICATIONS_DIR/Project Theseus Hive.app/Contents/MacOS/Project Theseus Hive</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$RUNTIME_ROOT/logs/project-theseus-menubar.out.log</string>
  <key>StandardErrorPath</key>
  <string>$RUNTIME_ROOT/logs/project-theseus-menubar.err.log</string>
</dict>
</plist>
EOF_MENUBAR
  if [ "$ENABLE_SERVICE" = "1" ]; then
    log_step "Loading menu bar LaunchAgent."
    launchctl unload "$LAUNCH_DIR/local.project-theseus.hive-menubar.plist" >/dev/null 2>&1 || true
    launchctl load "$LAUNCH_DIR/local.project-theseus.hive-menubar.plist" >/dev/null 2>&1 || true
  fi
fi

if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
  log_step "Writing soft-update LaunchAgent."
  LAUNCH_DIR="$HOME/Library/LaunchAgents"
  mkdir -p "$LAUNCH_DIR"
  cat > "$LAUNCH_DIR/local.project-theseus.update.plist" <<EOF_UPDATE
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.project-theseus.update</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>$INSTALL_ROOT/scripts/update_manager.py</string>
    <string>check</string>
    <string>--apply</string>
    <string>--respect-interval</string>
    <string>--out</string>
    <string>reports/update_checkin.json</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$INSTALL_ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>THESEUS_APP_SUPPORT_ROOT</key>
    <string>$SUPPORT_ROOT</string>
    <key>THESEUS_RUNTIME_ROOT</key>
    <string>$RUNTIME_ROOT</string>
    <key>THESEUS_DATA_DIR</key>
    <string>$RUNTIME_ROOT/data</string>
    <key>THESEUS_CACHE_DIR</key>
    <string>$CACHE_ROOT</string>
    <key>THESEUS_REPORTS_DIR</key>
    <string>$RUNTIME_ROOT/reports</string>
    <key>THESEUS_CHECKPOINTS_DIR</key>
    <string>$RUNTIME_ROOT/checkpoints</string>
    <key>CARGO_TARGET_DIR</key>
    <string>$RUNTIME_ROOT/cargo-target</string>
  </dict>
  <key>StartInterval</key>
  <integer>1800</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$RUNTIME_ROOT/logs/theseus-update.out.log</string>
  <key>StandardErrorPath</key>
  <string>$RUNTIME_ROOT/logs/theseus-update.err.log</string>
</dict>
</plist>
EOF_UPDATE
  if [ "$ENABLE_SERVICE" = "1" ]; then
    log_step "Loading soft-update LaunchAgent."
    launchctl unload "$LAUNCH_DIR/local.project-theseus.update.plist" >/dev/null 2>&1 || true
    launchctl load "$LAUNCH_DIR/local.project-theseus.update.plist" || true
  fi
fi

if [ "$VACATION_MODE_SERVICE" = "1" ]; then
  log_step "Writing vacation-mode LaunchAgent."
  LAUNCH_DIR="$HOME/Library/LaunchAgents"
  mkdir -p "$LAUNCH_DIR"
  cat > "$LAUNCH_DIR/local.project-theseus.vacation.plist" <<EOF_VACATION
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.project-theseus.vacation</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>$INSTALL_ROOT/scripts/vacation_mode_supervisor.py</string>
    <string>--cycles</string>
    <string>1</string>
    <string>--execute</string>
    <string>--start-services</string>
    <string>--out</string>
    <string>reports/vacation_mode_supervisor.json</string>
    <string>--markdown-out</string>
    <string>reports/vacation_mode_supervisor.md</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$INSTALL_ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>THESEUS_APP_SUPPORT_ROOT</key>
    <string>$SUPPORT_ROOT</string>
    <key>THESEUS_RUNTIME_ROOT</key>
    <string>$RUNTIME_ROOT</string>
    <key>THESEUS_DATA_DIR</key>
    <string>$RUNTIME_ROOT/data</string>
    <key>THESEUS_CACHE_DIR</key>
    <string>$CACHE_ROOT</string>
    <key>THESEUS_REPORTS_DIR</key>
    <string>$RUNTIME_ROOT/reports</string>
    <key>THESEUS_CHECKPOINTS_DIR</key>
    <string>$RUNTIME_ROOT/checkpoints</string>
    <key>CARGO_TARGET_DIR</key>
    <string>$RUNTIME_ROOT/cargo-target</string>
  </dict>
  <key>StartInterval</key>
  <integer>1800</integer>
  <key>StandardOutPath</key>
  <string>$RUNTIME_ROOT/logs/project-theseus-vacation.out.log</string>
  <key>StandardErrorPath</key>
  <string>$RUNTIME_ROOT/logs/project-theseus-vacation.err.log</string>
</dict>
</plist>
EOF_VACATION
  if [ "$ENABLE_SERVICE" = "1" ]; then
    log_step "Loading vacation-mode LaunchAgent."
    launchctl unload "$LAUNCH_DIR/local.project-theseus.vacation.plist" >/dev/null 2>&1 || true
    launchctl load "$LAUNCH_DIR/local.project-theseus.vacation.plist" >/dev/null 2>&1 || true
  fi
fi

if command -v xattr >/dev/null 2>&1; then
  log_step "Clearing quarantine attributes from installed apps."
  xattr -dr com.apple.quarantine "$INSTALL_ROOT" >/dev/null 2>&1 || true
  xattr -dr com.apple.quarantine "$APPLICATIONS_DIR/Project Theseus Hive.app" >/dev/null 2>&1 || true
  xattr -dr com.apple.quarantine "$APPLICATIONS_DIR/Project Theseus Setup.app" >/dev/null 2>&1 || true
  xattr -dr com.apple.quarantine "$APPLICATIONS_DIR/Project Theseus Doctor.app" >/dev/null 2>&1 || true
fi

log_step "Probing Hive node capabilities."
"$PYTHON" scripts/hive_node.py probe --out reports/hive_status.json >/dev/null
log_step "Running one scheduler/artifact sync pass."
"$PYTHON" scripts/hive_scheduler.py --out reports/hive_scheduler.json >/dev/null || true
log_step "Refreshing installed Hive version status."
"$PYTHON" scripts/hive_version_manager.py status --out reports/hive_version_status.json >/dev/null || true

if [ "$START_NOW" = "1" ]; then
  log_step "Starting Project Theseus Hive services."
  "$INSTALL_ROOT/scripts/start_theseus_hive.sh"
fi

log_step "Install complete."
printf '%s\n' "Project Theseus payload installed at: $INSTALL_ROOT"
printf '%s\n' "Project Theseus runtime root: $RUNTIME_ROOT"
printf '%s\n' "Project Theseus apps installed in: $APPLICATIONS_DIR"
printf '%s\n' "Hive app: $APPLICATIONS_DIR/Project Theseus Hive.app"
printf '%s\n' "Setup app: $APPLICATIONS_DIR/Project Theseus Setup.app"
printf '%s\n' "Doctor app: $APPLICATIONS_DIR/Project Theseus Doctor.app"
if [ "$INSTALL_SERVICE" = "1" ]; then
  printf '%s\n' "LaunchAgent written: $HOME/Library/LaunchAgents/local.project-theseus.hive.plist"
  printf '%s\n' "Menu bar LaunchAgent written: $HOME/Library/LaunchAgents/local.project-theseus.hive-menubar.plist"
fi
if [ "$AUTO_UPDATE_SOFT" = "1" ]; then
  printf '%s\n' "Update LaunchAgent written: $HOME/Library/LaunchAgents/local.project-theseus.update.plist"
fi
if [ "$VACATION_MODE_SERVICE" = "1" ]; then
  printf '%s\n' "Vacation LaunchAgent written: $HOME/Library/LaunchAgents/local.project-theseus.vacation.plist"
fi
