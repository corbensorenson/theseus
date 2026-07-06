#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

if [ "$(uname -s)" != "Darwin" ]; then
  printf '%s\n' "macOS packaging must run on macOS because codesign/notarytool/productbuild are Apple tools." >&2
  exit 2
fi

DIST_DIR="${DIST_DIR:-$ROOT/dist/macos}"
APP_NAME="${APP_NAME:-ProjectTheseusHive}"
APP_DIR="$DIST_DIR/$APP_NAME.app"
PAYLOAD_DIR="$APP_DIR/Contents/Resources/payload"
ZIP_PATH="$DIST_DIR/$APP_NAME.zip"
PKG_PATH="$DIST_DIR/$APP_NAME.pkg"
DMG_PATH="$DIST_DIR/$APP_NAME.dmg"
MLX_README="$DIST_DIR/README-MLX.txt"
COMPAT_README="$DIST_DIR/README-MACOS-COMPATIBILITY.txt"
ARTIFACT_REPORT="$DIST_DIR/hive-installer-artifacts.json"
SIGN_IDENTITY="${THESEUS_MACOS_CODESIGN_IDENTITY:--}"
PKG_IDENTITY="${THESEUS_MACOS_INSTALLER_IDENTITY:-}"
NOTARY_PROFILE="${THESEUS_NOTARYTOOL_PROFILE:-}"
BUILD_PKG="${THESEUS_MACOS_BUILD_PKG:-1}"
BUILD_DMG="${THESEUS_MACOS_BUILD_DMG:-1}"
MACOS_MIN_VERSION="${THESEUS_MACOS_MIN_VERSION:-10.15}"
MACOS_ARCH_TARGETS="${THESEUS_MACOS_ARCH_TARGETS:-x86_64,arm64}"
JOIN_BUNDLE="${THESEUS_MACOS_JOIN_BUNDLE:-}"

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources" "$PAYLOAD_DIR" "$DIST_DIR"

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
    "$ROOT/" "$PAYLOAD_DIR/"
else
  ditto "$ROOT" "$PAYLOAD_DIR"
  rm -rf "$PAYLOAD_DIR/.git" "$PAYLOAD_DIR/.attd_tmp" "$PAYLOAD_DIR"/.venv* "$PAYLOAD_DIR/dist" "$PAYLOAD_DIR/games" "$PAYLOAD_DIR/node_modules" "$PAYLOAD_DIR/target" "$PAYLOAD_DIR/reports" "$PAYLOAD_DIR/checkpoints" "$PAYLOAD_DIR/tmp" "$PAYLOAD_DIR/updates" "$PAYLOAD_DIR/vendor"
  rm -f "$PAYLOAD_DIR"/configs/*.local.json "$PAYLOAD_DIR"/configs/*.secret.json
  rm -rf "$PAYLOAD_DIR/data/.cache" "$PAYLOAD_DIR/data/external_benchmark_candidates" "$PAYLOAD_DIR/data/local_roms" "$PAYLOAD_DIR/data/public_benchmarks" "$PAYLOAD_DIR/data/rom_manifests" "$PAYLOAD_DIR/data/synthetic"
fi

if [ -n "$JOIN_BUNDLE" ]; then
  if [ ! -f "$JOIN_BUNDLE" ]; then
    printf '%s\n' "THESEUS_MACOS_JOIN_BUNDLE does not exist: $JOIN_BUNDLE" >&2
    exit 2
  fi
  JOIN_BUNDLE_ABS="$(CDPATH= cd -- "$(dirname -- "$JOIN_BUNDLE")" && pwd)/$(basename -- "$JOIN_BUNDLE")"
  DIST_JOIN_BUNDLE="$DIST_DIR/ProjectTheseusHive.join.json"
  cp "$JOIN_BUNDLE" "$PAYLOAD_DIR/ProjectTheseusHive.join.json"
  if [ "$JOIN_BUNDLE_ABS" != "$DIST_JOIN_BUNDLE" ]; then
    cp "$JOIN_BUNDLE" "$DIST_JOIN_BUNDLE"
  fi
fi

python3 - "$PAYLOAD_DIR" <<'PY' >/dev/null
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
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
python3 scripts/hive_version_manager.py build-manifest --out "$PAYLOAD_DIR/configs/hive_build_version.json" >/dev/null
if command -v swiftc >/dev/null 2>&1 && [ -f "$PAYLOAD_DIR/packaging/macos/TheseusHiveMenuBar.swift" ]; then
  MENU_SRC="$PAYLOAD_DIR/packaging/macos/TheseusHiveMenuBar.swift"
  MENU_BIN="$PAYLOAD_DIR/packaging/macos/TheseusHiveMenuBar"
  rm -f "$MENU_BIN" "$MENU_BIN.arm64" "$MENU_BIN.x86_64"
  BUILT_ARCHES=""
  for arch in x86_64 arm64; do
    case ",$MACOS_ARCH_TARGETS," in
      *",$arch,"*) ;;
      *) continue ;;
    esac
    if swiftc -parse-as-library -O -target "$arch-apple-macosx$MACOS_MIN_VERSION" -o "$MENU_BIN.$arch" "$MENU_SRC" >/dev/null 2>&1; then
      BUILT_ARCHES="$BUILT_ARCHES $arch"
    fi
  done
  if [ -n "$BUILT_ARCHES" ] && command -v lipo >/dev/null 2>&1; then
    set -- $BUILT_ARCHES
    if [ "$#" -gt 1 ]; then
      lipo -create -output "$MENU_BIN" "$MENU_BIN.x86_64" "$MENU_BIN.arm64" >/dev/null 2>&1 || rm -f "$MENU_BIN"
    fi
  fi
  if [ ! -f "$MENU_BIN" ] && [ -f "$MENU_BIN.$(uname -m)" ]; then
    cp "$MENU_BIN.$(uname -m)" "$MENU_BIN"
  fi
  if [ ! -f "$MENU_BIN" ]; then
    for arch in $BUILT_ARCHES; do
      cp "$MENU_BIN.$arch" "$MENU_BIN"
      break
    done
  fi
  [ ! -f "$MENU_BIN" ] || chmod +x "$MENU_BIN"
fi

cat > "$APP_DIR/Contents/MacOS/$APP_NAME" <<'EOF'
#!/usr/bin/env sh
set -eu
APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PAYLOAD="$APP_DIR/Resources/payload"
DEFAULT_ARGS="${THESEUS_MACOS_INSTALLER_ARGS:---auto-update-soft --install-service --enable-service}"
LOG_DIR="$HOME/Library/Logs/Project Theseus Hive"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install-$(date +%Y%m%d-%H%M%S).log"
STATUS_FILE="${TMPDIR:-/tmp}/project-theseus-hive-install-$$.status"

quote_shell() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

run_direct() {
  printf '%s\n' "Project Theseus Hive installer"
  printf '%s\n' "Log: $LOG_FILE"
  rm -f "$STATUS_FILE"
  (
    set +e
    # shellcheck disable=SC2086
    "$PAYLOAD/scripts/install_theseus_hive_macos.sh" --source "$PAYLOAD" $DEFAULT_ARGS
    printf '%s' "$?" > "$STATUS_FILE"
  ) 2>&1 | tee -a "$LOG_FILE"
  status="$(cat "$STATUS_FILE" 2>/dev/null || printf '1')"
  rm -f "$STATUS_FILE"
  printf '\n'
  if [ "$status" = "0" ]; then
    printf '%s\n' 'Checking local Hive API...'
    hive_ready=0
    i=0
    while [ "$i" -lt 30 ]; do
      if /usr/bin/curl -fsS --max-time 2 'http://127.0.0.1:8791/api/hive/status' >/dev/null 2>&1; then
        hive_ready=1
        break
      fi
      i=$((i + 1))
      sleep 1
    done
    if [ "$hive_ready" = "1" ]; then
      printf '%s\n' 'Done. Project Theseus Hive is installed and running.'
      INSTALL_ROOT="$HOME/Library/Application Support/Project Theseus Hive/app/current"
      PYTHON="$INSTALL_ROOT/.venv-puffer/bin/python"
      if [ ! -x "$PYTHON" ]; then
        PYTHON="python3"
      fi
      if [ -f "$INSTALL_ROOT/scripts/hive_macos_canary.py" ]; then
        printf '\n%s\n' 'Installed app status:'
        (cd "$INSTALL_ROOT" && "$PYTHON" scripts/hive_macos_canary.py app-status --text --out reports/macos_app_install_status.json) || true
      fi
    else
      printf '%s\n' 'Install finished, but the local Hive API did not answer within 30 seconds.'
      printf '%s\n' 'Open Project Theseus Doctor.app or check the LaunchAgent logs listed above.'
    fi
    printf '%s\n' 'Menu bar app: /Applications/Project Theseus Hive.app'
    printf '%s\n' 'Operator UI: http://127.0.0.1:8791/mobile'
  else
    printf '%s\n' "Install failed with exit code $status."
    printf '%s\n' "Check the log above: $LOG_FILE"
  fi
  exit "$status"
}

if [ -t 1 ] || [ "${THESEUS_MACOS_INSTALLER_NO_TERMINAL:-0}" = "1" ]; then
  run_direct
fi

RUNNER="${TMPDIR:-/tmp}/project-theseus-hive-install-$$.command"
PAYLOAD_Q="$(quote_shell "$PAYLOAD")"
LOG_FILE_Q="$(quote_shell "$LOG_FILE")"
STATUS_FILE_Q="$(quote_shell "$STATUS_FILE")"
DEFAULT_ARGS_Q="$(quote_shell "$DEFAULT_ARGS")"

cat > "$RUNNER" <<EOF_RUNNER
#!/usr/bin/env sh
set -u
PAYLOAD=$PAYLOAD_Q
LOG_FILE=$LOG_FILE_Q
STATUS_FILE=$STATUS_FILE_Q
DEFAULT_ARGS=$DEFAULT_ARGS_Q
rm -f "\$STATUS_FILE"
(
  clear 2>/dev/null || true
  printf '%s\n' 'Project Theseus Hive installer'
  printf '%s\n' 'This window shows live install progress. It is safe to leave it open until it says Done.'
  printf '%s\n' "Log: \$LOG_FILE"
  printf '%s\n\n' 'Installing Hive payload, LaunchAgent service, menu bar helper, and soft-update client...'
  # shellcheck disable=SC2086
  "\$PAYLOAD/scripts/install_theseus_hive_macos.sh" --source "\$PAYLOAD" \$DEFAULT_ARGS
  printf '%s' "\$?" > "\$STATUS_FILE"
) 2>&1 | tee -a "\$LOG_FILE"
status="\$(cat "\$STATUS_FILE" 2>/dev/null || printf '1')"
rm -f "\$STATUS_FILE"
printf '\n'
if [ "\$status" = "0" ]; then
  printf '%s\n' 'Checking local Hive API...'
  hive_ready=0
  i=0
  while [ "\$i" -lt 30 ]; do
    if /usr/bin/curl -fsS --max-time 2 'http://127.0.0.1:8791/api/hive/status' >/dev/null 2>&1; then
      hive_ready=1
      break
    fi
    i=\$((i + 1))
    sleep 1
  done
  if [ "\$hive_ready" = "1" ]; then
    printf '%s\n' 'Done. Project Theseus Hive is installed and running.'
    INSTALL_ROOT="\$HOME/Library/Application Support/Project Theseus Hive/app/current"
    PYTHON="\$INSTALL_ROOT/.venv-puffer/bin/python"
    if [ ! -x "\$PYTHON" ]; then
      PYTHON="python3"
    fi
    if [ -f "\$INSTALL_ROOT/scripts/hive_macos_canary.py" ]; then
      printf '\n%s\n' 'Installed app status:'
      (cd "\$INSTALL_ROOT" && "\$PYTHON" scripts/hive_macos_canary.py app-status --text --out reports/macos_app_install_status.json) || true
    fi
  else
    printf '%s\n' 'Install finished, but the local Hive API did not answer within 30 seconds.'
    printf '%s\n' 'Open Project Theseus Doctor.app or check the LaunchAgent logs listed above.'
  fi
  printf '%s\n' 'Menu bar app: /Applications/Project Theseus Hive.app'
  printf '%s\n' 'Operator UI: http://127.0.0.1:8791/mobile'
  open 'http://127.0.0.1:8791/mobile' >/dev/null 2>&1 || true
else
  printf '%s\n' "Install failed with exit code \$status."
  printf '%s\n' "Keep this window open and check the log above: \$LOG_FILE"
fi
printf '\n%s' 'Press Return to close this installer window... '
read -r _unused
exit "\$status"
EOF_RUNNER
chmod +x "$RUNNER"
osascript -e 'display notification "Opening installer progress in Terminal." with title "Project Theseus Hive"' >/dev/null 2>&1 || true
if open -a Terminal "$RUNNER" >/dev/null 2>&1; then
  exit 0
fi
if open "$RUNNER" >/dev/null 2>&1; then
  exit 0
fi
osascript -e 'display dialog "Project Theseus Hive could not open Terminal for installer progress. Run the app from Terminal or install the PKG from the DMG." buttons {"OK"} default button "OK"' >/dev/null 2>&1 || true
exit 1
EOF
chmod +x "$APP_DIR/Contents/MacOS/$APP_NAME"

cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>local.project-theseus.installer</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MACOS_MIN_VERSION</string>
</dict>
</plist>
EOF

if [ "$SIGN_IDENTITY" = "-" ]; then
  codesign --force --deep --sign - "$APP_DIR"
else
  codesign --force --deep --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_DIR"
fi
codesign --verify --deep --strict --verbose=2 "$APP_DIR"

rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP_DIR" "$ZIP_PATH"

if [ "$BUILD_PKG" = "1" ]; then
  rm -f "$PKG_PATH"
  if [ -n "$PKG_IDENTITY" ]; then
    productbuild --component "$APP_DIR" /Applications --sign "$PKG_IDENTITY" "$PKG_PATH"
  else
    productbuild --component "$APP_DIR" /Applications "$PKG_PATH"
  fi
fi

cat > "$MLX_README" <<'EOF'
Project Theseus Hive macOS MLX notes

Apple Silicon workers should run the installer normally first. The installer
creates .venv-puffer and runs scripts/macos_dependency_bootstrap.py, which
installs numpy and checks/installs MLX when dependency bootstrap is enabled.

Intel Macs are supported as Hive CPU/storage/operator nodes. They should not
use --require-mlx, and they will not advertise the apple_mlx capability.

Useful follow-up commands on the Mac:
  ./scripts/install_theseus_hive_macos.sh --require-mlx --auto-update-soft --install-service --enable-service
  python3 scripts/macos_dependency_bootstrap.py --venv ".venv-puffer" --install-missing --require-mlx
  python3 scripts/hive_node.py probe --out reports/hive_status.json

When the DMG/package is built on a Mac worker, the generated artifact manifest
can be pulled back by the Windows coordinator with:
  python scripts/hive_artifact_sync.py --peer-url http://MAC_NODE:8791 --limit 50
EOF

cat > "$COMPAT_README" <<EOF
Project Theseus Hive macOS compatibility

This macOS installer is built as an architecture-neutral shell app wrapper and
targets both Intel (x86_64) and Apple Silicon (arm64) Macs.

Compatibility contract:
  Minimum macOS declared by app bundles: $MACOS_MIN_VERSION
  Target architectures: $MACOS_ARCH_TARGETS
  Opening ProjectTheseusHive.app from the DMG installs the payload, creates
    the user LaunchAgents, enables safe soft auto-updates, and starts the Hive
    node through launchd. Override with THESEUS_MACOS_INSTALLER_ARGS only for
    diagnostics.
    A Finder double-click opens a Terminal progress window and writes a log to
    ~/Library/Logs/Project Theseus Hive/install-*.log so the app does not look
    stuck while dependency/bootstrap work runs.
  Intel Macs: supported for CPU Hive work, storage, remote control, voice
    following, relays, coordinator/operator UI, artifact sync, and checkpoint
    chat/report tasks.
  Apple Silicon Macs: supported for all of the above plus Apple MLX worker
    chunks when MLX is installed and the machine advertises mlx_apple.

Menu bar helper:
  The packager tries to build a universal Swift menu bar helper. If a universal
  helper cannot be built, the installer validates the helper architecture on
  the target Mac and recompiles locally when swiftc is available. If that is not
  possible, it installs a shell fallback that still starts/opens the Hive.

Python:
  python3 is required on the target Mac. If it is missing, install Xcode Command
  Line Tools or Homebrew Python, then rerun the installer.

MLX:
  MLX is optional and Apple Silicon-oriented. Intel Macs should join the Hive
  normally and let the scheduler route MLX work to M-series Macs.
EOF

if [ "$BUILD_DMG" = "1" ]; then
  rm -f "$DMG_PATH"
  hdiutil create -volname "Project Theseus Hive" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG_PATH"
fi

if [ -n "$NOTARY_PROFILE" ]; then
  xcrun notarytool submit "$ZIP_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP_DIR"
  if [ -f "$PKG_PATH" ]; then
    xcrun notarytool submit "$PKG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$PKG_PATH"
  fi
  if [ -f "$DMG_PATH" ]; then
    xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG_PATH" || true
  fi
fi

spctl -a -vv -t exec "$APP_DIR" || true
python3 - "$DIST_DIR" "$ARTIFACT_REPORT" "$MACOS_MIN_VERSION" "$MACOS_ARCH_TARGETS" "$PAYLOAD_DIR/packaging/macos/TheseusHiveMenuBar" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
min_macos = sys.argv[3]
arch_targets = [item.strip() for item in sys.argv[4].split(",") if item.strip()]
menu_binary = Path(sys.argv[5])
suffixes = (".dmg", ".pkg", ".zip", ".app", ".txt", ".json")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or not path.name.endswith(suffixes):
        continue
    stat = path.stat()
    rows.append({
        "path": str(path).replace("\\", "/"),
        "name": path.name,
        "size_bytes": stat.st_size,
        "sha256": digest(path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    })
menu_arches: list[str] = []
if menu_binary.exists():
    try:
        result = subprocess.run(["lipo", "-archs", str(menu_binary)], text=True, capture_output=True, timeout=5)
        if result.returncode == 0:
            menu_arches = [item.strip() for item in result.stdout.split() if item.strip()]
    except (OSError, subprocess.TimeoutExpired):
        menu_arches = []
payload = {
    "ok": True,
    "policy": "project_theseus_macos_installer_artifacts_v1",
    "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "platform": "macos",
    "artifact_count": len(rows),
    "artifacts": rows,
    "mlx_support": {
        "dependency_bootstrap": "scripts/macos_dependency_bootstrap.py",
        "notes": "README-MLX.txt",
    },
    "macos_compatibility": {
        "minimum_system_version": min_macos,
        "target_architectures": arch_targets,
        "installer_wrapper": "architecture_neutral_shell",
        "menu_bar_helper_architectures": menu_arches,
        "intel_support": "cpu_storage_remote_control_voice_relay_operator_artifact_sync",
        "apple_silicon_support": "intel_support_plus_optional_apple_mlx",
        "notes": "README-MACOS-COMPATIBILITY.txt",
    },
    "join_bundle": {
        "included": (root / "ProjectTheseusHive.join.json").exists(),
        "path": str(root / "ProjectTheseusHive.join.json") if (root / "ProjectTheseusHive.join.json").exists() else "",
        "install_behavior": "bundled profile is auto-applied by scripts/install_theseus_hive_macos.sh when present",
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mkdir -p "$ROOT/reports"
cp "$ARTIFACT_REPORT" "$ROOT/reports/hive_installer_artifacts_macos.json"
printf '%s\n' "macOS installer app: $APP_DIR"
printf '%s\n' "macOS installer zip: $ZIP_PATH"
if [ -f "$PKG_PATH" ]; then
  printf '%s\n' "macOS installer pkg: $PKG_PATH"
fi
if [ -f "$DMG_PATH" ]; then
  printf '%s\n' "macOS installer dmg: $DMG_PATH"
fi
printf '%s\n' "macOS installer artifact manifest: $ARTIFACT_REPORT"
