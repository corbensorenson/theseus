#!/usr/bin/env bash
set -u

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
IOS_DIR="$ROOT_DIR/ios/TheseusHive"
PROJECT="$IOS_DIR/TheseusHive.xcodeproj"
REPORT_DIR="$ROOT_DIR/reports/ios"
LOG_DIR="$REPORT_DIR/logs"
TMP_DIR="$ROOT_DIR/tmp/ios-build"
DIST_DIR="$ROOT_DIR/dist/ios"
STATUS_TSV="$REPORT_DIR/build_steps.tsv"
REPORT_JSON="$REPORT_DIR/build_status.json"

MODE="validate"
CONFIGURATION="Release"
TEAM="${THESEUS_IOS_DEVELOPMENT_TEAM:-}"
APP_BUNDLE_ID="${THESEUS_IOS_APP_BUNDLE_ID:-com.corbensorenson.theseushive}"
WATCH_BUNDLE_ID="${THESEUS_WATCH_APP_BUNDLE_ID:-${APP_BUNDLE_ID}.watch}"
ARCHIVE_PATH="${THESEUS_IOS_ARCHIVE_PATH:-$DIST_DIR/TheseusHive.xcarchive}"
EXPORT_OPTIONS="${THESEUS_IOS_EXPORT_OPTIONS:-}"
EXPORT_PATH="${THESEUS_IOS_EXPORT_PATH:-$DIST_DIR/export}"
ALLOW_PROVISIONING_UPDATES=0

usage() {
  cat <<'EOF'
Usage: scripts/build_theseus_ios.sh [options]

Build and validate the native Theseus Hive iPhone + Apple Watch apps.

Modes:
  --validate              Unsigned local validation build and report (default).
  --archive               Create a signed .xcarchive for Xcode Organizer/TestFlight.
  --export                Export an existing/new archive using an ExportOptions.plist.

Options:
  --team TEAM_ID          Apple Developer Team ID for signing.
  --app-bundle-id ID      iPhone bundle id. Default: com.corbensorenson.theseushive
  --watch-bundle-id ID    Watch bundle id. Default: <app-bundle-id>.watch
  --archive-path PATH     Archive output path. Default: dist/ios/TheseusHive.xcarchive
  --export-options PATH   ExportOptions.plist for --export.
  --export-path PATH      Export output directory. Default: dist/ios/export
  --configuration NAME    Xcode configuration. Default: Release
  --allow-provisioning-updates
                          Let Xcode create/update signing profiles.
  -h, --help              Show this help.

Environment equivalents:
  THESEUS_IOS_DEVELOPMENT_TEAM, THESEUS_IOS_APP_BUNDLE_ID,
  THESEUS_WATCH_APP_BUNDLE_ID, THESEUS_IOS_ARCHIVE_PATH,
  THESEUS_IOS_EXPORT_OPTIONS, THESEUS_IOS_EXPORT_PATH
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --validate)
      MODE="validate"
      ;;
    --archive)
      MODE="archive"
      ;;
    --export)
      MODE="export"
      ;;
    --team)
      shift
      TEAM="${1:-}"
      ;;
    --app-bundle-id)
      shift
      APP_BUNDLE_ID="${1:-}"
      ;;
    --watch-bundle-id)
      shift
      WATCH_BUNDLE_ID="${1:-}"
      ;;
    --archive-path)
      shift
      ARCHIVE_PATH="${1:-}"
      ;;
    --export-options)
      shift
      EXPORT_OPTIONS="${1:-}"
      ;;
    --export-path)
      shift
      EXPORT_PATH="${1:-}"
      ;;
    --configuration)
      shift
      CONFIGURATION="${1:-Release}"
      ;;
    --allow-provisioning-updates)
      ALLOW_PROVISIONING_UPDATES=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$REPORT_DIR" "$LOG_DIR" "$TMP_DIR" "$DIST_DIR"
: > "$STATUS_TSV"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step() {
  step_name="$1"
  shift
  step_log="$LOG_DIR/${step_name}.log"
  started="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "Running $step_name"
  if "$@" >"$step_log" 2>&1; then
    rc=0
    state="passed"
    log "$step_name passed"
  else
    rc=$?
    state="failed"
    log "$step_name failed with exit code $rc"
    tail -60 "$step_log" >&2 || true
  fi
  ended="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%s\t%s\t%s\t%s\t%s\n' "$step_name" "$state" "$rc" "$started" "$ended" >> "$STATUS_TSV"
  return "$rc"
}

write_report() {
  export THESEUS_IOS_REPORT_JSON="$REPORT_JSON"
  export THESEUS_IOS_STATUS_TSV="$STATUS_TSV"
  export THESEUS_IOS_MODE="$MODE"
  export THESEUS_IOS_CONFIGURATION="$CONFIGURATION"
  export THESEUS_IOS_PROJECT="$PROJECT"
  export THESEUS_IOS_APP_BUNDLE_ID_EFFECTIVE="$APP_BUNDLE_ID"
  export THESEUS_WATCH_BUNDLE_ID_EFFECTIVE="$WATCH_BUNDLE_ID"
  export THESEUS_IOS_TEAM_EFFECTIVE="$TEAM"
  export THESEUS_IOS_ARCHIVE_PATH_EFFECTIVE="$ARCHIVE_PATH"
  export THESEUS_IOS_EXPORT_PATH_EFFECTIVE="$EXPORT_PATH"
  python3 - <<'PY'
import json
import os
from pathlib import Path

rows = []
status_path = Path(os.environ["THESEUS_IOS_STATUS_TSV"])
if status_path.exists():
    for line in status_path.read_text(encoding="utf-8").splitlines():
        name, state, rc, started, ended = line.split("\t")
        rows.append({
            "name": name,
            "state": state,
            "returncode": int(rc),
            "started_utc": started,
            "ended_utc": ended,
            "log": f"reports/ios/logs/{name}.log",
        })

report = {
    "ok": bool(rows) and all(row["state"] == "passed" for row in rows),
    "policy": "project_theseus_ios_build_status_v1",
    "mode": os.environ["THESEUS_IOS_MODE"],
    "configuration": os.environ["THESEUS_IOS_CONFIGURATION"],
    "project": os.environ["THESEUS_IOS_PROJECT"],
    "app_bundle_id": os.environ["THESEUS_IOS_APP_BUNDLE_ID_EFFECTIVE"],
    "watch_bundle_id": os.environ["THESEUS_WATCH_BUNDLE_ID_EFFECTIVE"],
    "development_team": os.environ["THESEUS_IOS_TEAM_EFFECTIVE"],
    "archive_path": os.environ["THESEUS_IOS_ARCHIVE_PATH_EFFECTIVE"],
    "export_path": os.environ["THESEUS_IOS_EXPORT_PATH_EFFECTIVE"],
    "steps": rows,
}
out = Path(os.environ["THESEUS_IOS_REPORT_JSON"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
}

common_settings=(
  "THESEUS_IOS_APP_BUNDLE_ID=$APP_BUNDLE_ID"
  "THESEUS_WATCH_APP_BUNDLE_ID=$WATCH_BUNDLE_ID"
)

signing_settings=("${common_settings[@]}")
if [ -n "$TEAM" ]; then
  signing_settings+=("DEVELOPMENT_TEAM=$TEAM")
fi

provisioning_flags=()
if [ "$ALLOW_PROVISIONING_UPDATES" -eq 1 ]; then
  provisioning_flags+=("-allowProvisioningUpdates")
fi

overall_rc=0

if [ "$MODE" = "validate" ]; then
  run_step "watch_simulator_build" \
    xcodebuild -project "$PROJECT" -scheme TheseusHiveWatch \
      -destination "generic/platform=watchOS Simulator" \
      -configuration Debug \
      -derivedDataPath "$TMP_DIR/DerivedData" \
      CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY= \
      "${common_settings[@]}" build || overall_rc=1

  run_step "iphone_watch_unsigned_build" \
    xcodebuild -project "$PROJECT" -target TheseusHive \
      -configuration "$CONFIGURATION" \
      SYMROOT="$TMP_DIR/products" OBJROOT="$TMP_DIR/intermediates" \
      CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY= \
      "${common_settings[@]}" build || overall_rc=1
elif [ "$MODE" = "archive" ] || [ "$MODE" = "export" ]; then
  if [ -z "$TEAM" ]; then
    echo "--team or THESEUS_IOS_DEVELOPMENT_TEAM is required for $MODE." >&2
    overall_rc=2
  else
    mkdir -p "$(dirname "$ARCHIVE_PATH")"
    run_step "signed_archive" \
      xcodebuild -project "$PROJECT" -scheme TheseusHive \
        -configuration "$CONFIGURATION" \
        -archivePath "$ARCHIVE_PATH" \
        "${provisioning_flags[@]}" \
        "${signing_settings[@]}" archive || overall_rc=1
  fi

  if [ "$MODE" = "export" ] && [ "$overall_rc" -eq 0 ]; then
    if [ -z "$EXPORT_OPTIONS" ] || [ ! -f "$EXPORT_OPTIONS" ]; then
      echo "--export-options PATH is required for --export." >&2
      overall_rc=2
    else
      mkdir -p "$EXPORT_PATH"
      run_step "export_archive" \
        xcodebuild -exportArchive \
          -archivePath "$ARCHIVE_PATH" \
          -exportPath "$EXPORT_PATH" \
          -exportOptionsPlist "$EXPORT_OPTIONS" \
          "${provisioning_flags[@]}" || overall_rc=1
    fi
  fi
else
  echo "Unsupported mode: $MODE" >&2
  overall_rc=2
fi

write_report || true
log "iOS build report: $REPORT_JSON"
exit "$overall_rc"
