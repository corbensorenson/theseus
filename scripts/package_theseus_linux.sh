#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

DIST_DIR="${DIST_DIR:-$ROOT/dist/linux}"
BUNDLE="$DIST_DIR/ProjectTheseusHive"
TAR_PATH="$DIST_DIR/ProjectTheseusHive.tar.gz"
REPORT="$DIST_DIR/hive-installer-artifacts.json"
FORCE="${THESEUS_LINUX_PACKAGE_FORCE:-0}"

if [ -d "$BUNDLE" ] && [ "$FORCE" != "1" ]; then
  printf '%s\n' "Output already exists: $BUNDLE. Set THESEUS_LINUX_PACKAGE_FORCE=1 to replace." >&2
  exit 2
fi

rm -rf "$BUNDLE" "$TAR_PATH"
mkdir -p "$DIST_DIR"

python3 scripts/hive_usb_writer.py write \
  --out "$BUNDLE" \
  --hive-mode public \
  --public-mode off \
  --no-zip \
  --force >/dev/null

cat > "$BUNDLE/install-linux-cli.sh" <<'EOF'
#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$ROOT/install-from-usb.sh" "$@"
EOF
chmod +x "$BUNDLE/install-linux-cli.sh"

tar -C "$DIST_DIR" -czf "$TAR_PATH" "ProjectTheseusHive"

python3 - "$DIST_DIR" "$REPORT" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
suffixes = (".tar.gz", ".sh", ".desktop", ".zip")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or not any(path.name.endswith(suffix) for suffix in suffixes):
        continue
    stat = path.stat()
    rows.append({
        "path": str(path).replace("\\", "/"),
        "name": path.name,
        "size_bytes": stat.st_size,
        "sha256": sha(path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    })

payload = {
    "ok": True,
    "policy": "project_theseus_linux_installer_artifacts_v1",
    "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "platform": "linux",
    "artifact_count": len(rows),
    "artifacts": rows,
    "cli_entrypoint": "install-linux-cli.sh",
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mkdir -p "$ROOT/reports"
cp "$REPORT" "$ROOT/reports/hive_installer_artifacts_linux.json"

printf '%s\n' "Linux installer bundle: $BUNDLE"
printf '%s\n' "Linux installer tar: $TAR_PATH"
printf '%s\n' "Linux installer artifact manifest: $REPORT"
