#!/usr/bin/env sh
set -eu

TARGET_DIR="${1:-$HOME/.local/bin}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="python3"
if [ -x "$ROOT/.venv-puffer/bin/python" ]; then
  PYTHON="$ROOT/.venv-puffer/bin/python"
fi

mkdir -p "$TARGET_DIR"
cat > "$TARGET_DIR/theseus" <<EOF
#!/usr/bin/env sh
cd "$ROOT" || exit 1
exec "$PYTHON" "$ROOT/scripts/theseus_cli.py" "\$@"
EOF
chmod +x "$TARGET_DIR/theseus"

"$PYTHON" "$ROOT/scripts/theseus_cli.py" install --target-dir "$TARGET_DIR"

echo
echo "Theseus CLI installed at $TARGET_DIR/theseus"
echo "Try: theseus status"
echo "If that command is not found, add $TARGET_DIR to your PATH."
