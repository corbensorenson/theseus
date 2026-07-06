#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CSL="${CSL:-/usr/share/texlive/texmf-dist/tex/latex/citation-style-language/styles/ieee.csl}"
OUT="$ROOT/build"
mkdir -p "$OUT"

command -v pandoc >/dev/null || { echo 'pandoc is required' >&2; exit 1; }
[[ -f "$CSL" ]] || { echo "IEEE CSL not found: $CSL" >&2; exit 1; }

pandoc "$ROOT/source/Virtual_Context_Memory_v1.0.md" \
  --from markdown+tex_math_dollars \
  --citeproc \
  --bibliography="$ROOT/source/references_v1.0.bib" \
  --csl="$CSL" \
  --reference-doc="$ROOT/source/reference.docx" \
  --resource-path="$ROOT" \
  --metadata link-citations=true \
  -o "$OUT/Virtual_Context_Memory_v1.0.docx"

if command -v libreoffice >/dev/null; then
  libreoffice --headless --convert-to pdf --outdir "$OUT" \
    "$OUT/Virtual_Context_Memory_v1.0.docx" >/dev/null
  echo "Built DOCX and PDF in $OUT"
else
  echo "Built DOCX in $OUT; LibreOffice not found, so PDF conversion was skipped."
fi
