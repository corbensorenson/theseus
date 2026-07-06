"""Patch mGBA's Python CFFI builder for local Windows/MSVC builds.

The upstream builder was written with Unix preprocessors in mind. On Windows,
MSVC can expand a large amount of CRT/Windows declarations into the CFFI cdef
stream. Those declarations are not part of the mGBA API and can make CFFI fail
before required types such as mColor and symbols such as mCoreFind are exposed.

This helper is intentionally narrow and idempotent. It patches only the staged
source copy in the resource pantry; it does not modify vendored project files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "THESEUS_MGBA_WINDOWS_CFFI_PATCH"


def patch_builder(source: Path) -> dict[str, object]:
    builder = source / "src" / "platform" / "python" / "_builder.py"
    if not builder.exists():
        return {
            "patched": False,
            "builder": str(builder),
            "reason": "mGBA Python _builder.py not found",
        }

    text = builder.read_text(encoding="utf-8")
    off_t_filter = '("typedef" in line and (" off_t" in line or " ssize_t" in line))'
    if PATCH_MARKER in text and "_invalid_parameter_noinfo" in text and off_t_filter in text:
        return {"patched": False, "builder": str(builder), "reason": "already patched"}

    original = text
    if PATCH_MARKER not in text:
        text = text.replace(
            'preprocessed = preprocess(os.path.join(pydir, "_builder.h"))',
            (
                f'preprocessed = preprocess(os.path.join(pydir, "_builder.h"))\n'
                f"# {PATCH_MARKER}: MSVC pulls CRT/Windows declarations into the cdef stream.\n"
                "if sys.platform == \"win32\":\n"
                "    for marker in (\"_invalid_parameter_noinfo\", \"typedef BYTE  BOOLEAN;\", \"typedef BYTE BOOLEAN;\"):\n"
                "        index = preprocessed.find(marker)\n"
                "        if index >= 0:\n"
                "            preprocessed = preprocessed[:index]\n"
                "            break"
            ),
        )
    text = text.replace(
        'for marker in ("typedef BYTE  BOOLEAN;", "typedef BYTE BOOLEAN;"):',
        'for marker in ("_invalid_parameter_noinfo", "typedef BYTE  BOOLEAN;", "typedef BYTE BOOLEAN;"):',
    )
    text = text.replace(
        'line.startswith("__inline")\n        or "__builtin" in line',
        (
            'line.startswith("__inline")\n'
            '        or line.startswith("__forceinline")\n'
            '        or line.startswith("__declspec")\n'
            '        or "__cdecl" in line\n'
            '        or "__stdcall" in line\n'
            '        or "__CRTDECL" in line\n'
            '        or "__builtin" in line'
        ),
    )
    text = text.replace(
        '        or "__cdecl" in line\n        or "__stdcall" in line',
        '        or line.startswith("__declspec")\n        or "__cdecl" in line\n        or "__stdcall" in line',
    )
    if off_t_filter not in text:
        text = text.replace(
            '        or "__builtin" in line',
            '        or "__builtin" in line\n        or ("typedef" in line and (" off_t" in line or " ssize_t" in line))',
        )
    text = text.replace(
        'line.replace("__int64", "long long").replace("__int32", "int").replace("__int16", "short").replace("__int8", "char")',
        (
            'line.replace("__int64", "long long").replace("__int32", "int").replace("__int16", "short").replace("__int8", "char")'
            '.replace("__ptr64", "").replace("__unaligned", "")'
        ),
    )
    text = text.replace(
        "ffi.cdef('\\n'.join(lines))",
        "ffi.cdef('\\n'.join(lines), override=sys.platform == \"win32\")",
    )

    if text == original:
        return {"patched": False, "builder": str(builder), "reason": "patch patterns not found"}

    builder.write_text(text, encoding="utf-8")
    return {"patched": True, "builder": str(builder), "reason": "patched"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to staged mGBA source checkout")
    args = parser.parse_args()
    result = patch_builder(Path(args.source))
    print(result)
    return 0 if result.get("patched") or result.get("reason") == "already patched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
