from pathlib import Path as LocalPath
import os
import shutil
import sys


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("Python 3.10+ is required for the sealed evaluator")
    os.execv(python, [python, *sys.argv])


sys.path.insert(0, str(LocalPath(__file__).resolve().parent / "src"))

from anyio import Path  # noqa: E402


try:
    Path("/xyz/foo.txt").with_stem("")
except ValueError as exc:
    assert "non-empty suffix" in str(exc)
else:
    raise AssertionError("empty stem with non-empty suffix was accepted")

assert Path("/xyz/foo.txt").with_stem("bar").name == "bar.txt"
