from pathlib import Path
import os
import shutil
import sys


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("Python 3.10+ is required for the sealed evaluator")
    os.execv(python, [python, *sys.argv])


sys.path.insert(0, str(Path(__file__).resolve().parent))

from tomlkit import api  # noqa: E402


document = api.document()
document["x"] = 1
try:
    document["x"].comment("first line\nsecond line")
except ValueError as exc:
    assert "line breaks" in str(exc)
else:
    raise AssertionError("multiline item comment was accepted")
assert document.as_string() == "x = 1\n"

array = api.array()
try:
    array.add_line(1, 2, 3, comment="first line\rsecond line")
except ValueError as exc:
    assert "line breaks" in str(exc)
else:
    raise AssertionError("multiline array comment was accepted")
assert array.as_string() == "[]"
