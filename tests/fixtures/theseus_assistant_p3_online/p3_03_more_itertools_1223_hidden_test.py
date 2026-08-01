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

from more_itertools import chunked  # noqa: E402


try:
    list(chunked("ABCDE", -1))
except ValueError as exc:
    assert str(exc) == "n must be at least 0", "negative chunk size error is unclear"
else:
    raise AssertionError("negative chunk size was accepted")

assert list(chunked("ABCDE", 2)) == [["A", "B"], ["C", "D"], ["E"]]
