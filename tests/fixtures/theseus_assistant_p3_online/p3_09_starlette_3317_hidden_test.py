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

from starlette.datastructures import URL  # noqa: E402


url = URL("/path?a=1")
with_port = url.replace(port=8080)
assert with_port == URL("//:8080/path?a=1")
assert with_port.port == 8080
assert url.replace(username="u") == URL("//u@/path?a=1")
