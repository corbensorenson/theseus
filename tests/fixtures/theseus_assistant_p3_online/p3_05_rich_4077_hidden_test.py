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

from rich.console import Console  # noqa: E402
from rich.file_proxy import FileProxy  # noqa: E402


class TTYFile:
    def isatty(self) -> bool:
        return True


proxy = FileProxy(Console(), TTYFile())  # type: ignore[arg-type]
assert proxy.isatty() is True, "FileProxy did not forward isatty"
