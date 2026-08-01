from pathlib import Path
import os
import shutil
from tempfile import NamedTemporaryFile
import sys


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("python3.12 evaluator runtime is required")
    os.execv(python, [python, __file__])

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from requests.models import RequestEncodingMixin  # noqa: E402


with NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
    handle.write("named temp file contents\n")
    handle.seek(0)
    body, content_type = RequestEncodingMixin._encode_files({"file": handle}, {})

assert content_type.startswith("multipart/form-data; boundary=")
assert b"named temp file contents\n" in body
