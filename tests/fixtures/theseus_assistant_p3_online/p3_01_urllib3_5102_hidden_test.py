from pathlib import Path
import os
import shutil
import sys


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("Python 3.10+ is required for the sealed evaluator")
    os.execv(python, [python, *sys.argv])


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# The GitHub source archive intentionally omits generated build metadata.
version_file = Path(__file__).resolve().parent / "src" / "urllib3" / "_version.py"
version_file.write_text('__version__ = "0+sealed-evaluator"\n', encoding="utf-8")

from urllib3 import PoolManager  # noqa: E402
from urllib3.connectionpool import HTTPSConnectionPool  # noqa: E402


fingerprint = "92:81:FE:85:F7:0C:26:60:EC:D6:B3:BF:93:CF:F9:71:CC:07:7D:0A"
manager = PoolManager(assert_hostname=True, assert_fingerprint=fingerprint)
pool = manager.connection_from_url("http://example.com/")
assert not isinstance(pool, HTTPSConnectionPool)
assert "assert_hostname" not in pool.conn_kw, "HTTP pool retained assert_hostname"
assert "assert_fingerprint" not in pool.conn_kw, "HTTP pool retained assert_fingerprint"
