from pathlib import Path
import importlib.util
import os
import shutil
import sys
import types


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("Python 3.10+ is required for the sealed evaluator")
    os.execv(python, [python, *sys.argv])


source_root = Path(__file__).resolve().parent / "src"
package = types.ModuleType("filelock")
package.__path__ = [str(source_root / "filelock")]  # type: ignore[attr-defined]
sys.modules["filelock"] = package
identity = types.ModuleType("filelock._identity")
identity.host_name = lambda: "sealed-host"
identity.process_start_token = lambda _pid: 0
sys.modules[identity.__name__] = identity
soft = types.ModuleType("filelock._soft")
soft.SoftFileLock = type("SoftFileLock", (), {})
soft._read_lock_file = lambda _path: (None, None)
sys.modules[soft.__name__] = soft
util = types.ModuleType("filelock._util")
util.write_all = lambda _fd, _value: None
sys.modules[util.__name__] = util
spec = importlib.util.spec_from_file_location(
    "filelock._marker", source_root / "filelock" / "_marker.py"
)
assert spec is not None and spec.loader is not None
marker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = marker
spec.loader.exec_module(marker)
parse_marker = marker.parse_marker


prefix = "filelock/2\npid=1\nhost=h\nmode=lease\ntoken=t\n"
assert parse_marker(prefix + "duration=nan\n") is None, (
    "NaN lease duration was accepted"
)
assert parse_marker(prefix + "duration=inf\n") is None, (
    "infinite lease duration was accepted"
)
assert parse_marker(prefix + "duration=1.5\n") is not None, (
    "finite positive lease duration was rejected"
)
