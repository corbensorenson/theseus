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
protocol_path = source_root / "prompt_toolkit" / "contrib" / "telnet" / "protocol.py"
for name, path in (
    ("prompt_toolkit", source_root / "prompt_toolkit"),
    ("prompt_toolkit.contrib", source_root / "prompt_toolkit" / "contrib"),
    ("prompt_toolkit.contrib.telnet", source_root / "prompt_toolkit" / "contrib" / "telnet"),
):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = package
log_module = types.ModuleType("prompt_toolkit.contrib.telnet.log")
log_module.logger = __import__("logging").getLogger("sealed-prompt-toolkit")
sys.modules[log_module.__name__] = log_module
spec = importlib.util.spec_from_file_location(
    "prompt_toolkit.contrib.telnet.protocol", protocol_path
)
assert spec is not None and spec.loader is not None
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)
IS = protocol.IS
TelnetProtocol = protocol.TelnetProtocolParser


seen: list[str] = []


class Receiver:
    ttype_received_callback = staticmethod(seen.append)


TelnetProtocol.ttype(Receiver(), IS + b"x\xffterm")  # type: ignore[arg-type]
assert seen == ["x\ufffdterm"], "invalid ASCII terminal type was not replaced"
