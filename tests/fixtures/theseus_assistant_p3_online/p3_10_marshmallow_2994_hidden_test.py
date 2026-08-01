from enum import Enum
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

from marshmallow import fields  # noqa: E402


class Outcome(Enum):
    all = float("inf")
    nothing = None


default_field = fields.Enum(Outcome, by_value=True)
assert default_field.deserialize(None) is None, (
    "None-valued enum did not default allow_none"
)

explicit_field = fields.Enum(Outcome, by_value=True, allow_none=False)
try:
    explicit_field.deserialize(None)
except Exception:
    pass
else:
    raise AssertionError("explicit allow_none=False was ignored")
