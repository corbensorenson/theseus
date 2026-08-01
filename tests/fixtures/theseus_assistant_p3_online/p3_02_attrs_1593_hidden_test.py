from pathlib import Path
import os
import shutil
import sys
import typing


if sys.version_info < (3, 10):
    python = shutil.which("python3.12")
    if python is None:
        raise RuntimeError("Python 3.10+ is required for the sealed evaluator")
    os.execv(python, [python, *sys.argv])


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from attr._make import _is_class_var  # noqa: E402


assert _is_class_var(typing.ForwardRef("ClassVar[str]")), (
    "ForwardRef ClassVar was not recognized"
)
assert _is_class_var(typing.ForwardRef("typing.ClassVar[int]")), (
    "qualified ForwardRef ClassVar was not recognized"
)
assert not _is_class_var(typing.ForwardRef("list[str]")), (
    "non-ClassVar ForwardRef was misclassified"
)
