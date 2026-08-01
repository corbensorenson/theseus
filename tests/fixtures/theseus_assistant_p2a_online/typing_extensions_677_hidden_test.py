"""Evaluator-only assertions adapted from python/typing_extensions PR #677."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from typing_extensions import NotRequired, Required, TypedDict  # noqa: E402


class NotTotal(TypedDict, total=False):
    a: int


class Total(NotTotal):
    a: int


assert NotTotal.__required_keys__ == frozenset()
assert NotTotal.__optional_keys__ == frozenset({"a"})
assert Total.__required_keys__ == frozenset({"a"})
assert Total.__optional_keys__ == frozenset()


class Base(TypedDict):
    a: NotRequired[int]
    b: Required[int]


class Child(Base):
    a: Required[int]
    b: NotRequired[int]


assert Base.__required_keys__ == frozenset({"b"})
assert Base.__optional_keys__ == frozenset({"a"})
assert Child.__required_keys__ == frozenset({"a"})
assert Child.__optional_keys__ == frozenset({"b"})


class Base1(TypedDict):
    a: NotRequired[int]


class Base2(TypedDict):
    a: Required[str]


class MultipleChild(Base1, Base2):
    pass


assert MultipleChild.__annotations__ == {"a": Required[str]}
assert MultipleChild.__required_keys__ == frozenset({"a"})
assert MultipleChild.__optional_keys__ == frozenset()
