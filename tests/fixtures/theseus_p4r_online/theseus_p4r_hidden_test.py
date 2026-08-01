#!/usr/bin/env python3
"""Independent hidden behavior checks for the three new P4R tasks."""

from __future__ import annotations

import ast
import copy
import pickle as stdlib_pickle
import re
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4


class Strip(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.decorator_list = []
        return node


def function(path: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def method(path: str, class_name: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    matches = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def black() -> None:
    class FileData(tuple):
        def __new__(cls, *values: Any):
            return tuple.__new__(cls, values)

    with tempfile.TemporaryDirectory() as tmp:
        cache_file = Path(tmp) / "cache.pickle"
        cache_file.write_bytes(b"cache")

        def invoke(load: Any) -> Any:
            fake_pickle = SimpleNamespace(
                load=load,
                UnpicklingError=stdlib_pickle.UnpicklingError,
            )
            read = method(
                "src/black/cache.py", "Cache", "read",
                {"get_cache_file": lambda _mode: cache_file, "pickle": fake_pickle, "FileData": FileData},
            )

            class Cache:
                def __init__(self, mode: Any, path: Path, file_data: dict[str, Any] | None = None) -> None:
                    self.mode, self.cache_file = mode, path
                    self.file_data = file_data or {}

            Cache.read = classmethod(read)
            return Cache.read("mode")

        assert invoke(lambda _handle: (_ for _ in ()).throw(PermissionError("denied"))).file_data == {}
        valid = invoke(lambda _handle: {"a.py": (1.0, 2, "digest")})
        assert valid.file_data["a.py"] == (1.0, 2, "digest")
        assert invoke(lambda _handle: (_ for _ in ()).throw(EOFError())).file_data == {}


def django() -> None:
    path = "django/urls/utils.py"
    get_group = function(path, "_get_group_start_end", {})
    find_groups = function(path, "_find_groups", {"_get_group_start_end": get_group})
    replace = function(
        path,
        "replace_unnamed_groups",
        {"_find_groups": find_groups, "_UNNAMED_GROUP_MATCHER": re.compile(r"\(")},
    )
    assert replace(r"^(\w+)/b/(\w+)$") == r"^<var>/b/<var>$"
    assert replace(r"^a/(\w+)/b/(\d+)/c/(\w+)$") == r"^a/<var>/b/<var>/c/<var>$"
    assert replace(r"^<a>/b/(\w+)/(\d+)$") == r"^<a>/b/<var>/<var>$"
    assert replace(r"^(\w+)/\((\d+)\)$") == r"^<var>/\(<var>\)$"


def celery() -> None:
    def key_t(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode()
        raise TypeError(type(value).__name__)

    path = "celery/backends/base.py"
    get_key = method(path, "BaseKeyValueStoreBackend", "_get_key_for", {"UUID": UUID})
    get_task = method(path, "BaseKeyValueStoreBackend", "get_key_for_task", {})
    get_group = method(path, "BaseKeyValueStoreBackend", "get_key_for_group", {})
    get_chord = method(path, "BaseKeyValueStoreBackend", "get_key_for_chord", {})

    class Backend:
        task_keyprefix = b"task-"
        group_keyprefix = b"group-"
        chord_keyprefix = b"chord-"
        _get_key_for = get_key
        get_key_for_task = get_task
        get_key_for_group = get_group
        get_key_for_chord = get_chord

    Backend.key_t = staticmethod(key_t)

    backend = Backend()
    identifier = uuid4()
    assert backend.get_key_for_task(identifier) == backend.get_key_for_task(str(identifier))
    assert backend.get_key_for_group(identifier) == backend.get_key_for_group(str(identifier))
    assert backend.get_key_for_chord(identifier) == backend.get_key_for_chord(str(identifier))
    assert backend.get_key_for_task("abc", "tail") == b"task-abctail"
    assert backend.get_key_for_task(b"abc") == b"task-abc"


CASES = {"black": black, "django": django, "celery": celery}


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        CASES[case]()
    except Exception as exc:
        traceback.print_exc()
        print(f"P4R_FAIL_{case}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"P4R_PASS_{case}")
