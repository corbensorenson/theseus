#!/usr/bin/env python3
"""Bounded, non-hidden validation feedback for Theseus P4 candidates."""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib
import os
import pickle
import posixpath
import secrets
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote


class Strip(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.decorator_list = []
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.decorator_list = []
        return node


def function(path: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"P4_VISIBLE_{name}_IDENTITY"
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def method(path: str, class_name: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    matches = [
        node for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"P4_VISIBLE_{class_name}_{name}_IDENTITY"
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def local(module: str, src: str = "src") -> Any:
    sys.path.insert(0, str(Path(src).resolve()))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


def pydantic() -> None:
    eq = method("pydantic/types.py", "SecretStr", "__eq__", {"Any": Any, "secrets": secrets})
    cls = type("Secret", (), {
        "__init__": lambda self, value: setattr(self, "value", value),
        "get_secret_value": lambda self: self.value,
        "__eq__": eq,
    })
    assert cls("café") == cls("café"), "P4_VISIBLE_pydantic_unicode"


def pytest_case() -> None:
    fn = function("src/_pytest/logging.py", "_get_auto_indent", {"_strtobool": lambda value: False})
    assert fn(-5) == 0, "P4_VISIBLE_pytest_negative_indent"


def fastapi() -> None:
    path = "fastapi/sse.py"
    namespace: dict[str, Any] = {"Annotated": Any, "Doc": lambda value: value}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [
        Strip().visit(copy.deepcopy(node)) for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_split_sse_lines", "format_sse_event"}
    ]
    ast.fix_missing_locations(module := ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    assert namespace["format_sse_event"](data_str="Hello\n") == b"data: Hello\ndata: \n\n", "P4_VISIBLE_fastapi_trailing_line"


def yarl() -> None:
    os.environ["YARL_NO_EXTENSIONS"] = "1"
    URL = local("yarl", ".").URL
    built = URL.build(path="javascript:alert(1)")
    assert URL(str(built)).scheme == "", "P4_VISIBLE_yarl_scheme_materialization"


def packaging() -> None:
    tags = local("packaging.tags")
    try:
        tags.parse_tag("2.7.6-none-any")
    except tags.InvalidTag:
        return
    raise AssertionError("P4_VISIBLE_packaging_invalid_interpreter")


def structlog() -> None:
    output = local("structlog._output")
    restored = pickle.loads(pickle.dumps(output.WriteLogger(sys.stdout)))
    assert hasattr(restored, "_write"), "P4_VISIBLE_structlog_write_binding"


def trio() -> None:
    ignored = object()
    fn = function(
        "src/trio/_subprocess.py", "_run_process",
        {"subprocess": subprocess, "trio": SimpleNamespace(TASK_STATUS_IGNORED=ignored)},
    )

    async def invoke() -> None:
        try:
            await fn(["unused"], stdin=subprocess.PIPE)
        except ValueError as exc:
            assert str(exc).startswith("stdin=subprocess.PIPE"), "P4_VISIBLE_trio_stdin_label"
        else:
            raise AssertionError("P4_VISIBLE_trio_pipe_accepted")

    asyncio.run(invoke())


def tox() -> None:
    fn = function("src/tox/config/set_env.py", "_split_value_marker", {})
    assert fn("can't; sys_platform == 'darwin'")[0] == "can't", "P4_VISIBLE_tox_apostrophe"


def uvicorn() -> None:
    fn = function(
        "uvicorn/protocols/websockets/websockets_sansio_impl.py", "handle_connect",
        {"unquote": unquote},
    )

    class Headers:
        def raw_items(self) -> list[tuple[str, str]]:
            return [("Sec-WebSocket-Protocol", "proto1, proto2")]

        def get_all(self, name: str) -> list[str]:
            return ["proto1, proto2"]

    class Task:
        def add_done_callback(self, callback: Any) -> None:
            pass

    async def empty() -> None:
        pass

    def create_task(awaitable: Any) -> Task:
        awaitable.close()
        return Task()

    holder = SimpleNamespace(
        conn=SimpleNamespace(accept=lambda event: SimpleNamespace(status_code=101)),
        root_path="", asgi_version="3.0", scheme="ws", server=None, client=None,
        app_state={}, queue=SimpleNamespace(put_nowait=lambda value: None),
        loop=SimpleNamespace(create_task=create_task), run_asgi=empty,
        on_task_complete=lambda task: None, tasks=set(),
    )
    fn(holder, SimpleNamespace(headers=Headers(), path="/"))
    assert holder.scope["subprotocols"] == ["proto1", "proto2"], "P4_VISIBLE_uvicorn_split"


def installer() -> None:
    class InvalidWheelSource(Exception):
        pass

    fn = function(
        "src/installer/_core.py", "_determine_scheme",
        {"posixpath": posixpath, "SCHEME_NAMES": {"purelib", "scripts"},
         "cast": lambda _kind, value: value, "InvalidWheelSource": InvalidWheelSource},
    )
    source = SimpleNamespace(data_dir="demo-1.0.data")

    def alarm(_signum: int, _frame: Any) -> None:
        raise AssertionError("P4_VISIBLE_installer_sibling_prefix")

    prior = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1)
    try:
        assert fn("demo-1.0.database/payload.py", source, "purelib")[0] == "purelib"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)


CASES = {
    "pydantic": pydantic, "pytest": pytest_case, "fastapi": fastapi,
    "yarl": yarl, "packaging": packaging, "structlog": structlog,
    "trio": trio, "tox": tox, "uvicorn": uvicorn, "installer": installer,
}


if __name__ == "__main__":
    selected = sys.argv[1]
    try:
        CASES[selected]()
    except Exception:
        print(f"P4_VISIBLE_{selected}_PRIMARY", file=sys.stderr)
        raise
