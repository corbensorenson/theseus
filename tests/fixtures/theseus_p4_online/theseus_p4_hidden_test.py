#!/usr/bin/env python3
"""Independent behavioral checks for the sealed Theseus P4 task pool."""

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
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote


class AnnotationStripper(ast.NodeTransformer):
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


def extracted_function(path: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name}, found {len(matches)}")
    node = AnnotationStripper().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(node)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def extracted_method(
    path: str, class_name: str, name: str, namespace: dict[str, Any]
) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise AssertionError(f"expected one {class_name}, found {len(classes)}")
    matches = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {class_name}.{name}, found {len(matches)}"
        )
    node = AnnotationStripper().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(node)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def import_local(module: str, src: str = "src") -> Any:
    sys.path.insert(0, str(Path(src).resolve()))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


def check_pydantic() -> None:
    eq = extracted_method(
        "pydantic/types.py", "SecretStr", "__eq__", {"Any": Any, "secrets": secrets}
    )

    class Secret:
        def __init__(self, value: str) -> None:
            self.value = value

        def get_secret_value(self) -> str:
            return self.value

        __eq__ = eq

    assert Secret("café") == Secret("café")
    assert Secret("café") != Secret("cafe")
    assert Secret("\ud800") == Secret("\ud800")
    assert Secret("ascii") == Secret("ascii")
    assert (Secret("ascii") == "ascii") is False


def check_pytest() -> None:
    def strtobool(value: str) -> bool:
        lowered = value.lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
        raise ValueError(value)

    function = extracted_function(
        "src/_pytest/logging.py", "_get_auto_indent", {"_strtobool": strtobool}
    )
    assert function(-5) == 0
    assert function("-5") == 0
    assert function(7) == 7
    assert function(True) == -1
    assert function("off") == 0


def check_fastapi() -> None:
    path = "fastapi/sse.py"
    namespace: dict[str, Any] = {"Annotated": Any, "Doc": lambda value: value}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    wanted = [
        AnnotationStripper().visit(copy.deepcopy(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_split_sse_lines", "format_sse_event"}
    ]
    ast.fix_missing_locations(module := ast.Module(body=wanted, type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    function = namespace["format_sse_event"]
    assert function(data_str="Hello\n") == b"data: Hello\ndata: \n\n"
    assert function(data_str="Hello\r\nWorld") == b"data: Hello\ndata: World\n\n"
    assert function(data_str="A\u2028B") == "data: A\u2028B\n\n".encode()
    assert function(data_str="") == b"data: \n\n"
    assert function(comment="hi\n") == b": hi\n: \n\n"


def check_yarl() -> None:
    os.environ["YARL_NO_EXTENSIONS"] = "1"
    module = import_local("yarl", ".")
    URL = module.URL
    for value in (
        "http://127.0.0.1/internal-secret",
        "javascript:alert(1)",
        "data:text/html,xss",
    ):
        built = URL.build(path=value)
        reparsed = URL(str(built))
        assert built.scheme == ""
        assert built.raw_host is None
        assert reparsed.scheme == ""
        assert reparsed.raw_host is None
    assert str(URL("http://example.com").joinpath("a:b")) == "http://example.com/a:b"
    assert str(URL.build(path="/path/with:colon")) == "/path/with:colon"


def check_packaging() -> None:
    tags = import_local("packaging.tags")
    for value in ("2-none-any", "2.7.6-none-any", "py3.2-none-any", "py+3-none-any"):
        try:
            tags.parse_tag(value)
        except tags.InvalidTag:
            pass
        else:
            raise AssertionError(f"invalid interpreter accepted: {value}")
    for interpreter in ("sillywalk", "graalpy311", "_custom"):
        parsed = tags.parse_tag(f"{interpreter}-none-any")
        assert parsed == {tags.Tag(interpreter, "none", "any")}


def check_structlog() -> None:
    output = import_local("structlog._output")
    for file in (sys.stdout, sys.stderr):
        restored = pickle.loads(pickle.dumps(output.WriteLogger(file)))
        assert restored._file is file
        assert restored._write == file.write
        assert restored._flush == file.flush
        restored.msg("P4_STRUCTLOG_PICKLE_OK")


def check_trio() -> None:
    ignored = object()
    trio_stub = SimpleNamespace(TASK_STATUS_IGNORED=ignored)
    function = extracted_function(
        "src/trio/_subprocess.py",
        "_run_process",
        {"subprocess": subprocess, "trio": trio_stub},
    )

    async def invoke() -> None:
        try:
            await function(["unused"], stdin=subprocess.PIPE)
        except ValueError as exc:
            message = str(exc)
            assert message.startswith("stdin=subprocess.PIPE")
            assert "stdout=subprocess.PIPE" not in message
        else:
            raise AssertionError("stdin PIPE was accepted")

    asyncio.run(invoke())


def check_tox() -> None:
    function = extracted_function("src/tox/config/set_env.py", "_split_value_marker", {})
    assert function("can't; sys_platform == 'darwin'") == (
        "can't",
        "sys_platform == 'darwin'",
    )
    assert function("'a;b'; sys_platform == 'darwin'") == (
        "'a;b'",
        "sys_platform == 'darwin'",
    )
    assert function(r"a\;b; python_version > '3'") == (
        "a;b",
        "python_version > '3'",
    )


def check_uvicorn() -> None:
    function = extracted_function(
        "uvicorn/protocols/websockets/websockets_sansio_impl.py",
        "handle_connect",
        {"unquote": unquote},
    )

    class Headers:
        def raw_items(self) -> list[tuple[str, str]]:
            return [
                ("Sec-WebSocket-Protocol", "proto1, proto2"),
                ("Sec-WebSocket-Protocol", " proto3 "),
            ]

        def get_all(self, name: str) -> list[str]:
            assert name == "Sec-WebSocket-Protocol"
            return ["proto1, proto2", " proto3 "]

    class Task:
        def add_done_callback(self, callback: Any) -> None:
            self.callback = callback

    class Loop:
        def create_task(self, awaitable: Any) -> Task:
            if hasattr(awaitable, "close"):
                awaitable.close()
            return Task()

    class Connection:
        def accept(self, event: Any) -> Any:
            return SimpleNamespace(status_code=101)

    holder = SimpleNamespace(
        conn=Connection(), handshake_initiated=False, handshake_complete=False,
        close_sent=False, root_path="", asgi_version="3.0", scheme="ws",
        server=("localhost", 80), client=("client", 123), app_state={"x": 1},
        queue=SimpleNamespace(put_nowait=lambda value: None), loop=Loop(),
        run_asgi=lambda: _empty_coroutine(), on_task_complete=lambda task: None,
        tasks=set(),
    )
    event = SimpleNamespace(headers=Headers(), path="/chat?x=1")
    function(holder, event)
    assert holder.scope["subprotocols"] == ["proto1", "proto2", "proto3"]
    assert holder.scope["query_string"] == b"x=1"
    assert holder.scope["state"] == {"x": 1}


async def _empty_coroutine() -> None:
    return None


def check_installer() -> None:
    class InvalidWheelSource(Exception):
        pass

    function = extracted_function(
        "src/installer/_core.py",
        "_determine_scheme",
        {
            "posixpath": posixpath,
            "SCHEME_NAMES": {"purelib", "platlib", "scripts", "headers", "data"},
            "cast": lambda _kind, value: value,
            "InvalidWheelSource": InvalidWheelSource,
        },
    )
    source = SimpleNamespace(data_dir="demo-1.0.data")

    def alarm(_signum: int, _frame: Any) -> None:
        raise TimeoutError("common-prefix sibling caused nontermination")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1)
    try:
        assert function("demo-1.0.database/payload.py", source, "purelib") == (
            "purelib",
            "demo-1.0.database/payload.py",
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert function("demo-1.0.data/scripts/tool", source, "purelib") == (
        "scripts",
        "tool",
    )


CASES = {
    "pydantic": check_pydantic,
    "pytest": check_pytest,
    "fastapi": check_fastapi,
    "yarl": check_yarl,
    "packaging": check_packaging,
    "structlog": check_structlog,
    "trio": check_trio,
    "tox": check_tox,
    "uvicorn": check_uvicorn,
    "installer": check_installer,
}


def main() -> int:
    case = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        CASES[case]()
    except Exception as exc:
        traceback.print_exc()
        print(f"P4_FAIL_{case}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"P4_PASS_{case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
