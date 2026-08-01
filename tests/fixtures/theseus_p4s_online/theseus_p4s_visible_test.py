#!/usr/bin/env python3
"""Bounded, obligation-addressable visible feedback for fresh P4S tasks."""

from __future__ import annotations

import ast
import copy
import importlib
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any


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
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def method(path: str, owner: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner
    )
    node = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node = Strip().visit(copy.deepcopy(node))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def function_source(path: str, name: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(text, node) or ""


def method_source(path: str, owner: str, name: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    class_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == owner
    )
    node = next(
        item for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(text, node) or ""


def module_all(path: str) -> list[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in item.targets)
    )
    value = ast.literal_eval(node.value)
    assert isinstance(value, list)
    return value


def tenacity() -> None:
    source = method_source(
        "tenacity/retry.py", "retry_if_exception_cause_type", "__call__"
    )
    assert "id(exc) not in seen" in source and "seen.add(id(exc))" in source, (
        "P4S_VISIBLE_TENACITY_CAUSE_CYCLE"
    )


def apscheduler() -> None:
    validate = function("src/apscheduler/_validators.py", "valid_metadata", {})
    validate(None, None, {"nested": None})
    try:
        validate(None, None, {"nested": {1: "bad"}})
    except ValueError as exc:
        assert "1" in str(exc), "P4S_VISIBLE_APSCHEDULER_KEY_REPORT"
    else:
        raise AssertionError("P4S_VISIBLE_APSCHEDULER_KEY_REPORT")


def poetry() -> None:
    class Credential:
        def __init__(self, username: str = "", password: Any = None) -> None:
            self.username, self.password = username, password

    calls: list[tuple[Any, ...]] = []
    repository = SimpleNamespace(url="https://repo.example/simple")
    owner = SimpleNamespace(
        get_repository_config_for_url=lambda _url, _exact: repository,
        _get_credentials_for_repository=lambda repository: Credential("repo-user", None),
        _password_manager=SimpleNamespace(
            get_credential=lambda *args, **kwargs: calls.append(args) or Credential("extra", "pw")
        ),
    )
    get_credentials = method(
        "src/poetry/utils/authenticator.py",
        "Authenticator",
        "_get_credentials_for_url",
        {"urllib": urllib, "HTTPAuthCredential": Credential},
    )
    credential = get_credentials(owner, "https://repo.example/simple/pkg")
    assert credential.username == "repo-user" and calls == [], (
        "P4S_VISIBLE_POETRY_CREDENTIAL_LOOKUPS"
    )


def pip() -> None:
    source = function_source("src/pip/_internal/commands/list.py", "_build_package_finder")
    assert "format_control=options.format_control" in source, (
        "P4S_VISIBLE_PIP_BINARY_SELECTION"
    )
    assert "prefer_binary=options.prefer_binary" in source, (
        "P4S_VISIBLE_PIP_BINARY_SELECTION"
    )


def aiohttp() -> None:
    text = Path("aiohttp/web_request.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseRequest"
    )
    attrs = next(
        node for node in owner.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ATTRS" for target in node.targets)
    )
    strings = [node.value for node in ast.walk(attrs.value) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert strings.count("_transport_sockname") == 1, "P4S_VISIBLE_AIOHTTP_ATTRS"


def httpx() -> None:
    assert module_all("httpx/_auth.py").count("FunctionAuth") == 1, (
        "P4S_VISIBLE_HTTPX_AUTH_EXPORT"
    )
    assert module_all("httpx/__init__.py").count("FunctionAuth") == 1, (
        "P4S_VISIBLE_HTTPX_PACKAGE_EXPORT"
    )


def httpcore() -> None:
    sync = function_source("httpcore/_sync/connection_pool.py", "_assign_requests_to_connections")
    asynchronous = function_source(
        "httpcore/_async/connection_pool.py", "_assign_requests_to_connections"
    )
    assert "sum(connection.is_idle() for connection in self._connections)" in sync, (
        "P4S_VISIBLE_HTTPCORE_SYNC_IDLE_COUNT"
    )
    assert "sum(connection.is_idle() for connection in self._connections)" in asynchronous, (
        "P4S_VISIBLE_HTTPCORE_ASYNC_IDLE_COUNT"
    )


def isort() -> None:
    parsed = function_source("isort/parse.py", "file_contents")
    rendered = function_source("isort/output.py", "sorted_imports")
    assert '.replace("\\r\\n", "\\n").replace("\\r", "\\n").split("\\n")' in parsed, (
        "P4S_VISIBLE_ISORT_PARSE_FORM_FEED"
    )
    assert "overflow_form_feeds" in rendered and "trailing_blank_lines" in rendered, (
        "P4S_VISIBLE_ISORT_OUTPUT_FORM_FEED"
    )


def cachetools() -> None:
    source_root = str(Path("src").resolve())
    sys.path.insert(0, source_root)
    sys.modules.pop("cachetools", None)
    try:
        module = importlib.import_module("cachetools")
    finally:
        sys.path.remove(source_root)
    cache = module.Cache(maxsize=5, getsizeof=lambda value: value)
    cache["keep"] = 2
    cache["grow"] = 2
    cache["grow"] = 3
    assert "keep" in cache and cache.currsize == 5, (
        "P4S_VISIBLE_CACHETOOLS_REPLACE_SIZE"
    )


def werkzeug() -> None:
    text = Path("src/werkzeug/routing/map.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MapAdapter"
    )
    build = next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    candidates = [
        node for node in ast.walk(build)
        if isinstance(node, ast.If)
        and "force_external" in (ast.get_source_segment(text, node.test) or "")
        and "host_matching" in (ast.get_source_segment(text, node.test) or "")
    ]
    assert len(candidates) == 1
    expression = ast.Expression(body=copy.deepcopy(candidates[0].test))
    ast.fix_missing_locations(expression)
    owner = SimpleNamespace(
        map=SimpleNamespace(subdomain_matching=False, host_matching=False),
        subdomain="",
        server_name="example.test",
    )
    assert eval(compile(expression, "<p4s-map-condition>", "eval"), {}, {
        "self": owner, "force_external": False, "domain_part": "elsewhere", "host": "elsewhere"
    }), "P4S_VISIBLE_WERKZEUG_RELATIVE_BUILD"


CASES = {
    "tenacity": tenacity,
    "apscheduler": apscheduler,
    "poetry": poetry,
    "pip": pip,
    "aiohttp": aiohttp,
    "httpx": httpx,
    "httpcore": httpcore,
    "isort": isort,
    "cachetools": cachetools,
    "werkzeug": werkzeug,
}


if __name__ == "__main__":
    selected = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        CASES[selected]()
    except Exception as exc:
        message = str(exc)
        print(message if message.startswith("P4S_VISIBLE_") else f"P4S_VISIBLE_{selected.upper()}_PRIMARY", file=sys.stderr)
        raise
