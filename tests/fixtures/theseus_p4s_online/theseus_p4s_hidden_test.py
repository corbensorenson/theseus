#!/usr/bin/env python3
"""Independent hidden behavior checks for the ten fresh P4S maintenance tasks."""

from __future__ import annotations

import ast
import copy
import importlib
import sys
import traceback
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
    path = "tenacity/retry.py"
    source = method_source(path, "retry_if_exception_cause_type", "__call__")
    assert "seen: set[int] = set()" in source
    assert "while exc is not None and id(exc) not in seen" in source
    assert "seen.add(id(exc))" in source
    call = method(path, "retry_if_exception_cause_type", "__call__", {})
    inner = KeyError("match")
    outer = RuntimeError("outer")
    outer.__cause__ = inner
    state = SimpleNamespace(outcome=SimpleNamespace(failed=True, exception=lambda: outer))
    owner = SimpleNamespace(exception_cause_types=(KeyError,))
    assert call(owner, state) is True
    assert call(owner, SimpleNamespace(outcome=SimpleNamespace(failed=False))) is False


def apscheduler() -> None:
    validate = function("src/apscheduler/_validators.py", "valid_metadata", {})
    validate(None, None, {"a": [None, {"b": None}], "ok": 3})
    try:
        validate(None, None, {"nested": {7: "bad"}})
    except ValueError as exc:
        assert "7" in str(exc) and "non-string key" in str(exc)
    else:
        raise AssertionError("non-string key accepted")
    try:
        validate(None, None, {"bad": object()})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid scalar accepted")


def poetry() -> None:
    class Credential:
        def __init__(self, username: str = "", password: Any = None) -> None:
            self.username, self.password = username, password

    get_credentials = method(
        "src/poetry/utils/authenticator.py",
        "Authenticator",
        "_get_credentials_for_url",
        {"urllib": urllib, "HTTPAuthCredential": Credential},
    )
    repository = SimpleNamespace(url="https://repo.example/simple")
    matched_calls: list[tuple[Any, ...]] = []
    matched = SimpleNamespace(
        get_repository_config_for_url=lambda _url, _exact: repository,
        _get_credentials_for_repository=lambda repository: Credential("repo-user", None),
        _password_manager=SimpleNamespace(
            get_credential=lambda *args, **kwargs: matched_calls.append(args) or Credential("wrong", "pw")
        ),
    )
    credential = get_credentials(matched, "https://repo.example/simple/pkg")
    assert (credential.username, credential.password) == ("repo-user", None)
    assert matched_calls == []

    unmatched_calls: list[tuple[Any, ...]] = []
    unmatched = SimpleNamespace(
        get_repository_config_for_url=lambda _url, _exact: None,
        _get_credentials_for_repository=lambda repository: None,
        _password_manager=SimpleNamespace(
            get_credential=lambda *args, **kwargs: unmatched_calls.append(args) or Credential("url-user", "url-pass")
        ),
    )
    credential = get_credentials(unmatched, "https://other.example/path")
    assert (credential.username, credential.password) == ("url-user", "url-pass")
    assert unmatched_calls == [("https://other.example/path", "other.example")]


def pip() -> None:
    source = function_source("src/pip/_internal/commands/list.py", "_build_package_finder")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SelectionPreferences"
    ]
    assert len(calls) == 1
    keywords = {item.arg: ast.unparse(item.value) for item in calls[0].keywords}
    assert keywords == {
        "allow_yanked": "False",
        "release_control": "options.release_control",
        "format_control": "options.format_control",
        "prefer_binary": "options.prefer_binary",
    }
    assert "uploaded_prior_to=options.uploaded_prior_to" in source


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
    assert strings.count("_transport_sockname") == 1
    assert strings.count("_transport_peername") == 1
    assert strings.count("_transport_sslcontext") == 1
    assert len(strings) == len(set(strings))


def httpx() -> None:
    auth = module_all("httpx/_auth.py")
    package = module_all("httpx/__init__.py")
    assert auth.count("FunctionAuth") == 1
    assert package.count("FunctionAuth") == 1
    assert all(auth.count(name) == 1 for name in ("Auth", "BasicAuth", "DigestAuth", "NetRCAuth"))
    assert package.index("DigestAuth") < package.index("FunctionAuth") < package.index("get")


def httpcore() -> None:
    for path in (
        "httpcore/_sync/connection_pool.py",
        "httpcore/_async/connection_pool.py",
    ):
        source = function_source(path, "_assign_requests_to_connections")
        assert source.count("sum(connection.is_idle() for connection in self._connections)") == 1
        assert "len([connection.is_idle() for connection in self._connections])" not in source
        assert "connection.is_closed()" in source and "connection.has_expired()" in source
        assert "queued_requests" in source and "available_connections" in source


def isort() -> None:
    parsed = function_source("isort/parse.py", "file_contents")
    assert '.replace("\\r\\n", "\\n").replace("\\r", "\\n").split("\\n")' in parsed
    assert "if not contents:" in parsed and "in_lines = []" in parsed
    assert "contents.splitlines()" not in parsed
    rendered = function_source("isort/output.py", "sorted_imports")
    assert "trailing_blank_lines.append(formatted_output.pop(imports_tail))" in rendered
    assert 'character == "\\f"' in rendered
    assert "blank_lines[index] += form_feeds" in rendered
    assert "overflow_form_feeds + formatted_output[imports_tail]" in rendered
    assert "formatted_output[imports_tail:0] = blank_lines" in rendered


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
    assert set(cache) == {"keep", "grow"} and cache.currsize == 5
    cache["grow"] = 1
    assert set(cache) == {"keep", "grow"} and cache.currsize == 3

    eviction = module.Cache(maxsize=4, getsizeof=lambda value: value)
    eviction["grow"] = 1
    eviction["other"] = 3
    eviction["grow"] = 3
    assert set(eviction) == {"grow"} and eviction.currsize == 3


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
    code = compile(expression, "<p4s-map-condition>", "eval")

    def condition(force: bool, subdomain_matching: bool, host_matching: bool, domain: str, host: str) -> bool:
        owner = SimpleNamespace(
            map=SimpleNamespace(
                subdomain_matching=subdomain_matching, host_matching=host_matching
            ),
            subdomain="api",
            server_name="example.test",
        )
        return bool(eval(code, {}, {
            "self": owner,
            "force_external": force,
            "domain_part": domain,
            "host": host,
        }))

    assert condition(False, False, False, "elsewhere", "elsewhere") is True
    assert condition(False, True, False, "api", "elsewhere") is True
    assert condition(False, True, False, "other", "elsewhere") is False
    assert condition(False, False, True, "other", "example.test") is True
    assert condition(False, False, True, "other", "other.test") is False
    assert condition(True, False, False, "api", "example.test") is False


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
    case = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        CASES[case]()
    except Exception as exc:
        traceback.print_exc()
        print(f"P4S_FAIL_{case}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"P4S_PASS_{case}")
