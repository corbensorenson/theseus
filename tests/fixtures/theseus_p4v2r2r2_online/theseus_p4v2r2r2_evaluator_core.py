#!/usr/bin/env python3
"""Dependency-free behavioral evaluator core for the fresh P4 recovery pool."""

from __future__ import annotations

import ast
import collections.abc
import copy
import functools
import hashlib
import hmac
import html
import io
import re
import types
import typing
from pathlib import Path
from typing import Any


SOURCES = {
    "h11": Path("h11/_headers.py"),
    "h2": Path("src/h2/utilities.py"),
    "pygments": Path("pygments/formatters/other.py"),
    "pluggy": Path("src/pluggy/_callers.py"),
    "asgiref": Path("asgiref/sync.py"),
    "platformdirs": Path("src/platformdirs/unix.py"),
    "markupsafe": Path("src/markupsafe/__init__.py"),
    "itsdangerous": Path("src/itsdangerous/signer.py"),
    "tryceratops": Path("src/tryceratops/analyzers/call.py"),
    "xarray": Path("xarray/indexes/range_index.py"),
}


def parse(case: str) -> tuple[Path, ast.Module]:
    path = SOURCES[case]
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level(tree: ast.Module, kind: type[ast.AST], name: str) -> ast.AST:
    matches = [
        node
        for node in tree.body
        if isinstance(node, kind) and getattr(node, "name", None) == name
    ]
    assert len(matches) == 1, f"missing or duplicate top-level {name}"
    return matches[0]


def last_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert matches, f"missing top-level function {name}"
    return matches[-1]


def owner_method(tree: ast.Module, owner: str, name: str) -> ast.FunctionDef:
    cls = top_level(tree, ast.ClassDef, owner)
    assert isinstance(cls, ast.ClassDef)
    matches = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"missing or duplicate {owner}.{name}"
    return matches[0]


def execute(nodes: list[ast.AST], path: Path, namespace: dict[str, Any]) -> dict[str, Any]:
    body: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    for original in nodes:
        node = copy.deepcopy(original)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
        body.append(typing.cast(ast.stmt, node))
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec", dont_inherit=True), namespace)
    return namespace


class ProtocolError(Exception):
    pass


def _validate(pattern: re.Pattern[bytes], value: bytes, message: str, *args: Any) -> None:
    if pattern.fullmatch(value) is None:
        raise ProtocolError(message.format(*args))


def evaluate_h11(level: str) -> None:
    path, tree = parse("h11")
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            name in {"CONTENT_LENGTH_MAX_DIGITS", "_content_length_re", "_field_name_re", "_field_value_re"}
            for name in (
                [target.id for target in node.targets if isinstance(target, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id] if isinstance(node.target, ast.Name) else []
            )
        )
    ]
    namespace = execute(
        [*assignments, top_level(tree, ast.ClassDef, "Headers"), last_function(tree, "normalize_and_validate")],
        path,
        {
            "re": re,
            "field_name": r"[-!#$%&'*+.^_`|~0-9A-Za-z]+",
            "field_value": r"[\x20-\x7e\x80-\xff]*",
            "Sequence": collections.abc.Sequence,
            "Tuple": typing.Tuple,
            "bytesify": lambda value: value.encode() if isinstance(value, str) else value,
            "LocalProtocolError": ProtocolError,
            "validate": _validate,
        },
    )
    normalize = namespace["normalize_and_validate"]
    accepted = normalize([(b"Content-Length", b"9" * 20)])
    assert list(accepted) == [(b"content-length", b"9" * 20)]
    try:
        normalize([(b"Content-Length", b"9" * 21)])
    except ProtocolError:
        pass
    else:
        raise AssertionError("21-digit Content-Length was accepted")
    if level == "hidden":
        for invalid in (b"12x", b"1, 2"):
            try:
                normalize([(b"content-length", invalid)])
            except ProtocolError:
                pass
            else:
                raise AssertionError(f"invalid Content-Length accepted: {invalid!r}")
        assert list(normalize([(b"X-Test", b"value")])) == [(b"x-test", b"value")]


class HeaderTuple(tuple):
    def __new__(cls, name: bytes, value: bytes) -> "HeaderTuple":
        return tuple.__new__(cls, (name, value))


class NeverIndexedHeaderTuple(HeaderTuple):
    pass


def evaluate_h2(level: str) -> None:
    path, tree = parse("h2")
    secure = execute(
        [last_function(tree, "_secure_headers")],
        path,
        {
            "_SECURE_HEADERS": frozenset({b"authorization", b"proxy-authorization"}),
            "NeverIndexedHeaderTuple": NeverIndexedHeaderTuple,
        },
    )["_secure_headers"]
    ordinary = HeaderTuple(b"cook", b"short")
    cookie = HeaderTuple(b"cookie", b"short")
    secured = list(secure([ordinary, cookie], None))
    assert secured[0] is ordinary
    assert isinstance(secured[1], NeverIndexedHeaderTuple)

    captured: list[tuple[set[bytes], bytes | None]] = []
    pseudo = execute(
        [last_function(tree, "_reject_pseudo_header_fields")],
        path,
        {
            "SIGIL": ord(b":"),
            "_ALLOWED_PSEUDO_HEADER_FIELDS": frozenset({b":me"}),
            "ProtocolError": ProtocolError,
            "_check_pseudo_header_field_acceptability": lambda seen, method, flags: captured.append((seen, method)),
        },
    )["_reject_pseudo_header_fields"]
    assert list(pseudo([(b":me", b"GET")], None)) == [(b":me", b"GET")]
    assert captured == [({b":me"}, None)]
    if level == "hidden":
        authorization = HeaderTuple(b"authorization", b"long enough to index")
        long_cookie = HeaderTuple(b"cookie", b"x" * 20)
        values = list(secure([authorization, long_cookie], None))
        assert isinstance(values[0], NeverIndexedHeaderTuple)
        assert values[1] is long_cookie


class _Token:
    Error = object()
    Text = object()


def evaluate_pygments(level: str) -> None:
    path, tree = parse("pygments")
    format_method = owner_method(tree, "RawTokenFormatter", "format")

    def colorize(color: str, value: str) -> str:
        assert isinstance(value, str)
        return f"<{color}>{value}</{color}>"

    function = execute([format_method], path, {"Token": _Token, "colorize": colorize})["format"]
    formatter = types.SimpleNamespace(compress="", error_color="red")
    output = io.BytesIO()
    function(formatter, [(_Token.Error, "bad"), (_Token.Text, "ok")], output)
    rendered = output.getvalue()
    assert rendered.startswith(b"<red>") and b"bad" in rendered and b"ok" in rendered
    if level == "hidden":
        formatter.error_color = None
        plain = io.BytesIO()
        function(formatter, [(_Token.Error, "bad")], plain)
        assert not plain.getvalue().startswith(b"<red>")
        assert b"bad" in plain.getvalue()


class _Result:
    def __class_getitem__(cls, item: Any) -> type["_Result"]:
        return cls

    def __init__(self, result: Any, exception: BaseException | None) -> None:
        self._result = result
        self._exception = exception

    def force_result(self, result: Any) -> None:
        self._result = result
        self._exception = None

    def force_exception(self, exception: BaseException) -> None:
        self._exception = exception

    def get_result(self) -> Any:
        if self._exception is not None:
            raise self._exception
        return self._result


def _hook(function: Any, *, wrapper: bool = False, hookwrapper: bool = False) -> Any:
    return types.SimpleNamespace(
        function=function,
        wrapper=wrapper,
        hookwrapper=hookwrapper,
        argnames=(),
        plugin_name="independent-evaluator",
    )


def evaluate_pluggy(level: str) -> None:
    path, tree = parse("pluggy")

    def wrapfail(controller: Any, message: str) -> None:
        raise AssertionError(message)

    multicall = execute(
        [last_function(tree, "_multicall")],
        path,
        {
            "cast": typing.cast,
            "Generator": typing.Generator,
            "HookCallError": RuntimeError,
            "Result": _Result,
            "_raise_wrapfail": wrapfail,
            "_warn_teardown_exception": lambda *args: None,
        },
    )["_multicall"]

    def new_wrapper():
        yield

    def old_wrapper():
        yield

    def raises_stop() -> None:
        raise StopIteration("implementation-stop")

    for hooks in (
        [_hook(raises_stop), _hook(new_wrapper, wrapper=True)],
        [_hook(raises_stop), _hook(new_wrapper, wrapper=True), _hook(old_wrapper, hookwrapper=True)],
    ):
        try:
            multicall("hook", hooks, {}, False)
        except StopIteration as exc:
            assert str(exc) == "implementation-stop"
        except RuntimeError as exc:
            raise AssertionError("StopIteration leaked as generator RuntimeError") from exc
        else:
            raise AssertionError("implementation StopIteration was swallowed")
    if level == "hidden":
        def raises_value() -> None:
            raise ValueError("preserve-me")

        try:
            multicall("hook", [_hook(raises_value), _hook(new_wrapper, wrapper=True)], {}, False)
        except ValueError as exc:
            assert str(exc) == "preserve-me"
        else:
            raise AssertionError("non-StopIteration exception was not preserved")


def evaluate_asgiref(level: str) -> None:
    path, tree = parse("asgiref")
    init = owner_method(tree, "SyncToAsync", "__init__")
    function = execute(
        [init],
        path,
        {
            "functools": functools,
            "iscoroutinefunction": lambda value: False,
            "markcoroutinefunction": lambda value: None,
        },
    )["__init__"]

    class CallableObject:
        __name__ = "callable_object"
        __qualname__ = "CallableObject"
        __module__ = __name__
        __doc__ = "callable object"
        __annotations__: dict[str, Any] = {}

        def __init__(self) -> None:
            self.context = "callable-owned-context"

        def __call__(self) -> str:
            return "ok"

    wrapper = types.SimpleNamespace()
    function(wrapper, CallableObject(), context=None)
    assert wrapper.context is None
    assert wrapper._thread_sensitive is True
    if level == "hidden":
        def ordinary() -> str:
            return "ok"

        configured = object()
        wrapper2 = types.SimpleNamespace()
        function(wrapper2, ordinary, thread_sensitive=False, context=configured)
        assert wrapper2.context is configured
        assert wrapper2.__wrapped__ is ordinary


def evaluate_platformdirs(level: str) -> None:
    path, tree = parse("platformdirs")
    config = owner_method(tree, "_UnixDefaults", "iter_config_dirs")
    data = owner_method(tree, "_UnixDefaults", "iter_data_dirs")
    namespace = execute([config, data], path, {})
    site = types.SimpleNamespace(
        _use_site=True,
        user_config_dir="/site/config",
        _site_config_dirs=["/site/config"],
        user_data_dir="/site/data",
        _site_data_dirs=["/site/data"],
    )
    assert list(namespace["iter_config_dirs"](site)) == ["/site/config"]
    assert list(namespace["iter_data_dirs"](site)) == ["/site/data"]
    if level == "hidden":
        user = types.SimpleNamespace(
            _use_site=False,
            user_config_dir="/user/config",
            _site_config_dirs=["/site/config"],
            user_data_dir="/user/data",
            _site_data_dirs=["/site/data"],
        )
        assert list(namespace["iter_config_dirs"](user)) == ["/user/config", "/site/config"]
        assert list(namespace["iter_data_dirs"](user)) == ["/user/data", "/site/data"]


class _MiniMarkup(str):
    def unescape(self) -> str:
        return html.unescape(str(self))


def evaluate_markupsafe(level: str) -> None:
    path, tree = parse("markupsafe")
    method = owner_method(tree, "Markup", "striptags")
    function = execute([method], path, {})["striptags"]
    assert function(_MiniMarkup("A <i> </i> B")) == "A B"
    if level == "hidden":
        assert function(_MiniMarkup("Main &raquo;\t<em>About</em>")) == "Main » About"
        assert function(_MiniMarkup("A<!-- hidden -->   B")) == "A B"


class _UnavailableSha1:
    @property
    def sha1(self) -> Any:
        raise RuntimeError("sha1 unavailable at import time")


def evaluate_itsdangerous(level: str) -> None:
    path, tree = parse("itsdangerous")
    names = {"SigningAlgorithm", "NoneAlgorithm", "_lazy_sha1", "HMACAlgorithm", "Signer"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = execute(
        nodes,
        path,
        {
            "hashlib": _UnavailableSha1(),
            "hmac": hmac,
            "cabc": collections.abc,
            "t": typing,
        },
    )
    namespace["hashlib"] = hashlib
    hmac_algorithm = namespace["HMACAlgorithm"]
    signer = namespace["Signer"]
    assert hmac_algorithm.default_digest_method(b"x").digest() == hashlib.sha1(b"x").digest()
    assert signer.default_digest_method(b"y").digest() == hashlib.sha1(b"y").digest()
    if level == "hidden":
        assert hmac_algorithm().get_signature(b"key", b"value") == hmac.new(
            b"key", msg=b"value", digestmod=hashlib.sha1
        ).digest()


def _raise_call(argument: str) -> tuple[ast.Raise, ast.Call, ast.Name]:
    node = ast.parse(f"raise Exception({argument})").body[0]
    assert isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
    assert isinstance(node.exc.func, ast.Name)
    return node, node.exc, node.exc.func


def evaluate_tryceratops(level: str) -> None:
    path, tree = parse("tryceratops")
    method = owner_method(tree, "CallRaiseLongArgsAnalyzer", "_check_raise_callable")
    function = execute([method], path, {"ast": ast})["_check_raise_callable"]

    class Recorder:
        def __init__(self) -> None:
            self.count = 0

        def _mark_violation(self, node: ast.AST) -> None:
            self.count += 1

    recorder = Recorder()
    function(recorder, *_raise_call("f'code_{value}'"))
    assert recorder.count == 0
    function(recorder, *_raise_call("f'bad value {value}'"))
    assert recorder.count == 1
    if level == "hidden":
        function(recorder, *_raise_call("'two words'"))
        assert recorder.count == 2
        function(recorder, *_raise_call("'single'"))
        assert recorder.count == 2


class _RangeCoordinateTransform:
    def __init__(
        self,
        start: float,
        stop: float,
        size: int,
        coord_name: Any,
        dim: str,
        dtype: Any = None,
    ) -> None:
        self.start = start
        self.stop = stop
        self.size = size
        self.coord_name = coord_name
        self.dim = dim
        self.dtype = dtype


class _RangeIndex:
    def __init__(self, transform: _RangeCoordinateTransform) -> None:
        self.transform = transform


def evaluate_xarray(level: str) -> None:
    path, tree = parse("xarray")
    method = owner_method(tree, "RangeIndex", "linspace")
    function = execute([method], path, {"RangeCoordinateTransform": _RangeCoordinateTransform})["linspace"]
    one = function(_RangeIndex, 2.0, 8.0, 1, True, dim="x")
    assert one.transform.start == 2.0 and one.transform.stop == 8.0 and one.transform.size == 1
    if level == "hidden":
        many = function(_RangeIndex, 0.0, 1.0, 5, True, dim="x")
        assert many.transform.stop == 1.25 and many.transform.size == 5
        open_interval = function(_RangeIndex, 0.0, 1.0, 5, False, dim="x")
        assert open_interval.transform.stop == 1.0


EVALUATORS = {
    "h11": evaluate_h11,
    "h2": evaluate_h2,
    "pygments": evaluate_pygments,
    "pluggy": evaluate_pluggy,
    "asgiref": evaluate_asgiref,
    "platformdirs": evaluate_platformdirs,
    "markupsafe": evaluate_markupsafe,
    "itsdangerous": evaluate_itsdangerous,
    "tryceratops": evaluate_tryceratops,
    "xarray": evaluate_xarray,
}


def evaluate(case: str, level: str) -> None:
    assert case in EVALUATORS, f"unknown evaluator case: {case}"
    assert level in {"visible", "hidden"}
    EVALUATORS[case](level)
