#!/usr/bin/env python3
"""Independent hidden behavior checks for the fresh P4-v2r2 denominator."""

from __future__ import annotations

import ast
import collections
import copy
import io
import re
import sys
import tempfile
import tokenize
import types
import warnings
import zipfile
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Any


class Strip(ast.NodeTransformer):
    def __init__(self, *, remove_relative_imports: bool = False) -> None:
        self.remove_relative_imports = remove_relative_imports

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.decorator_list = []
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.stmt | None:
        if self.remove_relative_imports and node.level:
            return None
        return node


def method(
    path: str,
    owner: str,
    name: str,
    namespace: dict[str, Any],
    *,
    remove_relative_imports: bool = False,
) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == owner
    )
    node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node = Strip(remove_relative_imports=remove_relative_imports).visit(
        copy.deepcopy(node)
    )
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def function(path: str, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    node = Strip().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(module := ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def class_object(path: str, name: str, namespace: dict[str, Any]) -> type:
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    source = ast.get_source_segment(text, node)
    assert source is not None
    exec(compile("from __future__ import annotations\n" + source, path, "exec"), namespace)
    return namespace[name]


def local_import(path: str, module: str) -> Any:
    sys.path.insert(0, str(Path(path).resolve()))
    try:
        return __import__(module, fromlist=["*"])
    finally:
        sys.path.remove(str(Path(path).resolve()))


def flask() -> None:
    add_url_rule = method(
        "src/flask/sansio/app.py",
        "App",
        "add_url_rule",
        {"_endpoint_from_view_func": lambda value: value.__name__},
    )

    class Rule:
        def __init__(self, rule: str, *, methods: set[str], **options: Any) -> None:
            self.rule, self.methods, self.options = rule, methods, options

    def owner(config: bool) -> SimpleNamespace:
        return SimpleNamespace(
            config={"PROVIDE_AUTOMATIC_OPTIONS": config},
            url_rule_class=Rule,
            url_map=SimpleNamespace(add=lambda _rule: None),
            view_functions={},
        )

    def enabled() -> str:
        return "enabled"

    enabled.provide_automatic_options = True
    enabled_owner = owner(False)
    captured: list[Rule] = []
    enabled_owner.url_map.add = captured.append
    add_url_rule(enabled_owner, "/enabled", view_func=enabled)
    assert "OPTIONS" in captured[0].methods

    def disabled() -> str:
        return "disabled"

    disabled.provide_automatic_options = False
    disabled_owner = owner(True)
    captured_disabled: list[Rule] = []
    disabled_owner.url_map.add = captured_disabled.append
    add_url_rule(disabled_owner, "/disabled", view_func=disabled)
    assert "OPTIONS" not in captured_disabled[0].methods

    manual_owner = owner(False)
    captured_manual: list[Rule] = []
    manual_owner.url_map.add = captured_manual.append
    add_url_rule(
        manual_owner, "/manual", view_func=lambda: "manual", methods=["OPTIONS"]
    )
    assert captured_manual[0].methods == {"OPTIONS"}


def jinja() -> None:
    module = local_import("src", "jinja2")
    asynchronous = module.Environment(enable_async=True)
    assert asynchronous.overlay().is_async is True
    assert asynchronous.overlay(enable_async=False).is_async is False
    synchronous = module.Environment(enable_async=False)
    assert synchronous.overlay().is_async is False
    assert synchronous.overlay(enable_async=True).is_async is True


def textual() -> None:
    module = local_import("src", "textual._xterm_parser")
    parser = module.XTermParser()
    events = parser._parse_extended_key("\x1b[58;2;126:47u")
    assert events is not None
    assert [event.character for event in events] == ["~", "/"]
    ordinary = parser._parse_extended_key("\x1b[97;1;97u")
    assert ordinary is not None and len(ordinary) == 1
    assert ordinary[0].character == "a"
    assert [event.character for event in parser.feed("\x1b[58;2;126:47u")] == [
        "~",
        "/",
    ]


def scrapy() -> None:
    cls = class_object(
        "scrapy/utils/datatypes.py",
        "CaseInsensitiveDict",
        {"Any": Any, "collections": collections},
    )
    original = cls({"Header1": "value1", "header2": "value2"})
    for copied in (copy.copy(original), original.copy()):
        del copied["HEADER1"]
        copied["Header3"] = "value3"
        assert "header1" in original and "header3" not in original
        assert dict(original) == {"Header1": "value1", "header2": "value2"}

    class Normalized(cls):
        def _normvalue(self, value: int) -> int:
            return value + 1

    normalized = Normalized({"key": 1})
    assert copy.copy(normalized)["key"] == normalized.copy()["key"] == 2
    union = cls({"Header1": "value1"})
    union |= {"HEADER1": "value2", "header2": "value3"}
    assert len(union) == 2
    assert union["header1"] == "value2" and union["HEADER2"] == "value3"


def pyflakes() -> None:
    module = local_import(".", "pyflakes.checker")
    code = "x: int\nx.__dict__\ndef f(): x = 1\n"
    messages = module.Checker(ast.parse(code), filename="p4v2r2.py").messages
    names = [item.__class__.__name__ for item in messages]
    assert names.count("UndefinedName") == 1
    assert names.count("UnusedVariable") == 1
    ordinary = module.Checker(
        ast.parse("x: int\nx = 1\nprint(x)\n"), filename="ordinary.py"
    ).messages
    assert ordinary == []


def pycodestyle() -> None:
    module = local_import(".", "pycodestyle")

    def messages(line: str) -> list[str]:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
        return [
            message
            for _, message in module.whitespace_around_named_parameter_equals(
                line.rstrip("\n"), tokens
            )
        ]

    assert any(
        message.startswith("E252")
        for message in messages("class C3[T, U: str=str]:\n")
    )
    assert not any(
        message.startswith(("E251", "E252"))
        for message in messages("class C3[T, U: str = str]:\n")
    )
    assert any(
        message.startswith("E252")
        for message in messages("def f(arg: int=3):\n")
    )


def django_rest_framework() -> None:
    map_min_max = method(
        "rest_framework/schemas/openapi.py", "AutoSchema", "_map_min_max", {}
    )
    for minimum, maximum in ((0, None), (None, 0), (0.0, None), (None, 0.0)):
        content: dict[str, Any] = {}
        map_min_max(None, SimpleNamespace(min_value=minimum, max_value=maximum), content)
        assert content == (
            {"minimum": minimum} if minimum is not None else {"maximum": maximum}
        )
    absent: dict[str, Any] = {}
    map_min_max(None, SimpleNamespace(min_value=None, max_value=None), absent)
    assert absent == {}
    nonzero: dict[str, Any] = {}
    map_min_max(None, SimpleNamespace(min_value=-2, max_value=7), nonzero)
    assert nonzero == {"minimum": -2, "maximum": 7}


def python_lsp_server() -> None:
    definition = SimpleNamespace(
        line=None,
        column=None,
        module_path=None,
        name="compiled",
        is_definition=lambda: True,
    )
    script = SimpleNamespace(goto=lambda **_kwargs: [definition])
    document = SimpleNamespace(
        uri="file:///candidate.py",
        jedi_script=lambda **_kwargs: script,
    )
    jedi = SimpleNamespace(settings=SimpleNamespace(auto_import_modules=["old"]))
    namespace = {
        "_utils": SimpleNamespace(position_to_jedi_linecolumn=lambda *_args: {}),
        "_resolve_definition": lambda value, _script, _settings: value,
        "_not_internal_definition": lambda _value: True,
        "jedi": jedi,
        "uris": SimpleNamespace(uri_with=lambda uri, **_kwargs: uri),
    }
    definitions = function(
        "pylsp/plugins/definition.py", "pylsp_definitions", namespace
    )
    config = SimpleNamespace(plugin_settings=lambda _name: {})
    assert definitions(config, document, {"line": 0, "character": 0}) == []
    assert jedi.settings.auto_import_modules == ["old"]


def hypothesis() -> None:
    prepare = method(
        "hypothesis/src/hypothesis/database.py",
        "GitHubArtifactDatabase",
        "_prepare_for_io",
        {
            "BadZipFile": zipfile.BadZipFile,
            "HypothesisWarning": UserWarning,
            "PurePath": PurePath,
            "ZipFile": zipfile.ZipFile,
            "warnings": warnings,
        },
    )
    with tempfile.TemporaryDirectory(prefix="p4v2r2-hypothesis-") as tmp:
        file_only = Path(tmp) / "file-only.zip"
        with zipfile.ZipFile(file_only, "w") as archive:
            archive.writestr("key/value", b"payload")
        owner = SimpleNamespace(
            _artifact=file_only,
            _initialized=False,
            _access_cache=None,
            _disabled=False,
        )
        prepare(owner)
        assert owner._access_cache == {PurePath("key"): {PurePath("key/value")}}
        assert owner._initialized is True and owner._disabled is False

        file_then_dir = Path(tmp) / "file-then-dir.zip"
        with zipfile.ZipFile(file_then_dir, "w") as archive:
            archive.writestr("key/value", b"payload")
            archive.writestr("key/", b"")
        owner2 = SimpleNamespace(
            _artifact=file_then_dir,
            _initialized=False,
            _access_cache=None,
            _disabled=False,
        )
        prepare(owner2)
        assert owner2._access_cache == {PurePath("key"): {PurePath("key/value")}}


def pillow() -> None:
    class ConvertedPalette:
        def __init__(self) -> None:
            self.converted: list[int] = []

        def putpalette(self, mode: str, rawmode: str, data: bytes) -> None:
            assert mode == "RGB" and rawmode == "BGRX"
            for blue, green, red, _padding in zip(*[iter(data)] * 4):
                self.converted.extend((red, green, blue))

        def getpalette(self, mode: str) -> list[int]:
            assert mode == "RGB"
            return self.converted

    image = SimpleNamespace(
        core=SimpleNamespace(new=lambda mode, size: ConvertedPalette())
    )
    colors = method(
        "src/PIL/ImagePalette.py",
        "ImagePalette",
        "colors",
        {"Image": image},
        remove_relative_imports=True,
    )
    owner = SimpleNamespace(
        _colors=None,
        mode="RGB",
        rawmode="BGRX",
        palette=(
            0,
            0,
            0,
            0,
            255,
            255,
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            255,
            0,
            0,
            255,
            0,
            0,
            0,
        ),
    )
    assert colors(owner) == {
        (0, 0, 0): 0,
        (255, 255, 255): 1,
        (255, 0, 0): 2,
        (0, 255, 0): 3,
        (0, 0, 255): 4,
    }
    assert colors(owner) is owner._colors


CASES = {
    "flask": flask,
    "jinja": jinja,
    "textual": textual,
    "scrapy": scrapy,
    "pyflakes": pyflakes,
    "pycodestyle": pycodestyle,
    "django_rest_framework": django_rest_framework,
    "python_lsp_server": python_lsp_server,
    "hypothesis": hypothesis,
    "pillow": pillow,
}


if __name__ == "__main__":
    selected = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        CASES[selected]()
    except Exception:
        print(f"P4V2R2_FAIL_{selected}", file=sys.stderr)
        raise
