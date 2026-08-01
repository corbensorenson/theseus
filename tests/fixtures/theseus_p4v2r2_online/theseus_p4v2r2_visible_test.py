#!/usr/bin/env python3
"""Bounded obligation-addressable feedback for fresh P4-v2r2 tasks."""

from __future__ import annotations

import ast
import collections
import copy
import io
import sys
import tempfile
import tokenize
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
        return None if self.remove_relative_imports and node.level else node


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
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node = Strip().visit(copy.deepcopy(node))
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
    resolved = str(Path(path).resolve())
    sys.path.insert(0, resolved)
    try:
        return __import__(module, fromlist=["*"])
    finally:
        sys.path.remove(resolved)


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

    def index() -> str:
        return "index"

    index.provide_automatic_options = True
    captured: list[Rule] = []
    owner = SimpleNamespace(
        config={"PROVIDE_AUTOMATIC_OPTIONS": False},
        url_rule_class=Rule,
        url_map=SimpleNamespace(add=captured.append),
        view_functions={},
    )
    add_url_rule(owner, "/", view_func=index)
    assert "OPTIONS" in captured[0].methods, (
        "P4V2R2_VISIBLE_FLASK_OPTIONS_OVERRIDE"
    )


def jinja() -> None:
    module = local_import("src", "jinja2")
    assert module.Environment(enable_async=True).overlay().is_async, (
        "P4V2R2_VISIBLE_JINJA_OVERLAY_ASYNC"
    )


def textual() -> None:
    module = local_import("src", "textual._xterm_parser")
    events = module.XTermParser()._parse_extended_key("\x1b[58;2;126:47u")
    assert events is not None and [event.character for event in events] == ["~", "/"], (
        "P4V2R2_VISIBLE_TEXTUAL_MULTI_CODEPOINT"
    )


def scrapy() -> None:
    cls = class_object(
        "scrapy/utils/datatypes.py",
        "CaseInsensitiveDict",
        {"Any": Any, "collections": collections},
    )
    original = cls({"Header": "value"})
    copied = original.copy()
    del copied["header"]
    assert "HEADER" in original, "P4V2R2_VISIBLE_SCRAPY_COPY_ISOLATION"
    original |= {"HEADER": "new"}
    assert len(original) == 1 and original["header"] == "new", (
        "P4V2R2_VISIBLE_SCRAPY_INPLACE_UNION"
    )


def pyflakes() -> None:
    module = local_import(".", "pyflakes.checker")
    code = "x: int\nx.__dict__\ndef f(): x = 1\n"
    names = [
        item.__class__.__name__
        for item in module.Checker(ast.parse(code), filename="visible.py").messages
    ]
    assert "UndefinedName" in names and "UnusedVariable" in names, (
        "P4V2R2_VISIBLE_PYFLAKES_ANNOTATION_SCOPE"
    )


def pycodestyle() -> None:
    module = local_import(".", "pycodestyle")
    line = "class C3[T, U: str=str]:\n"
    tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
    messages = [
        message
        for _, message in module.whitespace_around_named_parameter_equals(
            line.rstrip("\n"), tokens
        )
    ]
    assert any(message.startswith("E252") for message in messages), (
        "P4V2R2_VISIBLE_PYCODESTYLE_GENERIC_DEFAULT"
    )


def django_rest_framework() -> None:
    map_min_max = method(
        "rest_framework/schemas/openapi.py", "AutoSchema", "_map_min_max", {}
    )
    content: dict[str, Any] = {}
    map_min_max(None, SimpleNamespace(min_value=0, max_value=0), content)
    assert content == {"minimum": 0, "maximum": 0}, (
        "P4V2R2_VISIBLE_DRF_ZERO_BOUNDS"
    )


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
    namespace = {
        "_utils": SimpleNamespace(position_to_jedi_linecolumn=lambda *_args: {}),
        "_resolve_definition": lambda value, _script, _settings: value,
        "_not_internal_definition": lambda _value: True,
        "jedi": SimpleNamespace(settings=SimpleNamespace(auto_import_modules=[])),
        "uris": SimpleNamespace(uri_with=lambda uri, **_kwargs: uri),
    }
    definitions = function(
        "pylsp/plugins/definition.py", "pylsp_definitions", namespace
    )
    config = SimpleNamespace(plugin_settings=lambda _name: {})
    assert definitions(config, document, {"line": 0, "character": 0}) == [], (
        "P4V2R2_VISIBLE_PYLSP_POSITION_GUARD"
    )


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
    with tempfile.TemporaryDirectory(prefix="visible-hypothesis-") as tmp:
        artifact = Path(tmp) / "artifact.zip"
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr("key/value", b"payload")
        owner = SimpleNamespace(
            _artifact=artifact,
            _initialized=False,
            _access_cache=None,
            _disabled=False,
        )
        prepare(owner)
        assert owner._access_cache == {PurePath("key"): {PurePath("key/value")}}, (
            "P4V2R2_VISIBLE_HYPOTHESIS_ZIP_LAYOUT"
        )


def pillow() -> None:
    class ConvertedPalette:
        def putpalette(self, _mode: str, _rawmode: str, data: bytes) -> None:
            self.converted = []
            for blue, green, red, _padding in zip(*[iter(data)] * 4):
                self.converted.extend((red, green, blue))

        def getpalette(self, _mode: str) -> list[int]:
            return self.converted

    colors = method(
        "src/PIL/ImagePalette.py",
        "ImagePalette",
        "colors",
        {"Image": SimpleNamespace(core=SimpleNamespace(new=lambda *_args: ConvertedPalette()))},
        remove_relative_imports=True,
    )
    owner = SimpleNamespace(
        _colors=None,
        mode="RGB",
        rawmode="BGRX",
        palette=(0, 0, 255, 0, 0, 255, 0, 0),
    )
    assert colors(owner) == {(255, 0, 0): 0, (0, 255, 0): 1}, (
        "P4V2R2_VISIBLE_PILLOW_RAW_PALETTE"
    )


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
    except Exception as exc:
        message = str(exc)
        marker = f"P4V2R2_VISIBLE_{selected.upper()}_PRIMARY"
        print(message if message.startswith("P4V2R2_VISIBLE_") else marker, file=sys.stderr)
        raise
