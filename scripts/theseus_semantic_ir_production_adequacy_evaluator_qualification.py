#!/usr/bin/env python3
"""Independently qualify the 18-task Semantic-IR adequacy evaluators.

The evaluator owns target archives and behavioral probes. Candidate-visible task
packets are not materialized here, and no model, reference, teacher, training,
D1, or D2 authority is opened.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
import copy
import functools
import json
import re
import sys
import tarfile
import tempfile
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZATION = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_materialization_v4.json"
MATERIALIZATION_SHA256 = "7572e6ebb82ae6b16575298c42450a31d7c50ce2823fd5fc6346b12d6216f122"
CONSTRUCT_REVIEW = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_source_construct_review_v2.json"
CONSTRUCT_REVIEW_SHA256 = "bf2780ba8e38e2d9959aaee4603ca3e7d67907e9f8dda855291a72827476e053"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_evaluator_qualification.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_evaluator_qualification_v1"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()
    report = qualify()
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def qualify() -> dict[str, Any]:
    faults: list[str] = []
    if sha256_file(MATERIALIZATION) != MATERIALIZATION_SHA256:
        faults.append("materialization_binding_invalid")
    if sha256_file(CONSTRUCT_REVIEW) != CONSTRUCT_REVIEW_SHA256:
        faults.append("construct_review_binding_invalid")
    materialization = read_json(MATERIALIZATION)
    construct = read_json(CONSTRUCT_REVIEW)
    if materialization.get("trigger_state") != "GREEN":
        faults.append("materialization_not_green")
    if (
        construct.get("trigger_state") != "GREEN"
        or construct.get("panel_admitted_for_evaluator_qualification") is not True
    ):
        faults.append("construct_review_not_green")

    rows: list[dict[str, Any]] = []
    if not faults:
        for row in dictionaries(materialization.get("rows")):
            result = qualify_row(row)
            rows.append(result)
            faults.extend(
                f"task_{row.get('index')}:{fault}" for fault in result["faults"]
            )
    green = sum(row.get("trigger_state") == "GREEN" for row in rows)
    counters = zero_counters()
    counters.update({
        "parent_target_evaluator_executions": len(rows) * 2,
        "evaluator_control_executions": len(rows) * 4,
    })
    admitted = not faults and green == 18
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if admitted else "RED",
        "faults": sorted(set(faults)),
        "materialization_report": artifact(MATERIALIZATION),
        "construct_review": artifact(CONSTRUCT_REVIEW),
        "evaluator_owner": artifact(Path(__file__).resolve()),
        "runtime": {
            "python_implementation": sys.implementation.name,
            "python_version": sys.version.split()[0],
        },
        "qualification_method": (
            "Evaluator-custody AST extraction plus dependency-stubbed behavioral "
            "micro-harnesses for the named causal slice. Checks are source-disjoint "
            "from upstream tests and accept a benign comment variant rather than exact bytes."
        ),
        "task_count": len(rows),
        "green_task_count": green,
        "panel_admitted_for_task_packet_materialization": admitted,
        "candidate_or_model_exposure_authorized": False,
        "rows": rows,
        "counters": counters,
        "maximum_inference": (
            "A GREEN report establishes only that each independent causal-slice evaluator "
            "rejects the frozen parent, accepts the frozen target and a benign-equivalent "
            "variant, and rejects required-mechanism, missing-path, and unauthorized-path "
            "controls. It does not establish candidate competence, a Semantic-IR treatment "
            "effect, full upstream repository correctness, D1, D2, training value, serving, "
            "or ASI Stack book support."
        ),
    }


def qualify_row(row: dict[str, Any]) -> dict[str, Any]:
    index = integer(row.get("index"))
    expected_paths = tuple(str(value) for value in row.get("selected_source_paths") or [])
    parent = archive_sources(row, "parent")
    target = archive_sources(row, "target")
    observations: dict[str, dict[str, Any]] = {}

    def observe(name: str, sources: dict[str, str]) -> bool:
        try:
            passed = evaluate(index, sources, expected_paths)
            observations[name] = {"passed": passed, "fault": None}
            return passed
        except Exception as exc:  # evaluator failures fail closed and stay visible
            observations[name] = {
                "passed": False,
                "fault": f"{type(exc).__name__}:{exc}"[:1000],
            }
            return False

    parent_passed = observe("parent", parent)
    target_passed = observe("target", target)
    benign = dict(target)
    if expected_paths:
        benign[expected_paths[0]] += "\n# evaluator benign-equivalence probe\n"
    benign_passed = observe("benign_equivalent", benign)
    mutated = mutate_required_mechanism(index, target)
    mutation_passed = observe("required_mechanism_mutation", mutated)
    missing = dict(target)
    if expected_paths:
        missing.pop(expected_paths[0], None)
    missing_path_passed = observe("missing_required_path", missing)
    unauthorized = dict(target)
    unauthorized["unauthorized/effect.py"] = "value = 1\n"
    unauthorized_path_passed = observe("unauthorized_path", unauthorized)

    checks = {
        "parent_negative": parent_passed is False,
        "target_positive": target_passed is True,
        "benign_equivalent_positive": benign_passed is True,
        "required_mechanism_mutation_rejected": mutation_passed is False,
        "missing_required_path_rejected": missing_path_passed is False,
        "unauthorized_path_rejected": unauthorized_path_passed is False,
    }
    faults = [name for name, passed in checks.items() if not passed]
    return {
        "index": index,
        "opaque_evaluator_id": f"semantic-ir-adequacy-evaluator-{index:02d}",
        "selected_source_paths": list(expected_paths),
        "verification_kind": "isolated_behavior_plus_ast_dataflow",
        "trigger_state": "GREEN" if not faults else "RED",
        "checks": checks,
        "observations": observations,
        "faults": faults,
    }


def evaluate(index: int, sources: dict[str, str], expected_paths: tuple[str, ...]) -> bool:
    if set(sources) != set(expected_paths):
        return False
    try:
        for source in sources.values():
            ast.parse(source)
    except SyntaxError:
        return False
    checker = CHECKERS.get(index)
    return bool(checker and checker(sources))


def has(source: str, *fragments: str) -> bool:
    return all(fragment in source for fragment in fragments)


def source_at(sources: dict[str, str], suffix: str) -> str:
    matches = [value for path, value in sources.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"source path suffix {suffix!r} matched {len(matches)}")
    return matches[0]


def extracted_function(
    source: str,
    name: str,
    namespace: dict[str, Any],
    *,
    class_name: str | None = None,
) -> Any:
    tree = ast.parse(source)
    if class_name is None:
        matches = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
    else:
        classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        matches = [
            node for cls in classes for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {class_name or ''}.{name}, found {len(matches)}")
    node = AnnotationStripper().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(node)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<evaluator-extract>", "exec"), namespace)
    return namespace[name]


def extracted_class(source: str, name: str, namespace: dict[str, Any]) -> type:
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one class {name}, found {len(matches)}")
    node = AnnotationStripper().visit(copy.deepcopy(matches[0]))
    ast.fix_missing_locations(node)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<evaluator-class-extract>", "exec"), namespace)
    return namespace[name]


def check_01(sources: dict[str, str]) -> bool:
    source = source_at(sources, "storage.py")
    if not has(source, 'manifest.get("size_bytes")', '.stat().st_size', 'except FileNotFoundError:', '"size_bytes": size_bytes'):
        return False
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        legacy = root / "legacy.json"
        explicit = root / "explicit.json"
        base = {"provider": "p", "format": "f", "fetched_at": "now", "canonical_identifier": "id"}
        legacy.write_text(json.dumps(base))
        legacy.with_suffix(".data").write_bytes(b"abc")
        explicit.write_text(json.dumps({**base, "canonical_identifier": "id2", "size_bytes": 99}))
        namespace = {
            "self": SimpleNamespace(_manifest_paths=lambda: [legacy, explicit]),
            "provider": None,
            "format": None,
            "json": json,
            "FileNotFoundError": FileNotFoundError,
        }
        function = extracted_function(source, "_list", namespace)
        return [row["size_bytes"] for row in function()] == [3, 99]


def check_02(sources: dict[str, str]) -> bool:
    source = source_at(sources, "tsfresh.py")
    if not has(source, ".drop_duplicates().to_numpy()", "X[X.columns[0]]"):
        return False

    class Column:
        def unique(self): return [1, 2]
        def drop_duplicates(self): return self
        def to_numpy(self): return [2, 1]

    column = Column()

    class ILoc:
        def __getitem__(self, key): return column

    class XValue:
        columns = ["id", "sort", "kind", "value"]
        iloc = ILoc()
        def __getitem__(self, key): return column

    class Result:
        order = None
        def reindex(self, order): self.order = list(order); return self

    module = types.ModuleType("tsfresh")
    module.extract_features = lambda *args, **kwargs: Result()
    prior = sys.modules.get("tsfresh")
    sys.modules["tsfresh"] = module
    try:
        method = extracted_function(source, "_transform", {}, class_name="TSFreshFeatureExtractor")
        result = method(SimpleNamespace(default_fc_parameters_={}), XValue())
        return result.order == [2, 1]
    finally:
        if prior is None: sys.modules.pop("tsfresh", None)
        else: sys.modules["tsfresh"] = prior


def check_03(sources: dict[str, str]) -> bool:
    source = source_at(sources, "conftest.py")
    if not has(source, "PARAMS.get(parameter_code, {}).get", "parameter_code.value"):
        return False
    class Code:
        def __init__(self, value: int) -> None: self.value = value

    code = Code(9)
    param = SimpleNamespace(code=9, value=42)
    device = SimpleNamespace(
        definition=SimpleNamespace(code=7),
        state=SimpleNamespace(params=[param]),
    )
    function = extracted_function(
        source,
        "get_current_value",
        {"PARAMS": {code: {}}, "all_devices": {1: device}},
    )
    return function(1, code) == 42


def check_04(sources: dict[str, str]) -> bool:
    source = source_at(sources, "cli.py")
    if not has(source, "def _resolved_path", "if explicit is not None:", "return configured or fallback"):
        return False
    function = extracted_function(source, "_resolved_path", {})
    return function("", "configured", "fallback") == "" and function(None, "", "fallback") == "fallback"


def check_05(sources: dict[str, str]) -> bool:
    source = source_at(sources, "style_json.py")
    if not has(source, 'stashed_fill = builder.get("fillColorSaved")', '_references_builtin_fill_pattern', 'base["paint"].get("fill-color") is None'):
        return False
    namespace = {
        "_builder_style_config": lambda value: value,
        "_mvt_source_layer": lambda *args: "data",
        "_companion_visibility": lambda *args, **kwargs: {},
        "_finite_number": lambda value: value if isinstance(value, (int, float)) else None,
        "_clamp_number": lambda value, low, high: max(low, min(high, value)),
        "_extrusion_height_expression": lambda column, scale: [column, scale],
        "DEFAULT_STROKE_COLOR": "stroke",
        "DEFAULT_OUTLINE_WIDTH": 1,
        "DEFAULT_FILL_COLOR": "blue",
        "DEFAULT_EXTRUSION_MIN_ZOOM": 0,
        "EXTRUSION_OPACITY_CAP": 0.8,
    }
    function = extracted_function(source, "_fill_companion_layers", namespace)
    layer = SimpleNamespace(
        style_config={"heightColumn": "height", "fillColorSaved": "red"},
        paint={"fill-pattern": "pattern"}, filter=None, opacity=0.5, id="x",
    )
    rows = function(layer, "source", "layer")
    extrusion = next(row for row in rows if row["type"] == "fill-extrusion")
    return extrusion["paint"]["fill-extrusion-color"] == "red"


def check_06(sources: dict[str, str]) -> bool:
    source = source_at(sources, "auxfuncs.py")
    if not has(source, "from .capi_maps import f2cmap_all", "for kind_exp in range(5):", "return result"):
        return False
    package = types.ModuleType("probe")
    capi = types.ModuleType("probe.capi_maps")
    capi.f2cmap_all = {"real": {"named": "double", "8": "double"}}
    old_package, old_capi = sys.modules.get("probe"), sys.modules.get("probe.capi_maps")
    sys.modules["probe"], sys.modules["probe.capi_maps"] = package, capi
    try:
        function = extracted_function(source, "get_kind", {"__package__": "probe"})
        return function({"kindselector": {"kind": "named"}, "typespec": "real"}) == "8"
    finally:
        if old_package is None: sys.modules.pop("probe", None)
        else: sys.modules["probe"] = old_package
        if old_capi is None: sys.modules.pop("probe.capi_maps", None)
        else: sys.modules["probe.capi_maps"] = old_capi


def check_07(sources: dict[str, str]) -> bool:
    source = source_at(sources, "range_frame.py")
    if not has(source, "resolve_sort_keys(", "sort_order(", "sort_descending"):
        return False
    observed: dict[str, Any] = {}
    def resolve(columns, **kwargs): observed["descending"] = kwargs["sort_descending"]; return ["Start", "End"], [True, False]
    def order(self, keys, descending, **kwargs): observed["keys"] = keys; return [1, 0]
    namespace = {
        "resolve_sort_keys": resolve,
        "sort_order": order,
        "arg_to_list": lambda value: [] if value is None else [value] if isinstance(value, str) else value,
        "RANGE_COLS": ["Start", "End"],
        "_mypy_ensure_rangeframe": lambda value: value,
    }
    method = extracted_function(source, "sort_ranges", namespace, class_name="RangeFrame")
    holder = SimpleNamespace(columns=["Start", "End"], take=lambda indexes: tuple(indexes))
    result = method(holder, sort_descending="Start")
    return result == (1, 0) and observed == {"descending": ["Start"], "keys": ["Start", "End"]}


def check_08(sources: dict[str, str]) -> bool:
    source = source_at(sources, "worker.py")
    if not has(source, "centred_values = finite_values - origin", "(param_data[valid] - origin) / span"):
        return False
    import numpy as np
    method = extracted_function(source, "_shaped_axis_is_rectilinear", {"np": np})
    values = np.array([1e9, 1e9 + 0.01, 1e9 + 0.02])
    data = np.array([values, values[::-1]])
    holder = SimpleNamespace(_shaped_axis_values=lambda _data, _dimension: values)
    return method(holder, data, 1) is False


def check_09(sources: dict[str, str]) -> bool:
    source = source_at(sources, "ingest.py")
    if not has(source, "def _split_oversized", "MAX_CHUNK_CHARS", "_split_oversized(read_text(path))"):
        return False

    @dataclass
    class Chunk:
        text: str
        source: str = "s"
        section: str = "x"
        chunk_type: str = "text"
        position: int = 1

    function = extracted_function(source, "_split_oversized", {"Chunk": Chunk, "MAX_CHUNK_CHARS": 40, "re": re})
    text = "word " * 35
    rows = function([Chunk(text=text)])
    return len(rows) > 1 and all(20 < len(row.text) <= 40 for row in rows)


def check_10(sources: dict[str, str]) -> bool:
    source = source_at(sources, "utils.py")
    if not has(source, "if force_hf_offline_active():", "return True"):
        return False
    fake_os = SimpleNamespace(environ={})
    function = extracted_function(
        source, "hf_env_offline",
        {"force_hf_offline_active": lambda: True, "os": fake_os, "_HF_OFFLINE_TRUE_VALUES": {"1"}},
    )
    return function() is True


def check_11(sources: dict[str, str]) -> bool:
    source = source_at(sources, "fallback.py")
    tree = ast.parse(source)
    top_level_tavily = any(
        isinstance(node, ast.ImportFrom) and node.module == "tavily"
        for node in tree.body
    )
    if top_level_tavily or not has(source, "except ImportError as e:", "raise MissingAPIKeyError"):
        return False

    class MissingAPIKeyError(Exception): pass
    class EmptyContentError(Exception):
        def __init__(self, message, **kwargs): super().__init__(message)
    class FetchError(Exception):
        def __init__(self, message, **kwargs): super().__init__(message)
    real_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "tavily": raise ImportError("absent")
        return real_import(name, *args, **kwargs)
    custom_builtins = dict(vars(builtins)); custom_builtins["__import__"] = blocked_import
    function = extracted_function(
        source, "tavily_extract",
        {"__builtins__": custom_builtins, "MissingAPIKeyError": MissingAPIKeyError, "EmptyContentError": EmptyContentError, "FetchError": FetchError},
    )
    try:
        function("https://example.invalid")
    except MissingAPIKeyError:
        return True
    return False


def check_12(sources: dict[str, str]) -> bool:
    source = source_at(sources, "ed_push.py")
    if not has(source, 're.search(r"^#\\s+(.+)$", writeup, re.M)', "writeup = writeup[:m.start()] + writeup[m.end():]"):
        return False
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); (root / "writeup.md").write_text("# Duplicate title\n\nBody\n")
        captured: list[str] = []
        function = extracted_function(source, "cmd_push_challenge", {
            "need_token": lambda: "token", "load_config": lambda: {}, "Path": Path,
            "load_manifest": lambda _path: {"lesson": 1, "title": "Title", "testcase_file": "case.py"},
            "guard_lesson": lambda *args: None, "make_slide": lambda *args, **kwargs: None,
            "request": lambda *args, **kwargs: {},
            "markdown_to_ed_xml": lambda text, uploader: captured.append(text) or text,
            "make_image_uploader": lambda *args: None,
            "workspace_files": lambda _path, kind: [SimpleNamespace(name="case.py")] if kind == "testbase" else [],
            "re": re, "print": lambda *args, **kwargs: None,
        })
        function(SimpleNamespace(dir=directory, dry_run=True, force=False))
        return captured == ["\n\nBody\n"]


def check_13(sources: dict[str, str]) -> bool:
    source = source_at(sources, "mgmt_handlers.py")
    if not has(source, "status_code == 404 and condition == ERROR_CODE_MESSAGE_NOT_FOUND", "return []"):
        return False
    called: list[Any] = []
    transport = SimpleNamespace(
        get_message_value=lambda message: {b"sessions-ids": [b"one"]},
        handle_amqp_mgmt_error=lambda *args: called.append(args),
    )
    namespace = {
        "MGMT_RESPONSE_MESSAGE_ERROR_CONDITION": "condition",
        "ERROR_CODE_MESSAGE_NOT_FOUND": "not-found",
        "_LOGGER": object(),
    }
    function = extracted_function(source, "list_sessions_op", namespace)
    message = SimpleNamespace(application_properties={"condition": "not-found"})
    return function(404, message, "missing", transport) == [] and not called and function(204, message, "", transport) == []


def check_14(sources: dict[str, str]) -> bool:
    source = source_at(sources, "source_code_manager.py")
    if not has(source, "if clone_result != 0:", "return None", '"GIT_TERMINAL_PROMPT": "0"'):
        return False
    parsed = SimpleNamespace(valid=True, github=True, protocol="https", host="example", owner="o", repo="r", branch=None, path_raw="")
    class RefType: BRANCH = "branch"
    class SourceCodeReference:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    namespace = {
        "parse_git_url": lambda value: parsed, "extract_ref": lambda *args, **kwargs: "",
        "RefType": RefType, "NonAccessibleRepository": RuntimeError,
        "list_dir": lambda path: [], "path_exists": lambda path: False,
        "create_dirs": lambda path: None, "run_command": lambda *args, **kwargs: 1,
        "SourceCodeReference": SourceCodeReference, "logger": SimpleNamespace(debug=lambda *args: None, error=lambda *args: None),
        "datetime": __import__("datetime").datetime, "pytz": SimpleNamespace(UTC=__import__("datetime").timezone.utc),
        "NONINTERACTIVE_GIT_ENV": {"GIT_TERMINAL_PROMPT": "0"}, "NONINTERACTIVE_GIT_TIMEOUT_SECONDS": 30,
    }
    method = extracted_function(source, "get_code", namespace, class_name="SourceCodeManager")
    holder = SimpleNamespace(
        get_canonical_urls=lambda value: (value, "api"),
        _get_mirror_url_and_ref=lambda *args, **kwargs: ("https://example/o/r", "branch", "main", "main"),
        local_cache_dir="/cache", timestamped_dir="stamp", setup_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        local_cache_ttl=60,
    )
    return method(holder, "https://example/o/r") is None


def check_15(sources: dict[str, str]) -> bool:
    source = source_at(sources, "utils.py")
    if not has(source, "self._credentials[bucket] = credentials", "self._credentials.pop(oldest_bucket)", "self._refresh_locks.pop(oldest_bucket, None)"):
        return False
    calls: list[str] = []
    class AioIdentityCache:
        def __init__(self, client, credential_cls): pass
        async def get_credentials(self, **kwargs): calls.append(kwargs["bucket"]); return "credential:" + kwargs["bucket"]
    class S3ExpressIdentityCache: pass
    cls = extracted_class(source, "AioS3ExpressIdentityCache", {
        "AioIdentityCache": AioIdentityCache, "S3ExpressIdentityCache": S3ExpressIdentityCache,
        "OrderedDict": OrderedDict, "asyncio": asyncio, "functools": functools,
    })
    async def exercise() -> bool:
        cache = cls(None, None)
        for index in range(101): await cache.get_credentials(f"b{index}")
        again = await cache.get_credentials("b100")
        return len(cache._credentials) == 100 and "b0" not in cache._credentials and again == "credential:b100" and calls.count("b100") == 1
    return asyncio.run(exercise())


def check_16(sources: dict[str, str]) -> bool:
    coords = source_at(sources, "coords.py")
    locator = source_at(sources, "locator.py")
    if not has(coords, "def target_logical_bounds", "target.x / scale") or not has(
        locator,
        "from openframe.recognize.coords import target_logical_bounds",
        "target_logical_bounds(",
        "scale_factor=scale_factor",
    ):
        return False
    namespace: dict[str, Any] = {}
    uses = extracted_function(coords, "target_uses_physical_pixels", namespace)
    namespace["target_uses_physical_pixels"] = uses
    logical = extracted_function(coords, "target_logical_bounds", namespace)
    iou = extracted_function(locator, "_iou", {"target_logical_bounds": logical})
    physical = SimpleNamespace(coordinate_space="physical", x=20, y=20, width=20, height=20)
    logical_target = SimpleNamespace(coordinate_space="logical", x=10, y=10, width=10, height=10)
    return logical(physical, scale_factor=2) == (10, 10, 10, 10) and iou(physical, logical_target, scale_factor=2) == 1.0


def check_17(sources: dict[str, str]) -> bool:
    ordering = source_at(sources, "ordering.py")
    sorter = source_at(sources, "sorter.py")
    if not has(ordering, "def solve_order", "OrderingOutcome.REPAIRED") or not has(sorter, "from funcsort.ordering import", "solve_order("):
        return False
    namespace: dict[str, Any] = {}
    exec(compile(ordering, "<ordering-evaluator>", "exec"), namespace)
    Statement = namespace["Statement"]; Problem = namespace["OrderingProblem"]
    provider = Statement(0, frozenset({"x"}), frozenset())
    consumer = Statement(1, frozenset(), frozenset({"x"}))
    result = namespace["solve_order"](Problem((), (provider, consumer), (0, 1), (1, 0)))
    return result.order == (0, 1) and result.outcome is namespace["OrderingOutcome"].REPAIRED


def check_18(sources: dict[str, str]) -> bool:
    symbols = source_at(sources, "symbols.py")
    writer = source_at(sources, "timescale_writer.py")
    if not has(symbols, "def validate_instrument_type", "ALLOWED_INSTRUMENT_TYPES") or writer.count("validate_instrument_type(instrument_type)") < 4:
        return False
    function = extracted_function(symbols, "validate_instrument_type", {"ALLOWED_INSTRUMENT_TYPES": frozenset({"spot"})})
    function("spot")
    try:
        function("unknown")
    except ValueError:
        return True
    return False


CHECKERS: dict[int, Callable[[dict[str, str]], bool]] = {
    1: check_01, 2: check_02, 3: check_03, 4: check_04, 5: check_05, 6: check_06,
    7: check_07, 8: check_08, 9: check_09, 10: check_10, 11: check_11, 12: check_12,
    13: check_13, 14: check_14, 15: check_15, 16: check_16, 17: check_17, 18: check_18,
}


MUTATIONS: dict[int, tuple[str, str, str]] = {
    1: ("storage.py", 'manifest.get("size_bytes")', 'manifest.get_disabled("size_bytes")'),
    2: ("tsfresh.py", ".drop_duplicates().to_numpy()", ".unique()"),
    3: ("conftest.py", "PARAMS.get(parameter_code, {}).get", "PARAMS[parameter_code].get"),
    4: ("cli.py", "def _resolved_path(explicit, configured, fallback):", "def _resolved_path_disabled(explicit, configured, fallback):"),
    5: ("style_json.py", 'stashed_fill = builder.get("fillColorSaved")', "stashed_fill = None"),
    6: ("auxfuncs.py", "from .capi_maps import f2cmap_all", "f2cmap_all = {}"),
    7: ("range_frame.py", "resolve_sort_keys(", "resolve_sort_keys_disabled("),
    8: ("worker.py", "centred_values = finite_values - origin", "centred_values = finite_values"),
    9: ("ingest.py", "def _split_oversized", "def _split_oversized_disabled"),
    10: ("utils.py", "if force_hf_offline_active():", "if False:"),
    11: ("fallback.py", "except ImportError as e:", "except RuntimeError as e:"),
    12: ("ed_push.py", 'm = re.search(r"^#\\s+(.+)$", writeup, re.M)', "m = None"),
    13: ("mgmt_handlers.py", "if status_code == 404 and condition == ERROR_CODE_MESSAGE_NOT_FOUND:", "if False:"),
    14: ("source_code_manager.py", "if clone_result != 0:", "if False:"),
    15: ("utils.py", "self._credentials[bucket] = credentials", "self._credentials_disabled = credentials"),
    16: ("locator.py", "from openframe.recognize.coords import target_logical_bounds", "from openframe.recognize.coords import target_uses_physical_pixels"),
    17: ("sorter.py", "from funcsort.ordering import (", "from funcsort.ordering_disabled import ("),
    18: ("symbols.py", "def validate_instrument_type", "def validate_instrument_type_disabled"),
}


def mutate_required_mechanism(index: int, sources: dict[str, str]) -> dict[str, str]:
    suffix, needle, replacement = MUTATIONS[index]
    result = dict(sources)
    paths = [path for path in result if path.endswith(suffix)]
    if len(paths) != 1 or needle not in result[paths[0]]:
        raise AssertionError(f"mutation anchor missing for task {index}")
    result[paths[0]] = result[paths[0]].replace(needle, replacement, 1)
    return result


def archive_sources(row: dict[str, Any], role: str) -> dict[str, str]:
    receipt = mapping(mapping(row.get("archives")).get(role))
    archive_path = resolve(str(receipt.get("path") or ""))
    if sha256_file(archive_path) != str(receipt.get("sha256") or ""):
        raise AssertionError(f"archive hash invalid:{role}")
    root = str(receipt.get("root") or "")
    expected_paths = {str(value) for value in row.get("selected_source_paths") or []}
    result: dict[str, str] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                not member.isfile() or member.issym() or member.islnk()
                or pure.is_absolute() or ".." in pure.parts
                or not member.name.startswith(root + "/")
            ):
                raise AssertionError("unsafe archive member")
            logical = member.name[len(root) + 1 :]
            if logical not in expected_paths:
                continue
            handle = archive.extractfile(member)
            result[logical] = (handle.read() if handle else b"").decode("utf-8")
    return result


def zero_counters() -> dict[str, int]:
    return {
        "parent_target_evaluator_executions": 0,
        "evaluator_control_executions": 0,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value or [] if isinstance(row, dict)]


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "task_count": report.get("task_count"),
        "green_task_count": report.get("green_task_count"),
        "panel_admitted_for_task_packet_materialization": report.get("panel_admitted_for_task_packet_materialization"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
