"""Dataset loaders for real-code benchmark graduation."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import pickle
import re
import zlib
from pathlib import Path
from typing import Any

from real_code_benchmark_constants import *  # noqa: F403
from real_code_benchmark_support import (
    function_name,
    function_name_from_tests,
    mbpp_visible_signature,
    read_jsonl,
    rel_or_abs,
    rotate,
    safe_name,
    tags_from_text,
)


__all__ = [
    "load_cases",
    "loader_manifest_for",
    "load_humaneval_cases",
    "load_evalplus_cases",
    "evalplus_dataset_path",
    "stream_jsonl_file",
    "evalplus_inputs",
    "stable_int",
    "evalplus_tests_from_canonical",
    "safe_public_literal",
    "deep_close_helper_source",
    "load_mbpp_cases",
    "load_bigcodebench_cases",
    "bigcodebench_dataset_paths",
    "runnable_unittest_source",
    "load_livecodebench_cases",
    "livecodebench_visible_prompt",
    "expanded_livecodebench_dataset_paths",
    "livecodebench_dataset_paths",
    "load_parquet_records",
    "livecodebench_test_case_dicts",
    "livecodebench_function_tests",
    "livecodebench_stdin_tests",
    "livecodebench_args",
    "livecodebench_arg_names",
    "livecodebench_single_arg_should_receive_list",
    "parse_jsonish",
    "parse_literalish",
    "load_loader_manifest_cases",
]

def load_cases(
    card_id: str,
    source_id: str,
    source_path: Path,
    seed: int,
    max_cases: int,
) -> tuple[list[dict[str, Any]], str, str]:
    if source_id == "evalplus":
        tasks = load_evalplus_cases(card_id, source_path, seed, max_cases)
        if tasks:
            return tasks, "public_benchmark_task_regression", "evalplus_public_code_calibration_pass_rate_not_student_model_score"
    if source_id == "human_eval":
        tasks = load_humaneval_cases(card_id, source_path, seed, max_cases)
        if tasks:
            return tasks, "public_benchmark_task_regression", "public_code_calibration_pass_rate_not_student_model_score"
    if source_id == "mbpp":
        tasks = load_mbpp_cases(card_id, source_path, seed, max_cases)
        if tasks:
            return tasks, "public_benchmark_task_regression", "public_code_calibration_pass_rate_not_student_model_score"
    if source_id == "bigcodebench":
        tasks = load_bigcodebench_cases(card_id, source_path, seed, max_cases)
        if tasks:
            return tasks, "public_benchmark_task_regression", "bigcodebench_public_code_calibration_pass_rate_not_student_model_score"
    if source_id == "livecodebench":
        tasks = load_livecodebench_cases(card_id, source_path, seed, max_cases)
        if tasks:
            return tasks, "public_benchmark_task_regression", "livecodebench_public_code_calibration_pass_rate_not_student_model_score"
    loader_manifest = loader_manifest_for(card_id, seed)
    tasks = load_loader_manifest_cases(card_id, loader_manifest, max_cases)
    if tasks:
        return tasks, "public_loader_regression", "public_loader_regression_not_benchmark_score"
    return [], "source_staged_no_local_task_cases", "blocked_not_scored"


def loader_manifest_for(card_id: str, seed: int) -> Path:
    exact = ROOT / "data" / "public_code_benchmark_manifests" / f"{safe_name(card_id)}_seed{seed}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(
        (ROOT / "data" / "public_code_benchmark_manifests").glob(f"{safe_name(card_id)}_seed*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else exact


def load_humaneval_cases(card_id: str, source_path: Path, seed: int, max_cases: int) -> list[dict[str, Any]]:
    candidates = [
        source_path / "data" / "HumanEval.jsonl.gz",
        source_path / "HumanEval.jsonl.gz",
        Path("D:/ProjectTheseus/resource_pantry/git/human_eval/data/HumanEval.jsonl.gz"),
    ]
    dataset = next((path for path in candidates if path.exists()), None)
    if not dataset:
        return []
    rows = []
    with gzip.open(dataset, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(raw.get("prompt") or "")
            entry = str(raw.get("entry_point") or function_name(prompt))
            test = str(raw.get("test") or "")
            if not prompt or not entry or not test:
                continue
            rows.append(
                {
                    "task_id": f"{card_id}_{safe_name(str(raw.get('task_id') or entry))}",
                    "source_task_id": raw.get("task_id"),
                    "case_type": "public_humaneval_task",
                    "prompt": prompt,
                    "entry_point": entry,
                    "tests": f"{test}\ncheck({entry})\n",
                    "tags": tags_from_text(prompt),
                    "provenance": {
                        "dataset": rel_or_abs(dataset),
                        "source_task_id": raw.get("task_id"),
                        "canonical_solution_seen_by_solver": False,
                        "copied_public_benchmark_item_chars": len(prompt) + len(test),
                    },
                }
            )
    return rotate(rows, seed)[:max_cases]


def load_evalplus_cases(card_id: str, source_path: Path, seed: int, max_cases: int) -> list[dict[str, Any]]:
    dataset = evalplus_dataset_path(source_path)
    if not dataset:
        return []
    rows = []
    for raw in stream_jsonl_file(dataset):
        prompt = str(raw.get("prompt") or "")
        entry = str(raw.get("entry_point") or function_name(prompt))
        canonical = str(raw.get("canonical_solution") or "")
        task_id = str(raw.get("task_id") or entry)
        if not prompt or not entry or not canonical or not task_id:
            continue
        inputs = evalplus_inputs(raw, seed=seed, task_id=task_id)
        tests = evalplus_tests_from_canonical(prompt, canonical, entry, inputs, atol=raw.get("atol"))
        if not tests:
            public_test_source = str(raw.get("test") or "")
            if public_test_source and "def check" in public_test_source:
                tests = f"{public_test_source}\ncheck({entry})\n"
        if not tests:
            continue
        rows.append(
            {
                "task_id": f"{card_id}_{safe_name(task_id)}",
                "source_task_id": task_id,
                "case_type": "public_evalplus_task",
                "prompt": prompt,
                "entry_point": entry,
                "tests": tests,
                "tags": tags_from_text(prompt) + ["evalplus"],
                "provenance": {
                    "dataset": rel_or_abs(dataset),
                    "source_task_id": task_id,
                    "canonical_solution_seen_by_solver": False,
                    "canonical_solution_used_by_scorer": True,
                    "public_tests_exported_to_generator": False,
                    "reference_outputs_materialized_for_sandbox_only": True,
                    "copied_public_benchmark_item_chars": len(prompt) + len(tests),
                },
            }
        )
    return rotate(rows, seed)[:max_cases]


def evalplus_dataset_path(source_path: Path) -> Path | None:
    candidates = [
        Path("D:/ProjectTheseus/resource_pantry/datasets/evalplus/HumanEvalPlus-v0.1.10.jsonl"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/evalplus/HumanEvalPlus-v0.1.10.jsonl.gz"),
        source_path / "HumanEvalPlus-v0.1.10.jsonl",
        source_path / "HumanEvalPlus-v0.1.10.jsonl.gz",
        source_path / "evalplus" / "data" / "HumanEvalPlus-v0.1.10.jsonl",
        source_path / "evalplus" / "data" / "HumanEvalPlus-v0.1.10.jsonl.gz",
    ]
    return next((path for path in candidates if path.exists()), None)


def stream_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict):
                    rows.append(raw)
    except OSError:
        return []
    return rows


def evalplus_inputs(raw: dict[str, Any], *, seed: int, task_id: str, max_inputs: int = 8) -> list[Any]:
    inputs: list[Any] = []
    for key in ("base_input", "plus_input"):
        value = raw.get(key)
        if isinstance(value, list):
            inputs.extend(value)
    rotated = rotate(inputs, seed + stable_int(task_id))
    return rotated[: max(1, max_inputs)]


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def evalplus_tests_from_canonical(
    prompt: str,
    canonical_solution: str,
    entry_point: str,
    inputs: list[Any],
    *,
    atol: Any,
) -> str:
    namespace: dict[str, Any] = {}
    try:
        exec(PUBLIC_TEST_RUNTIME_PRELUDE + prompt + "\n" + canonical_solution, namespace)
    except Exception:
        return ""
    fn = namespace.get(entry_point)
    if not callable(fn):
        return ""
    assertions = [deep_close_helper_source()]
    tolerance = float(atol) if isinstance(atol, (int, float)) else 1e-6
    for raw_args in inputs:
        args = list(raw_args) if isinstance(raw_args, (list, tuple)) else [raw_args]
        try:
            expected = fn(*args)
        except Exception:
            continue
        args_repr = safe_public_literal(args)
        expected_repr = safe_public_literal(expected)
        if args_repr is None or expected_repr is None:
            continue
        assertions.append(
            f"assert _theseus_deep_close({entry_point}(*{args_repr}), {expected_repr}, {tolerance!r})"
        )
    return "\n".join(assertions) + "\n" if len(assertions) > 1 else ""


def safe_public_literal(value: Any, *, max_chars: int = MAX_PUBLIC_LITERAL_REPR_CHARS) -> str | None:
    try:
        text = repr(value)
    except ValueError:
        return None
    if len(text) > max_chars:
        return None
    return text


def deep_close_helper_source() -> str:
    return """
def _theseus_deep_close(left, right, atol=1e-6):
    if isinstance(left, float) or isinstance(right, float):
        return abs(left - right) <= atol
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_theseus_deep_close(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_theseus_deep_close(left[k], right[k], atol) for k in left)
    return left == right
""".strip()


def load_mbpp_cases(card_id: str, source_path: Path, seed: int, max_cases: int) -> list[dict[str, Any]]:
    candidates = [
        source_path / "sanitized-mbpp.json",
        source_path / "mbpp" / "sanitized-mbpp.json",
        Path("D:/ProjectTheseus/resource_pantry/git/evalplus/.cache/sanitized-mbpp.json"),
    ]
    dataset = next((path for path in candidates if path.exists()), None)
    if not dataset:
        return []
    try:
        raw_rows = json.loads(dataset.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = []
    for raw in raw_rows if isinstance(raw_rows, list) else []:
        tests = raw.get("test_list")
        code = str(raw.get("code") or "")
        task_id = str(raw.get("task_id") or "")
        if not isinstance(tests, list) or not tests or not code:
            continue
        entry = function_name(code)
        task_text = str(raw.get("prompt") or raw.get("text") or "").replace('"""', "'''")
        if not task_text.strip():
            continue
        entry = function_name_from_tests(tests, preferred=entry) or entry
        prompt = f"{mbpp_visible_signature(entry or 'solve', task_text)}\n    \"\"\"{task_text}\"\"\"\n"
        rows.append(
            {
                "task_id": f"{card_id}_mbpp_{safe_name(task_id)}",
                "source_task_id": task_id,
                "case_type": "public_mbpp_task",
                "prompt": prompt,
                "entry_point": entry or function_name(prompt),
                "tests": "\n".join(str(item) for item in tests) + "\n",
                "tags": tags_from_text(task_text),
                "provenance": {
                    "dataset": rel_or_abs(dataset),
                    "source_task_id": task_id,
                    "canonical_solution_seen_by_solver": False,
                    "reference_implementation_hidden_from_solver": True,
                    "copied_public_benchmark_item_chars": len(prompt) + sum(len(str(item)) for item in tests),
                },
            }
        )
    return rotate(rows, seed)[:max_cases]


def load_bigcodebench_cases(card_id: str, source_path: Path, seed: int, max_cases: int) -> list[dict[str, Any]]:
    dataset = next((path for path in bigcodebench_dataset_paths(source_path) if path.exists()), None)
    if not dataset:
        return []
    raw_rows = stream_jsonl_file(dataset)
    rows = []
    for raw in raw_rows:
        task_id = str(raw.get("task_id") or raw.get("id") or "")
        prompt = str(raw.get("complete_prompt") or raw.get("code_prompt") or "")
        entry = str(raw.get("entry_point") or function_name(prompt))
        tests = str(raw.get("test") or "")
        if not task_id or not prompt or not entry or not tests or "def " not in prompt:
            continue
        rows.append(
            {
                "task_id": f"{card_id}_{safe_name(task_id)}",
                "source_task_id": task_id,
                "case_type": "public_bigcodebench_task",
                "prompt": prompt,
                "entry_point": entry,
                "tests": runnable_unittest_source(tests),
                "tags": tags_from_text(prompt) + ["bigcodebench"],
                "provenance": {
                    "dataset": rel_or_abs(dataset),
                    "source_task_id": task_id,
                    "canonical_solution_seen_by_solver": False,
                    "public_tests_exported_to_generator": False,
                    "reference_tests_used_by_scorer_only": True,
                    "copied_public_benchmark_item_chars": len(prompt) + len(tests),
                },
            }
        )
    return rotate(rows, seed)[:max_cases]


def bigcodebench_dataset_paths(source_path: Path) -> list[Path]:
    return [
        Path("D:/ProjectTheseus/resource_pantry/datasets/bigcodebench/BigCodeBench-v0.1.4.jsonl"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/bigcodebench/BigCodeBench-v0.1.4.jsonl.gz"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/bigcodebench/BigCodeBench.jsonl"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/bigcodebench/BigCodeBench.jsonl.gz"),
        source_path / "BigCodeBench-v0.1.4.jsonl",
        source_path / "BigCodeBench-v0.1.4.jsonl.gz",
        source_path / "BigCodeBench.jsonl",
        source_path / "BigCodeBench.jsonl.gz",
    ]


def runnable_unittest_source(test_source: str) -> str:
    suffix = "\n\nif __name__ == '__main__':\n    import unittest\n    unittest.main()\n"
    return test_source if "unittest.main" in test_source else test_source + suffix


def load_livecodebench_cases(card_id: str, source_path: Path, seed: int, max_cases: int) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for path in expanded_livecodebench_dataset_paths(source_path):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            raw_rows = load_parquet_records(path)
        else:
            raw_rows = stream_jsonl_file(path)
        if raw_rows:
            dataset = path
            break
    else:
        return []
    rows = []
    for raw in raw_rows:
        metadata = parse_jsonish(raw.get("metadata"), default={})
        if not isinstance(metadata, dict):
            metadata = {}
        func_name = str(metadata.get("func_name") or "")
        starter = str(raw.get("starter_code") or "")
        arg_names = livecodebench_arg_names(starter, func_name)
        public_tests = livecodebench_test_case_dicts(raw.get("public_test_cases"))
        private_tests = livecodebench_test_case_dicts(raw.get("private_test_cases"))
        question = str(raw.get("question_content") or raw.get("question_title") or "")
        if func_name:
            tests = livecodebench_function_tests(func_name, public_tests + private_tests, arg_names=arg_names)
            prompt = livecodebench_visible_prompt(starter, func_name, question)
            case_type = "public_livecodebench_task"
            functional_only_adapter = True
            stdin_adapter = False
        else:
            func_name = "solve"
            tests = livecodebench_stdin_tests(func_name, public_tests + private_tests)
            prompt = livecodebench_visible_stdin_prompt(question)
            case_type = "public_livecodebench_stdin_task"
            functional_only_adapter = False
            stdin_adapter = True
        if not tests:
            continue
        task_id = str(raw.get("question_id") or raw.get("contest_id") or func_name)
        rows.append(
            {
                "task_id": f"{card_id}_{safe_name(task_id)}",
                "source_task_id": task_id,
                "case_type": case_type,
                "prompt": prompt,
                "entry_point": func_name,
                "tests": tests,
                "tags": tags_from_text(question) + ["livecodebench", str(raw.get("difficulty") or "")],
                "provenance": {
                    "dataset": rel_or_abs(dataset),
                    "source_task_id": task_id,
                    "platform": raw.get("platform"),
                    "contest_date": raw.get("contest_date"),
                    "canonical_solution_seen_by_solver": False,
                    "public_tests_exported_to_generator": False,
                    "reference_tests_used_by_scorer_only": True,
                    "functional_only_adapter": functional_only_adapter,
                    "stdin_adapter": stdin_adapter,
                    "copied_public_benchmark_item_chars": len(prompt) + len(tests),
                },
            }
        )
    return rotate(rows, seed)[:max_cases]


def livecodebench_visible_prompt(starter: str, func_name: str, question: str) -> str:
    doc = question[:1200].replace(chr(34), chr(39))
    match = re.search(rf"def\s+{re.escape(func_name)}\s*\(([^)]*)\)", starter)
    if match:
        args = []
        for raw in match.group(1).split(","):
            item = raw.strip()
            if not item or item == "self":
                continue
            args.append(item)
        return f"def {func_name}({', '.join(args)}):\n    \"\"\"{doc}\"\"\"\n"
    if "def " in starter and "class " not in starter:
        return starter
    return f"def {func_name}(*args):\n    \"\"\"{doc}\"\"\"\n"


def livecodebench_visible_stdin_prompt(question: str) -> str:
    doc = question[:1200].replace(chr(34), chr(39))
    return f"def solve(input_data):\n    \"\"\"{doc}\n\n    Return the output text for the given stdin input string.\"\"\"\n"


def expanded_livecodebench_dataset_paths(source_path: Path) -> list[Path]:
    paths: list[Path] = []
    for path in livecodebench_dataset_paths(source_path):
        if path.is_dir():
            paths.extend(sorted(path.glob("test-*.parquet")))
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def livecodebench_dataset_paths(source_path: Path) -> list[Path]:
    return [
        Path("D:/ProjectTheseus/resource_pantry/datasets/livecodebench/release_v2"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/livecodebench/release_v1"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/livecodebench/code_generation_lite_release_v2.jsonl"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/livecodebench/code_generation_lite_release_v1.jsonl"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/livecodebench/code_generation_lite_release_v2.parquet"),
        Path("D:/ProjectTheseus/resource_pantry/datasets/livecodebench/code_generation_lite_release_v1.parquet"),
        source_path / "release_v2",
        source_path / "release_v1",
        source_path / "code_generation_lite_release_v2.jsonl",
        source_path / "code_generation_lite_release_v1.jsonl",
        source_path / "code_generation_lite_release_v2.parquet",
        source_path / "code_generation_lite_release_v1.parquet",
        source_path / "code_generation_lite_release_v6.jsonl",
        source_path / "code_generation_lite_release_v5.jsonl",
        source_path / "code_generation_lite_release_v4.jsonl",
        source_path / "code_generation_lite_release_v3.jsonl",
        source_path / "test6.jsonl",
        source_path / "test5.jsonl",
        source_path / "test4.jsonl",
        source_path / "test3.jsonl",
        source_path / "test2.jsonl",
        source_path / "test.jsonl",
    ]


def load_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []
    try:
        return pd.read_parquet(path).to_dict(orient="records")
    except Exception:
        return []


def livecodebench_test_case_dicts(value: Any) -> list[dict[str, Any]]:
    decoded = parse_jsonish(value, default=[])
    if not isinstance(decoded, list) and isinstance(value, str):
        try:
            decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))
        except Exception:
            decoded = []
    return [item for item in decoded if isinstance(item, dict)]


def livecodebench_function_tests(
    entry: str,
    tests: list[dict[str, Any]],
    max_tests: int = 10,
    *,
    arg_names: list[str] | None = None,
) -> str:
    assertions = [deep_close_helper_source()]
    for test in tests:
        if str(test.get("testtype") or "").lower() != "functional":
            continue
        args = livecodebench_args(test.get("input"), arg_names=arg_names or [])
        if args is None:
            continue
        expected = parse_jsonish(test.get("output"), default=test.get("output"))
        assertions.append(f"assert _theseus_deep_close({entry}(*{args!r}), {expected!r})")
        if len(assertions) > max_tests:
            break
    return "\n".join(assertions) + "\n" if len(assertions) > 1 else ""


def livecodebench_stdin_tests(entry: str, tests: list[dict[str, Any]], max_tests: int = 10) -> str:
    assertions = [
        """
def _theseus_lcb_call(fn, input_text):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(input_text)
    return buf.getvalue() if result is None else result

def _theseus_lcb_output_close(left, right):
    return str(left).strip() == str(right).strip()
""".strip()
    ]
    for test in tests:
        if str(test.get("testtype") or "").lower() != "stdin":
            continue
        input_text = str(test.get("input") or "")
        expected = str(test.get("output") or "")
        assertions.append(f"assert _theseus_lcb_output_close(_theseus_lcb_call({entry}, {input_text!r}), {expected!r})")
        if len(assertions) > max_tests:
            break
    return "\n".join(assertions) + "\n" if len(assertions) > 1 else ""


def livecodebench_args(value: Any, *, arg_names: list[str] | None = None) -> list[Any] | None:
    parsed = parse_jsonish(value, default=None)
    if parsed is None:
        parsed = parse_literalish(value)
    if parsed is None:
        return None
    arg_names = arg_names or []
    if isinstance(parsed, dict):
        if "args" in parsed and isinstance(parsed["args"], list):
            return parsed["args"]
        if "input" in parsed:
            nested = livecodebench_args(parsed["input"], arg_names=arg_names)
            return nested
        return [parsed]
    if isinstance(parsed, list):
        if len(arg_names) == 1 and livecodebench_single_arg_should_receive_list(arg_names[0], parsed):
            return [parsed]
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return [parsed]


def livecodebench_arg_names(starter: str, func_name: str) -> list[str]:
    match = re.search(rf"def\s+{re.escape(func_name)}\s*\(([^)]*)\)", starter)
    if not match:
        return []
    names = []
    for raw in match.group(1).split(","):
        item = raw.strip()
        if not item or item == "self" or item.startswith("*"):
            continue
        name = item.split(":", 1)[0].split("=", 1)[0].strip()
        if name:
            names.append(name)
    return names


def livecodebench_single_arg_should_receive_list(arg_name: str, parsed: list[Any]) -> bool:
    name = arg_name.lower()
    collection_names = {
        "arr",
        "array",
        "edges",
        "grid",
        "intervals",
        "items",
        "mat",
        "matrix",
        "nums",
        "queries",
        "usage",
        "usagelimits",
        "values",
        "words",
    }
    scalar_names = {"k", "m", "n", "num", "number", "purchaseamount", "x", "y"}
    if name in scalar_names:
        return False
    if name in collection_names or name.endswith("s"):
        return True
    return len(parsed) > 1 and not all(isinstance(item, (str, int, float, bool, type(None))) for item in parsed)


def parse_jsonish(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict, int, float, bool)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def parse_literalish(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def load_loader_manifest_cases(card_id: str, manifest: Path, max_cases: int) -> list[dict[str, Any]]:
    rows = []
    for raw in read_jsonl(manifest):
        if raw.get("case_type") not in {"multi_stream_python_code_repair", "python_code_repair"}:
            continue
        task_id = str(raw.get("task_id") or raw.get("case_id") or "")
        buggy = str(raw.get("buggy") or "")
        tests = str(raw.get("tests") or "")
        if not task_id or not buggy or not tests:
            continue
        rows.append(
            {
                "task_id": task_id,
                "source_task_id": raw.get("case_id"),
                "case_type": "public_loader_regression",
                "prompt": buggy,
                "entry_point": function_name(buggy),
                "tests": tests,
                "tags": [str(tag) for tag in raw.get("tags", [])] if isinstance(raw.get("tags"), list) else [],
                "repair_templates": raw.get("repair_templates") if isinstance(raw.get("repair_templates"), dict) else {},
                "provenance": raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {},
            }
        )
    return rows[:max_cases]
