"""Private long-horizon programming curriculum for Theseus.

Function completion is too small to prove durable programming learning. This
script builds private SWE-style repo-repair tasks with hidden tests and STS
trace rows. Public benchmark tasks are not copied into this curriculum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_OUT = Path("D:/ProjectTheseus/training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl")
DEFAULT_STS_OUT = Path("D:/ProjectTheseus/training_data/long_horizon_programming/sts/repo_repair_sts_rows.jsonl")
DEFAULT_REPORT = ROOT / "reports" / "long_horizon_programming_curriculum.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "long_horizon_programming_curriculum.md"


TASK_TEMPLATES = [
    {
        "category": "off_by_one_loop",
        "bug": "Loop excludes the final item.",
        "files": {
            "app/series.py": "def inclusive_sum(n):\n    total = 0\n    for value in range(n):\n        total += value\n    return total\n",
            "tests/test_series.py": "from app.series import inclusive_sum\n\n\ndef test_visible_small():\n    assert inclusive_sum(3) == 6\n",
        },
        "hidden_tests": "from app.series import inclusive_sum\n\n\ndef test_zero_and_one():\n    assert inclusive_sum(0) == 0\n    assert inclusive_sum(1) == 1\n",
        "patch": "Change range(n) to range(n + 1).",
        "expected_files": {
            "app/series.py": "def inclusive_sum(n):\n    total = 0\n    for value in range(n + 1):\n        total += value\n    return total\n",
        },
    },
    {
        "category": "string_parsing_edge",
        "bug": "Parser rejects signed integer tokens.",
        "files": {
            "app/parser.py": "def parse_ints(text):\n    return [int(part) for part in text.split() if part.isdigit()]\n",
            "tests/test_parser.py": "from app.parser import parse_ints\n\n\ndef test_visible_positive():\n    assert parse_ints('1 x 2') == [1, 2]\n",
        },
        "hidden_tests": "from app.parser import parse_ints\n\n\ndef test_signed_values():\n    assert parse_ints('-1 0 +2 nope') == [-1, 0, 2]\n",
        "patch": "Accept optional leading + or - before digit checks.",
        "expected_files": {
            "app/parser.py": "def parse_ints(text):\n    out = []\n    for part in text.split():\n        normalized = part.lstrip('+-')\n        if normalized.isdigit():\n            out.append(int(part))\n    return out\n",
        },
    },
    {
        "category": "mutable_default_state",
        "bug": "Function keeps state across calls through a mutable default.",
        "files": {
            "app/cache.py": "def append_seen(value, seen=[]):\n    seen.append(value)\n    return seen\n",
            "tests/test_cache.py": "from app.cache import append_seen\n\n\ndef test_visible_single_call():\n    assert append_seen('a')[-1] == 'a'\n",
        },
        "hidden_tests": "from app.cache import append_seen\n\n\ndef test_calls_are_independent():\n    assert append_seen(1) == [1]\n    assert append_seen(2) == [2]\n",
        "patch": "Use None as the default and allocate a new list inside the function.",
        "expected_files": {
            "app/cache.py": "def append_seen(value, seen=None):\n    if seen is None:\n        seen = []\n    seen.append(value)\n    return seen\n",
        },
    },
    {
        "category": "schema_validation",
        "bug": "Validator accepts records missing required keys.",
        "files": {
            "app/schema.py": "def has_required(record, required):\n    return any(key in record for key in required)\n",
            "tests/test_schema.py": "from app.schema import has_required\n\n\ndef test_visible_one_key():\n    assert has_required({'a': 1}, ['a']) is True\n",
        },
        "hidden_tests": "from app.schema import has_required\n\n\ndef test_all_keys_required():\n    assert has_required({'a': 1}, ['a', 'b']) is False\n    assert has_required({}, []) is True\n",
        "patch": "Require all keys to be present, not any key.",
        "expected_files": {
            "app/schema.py": "def has_required(record, required):\n    return all(key in record for key in required)\n",
        },
    },
    {
        "category": "path_normalization",
        "bug": "Path join leaves repeated separators and current-directory markers.",
        "files": {
            "app/paths.py": "def clean_join(left, right):\n    return left + '/' + right\n",
            "tests/test_paths.py": "from app.paths import clean_join\n\n\ndef test_visible_join():\n    assert clean_join('a', 'b') == 'a/b'\n",
        },
        "hidden_tests": "from app.paths import clean_join\n\n\ndef test_extra_separators():\n    assert clean_join('a/', './b') == 'a/b'\n",
        "patch": "Strip separators and ignore a leading './' segment.",
        "expected_files": {
            "app/paths.py": "def clean_join(left, right):\n    right = right[2:] if right.startswith('./') else right\n    return left.rstrip('/') + '/' + right.lstrip('/')\n",
        },
    },
    {
        "category": "cli_argument_shape",
        "bug": "CLI parser treats a missing optional flag as truthy text.",
        "files": {
            "app/cli.py": "def parse_verbose(args):\n    return '--verbose' or '-v' in args\n",
            "tests/test_cli.py": "from app.cli import parse_verbose\n\n\ndef test_visible_verbose():\n    assert parse_verbose(['--verbose']) is True\n",
        },
        "hidden_tests": "from app.cli import parse_verbose\n\n\ndef test_missing_verbose():\n    assert parse_verbose([]) is False\n    assert parse_verbose(['-v']) is True\n",
        "patch": "Check membership for both verbose flags explicitly.",
        "expected_files": {
            "app/cli.py": "def parse_verbose(args):\n    return '--verbose' in args or '-v' in args\n",
        },
    },
    {
        "category": "csv_header_validation",
        "bug": "CSV loader silently accepts rows missing required columns.",
        "files": {
            "app/csv_loader.py": "import csv\n\n\ndef load_named_scores(path):\n    with open(path, newline='', encoding='utf-8') as handle:\n        return [row for row in csv.DictReader(handle)]\n",
            "tests/test_csv_loader.py": "from app.csv_loader import load_named_scores\nfrom pathlib import Path\nimport tempfile\n\n\ndef test_visible_reads_rows():\n    with tempfile.TemporaryDirectory() as tmp:\n        p = Path(tmp) / 'scores.csv'\n        p.write_text('name,score\\na,3\\n', encoding='utf-8')\n        assert load_named_scores(p)[0]['name'] == 'a'\n",
        },
        "hidden_tests": "from app.csv_loader import load_named_scores\nfrom pathlib import Path\nimport tempfile\n\n\ndef test_missing_score_header_rejected():\n    with tempfile.TemporaryDirectory() as tmp:\n        p = Path(tmp) / 'scores.csv'\n        p.write_text('name\\na\\n', encoding='utf-8')\n        try:\n            load_named_scores(p)\n        except ValueError:\n            return\n        raise AssertionError('missing score header should be rejected')\n",
        "patch": "Validate that DictReader exposes both required headers before consuming rows.",
        "expected_files": {
            "app/csv_loader.py": "import csv\n\n\ndef load_named_scores(path):\n    with open(path, newline='', encoding='utf-8') as handle:\n        reader = csv.DictReader(handle)\n        required = {'name', 'score'}\n        if not required.issubset(set(reader.fieldnames or [])):\n            raise ValueError('missing required csv columns')\n        return [row for row in reader]\n",
        },
    },
    {
        "category": "json_nested_update",
        "bug": "Nested JSON update overwrites unrelated keys.",
        "files": {
            "app/json_patch.py": "def set_user_flag(payload, user_id, flag):\n    payload['users'] = {user_id: {'flag': flag}}\n    return payload\n",
            "tests/test_json_patch.py": "from app.json_patch import set_user_flag\n\n\ndef test_visible_sets_flag():\n    assert set_user_flag({'users': {}}, 'u1', True)['users']['u1']['flag'] is True\n",
        },
        "hidden_tests": "from app.json_patch import set_user_flag\n\n\ndef test_preserves_existing_users_and_fields():\n    payload = {'users': {'u1': {'name': 'Ada'}, 'u2': {'flag': False}}}\n    out = set_user_flag(payload, 'u1', True)\n    assert out['users']['u1']['name'] == 'Ada'\n    assert out['users']['u1']['flag'] is True\n    assert out['users']['u2']['flag'] is False\n",
        "patch": "Use setdefault for nested dictionaries and mutate only the requested field.",
        "expected_files": {
            "app/json_patch.py": "def set_user_flag(payload, user_id, flag):\n    users = payload.setdefault('users', {})\n    record = users.setdefault(user_id, {})\n    record['flag'] = flag\n    return payload\n",
        },
    },
    {
        "category": "multi_file_import_contract",
        "bug": "Service imports a stale helper name after a module split.",
        "files": {
            "app/helpers.py": "def normalize_name(value):\n    return ' '.join(str(value).split()).title()\n",
            "app/service.py": "from app.helpers import clean_name\n\n\ndef display_user(record):\n    return clean_name(record['name'])\n",
            "tests/test_service.py": "from app.service import display_user\n\n\ndef test_visible_display():\n    assert display_user({'name': 'ada'}) == 'Ada'\n",
        },
        "hidden_tests": "from app.service import display_user\n\n\ndef test_collapses_internal_spaces():\n    assert display_user({'name': ' ada   lovelace '}) == 'Ada Lovelace'\n",
        "patch": "Import the actual helper function that performs normalization.",
        "expected_files": {
            "app/service.py": "from app.helpers import normalize_name\n\n\ndef display_user(record):\n    return normalize_name(record['name'])\n",
        },
    },
    {
        "category": "datetime_boundary",
        "bug": "Date window check excludes the ending day.",
        "files": {
            "app/dates.py": "from datetime import date\n\n\ndef within_window(value, start, end):\n    return start <= value < end\n",
            "tests/test_dates.py": "from app.dates import within_window\nfrom datetime import date\n\n\ndef test_visible_inside():\n    assert within_window(date(2025, 1, 2), date(2025, 1, 1), date(2025, 1, 3))\n",
        },
        "hidden_tests": "from app.dates import within_window\nfrom datetime import date\n\n\ndef test_end_day_included():\n    assert within_window(date(2025, 1, 3), date(2025, 1, 1), date(2025, 1, 3))\n    assert not within_window(date(2025, 1, 4), date(2025, 1, 1), date(2025, 1, 3))\n",
        "patch": "Use an inclusive upper bound for the business date window.",
        "expected_files": {
            "app/dates.py": "from datetime import date\n\n\ndef within_window(value, start, end):\n    return start <= value <= end\n",
        },
    },
    {
        "category": "path_traversal_guard",
        "bug": "Path sanitizer only checks string prefixes before normalization.",
        "files": {
            "app/safe_paths.py": "from pathlib import Path\n\n\ndef safe_child(base, user_path):\n    candidate = Path(base) / user_path\n    if not str(candidate).startswith(str(base)):\n        raise ValueError('outside base')\n    return candidate\n",
            "tests/test_safe_paths.py": "from app.safe_paths import safe_child\nfrom pathlib import Path\nimport tempfile\n\n\ndef test_visible_child():\n    with tempfile.TemporaryDirectory() as tmp:\n        assert safe_child(Path(tmp), 'a.txt').name == 'a.txt'\n",
        },
        "hidden_tests": "from app.safe_paths import safe_child\nfrom pathlib import Path\nimport tempfile\n\n\ndef test_parent_escape_rejected():\n    with tempfile.TemporaryDirectory() as tmp:\n        try:\n            safe_child(Path(tmp), '../escape.txt')\n        except ValueError:\n            return\n        raise AssertionError('parent traversal should be rejected')\n",
        "patch": "Resolve both paths and require the normalized candidate to remain under the base.",
        "expected_files": {
            "app/safe_paths.py": "from pathlib import Path\n\n\ndef safe_child(base, user_path):\n    base_path = Path(base).resolve()\n    candidate = (base_path / user_path).resolve()\n    if base_path != candidate and base_path not in candidate.parents:\n        raise ValueError('outside base')\n    return candidate\n",
        },
    },
    {
        "category": "retry_state_reset",
        "bug": "Retry loop keeps the first error even after a later success.",
        "files": {
            "app/retry.py": "def first_success(callables):\n    error = None\n    for fn in callables:\n        try:\n            value = fn()\n        except Exception as exc:\n            error = exc\n            continue\n        if error:\n            raise error\n        return value\n    raise error\n",
            "tests/test_retry.py": "from app.retry import first_success\n\n\ndef test_visible_success():\n    assert first_success([lambda: 3]) == 3\n",
        },
        "hidden_tests": "from app.retry import first_success\n\n\ndef test_success_after_failure():\n    def bad():\n        raise RuntimeError('boom')\n    assert first_success([bad, lambda: 7]) == 7\n",
        "patch": "Return immediately on success and raise the last error only if every attempt fails.",
        "expected_files": {
            "app/retry.py": "def first_success(callables):\n    error = None\n    for fn in callables:\n        try:\n            return fn()\n        except Exception as exc:\n            error = exc\n    if error is not None:\n        raise error\n    raise ValueError('no callables provided')\n",
        },
    },
    {
        "category": "structured_log_parsing",
        "bug": "Log parser splits on every equals sign and corrupts values.",
        "files": {
            "app/logs.py": "def parse_kv_line(line):\n    result = {}\n    for part in line.split():\n        key, value = part.split('=')\n        result[key] = value\n    return result\n",
            "tests/test_logs.py": "from app.logs import parse_kv_line\n\n\ndef test_visible_simple():\n    assert parse_kv_line('level=info user=ada')['level'] == 'info'\n",
        },
        "hidden_tests": "from app.logs import parse_kv_line\n\n\ndef test_value_with_equals():\n    assert parse_kv_line('token=a=b=c level=debug')['token'] == 'a=b=c'\n",
        "patch": "Split each key-value token at the first equals sign only.",
        "expected_files": {
            "app/logs.py": "def parse_kv_line(line):\n    result = {}\n    for part in line.split():\n        if '=' not in part:\n            continue\n        key, value = part.split('=', 1)\n        result[key] = value\n    return result\n",
        },
    },
    {
        "category": "collection_aliasing",
        "bug": "Matrix builder aliases the same row object across all rows.",
        "files": {
            "app/matrix.py": "def zeros(rows, cols):\n    return [[0] * cols] * rows\n",
            "tests/test_matrix.py": "from app.matrix import zeros\n\n\ndef test_visible_shape():\n    assert zeros(2, 3) == [[0, 0, 0], [0, 0, 0]]\n",
        },
        "hidden_tests": "from app.matrix import zeros\n\n\ndef test_rows_are_independent():\n    matrix = zeros(2, 2)\n    matrix[0][0] = 9\n    assert matrix[1][0] == 0\n",
        "patch": "Build each row with a fresh list comprehension.",
        "expected_files": {
            "app/matrix.py": "def zeros(rows, cols):\n    return [[0 for _ in range(cols)] for _ in range(rows)]\n",
        },
    },
    {
        "category": "subprocess_result_contract",
        "bug": "Command helper ignores process failure and returns stderr as output.",
        "files": {
            "app/commands.py": "import subprocess\n\n\ndef run_text(command):\n    result = subprocess.run(command, capture_output=True, text=True)\n    return result.stdout or result.stderr\n",
            "tests/test_commands.py": "from app.commands import run_text\nimport sys\n\n\ndef test_visible_stdout():\n    assert run_text([sys.executable, '-c', 'print(42)']).strip() == '42'\n",
        },
        "hidden_tests": "from app.commands import run_text\nimport sys\n\n\ndef test_failure_raises():\n    try:\n        run_text([sys.executable, '-c', 'import sys; sys.stderr.write(\"bad\"); sys.exit(2)'])\n    except RuntimeError:\n        return\n    raise AssertionError('failed command should raise RuntimeError')\n",
        "patch": "Raise on non-zero return code and return stdout only for successful commands.",
        "expected_files": {
            "app/commands.py": "import subprocess\n\n\ndef run_text(command):\n    result = subprocess.run(command, capture_output=True, text=True)\n    if result.returncode != 0:\n        raise RuntimeError(result.stderr.strip() or 'command failed')\n    return result.stdout\n",
        },
    },
    {
        "category": "type_coercion_boundary",
        "bug": "Config reader treats numeric strings and integers differently.",
        "files": {
            "app/config.py": "def max_items(config):\n    value = config.get('max_items', 100)\n    if value > 1000:\n        return 1000\n    return value\n",
            "tests/test_config.py": "from app.config import max_items\n\n\ndef test_visible_int():\n    assert max_items({'max_items': 5}) == 5\n",
        },
        "hidden_tests": "from app.config import max_items\n\n\ndef test_string_value_and_cap():\n    assert max_items({'max_items': '7'}) == 7\n    assert max_items({'max_items': '5000'}) == 1000\n",
        "patch": "Coerce configured numeric values before applying caps.",
        "expected_files": {
            "app/config.py": "def max_items(config):\n    value = int(config.get('max_items', 100))\n    if value > 1000:\n        return 1000\n    return value\n",
        },
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--task-out", default=str(DEFAULT_TASK_OUT))
    parser.add_argument("--sts-out", default=str(DEFAULT_STS_OUT))
    parser.add_argument("--out", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    tasks = build_tasks(seed=args.seed, repetitions=max(1, args.repetitions))
    sts_rows = build_sts_rows(tasks)
    write_jsonl(resolve(args.task_out), tasks)
    write_jsonl(resolve(args.sts_out), sts_rows)
    categories = sorted({row["category"] for row in tasks})
    gates = [
        gate("private_repo_repair_tasks_written", len(tasks) >= len(TASK_TEMPLATES), f"tasks={len(tasks)}"),
        gate("hidden_tests_present_private_only", all(bool(row.get("hidden_tests")) for row in tasks), "hidden tests stored only in private curriculum"),
        gate("public_benchmark_solutions_absent", not any("HumanEval" in json.dumps(row) or "MBPP" in json.dumps(row) for row in tasks), "no public benchmark task names copied"),
        gate("sts_rows_written", len(sts_rows) >= len(tasks), f"sts_rows={len(sts_rows)}"),
        gate("source_split_groups_present", len(categories) >= 4, categories),
        gate("external_inference_zero", True, "generated locally from private templates"),
    ]
    payload = {
        "policy": "project_theseus_long_horizon_programming_curriculum_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "RED",
        "methodology": "private_repo_repair_hidden_test_curriculum",
        "summary": {
            "task_count": len(tasks),
            "sts_row_count": len(sts_rows),
            "category_count": len(categories),
            "categories": categories,
            "task_out": rel_or_abs(resolve(args.task_out)),
            "sts_out": rel_or_abs(resolve(args.sts_out)),
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "promotion_evidence": False,
            "external_inference_calls": 0,
        },
        "gates": gates,
        "score_semantics": "private long-horizon programming pressure only; public SWE-style tasks remain calibration-only",
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] == "GREEN" else 2


def build_tasks(*, seed: int, repetitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(repetitions):
        for index, template in enumerate(TASK_TEMPLATES):
            task_id = f"private_repo_repair_{template['category']}_{seed}_{rep:02d}_{index:02d}"
            split = "eval" if stable_int(f"{seed}:{task_id}") % 5 == 0 else "train"
            visible_files = dict(template["files"])
            rows.append(
                {
                    "policy": "project_theseus_long_horizon_programming_task_v1",
                    "task_id": task_id,
                    "split": split,
                    "category": template["category"],
                    "prompt": (
                        "Inspect the repo files, identify the bug, patch the minimal source file, "
                        "run visible tests, then validate against hidden tests."
                    ),
                    "repo_files": visible_files,
                    "visible_tests": visible_files.get(next((path for path in visible_files if path.startswith("tests/")), ""), ""),
                    "hidden_tests": template["hidden_tests"],
                    "bug_summary": template["bug"],
                    "repair_rationale": template["patch"],
                    "expected_patch_files": template["expected_files"],
                    "source_group": f"private_repo_repair/{template['category']}",
                    "public_benchmark_solutions_included": False,
                    "public_tests_included": False,
                    "external_inference_calls": 0,
                }
            )
    return rows


def build_sts_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    streams = [
        "context_stream",
        "solver_stream",
        "critic_stream",
        "tool_stream",
        "patch_stream",
        "residual_stream",
        "artifact_stream",
        "visible_report_stream",
    ]
    for task in tasks:
        task_id = str(task["task_id"])
        files = task.get("repo_files") if isinstance(task.get("repo_files"), dict) else {}
        file_list = ", ".join(sorted(files))
        rows.append(
            {
                "policy": "project_theseus_long_horizon_programming_sts_v1",
                "task_id": task_id,
                "split": task["split"],
                "category": task["category"],
                "streams": streams,
                "input_streams": {
                    "context_stream": f"repo files={file_list}; bug={task['bug_summary']}",
                    "solver_stream": "plan inspect -> patch -> test -> repair",
                    "critic_stream": "look for edge cases and hidden-test failure modes",
                    "tool_stream": "pytest sandbox available; no network; no public benchmark answers",
                },
                "target_streams": {
                    "solver_stream": task["repair_rationale"],
                    "critic_stream": f"dominant residual category={task['category']}",
                    "tool_stream": "run visible tests then hidden private tests",
                    "patch_stream": json.dumps(task["expected_patch_files"], sort_keys=True),
                    "residual_stream": "none_after_private_patch",
                    "artifact_stream": f"private_repo_repair_artifact:{task_id}",
                    "visible_report_stream": "repo repair trace completed with private hidden tests",
                },
                "causal_contract": {
                    "strict_past_only": True,
                    "one_token_per_output_stream_target": True,
                    "same_row_cross_stream_attention": "forbidden",
                },
                "public_benchmark_solutions_included": False,
                "public_tests_included": False,
                "external_inference_calls": 0,
            }
        )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# Long-Horizon Programming Curriculum",
            "",
            f"Generated: {payload.get('created_utc')}",
            f"Trigger: **{payload.get('trigger_state')}**",
            "",
            f"- Tasks: {summary.get('task_count')}",
            f"- STS rows: {summary.get('sts_row_count')}",
            f"- Categories: {', '.join(summary.get('categories') or [])}",
            f"- Task output: {summary.get('task_out')}",
            f"- STS output: {summary.get('sts_out')}",
            "",
            "Private hidden-test pressure only; this is not public promotion evidence.",
            "",
        ]
    )


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
