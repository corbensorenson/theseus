"""Private same-seed A/B for the broad-transfer residual decoder router.

This is intentionally private-only. It builds a small heldout manifest from
generated residual curricula, runs fanout with the router disabled and enabled,
and compares candidate distribution plus contract-verifier coverage. Public
benchmark manifests are empty in both arms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from theseus_archive_resolver import resolve_archived_path  # noqa: E402

REPORTS = ROOT / "reports"


def release_binary() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return ROOT / "target" / "release" / f"symliquid-cli{suffix}"


def training_data_root() -> Path:
    configured = os.environ.get("THESEUS_TRAINING_DATA_ROOT", "").strip()
    if configured:
        return Path(configured)
    if sys.platform.startswith("win"):
        return Path("D:/ProjectTheseus/training_data")
    return ROOT / "data" / "training_data"


def training_data_path(*parts: str) -> Path:
    return training_data_root().joinpath(*parts)


DEFAULT_BINARY = release_binary()
DEFAULT_CHECKPOINTS = [
    REPORTS / "student_code_lm_checkpoint_train_once_fanout_smoke2.json",
    REPORTS / "student_code_lm_checkpoint_train_once_fanout_smoke.json",
    REPORTS / "student_code_lm_checkpoint_private_pressure_private_recovery_train_once_fanout_v1.json",
]
PRIVATE_SOURCES = [
    training_data_path(
        "residual_code_curriculum",
        "private_train",
        "frontier_private_transfer_ratchet_v1_source_mbpp_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("residual_code_curriculum", "private_train", "residual_code_lm_tasks.jsonl"),
    training_data_path(
        "high_transfer",
        "private_train",
        "residual_targeted_private_edge_case_contract_v1_residual_code_lm_tasks.jsonl",
    ),
    training_data_path(
        "high_transfer",
        "private_train",
        "private_type_shape_receiver_veto_ablation_residual_code_lm_tasks.jsonl",
    ),
    training_data_path("high_transfer", "private_train", "algorithmic_planning_residual_code_lm_tasks.jsonl"),
    training_data_path("high_transfer", "private_train", "execution_shaped_programs_residual_code_lm_tasks.jsonl"),
    training_data_path("high_transfer", "private_train", "admissibility_and_interface_residual_code_lm_tasks.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/broad_transfer_residual_decoder_ablation.json")
    parser.add_argument("--markdown-out", default="reports/broad_transfer_residual_decoder_ablation.md")
    parser.add_argument("--manifest-out", default="reports/broad_transfer_residual_decoder_ablation_private_manifest.jsonl")
    parser.add_argument(
        "--manifest-in",
        default="",
        help="Optional private-only manifest to use instead of building the default balanced residual heldout set.",
    )
    parser.add_argument("--empty-public-out", default="reports/broad_transfer_residual_decoder_ablation_public_empty.jsonl")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--task-limit", type=int, default=8)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--sts-streams", default="reports/sts_decoder_control_rows.jsonl")
    parser.add_argument("--semantic-test-timeout-seconds", type=float, default=2.5)
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def choose_checkpoint(explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        resolved = resolve_archived_path(path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(path)
    for path in DEFAULT_CHECKPOINTS:
        resolved = resolve_archived_path(path)
        if resolved.exists():
            return resolved
    raise FileNotFoundError("no Code LM checkpoint found for private residual ablation")


def residual_family(row: dict[str, Any]) -> str:
    explicit = explicit_residual_family(row)
    if explicit:
        return explicit
    text = " ".join(
        [
            str(row.get("residual_concept") or ""),
            str(row.get("concept_residual_label") or ""),
            str(row.get("category") or ""),
            str(row.get("prompt") or ""),
            " ".join(map(str, row.get("tags") or [])),
            json.dumps(row.get("decoder_contract") or {}, sort_keys=True),
        ]
    ).lower()
    if any(
        token in text
        for token in [
            "verification_cascade_compile",
            "lint_parse",
            "compile_or_import",
            "syntax_rejected",
            "module_loads",
            "compile",
        ]
    ):
        return "verification_cascade_compile"
    if any(
        token in text
        for token in [
            "adapter_runtime_dependency_handling",
            "runtime_load_failure",
            "optional_dependency",
            "pandas",
            "numpy",
            "beautifulsoup",
        ]
    ):
        return "external_dependency_missing"
    if any(token in text for token in ["local_code_generation_adapter_needed", "json", "csv", "file_path", "archive", "path", "system_api"]):
        return "local_code_generation_adapter_needed"
    if "edge_contract_v2" in text or "edge_case" in text:
        return "edge_case"
    if "algorithm" in text or "window" in text or "frequency" in text:
        return "algorithm_choice"
    if "type" in text or "return_shape" in text or "interface" in text:
        return "type_handling"
    if "edge" in text or "empty" in text or "none" in text:
        return "edge_case"
    return "edge_case"


def explicit_residual_family(row: dict[str, Any]) -> str:
    text_parts = [
        str(row.get("task_id") or ""),
        str(row.get("residual_concept") or ""),
        str(row.get("concept_residual_label") or ""),
        str(row.get("category") or ""),
        " ".join(map(str, row.get("tags") or [])),
    ]
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    plan = contract.get("generation_plan") if isinstance(contract.get("generation_plan"), dict) else {}
    text_parts.extend(map(str, plan.get("skeleton_bias") or []))
    text = " ".join(text_parts).lower()
    if (
        "verification_cascade_compile" in text
        or "lint_parse" in text
        or "compile_or_import" in text
        or "syntax_rejected" in text
        or "compile" in text
    ):
        return "verification_cascade_compile"
    if (
        "admissibility_and_interface" in text
        or "local_code_generation_adapter_needed" in text
    ):
        return "local_code_generation_adapter_needed"
    if (
        "adapter_runtime_dependency_handling" in text
        or "runtime_load_failure" in text
        or "optional_dependency" in text
    ):
        return "external_dependency_missing"
    if "type_and_return_shape" in text or "receiver_return_shape_contract" in text:
        return "type_handling"
    if "algorithmic_planning" in text or "algorithm_choice" in text:
        return "algorithm_choice"
    if "edge_contract_v2_private_residual_curriculum" in text or "edge_contract_v2" in text or "edge_case" in text:
        return "edge_case"
    return ""


def materialize_private_semantic_tests(row: dict[str, Any]) -> None:
    if str(row.get("tests") or "").strip():
        row["private_semantic_tests_materialized"] = False
        return
    entry_point = str(row.get("entry_point") or "").strip()
    category = str(row.get("category") or "").strip()
    tests = private_semantic_tests_for_category(category, entry_point)
    if not tests:
        row["private_semantic_tests_materialized"] = False
        row["private_semantic_tests_missing_reason"] = f"no_private_template_for_category:{category or 'unknown'}"
        return
    row["tests"] = tests
    row["private_semantic_tests_materialized"] = True
    row["private_semantic_test_policy"] = "project_theseus_private_contract_semantic_tests_v1"
    row["private_semantic_test_source"] = "private_visible_prompt_category_and_contract_only"
    row["public_tests_used"] = False
    row["public_solutions_used"] = False


def private_semantic_tests_for_category(category: str, entry_point: str) -> str:
    if not entry_point.isidentifier():
        return ""
    templates = {
        "base_digits": base_digits_tests,
        "private_count_records_at_threshold": count_records_at_threshold_tests,
        "private_exec_csv_command_outputs": csv_command_outputs_tests,
        "private_exec_csv_split_shuffle": csv_split_shuffle_tests,
        "private_exec_json_extract_field": json_extract_field_tests,
        "private_exec_log_backup_tar": log_backup_tar_tests,
        "private_exec_system_info_dict": system_info_dict_tests,
        "private_exec_zip_flat_directory": zip_flat_directory_tests,
        "private_base64_json_field": base64_json_field_tests,
        "private_json_payload_field": json_payload_field_tests,
        "private_typed_pairs_to_dict": typed_pairs_to_dict_tests,
        "private_url_query_value": url_query_value_tests,
        "private_residual_nested_string_paths": nested_string_paths_tests,
        "private_residual_sort_by_second": sort_by_second_tests,
        "sort_even_index_values": sort_even_index_values_tests,
        "two_sum_zero_exists": two_sum_zero_exists_tests,
    }
    factory = templates.get(category)
    return factory(entry_point) if factory else ""


def base_digits_tests(fn: str) -> str:
    return (
        f"assert {fn}(0, 2) == '0'\n"
        f"assert {fn}(10, 2) == '1010'\n"
        f"assert {fn}(8, 2) == '1000'\n"
    )


def count_records_at_threshold_tests(fn: str) -> str:
    return (
        f"assert {fn}([{{'score': 5}}, {{'score': 4}}, {{'score': 3}}, {{'other': 9}}], 4) == 2\n"
        f"assert {fn}([{{'score': -1}}, {{'score': 0}}, {{'score': 7}}], 1) == 1\n"
        f"assert {fn}([], 10) == 0\n"
    )


def csv_command_outputs_tests(fn: str) -> str:
    return (
        "import csv, os, shlex, sys, tempfile\n"
        "with tempfile.TemporaryDirectory() as root:\n"
        "    csv_path = os.path.join(root, 'commands.csv')\n"
        "    out_dir = os.path.join(root, 'out')\n"
        "    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(\"print('alpha')\")\n"
        "    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:\n"
        "        csv.writer(handle).writerow([command])\n"
        f"    paths = {fn}(csv_path, out_dir)\n"
        "    assert isinstance(paths, list) and len(paths) == 1\n"
        "    assert os.path.isfile(paths[0])\n"
        "    assert 'alpha' in open(paths[0], encoding='utf-8').read()\n"
    )


def csv_split_shuffle_tests(fn: str) -> str:
    return (
        "import csv, os, tempfile\n"
        "with tempfile.TemporaryDirectory() as root:\n"
        f"    assert {fn}(os.path.join(root, 'missing.csv')) == []\n"
        "    txt = os.path.join(root, 'data.txt')\n"
        "    open(txt, 'w', encoding='utf-8').write('x')\n"
        f"    assert {fn}(txt) == []\n"
        "    csv_path = os.path.join(root, 'data.csv')\n"
        "    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:\n"
        "        csv.writer(handle).writerows([[1], [2], [3], [4]])\n"
        f"    paths = {fn}(csv_path)\n"
        "    assert isinstance(paths, list) and paths and all(isinstance(path, str) for path in paths)\n"
        "    assert all(os.path.isfile(path) for path in paths)\n"
        "    recovered = []\n"
        "    for path in paths:\n"
        "        with open(path, newline='', encoding='utf-8') as handle:\n"
        "            recovered.extend(list(csv.reader(handle)))\n"
        "    assert len(recovered) == 4\n"
    )


def json_extract_field_tests(fn: str) -> str:
    return (
        "import json, os, tempfile\n"
        "with tempfile.TemporaryDirectory() as root:\n"
        "    path = os.path.join(root, 'payload.json')\n"
        "    with open(path, 'w', encoding='utf-8') as handle:\n"
        "        json.dump({'name': 'theseus', 'score': 7}, handle)\n"
        f"    assert {fn}(path, 'name') == 'theseus'\n"
        f"    assert {fn}(path, 'missing') is None\n"
        f"    assert {fn}(os.path.join(root, 'absent.json'), 'name') is None\n"
    )


def log_backup_tar_tests(fn: str) -> str:
    return (
        "import os, tarfile, tempfile\n"
        "with tempfile.TemporaryDirectory() as root:\n"
        "    log_dir = os.path.join(root, 'logs')\n"
        "    out_dir = os.path.join(root, 'out')\n"
        "    os.makedirs(log_dir)\n"
        f"    empty_result = {fn}(log_dir, out_dir)\n"
        "    assert isinstance(empty_result, str) and 'log' in empty_result.lower()\n"
        "    with open(os.path.join(log_dir, 'a.log'), 'w', encoding='utf-8') as handle:\n"
        "        handle.write('a')\n"
        "    with open(os.path.join(log_dir, 'b.txt'), 'w', encoding='utf-8') as handle:\n"
        "        handle.write('b')\n"
        f"    archive_path = {fn}(log_dir, out_dir)\n"
        "    assert os.path.isfile(archive_path)\n"
        "    assert not os.path.exists(os.path.join(log_dir, 'a.log'))\n"
        "    with tarfile.open(archive_path, 'r:gz') as archive:\n"
        "        assert 'a.log' in archive.getnames()\n"
    )


def system_info_dict_tests(fn: str) -> str:
    return (
        f"result = {fn}()\n"
        "assert isinstance(result, dict)\n"
        "assert isinstance(result.get('Operating System'), str) and result.get('Operating System')\n"
        "assert isinstance(result.get('Architecture'), str) and result.get('Architecture')\n"
        "assert isinstance(result.get('Memory Usage'), str) and result.get('Memory Usage')\n"
    )


def zip_flat_directory_tests(fn: str) -> str:
    return (
        "import os, tempfile, zipfile\n"
        "with tempfile.TemporaryDirectory() as root:\n"
        f"    assert {fn}(root) is None\n"
        "    with open(os.path.join(root, 'a.txt'), 'w', encoding='utf-8') as handle:\n"
        "        handle.write('a')\n"
        "    os.makedirs(os.path.join(root, 'nested'))\n"
        "    with open(os.path.join(root, 'nested', 'b.txt'), 'w', encoding='utf-8') as handle:\n"
        "        handle.write('b')\n"
        f"    zip_path = {fn}(root)\n"
        "    assert os.path.isfile(zip_path)\n"
        "    with zipfile.ZipFile(zip_path) as archive:\n"
        "        assert archive.namelist() == ['a.txt']\n"
        f"    assert {fn}(os.path.join(root, 'missing')) is None\n"
    )


def base64_json_field_tests(fn: str) -> str:
    return (
        "import base64, json\n"
        "payload = base64.b64encode(json.dumps({'name': 'theseus', 'score': 7}).encode()).decode()\n"
        f"assert {fn}(payload, 'name') == 'theseus'\n"
        f"assert {fn}(payload, 'score') == '7'\n"
        f"assert {fn}('not-base64', 'name') == ''\n"
    )


def json_payload_field_tests(fn: str) -> str:
    return (
        "import json\n"
        "payload = json.dumps({'name': 'theseus', 'score': 7})\n"
        f"assert {fn}(payload, 'name') == 'theseus'\n"
        f"assert {fn}(payload, 'score') == '7'\n"
        f"assert {fn}('not-json', 'name') == ''\n"
    )


def typed_pairs_to_dict_tests(fn: str) -> str:
    return (
        f"assert {fn}([('a', 1), ['b', 2], ('bad', 'x'), (3, 4), ('short',)]) == {{'a': 1, 'b': 2}}\n"
        f"assert {fn}([]) == {{}}\n"
        f"assert {fn}(None) == {{}}\n"
    )


def url_query_value_tests(fn: str) -> str:
    return (
        f"assert {fn}('https://example.test/path?a=one&b=two', 'a') == 'one'\n"
        f"assert {fn}('https://example.test/path?a=one&b=two', 'missing') == ''\n"
        f"assert {fn}('not a url', 'a') == ''\n"
    )


def nested_string_paths_tests(fn: str) -> str:
    return (
        f"assert {fn}({{'a': ['x', 3], 'b': {{'c': 'y'}}}}) == ['a/0', 'b/c']\n"
        f"assert {fn}({{}}) == []\n"
        f"assert {fn}(['x', ['y']]) == ['0', '1/0']\n"
    )


def sort_by_second_tests(fn: str) -> str:
    return (
        f"assert {fn}([(2, 3), (1, 2), (3, 2)]) == [(1, 2), (3, 2), (2, 3)]\n"
        f"assert {fn}([]) == []\n"
        f"assert {fn}([('b', 1), ('a', 1), (9,)]) == [('a', 1), ('b', 1)]\n"
    )


def sort_even_index_values_tests(fn: str) -> str:
    return (
        f"assert {fn}([5, 9, 1, 8, 3]) == [1, 9, 3, 8, 5]\n"
        f"assert {fn}([]) == []\n"
        f"assert {fn}([2, 1]) == [2, 1]\n"
    )


def two_sum_zero_exists_tests(fn: str) -> str:
    return (
        f"assert {fn}([3, 1, -3]) is True\n"
        f"assert {fn}([1, 2, 3]) is False\n"
        f"assert {fn}([0]) is False\n"
    )


def build_private_manifest(limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for source in PRIVATE_SOURCES:
        for row in read_jsonl(source):
            key = str(row.get("task_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            family = residual_family(row)
            row = dict(row)
            row["split"] = "eval"
            row["task_id"] = f"broad_transfer_residual_ablation_{family}_{len(buckets[family]):04d}"
            row["benchmark_evidence_level"] = "private_residual_decoder_ablation_eval_only"
            tags = list(map(str, row.get("tags") or []))
            if family not in tags:
                tags.append(family)
            tags.append("broad_transfer_residual_decoder_ablation_v1")
            row["tags"] = tags
            row["residual_concept"] = family
            row.setdefault("ablation_source", "private_generated_residual_curriculum")
            materialize_private_semantic_tests(row)
            buckets[family].append(row)
    target_order = [
        "edge_case",
        "external_dependency_missing",
        "type_handling",
        "algorithm_choice",
        "verification_cascade_compile",
        "local_code_generation_adapter_needed",
    ]
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets.values()):
        made_progress = False
        for family in target_order:
            if len(selected) >= limit:
                break
            if buckets[family]:
                selected.append(buckets[family].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def run_arm(
    *,
    name: str,
    enabled: bool,
    binary: Path,
    checkpoint: Path,
    manifest: Path,
    empty_public: Path,
    seed: int,
    candidates_per_task: int,
    sts_streams: Path,
    task_family_by_id: dict[str, str],
    manifest_rows_by_id: dict[str, dict[str, Any]],
    semantic_test_timeout_seconds: float,
) -> dict[str, Any]:
    private_out = REPORTS / f"broad_transfer_residual_decoder_ablation_{name}_private_candidates.jsonl"
    public_out = REPORTS / f"broad_transfer_residual_decoder_ablation_{name}_public_candidates.jsonl"
    rust_report = REPORTS / f"broad_transfer_residual_decoder_ablation_{name}_fanout.json"
    env = os.environ.copy()
    env["THESEUS_BROAD_TRANSFER_RESIDUAL_DECODER_V1"] = "1" if enabled else "0"
    env["THESEUS_BROAD_PUBLIC_FLOOR_RECOVERY_V1"] = "1" if enabled else "0"
    env["THESEUS_PRIVATE_TO_PUBLIC_RECEIVER_INVENTORY_BRIDGE_PRIVATE_SHADOW_V1"] = (
        "1" if enabled else "0"
    )
    env["THESEUS_CODE_LM_LOW_LATENCY_FANOUT"] = "1"
    env["THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_MULTI_CANDIDATE_FANOUT"] = "1"
    env["THESEUS_CODE_LM_LOW_LATENCY_EXPENSIVE_RESCUE"] = "0"
    started = time.perf_counter()
    command = [
        str(binary),
        "generate-code-lm-closure-fanout",
        "--private-curriculum",
        str(manifest),
        "--public-task-manifest",
        str(empty_public),
        "--checkpoint-in",
        str(checkpoint),
        "--seed",
        str(seed),
        "--candidates-per-task",
        str(candidates_per_task),
        "--private-candidate-out",
        str(private_out),
        "--public-candidate-out",
        str(public_out),
        "--report-out",
        str(rust_report),
        "--sts-streams",
        str(sts_streams),
        "--private-eval-limit",
        "0",
        "--public-task-limit",
        "0",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    candidate_rows = read_jsonl(private_out)
    return {
        "name": name,
        "enabled": enabled,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "private_candidate_out": str(private_out),
        "public_candidate_out": str(public_out),
        "rust_report": str(rust_report),
        "summary": summarize_candidates(
            candidate_rows,
            task_family_by_id,
            manifest_rows_by_id,
            semantic_test_timeout_seconds,
        ),
    }


def summarize_candidates(
    rows: list[dict[str, Any]],
    task_family_by_id: dict[str, str] | None = None,
    manifest_rows_by_id: dict[str, dict[str, Any]] | None = None,
    semantic_test_timeout_seconds: float = 2.5,
) -> dict[str, Any]:
    task_family_by_id = task_family_by_id or {}
    manifest_rows_by_id = manifest_rows_by_id or {}
    tasks = {str(row.get("task_id") or "") for row in rows}
    no_admissible = {
        str(row.get("task_id") or "")
        for row in rows
        if row.get("candidate_generation_mode") == "student_decoder_no_admissible_candidate_residual"
    }
    passed = {
        str(row.get("task_id") or "")
        for row in rows
        if row.get("decoder_contract_verifier_v1_passed") is True
        and row.get("candidate_generation_mode") != "student_decoder_no_admissible_candidate_residual"
    }
    receiver_eligible = {
        str(row.get("task_id") or "")
        for row in rows
        if private_receiver_inventory_eligible(row)
    }
    residual_rows = [
        row
        for row in rows
        if row.get("broad_transfer_residual_stage") is True
        or "broad_transfer_residual_router_v1" in str(row.get("candidate_generation_mode") or "")
    ]
    eligible_receiver_rows = [
        row
        for row in rows
        if row.get("eligible_receiver_inventory_stage") is True
        or "eligible_receiver_inventory_router_v1" in str(row.get("candidate_generation_mode") or "")
    ]
    bridge_shadow_rows = [
        row
        for row in rows
        if row.get("private_to_public_receiver_inventory_bridge_stage") is True
        or "private_to_public_receiver_inventory_bridge_v1"
        in str(row.get("candidate_generation_mode") or "")
    ]
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in rows)
    grammar_masked_rows = [row for row in rows if grammar_masked_learned_token_candidate(row)]
    prompt_program_rows = [
        row for row in rows if "prompt_program_decoder" in str(row.get("candidate_generation_mode") or "").lower()
    ]
    private_receiver_rows = [row for row in rows if private_receiver_inventory_eligible(row)]
    non_grammar_private_receiver_rows = [
        row for row in private_receiver_rows if not grammar_masked_learned_token_candidate(row)
    ]
    family_pass = Counter()
    family_seen = Counter()
    for row in rows:
        policy = row.get("broad_transfer_residual_policy")
        if not isinstance(policy, dict):
            policy = (row.get("provenance") or {}).get("broad_transfer_residual_policy", {})
        families = policy.get("families") if isinstance(policy, dict) else []
        for family in families or []:
            family_seen[str(family)] += 1
            if row.get("decoder_contract_verifier_v1_passed") is True:
                family_pass[str(family)] += 1
    task_family_metrics = summarize_task_families(rows, task_family_by_id)
    semantic_tests = summarize_private_semantic_tests(
        rows,
        manifest_rows_by_id,
        task_family_by_id,
        timeout_seconds=semantic_test_timeout_seconds,
    )
    task_count = len([task for task in tasks if task])
    return {
        "row_count": len(rows),
        "task_count": task_count,
        "passed_task_count": len(passed),
        "private_receiver_eligible_task_count": len(receiver_eligible),
        "no_admissible_task_count": len(no_admissible),
        "passed_task_rate": len(passed) / task_count if task_count else 0.0,
        "private_receiver_eligible_task_rate": len(receiver_eligible) / task_count if task_count else 0.0,
        "no_admissible_rate": len(no_admissible) / task_count if task_count else 0.0,
        "broad_transfer_residual_row_count": len(residual_rows),
        "broad_transfer_residual_task_count": len({str(row.get("task_id") or "") for row in residual_rows}),
        "eligible_receiver_inventory_row_count": len(eligible_receiver_rows),
        "eligible_receiver_inventory_task_count": len({str(row.get("task_id") or "") for row in eligible_receiver_rows}),
        "private_to_public_receiver_bridge_shadow_row_count": len(bridge_shadow_rows),
        "private_to_public_receiver_bridge_shadow_task_count": len({str(row.get("task_id") or "") for row in bridge_shadow_rows}),
        "grammar_masked_learned_token_candidate_count": len(grammar_masked_rows),
        "prompt_program_candidate_count": len(prompt_program_rows),
        "private_receiver_eligible_candidate_count": len(private_receiver_rows),
        "non_grammar_private_receiver_eligible_candidate_count": len(non_grammar_private_receiver_rows),
        "mode_counts": dict(modes.most_common(12)),
        "family_seen_counts": dict(family_seen),
        "family_pass_counts": dict(family_pass),
        "task_family_metrics": task_family_metrics,
        "semantic_tests": semantic_tests,
    }


def summarize_task_families(rows: list[dict[str, Any]], task_family_by_id: dict[str, str]) -> dict[str, dict[str, Any]]:
    family_tasks: dict[str, set[str]] = defaultdict(set)
    family_passed: dict[str, set[str]] = defaultdict(set)
    family_receiver: dict[str, set[str]] = defaultdict(set)
    family_no_admissible: dict[str, set[str]] = defaultdict(set)
    for task_id, family in task_family_by_id.items():
        if task_id and family:
            family_tasks[family].add(task_id)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        family = task_family_by_id.get(task_id)
        if not family:
            family = residual_family(row)
            family_tasks[family].add(task_id)
        if row.get("candidate_generation_mode") == "student_decoder_no_admissible_candidate_residual":
            family_no_admissible[family].add(task_id)
        if (
            row.get("decoder_contract_verifier_v1_passed") is True
            and row.get("candidate_generation_mode") != "student_decoder_no_admissible_candidate_residual"
        ):
            family_passed[family].add(task_id)
        if private_receiver_inventory_eligible(row):
            family_receiver[family].add(task_id)
    out: dict[str, dict[str, Any]] = {}
    for family in sorted(family_tasks):
        task_count = len(family_tasks[family])
        passed_count = len(family_passed[family])
        receiver_count = len(family_receiver[family])
        no_admissible_count = len(family_no_admissible[family])
        out[family] = {
            "task_count": task_count,
            "passed_task_count": passed_count,
            "private_receiver_eligible_task_count": receiver_count,
            "no_admissible_task_count": no_admissible_count,
            "passed_task_rate": passed_count / task_count if task_count else 0.0,
            "private_receiver_eligible_task_rate": receiver_count / task_count if task_count else 0.0,
            "no_admissible_rate": no_admissible_count / task_count if task_count else 0.0,
        }
    return out


def private_receiver_inventory_eligible(row: dict[str, Any]) -> bool:
    mode = str(row.get("candidate_generation_mode") or "").lower()
    if "no_admissible" in mode:
        return False
    if not grammar_masked_learned_token_candidate(row):
        return False
    if row.get("private_receiver_inventory_eligible") is True:
        return True
    return (
        row.get("token_level_code_generation_learned") is True
        and row.get("full_body_token_candidate") is True
        and row.get("decoder_contract_verifier_v1_passed") is True
        and row.get("deterministic_guardrail_passed") is not False
        and not bool(row.get("same_seed_non_sts_comparator"))
        and not bool(row.get("template_like_candidate"))
        and not bool(row.get("placeholder_scaffold_body"))
        and not bool(row.get("expression_memory_fallback"))
        and not bool(row.get("sts_candidate_expression_used"))
        and not bool(row.get("broad_transfer_residual_stage"))
    )


def grammar_masked_learned_token_candidate(row: dict[str, Any]) -> bool:
    if row_bool(row, "grammar_masked_learned_token_candidate"):
        return True
    loop = row.get("program_synthesis_loop_v1")
    if isinstance(loop, dict):
        decode_control = loop.get("decode_control")
        if isinstance(decode_control, dict):
            return bool(decode_control.get("grammar_masked_learned_token_candidate"))
    return False


def row_bool(row: dict[str, Any], key: str) -> bool:
    if row.get(key) is True:
        return True
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and provenance.get(key) is True:
        return True
    return False


def summarize_private_semantic_tests(
    rows: list[dict[str, Any]],
    manifest_rows_by_id: dict[str, dict[str, Any]],
    task_family_by_id: dict[str, str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    tested_candidates = 0
    passed_candidates = 0
    syntax_failed_candidates = 0
    runtime_failed_candidates = 0
    timeout_candidates = 0
    tasks_with_tests = {
        task_id
        for task_id, task in manifest_rows_by_id.items()
        if str(task.get("tests") or "").strip()
    }
    passed_tasks: set[str] = set()
    candidate_rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_tasks: dict[str, set[str]] = defaultdict(set)
    family_passed: dict[str, set[str]] = defaultdict(set)
    failure_samples: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id not in tasks_with_tests:
            continue
        candidate_rows_by_task[task_id].append(row)
    for task_id in tasks_with_tests:
        family = task_family_by_id.get(task_id, "unknown")
        family_tasks[family].add(task_id)
    for task_id, candidate_rows in candidate_rows_by_task.items():
        task = manifest_rows_by_id.get(task_id, {})
        family = task_family_by_id.get(task_id, "unknown")
        for row in candidate_rows:
            if str(row.get("candidate_generation_mode") or "") == "student_decoder_no_admissible_candidate_residual":
                continue
            code = str(row.get("code") or "").strip()
            tests = str(task.get("tests") or "").strip()
            if not code or not tests:
                continue
            tested_candidates += 1
            result = run_private_candidate_tests(code, tests, timeout_seconds=timeout_seconds)
            if result["passed"]:
                passed_candidates += 1
                passed_tasks.add(task_id)
                family_passed[family].add(task_id)
            elif result["stage"] == "syntax":
                syntax_failed_candidates += 1
            elif result["stage"] == "timeout":
                timeout_candidates += 1
            else:
                runtime_failed_candidates += 1
            if not result["passed"] and len(failure_samples) < 8:
                failure_samples.append(
                    {
                        "task_id": task_id,
                        "family": family,
                        "mode": row.get("candidate_generation_mode"),
                        "stage": result["stage"],
                        "stderr_tail": result.get("stderr_tail", ""),
                    }
                )
    family_metrics: dict[str, dict[str, Any]] = {}
    for family in sorted(family_tasks):
        task_count = len(family_tasks[family])
        passed_count = len(family_passed[family])
        family_metrics[family] = {
            "semantic_tested_task_count": task_count,
            "semantic_passed_task_count": passed_count,
            "semantic_passed_task_rate": passed_count / task_count if task_count else 0.0,
        }
    tested_task_count = len(tasks_with_tests)
    passed_task_count = len(passed_tasks)
    return {
        "policy": "project_theseus_private_semantic_candidate_test_v1",
        "score_semantics": "private behavioral tests only; no public tests or solutions",
        "semantic_tested_task_count": tested_task_count,
        "semantic_candidate_test_count": tested_candidates,
        "semantic_candidate_pass_count": passed_candidates,
        "semantic_passed_task_count": passed_task_count,
        "semantic_passed_task_rate": passed_task_count / tested_task_count if tested_task_count else 0.0,
        "syntax_failed_candidate_count": syntax_failed_candidates,
        "runtime_failed_candidate_count": runtime_failed_candidates,
        "timeout_candidate_count": timeout_candidates,
        "family_metrics": family_metrics,
        "failure_samples": failure_samples,
        "public_tests_used": False,
        "public_solutions_used": False,
    }


def run_private_candidate_tests(code: str, tests: str, *, timeout_seconds: float) -> dict[str, Any]:
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        return {"passed": False, "stage": "syntax", "stderr_tail": str(exc)}
    harness = code.rstrip() + "\n\n" + tests.rstrip() + "\n"
    with tempfile.TemporaryDirectory(prefix="theseus_private_semantic_") as tmp:
        harness_path = Path(tmp) / "_run_private_semantic_tests.py"
        harness_path.write_text(harness, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(harness_path)],
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(0.25, timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "stage": "timeout",
                "stderr_tail": str(exc)[-500:],
            }
    return {
        "passed": completed.returncode == 0,
        "stage": "behavior" if completed.returncode else "passed",
        "stderr_tail": completed.stderr[-500:],
    }


def delta_report(baseline: dict[str, Any], patched: dict[str, Any]) -> dict[str, Any]:
    b = baseline["summary"]
    p = patched["summary"]
    family_deltas = family_delta_report(
        b.get("task_family_metrics") if isinstance(b.get("task_family_metrics"), dict) else {},
        p.get("task_family_metrics") if isinstance(p.get("task_family_metrics"), dict) else {},
    )
    semantic_delta = semantic_delta_report(
        b.get("semantic_tests") if isinstance(b.get("semantic_tests"), dict) else {},
        p.get("semantic_tests") if isinstance(p.get("semantic_tests"), dict) else {},
    )
    return {
        "passed_task_rate_delta": p["passed_task_rate"] - b["passed_task_rate"],
        "passed_task_count_delta": p["passed_task_count"] - b["passed_task_count"],
        "private_receiver_eligible_task_rate_delta": p["private_receiver_eligible_task_rate"] - b["private_receiver_eligible_task_rate"],
        "private_receiver_eligible_task_count_delta": p["private_receiver_eligible_task_count"] - b["private_receiver_eligible_task_count"],
        "no_admissible_rate_delta": p["no_admissible_rate"] - b["no_admissible_rate"],
        "broad_transfer_residual_task_count_delta": p["broad_transfer_residual_task_count"] - b["broad_transfer_residual_task_count"],
        "broad_transfer_residual_row_count_delta": p["broad_transfer_residual_row_count"] - b["broad_transfer_residual_row_count"],
        "eligible_receiver_inventory_task_count_delta": p["eligible_receiver_inventory_task_count"] - b["eligible_receiver_inventory_task_count"],
        "eligible_receiver_inventory_row_count_delta": p["eligible_receiver_inventory_row_count"] - b["eligible_receiver_inventory_row_count"],
        "private_to_public_receiver_bridge_shadow_task_count_delta": p["private_to_public_receiver_bridge_shadow_task_count"]
        - b["private_to_public_receiver_bridge_shadow_task_count"],
        "private_to_public_receiver_bridge_shadow_row_count_delta": p["private_to_public_receiver_bridge_shadow_row_count"]
        - b["private_to_public_receiver_bridge_shadow_row_count"],
        "grammar_masked_learned_token_candidate_count_delta": p["grammar_masked_learned_token_candidate_count"]
        - b["grammar_masked_learned_token_candidate_count"],
        "prompt_program_candidate_count_delta": p["prompt_program_candidate_count"] - b["prompt_program_candidate_count"],
        "semantic_test_passed_task_rate_delta": semantic_delta["semantic_passed_task_rate_delta"],
        "semantic_test_passed_task_count_delta": semantic_delta["semantic_passed_task_count_delta"],
        "task_family_deltas": family_deltas,
        "semantic_task_family_deltas": semantic_delta["family_deltas"],
    }


def family_delta_report(
    baseline: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for family in sorted(set(baseline) | set(patched)):
        base = baseline.get(family) if isinstance(baseline.get(family), dict) else {}
        current = patched.get(family) if isinstance(patched.get(family), dict) else {}
        out[family] = {
            "task_count": int(current.get("task_count") or base.get("task_count") or 0),
            "passed_task_rate_delta": float(current.get("passed_task_rate") or 0.0) - float(base.get("passed_task_rate") or 0.0),
            "passed_task_count_delta": int(current.get("passed_task_count") or 0) - int(base.get("passed_task_count") or 0),
            "private_receiver_eligible_task_rate_delta": float(current.get("private_receiver_eligible_task_rate") or 0.0)
            - float(base.get("private_receiver_eligible_task_rate") or 0.0),
            "private_receiver_eligible_task_count_delta": int(current.get("private_receiver_eligible_task_count") or 0)
            - int(base.get("private_receiver_eligible_task_count") or 0),
            "no_admissible_rate_delta": float(current.get("no_admissible_rate") or 0.0) - float(base.get("no_admissible_rate") or 0.0),
            "no_admissible_task_count_delta": int(current.get("no_admissible_task_count") or 0) - int(base.get("no_admissible_task_count") or 0),
        }
    return out


def semantic_delta_report(baseline: dict[str, Any], patched: dict[str, Any]) -> dict[str, Any]:
    base_family = baseline.get("family_metrics") if isinstance(baseline.get("family_metrics"), dict) else {}
    current_family = patched.get("family_metrics") if isinstance(patched.get("family_metrics"), dict) else {}
    family_deltas: dict[str, dict[str, Any]] = {}
    for family in sorted(set(base_family) | set(current_family)):
        base = base_family.get(family) if isinstance(base_family.get(family), dict) else {}
        current = current_family.get(family) if isinstance(current_family.get(family), dict) else {}
        family_deltas[family] = {
            "semantic_tested_task_count": int(current.get("semantic_tested_task_count") or base.get("semantic_tested_task_count") or 0),
            "semantic_passed_task_count_delta": int(current.get("semantic_passed_task_count") or 0)
            - int(base.get("semantic_passed_task_count") or 0),
            "semantic_passed_task_rate_delta": float(current.get("semantic_passed_task_rate") or 0.0)
            - float(base.get("semantic_passed_task_rate") or 0.0),
        }
    return {
        "semantic_passed_task_rate_delta": float(patched.get("semantic_passed_task_rate") or 0.0)
        - float(baseline.get("semantic_passed_task_rate") or 0.0),
        "semantic_passed_task_count_delta": int(patched.get("semantic_passed_task_count") or 0)
        - int(baseline.get("semantic_passed_task_count") or 0),
        "family_deltas": family_deltas,
    }


def target_family_semantic_delta_ready(delta: dict[str, Any]) -> bool:
    family_deltas = delta.get("semantic_task_family_deltas")
    if not isinstance(family_deltas, dict) or not family_deltas:
        return False
    active = [
        row
        for row in family_deltas.values()
        if isinstance(row, dict) and int(row.get("semantic_tested_task_count") or 0) > 0
    ]
    if not active:
        return False
    positive = [
        row
        for row in active
        if float(row.get("semantic_passed_task_rate_delta") or 0.0) > 0.0
        or int(row.get("semantic_passed_task_count_delta") or 0) > 0
    ]
    minimum_positive_families = min(3, len(active))
    return len(positive) >= minimum_positive_families and all(
        float(row.get("semantic_passed_task_rate_delta") or 0.0) >= 0.0 for row in active
    )


def markdown(payload: dict[str, Any]) -> str:
    delta = payload["delta"]
    gates = payload["gates"]
    lines = [
        "# Broad Transfer Residual Decoder Ablation",
        "",
        f"- Status: **{payload['status']}**",
        f"- Private heldout tasks: {payload['manifest']['task_count']}",
        f"- Same seed: {payload['config']['seed']}",
        f"- Public tasks used: {payload['manifest']['public_task_count']}",
        f"- Passed task rate delta: {delta['passed_task_rate_delta']:.3f}",
        f"- Private receiver eligible rate delta: {delta['private_receiver_eligible_task_rate_delta']:.3f}",
        f"- No-admissible rate delta: {delta['no_admissible_rate_delta']:.3f}",
        f"- Private semantic test pass-rate delta: {delta.get('semantic_test_passed_task_rate_delta', 0.0):.3f}",
        f"- Residual-router task-count delta: {delta['broad_transfer_residual_task_count_delta']}",
        f"- Eligible receiver inventory task-count delta: {delta['eligible_receiver_inventory_task_count_delta']}",
        f"- Private-to-public bridge-shadow task-count delta: {delta['private_to_public_receiver_bridge_shadow_task_count_delta']}",
        "",
        "## Gates",
    ]
    for gate in gates:
        lines.append(f"- {gate['name']}: {'PASS' if gate['passed'] else 'FAIL'} - {gate['detail']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    markdown_out = Path(args.markdown_out)
    manifest = Path(args.manifest_out)
    empty_public = Path(args.empty_public_out)
    binary = Path(args.binary)
    checkpoint = choose_checkpoint(args.checkpoint)
    sts_streams = Path(args.sts_streams)

    rows = read_jsonl(Path(args.manifest_in)) if args.manifest_in else build_private_manifest(args.task_limit)
    rows = [dict(row, split="eval") for row in rows[: max(1, int(args.task_limit))]]
    task_family_by_id = {str(row.get("task_id") or ""): residual_family(row) for row in rows}
    manifest_rows_by_id = {str(row.get("task_id") or ""): row for row in rows if str(row.get("task_id") or "")}
    write_jsonl(manifest, rows)
    write_jsonl(empty_public, [])

    payload: dict[str, Any] = {
        "policy": "project_theseus_broad_transfer_residual_decoder_ablation_v1",
        "config": {
            "binary": str(binary),
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "candidates_per_task": args.candidates_per_task,
            "sts_streams": str(sts_streams),
            "manifest_in": str(args.manifest_in or ""),
            "broad_public_floor_recovery_v1": True,
        },
        "manifest": {
            "path": str(manifest),
            "task_count": len(rows),
            "public_task_count": 0,
            "private_semantic_test_task_count": sum(1 for row in rows if str(row.get("tests") or "").strip()),
            "family_counts": dict(Counter(residual_family(row) for row in rows)),
            "task_family_by_id": task_family_by_id,
            "public_prompts_used": False,
            "public_tests_used": False,
            "public_solutions_used": False,
        },
    }

    if args.skip_run:
        payload.update({"status": "SKIPPED", "gates": [], "delta": {}})
        write_json(out, payload)
        markdown_out.write_text(markdown(payload), encoding="utf-8")
        return 0

    baseline = run_arm(
        name="baseline",
        enabled=False,
        binary=binary,
        checkpoint=checkpoint,
        manifest=manifest,
        empty_public=empty_public,
        seed=args.seed,
        candidates_per_task=args.candidates_per_task,
        sts_streams=sts_streams,
        task_family_by_id=task_family_by_id,
        manifest_rows_by_id=manifest_rows_by_id,
        semantic_test_timeout_seconds=args.semantic_test_timeout_seconds,
    )
    patched = run_arm(
        name="patched",
        enabled=True,
        binary=binary,
        checkpoint=checkpoint,
        manifest=manifest,
        empty_public=empty_public,
        seed=args.seed,
        candidates_per_task=args.candidates_per_task,
        sts_streams=sts_streams,
        task_family_by_id=task_family_by_id,
        manifest_rows_by_id=manifest_rows_by_id,
        semantic_test_timeout_seconds=args.semantic_test_timeout_seconds,
    )
    delta = delta_report(baseline, patched)
    baseline_elapsed = float(baseline.get("elapsed_seconds") or 0.0)
    patched_elapsed = float(patched.get("elapsed_seconds") or 0.0)
    speed_ratio = patched_elapsed / baseline_elapsed if baseline_elapsed > 0.0 else 1.0
    baseline_receiver_rate = float(baseline["summary"].get("private_receiver_eligible_task_rate") or 0.0)
    patched_receiver_rate = float(patched["summary"].get("private_receiver_eligible_task_rate") or 0.0)
    receiver_coverage_lift_ready = (
        delta["private_receiver_eligible_task_rate_delta"] >= 0.03
        or (
            baseline_receiver_rate >= 0.97
            and patched_receiver_rate >= baseline_receiver_rate
            and patched["summary"]["no_admissible_task_count"] <= baseline["summary"]["no_admissible_task_count"]
        )
    )
    bridge_shadow_row_delta = max(0, int(delta["private_to_public_receiver_bridge_shadow_row_count_delta"]))
    bridge_shadow_speed_allowance_seconds = bridge_shadow_row_delta * 4.5
    patched_speed_ceiling = max(0.25, baseline_elapsed * 1.10 + bridge_shadow_speed_allowance_seconds)
    gates = [
        {
            "name": "baseline_completed",
            "passed": baseline["returncode"] == 0,
            "detail": f"returncode={baseline['returncode']}",
        },
        {
            "name": "patched_completed",
            "passed": patched["returncode"] == 0,
            "detail": f"returncode={patched['returncode']}",
        },
        {
            "name": "private_only",
            "passed": payload["manifest"]["public_task_count"] == 0,
            "detail": "public manifest is empty in both same-seed arms",
        },
        {
            "name": "no_public_candidates_emitted",
            "passed": baseline["summary"]["task_count"] > 0
            and len(read_jsonl(Path(baseline["public_candidate_out"]))) == 0
            and len(read_jsonl(Path(patched["public_candidate_out"]))) == 0,
            "detail": "both arms used an empty public manifest and emitted zero public candidates",
        },
        {
            "name": "candidate_distribution_changed",
            "passed": delta["broad_transfer_residual_row_count_delta"] > 0
            or delta["eligible_receiver_inventory_row_count_delta"] > 0
            or delta["private_to_public_receiver_bridge_shadow_row_count_delta"] > 0,
            "detail": (
                f"residual row delta={delta['broad_transfer_residual_row_count_delta']}; "
                f"eligible receiver row delta={delta['eligible_receiver_inventory_row_count_delta']}; "
                "private-to-public bridge-shadow row delta="
                f"{delta['private_to_public_receiver_bridge_shadow_row_count_delta']}"
            ),
        },
        {
            "name": "private_to_public_bridge_shadow_exercised",
            "passed": delta["private_to_public_receiver_bridge_shadow_task_count_delta"] > 0
            and patched["summary"]["private_to_public_receiver_bridge_shadow_task_count"] > 0,
            "detail": {
                "baseline_bridge_shadow_tasks": baseline["summary"]["private_to_public_receiver_bridge_shadow_task_count"],
                "patched_bridge_shadow_tasks": patched["summary"]["private_to_public_receiver_bridge_shadow_task_count"],
                "bridge_shadow_task_delta": delta["private_to_public_receiver_bridge_shadow_task_count_delta"],
                "bridge_shadow_row_delta": delta["private_to_public_receiver_bridge_shadow_row_count_delta"],
                "rule": "the public bridge must be exercised on private heldout before any public metadata fanout can claim bridge readiness",
            },
        },
        {
            "name": "private_receiver_eligible_coverage_lift",
            "passed": receiver_coverage_lift_ready,
            "detail": (
                "requires same-seed private receiver-eligible task-rate lift >= 0.03, "
                "or saturated baseline coverage with no receiver/no-admissible regression; "
                f"baseline={baseline_receiver_rate:.3f}; patched={patched_receiver_rate:.3f}; "
                f"delta={delta['private_receiver_eligible_task_rate_delta']:.3f}"
            ),
        },
        {
            "name": "targeted_private_heldout_non_regression",
            "passed": delta["passed_task_rate_delta"] >= 0.0 and delta["no_admissible_rate_delta"] <= 0.0,
            "detail": f"pass_delta={delta['passed_task_rate_delta']:.3f}; no_admissible_delta={delta['no_admissible_rate_delta']:.3f}",
        },
        {
            "name": "patched_eligible_candidates_are_grammar_masked_learned_tokens",
            "passed": patched["summary"]["private_receiver_eligible_candidate_count"] > 0
            and patched["summary"]["non_grammar_private_receiver_eligible_candidate_count"] == 0
            and patched["summary"]["grammar_masked_learned_token_candidate_count"] >= patched["summary"]["private_receiver_eligible_candidate_count"],
            "detail": {
                "grammar_masked_learned_token_candidate_count": patched["summary"]["grammar_masked_learned_token_candidate_count"],
                "private_receiver_eligible_candidate_count": patched["summary"]["private_receiver_eligible_candidate_count"],
                "non_grammar_private_receiver_eligible_candidate_count": patched["summary"]["non_grammar_private_receiver_eligible_candidate_count"],
                "prompt_program_candidate_count": patched["summary"]["prompt_program_candidate_count"],
            },
        },
        {
            "name": "private_semantic_tests_materialized",
            "passed": payload["manifest"]["private_semantic_test_task_count"] == payload["manifest"]["task_count"]
            and patched["summary"]["semantic_tests"]["semantic_tested_task_count"] == payload["manifest"]["task_count"],
            "detail": {
                "manifest_private_semantic_test_task_count": payload["manifest"]["private_semantic_test_task_count"],
                "patched_semantic_tested_task_count": patched["summary"]["semantic_tests"]["semantic_tested_task_count"],
            },
        },
        {
            "name": "private_semantic_correctness_lift",
            "passed": delta["semantic_test_passed_task_rate_delta"] > 0.0,
            "detail": (
                "requires same-seed private behavioral-test pass-rate lift; "
                f"delta={delta['semantic_test_passed_task_rate_delta']:.3f}; "
                f"count_delta={delta['semantic_test_passed_task_count_delta']}"
            ),
        },
        {
            "name": "target_families_have_semantic_delta",
            "passed": target_family_semantic_delta_ready(delta),
            "detail": delta.get("semantic_task_family_deltas", {}),
        },
        {
            "name": "patched_no_admissible_zero",
            "passed": patched["summary"]["no_admissible_task_count"] == 0,
            "detail": f"patched_no_admissible_task_count={patched['summary']['no_admissible_task_count']}",
        },
        {
            "name": "no_fanout_speed_regression",
            "passed": patched_elapsed <= patched_speed_ceiling,
            "detail": (
                f"baseline_elapsed_seconds={baseline_elapsed:.3f}; "
                f"patched_elapsed_seconds={patched_elapsed:.3f}; speed_ratio={speed_ratio:.3f}; "
                f"bridge_shadow_row_delta={bridge_shadow_row_delta}; "
                f"bridge_shadow_speed_allowance_seconds={bridge_shadow_speed_allowance_seconds:.3f}; "
                f"patched_speed_ceiling_seconds={patched_speed_ceiling:.3f}"
            ),
        },
        {
            "name": "targeted_private_heldout_delta",
            "passed": delta["passed_task_rate_delta"] > 0.0
            or delta["private_receiver_eligible_task_rate_delta"] >= 0.03
            or delta["broad_transfer_residual_task_count_delta"] > 0
            or delta["eligible_receiver_inventory_task_count_delta"] > 0
            or delta["private_to_public_receiver_bridge_shadow_task_count_delta"] > 0,
            "detail": (
                "requires private pass-rate lift, receiver-eligible lift, same-seed router task "
                "coverage, or private bridge-shadow task coverage"
            ),
        },
        {
            "name": "task_family_metrics_materialized",
            "passed": bool(delta.get("task_family_deltas")),
            "detail": delta.get("task_family_deltas", {}),
        },
    ]
    status = "GREEN" if all(gate["passed"] for gate in gates) else "YELLOW"
    payload.update(
        {
            "trigger_state": status,
            "status": status,
            "baseline": baseline,
            "patched": patched,
            "delta": delta,
            "gates": gates,
        }
    )
    write_json(out, payload)
    markdown_out.write_text(markdown(payload), encoding="utf-8")
    return 0 if status in {"GREEN", "YELLOW"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
