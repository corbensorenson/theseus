"""Summarize governed training and benchmark resources staged for Theseus.

The runway report is intentionally conservative: it separates private training
pressure from public calibration assets, calls out license/usage gates, and
does not download anything. Other scripts stage resources; this script tells the
autonomy loop what is ready to use and what must remain eval-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D_ROOT = Path("D:/ProjectTheseus")


TRAINING_FILES = {
    "open_conversation_sft": DEFAULT_D_ROOT
    / "training_data/open_conversation_pantry/private_train/conversation_sft_pressure.jsonl",
    "open_conversation_sts": DEFAULT_D_ROOT
    / "training_data/open_conversation_pantry/sts_streams/conversation_sts_streams.jsonl",
    "open_code_expressions": DEFAULT_D_ROOT
    / "training_data/open_code_pantry/private_train/open_code_expressions.jsonl",
    "residual_code_curriculum": DEFAULT_D_ROOT
    / "training_data/residual_code_curriculum/private_train/residual_code_lm_tasks.jsonl",
    "residual_code_curriculum_statefix": DEFAULT_D_ROOT
    / "training_data/residual_code_curriculum/private_train/residual_code_lm_tasks_statefix.jsonl",
    "repo_repair_tasks": DEFAULT_D_ROOT
    / "training_data/long_horizon_programming/private_train/repo_repair_tasks.jsonl",
    "repo_repair_code_lm": DEFAULT_D_ROOT
    / "training_data/long_horizon_programming/private_train/repo_repair_code_lm_rows.jsonl",
    "repo_repair_sts": DEFAULT_D_ROOT
    / "training_data/long_horizon_programming/sts/repo_repair_sts_rows.jsonl",
    "governed_web_sample": DEFAULT_D_ROOT
    / "training_data/governed_samples/approved_training_mix.jsonl",
    "governed_pairwise_sample": DEFAULT_D_ROOT
    / "training_data/governed_samples/approved_pairwise_distill.jsonl",
}

PUBLIC_BENCHMARK_REPORTS = {
    "evalplus": ROOT / "reports/stage_evalplus_public_data.json",
    "bigcodebench_livecodebench": ROOT / "reports/public_code_benchmark_data_stage.json",
}

PUBLIC_CALIBRATION_FILES = {
    "humaneval": DEFAULT_D_ROOT / "resource_pantry/git/human_eval/data/HumanEval.jsonl.gz",
    "mbpp": DEFAULT_D_ROOT / "resource_pantry/git/mbpp/sanitized-mbpp.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-root", default=str(DEFAULT_D_ROOT))
    parser.add_argument("--out", default="reports/training_resource_runway.json")
    parser.add_argument("--markdown-out", default="reports/training_resource_runway.md")
    args = parser.parse_args()

    d_root = Path(args.d_root)
    resource_root = d_root / "resource_pantry"
    training_root = d_root / "training_data"
    source_catalog = read_json(ROOT / "reports/online_source_catalog_report.json")
    resource_pantry = read_json(ROOT / "reports/resource_pantry.json")
    sampler = read_json(ROOT / "reports/training_data_sampler.json")
    conversation = read_json(ROOT / "reports/open_conversation_training_pantry.json")
    inventory = read_json(ROOT / "reports/training_data_inventory.json")
    evalplus = read_json(PUBLIC_BENCHMARK_REPORTS["evalplus"])
    public_code = read_json(PUBLIC_BENCHMARK_REPORTS["bigcodebench_livecodebench"])
    adapter_factory = read_json(ROOT / "reports/benchmark_adapter_factory.json")
    rl_registry = read_json(ROOT / "reports/rl_benchmark_registry.json")
    benchmaxx = read_json(ROOT / "reports/benchmaxx_curriculum.json")

    source_repo_summary = summarize_source_repos(resource_root / "git")
    benchmark_dataset_summary = summarize_benchmark_datasets(resource_root / "datasets")
    training_summary = summarize_training_files()
    catalog_summary = summarize_catalog(source_catalog)
    pantry_summary = summarize_pantry(resource_pantry)
    public_benchmark_summary = summarize_public_benchmarks(evalplus, public_code)
    adapter_summary = summarize_adapter_factory(adapter_factory)
    rl_summary = summarize_rl_registry(rl_registry)
    safety_checks = build_safety_checks(
        d_root=d_root,
        source_catalog=source_catalog,
        resource_pantry=resource_pantry,
        sampler=sampler,
        conversation=conversation,
        evalplus=evalplus,
        public_code=public_code,
    )
    readiness_checks = build_readiness_checks(
        source_repo_summary=source_repo_summary,
        benchmark_dataset_summary=benchmark_dataset_summary,
        training_summary=training_summary,
        public_benchmark_summary=public_benchmark_summary,
        adapter_summary=adapter_summary,
        rl_summary=rl_summary,
    )
    trigger_state = state_from_checks(safety_checks, readiness_checks)

    report = {
        "policy": "project_theseus_training_resource_runway_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "d_root": str(d_root).replace("\\", "/"),
            "resource_root": str(resource_root).replace("\\", "/"),
            "training_root": str(training_root).replace("\\", "/"),
            "source_repo_count": source_repo_summary["repo_count"],
            "benchmark_dataset_groups": benchmark_dataset_summary["group_count"],
            "benchmark_dataset_bytes": benchmark_dataset_summary["total_bytes"],
            "private_training_rows": training_summary["private_training_rows"],
            "sts_training_rows": training_summary["sts_rows"],
            "governed_small_sample_rows": training_summary["governed_small_sample_rows"],
            "public_eval_task_rows": public_benchmark_summary["task_rows"],
            "catalog_imported_or_approved": catalog_summary["imported_or_approved"],
            "catalog_blocked_or_queued": catalog_summary["blocked_or_queued"],
            "resource_pantry_adapter_ready": pantry_summary["adapter_ready"],
            "benchmark_adapter_cards": adapter_summary["cards"],
            "benchmark_adapter_smoke_passed": adapter_summary["smoke_passed"],
            "local_rl_envs": rl_summary["local_env_count"],
            "inventory_bytes": get_path(inventory, ["summary", "bytes"], 0),
            "near_term_focus": get_path(benchmaxx, ["global_rules", "near_term_focus"], ""),
            "next_frontier_runner": get_path(benchmaxx, ["next_frontier", "runner_family"], ""),
        },
        "source_repos": source_repo_summary,
        "benchmark_datasets": benchmark_dataset_summary,
        "training_files": training_summary,
        "public_benchmark_assets_eval_only": public_benchmark_summary,
        "catalog": catalog_summary,
        "resource_pantry": pantry_summary,
        "benchmark_adapter_factory": adapter_summary,
        "rl_benchmark_registry": rl_summary,
        "safety_checks": safety_checks,
        "readiness_checks": readiness_checks,
        "next_actions": next_actions(trigger_state, readiness_checks, safety_checks, benchmaxx),
        "score_semantics": {
            "public_benchmark_assets": "calibration/scoring only; never private training rows",
            "private_training_assets": "allowlisted or locally generated pressure rows; not public promotion evidence by themselves",
            "source_repos": "adapter/source runway; each benchmark still needs clean adapter evidence before promotion",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_markdown(resolve(args.markdown_out), report)
    print(json.dumps(compact_console_report(report), indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 2


def summarize_source_repos(git_root: Path) -> dict[str, Any]:
    repos: list[dict[str, Any]] = []
    if git_root.exists():
        for path in sorted(p for p in git_root.iterdir() if p.is_dir()):
            repos.append(
                {
                    "name": path.name,
                    "path": str(path).replace("\\", "/"),
                    "is_git_repo": (path / ".git").exists(),
                }
            )
    return {
        "root": str(git_root).replace("\\", "/"),
        "repo_count": len(repos),
        "git_repo_count": sum(1 for repo in repos if repo["is_git_repo"]),
        "sample": repos[:40],
    }


def summarize_benchmark_datasets(dataset_root: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "bytes": 0, "sample": []})
    total_files = 0
    total_bytes = 0
    if dataset_root.exists():
        for path in sorted(p for p in dataset_root.rglob("*") if p.is_file()):
            rel = path.relative_to(dataset_root)
            group = rel.parts[0] if rel.parts else path.stem
            size = path.stat().st_size
            total_files += 1
            total_bytes += size
            groups[group]["files"] += 1
            groups[group]["bytes"] += size
            if len(groups[group]["sample"]) < 8:
                groups[group]["sample"].append(str(path).replace("\\", "/"))
    return {
        "root": str(dataset_root).replace("\\", "/"),
        "group_count": len(groups),
        "file_count": total_files,
        "total_bytes": total_bytes,
        "groups": dict(sorted(groups.items())),
    }


def summarize_training_files() -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    private_rows = 0
    sts_rows = 0
    governed_rows = 0
    for name, path in TRAINING_FILES.items():
        count = count_jsonl(path)
        size = path.stat().st_size if path.exists() else 0
        is_sts = name.endswith("_sts") or name in {"open_conversation_sts", "repo_repair_sts"}
        is_governed = name.startswith("governed_")
        if is_sts:
            sts_rows += count
        elif is_governed:
            governed_rows += count
        else:
            private_rows += count
        rows[name] = {
            "path": str(path).replace("\\", "/"),
            "exists": path.exists(),
            "rows": count,
            "bytes": size,
            "usage": training_usage(name),
        }
    return {
        "files": rows,
        "private_training_rows": private_rows,
        "sts_rows": sts_rows,
        "governed_small_sample_rows": governed_rows,
        "total_rows": private_rows + sts_rows + governed_rows,
    }


def summarize_catalog(report: dict[str, Any]) -> dict[str, Any]:
    sources = report.get("sources", [])
    decisions = Counter()
    categories = Counter()
    source_kinds = Counter()
    imported_or_approved = 0
    blocked_or_queued = 0
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            decision = str(source.get("decision") or "unknown")
            decisions[decision] += 1
            categories[str(source.get("category") or "unknown")] += 1
            source_kinds[str(source.get("source_kind") or "unknown")] += 1
            if decision in {"approved_for_catalog_import", "imported"} or source.get("staged"):
                imported_or_approved += 1
            if decision.startswith("blocked") or decision == "queued_only":
                blocked_or_queued += 1
    return {
        "trigger_state": report.get("trigger_state"),
        "source_count": len(sources) if isinstance(sources, list) else 0,
        "decision_counts": dict(sorted(decisions.items())),
        "category_counts": dict(sorted(categories.items())),
        "source_kind_counts": dict(sorted(source_kinds.items())),
        "imported_or_approved": imported_or_approved,
        "blocked_or_queued": blocked_or_queued,
        "errors": report.get("errors", []),
    }


def summarize_pantry(report: dict[str, Any]) -> dict[str, Any]:
    sources = report.get("sources", [])
    present = 0
    adapter_ready = 0
    statuses = Counter()
    categories = Counter()
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            if source.get("present"):
                present += 1
            if source.get("adapter_ready"):
                adapter_ready += 1
            statuses[str(source.get("status") or "unknown")] += 1
            categories[str(source.get("category") or "unknown")] += 1
    return {
        "trigger_state": report.get("trigger_state"),
        "source_count": len(sources) if isinstance(sources, list) else 0,
        "present": present,
        "adapter_ready": adapter_ready,
        "status_counts": dict(sorted(statuses.items())),
        "category_counts": dict(sorted(categories.items())),
        "hard_errors": report.get("hard_errors", []),
        "safety": report.get("safety", {}),
    }


def summarize_adapter_factory(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "cards": int(summary.get("cards") or 0),
        "ready_cards": int(summary.get("ready_cards") or 0),
        "smoke_passed": int(summary.get("smoke_passed") or 0),
        "needs_smoke": int(summary.get("needs_smoke") or 0),
        "blocked": int(summary.get("blocked") or 0),
        "emulator_cards": int(summary.get("emulator_cards") or 0),
        "written_cards": int(summary.get("written_cards") or 0),
    }


def summarize_rl_registry(report: dict[str, Any]) -> dict[str, Any]:
    local_envs = get_path(report, ["local_rl_inventory", "local_envs"], [])
    kind_counts = Counter()
    if isinstance(local_envs, list):
        for item in local_envs:
            if isinstance(item, dict):
                kind_counts[str(item.get("kind") or "unknown")] += 1
    return {
        "local_env_count": len(local_envs) if isinstance(local_envs, list) else 0,
        "kind_counts": dict(sorted(kind_counts.items())),
        "updated_utc": report.get("updated_utc") or report.get("created_utc"),
    }


def summarize_public_benchmarks(evalplus: dict[str, Any], public_code: dict[str, Any]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    calibration_files: dict[str, Any] = {}
    evalplus_rows = int(evalplus.get("row_count") or 0)
    if evalplus:
        datasets["evalplus"] = {
            "available": evalplus_rows >= 32,
            "rows": evalplus_rows,
            "path": evalplus.get("jsonl", ""),
            "training_use_allowed": bool(evalplus.get("training_use_allowed")),
        }
    public_rows = public_code.get("rows", [])
    if isinstance(public_rows, list):
        for row in public_rows:
            if not isinstance(row, dict):
                continue
            dataset = str(row.get("dataset") or "unknown")
            entry = datasets.setdefault(dataset, {"available_files": 0, "rows": None, "paths": []})
            if row.get("available"):
                entry["available_files"] += 1
            if row.get("path"):
                entry["paths"].append(row.get("path"))
            entry["training_admission"] = get_path(public_code, ["training_admission", "use"], "")
    for name, path in PUBLIC_CALIBRATION_FILES.items():
        task_count = count_task_file(path)
        calibration_files[name] = {
            "available": path.exists() and task_count >= 32,
            "path": str(path).replace("\\", "/"),
            "tasks": task_count,
            "training_use_allowed": False,
        }
    return {
        "datasets": datasets,
        "calibration_files": calibration_files,
        "task_rows": evalplus_rows + sum(int(item.get("tasks") or 0) for item in calibration_files.values()),
        "stage_trigger_state": public_code.get("trigger_state"),
        "training_admission": public_code.get("training_admission", {}),
    }


def build_safety_checks(
    *,
    d_root: Path,
    source_catalog: dict[str, Any],
    resource_pantry: dict[str, Any],
    sampler: dict[str, Any],
    conversation: dict[str, Any],
    evalplus: dict[str, Any],
    public_code: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        check("d_drive_root", str(d_root).replace("\\", "/").upper().startswith("D:/"), str(d_root)),
        check("online_catalog_no_errors", not source_catalog.get("errors"), source_catalog.get("errors", [])),
        check("resource_pantry_no_hard_errors", not resource_pantry.get("hard_errors"), resource_pantry.get("hard_errors", [])),
        check(
            "public_code_benchmarks_eval_only",
            get_path(public_code, ["training_admission", "public_benchmark_solutions_or_tests_may_train"]) is False,
            public_code.get("training_admission", {}),
        ),
        check(
            "evalplus_training_disallowed",
            evalplus.get("training_use_allowed") is False,
            {"training_use_allowed": evalplus.get("training_use_allowed")},
        ),
        check(
            "open_conversation_public_eval_overlap_rejected",
            named_gate_passed(conversation, "no_public_eval_token_overlap_in_accepted_rows"),
            "rows containing public code eval tokens are rejected",
        ),
        check(
            "sampler_teacher_generation_not_used",
            get_path(sampler, ["governance", "teacher_generation_used"]) is False,
            get_path(sampler, ["governance", "teacher_generation_used"]),
        ),
        check(
            "commercial_rom_downloads_blocked",
            get_path(resource_pantry, ["safety", "commercial_rom_downloads"]) is False,
            get_path(resource_pantry, ["safety", "commercial_rom_downloads"]),
        ),
    ]
    return checks


def build_readiness_checks(
    *,
    source_repo_summary: dict[str, Any],
    benchmark_dataset_summary: dict[str, Any],
    training_summary: dict[str, Any],
    public_benchmark_summary: dict[str, Any],
    adapter_summary: dict[str, Any],
    rl_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = set(benchmark_dataset_summary.get("groups", {}).keys())
    datasets = public_benchmark_summary.get("datasets", {})
    return [
        check("source_repo_runway_present", source_repo_summary.get("repo_count", 0) >= 20, source_repo_summary.get("repo_count", 0)),
        check("benchmark_adapter_cards_present", adapter_summary.get("cards", 0) >= 50, adapter_summary),
        check("benchmark_adapter_smoke_passed_present", adapter_summary.get("smoke_passed", 0) >= 30, adapter_summary),
        check("rl_long_horizon_envs_present", rl_summary.get("local_env_count", 0) >= 20, rl_summary),
        check("humaneval_public_calibration_present", get_path(public_benchmark_summary, ["calibration_files", "humaneval", "available"]) is True, get_path(public_benchmark_summary, ["calibration_files", "humaneval"], {})),
        check("mbpp_public_calibration_present", get_path(public_benchmark_summary, ["calibration_files", "mbpp", "available"]) is True, get_path(public_benchmark_summary, ["calibration_files", "mbpp"], {})),
        check("evalplus_public_calibration_present", get_path(datasets, ["evalplus", "available"]) is True, datasets.get("evalplus", {})),
        check("bigcodebench_public_calibration_present", "bigcodebench" in groups, sorted(groups)),
        check("livecodebench_full_release_slice_present", get_path(datasets, ["livecodebench_code_generation_lite", "available_files"], 0) >= 3, datasets.get("livecodebench_code_generation_lite", {})),
        check("private_residual_code_rows_present", get_path(training_summary, ["files", "residual_code_curriculum", "rows"], 0) >= 720, get_path(training_summary, ["files", "residual_code_curriculum", "rows"], 0)),
        check("repo_repair_rows_present", get_path(training_summary, ["files", "repo_repair_code_lm", "rows"], 0) >= 48, get_path(training_summary, ["files", "repo_repair_code_lm", "rows"], 0)),
        check("open_conversation_rows_present", get_path(training_summary, ["files", "open_conversation_sft", "rows"], 0) > 0, get_path(training_summary, ["files", "open_conversation_sft", "rows"], 0)),
        check("sts_rows_present", training_summary.get("sts_rows", 0) > 0, training_summary.get("sts_rows", 0)),
    ]


def next_actions(
    trigger_state: str,
    readiness: list[dict[str, Any]],
    safety: list[dict[str, Any]],
    benchmaxx: dict[str, Any] | None = None,
) -> list[str]:
    actions: list[str] = []
    benchmaxx = benchmaxx or {}
    failed_safety = [item["gate"] for item in safety if not item["passed"]]
    failed_ready = [item["gate"] for item in readiness if not item["passed"]]
    if failed_safety:
        actions.append(f"Resolve safety gates before training use: {', '.join(failed_safety)}.")
    if failed_ready:
        actions.append(f"Fill runway readiness gaps: {', '.join(failed_ready)}.")
    if trigger_state == "GREEN":
        if str(get_path(benchmaxx, ["next_frontier", "runner_family"], "")) == "conversation_multiturn_local":
            actions.append(
                "Run the English multi-turn conversation/personality lane first; keep open conversation rows private pressure only."
            )
        actions.extend(
            [
                "Run broad-transfer closure on MBPP/EvalPlus first; keep public benchmark data calibration-only.",
                "Use private residual, repo-repair, open-code, and open-conversation rows for training pressure.",
                "Generate any larger future downloads through catalog allowlists and explicit usage cards before admission.",
            ]
        )
    return actions


def state_from_checks(safety: list[dict[str, Any]], readiness: list[dict[str, Any]]) -> str:
    if any(not item["passed"] for item in safety):
        return "RED"
    if any(not item["passed"] for item in readiness):
        return "YELLOW"
    return "GREEN"


def training_usage(name: str) -> str:
    if "eval" in name:
        return "unknown"
    if name.startswith("governed_"):
        return "tiny governed training sample; license-gated and overlap-checked"
    if name.endswith("_sts") or name in {"open_conversation_sts", "repo_repair_sts"}:
        return "STS/private stream conditioning pressure"
    if name.startswith("residual_code"):
        return "private generated code residual training pressure"
    if name.startswith("repo_repair"):
        return "private repo-repair curriculum"
    if name.startswith("open_conversation"):
        return "open conversation/personality/multi-turn training pressure"
    if name.startswith("open_code"):
        return "permissive open-code expression training pressure"
    return "private training pressure"


def named_gate_passed(report: dict[str, Any], name: str) -> bool:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return False
    return any(isinstance(item, dict) and item.get("gate") == name and item.get("passed") is True for item in checks)


def check(gate: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": gate, "passed": bool(passed), "evidence": evidence}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def count_task_file(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".gz":
        count = 0
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    if path.suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        if isinstance(raw, list):
            return len(raw)
        if isinstance(raw, dict):
            for key in ("tasks", "examples", "data"):
                value = raw.get(key)
                if isinstance(value, list):
                    return len(value)
            return 1
        return 0
    return count_jsonl(path)


def compact_console_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "safety_checks": report["safety_checks"],
        "readiness_checks": report["readiness_checks"],
        "next_actions": report["next_actions"],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Training Resource Runway",
        "",
        f"- State: **{report['trigger_state']}**",
        f"- Created UTC: `{report['created_utc']}`",
        f"- Source repos staged: `{summary['source_repo_count']}`",
        f"- Benchmark dataset groups: `{summary['benchmark_dataset_groups']}`",
        f"- Benchmark dataset bytes: `{summary['benchmark_dataset_bytes']}`",
        f"- Private training rows: `{summary['private_training_rows']}`",
        f"- STS rows: `{summary['sts_training_rows']}`",
        f"- Governed small-sample rows: `{summary['governed_small_sample_rows']}`",
        f"- Public eval rows counted directly: `{summary['public_eval_task_rows']}`",
        f"- Benchmark adapter cards: `{summary['benchmark_adapter_cards']}`",
        f"- Smoke-passed adapter cards: `{summary['benchmark_adapter_smoke_passed']}`",
        f"- Local RL environments: `{summary['local_rl_envs']}`",
        "",
        "## Safety Gates",
    ]
    for item in report["safety_checks"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{mark}` {item['gate']}: {short(item['evidence'])}")
    lines.append("")
    lines.append("## Readiness Gates")
    for item in report["readiness_checks"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- `{mark}` {item['gate']}: {short(item['evidence'])}")
    lines.append("")
    lines.append("## Dataset Groups")
    for name, item in report["benchmark_datasets"]["groups"].items():
        lines.append(f"- `{name}`: {item['files']} files, {item['bytes']} bytes")
    lines.append("")
    lines.append("## Source Categories")
    for name, count in report["catalog"].get("category_counts", {}).items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("## RL Registry")
    for name, count in report["rl_benchmark_registry"].get("kind_counts", {}).items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    lines.append("## Public Calibration Files")
    for name, item in report["public_benchmark_assets_eval_only"].get("calibration_files", {}).items():
        lines.append(f"- `{name}`: tasks={item['tasks']}, training_use_allowed={item['training_use_allowed']}")
    lines.append("")
    lines.append("## Training Files")
    for name, item in report["training_files"]["files"].items():
        lines.append(f"- `{name}`: {item['rows']} rows, usage={item['usage']}")
    lines.append("")
    lines.append("## Next Actions")
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def short(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
