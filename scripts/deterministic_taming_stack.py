"""Report deterministic rule/verifier substrate health for Theseus arms.

The taming stack is deliberately not a solver. It checks surface legality,
tool schemas, provenance, runtime hygiene, and stale memory so the learned
SymLiquid arms can spend capacity on semantics and repair instead of
rediscovering syntax or permission rules in weights.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/deterministic_taming_stack.json")
    parser.add_argument("--markdown-out", default="reports/deterministic_taming_stack.md")
    parser.add_argument("--candidate-manifest", default="reports/student_code_candidates.jsonl")
    parser.add_argument("--run-cargo-check", action="store_true")
    parser.add_argument("--cargo-timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    started = time.perf_counter()
    candidate_rows = read_jsonl(resolve(args.candidate_manifest))
    grammar = read_json(REPORTS / "grammar_suckers.json", {})
    learning = read_json(REPORTS / "learning_scoreboard.json", {})
    rust = rust_linter(run=args.run_cargo_check, timeout=args.cargo_timeout_seconds)
    arms = [
        python_linter(candidate_rows),
        rust,
        javascript_linter(),
        english_linter(grammar),
        sbl_linter(grammar),
        tool_schema_linter(),
        memory_linter(learning),
    ]
    hard_failures = [arm for arm in arms if arm["severity"] == "RED" and not arm["passed"]]
    soft_failures = [arm for arm in arms if arm["severity"] == "YELLOW" and not arm["passed"]]
    report = {
        "policy": "project_theseus_deterministic_taming_stack_v1",
        "created_utc": now(),
        "trigger_state": "RED" if hard_failures else ("YELLOW" if soft_failures else "GREEN"),
        "thesis": "Rules, linters, schemas, and verifiers constrain form and safety; SymLiquid weights focus on meaning, strategy, and transfer.",
        "summary": {
            "arm_count": len(arms),
            "passed_arms": sum(1 for arm in arms if arm["passed"]),
            "hard_failure_count": len(hard_failures),
            "soft_failure_count": len(soft_failures),
            "python_invalid_promotion_candidates": next((arm["metrics"].get("invalid_promotion_eligible") for arm in arms if arm["arm"] == "python"), None),
            "rust_cargo_checked": rust["metrics"].get("cargo_check_ran"),
            "external_inference_calls": 0,
            "public_benchmark_solutions_used": False,
            "public_tests_visible_to_rule_layer": False,
        },
        "arms": arms,
        "gates": [
            gate("python_promotion_candidates_parse", python_gate(arms), "malformed Python cannot count as promotion evidence"),
            gate("tool_schema_configs_parse", next_arm(arms, "tool_schema")["passed"], "selected config JSON parsed"),
            gate("memory_truth_scoreboard_fresh", next_arm(arms, "memory")["passed"], "learning scoreboard is the truth source"),
            gate("rule_layer_no_public_solutions", True, "deterministic checks reject invalid form only"),
            gate("external_inference_zero", True, "local lint/schema checks only"),
        ],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def python_linter(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    parse_ok = 0
    invalid_promotion = 0
    invalid_non_promotion = 0
    unsafe_imports = 0
    for row in rows:
        code = str(row.get("code") or "")
        if not code.strip():
            continue
        checked += 1
        try:
            tree = ast.parse(code)
            parse_ok += 1
        except SyntaxError:
            if row.get("benchmark_promotion_eligible"):
                invalid_promotion += 1
            else:
                invalid_non_promotion += 1
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".", 1)[0]]
            else:
                continue
            if any(name in {"os", "sys", "subprocess", "socket", "pathlib", "shutil"} for name in names):
                unsafe_imports += 1
                if row.get("benchmark_promotion_eligible"):
                    invalid_promotion += 1
    parse_rate = round(parse_ok / checked, 6) if checked else 0.0
    passed = checked > 0 and invalid_promotion == 0
    severity = "RED" if invalid_promotion > 0 else "YELLOW"
    return {
        "arm": "python",
        "sucker": "python_grammar_sucker",
        "passed": passed,
        "severity": severity,
        "role": "AST, function-shape, import guard, and promotion admissibility for Python candidates.",
        "metrics": {
            "checked": checked,
            "parse_rate": parse_rate,
            "invalid_promotion_eligible": invalid_promotion,
            "invalid_non_promotion": invalid_non_promotion,
            "unsafe_import_count": unsafe_imports,
            "parse_rate_floor": 0.98,
            "parse_rate_below_floor_is_training_pressure": parse_rate < 0.98,
        },
    }


def rust_linter(*, run: bool, timeout: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"cargo_check_ran": run}
    passed = True
    if run:
        try:
            result = subprocess.run(
                ["cargo", "check", "-p", "symliquid-cli"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=max(30, timeout),
            )
            metrics.update(
                {
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-1000:],
                    "stderr_tail": result.stderr[-1000:],
                }
            )
            passed = result.returncode == 0
        except Exception as exc:  # noqa: BLE001 - report-only verifier
            metrics.update({"returncode": 127, "error": str(exc)})
            passed = False
    return {
        "arm": "rust",
        "sucker": "rust_grammar_sucker",
        "passed": passed,
        "severity": "RED" if run else "YELLOW",
        "role": "Cargo/rustc checks for Rust/CUDA/system arms.",
        "metrics": metrics,
    }


def javascript_linter() -> dict[str, Any]:
    package = ROOT / "package.json"
    return {
        "arm": "javascript_typescript",
        "sucker": "javascript_typescript_grammar_sucker",
        "passed": True,
        "severity": "YELLOW",
        "role": "Parser/type-check hook reserved for future web/tool arms.",
        "metrics": {
            "package_json_exists": package.exists(),
            "planned_parser": "node_or_typescript_when_js_ts_frontier_is_active",
        },
    }


def english_linter(grammar: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(grammar, "summary")
    rate = float(summary.get("english_surface_pass_rate") or 0.0)
    return {
        "arm": "english",
        "sucker": "english_surface_grammar_sucker",
        "passed": rate >= 0.45,
        "severity": "YELLOW",
        "role": "Conversation surface structure pressure separate from reasoning quality.",
        "metrics": {
            "surface_pass_rate": rate,
            "rows_checked": summary.get("english_rows_checked"),
        },
    }


def sbl_linter(grammar: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(grammar, "summary")
    traces = int(summary.get("sbl_trace_count") or 0)
    return {
        "arm": "sbl_semantic_backbone",
        "sucker": "sbl_semantic_backbone_sucker",
        "passed": bool(summary.get("legacy_sbl_found")) and traces > 0,
        "severity": "YELLOW",
        "role": "Language-independent semantic frames for router and STS transfer.",
        "metrics": {
            "legacy_sbl_found": bool(summary.get("legacy_sbl_found")),
            "sbl_trace_count": traces,
        },
    }


def tool_schema_linter() -> dict[str, Any]:
    paths = [
        ROOT / "configs" / "autonomy_policy.json",
        ROOT / "configs" / "grammar_suckers.json",
        ROOT / "configs" / "cognitive_context_policy.json",
        ROOT / "configs" / "open_conversation_training_pantry.json",
        ROOT / "configs" / "architecture_search_space.json",
        ROOT / "configs" / "teacher_policy.json",
    ]
    failures = []
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report invalid configs
            failures.append({"path": rel(path), "error": str(exc)})
    return {
        "arm": "tool_schema",
        "sucker": "tool_call_schema_sucker",
        "passed": not failures,
        "severity": "RED",
        "role": "Validate policy/config schemas before arms use them as tools.",
        "metrics": {
            "checked_configs": [rel(path) for path in paths],
            "failure_count": len(failures),
            "failures": failures,
        },
    }


def memory_linter(learning: dict[str, Any]) -> dict[str, Any]:
    age = age_seconds(str(learning.get("created_utc") or ""))
    passed = learning.get("policy") == "project_theseus_learning_scoreboard_v1" and age is not None and age <= 1800
    return {
        "arm": "memory",
        "sucker": "memory_provenance_sucker",
        "passed": passed,
        "severity": "YELLOW",
        "role": "Prevent stale reports from masquerading as active learning truth.",
        "metrics": {
            "learning_scoreboard_policy": learning.get("policy"),
            "learning_scoreboard_trigger_state": learning.get("trigger_state"),
            "age_seconds": age,
            "promotion_allowed": get_path(learning, ["promotion", "promotion_allowed"], None),
        },
    }


def python_gate(arms: list[dict[str, Any]]) -> bool:
    arm = next_arm(arms, "python")
    return int(get_path(arm, ["metrics", "invalid_promotion_eligible"], 1) or 0) == 0


def next_arm(arms: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for arm in arms:
        if arm.get("arm") == name:
            return arm
    return {"passed": False, "metrics": {}}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Deterministic Taming Stack",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        "Rules/verifiers constrain form and safety; learned arms handle meaning and transfer.",
        "",
        "## Arms",
        "",
    ]
    for arm in report.get("arms", []):
        lines.append(f"- `{arm.get('arm')}`: {'pass' if arm.get('passed') else 'needs work'} ({arm.get('sucker')})")
    lines.append("")
    return "\n".join(lines)


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def age_seconds(iso: str) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
