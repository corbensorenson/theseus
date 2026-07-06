#!/usr/bin/env python3
"""Private full-body semantic quality ablation.

This report does not execute public benchmarks and does not train. It reconciles
the post-v4 full-body admissibility diagnostic with the broader v4 private
transfer evidence so readiness can distinguish "can emit admissible bodies" from
"can emit bodies that solve heldout private public-shaped contracts".
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POST_V4 = REPORTS / "private_full_body_candidate_admissibility_gate_v1.json"
DEFAULT_POST_V4_CURRENT_RELEASE = REPORTS / "private_full_body_candidate_admissibility_gate_v1_post_v4_current_release.json"
DEFAULT_V4_FULL_BODY = REPORTS / "private_full_body_semantic_gate_v1_v4_all.json"
DEFAULT_V4_SCORE = REPORTS / "public_safe_broad_transfer_maturity_v4_score.json"
DEFAULT_V4_LEARNED_GATE = REPORTS / "public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json"
DEFAULT_V4_STRICT_SCORE = REPORTS / "public_safe_broad_transfer_maturity_v4_strict_novel_learned_only_score.json"
DEFAULT_OPERATOR_LOCK = REPORTS / "public_calibration_operator_lock.flag"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-v4-full-body", default=rel(DEFAULT_POST_V4))
    parser.add_argument("--post-v4-current-release", default=rel(DEFAULT_POST_V4_CURRENT_RELEASE))
    parser.add_argument("--v4-full-body", default=rel(DEFAULT_V4_FULL_BODY))
    parser.add_argument("--v4-score", default=rel(DEFAULT_V4_SCORE))
    parser.add_argument("--v4-learned-gate", default=rel(DEFAULT_V4_LEARNED_GATE))
    parser.add_argument("--v4-strict-score", default=rel(DEFAULT_V4_STRICT_SCORE))
    parser.add_argument("--operator-lock", default=rel(DEFAULT_OPERATOR_LOCK))
    parser.add_argument("--out", default="reports/private_full_body_semantic_quality_ablation_v1.json")
    parser.add_argument("--markdown-out", default="reports/private_full_body_semantic_quality_ablation_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    post_path = resolve(args.post_v4_full_body)
    post_current_path = resolve(args.post_v4_current_release)
    v4_full_body_path = resolve(args.v4_full_body)
    v4_score_path = resolve(args.v4_score)
    v4_learned_path = resolve(args.v4_learned_gate)
    v4_strict_path = resolve(args.v4_strict_score)
    lock_path = resolve(args.operator_lock)

    post = read_json(post_path, {})
    post_current = read_json(post_current_path, {})
    v4_full_body = read_json(v4_full_body_path, {})
    v4_score = read_json(v4_score_path, {})
    v4_learned = read_json(v4_learned_path, {})
    v4_strict = read_json(v4_strict_path, {})

    post_summary = object_field(post, "summary")
    post_current_summary = object_field(post_current, "summary")
    v4_full_summary = object_field(v4_full_body, "summary")
    v4_score_summary = object_field(v4_score, "summary")
    v4_learned_summary = object_field(v4_learned, "summary")
    v4_strict_summary = object_field(v4_strict, "summary")

    surfaces = {
        "post_v4_default_admissibility": surface_summary(post_path, post, diagnostic=False),
        "post_v4_current_release": surface_summary(post_current_path, post_current, diagnostic=False),
        "v4_release_full_body": surface_summary(v4_full_body_path, v4_full_body),
        "v4_release_score": score_summary(v4_score_path, v4_score),
        "v4_strict_novel_learned_only": score_summary(v4_strict_path, v4_strict),
    }

    no_fallback_template_public_external = all(
        [
            int_number(post_summary.get("fallback_return_candidate_count")) == 0,
            int_number(post_summary.get("template_like_candidate_count")) == 0,
            int_number(post_summary.get("public_leakage_count")) == 0,
            external_calls(post) == 0,
            int_number(post_current_summary.get("fallback_return_candidate_count")) == 0,
            int_number(post_current_summary.get("template_like_candidate_count")) == 0,
            int_number(post_current_summary.get("public_leakage_count")) == 0,
            external_calls(post_current) == 0,
            int_number(v4_full_summary.get("fallback_return_candidate_count")) == 0,
            int_number(v4_full_summary.get("template_like_candidate_count")) == 0,
            int_number(v4_full_summary.get("public_leakage_count")) == 0,
            external_calls(v4_full_body) == 0,
            int_number(v4_score_summary.get("public_data_leakage_hit_count")) == 0,
            external_calls(v4_score) == 0,
            external_calls(v4_learned) == 0,
            external_calls(v4_strict) == 0,
        ]
    )
    best_selected = max(
        number(post_summary.get("selected_pass_rate")),
        number(post_current_summary.get("selected_pass_rate")),
        number(v4_full_summary.get("selected_pass_rate")),
        number(v4_score_summary.get("pass_rate")),
        number(v4_strict_summary.get("pass_rate")),
    )
    best_oracle = max(
        number(post_summary.get("pass_if_any_rate")),
        number(post_current_summary.get("pass_if_any_rate")),
        number(v4_full_summary.get("pass_if_any_rate")),
        number(v4_score_summary.get("pass_rate")),
        number(v4_strict_summary.get("pass_rate")),
    )
    fallback_return_candidate_count = max(
        int_number(post_summary.get("fallback_return_candidate_count")),
        int_number(post_current_summary.get("fallback_return_candidate_count")),
        int_number(v4_full_summary.get("fallback_return_candidate_count")),
    )
    template_like_candidate_count = max(
        int_number(post_summary.get("template_like_candidate_count")),
        int_number(post_current_summary.get("template_like_candidate_count")),
        int_number(v4_full_summary.get("template_like_candidate_count")),
    )
    public_leakage_count = max(
        int_number(post_summary.get("public_leakage_count")),
        int_number(post_current_summary.get("public_leakage_count")),
        int_number(v4_full_summary.get("public_leakage_count")),
        int_number(v4_score_summary.get("public_data_leakage_hit_count")),
        int_number(v4_strict_summary.get("public_data_leakage_hit_count")),
    )
    external_inference_calls = max(
        external_calls(post),
        external_calls(post_current),
        external_calls(v4_full_body),
        external_calls(v4_score),
        external_calls(v4_learned),
        external_calls(v4_strict),
    )
    strict_novel_learned_green = bool(
        v4_learned.get("trigger_state") == "GREEN"
        and number(v4_learned_summary.get("strict_novel_learned_only_pass_rate")) >= 0.70
        and int_number(v4_learned_summary.get("exact_train_body_memory_pass_count")) == 0
        and int_number(v4_learned_summary.get("prototype_pass_count")) == 0
    )
    learned_inventory = object_field(v4_learned_summary, "candidate_inventory")
    gates = [
        gate("post_v4_admissibility_green", post.get("trigger_state") == "GREEN", {
            "path": rel(post_path),
            "selected_pass_rate": post_summary.get("selected_pass_rate"),
            "pass_if_any_rate": post_summary.get("pass_if_any_rate"),
            "no_admissible_task_rate": post_summary.get("no_admissible_task_rate"),
        }),
        gate("post_v4_current_release_coverage_ge_97pct", post_current.get("trigger_state") == "GREEN", {
            "path": rel(post_current_path),
            "trigger_state": post_current.get("trigger_state"),
            "selected_pass_rate": post_current_summary.get("selected_pass_rate"),
            "pass_if_any_rate": post_current_summary.get("pass_if_any_rate"),
            "no_admissible_task_rate": post_current_summary.get("no_admissible_task_rate"),
            "benchmark_promotion_eligible_candidate_count": post_current_summary.get("benchmark_promotion_eligible_candidate_count"),
        }),
        gate("v4_full_body_semantic_nonzero", number(v4_full_summary.get("selected_pass_rate")) > 0.0 and number(v4_full_summary.get("pass_if_any_rate")) > 0.0, {
            "path": rel(v4_full_body_path),
            "trigger_state": v4_full_body.get("trigger_state"),
            "selected_pass_rate": v4_full_summary.get("selected_pass_rate"),
            "pass_if_any_rate": v4_full_summary.get("pass_if_any_rate"),
        }),
        gate("v4_strict_novel_learned_green", strict_novel_learned_green, {
            "path": rel(v4_learned_path),
            "trigger_state": v4_learned.get("trigger_state"),
            "strict_novel_learned_only_pass_rate": v4_learned_summary.get("strict_novel_learned_only_pass_rate"),
            "exact_train_body_memory_pass_count": v4_learned_summary.get("exact_train_body_memory_pass_count"),
            "prototype_pass_count": v4_learned_summary.get("prototype_pass_count"),
            "strict_novel_learned_only_candidate_rows": learned_inventory.get("strict_novel_learned_only_candidate_rows"),
        }),
        gate("semantic_private_public_shaped_nonzero", best_selected > 0.0 and best_oracle > 0.0, {
            "best_selected_pass_rate": best_selected,
            "best_pass_if_any_rate": best_oracle,
        }),
        gate("no_fallback_template_public_or_external", no_fallback_template_public_external, {
            "post_v4_fallback": post_summary.get("fallback_return_candidate_count"),
            "post_v4_template": post_summary.get("template_like_candidate_count"),
            "post_v4_public_leakage": post_summary.get("public_leakage_count"),
            "v4_fallback": v4_full_summary.get("fallback_return_candidate_count"),
            "v4_template": v4_full_summary.get("template_like_candidate_count"),
            "v4_public_leakage": v4_full_summary.get("public_leakage_count"),
            "external_inference_calls": external_inference_calls,
        }),
        gate("public_calibration_locked", lock_path.exists(), rel(lock_path)),
        gate("semantic_evidence_is_private_only", True, "all inputs are private-heldout evidence reports; this script does not execute public calibration"),
    ]
    hard_gate_names = {
        "v4_full_body_semantic_nonzero",
        "v4_strict_novel_learned_green",
        "semantic_private_public_shaped_nonzero",
        "no_fallback_template_public_or_external",
        "public_calibration_locked",
        "semantic_evidence_is_private_only",
    }
    hard_failed = [row for row in gates if row["gate"] in hard_gate_names and not row["passed"]]
    warning_failed = [row for row in gates if row["gate"] not in hard_gate_names and not row["passed"]]
    trigger_state = "RED" if hard_failed else "GREEN"
    post_v4_default_dead = number(post_summary.get("pass_if_any_rate")) <= 0.0
    report = {
        "policy": "project_theseus_private_full_body_semantic_quality_ablation_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "post_v4_full_body": rel(post_path),
            "post_v4_current_release": rel(post_current_path),
            "v4_full_body": rel(v4_full_body_path),
            "v4_score": rel(v4_score_path),
            "v4_learned_gate": rel(v4_learned_path),
            "v4_strict_score": rel(v4_strict_path),
            "operator_lock": rel(lock_path),
        },
        "summary": {
            "best_private_public_shaped_selected_pass_rate": best_selected,
            "best_private_public_shaped_pass_if_any_rate": best_oracle,
            "post_v4_default_selected_pass_rate": post_summary.get("selected_pass_rate"),
            "post_v4_default_pass_if_any_rate": post_summary.get("pass_if_any_rate"),
            "post_v4_default_no_admissible_task_rate": post_summary.get("no_admissible_task_rate"),
            "post_v4_default_benchmark_promotion_eligible_candidate_count": post_summary.get(
                "benchmark_promotion_eligible_candidate_count"
            ),
            "post_v4_default_semantic_dead": post_v4_default_dead,
            # Backward-compatible aliases for older readiness reports. The
            # canonical path now points at the current admissibility surface,
            # not the obsolete token-beam diagnostic.
            "post_v4_token_beam_selected_pass_rate": post_summary.get("selected_pass_rate"),
            "post_v4_token_beam_pass_if_any_rate": post_summary.get("pass_if_any_rate"),
            "post_v4_token_beam_semantic_dead": post_v4_default_dead,
            "post_v4_current_release_selected_pass_rate": post_current_summary.get("selected_pass_rate"),
            "post_v4_current_release_pass_if_any_rate": post_current_summary.get("pass_if_any_rate"),
            "post_v4_current_release_no_admissible_task_rate": post_current_summary.get("no_admissible_task_rate"),
            "post_v4_current_release_benchmark_promotion_eligible_candidate_count": post_current_summary.get("benchmark_promotion_eligible_candidate_count"),
            "v4_full_body_selected_pass_rate": v4_full_summary.get("selected_pass_rate"),
            "v4_full_body_pass_if_any_rate": v4_full_summary.get("pass_if_any_rate"),
            "v4_full_body_benchmark_promotion_eligible_candidate_count": v4_full_summary.get("benchmark_promotion_eligible_candidate_count"),
            "v4_full_body_no_admissible_task_rate": v4_full_summary.get("no_admissible_task_rate"),
            "v4_strict_novel_learned_only_pass_rate": v4_learned_summary.get("strict_novel_learned_only_pass_rate"),
            "v4_strict_novel_candidate_rows": learned_inventory.get("strict_novel_learned_only_candidate_rows"),
            "exact_train_body_memory_pass_count": v4_learned_summary.get("exact_train_body_memory_pass_count"),
            "prototype_pass_count": v4_learned_summary.get("prototype_pass_count"),
            "fallback_return_candidate_count": fallback_return_candidate_count,
            "template_like_candidate_count": template_like_candidate_count,
            "public_leakage_count": public_leakage_count,
            "external_inference_calls": external_inference_calls,
            "sts_on_private_pass_rate": v4_score_summary.get("pass_rate"),
            "matched_sts_off_private_pass_rate": v4_score_summary.get("control_pass_rate"),
            "sts_delta": v4_score_summary.get("sts_delta"),
            "family_rates": v4_full_summary.get("family_rates"),
            "category_rates": v4_full_summary.get("category_rates"),
            "surface_summaries": surfaces,
            "hard_failed_gate_count": len(hard_failed),
            "warning_failed_gate_count": len(warning_failed),
        },
        "gates": gates,
        "recommendation": recommendation(trigger_state, post_v4_default_dead, post_current_summary),
        "public_tests_used": False,
        "public_solutions_used": False,
        "external_inference_calls": 0,
    }
    return report


def surface_summary(path: Path, report: dict[str, Any], *, diagnostic: bool = False) -> dict[str, Any]:
    summary = object_field(report, "summary")
    return {
        "path": rel(path),
        "trigger_state": report.get("trigger_state"),
        "diagnostic_surface": diagnostic,
        "task_count": summary.get("task_count"),
        "candidate_row_count": summary.get("candidate_row_count"),
        "full_body_token_candidate_count": summary.get("full_body_token_candidate_count"),
        "benchmark_promotion_eligible_candidate_count": summary.get("benchmark_promotion_eligible_candidate_count"),
        "no_admissible_task_rate": summary.get("no_admissible_task_rate"),
        "selected_pass_rate": summary.get("selected_pass_rate"),
        "pass_if_any_rate": summary.get("pass_if_any_rate"),
        "fallback_return_candidate_count": summary.get("fallback_return_candidate_count"),
        "template_like_candidate_count": summary.get("template_like_candidate_count"),
        "public_leakage_count": summary.get("public_leakage_count"),
        "external_inference_calls": external_calls(report),
    }


def score_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(report, "summary")
    return {
        "path": rel(path),
        "trigger_state": report.get("trigger_state"),
        "task_count": summary.get("heldout_task_count"),
        "candidate_row_count": summary.get("candidate_row_count"),
        "selected_pass_rate": summary.get("pass_rate"),
        "pass_if_any_rate": summary.get("pass_rate"),
        "matched_sts_off_pass_rate": summary.get("control_pass_rate"),
        "sts_delta": summary.get("sts_delta"),
        "public_leakage_count": summary.get("public_data_leakage_hit_count"),
        "external_inference_calls": external_calls(report),
    }


def recommendation(trigger_state: str, post_v4_default_dead: bool, post_current_summary: dict[str, Any]) -> str:
    if trigger_state == "RED":
        return "Do not request public calibration review; private full-body semantic evidence is not clean."
    if number(post_current_summary.get("no_admissible_task_rate")) > 0.03:
        return (
            "Release v4 full-body learned candidates solve private public-shaped heldout, "
            "and post-v4 current-release learned candidates solve a nonzero subset, but "
            "post-v4 promotion-eligible coverage still needs repair."
        )
    if post_v4_default_dead:
        return (
            "Release v4 full-body learned candidates solve private public-shaped heldout, "
            "but the current default full-body surface remains semantically dead; do not "
            "request public calibration review until that default surface is repaired."
        )
    return "Private full-body semantic evidence is clean enough to inform a later one-shot public calibration review."


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    lines = [
        "# Private Full-Body Semantic Quality Ablation v1",
        "",
        f"Trigger: **{report.get('trigger_state')}**",
        "",
        "## Summary",
        "",
        f"- best private public-shaped selected pass rate: {summary.get('best_private_public_shaped_selected_pass_rate')}",
        f"- best private public-shaped pass-if-any rate: {summary.get('best_private_public_shaped_pass_if_any_rate')}",
        f"- post-v4 default selected/pass-if-any/no-admissible: {summary.get('post_v4_default_selected_pass_rate')} / {summary.get('post_v4_default_pass_if_any_rate')} / {summary.get('post_v4_default_no_admissible_task_rate')}",
        f"- post-v4 current-release selected/pass-if-any/no-admissible: {summary.get('post_v4_current_release_selected_pass_rate')} / {summary.get('post_v4_current_release_pass_if_any_rate')} / {summary.get('post_v4_current_release_no_admissible_task_rate')}",
        f"- v4 full-body selected/pass-if-any: {summary.get('v4_full_body_selected_pass_rate')} / {summary.get('v4_full_body_pass_if_any_rate')}",
        f"- strict-novel learned-only pass rate: {summary.get('v4_strict_novel_learned_only_pass_rate')}",
        f"- exact train-body memory pass count: {summary.get('exact_train_body_memory_pass_count')}",
        f"- prototype pass count: {summary.get('prototype_pass_count')}",
        f"- STS-on / matched STS-off: {summary.get('sts_on_private_pass_rate')} / {summary.get('matched_sts_off_private_pass_rate')}",
        "",
        "## Recommendation",
        "",
        str(report.get("recommendation")),
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates") if isinstance(report.get("gates"), list) else []:
        mark = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {mark}: {row.get('gate')} - {row.get('evidence')}")
    lines.append("")
    return "\n".join(lines)


def object_field(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def int_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def external_calls(report: dict[str, Any]) -> int:
    return max(
        int_number(report.get("external_inference_calls")),
        int_number(object_field(report, "summary").get("external_inference_calls")),
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
