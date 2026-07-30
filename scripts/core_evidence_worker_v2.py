#!/usr/bin/env python3
"""Bounded, completely local repository patch worker.

The model sees only the natural request, declared runtime context, authority
grant, and a git-free parent snapshot. It can use a small structured tool set;
it never receives a shell, target identity, hidden tests, or evaluator output.
All writes are confined to the disposable snapshot and returned as a unified
diff for independent evaluation.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "core_evidence_local_8b_worker.json"
ALLOWED_SUFFIXES = {
    ".css", ".html", ".js", ".json", ".md", ".py", ".qmd", ".rs", ".toml",
    ".ts", ".tsx", ".yml", ".yaml",
}
ALLOWED_TOP_LEVEL = {"configs", "crates", "docs", "examples", "scripts", "tests"}
VISIBLE_FIELDS = {
    "natural_request", "parent_source_commit", "allowed_runtime_context",
    "authority_grant",
}
FORBIDDEN_TERMS = {
    "target_commit", "source_task_id", "hidden_tests", "gold_effects", "solution",
    "expected", "answer_family", "evaluator_score", "required_constructs",
}
VERIFY_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"


class WorkerFault(ValueError):
    """A fail-closed worker boundary fault."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--events-out", default="")
    args = parser.parse_args()
    visible = read_json(Path(args.input))
    config = read_json(Path(args.config))
    event_path = Path(args.events_out) if args.events_out else None
    result = run_worker(
        visible,
        Path(args.snapshot_root),
        config,
        event_sink=(
            (lambda event: append_event(event_path, event))
            if event_path is not None
            else None
        ),
    )
    Path(args.out).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def run_worker(
    visible: dict[str, Any],
    snapshot_root: Path,
    config: dict[str, Any],
    *,
    generator: Callable[[list[dict[str, str]]], str] | None = None,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_inputs(visible, snapshot_root, config)
    request = str(visible["natural_request"])
    state = RepositoryState(snapshot_root, config, request=request)
    local_model = None
    if generator is None:
        local_model = LocalMlxModel(config["model"])
        generator = local_model.generate
    request_id = sha256_text(request)[:16]
    messages = initial_messages(visible, state, config)
    budgets = config["budgets"]
    format_faults = 0
    format_repairs = 0
    terminal_reason = "turn_budget_exhausted"
    previous_stagnant_signature: str | None = None
    repeated_stagnant_actions = 0
    for turn in range(1, int(budgets["maximum_agent_turns"]) + 1):
        generation_started = time.perf_counter()
        retained = int(budgets.get("maximum_retained_tool_turns") or 6) * 2
        active_messages = (
            messages
            if len(messages) <= 2 + retained
            else messages[:2] + messages[-retained:]
        )
        raw = generator(active_messages)
        generation_wall_ms = round(
            (time.perf_counter() - generation_started) * 1000.0, 3
        )
        generation_metrics = (
            dict(local_model.last_generation_metrics)
            if local_model is not None
            else {}
        )
        state.inference_calls += 1
        action_kind = "parse_fault"
        try:
            action = parse_action(raw)
            if action.pop("_format_repaired", False):
                format_repairs += 1
            action_kind = str(action.get("action") or "")
            action_detail = safe_action_detail(action)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            action_detail = {}
            format_faults += 1
            result = {
                "ok": False,
                "fault": f"{type(exc).__name__}:{exc}",
                "instruction": "Return exactly one valid JSON tool action.",
            }
            if format_faults > int(budgets["maximum_format_retries"]):
                terminal_reason = "format_retry_budget_exhausted"
                emit_event(event_sink, {
                    "turn": turn,
                    "request_id": request_id,
                    "action": action_kind,
                    "action_detail": action_detail,
                    "ok": False,
                    "generation_wall_ms": generation_wall_ms,
                    **generation_metrics,
                    "raw_response_sha256": sha256_text(raw),
                    "raw_response_characters": len(raw),
                    "raw_response_tail": raw[-240:],
                    "terminal": True,
                    "reason": terminal_reason,
                })
                break
        else:
            format_faults = 0
            try:
                result = state.execute(action)
            except WorkerFault as exc:
                state.actions.append({
                    "action": action_kind,
                    "ok": False,
                })
                fault = f"{type(exc).__name__}:{exc}"
                result = {
                    "ok": False,
                    "fault": fault,
                    "instruction": state.recovery_instruction(str(exc)),
                }
        if result.get("ok") is not True and "instruction" not in result:
            result["instruction"] = state.recovery_instruction(
                str(result.get("fault") or "unsuccessful_action")
            )
        stagnant_signature = sha256_text(json.dumps({
            "action": action_kind,
            "detail": action_detail,
            "fault": result.get("fault"),
            "changed_paths": state.changed_paths(),
            "verification_count": len(state.verification_receipts),
        }, sort_keys=True))
        if result.get("ok") is False:
            repeated_stagnant_actions = (
                repeated_stagnant_actions + 1
                if stagnant_signature == previous_stagnant_signature
                else 1
            )
            previous_stagnant_signature = stagnant_signature
        else:
            repeated_stagnant_actions = 0
            previous_stagnant_signature = None
        if repeated_stagnant_actions >= int(
            budgets["maximum_repeated_denied_actions"]
        ):
            result.update({
                "terminal": True,
                "reason": "stalled_repeated_denied_action",
                "instruction": (
                    "The same denied action reached the configured repetition "
                    "ceiling without "
                    "repository progress; stop honestly instead of wasting budget."
                ),
            })
        result["state"] = {
            "changed_paths": state.changed_paths(),
            "verification_count": len(state.verification_receipts),
            "last_verification_green": bool(
                state.verification_receipts
                and state.verification_receipts[-1]["passed"]
            ),
            "repair_attempts": state.repair_attempts,
            "format_repairs": format_repairs,
            "pre_mutation_inspection": {
                "read_actions": state.read_actions,
                "distinct_read_paths": sorted(state.read_paths),
                "test_read_done": any(
                    path.startswith("tests/") for path in state.read_paths
                ),
                "required_read_actions": int(
                    budgets["minimum_pre_mutation_read_actions"]
                ),
                "required_distinct_paths": int(
                    budgets["minimum_pre_mutation_distinct_paths"]
                ),
            },
            "remaining_turns": int(budgets["maximum_agent_turns"]) - turn,
            "required_next_phase": state.required_next_phase(),
        }
        emit_event(event_sink, {
            "turn": turn,
            "request_id": request_id,
            "action": action_kind,
            "action_detail": action_detail,
            "ok": result.get("ok") is True,
            "generation_wall_ms": generation_wall_ms,
            **generation_metrics,
            "raw_response_sha256": sha256_text(raw),
            "raw_response_characters": len(raw),
            "raw_response_tail": (
                raw[-240:] if action_kind == "parse_fault" else ""
            ),
            "changed_path_count": len(state.changed_paths()),
            "verification_count": len(state.verification_receipts),
            "terminal": result.get("terminal") is True,
            "fault": result.get("fault"),
        })
        messages.extend([
            {"role": "assistant", "content": raw[: int(budgets["maximum_tool_result_characters"])]},
            {
                "role": "user",
                "content": "TOOL_RESULT\n" + json.dumps(
                    result, sort_keys=True
                )[: int(budgets["maximum_tool_result_characters"])],
            },
        ])
        if result.get("terminal") is True:
            terminal_reason = str(result.get("reason") or "finished")
            break

    patch = state.unified_diff()
    patch_bytes = len(patch.encode("utf-8"))
    if patch_bytes > int(budgets["maximum_patch_bytes"]):
        patch = ""
        terminal_reason = "patch_budget_exceeded"
    changed_paths = state.changed_paths()
    verified = bool(state.verification_receipts) and state.verification_receipts[-1]["passed"]
    residuals = []
    if not patch:
        residuals.append("no_effect_capable_patch")
    if changed_paths and not verified:
        residuals.append("final_candidate_verification_not_green")
    if terminal_reason != "finished":
        residuals.append(terminal_reason)
    model_card = config["model"]
    return {
        "policy": "project_theseus_local_repository_worker_v2",
        "worker_id": "theseus_local_repository_worker_v2",
        "worker_kind": "bounded_local_learned_repository_patch_agent",
        "model_identity": {
            "repo_id": model_card["repo_id"],
            "revision": model_card["revision"],
            "snapshot_manifest_sha256": (
                local_model.snapshot_manifest_sha256 if local_model else "injected_test_generator"
            ),
            "runtime": "mlx_lm_local_metal",
        },
        "learned_generation_credit": 1,
        "local_model_inference_calls": state.inference_calls,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "user_facing_effects": 0,
        "natural_request_sha256": sha256_text(request),
        "parent_source_commit": str(visible["parent_source_commit"]),
        "patch_unified_diff": patch,
        "proposed_paths": changed_paths,
        "verification_commands": [
            command for receipt in state.verification_receipts
            for command in receipt["commands"]
        ],
        "verification_receipts": state.verification_receipts,
        "effect_inventory": state.effect_inventory(),
        "action_summary": state.action_summary(),
        "advisory_plan": state.advisory_plan,
        "repair_attempts": state.repair_attempts,
        "format_repairs": format_repairs,
        "terminal_reason": terminal_reason,
        "abstained": not bool(patch),
        "abstention_reason": state.abstention_reason,
        "residuals": residuals,
        "candidate_authored_success_flags": [],
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "non_claims": [
            "This local third-party model is not the Theseus student.",
            "Candidate verification is advisory; the hidden evaluator independently recomputes completion.",
        ],
    }


class RepositoryState:
    def __init__(
        self,
        root: Path,
        config: dict[str, Any],
        *,
        request: str = "",
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.baseline = text_inventory(self.root)
        self.request = request
        self.request_tokens = keywords(request)
        self.request_scoped_paths = request_effect_paths(
            self.baseline, request
        )
        self.actions: list[dict[str, Any]] = []
        self.verification_receipts: list[dict[str, Any]] = []
        self.inference_calls = 0
        self.mutations = 0
        self.repair_attempts = 0
        self.failed_verification_seen = False
        self.read_actions = 0
        self.navigation_actions = 0
        self.read_paths: set[str] = set()
        self.read_spans: set[tuple[str, int, int]] = set()
        self.abstention_reason: str | None = None
        self.advisory_plan: dict[str, Any] | None = None

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = str(action.get("action") or "")
        if kind in {"list", "search", "read"}:
            self.navigation_actions += 1
            maximum_navigation = self.config["budgets"].get(
                "maximum_pre_mutation_navigation_actions"
            )
            if (
                maximum_navigation is not None
                and not self.changed_paths()
                and self.navigation_actions > int(maximum_navigation)
            ):
                raise WorkerFault(
                    "pre_mutation_navigation_ceiling_reached"
                )
        if kind == "list":
            result = self.list_paths(str(action.get("prefix") or ""))
        elif kind == "search":
            result = self.search(str(action.get("query") or ""))
        elif kind == "read":
            result = self.read(
                str(action.get("path") or ""),
                int(action.get("start_line") or 1),
                int(action.get("end_line") or 240),
            )
        elif kind == "replace":
            result = self.replace(
                str(action.get("path") or ""),
                str(action.get("old") or ""),
                str(action.get("new") or ""),
            )
        elif kind == "create":
            result = self.create(
                str(action.get("path") or ""),
                str(action.get("content") or ""),
            )
        elif kind == "delete":
            result = self.delete(str(action.get("path") or ""))
        elif kind == "verify":
            result = self.verify(
                strings(action.get("pytest")),
                strings(action.get("py_compile")),
                strings(action.get("json")),
            )
        elif kind == "plan":
            result = self.record_plan(action)
        elif kind == "finish":
            if not self.changed_paths():
                return {"ok": False, "terminal": False, "fault": "no_changed_paths"}
            if not self.verification_receipts or not self.verification_receipts[-1]["passed"]:
                return {"ok": False, "terminal": False, "fault": "final_verification_not_green"}
            result = {"ok": True, "terminal": True, "reason": "finished"}
        elif kind == "abstain":
            discarded_paths = self.changed_paths()
            self.restore_baseline()
            self.abstention_reason = str(action.get("reason") or "")[:1000]
            result = {
                "ok": True,
                "terminal": True,
                "reason": "explicit_abstention",
                "abstention_reason": self.abstention_reason,
                "discarded_paths": discarded_paths,
            }
        else:
            raise WorkerFault(f"unknown_action:{kind}")
        self.actions.append({"action": kind, "ok": result.get("ok") is True})
        return result

    def safe_path(self, value: str, *, must_exist: bool = True) -> Path:
        pure = PurePosixPath(value)
        if (
            not value or pure.is_absolute() or ".." in pure.parts
            or not pure.parts or pure.parts[0] not in ALLOWED_TOP_LEVEL
            or pure.suffix.lower() not in ALLOWED_SUFFIXES
        ):
            raise WorkerFault("unsafe_or_unsupported_path")
        path = self.root.joinpath(*pure.parts)
        cursor = self.root
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise WorkerFault("symlink_path_forbidden")
        resolved_parent = path.parent.resolve()
        if not resolved_parent.is_relative_to(self.root):
            raise WorkerFault("path_escape")
        if must_exist and (not path.is_file() or path.is_symlink()):
            raise WorkerFault("file_missing_or_not_regular")
        return path

    def list_paths(self, prefix: str) -> dict[str, Any]:
        if prefix:
            pure = PurePosixPath(prefix)
            if pure.is_absolute() or ".." in pure.parts:
                raise WorkerFault("unsafe_prefix")
        paths = [
            path for path in sorted(text_inventory(self.root))
            if not prefix or path.startswith(prefix)
        ][: int(self.config["budgets"]["maximum_search_results"])]
        return {"ok": True, "paths": paths}

    def search(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query or len(query) > 200:
            raise WorkerFault("invalid_search_query")
        rows = []
        maximum = int(self.config["budgets"]["maximum_search_results"])
        inventory = text_inventory(self.root)
        for relative, text in inventory.items():
            for line_number, line in enumerate(text.splitlines(), 1):
                if query.casefold() in line.casefold():
                    rows.append({
                        "path": relative,
                        "line": line_number,
                        "text": line[:500],
                    })
                    if len(rows) >= maximum:
                        return {
                            "ok": True,
                            "matches": rows,
                            "truncated": True,
                            "match_mode": "exact_substring",
                        }
        if rows:
            return {
                "ok": True,
                "matches": rows,
                "truncated": False,
                "match_mode": "exact_substring",
            }
        terms = keywords(query)
        ranked = []
        for relative, text in inventory.items():
            path_text = relative.casefold().replace("_", " ").replace("-", " ")
            for line_number, line in enumerate(text.splitlines(), 1):
                lowered = line.casefold()
                hits = sum(token in lowered for token in terms)
                if not hits:
                    continue
                score = hits * 10 + sum(
                    token in path_text for token in terms
                ) * 3
                ranked.append((score, relative, line_number, line[:500]))
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        fuzzy = [
            {"path": path, "line": line_number, "text": line}
            for _, path, line_number, line in ranked[:maximum]
        ]
        return {
            "ok": True,
            "matches": fuzzy,
            "truncated": len(ranked) > maximum,
            "match_mode": "token_ranked_fallback",
        }

    def read(self, relative: str, start: int, end: int) -> dict[str, Any]:
        path = self.safe_path(relative)
        budgets = self.config["budgets"]
        maximum = int(budgets["maximum_read_lines"])
        maximum_pre_mutation_reads = int(
            budgets.get(
                "maximum_pre_mutation_read_actions",
                budgets["maximum_agent_turns"],
            )
        )
        if (
            not self.changed_paths()
            and self.read_actions >= maximum_pre_mutation_reads
        ):
            raise WorkerFault("pre_mutation_inspection_ceiling_reached")
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(1, start)
        end = min(max(start, end), start + maximum - 1)
        minimum = min(
            maximum,
            max(1, int(budgets.get("minimum_read_context_lines", 1))),
        )
        if end - start + 1 < minimum:
            # Normalize tiny reads to stable context blocks. This prevents a
            # local model from spending its whole budget crawling six lines
            # at a time while keeping repeated blocks detectable.
            start = ((start - 1) // minimum) * minimum + 1
            end = min(len(lines), start + minimum - 1)
        span = (relative, start, min(end, len(lines)))
        if span in self.read_spans:
            raise WorkerFault("duplicate_read_span_no_new_information")
        self.read_spans.add(span)
        self.read_actions += 1
        self.read_paths.add(relative)
        numbered = [f"{index}: {lines[index - 1]}" for index in range(start, min(end, len(lines)) + 1)]
        return {
            "ok": True,
            "path": relative,
            "start_line": start,
            "end_line": min(end, len(lines)),
            "content": "\n".join(numbered),
        }

    def required_next_phase(self) -> str:
        """Return controller-owned progress guidance, not a success claim."""
        changed = self.changed_paths()
        if self.failed_verification_seen:
            return "repair_from_last_failure_before_reverification"
        if changed and not self.verification_receipts:
            return "verify_changed_paths"
        if (
            changed
            and self.verification_receipts
            and self.verification_receipts[-1]["passed"]
        ):
            return "finish_or_make_only_request_required_additional_edits"
        if (
            self.inspection_complete()
            and self.config["budgets"].get("require_plan_before_mutation")
            and self.advisory_plan is None
        ):
            return "record_request_criteria_plan_before_edit"
        if self.inspection_complete():
            return "edit_now_or_abstain_with_specific_missing_information"
        return "inspect_request_relevant_implementation_config_and_test"

    def inspection_complete(self) -> bool:
        budgets = self.config["budgets"]
        return bool(
            self.read_actions
            >= int(budgets["minimum_pre_mutation_read_actions"])
            and len(self.read_paths)
            >= int(budgets["minimum_pre_mutation_distinct_paths"])
            and (
                not budgets["require_test_read_before_mutation"]
                or any(path.startswith("tests/") for path in self.read_paths)
            )
        )

    def before_mutation(self) -> None:
        budgets = self.config["budgets"]
        if self.read_actions < int(
            budgets["minimum_pre_mutation_read_actions"]
        ):
            raise WorkerFault("insufficient_pre_mutation_read_actions")
        if len(self.read_paths) < int(
            budgets["minimum_pre_mutation_distinct_paths"]
        ):
            raise WorkerFault("insufficient_pre_mutation_distinct_paths")
        if (
            budgets["require_test_read_before_mutation"]
            and not any(path.startswith("tests/") for path in self.read_paths)
        ):
            raise WorkerFault("test_read_required_before_mutation")
        if (
            budgets.get("require_plan_before_mutation")
            and self.advisory_plan is None
        ):
            raise WorkerFault("request_criteria_plan_required_before_mutation")
        if self.mutations >= int(budgets["maximum_file_mutations"]):
            raise WorkerFault("mutation_budget_exhausted")
        if self.failed_verification_seen:
            if self.repair_attempts >= int(budgets["maximum_repair_attempts"]):
                raise WorkerFault("repair_budget_exhausted")
            self.repair_attempts += 1
            self.failed_verification_seen = False
        self.mutations += 1

    def require_request_scoped_effect(self, relative: str) -> None:
        if (
            self.config["budgets"].get(
                "enforce_request_scoped_effect_paths", False
            )
            and relative not in self.request_scoped_paths
        ):
            raise WorkerFault(
                "effect_path_outside_request_scoped_authority"
            )

    def record_plan(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.inspection_complete():
            raise WorkerFault("plan_requires_complete_inspection")
        if self.changed_paths():
            raise WorkerFault("plan_must_precede_mutation")
        criteria = action.get("criteria")
        target_paths = action.get("target_paths")
        implementation = action.get("implementation")
        verification = action.get("verification")
        if (
            not isinstance(criteria, list)
            or not criteria
            or len(criteria) > 12
            or any(not isinstance(item, str) or not item.strip() for item in criteria)
            or not isinstance(target_paths, list)
            or not target_paths
            or len(target_paths) > 8
            or any(
                not isinstance(path, str) or path not in self.baseline
                for path in target_paths
            )
            or not isinstance(implementation, str)
            or not implementation.strip()
            or not isinstance(verification, str)
            or not verification.strip()
        ):
            raise WorkerFault("request_criteria_plan_schema_invalid")
        self.advisory_plan = {
            "criteria": [item.strip()[:500] for item in criteria],
            "target_paths": target_paths,
            "implementation": implementation.strip()[:2000],
            "verification": verification.strip()[:1000],
        }
        return {
            "ok": True,
            "advisory_only": True,
            "candidate_authored_success_claim": False,
            "plan": self.advisory_plan,
        }

    def relevant_paths(
        self,
        *,
        prefix: str = "",
        unread_only: bool = False,
        maximum: int = 4,
    ) -> list[str]:
        inventory = {
            path: text for path, text in self.baseline.items()
            if (not prefix or path.startswith(prefix))
            and (not unread_only or path not in self.read_paths)
        }
        return lexical_rank(inventory, self.request_tokens)[:maximum]

    def restore_baseline(self) -> None:
        """Discard every provisional text effect inside the disposable snapshot."""
        current = text_inventory(self.root)
        for relative in sorted(set(current) - set(self.baseline)):
            self.safe_path(relative).unlink()
        for relative, text in self.baseline.items():
            if current.get(relative) == text:
                continue
            path = self.safe_path(
                relative, must_exist=relative in current
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            write_normalized_text(path, text)
        self.failed_verification_seen = False

    def recovery_instruction(self, fault: str) -> str:
        unread = self.relevant_paths(unread_only=True)
        tests = self.relevant_paths(
            prefix="tests/", unread_only=True, maximum=3
        )
        if "test_read_required_before_mutation" in fault:
            return (
                "NEXT ACTION MUST BE read on an existing relevant test. "
                f"Target-blind suggestions: {tests or unread}. Do not create a test yet."
            )
        phase = self.required_next_phase()
        if phase == "repair_from_last_failure_before_reverification":
            return (
                "Candidate verification failed. NEXT ACTION MUST BE a bounded repair "
                "with replace/create/delete using the failure_summaries already "
                "returned. Do not read, search, verify, finish, or abandon the "
                "provisional patch."
            )
        if phase == "verify_changed_paths":
            return (
                "A provisional effect exists without verification. NEXT ACTION MUST "
                "BE verify with empty pytest, py_compile, and json arrays so the "
                "controller selects checks. Do not read, search, or mutate again."
            )
        if phase.startswith("finish_or"):
            return (
                "The current provisional effect has green candidate verification. "
                "NEXT ACTION MUST BE finish. Do not read, search, mutate, or verify "
                "again."
            )
        if "request_criteria_plan_required" in fault:
            return (
                "NEXT ACTION MUST BE plan. Copy every acceptance criterion from the "
                "natural request, name only inspected existing target paths, describe "
                "the general implementation, and state how it will be verified."
            )
        if "duplicate_read_span" in fault and self.inspection_complete():
            return (
                "The requested span was already returned and the required "
                "inspection is complete. NEXT ACTION MUST BE replace/create/delete "
                "using exact text already read, or abstain with the specific missing "
                "information. Do not read or search again."
            )
        if (
            "insufficient_pre_mutation" in fault
            or "duplicate_read_span" in fault
        ):
            return (
                "NEXT ACTION MUST BE read on a new relevant existing path. "
                f"Target-blind suggestions: {unread}."
            )
        if "pre_mutation_inspection_ceiling_reached" in fault:
            return (
                "Inspection reached its prospective budget. NEXT ACTION MUST BE "
                "replace/create/delete using exact text already read, or abstain "
                "with the specific missing information. Do not read or search again."
            )
        if "pre_mutation_navigation_ceiling_reached" in fault:
            return (
                "Navigation reached its prospective budget. NEXT ACTION MUST BE "
                + (
                    "plan using the inspected request-scoped paths."
                    if self.inspection_complete()
                    and self.advisory_plan is None
                    else (
                        "replace/create/delete within request_scoped_effect_paths, "
                        "or abstain with the specific missing information."
                        if self.inspection_complete()
                        else "abstain with the specific missing information."
                    )
                )
                + " Do not read, list, or search again."
            )
        if "effect_path_outside_request_scoped_authority" in fault:
            return (
                "The proposed effect path is outside the request-derived authority "
                f"boundary. Use only one of: {self.request_scoped_paths}."
            )
        if (
            "verification_requires_changed_paths" in fault
            or "no_changed_paths" in fault
        ):
            return (
                "Do not verify or finish again. Produce a valid replace/create/delete "
                "effect. If exact text is unavailable, read the intended file first."
            )
        if "verification_requires_repair_after_failure" in fault:
            return (
                "Do not repeat verification. Use the failure_summaries from the last "
                "receipt to make one bounded repair, then verify again."
            )
        if "final_verification_not_green" in fault:
            return (
                "Do not finish or repeat verification. Repair the failing behavior "
                "from the last failure_summaries, then verify."
            )
        if (
            "replace_requires_distinct_nonempty_old_text" in fault
            or "replace_old_occurrence_count" in fault
        ):
            return (
                "Read the intended file segment, then replace one exact, nonempty, "
                "uniquely occurring old string with distinct new text."
            )
        return (
            "Choose a different safe action that directly resolves this fault. "
            f"Relevant unread paths: {unread}."
        )

    def replace(self, relative: str, old: str, new: str) -> dict[str, Any]:
        if not old or old == new:
            raise WorkerFault("replace_requires_distinct_nonempty_old_text")
        path = self.safe_path(relative)
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(old)
        if occurrences != 1:
            raise WorkerFault(f"replace_old_occurrence_count:{occurrences}")
        self.require_request_scoped_effect(relative)
        self.before_mutation()
        write_normalized_text(path, text.replace(old, new, 1))
        return {"ok": True, "path": relative, "changed": True}

    def create(self, relative: str, content: str) -> dict[str, Any]:
        path = self.safe_path(relative, must_exist=False)
        if path.exists() or not content:
            raise WorkerFault("create_requires_new_nonempty_file")
        if (
            self.config["budgets"][
                "require_existing_integration_before_new_implementation"
            ]
            and PurePosixPath(relative).parts[0] in {"scripts", "crates"}
            and not any(
                changed in self.baseline for changed in self.changed_paths()
            )
        ):
            raise WorkerFault(
                "new_implementation_requires_existing_integration_effect"
            )
        self.require_request_scoped_effect(relative)
        self.before_mutation()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_normalized_text(path, content)
        return {"ok": True, "path": relative, "changed": True}

    def delete(self, relative: str) -> dict[str, Any]:
        path = self.safe_path(relative)
        self.require_request_scoped_effect(relative)
        self.before_mutation()
        path.unlink()
        return {"ok": True, "path": relative, "changed": True}

    def verify(
        self,
        pytest_targets: list[str],
        py_compile_targets: list[str],
        json_targets: list[str],
    ) -> dict[str, Any]:
        if not self.changed_paths():
            raise WorkerFault("verification_requires_changed_paths")
        if self.failed_verification_seen:
            raise WorkerFault("verification_requires_repair_after_failure")
        automatic = self.automatic_verification_targets()
        denied_targets: list[str] = []
        pytest_targets = self.accept_verification_suggestions(
            pytest_targets, ".py", "tests/", denied_targets
        )
        py_compile_targets = self.accept_verification_suggestions(
            py_compile_targets, ".py", "", denied_targets
        )
        json_targets = self.accept_verification_suggestions(
            json_targets, ".json", "", denied_targets
        )
        pytest_targets = sorted(set(pytest_targets) | set(automatic["pytest"]))
        py_compile_targets = sorted(
            set(py_compile_targets) | set(automatic["py_compile"])
        )
        json_targets = sorted(set(json_targets) | set(automatic["json"]))
        commands: list[list[str]] = []
        if pytest_targets:
            commands.append([
                VERIFY_PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                *pytest_targets,
            ])
        if py_compile_targets:
            commands.append([
                VERIFY_PYTHON, "-m", "py_compile", *py_compile_targets
            ])
        for target in json_targets:
            commands.append([VERIFY_PYTHON, "-m", "json.tool", target])
        if not commands:
            receipt = {
                "passed": True,
                "commands": ["internal:text_inventory_and_utf8_validation"],
                "results": [],
                "denied_candidate_targets": denied_targets,
                "selection": automatic,
                "wall_ms": 0.0,
            }
            self.verification_receipts.append(receipt)
            self.failed_verification_seen = False
            return {"ok": True, **receipt}
        started = time.perf_counter()
        rows = []
        passed = True
        for command in commands:
            process = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=int(self.config["budgets"]["verification_timeout_seconds"]),
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "NO_PROXY": "*",
                    "no_proxy": "*",
                },
                check=False,
            )
            passed = passed and process.returncode == 0
            rows.append({
                "argv": [relative_command_part(value, self.root) for value in command],
                "returncode": process.returncode,
                "stdout_tail": process.stdout[-4000:],
                "stderr_tail": process.stderr[-4000:],
            })
        receipt = {
            "passed": passed,
            "commands": [" ".join(row["argv"]) for row in rows],
            "results": rows,
            "failure_summaries": [
                {
                    "command": " ".join(row["argv"]),
                    "returncode": row["returncode"],
                    "stdout_tail": row["stdout_tail"][-1800:],
                    "stderr_tail": row["stderr_tail"][-1800:],
                }
                for row in rows if row["returncode"] != 0
            ],
            "denied_candidate_targets": denied_targets,
            "selection": automatic,
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        self.verification_receipts.append(receipt)
        self.failed_verification_seen = not passed
        return {"ok": True, **receipt}

    def accept_verification_suggestions(
        self,
        targets: list[str],
        suffix: str,
        prefix: str,
        denied: list[str],
    ) -> list[str]:
        accepted = []
        for target in targets:
            try:
                accepted.append(self.verification_path(target, suffix, prefix))
            except WorkerFault:
                denied.append(target)
        return accepted

    def automatic_verification_targets(self) -> dict[str, list[str]]:
        changed = self.changed_paths()
        pytest_targets = {
            path for path in changed
            if path.startswith("tests/") and path.endswith(".py")
        }
        py_compile_targets = {
            path for path in changed if path.endswith(".py")
        }
        json_targets = {
            path for path in changed if path.endswith(".json")
        }
        test_inventory = {
            path: text for path, text in text_inventory(self.root).items()
            if path.startswith("tests/") and path.endswith(".py")
        }
        for changed_path in changed:
            if not changed_path.endswith(".py"):
                continue
            stem = PurePosixPath(changed_path).stem
            direct = f"tests/test_{stem}.py"
            if direct in test_inventory:
                pytest_targets.add(direct)
            references = [
                path for path, text in test_inventory.items()
                if stem in text or changed_path in text
            ][:4]
            pytest_targets.update(references)
        return {
            "pytest": sorted(pytest_targets),
            "py_compile": sorted(py_compile_targets),
            "json": sorted(json_targets),
        }

    def verification_path(
        self, relative: str, suffix: str, prefix: str = ""
    ) -> str:
        path = self.safe_path(relative)
        if path.suffix != suffix or (prefix and not relative.startswith(prefix)):
            raise WorkerFault("verification_target_not_allowlisted")
        return relative

    def changed_paths(self) -> list[str]:
        current = text_inventory(self.root)
        return sorted(
            path for path in set(self.baseline) | set(current)
            if self.baseline.get(path) != current.get(path)
        )

    def effect_inventory(self) -> list[dict[str, Any]]:
        current = text_inventory(self.root)
        rows = []
        for path in self.changed_paths():
            before = self.baseline.get(path)
            after = current.get(path)
            rows.append({
                "path": path,
                "effect": "create" if before is None else "delete" if after is None else "modify",
                "before_sha256": sha256_text(before) if before is not None else None,
                "after_sha256": sha256_text(after) if after is not None else None,
            })
        return rows

    def unified_diff(self) -> str:
        current = text_inventory(self.root)
        chunks = []
        for path in self.changed_paths():
            before = self.baseline.get(path)
            after = current.get(path)
            chunks.extend(difflib.unified_diff(
                [] if before is None else before.splitlines(keepends=True),
                [] if after is None else after.splitlines(keepends=True),
                fromfile="/dev/null" if before is None else f"a/{path}",
                tofile="/dev/null" if after is None else f"b/{path}",
                lineterm="\n",
            ))
        return "".join(chunks)

    def action_summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for row in self.actions:
            key = str(row["action"])
            counts[key] = counts.get(key, 0) + 1
        return {
            "action_count": len(self.actions),
            "action_counts": counts,
            "file_mutations": self.mutations,
            "failed_actions": sum(row["ok"] is False for row in self.actions),
            "pre_mutation_read_actions": self.read_actions,
            "pre_mutation_navigation_actions": self.navigation_actions,
            "distinct_read_paths": sorted(self.read_paths),
        }


class LocalMlxModel:
    def __init__(self, card: dict[str, Any]) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mlx_lm import load
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
        )
        from mlx_lm.server import LRUPromptCache
        from mlx_lm.sample_utils import (
            make_logits_processors,
            make_sampler,
        )
        import mlx.core as mx

        self.card = card
        self.snapshot = local_snapshot(card)
        self.snapshot_manifest_sha256 = snapshot_manifest(self.snapshot)
        self.model, self.tokenizer = load(str(self.snapshot), lazy=False)
        mx.eval(self.model.parameters())
        self.prompt_cache_trimmable = can_trim_prompt_cache(
            make_prompt_cache(self.model)
        )
        self.sampler = make_sampler(temp=float(card["temperature"]))
        self.logits_processors = make_logits_processors(
            repetition_penalty=float(card["repetition_penalty"]),
            repetition_context_size=int(card["repetition_context_size"]),
        )
        # This worker serves one request at a time, so only its longest dialogue
        # boundary is useful. Keeping one entry also avoids retaining an extra
        # full hybrid cache in unified memory.
        self.prompt_cache = LRUPromptCache(max_size=1)
        self.model_key = (
            str(card["repo_id"]),
            str(card["revision"]),
        )
        self.generation_cache_kwargs = {
            key: card[key]
            for key in (
                "kv_bits",
                "kv_group_size",
                "quantized_kv_start",
            )
            if card.get(key) is not None
        }
        self.last_generation_metrics: dict[str, Any] = {}

    def generate(self, messages: list[dict[str, str]]) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.core as mx

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **dict(self.card.get("chat_template_kwargs") or {}),
        )
        context_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            **dict(self.card.get("chat_template_kwargs") or {}),
        )
        bos = self.tokenizer.bos_token
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=(
                bos is None or not prompt.startswith(bos)
            ),
        )
        context_tokens = self.tokenizer.encode(
            context_prompt,
            add_special_tokens=(
                bos is None or not context_prompt.startswith(bos)
            ),
        )
        def prefill(cache: list[Any], tokens: list[int]) -> None:
            stream = generate_step(
                mx.array(tokens),
                self.model,
                max_tokens=0,
                sampler=self.sampler,
                logits_processors=self.logits_processors,
                prompt_cache=cache,
                **self.generation_cache_kwargs,
            )
            for _ in stream:
                pass

        cache_key = None
        if self.prompt_cache_trimmable:
            prompt_cache, uncached_tokens = (
                self.prompt_cache.fetch_nearest_cache(
                    self.model_key, prompt_tokens
                )
            )
            cache_reused = prompt_cache is not None
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self.model)
            else:
                # fetch_nearest_cache returns a private copy. The stored source
                # would otherwise remain resident through generation, doubling
                # long-context KV memory and causing 14B Metal OOM on this host.
                # This single-request worker can move ownership to the fetched
                # copy and reinsert its updated state after generation.
                self.prompt_cache.trim_to(n_sequences=0)
                mx.clear_cache()
            template_suffix_tokens = max(
                0, len(prompt_tokens) - len(context_tokens)
            )
            generation_uncached_tokens = min(
                len(uncached_tokens), template_suffix_tokens
            )
            uncached_context_tokens = (
                len(uncached_tokens) - generation_uncached_tokens
            )
            total_uncached_tokens = len(uncached_tokens)
            cache_key = prompt_tokens[:]
            boundary_cache_used = False
        else:
            (
                prompt_cache,
                uncached_tokens,
                cache_reused,
                uncached_context_tokens,
            ) = prepare_prompt_boundary_cache(
                self.prompt_cache,
                self.model_key,
                prompt_tokens,
                context_tokens,
                make_cache=lambda: make_prompt_cache(self.model),
                prefill=prefill,
            )
            generation_uncached_tokens = len(uncached_tokens)
            total_uncached_tokens = (
                uncached_context_tokens + generation_uncached_tokens
            )
            boundary_cache_used = True
        stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt=uncached_tokens,
            max_tokens=int(self.card["maximum_action_tokens"]),
            sampler=self.sampler,
            logits_processors=self.logits_processors,
            prompt_cache=prompt_cache,
            **self.generation_cache_kwargs,
        )
        text = ""
        complete = None
        generated_tokens = 0
        try:
            for response in stream:
                if cache_key is not None:
                    cache_key.append(int(response.token))
                generated_tokens += 1
                text += response.text
                complete = complete_action_json(text)
                if complete is not None:
                    break
        finally:
            stream.close()
            if cache_key is not None:
                self.prompt_cache.insert_cache(
                    self.model_key, cache_key, prompt_cache
                )
        self.last_generation_metrics = {
            "prompt_tokens": len(prompt_tokens),
            "uncached_prompt_tokens": total_uncached_tokens,
            "uncached_context_tokens": uncached_context_tokens,
            "generation_uncached_prompt_tokens": generation_uncached_tokens,
            "context_boundary_tokens": len(context_tokens),
            "generated_tokens": generated_tokens,
            "prefix_cache_reused": cache_reused,
            "prompt_boundary_cache_used": boundary_cache_used,
            "prompt_cache_trimmable": self.prompt_cache_trimmable,
            "kv_bits": self.generation_cache_kwargs.get("kv_bits"),
        }
        return complete if complete is not None else text


def initial_messages(
    visible: dict[str, Any], state: RepositoryState, config: dict[str, Any]
) -> list[dict[str, str]]:
    ranked = lexical_rank(
        state.baseline, keywords(str(visible["natural_request"]))
    )[:12]
    retrieval = initial_retrieval_context(
        state.baseline,
        keywords(str(visible["natural_request"])),
        ranked,
        int(config["budgets"]["maximum_initial_retrieval_characters"]),
    )
    inspection_plan = priority_inspection_paths(
        state.baseline, state.request_tokens, request=state.request
    )
    request_scope = [
        {
            "path": path,
            "exists": path in state.baseline,
        }
        for path in state.request_scoped_paths
    ]
    system = """You are a local repository editing agent inside a disposable parent snapshot.
Complete the user's request by inspecting and editing the repository, then verify it.
You have no target answer, git history, network, shell, or hidden tests.
Return exactly one JSON object per turn and no Markdown.
Allowed actions, with exact JSON shapes:
{"action":"list","prefix":"scripts/"}
{"action":"search","query":"literal or semantic terms"}
{"action":"read","path":"scripts/x.py","start_line":1,"end_line":120}
{"action":"replace","path":"scripts/x.py","old":"exact existing text","new":"replacement"}
{"action":"create","path":"tests/test_x.py","content":"complete nonempty file"}
{"action":"delete","path":"obsolete allowed path"}
{"action":"plan","criteria":["request-derived criterion"],"target_paths":["scripts/x.py"],"implementation":"general edit strategy","verification":"checks to run"}
{"action":"verify","pytest":[],"py_compile":[],"json":[]}
{"action":"finish"} or {"action":"abstain","reason":"specific missing information"}
Encode exactly one chosen action as one JSON object. For replace, old and new are
mandatory, distinct, nonempty strings; copy old exactly from a read result. Never
copy schema placeholders or invent a path. Use abstain only when the request cannot
be safely completed.
Use read/search before editing. A behavioral request requires behavioral code; changing only
messages, comments, reports, or roadmap prose is not a solution. Inspect the primary
implementation, an analogous implementation or config, and at least one relevant test before
mutation; the harness reports the exact minimum read counts in every tool result.
Integrate behavior into the existing request-specific code before adding any new implementation
module; an orphan helper is not completion.
Keep edits minimal and general. Never alter tests merely to make incorrect behavior pass.
Do not verify before creating an effect. When TOOL_RESULT.state.required_next_phase says
edit, verify, repair, or finish, follow that phase instead of continuing to browse.
The request_scoped_effect_paths are a controller-derived authority boundary, not target
answers. Never mutate another path. A listed path with exists=false is an authorized
request-derived creation path: do not try to read it before creating it. Read the existing
priority inspection paths instead.
For verification, prefer empty arrays so the worker independently selects tests, compilation,
and JSON checks from the actual changed paths. Finish only after green verification.
Candidate claims of success are ignored."""
    user = json.dumps({
        "natural_request": visible["natural_request"],
        "allowed_runtime_context": visible["allowed_runtime_context"],
        "authority_grant": visible["authority_grant"],
        "lexically_ranked_paths": ranked,
        "priority_inspection_paths": inspection_plan,
        "request_scoped_effect_paths": request_scope,
        "retrieved_parent_context": retrieval,
        "budgets": {
            key: config["budgets"][key] for key in (
                "maximum_agent_turns", "maximum_file_mutations",
                "maximum_repair_attempts",
            )
        },
    }, sort_keys=True)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def priority_inspection_paths(
    inventory: dict[str, str],
    tokens: list[str],
    *,
    request: str = "",
) -> list[str]:
    """Select a target-blind, structurally coupled test/code/config trio."""
    explicit = [
        path for path in explicit_request_paths(request)
        if path in inventory
    ]
    explicit_code = [
        path for path in explicit
        if path.startswith(("scripts/", "crates/"))
    ]
    tests = lexical_rank(
        {
            path: text for path, text in inventory.items()
            if path.startswith("tests/")
        },
        tokens,
    )
    selected_test = tests[0] if tests else None
    selected_code = explicit_code[0] if explicit_code else None
    if selected_code is not None:
        stem = PurePosixPath(selected_code).stem
        coupled = [
            path for path in tests
            if stem in PurePosixPath(path).stem
            or stem in inventory[path]
            or selected_code in inventory[path]
        ]
        if coupled:
            selected_test = coupled[0]
    if selected_test is not None:
        test_stem = PurePosixPath(selected_test).stem
        implementation_stem = (
            test_stem[5:] if test_stem.startswith("test_") else test_stem
        )
        matches = [
            path for path in inventory
            if path.startswith(("scripts/", "crates/"))
            and PurePosixPath(path).stem == implementation_stem
        ]
        if matches:
            selected_code = sorted(matches)[0]
    if selected_code is None:
        code = lexical_rank(
            {
                path: text for path, text in inventory.items()
                if path.startswith(("scripts/", "crates/"))
            },
            tokens,
        )
        selected_code = code[0] if code else None

    selected_config = None
    if selected_code is not None:
        referenced = sorted(set(re.findall(
            r"""configs/[A-Za-z0-9_./-]+\.json""",
            inventory[selected_code],
        )))
        existing = {
            path: inventory[path] for path in referenced if path in inventory
        }
        ranked_references = lexical_rank(existing, tokens)
        if ranked_references:
            selected_config = ranked_references[0]
        elif existing:
            selected_config = sorted(existing)[0]
    if selected_config is None:
        configs = lexical_rank(
            {
                path: text for path, text in inventory.items()
                if path.startswith("configs/")
                and all(
                    generic not in PurePosixPath(path).name
                    for generic in ("manifest", "registry", "roadmap")
                )
            },
            tokens,
        )
        selected_config = configs[0] if configs else None

    return [
        path for path in (selected_code, selected_config, selected_test)
        if path is not None
    ]


def explicit_request_paths(request: str) -> list[str]:
    """Extract only user-visible repository paths named in the request."""
    pattern = (
        r"(?<![A-Za-z0-9_.-])"
        r"(?:configs|crates|docs|examples|scripts|tests)/"
        r"[A-Za-z0-9_./-]+"
        r"\.(?:css|html|js|json|md|py|qmd|rs|toml|ts|tsx|ya?ml)"
    )
    return list(dict.fromkeys(re.findall(pattern, request)))


def request_effect_paths(
    inventory: dict[str, str], request: str
) -> list[str]:
    """Derive a narrow, target-blind effect boundary from request text."""
    explicit = explicit_request_paths(request)
    paths = list(explicit)
    for path in explicit:
        pure = PurePosixPath(path)
        if pure.parts[0] == "scripts" and pure.suffix == ".py":
            paths.append(f"tests/test_{pure.stem}.py")
    return [
        path for path in dict.fromkeys(paths)
        if path in inventory
        or (
            path.startswith("tests/test_")
            and path.endswith(".py")
        )
    ]


def validate_inputs(
    visible: dict[str, Any], snapshot_root: Path, config: dict[str, Any]
) -> None:
    if set(visible) != VISIBLE_FIELDS:
        raise WorkerFault(f"visible_fields_must_equal:{sorted(VISIBLE_FIELDS)}")
    if FORBIDDEN_TERMS.intersection(visible):
        raise WorkerFault("forbidden_visible_field")
    if (snapshot_root / ".git").exists():
        raise WorkerFault("git_metadata_forbidden")
    if config.get("policy") not in {
        "project_theseus_local_repository_worker_v2_development_v1",
        "project_theseus_local_8b_stack_worker_v1",
    }:
        raise WorkerFault("unexpected_config_policy")
    boundaries = config.get("boundaries") or {}
    required = {
        "network": "forbidden",
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "user_facing_effects": 0,
        "learned_generation_credit": 1,
        "candidate_snapshot_has_git_metadata": False,
        "arbitrary_shell": False,
    }
    if any(boundaries.get(key) != value for key, value in required.items()):
        raise WorkerFault("boundary_contract_mismatch")


def parse_action(raw: str) -> dict[str, Any]:
    complete = complete_action_json(raw)
    if complete is None:
        repaired = repair_common_replace_concatenation(raw)
        if repaired is not None:
            return {**repaired, "_format_repaired": True}
        raise json.JSONDecodeError("no complete JSON action", raw, 0)
    result = json.loads(complete)
    if not isinstance(result, dict):
        raise WorkerFault("action_must_be_object")
    return result


def repair_common_replace_concatenation(
    raw: str,
) -> dict[str, Any] | None:
    """Repair one observed Qwen tool-format error without semantic inference.

    Qwen sometimes emits ``"old":"..."+"..."`` where the second literal was
    plainly intended as ``new``. Only the exact replace-action shape is
    accepted, and each use is surfaced in candidate accounting.
    """
    string_literal = r'"(?:\\.|[^"\\])*"'
    match = re.fullmatch(
        rf"""\s*\{{\s*
        "action"\s*:\s*"replace"\s*,\s*
        "path"\s*:\s*(?P<path>{string_literal})\s*,\s*
        "old"\s*:\s*(?P<old>{string_literal})\s*
        \+\s*(?P<new>{string_literal})\s*
        \}}\s*""",
        raw,
        flags=re.VERBOSE,
    )
    if match is None:
        return None
    try:
        path = json.loads(match.group("path"))
        old = json.loads(match.group("old"))
        new = json.loads(match.group("new"))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(value, str) for value in (path, old, new)):
        return None
    if not path or not old or not new or old == new:
        return None
    return {
        "action": "replace",
        "path": path,
        "old": old,
        "new": new,
    }


def complete_action_json(raw: str) -> str | None:
    """Return the first complete JSON object without waiting for model EOS."""
    start = raw.find("{")
    if start < 0:
        return None
    try:
        result, consumed = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        raise WorkerFault("action_must_be_object")
    return raw[start:start + consumed]


def prepare_prompt_boundary_cache(
    cache_store: Any,
    model_key: Any,
    prompt_tokens: list[int],
    boundary_tokens: list[int],
    *,
    make_cache: Any,
    prefill: Any,
) -> tuple[Any, list[int], bool, int]:
    """Create a reusable pre-generation cache even for hybrid recurrent models.

    A normal streamed cache includes generated assistant tokens. Hybrid caches
    cannot always be trimmed when the next chat turn diverges at that boundary,
    which forces a full prompt prefill every turn. Prefilling through the
    completed-message boundary with ``max_tokens=0`` records a prefix that is
    stable when the next assistant/tool turn is rendered, without needing to
    rewind recurrent state.
    """
    if (
        not boundary_tokens
        or prompt_tokens[:len(boundary_tokens)] != boundary_tokens
    ):
        raise WorkerFault("prompt_context_boundary_not_prefix")
    boundary_cache, boundary_uncached = cache_store.fetch_nearest_cache(
        model_key, boundary_tokens
    )
    reused = boundary_cache is not None
    if boundary_cache is None:
        boundary_cache = make_cache()
    if boundary_uncached:
        prefill(boundary_cache, boundary_uncached)
    cache_store.insert_cache(model_key, boundary_tokens, boundary_cache)
    generation_cache, generation_uncached = cache_store.fetch_nearest_cache(
        model_key, prompt_tokens
    )
    expected_generation_suffix = prompt_tokens[len(boundary_tokens):]
    if (
        generation_cache is None
        or generation_uncached != expected_generation_suffix
    ):
        raise WorkerFault("prompt_boundary_cache_retrieval_failed")
    return (
        generation_cache,
        generation_uncached,
        reused,
        len(boundary_uncached),
    )


def write_normalized_text(path: Path, text: str) -> None:
    path.write_text(
        text if text.endswith("\n") else text + "\n",
        encoding="utf-8",
    )


def safe_action_detail(action: dict[str, Any]) -> dict[str, Any]:
    kind = str(action.get("action") or "")
    if kind == "search":
        return {"query": str(action.get("query") or "")[:200]}
    if kind == "list":
        return {"prefix": str(action.get("prefix") or "")[:200]}
    if kind in {"read", "replace", "create", "delete"}:
        detail = {"path": str(action.get("path") or "")[:300]}
        if kind == "read":
            detail.update({
                "start_line": int(action.get("start_line") or 1),
                "end_line": int(action.get("end_line") or 240),
            })
        elif kind == "replace":
            detail.update({
                "old_characters": len(str(action.get("old") or "")),
                "new_characters": len(str(action.get("new") or "")),
            })
        elif kind == "create":
            detail["content_characters"] = len(
                str(action.get("content") or "")
            )
        return detail
    if kind == "verify":
        return {
            key: strings(action.get(key))[:16]
            for key in ("pytest", "py_compile", "json")
        }
    if kind == "plan":
        return {
            "criterion_count": len(action.get("criteria") or []),
            "target_path_count": len(action.get("target_paths") or []),
        }
    return {}


def initial_retrieval_context(
    inventory: dict[str, str],
    tokens: list[str],
    ranked_paths: list[str],
    maximum_characters: int,
) -> list[dict[str, Any]]:
    """Build compact target-blind excerpts from the parent snapshot."""
    rows = []
    used = 0
    for path in ranked_paths[:8]:
        lines = inventory[path].splitlines()
        matching = [
            index for index, line in enumerate(lines)
            if any(token in line.casefold() for token in tokens)
        ]
        selected: set[int] = set()
        for index in matching[:10]:
            selected.update(range(max(0, index - 2), min(len(lines), index + 3)))
        if path.endswith(".py"):
            selected.update(
                index for index, line in enumerate(lines)
                if re.match(r"^(?:async\s+)?(?:def|class)\s+", line)
            )
        selected_lines = [
            f"{index + 1}: {lines[index]}" for index in sorted(selected)[:60]
        ]
        excerpt = "\n".join(selected_lines)
        if not excerpt:
            excerpt = "\n".join(
                f"{index + 1}: {line}"
                for index, line in enumerate(lines[:20])
            )
        remaining = maximum_characters - used
        if remaining <= 0:
            break
        excerpt = excerpt[:remaining]
        rows.append({"path": path, "excerpt": excerpt})
        used += len(excerpt)
    return rows


def text_inventory(root: Path) -> dict[str, str]:
    rows = {}
    for path in root.rglob("*"):
        if (
            not path.is_file() or path.is_symlink()
            or path.suffix.lower() not in ALLOWED_SUFFIXES
            or path.stat().st_size > 1_000_000
        ):
            continue
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] not in ALLOWED_TOP_LEVEL:
            continue
        rows[relative.as_posix()] = path.read_text(encoding="utf-8", errors="strict")
    return dict(sorted(rows.items()))


def lexical_rank(inventory: dict[str, str], tokens: list[str]) -> list[str]:
    scores = []
    for path, text in inventory.items():
        path_text = path.lower().replace("_", " ").replace("-", " ")
        lowered = text.lower()
        # Binary term presence prevents giant registries from winning merely
        # because they repeat generic words hundreds of times. Paths carry
        # more intent than raw corpus frequency for repository work.
        path_hits = sum(token in path_text for token in tokens)
        content_hits = sum(token in lowered for token in tokens)
        score = path_hits * 12 + content_hits
        if score:
            scores.append((score, path_hits, path))
    return [
        path for _, _, path in sorted(
            scores, key=lambda row: (-row[0], -row[1], row[2])
        )
    ]


def keywords(value: str) -> list[str]:
    stop = {"and", "for", "from", "into", "the", "this", "with"}
    return sorted({
        token for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) >= 3 and token not in stop
    })


def local_snapshot(card: dict[str, Any]) -> Path:
    repo = str(card["repo_id"]).replace("/", "--")
    path = (
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo}"
        / "snapshots" / str(card["revision"])
    )
    if not path.is_dir() or not complete_model_snapshot(path):
        raise WorkerFault("complete_local_model_snapshot_missing")
    return path


def complete_model_snapshot(path: Path) -> bool:
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not all((path / item).is_file() for item in required):
        return False
    if (path / "model.safetensors").is_file():
        return True
    index = path / "model.safetensors.index.json"
    return bool(
        index.is_file()
        and list(path.glob("model-*-of-*.safetensors"))
    )


def snapshot_manifest(path: Path) -> str:
    rows = []
    for item in sorted(path.iterdir()):
        if item.is_file():
            resolved = item.resolve()
            rows.append({
                "path": item.name,
                "bytes": resolved.stat().st_size,
                "blob_identity": resolved.name,
            })
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def relative_command_part(value: str, root: Path) -> str:
    try:
        path = Path(value)
        return path.relative_to(root).as_posix() if path.is_absolute() else value
    except ValueError:
        return value


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def emit_event(
    sink: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if sink is not None:
        sink(event)


def strings(value: Any) -> list[str]:
    return (
        [item for item in value if isinstance(item, str)]
        if isinstance(value, list)
        else []
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
