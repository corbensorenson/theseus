#!/usr/bin/env python3
"""Deterministically materialize the frozen v8 functional-utility cases."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Callable


ARMS = ("english", "python", "javascript_typescript", "html_css", "rust")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def materialize_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["seed"])
    variants = int(config["variants_per_family"])
    cases: list[dict[str, Any]] = []
    for arm_id in ARMS:
        arm = config["arms"][arm_id]
        for family in arm["families"]:
            for variant in range(variants):
                case = build_case(arm_id, str(family), variant, seed)
                visible = {
                    "case_id": case["case_id"],
                    "arm_id": arm_id,
                    "prompt": case["prompt"],
                }
                case["model_visible"] = visible
                case["prompt_sha256"] = hashlib.sha256(case["prompt"].encode()).hexdigest()
                case["verifier_sha256"] = stable_hash(case["verifier"])
                cases.append(case)
    return cases


def build_case(arm_id: str, family: str, variant: int, seed: int) -> dict[str, Any]:
    identity = f"{arm_id}:{family}:{variant}:{seed}"
    case_id = f"fu-{arm_id}-{family}-{variant + 1:02d}-{hashlib.sha256(identity.encode()).hexdigest()[:10]}"
    rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16))
    if arm_id == "english":
        payload = english_case(family, variant)
    elif arm_id in {"python", "javascript_typescript"}:
        payload = dynamic_language_case(arm_id, family, variant, rng)
    elif arm_id == "rust":
        payload = rust_case(family, variant, rng)
    elif arm_id == "html_css":
        payload = html_case(family, variant)
    else:
        raise ValueError(f"unsupported arm: {arm_id}")
    return {
        "case_id": case_id,
        "arm_id": arm_id,
        "task_family": family,
        "variant": variant,
        "prompt": payload["prompt"],
        "verifier": payload["verifier"],
        "public_benchmark": False,
        "training_eligible": False,
        "candidate_family_required": "direct_autoregressive_model_text",
    }


def english_case(family: str, variant: int) -> dict[str, Any]:
    names = ("Mara", "Jonah", "Priya", "Luis")
    projects = ("Atlas", "Lantern", "Harbor", "Juniper")
    colors = ("teal", "amber", "indigo", "crimson")
    name = names[variant]
    project = projects[variant]
    color = colors[variant]
    builders: dict[str, Callable[[], tuple[str, list[str], list[str]]]] = {
        "correction_following": lambda: (
            f"Conversation:\nUser: For {project}, use {color} and deliver a PDF on Friday.\n"
            f"User: Correction: use grayscale, deliver a Markdown file on Thursday, and keep the project name.\n"
            "Assistant: Understood.\nUser: State the final deliverable in one sentence.",
            [project, "grayscale", "Markdown", "Thursday"],
            [color, "PDF", "Friday"],
        ),
        "constraint_tracking": lambda: (
            f"Conversation:\nUser: Plan a review for {project}. It must last 25 minutes, include {name} and two reviewers, "
            "avoid Monday, and end with one written decision.\nUser: Give a concise plan that preserves every constraint.",
            [project, "25", name, "two reviewers", "written decision", "not Monday"],
            ["Monday meeting", "30 minutes"],
        ),
        "grounded_synthesis": lambda: (
            f"Context:\n- {project} processed {40 + variant * 7} jobs.\n- {3 + variant} jobs failed validation.\n"
            f"- The retry queue is owned by {name}.\nQuestion: Summarize the operational state using only supplied facts and name the owner.",
            [project, str(40 + variant * 7), str(3 + variant), name],
            ["root cause", "all retries succeeded"],
        ),
        "ambiguity_clarification": lambda: (
            f"Conversation:\nUser: Send the {project} report when it is ready.\nAssistant: I can help prepare that.\n"
            "User: Go ahead.\nRespond with the clarification needed before any external effect.",
            ["recipient", "destination", "approval"],
            ["sent", "emailed", "uploaded"],
        ),
        "calibrated_abstention": lambda: (
            f"Context:\nThe {project} build started at 09:00. The log ends after dependency resolution.\n"
            "Question: What exact time did the build finish? State what is known and what evidence is missing.",
            ["cannot determine", "09:00", "missing", "completion"],
            ["finished at", "09:30", "10:00"],
        ),
        "conversation_continuity": lambda: (
            f"Conversation:\nUser: My name is {name}. For status notes, use bullets and never include emojis.\n"
            f"User: The current project is {project}; its status is blocked on review.\n"
            "User: Write my status note while following my saved preferences.",
            [name, project, "blocked", "review", "-"],
            ["emoji", "😀"],
        ),
        "structured_transformation": lambda: (
            f"Source: {project} added caching. Startup improved by {12 + variant * 3} percent. Memory rose by {2 + variant} percent.\n"
            "Transform this into exactly three labeled bullets: Change, Benefit, Cost. Do not add facts.",
            ["Change", "Benefit", "Cost", "caching", str(12 + variant * 3), str(2 + variant)],
            ["latency", "customers"],
        ),
        "repository_planning": lambda: (
            f"Repository task: rename config key `{color}_mode` to `display_mode` in parser.py and settings.ts. "
            "Preserve backward-compatible reads for one release, update tests, and provide rollback. "
            "Return an ordered implementation and verification plan.",
            ["parser.py", "settings.ts", f"{color}_mode", "display_mode", "test", "rollback", "backward"],
            ["delete compatibility immediately"],
        ),
    }
    if family not in builders:
        raise ValueError(f"unknown English family: {family}")
    prompt, required, forbidden = builders[family]()
    return {
        "prompt": prompt,
        "verifier": {
            "kind": "blind_rubric_v1",
            "required_concepts": required,
            "forbidden_claims": forbidden,
            "dimensions": [
                "instruction_fulfillment",
                "correctness_and_grounding",
                "conversation_state",
                "calibration",
                "clarity",
            ],
            "minimum_raters": 2,
        },
    }


def dynamic_language_case(
    arm_id: str, family: str, variant: int, rng: random.Random
) -> dict[str, Any]:
    is_python = arm_id == "python"
    language = "Python" if is_python else ("TypeScript" if variant % 2 else "JavaScript")
    extension = "py" if is_python else ("ts" if variant % 2 else "js")
    function = f"{family}_{variant + 1}"
    calls = generated_calls(family, variant, rng)
    signature = dynamic_signature(language, family, function)
    prompt = (
        f"Write a complete standalone {language} module implementing `{signature}`. "
        f"{family_contract(family)} Return only the complete module source."
    )
    if family == "repository_edit":
        old_limit = 2 + variant
        new_limit = 6 + variant
        if is_python:
            current = f"RETRY_LIMIT = {old_limit}\n\ndef should_retry(attempt):\n    return attempt < RETRY_LIMIT\n"
        else:
            export = "export " if extension == "ts" else "export "
            current = f"{export}const RETRY_LIMIT = {old_limit};\n{export}function shouldRetry(attempt) {{\n  return attempt < RETRY_LIMIT;\n}}\n"
        prompt = (
            f"Apply this repository edit to the complete {language} module: change RETRY_LIMIT from "
            f"{old_limit} to {new_limit}, preserve the retry boundary behavior, and add a named "
            f"DEFAULT_TIMEOUT_MS constant set to {1000 + variant * 250}.\n\nCurrent module:\n{current}\n"
            "Return only the complete revised module."
        )
        calls = {
            "old_limit": old_limit,
            "new_limit": new_limit,
            "timeout_ms": 1000 + variant * 250,
        }
    return {
        "prompt": prompt,
        "verifier": {
            "kind": "python_isolated_tests_v1" if is_python else "deno_typescript_tests_v1",
            "language": language.lower(),
            "candidate_filename": f"candidate.{extension}",
            "function_name": function,
            "family": family,
            "cases": calls,
        },
    }


def generated_calls(family: str, variant: int, rng: random.Random) -> list[dict[str, Any]]:
    if family == "stable_unique":
        rows = []
        for _ in range(4):
            values = [rng.randint(0, 5 + variant) for _ in range(10 + variant)]
            rows.append({"args": [values], "expected": list(dict.fromkeys(values))})
        return rows
    if family == "clamp_values":
        rows = []
        for _ in range(4):
            low, high = sorted(rng.sample(range(-10, 16), 2))
            values = [rng.randint(-20, 25) for _ in range(8)]
            rows.append({"args": [values, low, high], "expected": [min(high, max(low, x)) for x in values]})
        return rows
    if family == "merge_counts":
        rows = []
        for _ in range(4):
            left = {f"k{i}": rng.randint(0, 7) for i in range(3 + variant % 2)}
            right = {f"k{i}": rng.randint(0, 7) for i in range(1, 5)}
            expected = dict(left)
            for key, value in right.items():
                expected[key] = expected.get(key, 0) + value
            rows.append({"args": [left, right], "expected": expected})
        return rows
    if family == "chunk_values":
        rows = []
        for size in (1, 2, 3, 4):
            values = list(range(variant, variant + 7 + size))
            rows.append({"args": [values, size], "expected": [values[i : i + size] for i in range(0, len(values), size)]})
        return rows
    if family == "parse_duration":
        return [
            {"args": [f"{variant + 1}h {minutes}m {seconds}s"], "expected": (variant + 1) * 3600 + minutes * 60 + seconds}
            for minutes, seconds in ((0, 5), (2, 0), (7, 11), (59, 59))
        ]
    if family == "normalize_slug":
        values = [
            f"  Project {variant} Status  ",
            f"Alpha__Beta {variant}",
            f"Already-clean-{variant}",
            f"Symbols! and @ spaces {variant}",
        ]
        return [{"args": [value], "expected": reference_slug(value)} for value in values]
    if family == "select_active":
        rows = []
        for _ in range(4):
            records = [
                {"name": f"item-{i}", "enabled": bool(rng.randint(0, 1)), "priority": rng.randint(0, 9)}
                for i in range(7)
            ]
            expected = [
                row["name"]
                for row in sorted((row for row in records if row["enabled"]), key=lambda row: (-row["priority"], row["name"]))
            ]
            rows.append({"args": [records], "expected": expected})
        return rows
    if family == "repository_edit":
        return []
    raise ValueError(f"unknown dynamic family: {family}")


def dynamic_signature(language: str, family: str, function: str) -> str:
    python = {
        "stable_unique": f"def {function}(values: list[int]) -> list[int]",
        "clamp_values": f"def {function}(values: list[int], low: int, high: int) -> list[int]",
        "merge_counts": f"def {function}(left: dict[str, int], right: dict[str, int]) -> dict[str, int]",
        "chunk_values": f"def {function}(values: list[int], size: int) -> list[list[int]]",
        "parse_duration": f"def {function}(text: str) -> int",
        "normalize_slug": f"def {function}(text: str) -> str",
        "select_active": f"def {function}(records: list[dict]) -> list[str]",
    }
    typescript = {
        "stable_unique": f"export function {function}(values: number[]): number[]",
        "clamp_values": f"export function {function}(values: number[], low: number, high: number): number[]",
        "merge_counts": f"export function {function}(left: Record<string, number>, right: Record<string, number>): Record<string, number>",
        "chunk_values": f"export function {function}(values: number[], size: number): number[][]",
        "parse_duration": f"export function {function}(text: string): number",
        "normalize_slug": f"export function {function}(text: string): string",
        "select_active": f"export function {function}(records: {{name: string; enabled: boolean; priority: number}}[]): string[]",
    }
    javascript = {
        "stable_unique": f"export function {function}(values)",
        "clamp_values": f"export function {function}(values, low, high)",
        "merge_counts": f"export function {function}(left, right)",
        "chunk_values": f"export function {function}(values, size)",
        "parse_duration": f"export function {function}(text)",
        "normalize_slug": f"export function {function}(text)",
        "select_active": f"export function {function}(records)",
    }
    table = python if language == "Python" else (typescript if language == "TypeScript" else javascript)
    return table.get(family, "complete module edit")


def family_contract(family: str) -> str:
    return {
        "stable_unique": "Return values in first-seen order with duplicates removed and do not mutate the input.",
        "clamp_values": "Clamp every value inclusively to [low, high], preserving order; reject low greater than high.",
        "merge_counts": "Return a new key-sorted mapping whose values sum matching keys without mutating inputs.",
        "chunk_values": "Split values into consecutive chunks, retaining a short final chunk; reject non-positive size.",
        "parse_duration": "Parse optional h, m, and s integer components separated by spaces into total seconds; reject malformed text.",
        "normalize_slug": "Lowercase, collapse every run of non-alphanumeric characters to one hyphen, and trim boundary hyphens.",
        "select_active": "Select enabled records, sort by descending priority then ascending name, and return names without mutating input.",
        "repository_edit": "Apply the requested complete-module edit.",
    }[family]


def reference_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def rust_case(family: str, variant: int, rng: random.Random) -> dict[str, Any]:
    function = f"{family}_{variant + 1}"
    signatures = {
        "stable_unique": f"pub fn {function}(values: &[i32]) -> Vec<i32>",
        "clamp_values": f"pub fn {function}(values: &[i32], low: i32, high: i32) -> Result<Vec<i32>, String>",
        "merge_counts": f"pub fn {function}(left: &[(String, i32)], right: &[(String, i32)]) -> Vec<(String, i32)>",
        "chunk_values": f"pub fn {function}(values: &[i32], size: usize) -> Result<Vec<Vec<i32>>, String>",
        "parse_duration": f"pub fn {function}(text: &str) -> Result<u64, String>",
        "normalize_slug": f"pub fn {function}(text: &str) -> String",
        "select_active": f"pub fn {function}(records: &[(String, bool, i32)]) -> Vec<String>",
    }
    if family == "repository_edit":
        old_limit, new_limit = 2 + variant, 6 + variant
        prompt = (
            f"Apply this repository edit to the complete Rust library: change RETRY_LIMIT from {old_limit} to {new_limit}, "
            f"preserve should_retry's boundary behavior, and add `pub const DEFAULT_TIMEOUT_MS: u64 = {1000 + variant * 250};`.\n\n"
            f"Current library:\npub const RETRY_LIMIT: u32 = {old_limit};\npub fn should_retry(attempt: u32) -> bool {{ attempt < RETRY_LIMIT }}\n\n"
            "Return only the complete revised src/lib.rs."
        )
        spec: Any = {"old_limit": old_limit, "new_limit": new_limit, "timeout_ms": 1000 + variant * 250}
    else:
        prompt = (
            f"Write a complete standalone Rust library implementing `{signatures[family]}`. "
            f"{family_contract(family)} Return only the complete src/lib.rs source."
        )
        spec = generated_calls(family, variant, rng)
    return {
        "prompt": prompt,
        "verifier": {
            "kind": "rust_cargo_tests_v1",
            "candidate_filename": "src/lib.rs",
            "function_name": function,
            "family": family,
            "cases": spec,
        },
    }


def html_case(family: str, variant: int) -> dict[str, Any]:
    suffix = variant + 1
    specs: dict[str, dict[str, Any]] = {
        "accessible_form": {
            "request": f"Create a signup form titled Account {suffix} with labeled email and password fields, validation help, and a submit button.",
            "required_tags": ["main", "form", "label", "input", "button"],
            "required_attrs": [["input", "type", "email"], ["input", "type", "password"], ["main", "aria-labelledby", "signup-title"]],
            "required_text": [f"Account {suffix}", "Email", "Password"],
            "required_css": [":focus-visible", "@media"],
        },
        "landmark_navigation": {
            "request": f"Create a documentation page for Release {suffix} with skip link, header, labeled navigation, main article, and footer.",
            "required_tags": ["header", "nav", "main", "article", "footer", "a"],
            "required_attrs": [["nav", "aria-label", "Documentation"], ["main", "id", "content"]],
            "required_text": [f"Release {suffix}", "Skip to content"],
            "required_css": [":focus-visible", "max-width"],
        },
        "data_table": {
            "request": f"Create a responsive table captioned Build {suffix} results with Name, Status, and Duration columns and two data rows.",
            "required_tags": ["table", "caption", "thead", "tbody", "th", "td"],
            "required_attrs": [["th", "scope", "col"]],
            "required_text": [f"Build {suffix} results", "Status", "Duration"],
            "required_css": ["overflow", "@media"],
        },
        "status_alert": {
            "request": f"Create a status panel announcing Sync {suffix} failed, with an accessible alert, retry button, and visually distinct error state.",
            "required_tags": ["section", "button"],
            "required_attrs": [["section", "role", "alert"], ["button", "type", "button"]],
            "required_text": [f"Sync {suffix} failed", "Retry"],
            "required_css": ["border", ":focus-visible"],
        },
        "responsive_cards": {
            "request": f"Create a Projects {suffix} card grid with three articles that is one column on narrow screens and three columns at 48rem.",
            "required_tags": ["main", "section", "article", "h1", "h2"],
            "required_attrs": [["section", "aria-label", "Projects"]],
            "required_text": [f"Projects {suffix}"],
            "required_css": ["grid-template-columns", "48rem", "@media"],
        },
        "theme_variables": {
            "request": f"Create a Theme {suffix} preview using CSS custom properties for surface, text, accent, and focus colors plus dark-scheme overrides.",
            "required_tags": ["main", "button"],
            "required_attrs": [["main", "aria-labelledby", "theme-title"]],
            "required_text": [f"Theme {suffix}"],
            "required_css": ["--surface", "--text", "--accent", "prefers-color-scheme", ":focus-visible"],
        },
        "modal_dialog": {
            "request": f"Create a visible confirmation dialog for Archive {suffix} with title, description, cancel, and confirm controls.",
            "required_tags": ["dialog", "button", "h2"],
            "required_attrs": [["dialog", "open", ""], ["dialog", "aria-labelledby", "dialog-title"], ["dialog", "aria-describedby", "dialog-description"]],
            "required_text": [f"Archive {suffix}", "Cancel", "Confirm"],
            "required_css": ["::backdrop", ":focus-visible"],
        },
        "media_figure": {
            "request": f"Create a performance figure titled Throughput {suffix} with an image, useful alt text, caption, and responsive sizing.",
            "required_tags": ["main", "figure", "img", "figcaption"],
            "required_attrs": [["img", "alt", f"Throughput chart {suffix}"]],
            "required_text": [f"Throughput {suffix}"],
            "required_css": ["max-width", "height: auto"],
        },
    }
    spec = specs[family]
    return {
        "prompt": spec["request"] + " Return one complete HTML document with embedded CSS and no JavaScript or external resources.",
        "verifier": {
            "kind": "html_dom_a11y_render_v1",
            "candidate_filename": "index.html",
            **{key: value for key, value in spec.items() if key != "request"},
            "forbid_external_resources": True,
            "forbid_javascript": True,
            "viewport": {"width": 800, "height": 600},
        },
    }


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "arm_id": case["arm_id"],
        "task_family": case["task_family"],
        "variant": case["variant"],
        "prompt_sha256": case["prompt_sha256"],
        "verifier_kind": case["verifier"]["kind"],
        "verifier_sha256": case["verifier_sha256"],
        "public_benchmark": False,
        "training_eligible": False,
    }
