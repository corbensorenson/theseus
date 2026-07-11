"""Audit that external provider inference is teacher-only.

The project may fetch benchmark/data metadata under license gates, but it must
not use outside intelligence for training, scoring, synthesis, routing, or
normal autonomy. Local model libraries are allowed for local training/runtime;
approved OpenAI inference is allowed only through the sparse teacher wrapper
in ``scripts/teacher_oracle.py``. Anthropic and Claude are forbidden.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from teacher_provider_policy import teacher_receipt_decision
from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TEACHER_POLICY_PATH = ROOT / "configs" / "teacher_distillation_policy.json"
TEACHER_CALLS_PATH = REPORTS / "teacher_calls.jsonl"
REPORT_JSON_PARSE_MAX_BYTES = 8 * 1024 * 1024
REPORT_EXTERNAL_KEYS = (
    b'"external_inference_calls"',
    b'"external_inference_violations"',
)

TEACHER_FILES = {
    Path("scripts/teacher_oracle.py"),
    Path("configs/teacher_policy.json"),
    Path("configs/teacher_response_schema.json"),
}

TEACHER_DELEGATE_FILES = {
    Path("scripts/autonomy_cycle.py"),
    Path("scripts/autonomous_goal_runner.py"),
    Path("scripts/checkpoint_chat.py"),
    Path("scripts/sparkstream_dashboard.py"),
    Path("scripts/sparkstream_daemon.py"),
}

TEACHER_POLICY_OBSERVER_FILES = {
    Path("scripts/autonomy_launch_readiness.py"),
    Path("scripts/teacher_provider_policy.py"),
}
TEACHER_REPORT_PREFIXES = (
    "teacher_budget",
    "teacher_distillation",
    "teacher_oracle",
    "teacher_self_edit",
)
TEACHER_GUIDANCE_REPORT_PREFIXES = (
    "architecture_guidance_loop",
)
TEACHER_AUDIT_REPORT_POLICIES = {
    "project_theseus_teacher_budget_audit_v1",
    "project_theseus_teacher_distillation_gate_v0",
    "project_theseus_teacher_distillation_manifest_builder_v0",
    "project_theseus_teacher_distillation_manifest_v0",
}

SELF_FILES = {Path("scripts/external_inference_audit.py")}
LOCAL_COMPAT_FILES = {
    Path("scripts/openai_compat_server.py"),
    Path("configs/openai_compat_policy.json"),
}
LOCAL_SECRET_SCANNER_FILES = {
    Path("scripts/personality_core.py"),
}

SCAN_ROOTS = [
    Path("scripts"),
    Path("configs"),
    Path("crates"),
    Path("adapters"),
]

TEXT_SUFFIXES = {
    ".py",
    ".rs",
    ".ps1",
    ".json",
    ".toml",
    ".lock",
}

ACTIVE_INFERENCE_PATTERNS = [
    ("codex_cli_policy", re.compile(r"\bcodex_command\b")),
    ("openai_sdk_import", re.compile(r"(?m)^\s*(?:from\s+openai\b|import\s+openai\b)")),
    ("openai_api_endpoint", re.compile(r"https?://api\.openai\.com|/v1/(?:chat/completions|responses)\b", re.I)),
    ("openai_api_key", re.compile(r"\bOPENAI_API_KEY\b")),
    ("anthropic_sdk_import", re.compile(r"(?m)^\s*(?:from\s+anthropic\b|import\s+anthropic\b)")),
    ("anthropic_api_endpoint", re.compile(r"https?://api\.anthropic\.com", re.I)),
    ("anthropic_api_key", re.compile(r"\bANTHROPIC_API_KEY\b")),
    ("gemini_sdk_import", re.compile(r"google\.generativeai|google\.genai", re.I)),
    ("gemini_api_endpoint", re.compile(r"generativelanguage\.googleapis\.com", re.I)),
    ("gemini_api_key", re.compile(r"\b(?:GEMINI_API_KEY|GOOGLE_API_KEY)\b")),
    ("mistral_sdk_import", re.compile(r"(?m)^\s*(?:from\s+mistral(?:ai)?\b|import\s+mistral(?:ai)?\b)")),
    ("mistral_api_key", re.compile(r"\bMISTRAL_API_KEY\b")),
    ("cohere_sdk_import", re.compile(r"(?m)^\s*(?:from\s+cohere\b|import\s+cohere\b)")),
    ("cohere_api_key", re.compile(r"\bCOHERE_API_KEY\b")),
    ("together_api_key", re.compile(r"\bTOGETHER_API_KEY\b")),
    ("replicate_api_key", re.compile(r"\bREPLICATE_API_TOKEN\b")),
    ("groq_api_key", re.compile(r"\bGROQ_API_KEY\b")),
    ("hf_inference_endpoint", re.compile(r"api-inference\.huggingface\.co", re.I)),
    ("hf_inference_client", re.compile(r"\bInferenceClient\b")),
    ("transformers_local_model", re.compile(r"\b(?:transformers|AutoModel|AutoTokenizer|pipeline\()\b")),
    ("sentence_transformers", re.compile(r"\bsentence[_-]transformers\b")),
    ("ollama_local_model", re.compile(r"\bollama\b", re.I)),
    ("vllm_local_model", re.compile(r"\bvllm\b", re.I)),
    ("llama_cpp_local_model", re.compile(r"\b(?:llama_cpp|llama-cpp|llama\.cpp)\b", re.I)),
    ("speech_inference_import", re.compile(r"(?m)^\s*(?:from\s+(?:whisper|faster_whisper|vosk|pyttsx3|TTS|speechbrain\.pretrained)\b|import\s+(?:whisper|faster_whisper|vosk|pyttsx3|TTS)\b)")),
    ("generic_bearer_auth", re.compile(r"\bAuthorization\b.*\bBearer\b", re.I)),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/external_inference_audit.json")
    parser.add_argument("--scan-reports", action="store_true", default=False)
    parser.add_argument("--no-scan-reports", action="store_false", dest="scan_reports")
    parser.add_argument("--include-large-reports", action="store_true")
    args = parser.parse_args()

    files = scannable_files()
    code_hits, violations, delegate_hits = scan_code(files)
    report_violations, report_scan = (
        scan_report_external_inference(include_large_reports=args.include_large_reports)
        if args.scan_reports
        else ([], {"scanned_report_files": 0, "skipped_large_report_files": 0})
    )
    teacher_receipt_violations, teacher_receipt_scan = scan_teacher_receipts()
    all_violations = violations + report_violations + teacher_receipt_violations
    allowed_teacher_hits = [hit for hit in code_hits if hit["classification"] == "allowed_teacher"]
    benign_hits = [hit for hit in code_hits if hit["classification"].startswith("benign")]

    payload = {
        "policy": "sparkstream_external_inference_teacher_only_audit_v0",
        "created_utc": now(),
        "ok": not all_violations,
        "teacher_only_invariant": not all_violations,
        "rule": (
            "Approved OpenAI inference is allowed only through "
            "scripts/teacher_oracle.py in sparse teacher mode; Anthropic and "
            "Claude are forbidden. Local model "
            "libraries are allowed for local training/runtime. Network data, "
            "benchmark, and RL source discovery is not inference and remains "
            "governed by license/fetch policy."
        ),
        "summary": {
            "scanned_files": len(files),
            "active_inference_hits": len(code_hits),
            "allowed_teacher_hits": len(allowed_teacher_hits),
            "teacher_delegate_hits": len(delegate_hits),
            "benign_metadata_or_policy_hits": len(benign_hits),
            "code_violations": len(violations),
            "report_violations": len(report_violations),
            "teacher_receipt_violations": len(teacher_receipt_violations),
            "total_violations": len(all_violations),
            **report_scan,
            **teacher_receipt_scan,
        },
        "allowed_teacher_surfaces": [
            {
                "path": as_posix(path),
                "role": "teacher_wrapper" if path in TEACHER_FILES else "teacher_delegate",
            }
            for path in sorted(TEACHER_FILES | TEACHER_DELEGATE_FILES, key=as_posix)
        ],
        "teacher_delegate_hits": delegate_hits,
        "allowed_teacher_hits": allowed_teacher_hits,
        "benign_metadata_or_policy_hits": benign_hits[:100],
        "violations": all_violations,
    }
    write_json(ROOT / args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


def scan_teacher_receipts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = read_json(TEACHER_POLICY_PATH)
    rows = read_jsonl_follow_pointer(TEACHER_CALLS_PATH)
    if not isinstance(policy, dict):
        return [
            {
                "path": as_posix(TEACHER_POLICY_PATH.relative_to(ROOT)),
                "kind": "teacher_provider_policy_missing_or_invalid",
                "classification": "violation",
            }
        ], {"scanned_teacher_receipts": 0, "teacher_provider_counts": {}}
    return audit_teacher_receipt_rows(rows, policy)


def audit_teacher_receipt_rows(
    rows: list[Any], policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    provider_counts: dict[str, int] = {}
    scanned = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(
                {
                    "path": as_posix(TEACHER_CALLS_PATH.relative_to(ROOT)),
                    "kind": "teacher_receipt_not_object",
                    "row_index": index,
                    "classification": "violation",
                }
            )
            continue
        scanned += 1
        provider = str(row.get("provider") or "missing").strip().lower()
        model = str(row.get("model") or "missing").strip().lower()
        provider_key = f"{provider}/{model}"
        provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1
        decision = teacher_receipt_decision(policy, row)
        if not decision["accepted"]:
            violations.append(
                {
                    "path": as_posix(TEACHER_CALLS_PATH.relative_to(ROOT)),
                    "kind": "teacher_receipt_provider_provenance_invalid",
                    "row_index": index,
                    "request_id": row.get("request_id"),
                    "provider": provider,
                    "model": model,
                    "reject_reasons": decision["reject_reasons"],
                    "classification": "violation",
                }
            )
    return violations, {
        "scanned_teacher_receipts": scanned,
        "teacher_provider_counts": dict(sorted(provider_counts.items())),
    }


def scan_code(files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hits: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    delegates: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(ROOT)
        text = read_text(path)
        if not text:
            continue
        if rel in TEACHER_DELEGATE_FILES and "scripts/teacher_oracle.py" in text:
            delegates.append(
                {
                    "path": as_posix(rel),
                    "kind": "teacher_delegate",
                    "classification": "allowed_teacher_delegate",
                }
            )
        if (
            rel not in TEACHER_FILES
            and rel not in TEACHER_POLICY_OBSERVER_FILES
            and rel not in SELF_FILES
        ):
            codex_match = re.search(
                r"(?is)(codex.{0,240}[\"']exec[\"']|[\"']exec[\"'].{0,240}codex)",
                text,
            )
            if codex_match:
                violations.append(
                    hit(rel, "codex_exec_outside_teacher", codex_match.group(0), "violation")
                )
        if rel not in SELF_FILES:
            claude_match = re.search(
                r"(?is)(?:subprocess\.(?:run|Popen|check_output|check_call)|os\.(?:system|execv|execve))"
                r".{0,240}[\"']claude[\"']",
                text,
            )
            if claude_match:
                violations.append(
                    hit(rel, "claude_cli_invocation_forbidden", claude_match.group(0), "violation")
                )
        for name, pattern in ACTIVE_INFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                item = hit(rel, name, match.group(0), classify(rel, name))
                if item["classification"] == "violation":
                    violations.append(item)
                elif item["classification"] != "ignored_self":
                    hits.append(item)
    return hits, violations, delegates


def classify(rel: Path, pattern_name: str) -> str:
    if rel in SELF_FILES:
        return "ignored_self"
    if rel in TEACHER_FILES:
        return "allowed_teacher"
    if rel in TEACHER_POLICY_OBSERVER_FILES and pattern_name == "codex_cli_policy":
        return "benign_teacher_policy_observer"
    if rel in LOCAL_COMPAT_FILES and pattern_name in {
        "openai_api_endpoint",
        "generic_bearer_auth",
    }:
        return "benign_local_openai_compatible_endpoint"
    if rel in LOCAL_SECRET_SCANNER_FILES and pattern_name == "generic_bearer_auth":
        return "benign_local_secret_redaction_pattern"
    if rel == Path("scripts/hive_rented_compute.py") and pattern_name == "generic_bearer_auth":
        return "benign_cloud_compute_auth_template"
    if rel == Path("configs/autonomy_policy.json") and pattern_name in {
        "ollama_local_model",
        "vllm_local_model",
        "llama_cpp_local_model",
    }:
        # Policy text may mention a blocked capability name; it is not an invocation.
        return "benign_policy_text"
    if pattern_name in {
        "transformers_local_model",
        "sentence_transformers",
        "ollama_local_model",
        "vllm_local_model",
        "llama_cpp_local_model",
        "speech_inference_import",
    }:
        return "benign_local_model_library"
    return "violation"


def scan_report_external_inference(*, include_large_reports: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    violations: list[dict[str, Any]] = []
    stats = {"scanned_report_files": 0, "skipped_large_report_files": 0}
    if not REPORTS.exists():
        return violations, stats
    for path in REPORTS.glob("*.json"):
        rel = path.relative_to(ROOT)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > REPORT_JSON_PARSE_MAX_BYTES:
            if include_large_reports:
                stats["scanned_report_files"] += 1
                violations.extend(scan_large_report_external_inference(path, rel))
            else:
                stats["skipped_large_report_files"] += 1
            continue
        if not report_may_contain_external_inference(path):
            continue
        stats["scanned_report_files"] += 1
        value = read_json(path)
        if value is None:
            continue
        for item in report_external_values(value, []):
            key_path = ".".join(item["path"])
            key = item["path"][-1] if item["path"] else ""
            raw = item["value"]
            if is_allowed_teacher_report(rel, value):
                continue
            if key == "external_inference_calls" and number(raw) > 0:
                violations.append(
                    {
                        "path": as_posix(rel),
                        "kind": "report_external_inference_calls_nonzero",
                        "key_path": key_path,
                        "value": raw,
                        "classification": "violation",
                    }
                )
            if key == "external_inference_violations" and truthy_violation_value(raw):
                violations.append(
                    {
                        "path": as_posix(rel),
                        "kind": "report_external_inference_violations_nonempty",
                        "key_path": key_path,
                        "value": raw,
                        "classification": "violation",
                    }
                )
    return violations, stats


def report_may_contain_external_inference(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            carry = b""
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    return False
                window = carry + chunk
                if any(key in window for key in REPORT_EXTERNAL_KEYS):
                    return True
                carry = window[-128:]
    except OSError:
        return False


def scan_large_report_external_inference(path: Path, rel: Path) -> list[dict[str, Any]]:
    if rel.name.startswith(TEACHER_REPORT_PREFIXES):
        return []
    violations: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                if "external_inference_calls" in line:
                    match = re.search(r'"external_inference_calls"\s*:\s*(-?\d+(?:\.\d+)?)', line)
                    if match and number(match.group(1)) > 0:
                        violations.append(
                            {
                                "path": as_posix(rel),
                                "kind": "report_external_inference_calls_nonzero",
                                "key_path": f"line:{line_number}.external_inference_calls",
                                "value": match.group(1),
                                "classification": "violation",
                            }
                        )
                if "external_inference_violations" in line:
                    match = re.search(r'"external_inference_violations"\s*:\s*(.+?)(?:,)?\s*$', line)
                    if match and truthy_violation_value(match.group(1).strip()):
                        violations.append(
                            {
                                "path": as_posix(rel),
                                "kind": "report_external_inference_violations_nonempty",
                                "key_path": f"line:{line_number}.external_inference_violations",
                                "value": match.group(1).strip()[:240],
                                "classification": "violation",
                            }
                        )
    except OSError:
        return violations
    return violations


def is_allowed_teacher_report(rel: Path, value: Any) -> bool:
    if rel.parent != Path("reports"):
        return False
    if rel.name.startswith(TEACHER_REPORT_PREFIXES):
        return True
    if isinstance(value, dict) and value.get("policy") in TEACHER_AUDIT_REPORT_POLICIES:
        return True
    if isinstance(value, dict) and value.get("policy") == "project_theseus_permissive_growth_mode_report_v1":
        summary = value.get("summary") if isinstance(value.get("summary"), dict) else {}
        no_cheat = summary.get("no_cheat_counters") if isinstance(summary.get("no_cheat_counters"), dict) else {}
        return (
            no_cheat.get("runtime_external_inference_forbidden") is True
            and number(no_cheat.get("runtime_external_inference_calls")) == 0
            and no_cheat.get("growth_loop_runtime_external_violation") is not True
            and no_cheat.get("teacher_training_external_inference_calls") is not None
        )
    if not rel.name.startswith(TEACHER_GUIDANCE_REPORT_PREFIXES):
        return False
    if not isinstance(value, dict):
        return False
    if value.get("policy") != "project_theseus_architecture_guidance_loop_v1":
        return False
    teacher = value.get("teacher") if isinstance(value.get("teacher"), dict) else {}
    if teacher.get("mode") != "proposal":
        return False
    gates = value.get("gates") if isinstance(value.get("gates"), list) else []
    proposal_gate_passed = any(
        isinstance(gate, dict)
        and gate.get("gate") == "teacher_proposal_only"
        and gate.get("passed") is True
        for gate in gates
    )
    if not proposal_gate_passed:
        return False
    return True


def report_external_values(value: Any, path: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = [*path, str(key)]
            if key in {"external_inference_calls", "external_inference_violations"}:
                rows.append({"path": next_path, "value": child})
            rows.extend(report_external_values(child, next_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            rows.extend(report_external_values(child, [*path, str(idx)]))
    return rows


def truthy_violation_value(value: Any) -> bool:
    if value in (None, False, 0, "", [], {}):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "[]", "{}", "false", "null", "0"}
    if isinstance(value, list):
        return any(truthy_violation_value(item) for item in value)
    if isinstance(value, dict):
        return any(truthy_violation_value(item) for item in value.values())
    return True


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def hit(rel: Path, kind: str, evidence: str, classification: str) -> dict[str, Any]:
    return {
        "path": as_posix(rel),
        "kind": kind,
        "evidence": evidence[:240].replace("\n", "\\n"),
        "classification": classification,
    }


def scannable_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    for name in ["Cargo.toml", "Cargo.lock"]:
        path = ROOT / name
        if path.exists():
            files.append(path)
    return sorted(set(files))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
