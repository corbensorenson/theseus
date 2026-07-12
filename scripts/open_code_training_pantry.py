"""Governed open-source code data pantry for Code LM training.

The pantry is intentionally conservative:
- only GitHub repositories with allowlisted SPDX licenses are admitted;
- public benchmark names are excluded from paths and repo names;
- downloaded tarballs and extracted rows live on D: by default;
- extracted function/expression rows are train-only and are not public benchmark evidence;
- arbitrary open-source expressions are not candidate-bank eligible by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "training_data" / "open_code_pantry"
DEFAULT_REPOS = "psf/requests,pallets/click,pallets/flask,pypa/packaging,pytest-dev/pluggy"
ALLOWED_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
    "Unlicense",
}
BENCHMARK_EXCLUSION_TOKENS = {
    "apps",
    "bigcodebench",
    "evalplus",
    "humaneval",
    "human_eval",
    "livecodebench",
    "mbpp",
    "swe-bench",
    "swe_bench",
}
SOURCE_EXTENSIONS = {
    ".py", ".rs", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".html", ".htm", ".css", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
}
BANNED_EXPR_FRAGMENTS = {
    "__",
    "eval(",
    "exec(",
    "open(",
    "compile(",
    "subprocess",
    "socket",
    "requests.",
    "urllib.",
    "os.",
    "sys.",
    "shutil.",
    "pathlib.",
}
BANNED_BODY_FRAGMENTS = BANNED_EXPR_FRAGMENTS | {
    "input(",
    "getattr(",
    "setattr(",
    "delattr(",
    "globals(",
    "locals(",
    "memoryview(",
    "__import__",
    "importlib",
    "pickle",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio.",
    "pytest",
    "unittest",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--repo-config",
        default="",
        help="Optional JSON file with a repos list and default cap overrides.",
    )
    parser.add_argument("--repos", default=DEFAULT_REPOS)
    parser.add_argument("--max-repos", type=int, default=3)
    parser.add_argument("--max-files-per-repo", type=int, default=250)
    parser.add_argument("--max-bytes-per-file", type=int, default=180_000)
    parser.add_argument("--max-expressions", type=int, default=800)
    parser.add_argument(
        "--max-full-body-functions",
        type=int,
        default=0,
        help="Maximum full-body Python function rows. Defaults to max-expressions when omitted or zero.",
    )
    parser.add_argument("--candidate-expression-eligible", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default="reports/open_code_training_pantry.json")
    args = parser.parse_args()
    config = read_repo_config(args.repo_config)
    config_defaults = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}
    for field in (
        "max_repos",
        "max_files_per_repo",
        "max_bytes_per_file",
        "max_expressions",
        "max_full_body_functions",
    ):
        arg_name = field.replace("_", "-")
        if field in config_defaults and option_omitted(f"--{arg_name}"):
            setattr(args, field, int(config_defaults[field]))
    if args.max_full_body_functions <= 0:
        args.max_full_body_functions = args.max_expressions

    root = Path(args.root)
    tarball_dir = root / "tarballs"
    samples_path = root / "samples" / "open_code_samples.jsonl"
    samples_manifest_path = root / "samples_manifest.json"
    train_path = root / "private_train" / "open_code_expressions.jsonl"
    tarball_dir.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    config_repos = repos_from_config(config)
    repos = config_repos or [repo.strip() for repo in str(args.repos).split(",") if repo.strip()]
    repos = [repo for repo in repos if not excluded_by_benchmark_name(repo)][: max(0, args.max_repos)]
    admitted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for repo in repos:
        try:
            meta = github_json(f"https://api.github.com/repos/{repo}")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"repo": repo, "reason": "metadata_fetch_failed", "error": str(exc)[:400]})
            continue
        license_spdx = str((meta.get("license") or {}).get("spdx_id") or "NOASSERTION")
        if license_spdx not in ALLOWED_LICENSES:
            skipped.append({"repo": repo, "reason": "license_not_allowlisted", "license_spdx": license_spdx})
            continue
        default_branch = str(meta.get("default_branch") or "main")
        tarball_path = tarball_dir / f"{repo.replace('/', '__')}__{default_branch}.tar.gz"
        if args.refresh or not tarball_path.exists():
            try:
                download_bytes(
                    f"https://api.github.com/repos/{repo}/tarball/{default_branch}",
                    tarball_path,
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"repo": repo, "reason": "tarball_download_failed", "error": str(exc)[:400]})
                continue

        repo_sample_count = 0
        repo_expr_count = 0
        repo_full_body_count = 0
        repo_row = {
            "repo": repo,
            "license_spdx": license_spdx,
            "default_branch": default_branch,
            "html_url": meta.get("html_url"),
            "tarball": str(tarball_path).replace("\\", "/"),
            "tarball_sha256": stable_hash_hex(tarball_path.read_bytes()),
        }
        try:
            for file_row in iter_tar_source_files(
                tarball_path,
                repo=repo,
                license_spdx=license_spdx,
                max_files=max(1, args.max_files_per_repo),
                max_bytes=max(1024, args.max_bytes_per_file),
            ):
                sample_rows.append(file_row)
                repo_sample_count += 1
                if file_row["language"] == "python" and (
                    expression_count(train_rows) < max(0, args.max_expressions)
                    or full_body_count(train_rows) < max(0, args.max_full_body_functions)
                ):
                    for expr_row in extract_python_training_rows(
                        file_row,
                        candidate_expression_eligible=bool(args.candidate_expression_eligible),
                        expression_remaining=max(0, args.max_expressions - expression_count(train_rows)),
                        full_body_remaining=max(0, args.max_full_body_functions - full_body_count(train_rows)),
                    ):
                        train_rows.append(expr_row)
                        if expr_row.get("category") == "open_code_expr":
                            repo_expr_count += 1
                        if expr_row.get("category") == "open_code_full_body":
                            repo_full_body_count += 1
                        if (
                            expression_count(train_rows) >= max(0, args.max_expressions)
                            and full_body_count(train_rows) >= max(0, args.max_full_body_functions)
                        ):
                            break
        except Exception as exc:  # noqa: BLE001
            skipped.append({"repo": repo, "reason": "tarball_scan_failed", "error": str(exc)[:400]})
            continue
        repo_row.update(
            {
                "sample_count": repo_sample_count,
                "expression_train_count": repo_expr_count,
                "full_body_train_count": repo_full_body_count,
            }
        )
        admitted.append(repo_row)

    write_jsonl(samples_path, sample_rows)
    write_jsonl(train_path, train_rows)
    samples_manifest = {
        "policy": "project_theseus_open_code_canonical_shard_manifest_v1",
        "created_utc": now(),
        "sample_jsonl": str(samples_path.relative_to(root)).replace("\\", "/"),
        "sample_jsonl_sha256": stable_hash_hex(samples_path.read_bytes()),
        "sample_count": len(sample_rows),
        "allowed_licenses": sorted(ALLOWED_LICENSES),
        "admitted_sources": [
            {
                "repo": row["repo"],
                "license_spdx": row["license_spdx"],
                "default_branch": row["default_branch"],
                "tarball": row["tarball"],
                "tarball_sha256": row["tarball_sha256"],
                "sample_count": row["sample_count"],
            }
            for row in admitted
        ],
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "benchmark_excluded": True,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json(samples_manifest_path, samples_manifest)
    report = {
        "policy": "project_theseus_open_code_training_pantry_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if admitted else "YELLOW",
        "root": str(root).replace("\\", "/"),
        "sample_jsonl": str(samples_path).replace("\\", "/"),
        "sample_manifest": str(samples_manifest_path).replace("\\", "/"),
        "private_train_jsonl": str(train_path).replace("\\", "/"),
        "summary": {
            "admitted_repo_count": len(admitted),
            "skipped_repo_count": len(skipped),
            "source_sample_count": len(sample_rows),
            "private_train_expression_count": expression_count(train_rows),
            "private_train_full_body_count": full_body_count(train_rows),
            "private_train_row_count": len(train_rows),
            "allowed_licenses": sorted(ALLOWED_LICENSES),
            "benchmark_exclusion_tokens": sorted(BENCHMARK_EXCLUSION_TOKENS),
            "candidate_expression_eligible": bool(args.candidate_expression_eligible),
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
            "benchmark_evidence_level": "permissive_open_source_train_only",
            "benchmark_excluded": True,
            "repo_config": args.repo_config,
        },
        "admitted_repos": admitted,
        "skipped_repos": skipped,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if admitted else 1


def github_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "ProjectTheseusCodePantry/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def read_repo_config(value: str) -> dict[str, Any]:
    if not value:
        return {}
    path = resolve(value)
    if not path.exists():
        raise FileNotFoundError(f"repo config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("repo config must be a JSON object")
    return raw


def repos_from_config(config: dict[str, Any]) -> list[str]:
    raw_repos = config.get("repos", [])
    if not isinstance(raw_repos, list):
        return []
    repos: list[str] = []
    for row in raw_repos:
        if isinstance(row, str):
            repo = row
            enabled = True
        elif isinstance(row, dict):
            repo = str(row.get("repo") or "")
            enabled = bool(row.get("enabled", True))
        else:
            continue
        repo = repo.strip()
        if enabled and repo:
            repos.append(repo)
    return repos


def option_omitted(name: str) -> bool:
    import sys

    return not any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv)


def download_bytes(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ProjectTheseusCodePantry/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def iter_tar_source_files(
    tarball_path: Path,
    *,
    repo: str,
    license_spdx: str,
    max_files: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(tarball_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if len(rows) >= max_files:
                break
            if not member.isfile() or member.size <= 0 or member.size > max_bytes:
                continue
            path = member.name.replace("\\", "/")
            if excluded_by_benchmark_name(path):
                continue
            ext = Path(path).suffix.lower()
            if ext not in SOURCE_EXTENSIONS:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            text = decode_source(raw)
            if not text.strip():
                continue
            rows.append(
                {
                    "repo": repo,
                    "path": strip_tar_root(path),
                    "language": language_for_extension(ext),
                    "license_spdx": license_spdx,
                    "size_bytes": len(raw),
                    "sha256": stable_hash_hex(raw),
                    "text_sha256": stable_hash_hex(text.encode("utf-8")),
                    "text": text,
                    "benchmark_evidence_level": "permissive_open_source_train_only",
                    "public_benchmark": False,
                    "public_benchmark_solutions_included": False,
                    "public_tests_included": False,
                    "benchmark_excluded": True,
                    "provenance": {
                        "source": "github_tarball",
                        "repo": repo,
                        "path": strip_tar_root(path),
                        "license_spdx": license_spdx,
                        "tarball": str(tarball_path).replace("\\", "/"),
                        "source_file_sha256": stable_hash_hex(raw),
                        "benchmark_excluded": True,
                        "public_benchmark": False,
                        "public_benchmark_solutions_included": False,
                        "public_tests_included": False,
                    },
                }
            )
    return rows


def extract_python_training_rows(
    file_row: dict[str, Any],
    *,
    candidate_expression_eligible: bool,
    expression_remaining: int,
    full_body_remaining: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = str(file_row.get("text") or "")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return rows
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        entry_point = safe_identifier(node.name)
        if full_body_remaining > full_body_count(rows):
            body = function_body_text(node)
            if useful_body(body):
                rows.append(open_code_row(file_row, node, entry_point, body=body, category="open_code_full_body"))
        if expression_remaining > expression_count(rows):
            expr = single_return_expression(node)
            if expr:
                expr_text = ast.unparse(expr).strip()
                if useful_expression(expr_text):
                    rows.append(
                        open_code_row(
                            file_row,
                            node,
                            entry_point,
                            expr=expr_text,
                            body=f"return {expr_text}",
                            category="open_code_expr",
                            candidate_expression_eligible=candidate_expression_eligible,
                        )
                    )
    return rows


def open_code_row(
    file_row: dict[str, Any],
    node: ast.FunctionDef,
    entry_point: str,
    *,
    category: str,
    body: str = "",
    expr: str = "",
    candidate_expression_eligible: bool = False,
) -> dict[str, Any]:
    signature = visible_signature(node)
    source_function_id = f"{file_row['repo']}:{file_row['path']}:{node.name}:{node.lineno}"
    body_or_expr = body or f"return {expr}"
    row_hash = stable_hash_hex(f"{source_function_id}:{category}:{body_or_expr}".encode("utf-8"))[:16]
    tags = [
        "open_code_permissive_pantry",
        "train_only",
        "no_public_benchmark",
        "benchmark_excluded",
        category,
    ]
    return {
        "task_id": f"{category}_{row_hash}",
        "source_task_id": source_function_id,
        "card_id": "open_code_permissive_pantry",
        "source_id": f"github:{file_row['repo']}",
        "split": "train",
        "category": category,
        "prompt": prompt_for_function(entry_point, signature, file_row, category),
        "entry_point": entry_point,
        "solution_expr": expr or first_return_expression_from_body(body_or_expr),
        "solution_body": body_or_expr,
        "tests": "",
        "tags": tags,
        "benchmark_evidence_level": "permissive_open_source_train_only",
        "public_benchmark": False,
        "public_benchmark_solutions_included": False,
        "public_tests_included": False,
        "benchmark_excluded": True,
        "license_spdx": file_row["license_spdx"],
        "candidate_expression_eligible": bool(candidate_expression_eligible),
        "provenance": {
            **(file_row.get("provenance") if isinstance(file_row.get("provenance"), dict) else {}),
            "function_name": node.name,
            "entry_point": entry_point,
            "line_start": getattr(node, "lineno", None),
            "line_end": getattr(node, "end_lineno", None),
            "source_file_sha256": file_row.get("sha256"),
            "row_sha256": stable_hash_hex(body_or_expr),
            "benchmark_excluded": True,
            "public_benchmark": False,
            "public_benchmark_solutions_included": False,
            "public_tests_included": False,
        },
        "decoder_contract": decoder_contract_for_function(node, signature, category),
    }


def expression_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("category") == "open_code_expr")


def full_body_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("category") == "open_code_full_body")


def single_return_expression(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.expr | None:
    body = [item for item in node.body if not isinstance(item, ast.Expr) or not isinstance(getattr(item, "value", None), ast.Constant)]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    return body[0].value


def function_body_text(node: ast.FunctionDef) -> str:
    body_items = [
        item
        for item in node.body
        if not (
            isinstance(item, ast.Expr)
            and isinstance(getattr(item, "value", None), ast.Constant)
            and isinstance(getattr(item.value, "value", None), str)
        )
    ]
    try:
        return "\n".join(ast.unparse(item).strip() for item in body_items if ast.unparse(item).strip()).strip()
    except Exception:  # noqa: BLE001
        return ""


def useful_body(body: str) -> bool:
    if not body or len(body) < 12 or len(body) > 2400:
        return False
    lowered = body.lower()
    if "return" not in lowered:
        return False
    if any(fragment in lowered for fragment in BANNED_BODY_FRAGMENTS):
        return False
    return True


def first_return_expression_from_body(body: str) -> str:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                return ast.unparse(node.value).strip()
            except Exception:  # noqa: BLE001
                return ""
    return ""


def visible_signature(node: ast.FunctionDef) -> str:
    names: list[str] = []
    args = list(node.args.posonlyargs) + list(node.args.args)
    for arg in args:
        if arg.arg in {"self", "cls"}:
            continue
        names.append(safe_identifier(arg.arg))
    if node.args.vararg:
        names.append(f"*{safe_identifier(node.args.vararg.arg)}")
    for arg in node.args.kwonlyargs:
        names.append(f"{safe_identifier(arg.arg)}=None")
    if node.args.kwarg:
        names.append(f"**{safe_identifier(node.args.kwarg.arg)}")
    return f"{safe_identifier(node.name)}({', '.join(names)})"


def prompt_for_function(entry_point: str, signature: str, file_row: dict[str, Any], category: str) -> str:
    row_kind = "full function body" if category == "open_code_full_body" else "return expression"
    return (
        f"Write a Python {row_kind} for {signature}. "
        f"Preserve the visible interface for {entry_point} and infer robust return-shape behavior "
        f"from a permissive open-source training example in {file_row['repo']}."
    )


def decoder_contract_for_function(node: ast.FunctionDef, signature: str, category: str) -> dict[str, Any]:
    argument_names = [
        safe_identifier(arg.arg)
        for arg in list(node.args.posonlyargs) + list(node.args.args)
        if arg.arg not in {"self", "cls"}
    ]
    constructs = required_constructs(node)
    return {
        "policy": "project_theseus_open_code_decoder_contract_v1",
        "source": "permissive_open_code_train_only",
        "entry_point": safe_identifier(node.name),
        "visible_signature": signature,
        "argument_names": argument_names,
        "argument_count": len(argument_names),
        "return_shape_hint": return_shape_hint(node),
        "required_constructs": constructs,
        "category": category,
        "public_tests_used": False,
        "public_solutions_used": False,
        "benchmark_excluded": True,
        "generation_plan": {
            "receiver_family": "open_code_real_function_body" if category == "open_code_full_body" else "open_code_expression",
            "preserve_visible_signature": True,
            "infer_argument_roles": True,
            "infer_return_shape": True,
            "prefer_full_body_token_decode": category == "open_code_full_body",
            "public_tests_used": False,
            "public_solutions_used": False,
            "benchmark_excluded": True,
        },
    }


def required_constructs(node: ast.FunctionDef) -> list[str]:
    constructs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.If):
            constructs.add("branch")
        elif isinstance(child, (ast.For, ast.While)):
            constructs.add("loop")
        elif isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            constructs.add("local_state")
        elif isinstance(child, ast.Try):
            constructs.add("exception_path")
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            constructs.add("comprehension")
        elif isinstance(child, ast.Call):
            constructs.add("call")
        elif isinstance(child, ast.Return):
            constructs.add("return")
    order = ["return", "branch", "loop", "local_state", "comprehension", "call", "exception_path"]
    return [item for item in order if item in constructs]


def return_shape_hint(node: ast.FunctionDef) -> str:
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            value = child.value
            if isinstance(value, ast.List):
                return "list"
            if isinstance(value, ast.Tuple):
                return "tuple"
            if isinstance(value, ast.Dict):
                return "dict"
            if isinstance(value, ast.Set):
                return "set"
            if isinstance(value, ast.Constant):
                if isinstance(value.value, bool):
                    return "bool"
                if isinstance(value.value, int):
                    return "int"
                if isinstance(value.value, float):
                    return "float"
                if isinstance(value.value, str):
                    return "str"
                if value.value is None:
                    return "none"
            if isinstance(value, ast.Compare):
                return "bool"
            if isinstance(value, ast.BinOp):
                return "computed"
            if isinstance(value, ast.Call):
                return "call_result"
            return type(value).__name__.removeprefix("ast.").lower()
    return "unknown"


def useful_expression(expr: str) -> bool:
    if not expr or len(expr) > 220:
        return False
    lowered = expr.lower()
    if any(fragment in lowered for fragment in BANNED_EXPR_FRAGMENTS):
        return False
    return True


def excluded_by_benchmark_name(value: str) -> bool:
    lowered = value.lower().replace("\\", "/")
    return any(token in lowered for token in BENCHMARK_EXCLUSION_TOKENS)


def decode_source(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def strip_tar_root(path: str) -> str:
    parts = path.split("/", 1)
    return parts[1] if len(parts) == 2 else path


def language_for_extension(ext: str) -> str:
    return {
        ".py": "python",
        ".rs": "rust",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c_header",
        ".hpp": "cpp_header",
    }.get(ext, ext.lstrip("."))


def safe_identifier(value: str) -> str:
    cleaned = re.sub(r"\W+", "_", str(value or "open_code_func")).strip("_")
    if not cleaned:
        cleaned = "open_code_func"
    if cleaned[0].isdigit():
        cleaned = f"open_code_{cleaned}"
    return cleaned


def stable_hash_hex(data: bytes | str) -> str:
    import hashlib

    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
