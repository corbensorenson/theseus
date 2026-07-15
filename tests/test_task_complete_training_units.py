from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import task_complete_training_units as units  # noqa: E402
import training_data_admission_v1 as admission  # noqa: E402
import training_data_lineage_audit as lineage  # noqa: E402


def config() -> dict:
    return json.loads((ROOT / "configs" / "task_complete_training_units.json").read_text())


def clean_contamination() -> dict:
    return lineage.contamination_index_from_rows([])


def test_exact_contamination_is_normalized_and_quarantined() -> None:
    public = {"prompt": "Return the stable checksum for every accepted record."}
    contamination = lineage.contamination_index_from_rows([public])
    row = units.base_unit(
        config(),
        source={"id": "fixture"},
        source_task_id="fixture-1",
        arm_id="python",
        task_family="fixture",
        visible_context="  RETURN the stable checksum for every accepted record.  ",
        target="def solve(records):\n    return 17\n",
        license_spdx="MIT",
        provenance={"static_open_corpus": True},
        contamination=contamination,
    )
    units.finish_unit(
        row,
        {"state": "passed", "strength": "executable_target_pass_starter_fail"},
        config(),
    )
    assert row["contamination"]["exact_overlap"] is True
    assert row["decision"] == "quarantine"


def test_split_is_deterministic_and_cross_split_source_is_rejected() -> None:
    cfg = config()
    observed = units.split_for(cfg, "same-source-task")
    assert observed == units.split_for(cfg, "same-source-task")
    fixture = [
        {"source_task_id": "same", "split": "train"},
        {"source_task_id": "same", "split": "confirmation"},
    ]
    assert units.split_leakage(fixture) == ["same"]


def test_coverage_counts_only_admitted_required_strength() -> None:
    cfg = config()
    for arm in cfg["coverage_floors_for_50m_scale_proposal"].values():
        arm["minimum_verified_units"] = 1
        arm["minimum_target_positions"] = 1
    cfg["coverage_floors_for_50m_scale_proposal"]["english"].update(
        minimum_human_contributed_share=1.0,
        minimum_multi_turn_share=1.0,
    )
    rows = [{
        "decision": "admit",
        "arm_id": "english",
        "target_positions": 10,
        "verification": {
            "strength": "governed_conversation_target_bound",
            "multi_turn": True,
        },
        "provenance": {"provenance_class": "human_contributed"},
    }]
    for arm in ("python", "javascript_typescript", "rust"):
        rows.append({
            "decision": "admit",
            "arm_id": arm,
            "target_positions": 10,
            "verification": {"strength": "parser_only"},
            "provenance": {},
        })
    rows.append({
        "decision": "admit",
        "arm_id": "html_css",
        "target_positions": 10,
        "verification": {"strength": "dom_a11y_layout_render_delta"},
        "provenance": {},
    })
    coverage, gaps = units.coverage_summary(cfg, rows)
    assert coverage["english"]["ready"] is True
    assert coverage["html_css"]["ready"] is True
    assert coverage["python"]["verified_units"] == 0
    assert {row["arm_id"] for row in gaps} == {"python", "javascript_typescript", "rust"}


def test_executable_unit_requires_target_pass_and_starter_fail(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "repo"
    exercise = source_root / "exercise"
    exercise.mkdir(parents=True)
    (exercise / "solution.py").write_text("raise NotImplementedError\n")
    (exercise / "solution_test.py").write_text("# fixture\n")
    outcomes = iter([
        {"ok": False, "returncode": 1, "stdout": "", "stderr": "starter failed", "duration_ms": 1},
        {"ok": True, "returncode": 0, "stdout": "target passed", "stderr": "", "duration_ms": 1},
    ])
    monkeypatch.setattr(units, "run_exercise", lambda *args, **kwargs: next(outcomes))
    result = units.verify_exercism(
        source_root,
        exercise,
        "python",
        {"solution.py": "def solve():\n    return 1\n"},
        ["solution_test.py"],
        {"timeout_seconds": 5},
    )
    assert result["state"] == "passed"
    assert result["target_passed"] is True
    assert result["starter_failed"] is True


def test_starter_pass_is_not_task_complete_evidence(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "repo"
    exercise = source_root / "exercise"
    exercise.mkdir(parents=True)
    (exercise / "solution.py").write_text("return_value = 1\n")
    (exercise / "solution_test.py").write_text("# fixture\n")
    outcomes = iter([
        {"ok": True, "returncode": 0, "stdout": "starter passed", "stderr": "", "duration_ms": 1},
        {"ok": True, "returncode": 0, "stdout": "target passed", "stderr": "", "duration_ms": 1},
    ])
    monkeypatch.setattr(units, "run_exercise", lambda *args, **kwargs: next(outcomes))
    result = units.verify_exercism(
        source_root,
        exercise,
        "python",
        {"solution.py": "return_value = 2\n"},
        ["solution_test.py"],
        {"timeout_seconds": 5},
    )
    assert result["state"] == "failed"
    assert result["starter_failed"] is False


def test_javascript_ast_rows_are_bound_to_source_spans(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "repo"
    source_path = source_root / "src" / "value.ts"
    source_path.parent.mkdir(parents=True)
    text = "export function value() {\n  return 42\n}\n"
    source_path.write_text(text)
    start = text.index("{") + 1
    end = text.rindex("}")

    def fake_run(command, workdir, timeout, **kwargs):
        Path(command[-1]).write_text(json.dumps({
            "typescriptVersion": "5.9.3",
            "records": [{
                "path": str(source_path),
                "qualified_name": "value",
                "start_byte": start,
                "end_byte": end,
                "target_body": text[start:end],
                "starter_body": '\n  throw new Error("hole")\n',
            }],
        }))
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(units, "run_sandboxed", fake_run)
    rows = units.discover_javascript_function_holes(
        source_root,
        {
            "source_globs": ["src/**/*.ts"],
            "minimum_body_bytes": 1,
            "maximum_body_bytes": 1000,
            "context_lines": 5,
            "test_suite_paths": ["src/value.spec.ts"],
        },
        parser_toolchain={
            "runtime_root": str(tmp_path / "toolchain"),
            "node_executable": sys.executable,
            "parser_script": str(tmp_path / "parser.mjs"),
            "typescript_version": "5.9.3",
        },
    )
    assert len(rows) == 1
    assert rows[0]["target_body"] == "\n  return 42\n"
    assert rows[0]["path"] == "src/value.ts"
    assert rows[0]["visible_source"].count("<THESEUS_IMPLEMENTATION_HOLE>") == 1


def test_javascript_mutation_fault_is_not_a_kill_and_source_is_restored(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "value.ts"
    text = "export function value() {\n  return 42\n}\n"
    source_path.write_text(text)
    start = text.index("{") + 1
    end = text.rindex("}")
    monkeypatch.setattr(
        units,
        "run_javascript_project_tests",
        lambda *args, **kwargs: {"ok": False, "fault": "timeout", "duration_ms": 10},
    )
    result = units.verify_javascript_function_hole(
        tmp_path,
        {
            "path": "value.ts",
            "start_byte": start,
            "end_byte": end,
            "target_body": text[start:end],
            "starter_body": '\n  throw new Error("hole")\n',
        },
        {"command": ["false"], "environment": {}, "timeout_seconds": 1},
        baseline_run={"ok": True, "returncode": 0},
        baseline_receipt={"ok": True},
    )
    assert result["state"] == "failed"
    assert result["starter_failed"] is False
    assert source_path.read_text() == text


def test_javascript_mutation_requires_normal_nonzero_test_exit(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "value.ts"
    text = "export function value() {\n  return 42\n}\n"
    source_path.write_text(text)
    start = text.index("{") + 1
    end = text.rindex("}")
    monkeypatch.setattr(
        units,
        "run_javascript_project_tests",
        lambda *args, **kwargs: {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "assertion failed",
            "duration_ms": 10,
        },
    )
    result = units.verify_javascript_function_hole(
        tmp_path,
        {
            "path": "value.ts",
            "start_byte": start,
            "end_byte": end,
            "target_body": text[start:end],
            "starter_body": '\n  throw new Error("hole")\n',
        },
        {"command": ["false"], "environment": {}, "timeout_seconds": 1},
        baseline_run={"ok": True, "returncode": 0},
        baseline_receipt={"ok": True},
    )
    assert result["state"] == "passed"
    assert result["starter_failed"] is True
    assert source_path.read_text() == text


def test_javascript_test_selection_uses_bounded_relative_import_graph(tmp_path: Path) -> None:
    source_root = tmp_path / "repo"
    util = source_root / "src" / "util.ts"
    api = source_root / "src" / "api.ts"
    test = source_root / "src" / "__tests__" / "api.spec.ts"
    test.parent.mkdir(parents=True)
    util.write_text("export const value = 42\n")
    api.write_text("export { value } from './util'\n")
    test.write_text("import { value } from '../api'\n")
    observed = units.javascript_import_graph_test_map(
        source_root,
        {
            "test_selection": {
                "kind": "bounded_import_graph_v1",
                "maximum_import_depth": 3,
                "maximum_tests_per_source_file": 1,
                "basename_fallback": True,
            }
        },
        source_paths={util.resolve(), api.resolve()},
        test_paths={test.resolve()},
        imports={
            str(util.resolve()): [],
            str(api.resolve()): ["./util"],
            str(test.resolve()): ["../api"],
        },
    )
    expected = ["src/__tests__/api.spec.ts"]
    assert observed["src/api.ts"] == expected
    assert observed["src/util.ts"] == expected


def test_javascript_baseline_resolution_keeps_only_clean_tests(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_run(_workdir, _toolchain, *, test_paths=None):
        paths = tuple(test_paths or [])
        return {
            "ok": paths == ("clean.spec.ts",),
            "returncode": 0 if paths == ("clean.spec.ts",) else 1,
            "stdout": "",
            "stderr": "baseline failure" if paths != ("clean.spec.ts",) else "",
            "duration_ms": 1,
        }

    monkeypatch.setattr(units, "run_javascript_project_tests", fake_run)
    effective, run, receipt = units.resolve_javascript_test_baseline(
        tmp_path,
        {"command": [], "environment": {}, "timeout_seconds": 1},
        requested_paths=["clean.spec.ts", "faulty.spec.ts"],
    )
    assert effective == ["clean.spec.ts"]
    assert run["ok"] is True
    assert receipt["effective_test_paths"] == ["clean.spec.ts"]
    assert len(receipt["attempts"]) == 3


def test_javascript_verification_selection_is_content_bound_and_outcome_blind() -> None:
    holes = [
        {
            "path": f"src/value-{index}.ts",
            "qualified_name": f"value{index}",
            "start_byte": index * 10,
            "end_byte": index * 10 + 5,
            "target_body": f"return {index}",
        }
        for index in range(4)
    ]
    observed = units.javascript_verification_candidate_selection(
        {
            "id": "fixture",
            "verification_candidate_selection": {
                "kind": "canonical_ast_order_prefix_v1",
                "maximum_candidates": 2,
                "selection_uses_verifier_outcomes": False,
                "rationale": "bounded fixture",
            },
        },
        holes,
    )
    assert observed["selected_indexes"] == [0, 1]
    receipt = observed["public_receipt"]
    assert receipt["selected_candidate_count"] == 2
    assert receipt["unselected_candidate_count"] == 2
    assert receipt["selection_uses_verifier_outcomes"] is False
    assert len(receipt["ordered_inventory_sha256"]) == 64
    assert len(receipt["selected_inventory_sha256"]) == 64


def test_javascript_verification_selection_rejects_outcome_conditioning() -> None:
    with pytest.raises(ValueError, match="may not use verifier outcomes"):
        units.javascript_verification_candidate_selection(
            {
                "id": "fixture",
                "verification_candidate_selection": {
                    "kind": "canonical_ast_order_prefix_v1",
                    "maximum_candidates": 1,
                    "selection_uses_verifier_outcomes": True,
                },
            },
            [{
                "path": "src/value.ts",
                "qualified_name": "value",
                "start_byte": 0,
                "end_byte": 5,
                "target_body": "value",
            }],
        )


def test_tailwind_campaign_cap_does_not_bound_vite_or_python_sources() -> None:
    sources = {row["id"]: row for row in config()["sources"]}
    tailwind = sources["open_repo_tailwind_35a3e9c_test_killed_function_holes"]
    vite = sources["open_repo_vite_c961cae_test_killed_function_holes"]
    python_sources = [
        row for row in sources.values()
        if row.get("adapter") == "python_test_killed_function_holes"
    ]
    assert tailwind["verification_candidate_selection"] == {
        "kind": "canonical_ast_order_prefix_v1",
        "maximum_candidates": 600,
        "selection_uses_verifier_outcomes": False,
        "rationale": (
            "Bound correlated same-repository verification after cross-package "
            "coverage and the frozen JS/TS unit and position floors are satisfied."
        ),
    }
    assert "verification_candidate_selection" not in vite
    assert all("verification_candidate_selection" not in row for row in python_sources)


def test_rust_example_maps_code_and_manifest_without_dropping_starter(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / ".meta").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("todo!()\n")
    (tmp_path / "Cargo.toml").write_text("[package]\nname='fixture'\n")
    (tmp_path / ".meta" / "example.rs").write_text("pub fn answer() -> u8 { 42 }\n")
    observed = units.map_example_to_solution(
        tmp_path,
        ["src/lib.rs", "Cargo.toml"],
        [".meta/example.rs"],
    )
    assert observed["src/lib.rs"].startswith("pub fn answer")
    assert observed["Cargo.toml"].startswith("[package]")


def test_archive_traversal_and_links_fail_closed(tmp_path: Path) -> None:
    traversal = io.BytesIO()
    with tarfile.open(fileobj=traversal, mode="w") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("root/../../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    traversal.seek(0)
    with tarfile.open(fileobj=traversal, mode="r") as archive:
        try:
            units.safe_extract_source_archive(archive, tmp_path / "traversal")
        except ValueError as exc:
            assert "unsafe archive member" in str(exc)
        else:
            raise AssertionError("path traversal was extracted")

    linked = io.BytesIO()
    with tarfile.open(fileobj=linked, mode="w") as archive:
        member = tarfile.TarInfo("root/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/escape"
        archive.addfile(member)
    linked.seek(0)
    with tarfile.open(fileobj=linked, mode="r") as archive:
        try:
            units.safe_extract_source_archive(archive, tmp_path / "linked")
        except ValueError as exc:
            assert "unsafe archive link" in str(exc)
        else:
            raise AssertionError("archive link was extracted")

    internal = io.BytesIO()
    with tarfile.open(fileobj=internal, mode="w") as archive:
        payload = b"internal"
        target = tarfile.TarInfo("root/src/value.js")
        target.size = len(payload)
        archive.addfile(target, io.BytesIO(payload))
        member = tarfile.TarInfo("root/link.js")
        member.type = tarfile.SYMTYPE
        member.linkname = "./src/value.js"
        archive.addfile(member)
    internal.seek(0)
    with tarfile.open(fileobj=internal, mode="r") as archive:
        units.safe_extract_source_archive(archive, tmp_path / "internal")
    assert (tmp_path / "internal" / "link.js").read_bytes() == b"internal"


def test_missing_platform_sandbox_never_runs_unsandboxed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(units.platform, "system", lambda: "UnknownOS")
    result = units.run_sandboxed(["echo", "unsafe"], tmp_path, 1)
    assert result["ok"] is False
    assert result["fault"] == "fail_closed_sandbox_unavailable"


def test_canonical_admission_replays_task_unit_ledger_identity(tmp_path: Path) -> None:
    ledger = tmp_path / "units.jsonl.gz"
    ledger.write_bytes(b"content-bound-ledger")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "policy": "project_theseus_task_complete_training_units_v1",
        "contract_state": "GREEN",
        "coverage_state": "YELLOW",
        "summary": {
            "admitted_unit_count": 3,
            "task_complete_unique_target_positions": 100,
            "contract_hard_gap_count": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "fallback_return_count": 0,
        },
        "ledger_receipt": {
            "path": str(ledger),
            "sha256": admission.sha256_file(ledger),
            "replay_valid": True,
        },
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "fallback_return_count": 0,
        },
        "coverage": {},
    }))
    observed = admission.audit_task_complete_training_units(report)
    assert observed["contract_ready"] is True
    assert observed["coverage_ready"] is False
    ledger.write_bytes(b"tampered")
    assert admission.audit_task_complete_training_units(report)["contract_ready"] is False


def test_prior_contamination_cache_requires_ledger_hash_and_public_index(tmp_path: Path) -> None:
    ledger = tmp_path / "units.jsonl.gz"
    report = tmp_path / "report.json"
    row = {
        "unit_id": "unit-1",
        "public_benchmark_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "contamination": {"public_index_digest": "index-a"},
    }
    receipt = units.write_unit_ledger(ledger, [row])
    report.write_text(json.dumps({"ledger_receipt": receipt}))
    assert set(units.load_prior_unit_cache(ledger, report, contamination_digest="index-a")) == {"unit-1"}
    assert units.load_prior_unit_cache(ledger, report, contamination_digest="index-b") == {}
    with ledger.open("ab") as handle:
        handle.write(b"tamper")
    assert units.load_prior_unit_cache(ledger, report, contamination_digest="index-a") == {}


def test_python_function_holes_are_file_grouped_and_exclude_nested_functions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "fixture"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text(
        "def outer(value):\n"
        "    \"\"\"Return a transformed value.\"\"\"\n"
        "    def nested():\n"
        "        return value + 99\n"
        "    return value + 1\n\n"
        "class Worker:\n"
        "    def run(self, value):\n"
        "        return value * 2\n",
        encoding="utf-8",
    )
    holes = units.discover_python_function_holes(
        tmp_path,
        {
            "source_globs": ["src/**/*.py"],
            "minimum_body_bytes": 4,
            "maximum_body_bytes": 4096,
            "context_lines": 8,
        },
    )
    assert [row["qualified_name"] for row in holes] == ["outer", "Worker.run"]
    assert {row["path"] for row in holes} == {"src/fixture/module.py"}
    assert all(row["target_body"] not in row["visible_source"] for row in holes)
    for row in holes:
        text = source.read_text(encoding="utf-8").splitlines(keepends=True)
        starter = (
            "".join(text[: row["start_line"] - 1])
            + row["starter_body"]
            + "".join(text[row["end_line"] :])
        )
        compile(starter, row["path"], "exec")


def test_python_function_hole_requires_nonfaulting_mutation_kill(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "module.py"
    target = "def answer():\n    return 42\n"
    path.write_text(target, encoding="utf-8")
    hole = {
        "path": "module.py",
        "start_line": 2,
        "end_line": 2,
        "target_body": "    return 42\n",
        "starter_body": "    raise NotImplementedError('hole')\n",
    }
    toolchain = {"command": ["python", "-m", "pytest"], "environment": {}, "timeout_seconds": 5}
    monkeypatch.setattr(
        units,
        "run_python_project_tests",
        lambda *args, **kwargs: {
            "ok": False,
            "returncode": 1,
            "stdout": "test failed",
            "stderr": "",
            "duration_ms": 1,
        },
    )
    observed = units.verify_python_function_hole(
        tmp_path,
        hole,
        toolchain,
        baseline_run={"ok": True},
        baseline_receipt={"ok": True},
    )
    assert observed["state"] == "passed"
    assert path.read_text(encoding="utf-8") == target

    monkeypatch.setattr(
        units,
        "run_python_project_tests",
        lambda *args, **kwargs: {
            "ok": False,
            "returncode": None,
            "fault": "timeout",
            "stdout": "",
            "stderr": "",
            "duration_ms": 5000,
        },
    )
    faulted = units.verify_python_function_hole(
        tmp_path,
        hole,
        toolchain,
        baseline_run={"ok": True},
        baseline_receipt={"ok": True},
    )
    assert faulted["state"] == "failed"
    assert faulted["starter_failed"] is False


def test_write_cache_is_atomic_and_replayable(tmp_path: Path) -> None:
    path = tmp_path / "verification.jsonl"
    first = {
        "unit-a": {
            "unit_id": "unit-a",
            "verification_digest": "digest-a",
            "verification": {"state": "passed"},
        }
    }
    units.write_cache(path, first)
    assert units.load_cache(path) == first
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    replacement = {
        **first,
        "unit-b": {
            "unit_id": "unit-b",
            "verification_digest": "digest-b",
            "verification": {"state": "failed"},
        },
    }
    units.write_cache(path, replacement)
    assert units.load_cache(path) == replacement
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith('{"unit_id":"unit-a"')


def test_python_toolchain_expands_source_environment_and_binds_lock(
    tmp_path: Path,
) -> None:
    environment_python = tmp_path / ".venv" / "bin" / "python"
    environment_python.parent.mkdir(parents=True)
    environment_python.symlink_to(Path(units.sys.executable).resolve())
    locked = {
        "manager": "uv",
        "lock_sha256": "lock-digest",
        "identity_sha256": "environment-digest",
        "prepared_with_network_this_run": True,
        "uv_bootstrapped_with_network_this_run": True,
        "offline_replay": {"ok": True},
    }
    observed = units.resolve_python_project_toolchain(
        {
            "python_executable_candidates": ["{source_root}/.venv/bin/python"],
            "minimum_python_version": "3.9",
            "test_command": ["{python}", "-m", "pytest"],
            "environment": {"PYTHONPATH": "src"},
            "timeout_seconds": 30,
        },
        source_root=tmp_path,
        locked_environment=locked,
    )
    assert observed["command"][0] == str(environment_python.absolute())
    assert observed["locked_environment_identity"]["identity_sha256"] == "environment-digest"
    assert "prepared_with_network_this_run" not in observed["locked_environment_identity"]
    assert "uv_bootstrapped_with_network_this_run" not in observed["locked_environment_identity"]
    assert "offline_replay" not in observed["locked_environment_identity"]
