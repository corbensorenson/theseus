from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import task_complete_training_units as units  # noqa: E402
import training_admission_epistemic_tcb as admission_tcb  # noqa: E402
import task_complete_css_holes as css_holes  # noqa: E402
import task_complete_rust_holes as rust_holes  # noqa: E402
import task_complete_web_holes as web_holes  # noqa: E402
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
        arm.pop("required_capability_floors", None)
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


def test_html_css_coverage_requires_both_subcapabilities() -> None:
    cfg = config()
    floor = cfg["coverage_floors_for_50m_scale_proposal"]["html_css"]
    floor.update(minimum_verified_units=2, minimum_target_positions=20)
    floor["required_capability_floors"] = {
        "html": {
            "minimum_verified_units": 1,
            "minimum_target_positions": 10,
            "required_verification_strength": "dom_a11y_layout_render_delta",
        },
        "css": {
            "minimum_verified_units": 1,
            "minimum_target_positions": 10,
            "required_verification_strength": "layout_render_delta",
        },
    }
    html_only = [{
        "decision": "admit",
        "arm_id": "html_css",
        "capability_tags": ["html"],
        "target_positions": 20,
        "verification": {"strength": "dom_a11y_layout_render_delta"},
        "provenance": {},
    }, {
        "decision": "admit",
        "arm_id": "html_css",
        "capability_tags": ["html"],
        "target_positions": 20,
        "verification": {"strength": "dom_a11y_layout_render_delta"},
        "provenance": {},
    }]
    coverage, gaps = units.coverage_summary(cfg, html_only)
    assert coverage["html_css"]["ready"] is False
    assert coverage["html_css"]["capabilities"]["html"]["ready"] is True
    assert coverage["html_css"]["capabilities"]["css"]["ready"] is False
    assert "capability:css:unit_floor" in next(
        row["failed_checks"] for row in gaps if row["arm_id"] == "html_css"
    )

    wrong_strength_css = [*html_only, {
        "decision": "admit",
        "arm_id": "html_css",
        "capability_tags": ["css"],
        "target_positions": 10,
        "verification": {"strength": "dom_a11y_layout_render_delta"},
        "provenance": {},
    }]
    floor["minimum_verified_units"] = 3
    coverage, _ = units.coverage_summary(cfg, wrong_strength_css)
    assert coverage["html_css"]["capabilities"]["css"]["verified_units"] == 0
    assert coverage["html_css"]["ready"] is False

    html_and_css = [*html_only, {
        "decision": "admit",
        "arm_id": "html_css",
        "capability_tags": ["css"],
        "target_positions": 10,
        "verification": {"strength": "layout_render_delta"},
        "provenance": {},
    }]
    coverage, _ = units.coverage_summary(cfg, html_and_css)
    assert coverage["html_css"]["ready"] is True


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


def test_javascript_package_inventory_identity_is_order_independent(tmp_path: Path) -> None:
    first = json.dumps([
        {"path": str(tmp_path / "b"), "dependencies": {"z": {"version": "2"}}},
        {"path": str(tmp_path / "a"), "dependencies": {"a": {"version": "1"}}},
    ])
    second = json.dumps([
        {"dependencies": {"a": {"version": "1"}}, "path": str(tmp_path / "a")},
        {"dependencies": {"z": {"version": "2"}}, "path": str(tmp_path / "b")},
    ])
    assert units.canonical_javascript_package_inventory(
        first, source_root=tmp_path
    ) == units.canonical_javascript_package_inventory(second, source_root=tmp_path)


def test_javascript_cache_migration_allows_only_inventory_identity_drift() -> None:
    basis = {
        "unit_id": "unit-1",
        "archive_sha256": "archive",
        "target_body": "return 42",
        "starter_body": 'throw new Error("hole")',
        "test_paths": ["value.test.ts"],
    }
    prior_toolchain = {
        "command": ["vitest", "run"],
        "environment": {},
        "timeout_seconds": 30,
        "parser_toolchain_identity": {"identity_sha256": "parser"},
        "locked_environment_identity": {
            "identity_sha256": "prior-derived",
            "installed_packages_sha256": "prior-raw-order",
            "lock_sha256": "lock",
            "pnpm_version": "11.9.0",
            "build_outputs": {"tree_sha256": "build"},
        },
    }
    current_toolchain = json.loads(json.dumps(prior_toolchain))
    current_toolchain["locked_environment_identity"].update({
        "identity_sha256": "current-derived",
        "installed_packages_sha256": "canonical-order",
        "installed_packages_identity_kind": "canonical_unordered_json_v1",
    })
    cached = {
        "verification_digest": units.stable_hash({**basis, "toolchain": prior_toolchain}),
        "verification": {
            "kind": "javascript_typescript_repository_test_killed_function_hole_v1",
            "strength": "executable_target_pass_starter_fail",
            "state": "passed",
            "target_passed": True,
            "starter_failed": True,
            "toolchain": prior_toolchain,
        },
    }
    assert units.javascript_cached_verification_compatible(
        cached, verification_basis=basis, current_toolchain=current_toolchain
    )
    rebound = units.javascript_rebind_cached_verification(
        cached["verification"], current_toolchain=current_toolchain
    )
    assert rebound["toolchain"] == current_toolchain
    assert rebound["cache_revalidation"]["tests_reexecuted"] is False

    changed_command = json.loads(json.dumps(current_toolchain))
    changed_command["command"].append("--changed")
    assert not units.javascript_cached_verification_compatible(
        cached, verification_basis=basis, current_toolchain=changed_command
    )
    assert not units.javascript_cached_verification_compatible(
        cached,
        verification_basis={**basis, "starter_body": "different mutation"},
        current_toolchain=current_toolchain,
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

    directory = io.BytesIO()
    with tarfile.open(fileobj=directory, mode="w") as archive:
        root = tarfile.TarInfo("root/shared")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        payload = b"asset"
        target = tarfile.TarInfo("root/shared/value.css")
        target.size = len(payload)
        archive.addfile(target, io.BytesIO(payload))
        member = tarfile.TarInfo("root/public")
        member.type = tarfile.SYMTYPE
        member.linkname = "./shared"
        archive.addfile(member)
    directory.seek(0)
    with tarfile.open(fileobj=directory, mode="r") as archive:
        units.safe_extract_source_archive(archive, tmp_path / "directory")
    assert (tmp_path / "directory" / "public" / "value.css").read_bytes() == b"asset"


def test_html_subtree_discovery_binds_exact_source_span(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "index.html"
    page.parent.mkdir(parents=True)
    original = (
        "<!doctype html><html><body>"
        "<main><section aria-label='Account'><h1>Account</h1>"
        "<p>Manage your profile and notification preferences.</p></section></main>"
        "</body></html>"
    )
    page.write_text(original)
    source = {
        "source_globs": ["pages/*.html"],
        "minimum_target_bytes": 32,
        "maximum_target_bytes": 4096,
        "minimum_visible_text_chars": 8,
        "minimum_descendant_tag_count": 1,
    }
    holes, receipt = web_holes.discover_html_subtree_holes(tmp_path, source)
    section = next(row for row in holes if row["tag"] == "section")
    assert original[section["start_char"]:section["end_char"]] == section["target_body"]
    assert section["target_body"] not in section["visible_source"]
    assert section["visible_source"].count("<THESEUS_IMPLEMENTATION_HOLE:") == 1
    assert web_holes.reconstruct_document(
        section["visible_source"],
        {
            "start_char": section["start_char"],
            "end_char": section["start_char"] + len(section["visible_starter_body"]),
        },
        section["target_body"],
    ) == original
    assert receipt["candidate_count"] >= 1


def test_html_discovery_excludes_statically_non_rendered_subtrees(tmp_path: Path) -> None:
    page = tmp_path / "index.html"
    page.write_text(
        "<!doctype html><html><body>"
        "<template><section><h1>Template only</h1><p>Never painted content.</p></section></template>"
        "<section hidden><h1>Hidden</h1><p>Also never painted.</p></section>"
        "<section><h1>Visible</h1><p>This component is rendered for the user.</p></section>"
        "</body></html>"
    )
    holes, receipt = web_holes.discover_html_subtree_holes(
        tmp_path,
        {
            "source_globs": ["*.html"],
            "minimum_target_bytes": 16,
            "maximum_target_bytes": 4096,
            "minimum_visible_text_chars": 4,
            "minimum_descendant_tag_count": 1,
        },
    )
    sections = [row for row in holes if row["tag"] == "section"]
    assert len(sections) == 1
    assert "Visible" in sections[0]["target_body"]
    assert receipt["diagnostics"]["statically_non_rendered_rejected"] == 2


def test_html_selection_is_file_stratified_and_outcome_blind() -> None:
    holes = []
    for path in ("a.html", "b.html", "c.html"):
        for offset in range(3):
            holes.append({
                "path": path,
                "start_char": offset * 10,
                "end_char": offset * 10 + 5,
                "tag": "section",
                "source_sha256": f"source-{path}",
                "target_sha256": f"target-{path}-{offset}",
                "selection_key": f"{offset}-{path}",
            })
    source = {
        "id": "fixture",
        "verification_candidate_selection": {
            "kind": "content_hash_file_round_robin_v1",
            "maximum_candidates": 4,
            "maximum_candidates_per_file": 2,
            "selection_uses_verifier_outcomes": False,
            "rationale": "fixture",
        },
    }
    first = web_holes.select_verification_candidates(source, holes)
    second = web_holes.select_verification_candidates(source, list(reversed(holes)))
    first_records = [web_holes.selection_record(row) for row in first["selected"]]
    second_records = [web_holes.selection_record(row) for row in second["selected"]]
    assert first_records == second_records
    assert len({row["path"] for row in first["selected"][:3]}) == 3
    assert first["public_receipt"]["selection_uses_verifier_outcomes"] is False
    with pytest.raises(ValueError, match="may not use verifier outcomes"):
        web_holes.select_verification_candidates(
            {
                **source,
                "verification_candidate_selection": {
                    **source["verification_candidate_selection"],
                    "selection_uses_verifier_outcomes": True,
                },
            },
            holes,
        )


def test_css_discovery_uses_ast_spans_and_source_only_page_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    css_path = tmp_path / "styles.css"
    page_path = tmp_path / "index.html"
    css_text = ".card { color: red; padding: 1rem; }\n"
    css_path.write_text(css_text, encoding="utf-8")
    (tmp_path / "styles.min.css").write_text(".card{color:red}", encoding="utf-8")
    page_path.write_text(
        '<html><head><link rel="stylesheet" href="styles.css"></head>'
        '<body><article class="card">Visible card</article></body></html>',
        encoding="utf-8",
    )
    observed_paths: list[Path] = []

    def parse_fixture(paths, toolchain):
        observed_paths.extend(paths)
        return ([{
            "path": str(css_path.resolve()),
            "start_byte": 0,
            "end_byte": len(css_text.strip().encode()),
            "start_char": 0,
            "end_char": len(css_text.strip()),
            "selector": ".card",
            "declaration_count": 2,
            "ancestor_kinds": [],
            "target_body": css_text.strip(),
        }], {"parsed_file_count": 1, "parse_error_count": 0})

    monkeypatch.setattr(
        css_holes,
        "parse_css_records",
        parse_fixture,
    )
    source = {
        "id": "css-fixture",
        "source_globs": ["*.css"],
        "exclude_source_globs": ["**/*.min.css", "*.min.css"],
        "page_globs": ["*.html"],
        "minimum_target_bytes": 10,
        "maximum_target_bytes": 1000,
        "verification_candidate_selection": {
            "kind": "content_hash_file_round_robin_v1",
            "maximum_candidates": 1,
            "maximum_candidates_per_file": 1,
            "selection_uses_verifier_outcomes": False,
        },
    }
    holes, receipt = css_holes.discover_css_rule_holes(tmp_path.resolve(), source, {})
    assert receipt["candidate_count"] == 1
    assert observed_paths == [css_path.resolve()]
    assert holes[0]["target_body"] == css_text.strip()
    assert holes[0]["page_path"] == "index.html"
    assert holes[0]["target_body"] not in holes[0]["stylesheet_context"]
    selection = css_holes.select_verification_candidates(source, holes)
    assert selection["public_receipt"]["selection_uses_verifier_outcomes"] is False
    with pytest.raises(ValueError, match="may not use verifier outcomes"):
        css_holes.select_verification_candidates(
            {**source, "verification_candidate_selection": {
                **source["verification_candidate_selection"],
                "selection_uses_verifier_outcomes": True,
            }},
            holes,
        )


def test_css_link_rewrite_is_exact_and_preserves_other_stylesheets(tmp_path: Path) -> None:
    page_path = tmp_path / "index.html"
    css_path = tmp_path / ".theseus-css-rule-fixture.css"
    page = (
        '<link rel="stylesheet" href="base.css">\n'
        '<link href="styles.css" rel="stylesheet" />\n'
    )
    rewritten = css_holes.replace_stylesheet_link(page, page_path, css_path, ["styles.css"])
    assert 'href="base.css"' in rewritten
    assert 'href=".theseus-css-rule-fixture.css"' in rewritten
    assert 'href="styles.css"' not in rewritten


def test_css_verifier_accepts_one_responsive_viewport_and_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    css_path = tmp_path / "styles.css"
    page_path = tmp_path / "index.html"
    css_text = ".card { display: grid; gap: 1rem; }"
    page_text = '<link rel="stylesheet" href="styles.css"><div class="card">Card</div>'
    css_path.write_text(css_text, encoding="utf-8")
    page_path.write_text(page_text, encoding="utf-8")
    monkeypatch.setattr(
        css_holes,
        "render_css_pair",
        lambda **kwargs: {
            "fault": None,
            "viewports": [
                {
                    "label": "wide",
                    "target": {"ok": True},
                    "starter": {"ok": True},
                    "pixel_delta": {"changed_pixel_fraction": 0.002},
                },
                {
                    "label": "narrow",
                    "target": {"ok": True},
                    "starter": {"ok": True},
                    "pixel_delta": {"changed_pixel_fraction": 0.0},
                },
            ],
        },
    )
    hole = {
        "path": "styles.css",
        "page_path": "index.html",
        "stylesheet_aliases": ["styles.css"],
        "start_char": 0,
        "end_char": len(css_text),
        "selector": ".card",
        "declaration_count": 2,
        "source_sha256": css_holes.sha256_text(css_text),
        "page_sha256": css_holes.sha256_text(page_text),
        "target_sha256": css_holes.sha256_text(css_text),
        "target_body": css_text,
    }
    receipt = css_holes.verify_css_rule_hole(
        tmp_path.resolve(), hole, {"timeout_seconds": 5, "minimum_changed_pixel_fraction": 0.0005}
    )
    assert receipt["state"] == "passed"
    assert receipt["changed_viewports"] == ["wide"]
    assert receipt["all_viewports_rendered"] is True
    assert receipt["canonical_sources_unchanged"] is True
    assert css_path.read_text() == css_text
    assert page_path.read_text() == page_text


def test_css_verifier_rejects_no_render_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    css_path = tmp_path / "styles.css"
    page_path = tmp_path / "index.html"
    css_text = ".unused { color: red; }"
    page_text = '<link rel="stylesheet" href="styles.css"><p>Text</p>'
    css_path.write_text(css_text)
    page_path.write_text(page_text)
    monkeypatch.setattr(
        css_holes,
        "render_css_pair",
        lambda **kwargs: {
            "fault": None,
            "viewports": [
                {"label": label, "target": {"ok": True}, "starter": {"ok": True},
                 "pixel_delta": {"changed_pixel_fraction": 0.0}}
                for _, _, label in css_holes.VIEWPORTS
            ],
        },
    )
    hole = {
        "path": "styles.css", "page_path": "index.html", "stylesheet_aliases": ["styles.css"],
        "start_char": 0, "end_char": len(css_text), "selector": ".unused", "declaration_count": 1,
        "source_sha256": css_holes.sha256_text(css_text), "page_sha256": css_holes.sha256_text(page_text),
        "target_sha256": css_holes.sha256_text(css_text), "target_body": css_text,
    }
    receipt = css_holes.verify_css_rule_hole(
        tmp_path.resolve(), hole, {"timeout_seconds": 5, "minimum_changed_pixel_fraction": 0.0005}
    )
    assert receipt["state"] == "failed"
    assert receipt["changed_viewport_count"] == 0


def test_css_campaigns_are_source_only_and_exclude_minified_targets() -> None:
    sources = [
        row for row in config()["sources"]
        if row.get("adapter") == "css_render_killed_rule_holes"
    ]
    assert {row["id"] for row in sources} == {
        "open_repo_sb_admin_f030988_render_killed_css_rules",
        "open_repo_agency_b2d5d5c_render_killed_css_rules",
        "open_repo_semantic_ui_597843a_render_killed_css_rules",
        "open_repo_bootstrap_b37afd7_render_killed_css_rules",
    }
    assert all(
        row["verification_candidate_selection"]["selection_uses_verifier_outcomes"] is False
        for row in sources
    )
    semantic = next(row for row in sources if "semantic_ui" in row["id"])
    assert semantic["exclude_source_globs"] == ["**/*.min.css"]
    assert sum(
        row["verification_candidate_selection"]["maximum_candidates"] for row in sources
    ) == 1000


def test_all_html_campaigns_are_source_only_and_parallelize_by_file() -> None:
    sources = [
        row for row in config()["sources"]
        if row.get("adapter") == "html_render_killed_subtree_holes"
    ]
    assert len(sources) >= 6
    assert all(
        row["verification_candidate_selection"]["selection_uses_verifier_outcomes"] is False
        for row in sources
    )
    assert all(int(row["verification_parallelism"]) == 2 for row in sources)
    assert all(int(row["verification_checkpoint_interval"]) == 25 for row in sources)


def test_html_cache_reuse_requires_bound_verifier_and_toolchain() -> None:
    toolchain = {"verifier_abi": web_holes.VERIFIER_ABI, "chrome_version": "fixture"}
    cached = {
        "verification_digest": "legacy-campaign-bound-digest",
        "verification": {
            "kind": web_holes.VERIFIER_ABI,
            "state": "passed",
            "toolchain": toolchain,
        },
    }
    assert units.html_cached_verification_compatible(
        cached, verification_digest="candidate-only-digest", toolchain_identity=toolchain
    )
    cached["verification"]["toolchain"] = {"chrome_version": "changed"}
    assert not units.html_cached_verification_compatible(
        cached, verification_digest="candidate-only-digest", toolchain_identity=toolchain
    )


def test_html_adapter_parallelizes_distinct_files_and_admits_only_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    for name in ("a.html", "b.html"):
        (tmp_path / name).write_text(
            "<!doctype html><html><body><section><h1>Profile</h1>"
            f"<p>Complete account panel {name} with durable details.</p>"
            "</section></body></html>"
        )
    source = {
        "id": "html-fixture",
        "adapter": "html_render_killed_subtree_holes",
        "repo": "fixture/repo",
        "revision": "abc1234",
        "archive_sha256": "a" * 64,
        "license_spdx": "MIT",
        "source_globs": ["*.html"],
        "minimum_target_bytes": 16,
        "maximum_target_bytes": 4096,
        "minimum_visible_text_chars": 4,
        "minimum_descendant_tag_count": 1,
        "verification_parallelism": 2,
        "verification_checkpoint_interval": 1,
        "verification_candidate_selection": {
            "kind": "content_hash_file_round_robin_v1",
            "maximum_candidates": 2,
            "maximum_candidates_per_file": 1,
            "selection_uses_verifier_outcomes": False,
            "rationale": "fixture",
        },
    }
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_verify(_root, _hole, toolchain):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {
            "kind": web_holes.VERIFIER_ABI,
            "strength": "dom_a11y_layout_render_delta",
            "state": "passed",
            "toolchain": web_holes.web_toolchain_identity(toolchain),
        }

    monkeypatch.setattr(units, "ensure_source_root", lambda _source: tmp_path)
    monkeypatch.setattr(web_holes, "verify_html_subtree_hole", fake_verify)
    observed, summary = units.html_subtree_hole_units(
        config(),
        source,
        clean_contamination(),
        prior_units={},
        cache={},
        cache_updates={},
        cache_path=tmp_path / "cache.jsonl",
        inventory_only=False,
        max_verify=0,
    )
    assert len(observed) == 2
    assert all(row["decision"] == "admit" for row in observed)
    assert maximum_active == 2
    assert summary["completed_verification_count"] == 2
    assert summary["verification_cache_checkpoint_write_count"] >= 1


def test_code_unit_reuses_only_identity_and_public_index_bound_contamination(
    monkeypatch,
) -> None:
    cfg = config()
    contamination = clean_contamination()
    baseline = units.base_unit(
        cfg,
        source={"id": "fixture"},
        source_task_id="task",
        arm_id="rust",
        task_family="fixture",
        visible_context="Implement the missing checked operation.",
        target="return Ok(42);",
        license_spdx="MIT",
        provenance={},
        contamination=contamination,
    )
    monkeypatch.setattr(
        lineage,
        "semantic_overlap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recomputed")),
    )
    replay = units.base_unit(
        cfg,
        source={"id": "fixture"},
        source_task_id="task",
        arm_id="rust",
        task_family="fixture",
        visible_context="Implement the missing checked operation.",
        target="return Ok(42);",
        license_spdx="MIT",
        provenance={},
        contamination=contamination,
        prior_units={baseline["unit_id"]: baseline},
    )
    assert replay["contamination"]["reused_prior_audit"] is True
    altered_index = {**contamination, "digest": "changed-index"}
    with pytest.raises(AssertionError, match="recomputed"):
        units.base_unit(
            cfg,
            source={"id": "fixture"},
            source_task_id="task",
            arm_id="rust",
            task_family="fixture",
            visible_context="Implement the missing checked operation.",
            target="return Ok(42);",
            license_spdx="MIT",
            provenance={},
            contamination=altered_index,
            prior_units={baseline["unit_id"]: baseline},
        )


def test_html_render_verifier_requires_both_viewports_and_restores_source(
    tmp_path: Path, monkeypatch
) -> None:
    page = tmp_path / "index.html"
    original = (
        "<!doctype html><html><body><section><h1>Profile</h1>"
        "<p>Update your account details.</p></section></body></html>"
    )
    page.write_text(original)
    source = {
        "source_globs": ["*.html"],
        "minimum_target_bytes": 16,
        "maximum_target_bytes": 4096,
        "minimum_visible_text_chars": 4,
        "minimum_descendant_tag_count": 1,
    }
    holes, _ = web_holes.discover_html_subtree_holes(tmp_path, source)
    hole = next(row for row in holes if row["tag"] == "section")

    def fake_render(_root, source_path, screenshot, _user_data, width, height, *_args):
        has_target = "Update your account details" in source_path.read_text()
        from PIL import Image
        Image.new("RGB", (width, height), "black" if has_target else "white").save(screenshot)
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "duration_ms": 1}

    monkeypatch.setattr(web_holes, "run_chrome_render", fake_render)
    verifier = web_holes.verify_html_subtree_hole(
        tmp_path,
        hole,
        {"timeout_seconds": 2, "minimum_changed_pixel_fraction": 0.01},
    )
    assert verifier["state"] == "passed"
    assert verifier["both_viewports_changed"] is True
    assert verifier["source_restored"] is True
    assert page.read_text() == original

    def unchanged_render(_root, _source_path, screenshot, _user_data, width, height, *_args):
        from PIL import Image
        Image.new("RGB", (width, height), "white").save(screenshot)
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "duration_ms": 1}

    monkeypatch.setattr(web_holes, "run_chrome_render", unchanged_render)
    verifier = web_holes.verify_html_subtree_hole(
        tmp_path,
        hole,
        {"timeout_seconds": 2, "minimum_changed_pixel_fraction": 0.01},
    )
    assert verifier["state"] == "failed"
    assert verifier["both_viewports_changed"] is False
    assert page.read_text() == original


def test_missing_platform_sandbox_never_runs_unsandboxed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(units.platform, "system", lambda: "UnknownOS")
    result = units.run_sandboxed(["echo", "unsafe"], tmp_path, 1)
    assert result["ok"] is False
    assert result["fault"] == "fail_closed_sandbox_unavailable"


def test_sandbox_output_is_tail_bounded_unless_full_capture_is_explicit(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('x' * 5000)"]
    bounded = units.run_sandboxed(command, tmp_path, 5)
    full = units.run_sandboxed(command, tmp_path, 5, output_tail_characters=None)
    assert bounded["ok"] is True
    assert len(bounded["stdout"]) == 4000
    assert full["ok"] is True
    assert len(full["stdout"]) == 5001


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


def test_independent_admission_oracle_rejects_forged_executable_credit() -> None:
    visible = '{"prompt":"Implement answer"}'
    target = '{"replacement":"42"}'
    row = {
        "policy": admission_tcb.UNIT_POLICY,
        "unit_id": "unit-1",
        "source_id": "fixture",
        "source_task_id": "fixture:one",
        "arm_id": "rust",
        "task_family": "repository_rust_function_body_hole",
        "visible_context": visible,
        "visible_context_sha256": admission_tcb.sha256_text(visible),
        "target": target,
        "target_sha256": admission_tcb.sha256_text(target),
        "split": "train",
        "license_spdx": "MIT",
        "provenance": {"live_teacher_call": False},
        "contamination": {
            "exact_overlap": False,
            "semantic_overlap": False,
            "quarantine": False,
        },
        "verification": {
            "kind": "project_theseus_rust_test_killed_function_body_v3",
            "state": "passed",
            "strength": admission_tcb.EXECUTABLE_STRENGTH,
            "target_passed": True,
            "starter_test_failed": True,
            "source_restored": True,
            "baseline_run": {"ok": True},
            "checkpoint_baseline_run": {"ok": True},
        },
        "task_complete_verified": True,
        "decision": "admit",
        "decision_reasons": [],
        "public_benchmark_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    assert admission_tcb.independent_unit_faults(row) == []
    forged = json.loads(json.dumps(row))
    forged["verification"]["starter_test_failed"] = False
    assert "executable_starter_not_failed" in admission_tcb.independent_unit_faults(forged)


def test_canonical_admission_requires_current_epistemic_tcb_identity(tmp_path: Path) -> None:
    task_report = tmp_path / "task-report.json"
    ledger = tmp_path / "units.jsonl.gz"
    tcb = tmp_path / "tcb.json"
    task_report.write_text('{"policy":"task"}')
    ledger.write_bytes(b"ledger")
    tcb_payload = {
        "policy": admission_tcb.POLICY,
        "trigger_state": "GREEN",
        "hard_gaps": [],
        "summary": {
            "mutation_count": 17,
            "surviving_mutant_count": 0,
            "correlated_dependency_count": 3,
            "position_budgets_reported_separately": True,
        },
        "input_artifacts": {
            "task_report": {
                "path": admission.rel(task_report),
                "sha256": admission.sha256_file(task_report),
            },
            "unit_ledger": {
                "path": str(ledger),
                "sha256": admission.sha256_file(ledger),
            },
        },
    }
    tcb.write_text(json.dumps(tcb_payload))
    observed = admission.audit_training_admission_epistemic_tcb(
        tcb,
        task_complete_path=task_report,
        task_complete={"ledger_identity_valid": True},
    )
    assert observed["qualified"] is True
    task_report.write_text('{"policy":"tampered"}')
    assert admission.audit_training_admission_epistemic_tcb(
        tcb,
        task_complete_path=task_report,
        task_complete={"ledger_identity_valid": True},
    )["qualified"] is False


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


def test_rust_body_parser_ignores_literals_comments_and_lifetimes() -> None:
    source = r'''fn render<'a>(value: &'a str) -> String
where
    'a: 'static,
{
    let char_brace = '{';
    let raw = r###"not a body } or {"###;
    /* nested { comment /* } */ still comment } */
    format!("{{{value}}}: {raw}: {char_brace}")
}
'''
    start, end = rust_holes.find_function_body(source, 0, len(source))
    body = source[start:end]
    assert body.startswith("{\n")
    assert body.endswith("\n}")
    assert "format!" in body


def test_rust_selection_is_source_only_deterministic_and_capped() -> None:
    source = {
        "id": "rust-fixture",
        "verification_candidate_selection": {
            "kind": "content_hash_file_round_robin_v1",
            "maximum_candidates": 3,
            "maximum_candidates_per_file": 2,
            "maximum_candidates_per_package": 2,
            "selection_uses_verifier_outcomes": False,
        },
    }
    holes = []
    for index, path in enumerate(("src/a.rs", "src/a.rs", "src/a.rs", "src/b.rs")):
        holes.append({
            "candidate_id": f"candidate-{index}",
            "path": path,
            "package": "fixture" if index < 3 else "other",
            "function_name": f"function_{index}",
            "body_start_char": index * 10,
            "body_end_char": index * 10 + 8,
            "target_sha256": f"sha-{index}",
            "target_bytes": 8,
        })
    first = rust_holes.select_verification_candidates(source, holes)
    second = rust_holes.select_verification_candidates(source, list(reversed(holes)))
    assert first["public_receipt"] == second["public_receipt"]
    assert [row["candidate_id"] for row in first["selected"]] == [
        row["candidate_id"] for row in second["selected"]
    ]
    assert first["public_receipt"]["selected_count"] == 3
    assert first["public_receipt"]["selection_uses_verifier_outcomes"] is False


def test_rust_verifier_requires_compile_pass_test_kill_and_restore(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    source_path = source_root / "src" / "lib.rs"
    source_path.parent.mkdir(parents=True)
    source_text = "pub fn answer() -> u8 {\n    42\n}\n"
    source_path.write_text(source_text)
    body_start = source_text.index("{")
    body_end = source_text.rindex("}") + 1
    hole = {
        "candidate_id": "candidate-a",
        "path": "src/lib.rs",
        "package": "fixture",
        "verification_root_package": "fixture_integration",
        "body_start_char": body_start,
        "body_end_char": body_end,
        "source_sha256": rust_holes.sha256_text(source_text),
        "target_sha256": rust_holes.sha256_text(source_text[body_start:body_end]),
    }
    outcomes = iter((
        {"ok": True, "returncode": 0, "timed_out": False},
        {"ok": False, "returncode": 101, "timed_out": False},
        {"ok": True, "returncode": 0, "timed_out": False},
        {"ok": True, "returncode": 0, "timed_out": False},
    ))

    commands = []

    environments = []

    def fake_run(
        command, workdir, timeout_seconds, *, target_dir=None, cargo_home=None, environment=None
    ):
        commands.append(command)
        environments.append(environment)
        result = next(outcomes)
        return {
            **result,
            "command": command,
            "duration_ms": 1,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(rust_holes, "run_command", fake_run)
    results, package = rust_holes._verify_package_group(
        source_root,
        "fixture",
        [hole],
        {
            "cargo_path": "cargo",
            "verifier_abi": rust_holes.VERIFIER_ABI,
            "locked_environment": {"cargo_home_path": str(tmp_path / "cargo-home")},
        },
        {"work_root": str(tmp_path / "work"), "test_timeout_seconds": 5},
    )
    assert results["candidate-a"]["state"] == "passed"
    assert results["candidate-a"]["starter_compiled"] is True
    assert results["candidate-a"]["starter_test_failed"] is True
    assert results["candidate-a"]["source_restored"] is True
    assert package["final_baseline"]["ok"] is True
    assert package["verification_packages"] == ["fixture", "fixture_integration"]
    assert results["candidate-a"]["starter_compile_evidence"] == "cargo_check_after_test_failure"
    assert all(
        command.count("-p") == 2 and "fixture_integration" in command
        for command in commands
    )
    assert all(environment and environment["TMPDIR"] for environment in environments)


def test_rust_verifier_uses_passing_tests_as_compile_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    source_path = source_root / "src" / "lib.rs"
    source_path.parent.mkdir(parents=True)
    source_text = "pub fn unobserved() -> u8 {\n    42\n}\n"
    source_path.write_text(source_text)
    body_start = source_text.index("{")
    body_end = source_text.rindex("}") + 1
    hole = {
        "candidate_id": "candidate-missed",
        "path": "src/lib.rs",
        "package": "fixture",
        "verification_root_package": "fixture",
        "body_start_char": body_start,
        "body_end_char": body_end,
        "source_sha256": rust_holes.sha256_text(source_text),
        "target_sha256": rust_holes.sha256_text(source_text[body_start:body_end]),
    }
    outcomes = iter((
        {"ok": True, "returncode": 0, "timed_out": False},
        {"ok": True, "returncode": 0, "timed_out": False},
        {"ok": True, "returncode": 0, "timed_out": False},
    ))
    commands = []

    def fake_run(
        command, workdir, timeout_seconds, *, target_dir=None, cargo_home=None, environment=None
    ):
        commands.append(command)
        result = next(outcomes)
        return {
            **result,
            "command": command,
            "duration_ms": 1,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(rust_holes, "run_command", fake_run)
    results, _ = rust_holes._verify_package_group(
        source_root,
        "fixture",
        [hole],
        {
            "cargo_path": "cargo",
            "verifier_abi": rust_holes.VERIFIER_ABI,
            "locked_environment": {"cargo_home_path": str(tmp_path / "cargo-home")},
        },
        {"work_root": str(tmp_path / "work"), "test_timeout_seconds": 5},
    )
    result = results["candidate-missed"]
    assert result["state"] == "failed"
    assert result["reason"] == "starter_tests_passed"
    assert result["starter_compiled"] is True
    assert result["starter_compile_evidence"] == "cargo_test_completed"
    assert len(commands) == 3
    assert all(command[1] == "test" for command in commands)


def test_rust_timeout_terminates_the_entire_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    result = rust_holes.run_command(
        [
            "/bin/sh",
            "-c",
            "(trap '' TERM; while :; do sleep 30; done) & "
            "child=$!; echo $child > child.pid; wait",
        ],
        tmp_path,
        1,
    )
    assert result["timed_out"] is True
    assert result["timeout_termination"] == "process_group_sigkill_after_grace"
    child_pid = int(child_pid_path.read_text().strip())
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"timed-out child process {child_pid} survived")


def test_rust_timeout_taxonomy_prefers_the_test_phase() -> None:
    assert rust_holes._failure_reason(
        {"ok": False, "timed_out": True},
        {"ok": False, "timed_out": True},
        True,
    ) == "starter_test_timeout"


def test_rust_worktree_cleanup_retries_transient_directory_race(
    tmp_path: Path, monkeypatch
) -> None:
    real_rmtree = rust_holes.shutil.rmtree
    attempts = 0

    def transient_rmtree(path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(66, "Directory not empty", "debug")
        real_rmtree(path)

    monkeypatch.setattr(rust_holes.shutil, "rmtree", transient_rmtree)
    with rust_holes.resilient_temporary_directory(
        prefix="rust-cleanup-", directory=tmp_path
    ) as raw:
        worktree = Path(raw)
        (worktree / "debug").mkdir()
        (worktree / "debug" / "artifact").write_text("complete")

    assert attempts == 2
    assert not worktree.exists()


def test_verification_cache_compaction_only_prunes_after_complete_campaign() -> None:
    cache = {
        "live": {"unit_id": "live"},
        "stale": {"unit_id": "stale"},
    }
    unit_rows = [{"unit_id": "live"}]

    compacted, pruned = units.compact_cache_to_units(
        cache, unit_rows, allow_prune=True
    )
    preserved, preserved_pruned = units.compact_cache_to_units(
        cache, unit_rows, allow_prune=False
    )

    assert compacted == {"live": {"unit_id": "live"}}
    assert pruned == 1
    assert preserved == cache
    assert preserved_pruned == 0
