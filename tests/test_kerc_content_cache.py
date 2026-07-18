from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kerc_content_cache as cache  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_exact_receipt_replays_and_dependency_change_invalidates(tmp_path: Path) -> None:
    dependency = tmp_path / "source.txt"
    dependency.write_text("source-v1", encoding="utf-8")
    result = tmp_path / "report.json"
    write_json(result, {"trigger_state": "GREEN", "row_count": 2})
    dependencies = cache.dependency_bindings({"source": dependency})
    outputs = {"report": result}

    receipt = cache.publish_receipt(
        tmp_path / "cache",
        role="independent_verifier",
        dependencies=dependencies,
        outputs=outputs,
        result_output_id="report",
    )
    assert receipt.is_file()
    assert cache.load_receipt(
        tmp_path / "cache",
        role="independent_verifier",
        dependencies=dependencies,
        outputs=outputs,
        result_output_id="report",
    ) == {"trigger_state": "GREEN", "row_count": 2}

    dependency.write_text("source-v2", encoding="utf-8")
    changed = cache.dependency_bindings({"source": dependency})
    assert cache.load_receipt(
        tmp_path / "cache",
        role="independent_verifier",
        dependencies=changed,
        outputs=outputs,
        result_output_id="report",
    ) is None


def test_output_and_receipt_tampering_fail_closed(tmp_path: Path) -> None:
    dependency = tmp_path / "source.txt"
    dependency.write_text("source", encoding="utf-8")
    result = tmp_path / "report.json"
    write_json(result, {"trigger_state": "GREEN"})
    dependencies = cache.dependency_bindings({"source": dependency})
    outputs = {"report": result}
    receipt = cache.publish_receipt(
        tmp_path / "cache",
        role="producer",
        dependencies=dependencies,
        outputs=outputs,
        result_output_id="report",
    )

    write_json(result, {"trigger_state": "RED"})
    assert cache.load_receipt(
        tmp_path / "cache",
        role="producer",
        dependencies=dependencies,
        outputs=outputs,
        result_output_id="report",
    ) is None

    write_json(result, {"trigger_state": "GREEN"})
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["cache_key_sha256"] = "0" * 64
    write_json(receipt, payload)
    assert cache.load_receipt(
        tmp_path / "cache",
        role="producer",
        dependencies=dependencies,
        outputs=outputs,
        result_output_id="report",
    ) is None


def test_directory_binding_covers_relative_paths_and_bytes(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "a").mkdir(parents=True)
    (tree / "a" / "one.txt").write_text("one", encoding="utf-8")
    first = cache.hash_path(tree)
    (tree / "a" / "one.txt").write_text("two", encoding="utf-8")
    second = cache.hash_path(tree)
    assert first["kind"] == "directory_tree"
    assert first["file_count"] == 1
    assert first["sha256"] != second["sha256"]
