from __future__ import annotations

import json
import sqlite3
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


def test_object_cache_is_canonical_namespaced_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "objects.sqlite3"
    key = cache.object_key(
        role="producer",
        layer="economics",
        dependencies={"record": "sha256:abc", "lambda": 64},
    )
    with cache.ContentObjectCache(path, namespace="producer:economics") as store:
        store.put(key, {"z": [2, 1], "a": True})
        assert store.get(key) == {"a": True, "z": [2, 1]}
        assert store.count() == 1

    with cache.ContentObjectCache(path, namespace="verifier:semantic") as other:
        assert other.get(key) is None
        assert other.count() == 0

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE objects SET payload_json = ? WHERE namespace = ? AND object_key = ?",
        (b'{"forged":true}', "producer:economics", key),
    )
    connection.commit()
    connection.close()
    with cache.ContentObjectCache(path, namespace="producer:economics") as store:
        assert store.get(key) is None
        assert store.count() == 0


def test_object_cache_recomputes_only_changed_dependency(tmp_path: Path) -> None:
    path = tmp_path / "objects.sqlite3"
    original_dependencies = [
        {"source_id": "a", "importance": 1.0},
        {"source_id": "b", "importance": 2.0},
        {"source_id": "c", "importance": 3.0},
    ]
    with cache.ContentObjectCache(path, namespace="producer:economics") as store:
        for dependencies in original_dependencies:
            key = cache.object_key(
                role="producer", layer="economics", dependencies=dependencies
            )
            store.put(key, {"allocation": dependencies["importance"]})

    changed_dependencies = [dict(value) for value in original_dependencies]
    changed_dependencies[1]["importance"] = 2.5
    hits = 0
    misses = 0
    with cache.ContentObjectCache(path, namespace="producer:economics") as store:
        for dependencies in changed_dependencies:
            key = cache.object_key(
                role="producer", layer="economics", dependencies=dependencies
            )
            if store.get(key) is None:
                misses += 1
                store.put(key, {"allocation": dependencies["importance"]})
            else:
                hits += 1
        assert store.count() == 4
    assert hits == 2
    assert misses == 1
