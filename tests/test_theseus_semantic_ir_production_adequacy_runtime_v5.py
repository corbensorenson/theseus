from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_semantic_ir_production_adequacy_runtime_v5 as runtime  # noqa: E402


TASK = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v5_task_04.json"
PACKET = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v5_candidate_packet_04.json"


def extract_parent(tmp_path: Path) -> tuple[Path, dict]:
    task = json.loads(TASK.read_text(encoding="utf-8"))
    root = tmp_path / "source"
    with tarfile.open(ROOT / task["source_archive"]) as archive:
        archive.extractall(root, filter="data")
    return root / task["source_archive_root"], task


def test_compact_inventory_is_complete_unique_and_hides_redundant_fields(
    tmp_path: Path,
) -> None:
    root, task = extract_parent(tmp_path)
    symbols = runtime.semantic_scope_symbol_table(root, task)
    context = runtime.render_common_context(root, task, symbols)
    address_inventory = context.split("[COMPACT_SEMANTIC_NODE_ABI]\n", 1)[1].split(
        "[MISSING_ALLOWED_EFFECT_PATHS]", 1
    )[0]

    assert len(symbols["nodes"]) == 577
    assert len({row["id"] for row in symbols["nodes"]}) == 577
    assert all(len(row["id"]) == 34 for row in symbols["nodes"])
    assert symbols["inventory_complete"] is True
    assert symbols["inventory_truncated"] is False
    assert symbols["compact_integrity_abi"]["handle_bits"] == 128
    assert symbols["compact_integrity_abi"]["full_node_sha256_candidate_visible"] is False
    assert not any(row["sha256"] in address_inventory for row in symbols["nodes"])
    assert not any(
        str(row["label"]) in address_inventory
        for row in symbols["nodes"]
        if len(str(row["label"])) > 12
    )
    assert context.count("PATH src/black/nodes.py") == 1


def test_compact_parser_resolves_full_digest_independently(tmp_path: Path) -> None:
    root, task = extract_parent(tmp_path)
    symbols = runtime.semantic_scope_symbol_table(root, task)
    target = next(
        row
        for row in symbols["nodes"]
        if row["path"] == "src/black/nodes.py"
        and row["node_type"] == "Return"
        and row["start_line"] == 405
    )
    artifact = (
        f"{runtime.HEADER}\n"
        f"SOURCE {symbols['source_digest']}\n"
        "ALL_OBLIGATIONS O1,O2,O3\n"
        "CHANGE_OBLIGATIONS O1\n"
        "PRESERVE_OBLIGATIONS O2\n"
        "NON_GOAL_OBLIGATIONS O3\n"
        "UNIT U1\n"
        "OBLIGATIONS O1,O2\n"
        "OP REPLACE\n"
        "PATH src/black/nodes.py\n"
        f"NODE {target['id']}\n"
        "<<<\nreturn True\n>>>\n"
        "END_UNIT\n"
        "LOSS NONE\n"
        "END"
    )
    parsed = runtime.parse(artifact, task, root)

    assert parsed["faults"] == []
    assert parsed["actions"][0]["start_line"] == 405
    assert parsed["semantic_receipt"]["schema"] == runtime.HEADER
    assert parsed["semantic_receipt"]["candidate_supplied_full_node_sha256"] is False
    assert parsed["semantic_receipt"]["node_identity_resolution"][0]["resolved"] is True
    assert "NODE_SHA" not in parsed["canonical_ir"]


def test_compact_prompt_materially_reduces_consumed_task_without_truncation(
    tmp_path: Path,
) -> None:
    root, task = extract_parent(tmp_path)
    symbols = runtime.semantic_scope_symbol_table(root, task)
    prompt = runtime.render_prompt(task, runtime.render_common_context(root, task, symbols))
    prior = json.loads(PACKET.read_text(encoding="utf-8"))["serialized_prompt"]

    assert len(prompt.encode("utf-8")) < len(prior.encode("utf-8")) * 0.65
    assert p2a.render_visible_context(root, task) in prompt
    address_inventory = prompt.split("[COMPACT_SEMANTIC_NODE_ABI]\n", 1)[1].split(
        "[MISSING_ALLOWED_EFFECT_PATHS]", 1
    )[0]
    assert address_inventory.count("N-") == len(symbols["nodes"])
    assert "NODE_SHA" not in prompt
    assert runtime.complete(
        prompt.split("OUTPUT ONLY THIS SHAPE:\n", 1)[1]
        .replace("<copy exact semantic source digest>", symbols["source_digest"])
        .replace("<change ids plus dependency-required preserve ids>", "O1,O2", 1)
        .replace("<REPLACE|INSERT_BEFORE|INSERT_AFTER>", "REPLACE")
        .replace("<copy exact path from the nearest PATH group>", "src/black/nodes.py")
        .replace("<copy exact 128-bit semantic node handle>", symbols["nodes"][0]["id"])
        .replace("<replacement source for only that node>", "pass")
        .split("For a path listed under MISSING_ALLOWED_EFFECT_PATHS", 1)[0]
        .rstrip()
        + "\nLOSS NONE\nEND"
    ) is True
