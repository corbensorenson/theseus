from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import semantic_ir
from code_lm_private_verifier import evaluate_private_candidates
from neural_seed_token_decoder_rendering import (
    PLAN_BODY_START_TOKEN,
    decode_candidate_body_tokens,
    split_learned_plan_prefix_tokens,
)
from neural_seed_token_decoder_comparator import token_candidate_rows_for_view
from neural_seed_token_decoder_support import body_tokens, target_tokens
from strict_generator_semantic_ir_repair_apply import apply_local_repair
from strict_generator_mlx_decode_eval import semantic_ir_plan_prefix_choices


BODY = """out = {}
for item in data or []:
    if not isinstance(item, dict) or item.get("done"):
        continue
    owner = item.get("owner") or "unassigned"
    out.setdefault(owner, []).append(str(item.get("label")))
return {key: sorted(values) for key, values in sorted(out.items())}
"""


def canonical(body: str) -> str:
    return ast.dump(ast.parse("def f(data):\n" + "".join(f"    {line}\n" for line in body.splitlines())), include_attributes=False)


def test_ordered_plan_body_target_keeps_learned_body_tokens() -> None:
    body = "out = []\nfor value in data:\n    out.append(value + 1)\nreturn out"
    tokens = target_tokens(body, target_mode=semantic_ir.PLAN_BODY_TARGET_MODE)
    assert tokens[0] == semantic_ir.PLAN_BEGIN
    assert semantic_ir.PLAN_END in tokens
    assert PLAN_BODY_START_TOKEN in tokens
    decoded_body_tokens, metadata = split_learned_plan_prefix_tokens(tokens)
    assert decoded_body_tokens == body_tokens(body)
    assert metadata["semantic_ir_plan_complete"] is True


def test_ordered_plan_is_generic_bounded_and_non_compilable() -> None:
    body = "total = 0\nfor value in data:\n    if value > 0:\n        total += value\nreturn total"
    tokens = semantic_ir.body_to_plan_tokens(body)
    assert len(tokens) <= semantic_ir.PLAN_MAX_TOKENS
    assert "IRP:SEM:traversal" in tokens
    assert "IRP:SEM:state_update" in tokens
    assert "IRP:STEP:D0:iterate" in tokens
    assert all("total" not in token and "value" not in token for token in tokens)
    compiled, receipt = semantic_ir.compile_body_tokens(tokens)
    assert compiled == ""
    assert receipt["state"] == "FAULT"
    assert receipt["candidate_generation_credit"] == 0


def test_plan_obligation_labels_are_fixed_generic_and_identifier_free() -> None:
    first = "quux = 0\nfor widget in payload:\n    quux += widget\nreturn quux"
    renamed = "zorb = 0\nfor flarn in records:\n    zorb += flarn\nreturn zorb"
    features = semantic_ir.plan_obligation_features()
    first_labels = semantic_ir.body_to_plan_obligation_labels(first)
    renamed_labels = semantic_ir.body_to_plan_obligation_labels(renamed)
    assert len(features) == len(first_labels) == len(renamed_labels)
    assert first_labels == renamed_labels
    assert any(first_labels)
    assert len(features) == len(set(features))
    assert all(token.startswith("IRP:") for token in features)
    assert not any(
        identifier in token
        for token in features
        for identifier in ("quux", "widget", "payload", "zorb", "flarn", "records")
    )


def test_semantic_plan_decode_boundary_uses_only_plan_vocabulary() -> None:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        semantic_ir.PLAN_BEGIN: 3,
        "IRP:STEP:D0:return": 4,
        "IRP:SEM:return_closure": 5,
        "IRP:FLOW:R1:W0": 6,
        "IRP:DATA:RARG0:WNONE": 7,
        "IRP:VALUE:name": 8,
        semantic_ir.PLAN_END: 9,
        PLAN_BODY_START_TOKEN: 10,
        "NAME:return": 11,
    }
    inverse = {index: token for token, index in vocab.items()}
    probabilities = [0.01] * (max(inverse) + 1)
    probabilities[11] = 0.9
    probabilities[4] = 0.8
    first = semantic_ir_plan_prefix_choices(probabilities, inverse, vocab, [], max_choices=2)
    assert first == [(3, probabilities[3])]
    content = semantic_ir_plan_prefix_choices(
        probabilities,
        inverse,
        vocab,
        [semantic_ir.PLAN_BEGIN],
        max_choices=2,
    )
    assert content == [(4, probabilities[4])]
    boundary = semantic_ir_plan_prefix_choices(
        probabilities,
        inverse,
        vocab,
        [
            semantic_ir.PLAN_BEGIN,
            "IRP:STEP:D0:return",
            "IRP:SEM:return_closure",
            "IRP:FLOW:R1:W0",
            "IRP:DATA:RARG0:WNONE",
            "IRP:VALUE:name",
            semantic_ir.PLAN_END,
        ],
        max_choices=2,
    )
    assert boundary == [(10, probabilities[10])]


def test_compact_plan_keeps_complete_steps_within_budget() -> None:
    body = "\n".join([f"value_{index} = data" for index in range(20)] + ["return data"])
    tokens = semantic_ir.body_to_plan_tokens(body, max_tokens=17)
    assert len(tokens) <= 17
    assert tokens[-1] == semantic_ir.PLAN_END
    assert sum(token.startswith("IRP:STEP:") for token in tokens) == 3
    assert sum(token.startswith("IRP:SEM:") for token in tokens) == 3
    assert sum(token.startswith("IRP:FLOW:") for token in tokens) == 3
    assert sum(token.startswith("IRP:DATA:") for token in tokens) == 3
    assert sum(token.startswith("IRP:VALUE:") for token in tokens) == 3


def test_plan_dataflow_is_identifier_free_and_renaming_invariant() -> None:
    first = "out = []\nfor item in data:\n    out.append(item)\nreturn out"
    renamed = "result = []\nfor value in payload:\n    result.append(value)\nreturn result"
    first_plan = semantic_ir.body_to_plan_tokens(first)
    renamed_plan = semantic_ir.body_to_plan_tokens(renamed)
    assert first_plan == renamed_plan
    assert "IRP:DATA:RARG0:WLOCAL1" in first_plan
    assert "IRP:DATA:RLOCAL0:WNONE" in first_plan
    assert all(
        name not in token
        for token in first_plan
        for name in ("out", "item", "data", "result", "value", "payload")
    )


def test_plan_information_dropout_preserves_protocol_shape_and_mass() -> None:
    plan = semantic_ir.body_to_plan_tokens(
        "total = 0\nfor value in data:\n    total += value\nreturn total"
    )
    dropped = semantic_ir.dropout_plan_tokens(plan)
    assert len(dropped) == len(plan)
    assert dropped[0] == semantic_ir.PLAN_BEGIN
    assert dropped[-1] == semantic_ir.PLAN_END
    assert dropped != plan
    prefix: list[str] = []
    for token in dropped:
        assert semantic_ir.plan_prefix_token_allowed(
            prefix, token, body_start_token=PLAN_BODY_START_TOKEN
        )
        prefix.append(token)


def test_ordered_plan_slot_field_preserves_order_without_identifiers() -> None:
    slots = 16
    protocol_width = len(semantic_ir.plan_protocol_tokens())
    features = semantic_ir.ordered_plan_slot_features(slots)
    labels = semantic_ir.body_to_ordered_plan_slot_labels(
        "out = []\nfor item in data:\n    out.append(item)\nreturn out",
        slot_count=slots,
    )
    assert len(features) == len(labels) == slots * protocol_width
    active = [index for index, value in enumerate(labels) if value]
    expected_plan = semantic_ir.body_to_plan_tokens(
        "out = []\nfor item in data:\n    out.append(item)\nreturn out",
        max_tokens=slots,
    )
    assert len(active) == len(expected_plan)
    assert [index // protocol_width for index in active] == list(range(len(expected_plan)))
    assert all(name not in feature for feature in features for name in ("out", "item", "data"))

    renamed = semantic_ir.body_to_ordered_plan_slot_labels(
        "result = []\nfor value in payload:\n    result.append(value)\nreturn result",
        slot_count=slots,
    )
    assert labels == renamed


def test_factorized_step_field_breaks_truncated_algorithm_collision() -> None:
    rle = """out = []
for item in data:
    if out and out[-1][0] == item:
        out[-1] = (item, out[-1][1] + 1)
    else:
        out.append((item, 1))
return out
"""
    merge = """intervals = []
for item in data:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        a, b = item[0], item[1]
        if b > a:
            intervals.append((a, b))
merged = []
for a, b in sorted(intervals):
    if not merged or a > merged[-1][1]:
        merged.append([a, b])
    else:
        merged[-1][1] = max(merged[-1][1], b)
return [tuple(item) for item in merged]
"""
    assert semantic_ir.body_to_ordered_plan_slot_labels(
        rle, slot_count=16
    ) == semantic_ir.body_to_ordered_plan_slot_labels(merge, slot_count=16)

    step_count = semantic_ir.PLAN_MAX_STEPS
    group_sizes = semantic_ir.ordered_plan_step_factor_group_sizes()
    features = semantic_ir.ordered_plan_step_features(step_count)
    rle_labels = semantic_ir.body_to_ordered_plan_step_labels(
        rle, step_count=step_count
    )
    merge_labels = semantic_ir.body_to_ordered_plan_step_labels(
        merge, step_count=step_count
    )
    assert len(features) == len(rle_labels) == step_count * sum(group_sizes)
    assert rle_labels != merge_labels
    assert all(identifier not in feature for feature in features for identifier in ("out", "item", "data", "intervals", "merged"))

    slot_width = sum(group_sizes)
    for labels in (rle_labels, merge_labels):
        for slot in range(step_count):
            row = labels[slot * slot_width : (slot + 1) * slot_width]
            presence = row[0]
            offset = 1
            for width in group_sizes[1:]:
                assert sum(row[offset : offset + width]) == presence
                offset += width

    renamed = rle.replace("out", "result").replace("item", "entry").replace("data", "records")
    assert rle_labels == semantic_ir.body_to_ordered_plan_step_labels(
        renamed, step_count=step_count
    )


def test_closed_plan_protocol_covers_generated_tokens_and_rejects_bad_transitions() -> None:
    generated = semantic_ir.body_to_plan_tokens(BODY, max_tokens=32)
    protocol = set(semantic_ir.plan_protocol_tokens())
    assert set(generated) <= protocol
    prefix: list[str] = []
    for token in generated:
        assert semantic_ir.plan_prefix_token_allowed(
            prefix,
            token,
            body_start_token=PLAN_BODY_START_TOKEN,
        )
        prefix.append(token)
    assert semantic_ir.plan_prefix_token_allowed(
        prefix,
        PLAN_BODY_START_TOKEN,
        body_start_token=PLAN_BODY_START_TOKEN,
    )
    assert not semantic_ir.plan_prefix_token_allowed(
        [semantic_ir.PLAN_BEGIN],
        "NAME:return",
        body_start_token=PLAN_BODY_START_TOKEN,
    )
    assert not semantic_ir.plan_prefix_token_allowed(
        [semantic_ir.PLAN_BEGIN],
        PLAN_BODY_START_TOKEN,
        body_start_token=PLAN_BODY_START_TOKEN,
    )


def test_generic_semantic_ir_roundtrips_nested_program() -> None:
    tokens = semantic_ir.body_to_tokens(BODY)
    rendered, receipt = semantic_ir.compile_body_tokens(tokens)
    assert receipt["state"] == "READY"
    assert receipt["roundtrip_ast_equal"] is True
    assert receipt["candidate_generation_credit"] == 0
    assert receipt["deterministic_compiler_credit"] == 0
    assert canonical(rendered) == canonical(BODY)


def test_semantic_ir_normalizes_source_extraction_indentation() -> None:
    extracted = "    value = int(data)\n    if value > 0:\n        return value\n    return 0"
    plan = semantic_ir.body_to_plan_tokens(extracted)
    tokens = semantic_ir.body_to_tokens(extracted)
    rendered, receipt = semantic_ir.compile_body_tokens(tokens)
    assert plan[0] == semantic_ir.PLAN_BEGIN
    assert receipt["state"] == "READY"
    assert "return value" in rendered


def test_semantic_ir_rejects_malformed_stream_without_fallback() -> None:
    rendered, receipt = semantic_ir.compile_body_tokens(
        [f"{semantic_ir.PROGRAM_BEGIN}:1", "IR:NODE:Return"]
    )
    assert rendered == ""
    assert receipt["state"] == "FAULT"
    assert receipt["fallback_return_count"] == 0
    assert receipt["typed_faults"][0]["failure_behavior"] == "reject_without_fallback"


def test_semantic_ir_rejects_unknown_node_kind() -> None:
    rendered, receipt = semantic_ir.compile_body_tokens(
        [
            f"{semantic_ir.PROGRAM_BEGIN}:1",
            "IR:NODE:AnswerTemplateRouter",
        ]
    )
    assert rendered == ""
    assert receipt["typed_faults"][0]["fault_type"] == "IR_NODE_KIND_UNSUPPORTED"


def test_candidate_receipt_rejects_unrepresented_top_level_effects() -> None:
    receipt = semantic_ir.candidate_receipt("import os\n\ndef f(data):\n    return data\n")
    assert receipt["state"] == "FAULT"
    assert receipt["typed_faults"][0]["fault_type"] == "IR_SINGLE_FUNCTION_REQUIRED"


def test_candidate_program_hash_binds_actual_callable_signature() -> None:
    one = semantic_ir.candidate_receipt("def f(data):\n    return data\n")
    two = semantic_ir.candidate_receipt("def f(data, other):\n    return data\n")
    assert one["body_ast_sha256"] == two["body_ast_sha256"]
    assert one["actual_signature_sha256"] != two["actual_signature_sha256"]
    assert one["program_sha256"] != two["program_sha256"]


def test_candidate_receipt_has_typed_graph_and_no_hidden_input_boundary() -> None:
    code = "def collect(data):\n" + "".join(f"    {line}\n" for line in BODY.splitlines())
    receipt = semantic_ir.candidate_receipt(
        code,
        prompt="Group labels by owner.",
        callable_signature="collect(data)",
        learned_prefix_tokens=["SLOT:PLAN_GENERIC_BODY"],
        vcm_context_ref="vcm:test",
        residual_lineage=["private:heldout"],
        include_graph=True,
    )
    assert receipt["state"] == "READY"
    assert receipt["atom_count"] > 10
    assert receipt["dependency_edge_count"] > 0
    assert receipt["roundtrip_ast_equal"] is True
    assert receipt["generation_boundary"]["uses_eval_tests_or_solutions"] is False
    assert receipt["generation_boundary"]["uses_answer_metadata"] is False
    assert receipt["program_graph"]["vcm_context_ref"] == "vcm:test"


def test_candidate_receipt_localizes_undefined_binding() -> None:
    receipt = semantic_ir.candidate_receipt("def f(data):\n    return missing + data\n", include_graph=True)
    assert receipt["state"] == "READY"
    assert "undefined_binding" in receipt["open_obligation_types"]
    obligations = receipt["program_graph"]["open_obligations"]
    assert obligations[0]["binding"] == "missing"
    assert obligations[0]["repair_scope"]


def test_target_mode_uses_generic_ir_and_remains_noncredit() -> None:
    tokens = target_tokens("return data", target_mode=semantic_ir.TARGET_MODE)
    body, meta = decode_candidate_body_tokens(
        tokens,
        {"entry_point": "identity", "prompt": "Return the input."},
        target_mode=semantic_ir.TARGET_MODE,
    )
    assert body == "return data"
    assert meta["rendered_from_typed_semantic_ir"] is True
    assert meta["candidate_generation_credit"] == 0
    assert meta["fallback_return_used"] is False


def test_private_verifier_recomputes_semantic_ir_independently() -> None:
    code = "def add_one(data):\n    return data + 1\n"
    candidate = {
        "task_id": "semantic-ir-private-replay",
        "phase": "private_eval",
        "rank": 1,
        "entry_point": "add_one",
        "code": code,
        "candidate_sha256": semantic_ir.stable_hash(code),
        "candidate_generation_mode": "token_level_code_decoder",
        "semantic_ir": semantic_ir.candidate_receipt(code),
    }
    task = {
        "task_id": "semantic-ir-private-replay",
        "split": "eval",
        "category": "semantic_ir_test",
        "entry_point": "add_one",
        "tests": "assert add_one(2) == 3\n",
    }
    report = evaluate_private_candidates([task], [candidate])
    trace = next(row for row in report["verification_attempt_labels"] if row.get("attempt_index") == 1)
    assert trace["intended_behavior_passed"] is True
    assert trace["semantic_ir_independently_recomputed"] is True
    assert trace["semantic_ir_receipt_match"] is True
    assert trace["semantic_ir_roundtrip_ast_equal"] is True


def test_localized_repair_consumes_model_prefix_and_stays_noncredit() -> None:
    row = {
        "code": "def f(data):\n    out = []\n    for item in data:\n        out = out + item\n    return out\n",
        "candidate_sha256": "source",
        "rank_score": -1.0,
        "source_task_id": "private:heldout",
        "body_structure_decode": {
            "predicted_plan": "AST_RETURN_MAX_AGGREGATE",
            "learned_plan_prefix_tokens": [
                "SLOT:PLAN_AST_RETURN_MAX_AGGREGATE",
                "SLOT:EXPR_CALL_STR",
            ],
        },
        "loop_plan_adequacy": {
            "failures": ["missing_expected_update_call"],
            "expectation": {"plan": "AST_RETURN_MAX_AGGREGATE"},
        },
    }
    repaired, reason = apply_local_repair(row)
    assert reason == "repaired"
    assert repaired is not None
    assert "key=str" in repaired["code"].replace(" ", "")
    assert repaired["candidate_generation_credit"] == 0
    assert repaired["token_level_code_generation_learned"] is False
    assert repaired["semantic_ir"]["state"] == "READY"
    assert repaired["semantic_ir_repair_apply"]["changed_atom_ids"]
    assert repaired["semantic_ir_repair_apply"]["comparison_key"] == "str"


def test_localized_repair_rejects_unsupported_plan_without_body() -> None:
    repaired, reason = apply_local_repair(
        {
            "code": "def f(data):\n    return data\n",
            "body_structure_decode": {"predicted_plan": "AST_GENERIC_RETURN"},
            "loop_plan_adequacy": {
                "failures": ["missing_expected_update_call"],
                "expectation": {"plan": "AST_GENERIC_RETURN"},
            },
        }
    )
    assert repaired is None
    assert reason == "unsupported_plan_for_local_repair"


def test_direct_candidate_path_attaches_prompt_signature_ir_receipt() -> None:
    task = {
        "task_id": "semantic-ir-direct-candidate",
        "source_task_id": "private-source",
        "entry_point": "identity",
        "prompt": "Return the input unchanged.",
        "split": "eval",
        "public_benchmark": False,
    }
    decoded = [
        {
            "decoded_tokens": target_tokens("return data", target_mode="body_tokens") + ["<eos>"],
            "rank_score": -0.1,
            "decoded_token_count": 3,
            "decoded_token_sha256": "direct-candidate",
            "body": "return data",
        }
    ]
    rows = token_candidate_rows_for_view(
        [task],
        [decoded],
        arm_id="transformer_control",
        substrate="test_transformer",
        phase="private_eval",
        view="sts_on",
        config={
            "policy": "test",
            "candidate_row_schema": {},
            "text_views": {"sts_on": ["prompt", "entry_point"]},
        },
        seed=1,
        target_mode="body_tokens",
        residual_context={},
        output_top_k=1,
    )
    assert len(rows) == 1
    assert rows[0]["semantic_ir"]["state"] == "READY"
    assert rows[0]["semantic_ir"]["vcm_context_ref"]
    assert rows[0]["semantic_ir"]["generation_boundary"]["prompt_sha256"] != semantic_ir.stable_hash("")
    assert rows[0]["semantic_ir"]["generation_boundary"]["callable_signature_sha256"] != semantic_ir.stable_hash("")
