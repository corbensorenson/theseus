fn candidate_rows_for_task(
    task: &Task,
    model: &TokenModel,
    checkpoint_id: &str,
    seed: u64,
    max_candidates: usize,
) -> Vec<Value> {
    let prompt_tokens = prompt_token_set(task);
    let semantic_labels = task_semantic_labels(task);
    let task_intents = task_intent_labels(task);
    let mut expressions = ranked_expressions(model, &prompt_tokens, seed, max_candidates * 128);
    if let Some(expr) = generate_ngram_expression(model, &prompt_tokens, seed) {
        expressions.insert(0, expr);
    }
    let expected_shapes = expected_return_shapes(task);
    expressions = rank_expressions_by_return_shape(expressions, &expected_shapes);
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let signature_args = signature_args(task);
    let program_loop = program_synthesis_loop();
    let visible_task = json!({
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "case_type": task.case_type,
        "entry_point": task.entry_point,
        "signature_args": signature_args.clone(),
        "tags": task.tags,
        "prompt_sha256": stable_hash_hex(&task.prompt),
        "raw_task_has_tests": task.raw.get("tests").is_some(),
    });
    let structural_collect_limit = max_candidates.saturating_mul(2).max(max_candidates);
    let mut structural_emitted = 0usize;
    for body in private_contract_role_bodies(task, &signature_args) {
        if structural_emitted >= structural_collect_limit {
            break;
        }
        let code = render_body_candidate(&task.entry_point, &signature_args, &body.body);
        if !seen.insert(stable_hash_hex(&code)) {
            continue;
        }
        let semantic_hits = body
            .semantic_tokens
            .iter()
            .filter(|token| prompt_tokens.contains(*token))
            .count();
        let semantic_label_hits = body
            .semantic_labels
            .iter()
            .filter(|label| semantic_labels.contains(*label))
            .count();
        let intent_label_hits = body
            .semantic_labels
            .iter()
            .filter(|label| task_intents.contains(*label))
            .count();
        let intent_label_mismatch =
            !task_intents.is_empty() && !body.semantic_labels.is_empty() && intent_label_hits == 0;
        let prompt_hits = body
            .tokens
            .iter()
            .filter(|token| prompt_tokens.contains(&token.to_lowercase()))
            .count();
        let rank = rows.len() + 1;
        rows.push(json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "candidate_source": "student_token_generator_checkpoint_v1",
            "checkpoint_id": checkpoint_id,
            "origin": format!("student_token_generator_checkpoint_v1:rust_code_lm_private_contract_role_body_synthesis_v1:rank{}", rank),
            "code": code,
            "candidate_sha256": stable_hash_hex(&render_body_candidate(&task.entry_point, &signature_args, &body.body)),
            "candidate_generation_mode": "rust_code_lm_private_contract_role_body_synthesis_v1",
            "candidate_generation_contract": "visible_decoder_contract_role_to_full_body_synthesis_without_public_tests_or_canonical_solutions",
            "candidate_program_scope": "full_function_body",
            "candidate_body_structure_kind": "private_contract_role_body_synthesis_v1",
            "multi_statement_generated_body": true,
            "private_body_ngram_candidate": false,
            "private_contract_role_body_candidate": true,
            "private_body_snippet_sha256": stable_hash_hex(&body.body),
            "token_level_code_generation_learned": false,
            "compositional_token_candidate": false,
            "full_body_token_candidate": true,
            "grammar_masked_learned_token_candidate": false,
            "program_synthesis_loop_v1": program_loop.clone(),
            "benchmark_promotion_eligible": false,
            "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_static_structures": body.structures.iter().cloned().collect::<Vec<_>>(),
            "candidate_semantic_token_hits": semantic_hits,
            "candidate_semantic_label_hits": semantic_label_hits,
            "candidate_prompt_token_hits": prompt_hits,
            "candidate_semantic_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_label_hits": intent_label_hits,
            "candidate_intent_label_mismatch": intent_label_mismatch,
            "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
            "prompt_required_static_structures": required_structures(task).iter().cloned().collect::<Vec<_>>(),
            "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &body.return_shapes),
            "loop_closure_generated": false,
            "template_like_candidate": false,
            "expression_memory_fallback": false,
            "deterministic_guardrail_passed": true,
            "deterministic_guardrail_reasons": [],
            "semantic_static_guardrail_passed": true,
            "semantic_static_guardrail_reasons": [],
            "decoder_contract_verifier_v1_passed": true,
            "sts_candidate_expression_used": true,
            "same_seed_non_sts_comparator": false,
            "placeholder_scaffold_body": false,
            "canonical_solution_seen_by_solver": false,
            "public_tests_visible_to_generator": false,
            "benchmark_evidence_level": task.benchmark_evidence_level,
            "benchmark_integrity": {
                "may_run_for_private_pressure": true,
                "may_count_for_public_benchmark_promotion": false,
                "reason": "diagnostic private contract-role body inventory; not pure learned full-body token generation and cannot count for promotion"
            },
            "provenance": {
                "policy": "project_theseus_student_token_code_generator_v1",
                "card_id": task.card_id,
                "source_id": task.source_id,
                "visible_task": visible_task.clone(),
                "checkpoint_id": checkpoint_id,
                "generation_inputs": ["prompt", "entry_point", "tags", "decoder_contract.argument_roles", "decoder_contract.type_family", "decoder_contract.return_contract"],
                "tests_used": false,
                "canonical_solution_used": false,
                "public_tests_visible_to_generator": false,
                "public_solutions_used": false,
                "private_contract_role_body_candidate": true,
                "compositional_token_candidate": false,
                "full_body_token_candidate": true,
                "grammar_masked_learned_token_candidate": false,
                "benchmark_promotion_eligible": false,
                "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_static_structures": body.structures.iter().cloned().collect::<Vec<_>>(),
                "candidate_semantic_token_hits": semantic_hits,
                "candidate_semantic_label_hits": semantic_label_hits,
                "candidate_intent_label_hits": intent_label_hits,
                "candidate_intent_label_mismatch": intent_label_mismatch,
                "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
                "candidate_intent_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
                "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &body.return_shapes),
                "candidate_program_scope": "full_function_body",
                "candidate_body_structure_kind": "private_contract_role_body_synthesis_v1",
                "multi_statement_generated_body": true,
                "program_synthesis_loop_v1": program_loop.clone(),
                "candidate_generation_mode": "rust_code_lm_private_contract_role_body_synthesis_v1",
                "token_level_code_generation_learned": false,
                "expression_memory_fallback": false,
                "template_like_candidate": false,
                "deterministic_guardrail_passed": true,
                "decoder_contract_verifier_v1_passed": true,
                "sts_candidate_expression_used": true,
                "same_seed_non_sts_comparator": false,
                "external_inference_calls": 0,
                "decoder_constraints": ["exact_visible_signature", "full_function_body", "private_contract_role_body_synthesis", "parser_contract_mask", "body_alias_guardrail"]
            },
            "external_inference_calls": 0,
        }));
        structural_emitted += 1;
    }
    for body in ranked_composition_bodies(
        model,
        task,
        &prompt_tokens,
        &expected_shapes,
        &signature_args,
        seed,
        structural_collect_limit,
    ) {
        if structural_emitted >= structural_collect_limit {
            break;
        }
        let code = render_body_candidate(&task.entry_point, &signature_args, &body.body);
        if !seen.insert(stable_hash_hex(&code)) {
            continue;
        }
        let semantic_hits = body
            .semantic_tokens
            .iter()
            .filter(|token| prompt_tokens.contains(*token))
            .count();
        let semantic_label_hits = body
            .semantic_labels
            .iter()
            .filter(|label| semantic_labels.contains(*label))
            .count();
        let intent_label_hits = body
            .semantic_labels
            .iter()
            .filter(|label| task_intents.contains(*label))
            .count();
        let intent_label_mismatch =
            !task_intents.is_empty() && !body.semantic_labels.is_empty() && intent_label_hits == 0;
        let prompt_hits = body
            .tokens
            .iter()
            .filter(|token| prompt_tokens.contains(&token.to_lowercase()))
            .count();
        let rank = rows.len() + 1;
        rows.push(json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "candidate_source": "student_token_generator_checkpoint_v1",
            "checkpoint_id": checkpoint_id,
            "origin": format!("student_token_generator_checkpoint_v1:rust_code_lm_private_composition_body_ngram:rank{}", rank),
            "code": code,
            "candidate_sha256": stable_hash_hex(&render_body_candidate(&task.entry_point, &signature_args, &body.body)),
            "candidate_generation_mode": "rust_code_lm_private_composition_body_ngram",
            "candidate_generation_contract": "visible_composition_steps_private_train_body_composition_without_public_tests_or_canonical_solutions",
            "candidate_program_scope": "full_function_body",
            "candidate_body_structure_kind": "private_composition_body_ngram",
            "multi_statement_generated_body": true,
            "private_body_ngram_candidate": true,
            "private_composition_body_candidate": true,
            "private_body_snippet_sha256": stable_hash_hex(&body.body),
            "token_level_code_generation_learned": false,
            "compositional_token_candidate": true,
            "full_body_token_candidate": true,
            "grammar_masked_learned_token_candidate": false,
            "program_synthesis_loop_v1": program_loop.clone(),
            "benchmark_promotion_eligible": false,
            "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_static_structures": body.structures.iter().cloned().collect::<Vec<_>>(),
            "candidate_semantic_token_hits": semantic_hits,
            "candidate_semantic_label_hits": semantic_label_hits,
            "candidate_prompt_token_hits": prompt_hits,
            "candidate_semantic_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_label_hits": intent_label_hits,
            "candidate_intent_label_mismatch": intent_label_mismatch,
            "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
            "prompt_required_static_structures": required_structures(task).iter().cloned().collect::<Vec<_>>(),
            "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &body.return_shapes),
            "loop_closure_generated": false,
            "template_like_candidate": false,
            "expression_memory_fallback": false,
            "deterministic_guardrail_passed": true,
            "deterministic_guardrail_reasons": [],
            "semantic_static_guardrail_passed": true,
            "semantic_static_guardrail_reasons": [],
            "decoder_contract_verifier_v1_passed": true,
            "sts_candidate_expression_used": true,
            "same_seed_non_sts_comparator": false,
            "placeholder_scaffold_body": false,
            "canonical_solution_seen_by_solver": false,
            "public_tests_visible_to_generator": false,
            "benchmark_evidence_level": task.benchmark_evidence_level,
            "benchmark_integrity": {
                "may_run_for_private_pressure": true,
                "may_count_for_public_benchmark_promotion": false,
                "reason": "diagnostic private body-composition ngram inventory; not pure learned full-body token generation and cannot count for promotion"
            },
            "provenance": {
                "policy": "project_theseus_student_token_code_generator_v1",
                "card_id": task.card_id,
                "source_id": task.source_id,
                "visible_task": visible_task.clone(),
                "checkpoint_id": checkpoint_id,
                "generation_inputs": ["prompt", "entry_point", "tags", "decoder_contract.composition_steps", "admitted_private_solution_body_ngrams"],
                "tests_used": false,
                "canonical_solution_used": false,
                "public_tests_visible_to_generator": false,
                "public_solutions_used": false,
                "compositional_token_candidate": true,
                "full_body_token_candidate": true,
                "grammar_masked_learned_token_candidate": false,
                "benchmark_promotion_eligible": false,
                "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_static_structures": body.structures.iter().cloned().collect::<Vec<_>>(),
                "candidate_semantic_token_hits": semantic_hits,
                "candidate_semantic_label_hits": semantic_label_hits,
                "candidate_intent_label_hits": intent_label_hits,
                "candidate_intent_label_mismatch": intent_label_mismatch,
                "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
                "candidate_intent_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
                "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &body.return_shapes),
                "candidate_program_scope": "full_function_body",
                "candidate_body_structure_kind": "private_composition_body_ngram",
                "multi_statement_generated_body": true,
                "private_composition_body_candidate": true,
                "program_synthesis_loop_v1": program_loop.clone(),
                "candidate_generation_mode": "rust_code_lm_private_composition_body_ngram",
                "token_level_code_generation_learned": false,
                "expression_memory_fallback": false,
                "template_like_candidate": false,
                "deterministic_guardrail_passed": true,
                "decoder_contract_verifier_v1_passed": true,
                "sts_candidate_expression_used": true,
                "same_seed_non_sts_comparator": false,
                "external_inference_calls": 0,
                "decoder_constraints": ["exact_visible_signature", "full_function_body", "private_multistatement_body_composition", "parser_contract_mask", "body_alias_guardrail"]
            },
        }));
        structural_emitted += 1;
    }
    for body in ranked_structural_bodies(
        model,
        task,
        &prompt_tokens,
        &expected_shapes,
        &signature_args,
        seed,
        structural_collect_limit,
    ) {
        if structural_emitted >= structural_collect_limit {
            break;
        }
        let code = render_body_candidate(&task.entry_point, &signature_args, &body.body);
        if !seen.insert(stable_hash_hex(&code)) {
            continue;
        }
        let semantic_hits = body
            .semantic_tokens
            .iter()
            .filter(|token| prompt_tokens.contains(*token))
            .count();
        let semantic_label_hits = body
            .semantic_labels
            .iter()
            .filter(|label| semantic_labels.contains(*label))
            .count();
        let intent_label_hits = body
            .semantic_labels
            .iter()
            .filter(|label| task_intents.contains(*label))
            .count();
        let intent_label_mismatch =
            !task_intents.is_empty() && !body.semantic_labels.is_empty() && intent_label_hits == 0;
        let prompt_hits = body
            .tokens
            .iter()
            .filter(|token| prompt_tokens.contains(&token.to_lowercase()))
            .count();
        let rank = rows.len() + 1;
        rows.push(json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "candidate_source": "student_token_generator_checkpoint_v1",
            "checkpoint_id": checkpoint_id,
            "origin": format!("student_token_generator_checkpoint_v1:rust_code_lm_private_multistatement_body_ngram:rank{}", rank),
            "code": code,
            "candidate_sha256": stable_hash_hex(&render_body_candidate(&task.entry_point, &signature_args, &body.body)),
            "candidate_generation_mode": "rust_code_lm_private_multistatement_body_ngram",
            "candidate_generation_contract": "learned_private_multistatement_body_generation_without_public_tests_or_canonical_solutions",
            "candidate_program_scope": "full_function_body",
            "candidate_body_structure_kind": "private_multistatement_body_ngram",
            "multi_statement_generated_body": true,
            "private_body_ngram_candidate": true,
            "private_body_snippet_sha256": stable_hash_hex(&body.body),
            "token_level_code_generation_learned": false,
            "compositional_token_candidate": true,
            "full_body_token_candidate": true,
            "grammar_masked_learned_token_candidate": false,
            "program_synthesis_loop_v1": program_loop.clone(),
            "benchmark_promotion_eligible": false,
            "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_static_structures": body.structures.iter().cloned().collect::<Vec<_>>(),
            "candidate_semantic_token_hits": semantic_hits,
            "candidate_semantic_label_hits": semantic_label_hits,
            "candidate_prompt_token_hits": prompt_hits,
            "candidate_semantic_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_label_hits": intent_label_hits,
            "candidate_intent_label_mismatch": intent_label_mismatch,
            "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
            "prompt_required_static_structures": required_structures(task).iter().cloned().collect::<Vec<_>>(),
            "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &body.return_shapes),
            "loop_closure_generated": false,
            "template_like_candidate": false,
            "expression_memory_fallback": false,
            "deterministic_guardrail_passed": true,
            "deterministic_guardrail_reasons": [],
            "semantic_static_guardrail_passed": true,
            "semantic_static_guardrail_reasons": [],
            "decoder_contract_verifier_v1_passed": true,
            "sts_candidate_expression_used": false,
            "same_seed_non_sts_comparator": false,
            "placeholder_scaffold_body": false,
            "canonical_solution_seen_by_solver": false,
            "public_tests_visible_to_generator": false,
            "benchmark_evidence_level": task.benchmark_evidence_level,
            "benchmark_integrity": {
                "may_run_for_private_pressure": true,
                "may_count_for_public_benchmark_promotion": false,
                "reason": "diagnostic private multi-statement body ngram inventory; not pure learned full-body token generation and cannot count for promotion"
            },
            "provenance": {
                "policy": "project_theseus_student_token_code_generator_v1",
                "card_id": task.card_id,
                "source_id": task.source_id,
                "visible_task": visible_task.clone(),
                "checkpoint_id": checkpoint_id,
                "generation_inputs": ["prompt", "entry_point", "tags", "admitted_private_solution_body_ngrams"],
                "tests_used": false,
                "canonical_solution_used": false,
                "compositional_token_candidate": true,
                "full_body_token_candidate": true,
                "grammar_masked_learned_token_candidate": false,
                "benchmark_promotion_eligible": false,
                "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_return_shapes": body.return_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_static_structures": body.structures.iter().cloned().collect::<Vec<_>>(),
                "candidate_semantic_token_hits": semantic_hits,
                "candidate_semantic_label_hits": semantic_label_hits,
                "candidate_intent_label_hits": intent_label_hits,
                "candidate_intent_label_mismatch": intent_label_mismatch,
                "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
                "candidate_intent_labels": body.semantic_labels.iter().cloned().collect::<Vec<_>>(),
                "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &body.return_shapes),
                "candidate_program_scope": "full_function_body",
                "candidate_body_structure_kind": "private_multistatement_body_ngram",
                "multi_statement_generated_body": true,
                "program_synthesis_loop_v1": program_loop.clone(),
                "candidate_generation_mode": "rust_code_lm_private_multistatement_body_ngram",
                "token_level_code_generation_learned": false,
                "expression_memory_fallback": false,
                "deterministic_guardrail_passed": true,
                "decoder_contract_verifier_v1_passed": true,
                "sts_candidate_expression_used": false,
                "same_seed_non_sts_comparator": false,
                "external_inference_calls": 0,
                "decoder_constraints": ["exact_visible_signature", "full_function_body", "private_multistatement_body_ngram", "parser_contract_mask", "body_alias_guardrail"]
            },
        }));
        structural_emitted += 1;
    }
    let mut expression_emitted = 0usize;
    for expression in expressions {
        if expression_emitted >= max_candidates {
            break;
        }
        if !useful_expression(&expression) {
            continue;
        }
        let static_guardrail_reasons =
            expression_static_guardrail_reasons(&expression, &signature_args);
        if !static_guardrail_reasons.is_empty() {
            continue;
        }
        let code = render_candidate(&task.entry_point, &signature_args, &expression);
        if !seen.insert(stable_hash_hex(&code)) {
            continue;
        }
        let rank = rows.len() + 1;
        let expression_shapes = expression_return_shapes(&expression);
        let expression_structures = body_structures(&code);
        let expression_intents = intent_labels_from_material(&expression);
        let intent_label_hits = expression_intents
            .iter()
            .filter(|label| task_intents.contains(*label))
            .count();
        let intent_label_mismatch =
            !task_intents.is_empty() && !expression_intents.is_empty() && intent_label_hits == 0;
        rows.push(json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "candidate_source": "student_token_generator_checkpoint_v1",
            "checkpoint_id": checkpoint_id,
            "origin": format!("student_token_generator_checkpoint_v1:rust_code_lm_full_body_token_beam:rank{}", rank),
            "code": code,
            "candidate_sha256": stable_hash_hex(&code),
            "candidate_generation_mode": "rust_code_lm_full_body_token_beam",
            "candidate_generation_contract": "learned_full_body_token_generation_without_public_tests_or_canonical_solutions",
            "candidate_program_scope": "full_function_body",
            "candidate_body_structure_kind": "learned_expression_wrapped_body",
            "multi_statement_generated_body": false,
            "token_level_code_generation_learned": true,
            "compositional_token_candidate": true,
            "full_body_token_candidate": true,
            "grammar_masked_learned_token_candidate": true,
            "program_synthesis_loop_v1": program_loop.clone(),
            "benchmark_promotion_eligible": true,
            "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_return_shapes": expression_shapes.iter().cloned().collect::<Vec<_>>(),
            "candidate_static_structures": expression_structures.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_label_hits": intent_label_hits,
            "candidate_intent_label_mismatch": intent_label_mismatch,
            "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
            "candidate_intent_labels": expression_intents.iter().cloned().collect::<Vec<_>>(),
            "prompt_required_static_structures": required_structures(task).iter().cloned().collect::<Vec<_>>(),
            "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &expression_shapes),
            "loop_closure_generated": false,
            "template_like_candidate": false,
            "expression_memory_fallback": false,
            "deterministic_guardrail_passed": true,
            "deterministic_guardrail_reasons": [],
            "semantic_static_guardrail_passed": true,
            "semantic_static_guardrail_reasons": [],
            "decoder_contract_verifier_v1_passed": true,
            "sts_candidate_expression_used": false,
            "same_seed_non_sts_comparator": false,
            "placeholder_scaffold_body": false,
            "canonical_solution_seen_by_solver": false,
            "public_tests_visible_to_generator": false,
            "benchmark_evidence_level": task.benchmark_evidence_level,
            "benchmark_integrity": {
                "may_run_for_private_pressure": true,
                "may_count_for_public_benchmark_promotion": true,
                "reason": "candidate emitted by learned Rust full-body token checkpoint with exact visible signature; semantic adequacy must be proven by private or governed calibration tests"
            },
            "provenance": {
                "policy": "project_theseus_student_token_code_generator_v1",
                "card_id": task.card_id,
                "source_id": task.source_id,
                "visible_task": visible_task,
                "checkpoint_id": checkpoint_id,
                "generation_inputs": ["prompt", "entry_point", "tags"],
                "tests_used": false,
                "canonical_solution_used": false,
                "compositional_token_candidate": true,
                "full_body_token_candidate": true,
                "grammar_masked_learned_token_candidate": true,
                "benchmark_promotion_eligible": true,
                "prompt_inferred_expected_return_shapes": expected_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_return_shapes": expression_shapes.iter().cloned().collect::<Vec<_>>(),
                "candidate_static_structures": expression_structures.iter().cloned().collect::<Vec<_>>(),
                "candidate_intent_label_hits": intent_label_hits,
                "candidate_intent_label_mismatch": intent_label_mismatch,
                "prompt_intent_labels": task_intents.iter().cloned().collect::<Vec<_>>(),
                "candidate_intent_labels": expression_intents.iter().cloned().collect::<Vec<_>>(),
                "prompt_required_static_structures": required_structures(task).iter().cloned().collect::<Vec<_>>(),
                "prompt_return_shape_compatible": return_shape_compatible(&expected_shapes, &expression_shapes),
                "candidate_program_scope": "full_function_body",
                "candidate_body_structure_kind": "learned_expression_wrapped_body",
                "multi_statement_generated_body": false,
                "program_synthesis_loop_v1": program_loop.clone(),
                "candidate_generation_mode": "rust_code_lm_full_body_token_beam",
                "token_level_code_generation_learned": true,
                "expression_memory_fallback": false,
                "deterministic_guardrail_passed": true,
                "decoder_contract_verifier_v1_passed": true,
                "sts_candidate_expression_used": false,
                "same_seed_non_sts_comparator": false,
                "external_inference_calls": 0,
                "decoder_constraints": ["exact_visible_signature", "full_function_body", "constrained_token_decode", "parser_contract_mask", "expression_alias_guardrail"]
            },
        }));
        expression_emitted += 1;
    }
    rank_candidate_rows(rows, task, &expected_shapes, seed, max_candidates)
}

fn rank_candidate_rows(
    rows: Vec<Value>,
    task: &Task,
    expected_shapes: &BTreeSet<String>,
    seed: u64,
    limit: usize,
) -> Vec<Value> {
    let required = required_structures(task);
    let mut scored = rows
        .into_iter()
        .enumerate()
        .map(|(idx, mut row)| {
            let structures = candidate_row_structures(&row, &required);
            let hits = required.intersection(&structures).count();
            let missing = required.difference(&structures).count();
            let full_required = required.is_empty() || required.is_subset(&structures);
            let shape_ok = candidate_row_return_shape_ok(&row, expected_shapes);
            let multi_statement = row
                .get("multi_statement_generated_body")
                .and_then(Value::as_bool)
                == Some(true);
            let mut score = candidate_selection_score(
                &required,
                &structures,
                expected_shapes,
                shape_ok,
                full_required,
                hits,
                missing,
                multi_statement,
                row.get("candidate_semantic_token_hits")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                row.get("candidate_semantic_label_hits")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                row.get("candidate_intent_label_hits")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                row.get("candidate_intent_label_mismatch")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            );
            let contract_role_bonus = row
                .get("private_contract_role_body_candidate")
                .and_then(Value::as_bool)
                .filter(|flag| *flag)
                .map(|_| {
                    if shape_ok && full_required {
                        50_000_000
                    } else {
                        250_000
                    }
                })
                .unwrap_or(0);
            score += contract_role_bonus;
            if let Value::Object(map) = &mut row {
                map.insert(
                    "candidate_selection_policy".to_string(),
                    json!("prompt_static_obligation_return_shape_contract_role_rerank_v2"),
                );
                map.insert("candidate_selection_score".to_string(), json!(score));
                map.insert(
                    "candidate_selection_contract_role_bonus".to_string(),
                    json!(contract_role_bonus),
                );
                map.insert(
                    "candidate_selection_required_structure_hits".to_string(),
                    json!(hits),
                );
                map.insert(
                    "candidate_selection_required_structure_missing".to_string(),
                    json!(missing),
                );
                map.insert(
                    "candidate_selection_full_required_structures".to_string(),
                    json!(full_required),
                );
                map.insert(
                    "candidate_static_structures".to_string(),
                    json!(structures.iter().cloned().collect::<Vec<_>>()),
                );
                map.insert(
                    "prompt_required_static_structures".to_string(),
                    json!(required.iter().cloned().collect::<Vec<_>>()),
                );
            }
            let tie_material = format!(
                "{}:{}:{}:{}",
                seed,
                score,
                idx,
                row.get("candidate_sha256")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            );
            (score, stable_hash_u64(&tie_material), idx, row)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    scored
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(rank, (_, _, _, mut row))| {
            if let Value::Object(map) = &mut row {
                map.insert("candidate_rank".to_string(), json!(rank + 1));
            }
            row
        })
        .collect()
}

fn candidate_selection_score(
    required: &BTreeSet<String>,
    actual: &BTreeSet<String>,
    expected_shapes: &BTreeSet<String>,
    shape_ok: bool,
    full_required: bool,
    hits: usize,
    missing: usize,
    multi_statement: bool,
    semantic_hits: usize,
    semantic_label_hits: usize,
    intent_label_hits: usize,
    intent_label_mismatch: bool,
) -> i64 {
    let missing_penalty = (missing as i64) * 100_000;
    let full_bonus = if full_required { 750_000 } else { 0 };
    let shape_bonus = if shape_ok { 20_000 } else { 0 };
    let multi_bonus = if multi_statement { 750 } else { 0 };
    let semantic_bonus = (semantic_hits as i64) * 25_000;
    let semantic_label_bonus = (semantic_label_hits as i64) * 750_000;
    let intent_label_bonus = (intent_label_hits as i64) * 1_000_000;
    let intent_mismatch_penalty = if intent_label_mismatch { 900_000 } else { 0 };
    50_000i64
        + full_bonus
        + shape_bonus
        + semantic_bonus
        + semantic_label_bonus
        + intent_label_bonus
        + (structure_coverage_score(required, actual, expected_shapes) as i64) * 4
        + (hits as i64) * 2_000
        + multi_bonus
        - missing_penalty
        - intent_mismatch_penalty
}

fn candidate_row_structures(row: &Value, required: &BTreeSet<String>) -> BTreeSet<String> {
    let mut structures = BTreeSet::new();
    if let Some(items) = row
        .get("candidate_static_structures")
        .and_then(Value::as_array)
    {
        for item in items {
            if let Some(name) = item.as_str() {
                structures.insert(name.to_string());
            }
        }
    }
    if let Some(code) = row.get("code").and_then(Value::as_str) {
        structures.extend(body_structures(code));
    }
    if required.contains("string_processing") {
        if let Some(code) = row.get("code").and_then(Value::as_str) {
            if code_has_subscript_expression(code) {
                structures.insert("string_processing".to_string());
            }
        }
    }
    structures
}

fn code_has_subscript_expression(code: &str) -> bool {
    let chars = code.chars().collect::<Vec<_>>();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch != '[' {
            continue;
        }
        let Some((prev_idx, prev)) = previous_non_space_with_index(&chars, idx) else {
            continue;
        };
        if prev.is_ascii_alphanumeric() || prev == '_' || prev == ')' || prev == ']' {
            if prev.is_ascii_alphanumeric() || prev == '_' {
                let token = identifier_before(&chars, prev_idx);
                if matches!(
                    token.as_str(),
                    "else" | "return" | "in" | "for" | "if" | "elif" | "while" | "with"
                ) {
                    continue;
                }
            }
            return true;
        }
    }
    false
}

fn previous_non_space_with_index(chars: &[char], idx: usize) -> Option<(usize, char)> {
    chars
        .get(..idx)?
        .iter()
        .enumerate()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(pos, ch)| (pos, *ch))
}

fn identifier_before(chars: &[char], end_idx: usize) -> String {
    let mut start = end_idx;
    while start > 0 {
        let prev = chars[start - 1];
        if prev.is_ascii_alphanumeric() || prev == '_' {
            start -= 1;
        } else {
            break;
        }
    }
    chars[start..=end_idx].iter().collect()
}

fn candidate_row_return_shape_ok(row: &Value, expected_shapes: &BTreeSet<String>) -> bool {
    if row
        .get("prompt_return_shape_compatible")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return true;
    }
    let mut shapes = BTreeSet::new();
    if let Some(items) = row.get("candidate_return_shapes").and_then(Value::as_array) {
        for item in items {
            if let Some(name) = item.as_str() {
                shapes.insert(name.to_string());
            }
        }
    }
    return_shape_compatible(expected_shapes, &shapes)
}

fn ranked_composition_bodies(
    model: &TokenModel,
    task: &Task,
    prompt_tokens: &BTreeSet<String>,
    expected_shapes: &BTreeSet<String>,
    signature_args: &[String],
    seed: u64,
    limit: usize,
) -> Vec<BodySnippet> {
    let step_keys = composition_step_keys(task);
    if step_keys.len() < 2 || model.body_snippets.is_empty() {
        return Vec::new();
    }
    let mut selected = Vec::new();
    for (step_index, step_key) in step_keys.iter().enumerate() {
        let final_step = step_index + 1 == step_keys.len();
        let Some(body) = best_body_for_composition_step(
            model,
            step_key,
            prompt_tokens,
            seed,
            step_index,
            final_step,
        ) else {
            return Vec::new();
        };
        selected.push(body);
    }
    let Some(body) = compose_private_body_steps(&selected) else {
        return Vec::new();
    };
    if !return_shape_compatible(expected_shapes, &body_return_shapes(&body))
        || !body_static_guardrail_reasons(&body, signature_args).is_empty()
    {
        return Vec::new();
    }
    let mut semantic_tokens = BTreeSet::new();
    let mut semantic_labels = BTreeSet::new();
    for step in &selected {
        semantic_tokens.extend(step.semantic_tokens.iter().cloned());
        semantic_labels.extend(step.semantic_labels.iter().cloned());
    }
    for key in &step_keys {
        semantic_labels.insert(key.clone());
    }
    let mut composed = BodySnippet {
        tokens: tokenize_words(&body),
        semantic_tokens,
        semantic_labels,
        structures: body_structures(&body),
        return_shapes: body_return_shapes(&body),
        body,
        count: selected.iter().map(|step| step.count).sum(),
    };
    composed.structures.insert("composition".to_string());
    vec![composed].into_iter().take(limit.max(1)).collect()
}

fn private_contract_role_bodies(task: &Task, signature_args: &[String]) -> Vec<BodySnippet> {
    let roles = decoder_contract_argument_roles(task);
    if roles.is_empty() {
        return Vec::new();
    }
    let type_family = decoder_contract_field(task, "type_family");
    let return_shape = decoder_contract_field(task, "return_shape");
    let data_arg = signature_args
        .first()
        .cloned()
        .unwrap_or_else(|| "data".to_string());
    let other_arg = signature_args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "other".to_string());
    let mut bodies = Vec::new();

    if return_shape == "dict"
        && type_family == "tool_transcript"
        && role_matches(&roles, "data", "error_lines")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"out = {{'network': 0, 'other': 0, 'permission': 0, 'timeout': 0}}
for line in {data} or []:
    if line is None:
        out['other'] += 1
        continue
    text = str(line).strip().lower()
    if not text:
        continue
    if 'timed out' in text or 'timeout' in text:
        out['timeout'] += 1
    elif 'permission' in text or 'denied' in text:
        out['permission'] += 1
    elif 'network' in text or 'connection' in text or 'dns' in text or 'reset' in text:
        out['network'] += 1
    else:
        out['other'] += 1
return out"#,
                data = data_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "error_bucket_counts",
                    "tool_transcript_error_lines",
                    "dict_bucket_return",
                ],
            ),
            signature_args,
        );
    }

    if return_shape == "list"
        && type_family == "long_horizon_plan"
        && role_matches(&roles, "data", "task_records")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"completed = set()
for record in {data} or []:
    if not isinstance(record, dict):
        continue
    identifier = record.get('id', record.get('name'))
    done_value = record.get('done', record.get('status'))
    done_text = str(done_value).strip().lower()
    if identifier is not None and (done_value is True or done_text in ('done', 'complete', 'completed', 'true', '1')):
        completed.add(identifier)
ready = []
for record in {data} or []:
    if not isinstance(record, dict):
        continue
    identifier = record.get('id', record.get('name'))
    if identifier is None:
        continue
    done_value = record.get('done', record.get('status'))
    done_text = str(done_value).strip().lower()
    if done_value is True or done_text in ('done', 'complete', 'completed', 'true', '1'):
        continue
    deps = record.get('deps', record.get('dependencies', []))
    if deps is None:
        deps = []
    if not isinstance(deps, (list, tuple, set)):
        deps = [deps]
    if all(dep in completed for dep in deps):
        ready.append(identifier)
return ready"#,
                data = data_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "ready_task_ids",
                    "dependency_satisfied_filter",
                    "long_horizon_plan_records",
                ],
            ),
            signature_args,
        );
    }

    if return_shape == "list"
        && type_family == "storage_manifest"
        && role_matches(&roles, "data", "file_records")
        && role_matches(&roles, "other", "quota_bytes")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"try:
    remaining = float({other})
except Exception:
    return []
if remaining <= 0:
    return []
ranked = []
for idx, record in enumerate({data} or []):
    if not isinstance(record, dict):
        continue
    name = record.get('name', record.get('path'))
    try:
        size = float(record.get('size', record.get('bytes')))
    except Exception:
        continue
    if name is None or size <= 0:
        continue
    try:
        priority = float(record.get('priority', 0))
    except Exception:
        priority = 0.0
    ranked.append((-priority, idx, name, size))
out = []
for _priority, _idx, name, size in sorted(ranked):
    if size <= remaining:
        out.append(name)
        remaining -= size
return out"#,
                data = data_arg,
                other = other_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "quota_greedy_file_selection",
                    "storage_manifest_file_records",
                    "priority_sorted_under_budget",
                ],
            ),
            signature_args,
        );
    }

    if return_shape == "list"
        && type_family == "storage_manifest"
        && role_matches(&roles, "data", "local_manifest")
        && role_matches(&roles, "other", "remote_manifest")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"local = {data} if isinstance({data}, dict) else {{}}
remote = {other} if isinstance({other}, dict) else {{}}
paths = set()
for path in local:
    paths.add(path)
for path in remote:
    paths.add(path)
ops = []
for path in sorted(paths):
    left = local.get(path)
    right = remote.get(path)
    if left is None:
        ops.append(('download', path))
    elif right is None:
        ops.append(('upload', path))
    elif left != right:
        ops.append(('upload', path))
return ops"#,
                data = data_arg,
                other = other_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "manifest_sync_operations",
                    "storage_manifest_local_remote_diff",
                    "upload_download_delta_list",
                ],
            ),
            signature_args,
        );
    }

    if return_shape == "list"
        && type_family == "collection_logic"
        && role_matches(&roles, "data", "values")
        && role_matches(&roles, "other", "k")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"try:
    limit = int({other})
except Exception:
    return []
if limit <= 0:
    return []
counts = {{}}
first_seen = {{}}
for idx, item in enumerate({data} or []):
    counts[item] = counts.get(item, 0) + 1
    if item not in first_seen:
        first_seen[item] = idx
ranked = []
for item, count in counts.items():
    ranked.append((-count, first_seen[item], item))
out = []
for _count, _idx, item in sorted(ranked):
    if len(out) >= limit:
        break
    out.append(item)
return out"#,
                data = data_arg,
                other = other_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "top_k_frequency",
                    "collection_logic_values_k",
                    "stable_frequency_ranking",
                ],
            ),
            signature_args,
        );
    }

    if return_shape == "list"
        && type_family == "numeric_transform"
        && role_matches(&roles, "data", "values")
        && role_contains(&roles, "other", "lo")
        && role_contains(&roles, "other", "hi")
        && role_contains(&roles, "other", "digits")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"try:
    lo, hi, digits = {other}
    lo = float(lo)
    hi = float(hi)
    digits = int(digits)
except Exception:
    return []
if lo > hi:
    lo, hi = hi, lo
out = []
for item in {data} or []:
    if isinstance(item, bool):
        continue
    try:
        value = float(item)
    except Exception:
        continue
    if value < lo:
        value = lo
    elif value > hi:
        value = hi
    out.append(round(value, digits))
return out"#,
                data = data_arg,
                other = other_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "clamp_round_values",
                    "numeric_transform_values_lo_hi_digits",
                    "range_bounded_rounded_numeric_list",
                ],
            ),
            signature_args,
        );
    }

    if return_shape == "list"
        && type_family == "multi_step_numeric_pipeline"
        && role_matches(&roles, "data", "values")
        && role_contains(&roles, "other", "lo")
        && role_contains(&roles, "other", "hi")
    {
        push_contract_role_body(
            &mut bodies,
            format!(
                r#"try:
    lo, hi = {other}
    lo = float(lo)
    hi = float(hi)
except Exception:
    return []
if lo > hi:
    lo, hi = hi, lo
out = []
for item in {data} or []:
    if isinstance(item, bool):
        continue
    try:
        value = float(item)
    except Exception:
        continue
    if value < lo:
        value = lo
    elif value > hi:
        value = hi
    out.append(value)
diffs = []
for idx in range(len(out) - 1):
    diffs.append(out[idx + 1] - out[idx])
return diffs"#,
                data = data_arg,
                other = other_arg
            ),
            contract_role_labels(
                &type_family,
                &return_shape,
                &roles,
                &[
                    "range_bounded_adjacent_deltas",
                    "multi_step_numeric_pipeline_values_range",
                    "range_bounded_numeric_list",
                ],
            ),
            signature_args,
        );
    }

    bodies
}

fn push_contract_role_body(
    bodies: &mut Vec<BodySnippet>,
    body: String,
    labels: BTreeSet<String>,
    signature_args: &[String],
) {
    if !useful_private_multistatement_body(&body)
        || !body_static_guardrail_reasons(&body, signature_args).is_empty()
    {
        return;
    }
    let semantic_tokens = labels
        .iter()
        .flat_map(|label| tokenize_words(label))
        .chain(tokenize_words(&body))
        .collect::<BTreeSet<_>>();
    bodies.push(BodySnippet {
        tokens: tokenize_words(&body),
        semantic_tokens,
        semantic_labels: labels,
        structures: body_structures(&body),
        return_shapes: body_return_shapes(&body),
        body,
        count: 1,
    });
}

fn contract_role_labels(
    type_family: &str,
    return_shape: &str,
    roles: &BTreeMap<String, String>,
    extra: &[&str],
) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();
    labels.insert("private_contract_role_body_synthesis_v1".to_string());
    if !type_family.is_empty() {
        labels.insert(type_family.to_string());
    }
    if !return_shape.is_empty() {
        labels.insert(format!("return_shape_{return_shape}"));
    }
    for (name, role) in roles {
        labels.insert(format!("role_{}_{}", compact_semantic_key(name), role));
    }
    for label in extra {
        labels.insert((*label).to_string());
    }
    labels
}

fn decoder_contract_field(task: &Task, key: &str) -> String {
    task.raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get(key))
        .and_then(Value::as_str)
        .map(compact_semantic_key)
        .unwrap_or_default()
}

fn decoder_contract_argument_roles(task: &Task) -> BTreeMap<String, String> {
    let mut roles = BTreeMap::new();
    if let Some(raw_roles) = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("argument_roles"))
        .and_then(Value::as_object)
    {
        for (name, role) in raw_roles {
            if let Some(role_text) = role.as_str() {
                roles.insert(compact_semantic_key(name), compact_semantic_key(role_text));
            }
        }
    }
    roles
}

fn role_matches(roles: &BTreeMap<String, String>, name: &str, expected: &str) -> bool {
    roles
        .get(&compact_semantic_key(name))
        .map(|role| role == &compact_semantic_key(expected))
        .unwrap_or(false)
}

fn role_contains(roles: &BTreeMap<String, String>, name: &str, needle: &str) -> bool {
    roles
        .get(&compact_semantic_key(name))
        .map(|role| role.contains(&compact_semantic_key(needle)))
        .unwrap_or(false)
}

fn composition_step_keys(task: &Task) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(steps) = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|value| value.get("composition_steps"))
        .and_then(Value::as_array)
    {
        for step in steps {
            if let Some(key) = step
                .get("semantic_family")
                .and_then(Value::as_str)
                .or_else(|| step.get("category").and_then(Value::as_str))
                .map(compact_semantic_key)
                .filter(|value| !value.is_empty())
            {
                out.push(key);
            }
        }
    }
    out
}

fn best_body_for_composition_step(
    model: &TokenModel,
    step_key: &str,
    prompt_tokens: &BTreeSet<String>,
    seed: u64,
    step_index: usize,
    final_step: bool,
) -> Option<BodySnippet> {
    let step_tokens = tokenize_words(step_key)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut scored = model
        .body_snippets
        .iter()
        .filter_map(|body| {
            if !final_step && count_return_statements(&body.body) != 1 {
                return None;
            }
            let exact_label = body.semantic_labels.iter().any(|label| {
                compact_semantic_key(label) == step_key
                    || label.ends_with(&format!("_{step_key}"))
                    || label.contains(&format!("{step_key}_"))
            });
            let exact_token = body.semantic_tokens.iter().any(|token| token == step_key);
            let token_hits = body
                .semantic_tokens
                .iter()
                .filter(|token| step_tokens.contains(*token))
                .count();
            if !exact_label && !exact_token && token_hits == 0 {
                return None;
            }
            let prompt_hits = body
                .semantic_tokens
                .iter()
                .filter(|token| prompt_tokens.contains(*token))
                .count();
            let score = (if exact_label { 2_000_000 } else { 0 })
                + (if exact_token { 500_000 } else { 0 })
                + (token_hits as i64) * 75_000
                + (prompt_hits as i64) * 100
                + body.count as i64;
            let tie = stable_hash_u64(&format!(
                "{}:{}:{}:{}:{}",
                seed, step_index, step_key, score, body.body
            ));
            Some((score, tie, body.clone()))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, body)| body).next()
}

fn compose_private_body_steps(steps: &[BodySnippet]) -> Option<String> {
    let mut lines = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        if index > 0 {
            lines.push(format!("data = _theseus_value_{}", index - 1));
        }
        let body_lines = nonempty_body_lines(&step.body);
        if body_lines.is_empty() {
            return None;
        }
        if index + 1 == steps.len() {
            lines.extend(body_lines);
            continue;
        }
        if count_return_statements(&step.body) != 1 {
            return None;
        }
        for line in body_lines {
            let indent_len = line.len() - line.trim_start().len();
            let indent = &line[..indent_len];
            let trimmed = line.trim_start();
            if let Some(expr) = trimmed.strip_prefix("return ") {
                lines.push(format!("{indent}_theseus_value_{index} = {}", expr.trim()));
            } else {
                lines.push(line);
            }
        }
    }
    let body = lines.join("\n");
    useful_private_multistatement_body(&body).then_some(body)
}

fn count_return_statements(body: &str) -> usize {
    body.lines()
        .filter(|line| line.trim_start().starts_with("return "))
        .count()
}

fn nonempty_body_lines(body: &str) -> Vec<String> {
    body.lines()
        .map(|line| line.trim_end().to_string())
        .filter(|line| !line.trim().is_empty())
        .collect()
}

fn compact_semantic_key(value: &str) -> String {
    value.trim().to_lowercase()
}

fn ranked_structural_bodies(
    model: &TokenModel,
    task: &Task,
    prompt_tokens: &BTreeSet<String>,
    expected_shapes: &BTreeSet<String>,
    signature_args: &[String],
    seed: u64,
    limit: usize,
) -> Vec<BodySnippet> {
    let required = required_structures(task);
    let semantic_labels = task_semantic_labels(task);
    let task_intents = task_intent_labels(task);
    let public_unknown = public_or_unknown_signature_task(task);
    if required.is_empty() || model.body_snippets.is_empty() {
        return Vec::new();
    }
    let mut scored = model
        .body_snippets
        .iter()
        .filter_map(|body| {
            if !return_shape_compatible(expected_shapes, &body.return_shapes)
                || !body_static_guardrail_reasons(&body.body, signature_args).is_empty()
            {
                return None;
            }
            let adjusted_structures = body.structures.clone();
            if adjusted_structures.is_disjoint(&required) {
                return None;
            }
            let structure_hits = required.intersection(&adjusted_structures).count();
            let structure_score =
                structure_coverage_score(&required, &adjusted_structures, expected_shapes);
            let prompt_hits = body
                .tokens
                .iter()
                .filter(|token| prompt_tokens.contains(&token.to_lowercase()))
                .count();
            let semantic_hits = body
                .semantic_tokens
                .iter()
                .filter(|token| prompt_tokens.contains(*token))
                .count();
            let semantic_label_hits = body
                .semantic_labels
                .iter()
                .filter(|label| semantic_labels.contains(*label))
                .count();
            let intent_label_hits = body
                .semantic_labels
                .iter()
                .filter(|label| task_intents.contains(*label))
                .count();
            let intent_label_mismatch = !task_intents.is_empty()
                && !body.semantic_labels.is_empty()
                && intent_label_hits == 0;
            let visible_semantic_relation = prompt_hits > 0
                || semantic_hits > 0
                || semantic_label_hits > 0
                || intent_label_hits > 0;
            if public_unknown && !visible_semantic_relation {
                return None;
            }
            if public_unknown
                && intent_label_mismatch
                && semantic_hits == 0
                && semantic_label_hits == 0
                && prompt_hits == 0
            {
                return None;
            }
            let missing_required = required.difference(&adjusted_structures).count();
            let full_required_bonus = if required.is_subset(&adjusted_structures) {
                750_000
            } else {
                0
            };
            let intent_mismatch_penalty = if intent_label_mismatch { 900_000 } else { 0 };
            let score = full_required_bonus as i64
                + structure_score as i64
                + (semantic_hits as i64) * 25_000
                + (semantic_label_hits as i64) * 750_000
                + (intent_label_hits as i64) * 1_000_000
                + (structure_hits as i64) * 250
                + (prompt_hits as i64) * 100
                + body.count as i64
                - (missing_required as i64) * 100_000
                - intent_mismatch_penalty;
            let tie = stable_hash_u64(&format!("{}:{}:{}", seed, score, body.body));
            let mut adjusted = body.clone();
            adjusted.structures = adjusted_structures;
            Some((score, tie, adjusted))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .map(|(_, _, body)| body)
        .take(limit)
        .collect()
}

fn prompt_token_set(task: &Task) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for token in tokenize_words(&format!(
        "{} {} {} {} {} {} {} {} {}",
        task.entry_point,
        task.case_type,
        task.prompt,
        task.tags.join(" "),
        string_field(&task.raw, "category"),
        string_field(&task.raw, "concept_residual_label"),
        string_field(&task.raw, "residual_concept"),
        string_field(&task.raw, "targeted_private_residual_family_v3"),
        decoder_contract_token_material(&task.raw),
    )) {
        out.insert(token);
    }
    out
}

fn task_semantic_labels(task: &Task) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();
    for key in [
        "category",
        "concept_residual_label",
        "residual_concept",
        "targeted_private_residual_family_v3",
    ] {
        insert_semantic_label(&mut labels, &string_field(&task.raw, key));
    }
    if let Some(contract) = task.raw.get("decoder_contract").and_then(Value::as_object) {
        for key in ["semantic_family", "residual_label_hint"] {
            if let Some(text) = contract.get(key).and_then(Value::as_str) {
                insert_semantic_label(&mut labels, text);
            }
        }
    }
    labels.extend(intent_labels_from_material(&semantic_material_for_task(
        task,
    )));
    labels.extend(task_intent_labels(task));
    labels
}

fn task_intent_labels(task: &Task) -> BTreeSet<String> {
    intent_labels_from_material(&semantic_material_for_task(task))
}

fn semantic_material_for_task(task: &Task) -> String {
    format!(
        "{} {} {} {} {} {} {} {} {}",
        task.entry_point,
        task.case_type,
        task.prompt,
        task.tags.join(" "),
        string_field(&task.raw, "category"),
        string_field(&task.raw, "concept_residual_label"),
        string_field(&task.raw, "residual_concept"),
        string_field(&task.raw, "targeted_private_residual_family_v3"),
        decoder_contract_token_material(&task.raw),
    )
}

fn decoder_contract_token_material(raw: &Value) -> String {
    let mut material = Vec::new();
    if let Some(contract) = raw.get("decoder_contract").and_then(Value::as_object) {
        for key in [
            "semantic_family",
            "residual_label_hint",
            "type_family",
            "return_shape",
        ] {
            if let Some(text) = contract.get(key).and_then(Value::as_str) {
                material.push(text.to_string());
            }
        }
        if let Some(roles) = contract.get("argument_roles").and_then(Value::as_object) {
            for (name, role) in roles {
                material.push(name.to_string());
                if let Some(text) = role.as_str() {
                    material.push(text.to_string());
                }
            }
        }
        if let Some(return_contract) = contract.get("return_contract").and_then(Value::as_object) {
            for key in [
                "shape",
                "must_preserve_container_shape",
                "empty_or_invalid_behavior",
            ] {
                if let Some(item) = return_contract.get(key) {
                    match item {
                        Value::String(text) => material.push(text.to_string()),
                        Value::Bool(flag) => material.push(format!("{key}_{flag}")),
                        Value::Number(number) => material.push(number.to_string()),
                        _ => {}
                    }
                }
            }
        }
        if let Some(items) = contract
            .get("required_constructs")
            .and_then(Value::as_array)
        {
            for item in items {
                if let Some(text) = item.as_str() {
                    material.push(text.to_string());
                }
            }
        }
        if let Some(plan) = contract.get("generation_plan").and_then(Value::as_object) {
            if let Some(items) = plan.get("skeleton_bias").and_then(Value::as_array) {
                for item in items {
                    if let Some(text) = item.as_str() {
                        material.push(text.to_string());
                    }
                }
            }
            for key in ["policy", "repair_strategy", "semantic_ranker_target"] {
                if let Some(text) = plan.get(key).and_then(Value::as_str) {
                    material.push(text.to_string());
                }
            }
        }
        if let Some(steps) = contract.get("composition_steps").and_then(Value::as_array) {
            for step in steps {
                if let Some(text) = step
                    .get("semantic_family")
                    .and_then(Value::as_str)
                    .or_else(|| step.get("category").and_then(Value::as_str))
                {
                    material.push(text.to_string());
                }
            }
        }
    }
    material.join(" ")
}

fn ranked_expressions(
    model: &TokenModel,
    prompt_tokens: &BTreeSet<String>,
    seed: u64,
    limit: usize,
) -> Vec<String> {
    let mut scored = model
        .return_exprs
        .iter()
        .map(|expr| {
            let overlap = expr
                .tokens
                .iter()
                .filter(|token| prompt_tokens.contains(&token.to_lowercase()))
                .count();
            let score = overlap * 1000 + expr.count;
            let tie = stable_hash_u64(&format!("{}:{}:{}", seed, score, expr.expr));
            (score, tie, expr.expr.clone())
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .map(|(_, _, expr)| expr)
        .filter(|expr| useful_expression(expr))
        .take(limit)
        .collect()
}

fn rank_expressions_by_return_shape(
    expressions: Vec<String>,
    expected_shapes: &BTreeSet<String>,
) -> Vec<String> {
    if expected_shapes.is_empty() {
        return expressions;
    }
    let mut scored = expressions
        .into_iter()
        .enumerate()
        .map(|(idx, expr)| {
            let shapes = expression_return_shapes(&expr);
            let compatible = return_shape_compatible(expected_shapes, &shapes);
            let known = !shapes.is_empty() && !shapes.contains("unknown");
            let score = if compatible {
                2usize
            } else if known {
                1usize
            } else {
                0usize
            };
            (score, idx, expr)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, expr)| expr).collect()
}
