use super::linguistic_features::add_recurrent_grammar_slot_features;
use super::*;

#[test]
fn generated_suite_has_all_task_families() {
    let suite = generate_cgs_hard_suite(0, 2);
    assert_eq!(suite.cases.len(), 20);
    assert!(suite.cases.iter().any(|case| case.task == "role_filler"));
    assert!(suite
        .cases
        .iter()
        .any(|case| case.task == "missing_evidence_rag"));
    assert!(suite
        .cases
        .iter()
        .any(|case| case.task == "code_repair_verifier"));
    assert!(suite
        .cases
        .iter()
        .any(|case| case.task == "babylm_minimal_pair"));
    assert!(suite
        .cases
        .iter()
        .any(|case| case.task == "blimp_acceptability"));
    assert!(suite
        .cases
        .iter()
        .any(|case| case.task == "long_context_retrieval"));
    assert!(suite
        .cases
        .iter()
        .any(|case| case.task == "adversarial_rag"));
}

#[test]
fn symliquid_reference_scores_cleanly() {
    let suite = generate_cgs_hard_suite(1, 1);
    let report = run_symliquid_suite(&suite, "symliquid-reference", false);
    assert_eq!(report.summary.cases, 10);
    assert!((report.summary.accuracy - 1.0).abs() < 1e-6);
    assert_eq!(
        report.results.iter().filter(|r| r.invalid_action).count(),
        0
    );
}

#[test]
fn invalid_action_is_penalized() {
    let suite = generate_cgs_hard_suite(2, 1);
    let case = suite
        .cases
        .iter()
        .find(|case| case.expected_kind == ExpectedKind::Action)
        .unwrap();
    let result = score_output(
        &suite,
        case,
        ModelResponse {
            case_id: case.id.clone(),
            model_id: "bad".to_string(),
            mode: RunMode::LocalBaseline,
            output: "make_up_action".to_string(),
            runtime_ms: None,
            token_count: None,
            tool_calls: None,
            estimated_cost_usd: None,
            notes: Vec::new(),
        },
    );
    assert!(result.invalid_action);
    assert!(!result.correct);
}

#[test]
fn standalone_training_defaults_to_learned_only() {
    let report = train_standalone_symliquid(StandaloneTrainConfig {
        cases_per_task: 1,
        epochs: 1,
        hv_dim: 128,
        ..StandaloneTrainConfig::default()
    })
    .unwrap();
    assert!(!report.symbolic_fallback);
    assert_eq!(report.feature_set, "structured_cgs_vsa");
    assert_eq!(report.eval.summary.mode, RunMode::SymLiquid);
}

#[test]
fn stateful_babylm_features_add_grammar_slots() {
    let plain = sentence_features("The children near the window are ready.", 512, false);
    let stateful = sentence_features("The children near the window are ready.", 512, true);
    let delta = plain
        .iter()
        .zip(stateful.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    assert!(delta > 0.05);
}

#[test]
fn recurrent_grammar_slots_track_binding_and_animacy() {
    let mut features = vec![0.0; 512];
    add_recurrent_grammar_slot_features(
        &mut features,
        "Alice said that Carla praised herself near the stage.",
    );
    add_recurrent_grammar_slot_features(&mut features, "The window laughed near the stage.");
    assert!(features.iter().any(|value| value.abs() > 0.0));
}

#[test]
fn grammar_prior_uses_subject_head_over_prepositional_attractor() {
    let good = sentence_quality_prior("The green people beside the library are ready.");
    let bad = sentence_quality_prior("The green people beside the library is ready.");
    assert!(good > bad);
}

#[test]
fn grammar_prior_prefers_local_binding_domain_in_complement_clause() {
    let good = sentence_quality_prior("Bruce said that Diane praised herself.");
    let bad = sentence_quality_prior("Bruce said that Diane praised himself.");
    assert!(good > bad);
}

#[test]
fn grammar_prior_covers_mutated_argument_and_animacy_templates() {
    assert!(
        sentence_quality_prior("The student moved the package.")
            > sentence_quality_prior("The student moved.")
    );
    assert!(
        sentence_quality_prior("The student laughed.")
            > sentence_quality_prior("The student laughed the package.")
    );
    assert!(
        sentence_quality_prior("The student laughed near the stage.")
            > sentence_quality_prior("The window laughed near the stage.")
    );
}

#[test]
fn writes_breakdown_csv_for_report() {
    let suite = generate_cgs_hard_suite(3, 1);
    let report = run_symliquid_suite(&suite, "symliquid-reference", false);
    let path = std::env::temp_dir().join("symliquid_breakdown_test.csv");
    write_breakdown_csv(&path, &suite, &report, "task").unwrap();
    let text = fs::read_to_string(&path).unwrap();
    let _ = fs::remove_file(&path);
    assert!(text.contains("task,group_by,group,cases,accuracy"));
    assert!(text.contains("role_filler"));
}
