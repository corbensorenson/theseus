use std::fs;

use symliquid_core::benchmarks::{
    generate_babylm_probe_suite, generate_cgs_hard_suite, run_local_baseline_suite,
    run_symliquid_suite, train_babylm_probe_scorer, BabyLmProbeTrainConfig, LocalBaselineKind,
};

#[test]
fn cgs_hard_suite_scores_symliquid_reference() {
    let suite = generate_cgs_hard_suite(42, 2);
    let report = run_symliquid_suite(&suite, "symliquid-reference", false);
    assert_eq!(report.summary.cases, 20);
    assert!(report.summary.accuracy > 0.99);
    assert_eq!(report.summary.invalid_action_rate, 0.0);
}

#[test]
fn local_baseline_scores_without_external_inference() {
    let suite = generate_cgs_hard_suite(7, 1);
    let report =
        run_local_baseline_suite(&suite, LocalBaselineKind::FirstAllowed, 0, 128, 1, 0.01).unwrap();
    assert_eq!(
        report.summary.mode,
        symliquid_core::benchmarks::RunMode::LocalBaseline
    );
    assert_eq!(report.summary.total_tool_calls, 0);
}

#[test]
fn babylm_probe_imports_local_text() {
    let path = std::env::temp_dir().join("symliquid_babylm_probe_test.txt");
    fs::write(
        &path,
        "This dog is friendly. These cats are playful. A bird can fly over walls.",
    )
    .unwrap();
    let suite = generate_babylm_probe_suite(&path, 0, 4).unwrap();
    assert_eq!(suite.name, "babylm_local_probe");
    assert!(!suite.cases.is_empty());
    let report = run_symliquid_suite(&suite, "symliquid-babylm-probe-test", false);
    assert_eq!(report.summary.total_tool_calls, 0);
    let _ = fs::remove_file(path);
}

#[test]
fn babylm_probe_sequence_scorer_trains_locally() {
    let path = std::env::temp_dir().join("symliquid_babylm_train_test.txt");
    fs::write(
        &path,
        "This dog is friendly. These cats are playful. A bird can fly over walls. This area is quiet. These rooms are bright.",
    )
    .unwrap();
    let report = train_babylm_probe_scorer(BabyLmProbeTrainConfig {
        input_path: path.to_string_lossy().to_string(),
        eval_input_path: None,
        train_seed: 0,
        eval_seed: 1,
        train_limit: 4,
        eval_limit: 4,
        steps: 8,
        hv_dim: 128,
        lr: 0.05,
        stateful: false,
        pairwise_contrast: false,
        balance_rules: false,
        prior_weight: 0.0,
    })
    .unwrap();
    assert_eq!(report.eval.summary.total_tool_calls, 0);
    assert!(report.train_loss.is_finite());
    let _ = fs::remove_file(path);
}
