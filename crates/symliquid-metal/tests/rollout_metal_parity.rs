#![cfg(target_os = "macos")]

use symliquid_core::tensor::Tensor;
use symliquid_cuda::rollout_cuda::{
    rollout_sequence_cpu, RolloutConfig, RolloutParams, RolloutState,
};
use symliquid_metal::{
    metal_available, rollout_metal_feature_proof_report, rollout_metal_readout_proof_report,
    rollout_metal_readout_training_proof_report, rollout_metal_state_training_proof_report,
    rollout_metal_train_path_proof_report, rollout_sequence_metal,
    token_superposition_metal_readout_proof_report, RolloutMetalProofConfig,
    TokenSuperpositionMetalProofConfig,
};

#[test]
fn metal_rollout_state_update_matches_cpu() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let config = RolloutConfig {
        batch: 2,
        obs_dim: 3,
        hidden_dim: 4,
        reservoir_dim: 5,
        hv_dim: 7,
        dt: 0.1,
        alpha: 0.25,
        memory_decay: 0.9,
    };
    let observations = Tensor::new(6, 3, deterministic_values(18, 0.07)).unwrap();
    let state = RolloutState {
        h: Tensor::new(2, 4, deterministic_values(8, 0.03)).unwrap(),
        r: Tensor::new(2, 5, deterministic_values(10, 0.02)).unwrap(),
        memory: Tensor::zeros(2, 7),
    };
    let params = RolloutParams {
        obs_to_h: deterministic_values(config.hidden_dim * config.obs_dim, 0.04),
        h_recurrent: deterministic_values(config.hidden_dim * config.hidden_dim, 0.015),
        h_bias: deterministic_values(config.hidden_dim, 0.02),
        reservoir_input: deterministic_values(config.reservoir_dim * config.hidden_dim, 0.03),
        reservoir_recurrent: deterministic_values(
            config.reservoir_dim * config.reservoir_dim,
            0.01,
        ),
        reservoir_bias: deterministic_values(config.reservoir_dim, 0.015),
        hv_proj: deterministic_values(config.hv_dim * config.reservoir_dim, 0.025),
    };

    let cpu = rollout_sequence_cpu(&observations, &state, &params, &config).unwrap();
    let metal = rollout_sequence_metal(&observations, &state, &params, &config).unwrap();
    assert_close(&cpu.h.data, &metal.h.data, 1e-4);
    assert_close(&cpu.r.data, &metal.r.data, 1e-4);
    assert_close(&cpu.memory.data, &metal.memory.data, 1e-4);
}

#[test]
fn metal_rollout_feature_proof_report_stays_non_promotional() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let report = rollout_metal_feature_proof_report(&RolloutMetalProofConfig {
        cases: 4,
        batch: 2,
        steps: 3,
        obs_dim: 3,
        hidden_dim: 4,
        reservoir_dim: 5,
        hv_dim: 7,
        ..RolloutMetalProofConfig::default()
    });
    assert_eq!(
        report.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("train_rollout_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("external_inference_calls")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .get("public_training_rows")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
}

#[test]
fn metal_rollout_readout_proof_report_stays_non_promotional() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let report = rollout_metal_readout_proof_report(&RolloutMetalProofConfig {
        cases: 4,
        batch: 2,
        steps: 3,
        obs_dim: 3,
        hidden_dim: 4,
        reservoir_dim: 5,
        hv_dim: 7,
        output_dim: 4,
        ..RolloutMetalProofConfig::default()
    });
    assert_eq!(
        report.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("train_rollout_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("prediction_agreement")
            .and_then(|value| value.as_f64()),
        Some(1.0)
    );
    assert_eq!(
        report
            .get("external_inference_calls")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .get("public_training_rows")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
}

#[test]
fn metal_rollout_readout_training_proof_report_stays_non_promotional() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let report = rollout_metal_readout_training_proof_report(&RolloutMetalProofConfig {
        cases: 4,
        batch: 2,
        steps: 3,
        obs_dim: 3,
        hidden_dim: 4,
        reservoir_dim: 5,
        hv_dim: 7,
        output_dim: 4,
        readout_epochs: 2,
        readout_lr: 0.03,
        samples_per_launch: 2,
        ..RolloutMetalProofConfig::default()
    });
    assert_eq!(
        report.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("readout_training_subpath_proof")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("train_rollout_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("external_inference_calls")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .get("public_training_rows")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
}

#[test]
fn metal_train_path_proof_report_stays_non_promotional() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let report = rollout_metal_train_path_proof_report(&RolloutMetalProofConfig {
        cases: 4,
        batch: 2,
        steps: 3,
        obs_dim: 3,
        hidden_dim: 4,
        reservoir_dim: 5,
        hv_dim: 7,
        output_dim: 4,
        readout_epochs: 2,
        readout_lr: 0.03,
        samples_per_launch: 2,
        ..RolloutMetalProofConfig::default()
    });
    assert_eq!(
        report.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("symbolic_fallback")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("train_rollout_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("external_inference_calls")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .get("public_training_rows")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
}

#[test]
fn metal_state_training_proof_report_stays_non_promotional() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let report = rollout_metal_state_training_proof_report(&RolloutMetalProofConfig {
        cases: 4,
        batch: 2,
        steps: 3,
        obs_dim: 3,
        hidden_dim: 4,
        reservoir_dim: 5,
        hv_dim: 7,
        output_dim: 4,
        state_epochs: 2,
        state_lr: 0.03,
        tolerance: 0.0005,
        ..RolloutMetalProofConfig::default()
    });
    assert_eq!(
        report.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("state_training_semantics_proof")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("train_rollout_sweep_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("external_inference_calls")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .pointer("/guardrails/no_fallback_returns")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn metal_token_superposition_readout_proof_stays_non_promotional() {
    if !metal_available() {
        eprintln!("skipping: no default Metal device available");
        return;
    }
    let report =
        token_superposition_metal_readout_proof_report(&TokenSuperpositionMetalProofConfig {
            vocab_size: 12,
            hv_dim: 24,
            train_tokens: 128,
            train_samples: 16,
            eval_samples: 16,
            baseline_epochs: 2,
            bag_size: 4,
            recovery_ratio: 0.5,
            lr: 0.03,
            samples_per_launch: 4,
            tolerance: 0.0005,
        });
    assert_eq!(
        report.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .get("train_token_superposition_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("full_cli_parity_claim_allowed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        report
            .get("external_inference_calls")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .get("public_training_rows")
            .and_then(|value| value.as_i64()),
        Some(0)
    );
    assert_eq!(
        report
            .pointer("/guardrails/no_fallback_returns")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        report
            .pointer("/variant/prediction_agreement")
            .and_then(|value| value.as_f64()),
        Some(1.0)
    );
}

fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx * 37 + 11) % 23) as f32 - 11.0) * scale)
        .collect()
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "idx {idx}: {a} != {b} within {tolerance}"
        );
    }
}
